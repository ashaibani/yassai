#!/usr/bin/env python3
"""LoRA SFT for yassai local models (MiniCPM5 default; any ChatML base via BASE_MODEL)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer


NO_THINK_PREFIX = "<think>\n\n</think>\n\n"

# Training-only template for our no-think router path. MiniCPM5's official
# inference template renders add_generation_prompt=True, enable_thinking=False
# as: <|im_start|>assistant\n<think>\n\n</think>\n\n
# Qwen3.5 uses the same empty-think fencing. We put that exact prefix before
# the supervised assistant content and mask loss only over the content that
# follows it.
TRAIN_CHAT_TEMPLATE = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "{{- '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- elif message['role'] == 'user' %}"
    "{{- '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\n" + NO_THINK_PREFIX + "' }}"
    "{%- generation %}"
    "{{- message['content'] + '<|im_end|>' }}"
    "{%- endgeneration %}"
    "{{- '\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\n" + NO_THINK_PREFIX + "' }}"
    "{%- endif %}"
)


def getenv_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def getenv_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def assert_no_think_template_contract(tok: AutoTokenizer, base: str) -> str:
    """Pick a training chat template that matches the base model's official no-think render.

    MiniCPM5 and Qwen3.5 both emit ChatML + an empty <think> block when
    enable_thinking=False. We keep TRAIN_CHAT_TEMPLATE when it matches that
    official prompt; otherwise we fall back to the model's own template.
    """
    original_template = tok.chat_template
    if not original_template:
        raise RuntimeError(f"{base} tokenizer did not expose a chat_template")

    messages = [
        {"role": "system", "content": "Use run_python for arithmetic."},
        {"role": "user", "content": "What is 17*23?"},
    ]
    target = '<tool_call>{"name":"run_python","arguments":{"code":"print(17*23)"}}</tool_call>'

    try:
        official_prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        official_prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    tok.chat_template = TRAIN_CHAT_TEMPLATE
    train_prompt = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    train_render = tok.apply_chat_template(
        messages + [{"role": "assistant", "content": target}],
        tokenize=False,
        add_generation_prompt=False,
    )

    if train_prompt == official_prompt:
        expected = official_prompt + target + "<|im_end|>\n"
        if train_render != expected:
            tok.chat_template = original_template
            raise RuntimeError(
                "Training render does not append assistant target after the official no-think prompt:\n"
                f"expected={expected!r}\nactual={train_render!r}"
            )
        tok.chat_template = original_template
        return "train_chat_template"

    tok.chat_template = original_template
    if "<think>\n\n</think>" not in official_prompt:
        print(f"note: {base} official prompt has no empty-think prefix; using model template as-is")
    else:
        print(f"note: {base} official no-think prompt differs from TRAIN_CHAT_TEMPLATE; using model template")
    print(f"official_prompt_tail={official_prompt[-120:]!r}")
    return "model_chat_template"


def main() -> None:
    base = os.environ.get("BASE_MODEL", "openbmb/MiniCPM5-1B")
    data_path = Path(os.environ["DATA"])
    out = Path(os.environ.get("OUTPUT_DIR", "/checkpoints/minicpm5-yassai"))
    epochs = getenv_float("EPOCHS", 10.0)
    lr = getenv_float("LR", 1.0e-4)
    rank = getenv_int("LORA_RANK", 32)
    seed = getenv_int("SEED", 42)
    max_length = getenv_int("MAX_LENGTH", 3072)

    set_seed(seed)
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    tok = AutoTokenizer.from_pretrained(base, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    template_mode = assert_no_think_template_contract(tok, base)
    if template_mode == "train_chat_template":
        tok.chat_template = TRAIN_CHAT_TEMPLATE

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def load_base(path: str):
        # MiniCPM5 / classic causal LMs.
        try:
            return AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=dtype,
                attn_implementation="sdpa",
                device_map="auto",
                trust_remote_code=True,
            )
        except (ValueError, KeyError) as exc:
            print(f"AutoModelForCausalLM failed ({exc}); trying AutoModel for multimodal/hybrid bases")
            return AutoModel.from_pretrained(
                path,
                torch_dtype=dtype,
                attn_implementation="sdpa",
                device_map="auto",
                trust_remote_code=True,
            )

    model = load_base(base)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    # Classic LLaMA/Qwen MLP+attn + Qwen3.5 Gated DeltaNet projections.
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj",
    ]
    lora = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.03,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    sft_kwargs = dict(
        output_dir=str(out),
        num_train_epochs=epochs,
        per_device_train_batch_size=getenv_int("BATCH_SIZE", 2),
        gradient_accumulation_steps=getenv_int("GRAD_ACCUM", 4),
        learning_rate=lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        packing=False,
        logging_steps=1,
        save_steps=25,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=seed,
    )
    # TRL 0.20 uses max_length + assistant_only_loss; newer TRL may rename.
    for key, val in (("max_length", max_length), ("max_seq_length", max_length),
                     ("assistant_only_loss", True)):
        sft_kwargs[key] = val
    try:
        sft_args = SFTConfig(**sft_kwargs)
    except TypeError:
        for drop in ("assistant_only_loss", "max_length", "max_seq_length"):
            sft_kwargs.pop(drop, None)
        sft_kwargs["max_length"] = max_length
        try:
            sft_args = SFTConfig(**sft_kwargs)
        except TypeError:
            sft_kwargs.pop("max_length", None)
            sft_kwargs["max_seq_length"] = max_length
            sft_args = SFTConfig(**sft_kwargs)

    trainer_kwargs = dict(model=model, args=sft_args, train_dataset=ds)
    try:
        trainer = SFTTrainer(processing_class=tok, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tok, **trainer_kwargs)
    trainer.train()

    adapter = out / "adapter_final"
    trainer.model.save_pretrained(adapter)

    original_tok = AutoTokenizer.from_pretrained(base, use_fast=True, trust_remote_code=True)
    base_model = load_base(base)
    merged = PeftModel.from_pretrained(base_model, adapter).merge_and_unload()
    merged_dir = out / "merged_hf"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    original_tok.save_pretrained(merged_dir)

    print(f"saved adapter: {adapter}")
    print(f"saved merged HF model: {merged_dir}")
    print(f"template_mode: {template_mode}")


if __name__ == "__main__":
    main()
