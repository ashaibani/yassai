#!/usr/bin/env python3
"""LoRA SFT for MiniCPM5-1B tool-use behavior."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer


NO_THINK_PREFIX = "<think>\\n\\n</think>\\n\\n"

# Training-only template for our no-think router path. MiniCPM5's official
# inference template renders add_generation_prompt=True, enable_thinking=False
# as: <|im_start|>assistant\n<think>\n\n</think>\n\n
# We put that exact prefix before the supervised assistant content and mask loss
# only over the content that follows it.
TRAIN_CHAT_TEMPLATE = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'user' %}"
    "{{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n" + NO_THINK_PREFIX + "' }}"
    "{%- generation %}"
    "{{- message['content'] + '<|im_end|>' }}"
    "{%- endgeneration %}"
    "{{- '\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n" + NO_THINK_PREFIX + "' }}"
    "{%- endif %}"
)


def getenv_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def getenv_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def assert_minicpm5_template_contract(tok: AutoTokenizer) -> None:
    """Fail early if our training template diverges from MiniCPM5 no-think prompting."""
    original_template = tok.chat_template
    if not original_template:
        raise RuntimeError("MiniCPM5 tokenizer did not expose a chat_template")

    messages = [
        {"role": "system", "content": "Use run_python for arithmetic."},
        {"role": "user", "content": "What is 17*23?"},
    ]
    target = '<tool_call>{"name":"run_python","arguments":{"code":"print(17*23)"}}</tool_call>'

    official_prompt = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    tok.chat_template = TRAIN_CHAT_TEMPLATE
    train_render = tok.apply_chat_template(
        messages + [{"role": "assistant", "content": target}],
        tokenize=False,
        add_generation_prompt=False,
    )
    train_prompt = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    tok.chat_template = original_template

    if train_prompt != official_prompt:
        raise RuntimeError(
            "Training prompt does not match MiniCPM5 official no-think prompt:\n"
            f"official={official_prompt!r}\ntrain={train_prompt!r}"
        )
    expected = official_prompt + target + "<|im_end|>\n"
    if train_render != expected:
        raise RuntimeError(
            "Training render does not append assistant target after the official no-think prompt:\n"
            f"expected={expected!r}\nactual={train_render!r}"
        )


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

    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    assert_minicpm5_template_contract(tok)
    tok.chat_template = TRAIN_CHAT_TEMPLATE

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.03,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(out),
            num_train_epochs=epochs,
            per_device_train_batch_size=getenv_int("BATCH_SIZE", 2),
            gradient_accumulation_steps=getenv_int("GRAD_ACCUM", 4),
            learning_rate=lr,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            bf16=torch.cuda.is_available(),
            max_length=max_length,
            packing=False,
            assistant_only_loss=True,
            logging_steps=1,
            save_steps=25,
            save_total_limit=3,
            report_to="none",
            dataloader_num_workers=2,
            remove_unused_columns=False,
            seed=seed,
        ),
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()

    adapter = out / "adapter_final"
    trainer.model.save_pretrained(adapter)

    # Save a merged HF checkpoint for conversion/export. Reload the original
    # tokenizer so the training-only generation mask is never shipped.
    original_tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype, device_map="auto")
    merged = PeftModel.from_pretrained(base_model, adapter).merge_and_unload()
    merged_dir = out / "merged_hf"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    original_tok.save_pretrained(merged_dir)

    print(f"saved adapter: {adapter}")
    print(f"saved merged HF model: {merged_dir}")


if __name__ == "__main__":
    main()
