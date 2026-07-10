#!/usr/bin/env python3
"""GRPO/RLVR post-SFT pass using verifiable family rewards.

Env:
  BASE_MODEL   HF id or path to the SFT merged_hf dir
  DATA         JSONL of {family, system, prompt} (no golden/live prompts)
  OUTPUT_DIR   where to write adapter_final + merged_hf
  EPOCHS, LR, LORA_RANK, NUM_GENERATIONS (group size), MAX_COMPLETION_LENGTH

TRL 0.20 ships GRPOTrainer. We keep the reward pure-Python (rewards_rlvr.py)
so the Modal image needs no Fireworks key. Optional JUDGE_BONUS_FILE can mix
in precomputed judge scores (task_id -> float) without calling the API online.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from rewards_rlvr import reward_for


def main() -> None:
    base = os.environ["BASE_MODEL"]
    data_path = Path(os.environ["DATA"])
    out = Path(os.environ["OUTPUT_DIR"])
    epochs = float(os.environ.get("EPOCHS", "1"))
    lr = float(os.environ.get("LR", "5e-6"))
    rank = int(os.environ.get("LORA_RANK", "16"))
    num_gen = int(os.environ.get("NUM_GENERATIONS", "4"))
    max_comp = int(os.environ.get("MAX_COMPLETION_LENGTH", "512"))

    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # MiniCPM's remote-code generate() rejects token_type_ids, but its
    # tokenizer emits them and TRL forwards every tokenizer output.
    tok.model_input_names = ["input_ids", "attention_mask"]

    rows = [json.loads(l) for l in data_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def render(r: dict) -> dict:
        # Prefer official no-think render when the tokenizer supports it.
        try:
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return {
            "prompt": prompt,
            "family": r["family"],
            "user_prompt": r["prompt"],
            # logic_tool rows carry the puzzle's derived solution as a JSON
            # string; other families have no ground truth column.
            "truth": r.get("truth", ""),
        }

    ds = Dataset.from_list([render(r) for r in rows])
    print(f"GRPO prompts: {len(ds)} from {data_path}; generations={num_gen}")

    def reward_fn(completions, family, user_prompt, truth, **kwargs):
        # TRL passes columns as lists aligned with completions.
        out = []
        for comp, fam, up, tr in zip(completions, family, user_prompt, truth):
            # completions may be str or list[{role,content}]
            if isinstance(comp, list):
                text = comp[-1].get("content", "") if comp else ""
            else:
                text = str(comp)
            truth_obj = json.loads(tr) if tr else None
            out.append(float(reward_for(fam, up, text, truth=truth_obj)))
        return out

    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    peft_config = LoraConfig(
        r=rank, lora_alpha=rank * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    cfg = GRPOConfig(
        output_dir=str(out),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=1,
        # generation_batch_size (batch*accum) must be divisible by the group
        # size, so tie accumulation to num_generations.
        gradient_accumulation_steps=num_gen,
        num_generations=num_gen,
        max_completion_length=max_comp,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_config,
        reward_funcs=reward_fn,
    )
    trainer.train()

    adapter = out / "adapter_final"
    trainer.model.save_pretrained(adapter)
    print("saved adapter:", adapter)

    del trainer, model
    torch.cuda.empty_cache()
    base_model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    merged = PeftModel.from_pretrained(base_model, adapter).merge_and_unload()
    merged_dir = out / "merged_hf"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)
    print("saved merged HF model:", merged_dir)


if __name__ == "__main__":
    main()
