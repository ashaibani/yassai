#!/usr/bin/env python3
"""DPO calibration pass over the assist SFT model.

Preference pairs come from our own judge archives (scripts/mine_dpo_pairs.py):
answers to the SAME task where one passed the glm-5p2 judge and one failed.
DPO nudges the policy toward judge-passing behaviour - verdict calibration,
cause-line accuracy, rubric coverage - without new supervised targets, which
is exactly the residual SFT plateaued on.

Env: BASE_MODEL (the SFT merged_hf dir), DATA (pairs jsonl), OUTPUT_DIR,
optional EPOCHS/LR/BETA/LORA_RANK. Renders prompts with the official no-think
chat template so training and serving see identical token streams.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def main() -> None:
    base = os.environ["BASE_MODEL"]
    data_path = Path(os.environ["DATA"])
    out = Path(os.environ["OUTPUT_DIR"])
    epochs = float(os.environ.get("EPOCHS", "2"))
    lr = float(os.environ.get("LR", "5e-6"))
    beta = float(os.environ.get("BETA", "0.1"))
    rank = int(os.environ.get("LORA_RANK", "16"))

    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in data_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def render(r: dict) -> dict:
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return {"prompt": prompt, "chosen": r["chosen"], "rejected": r["rejected"]}

    ds = Dataset.from_list([render(r) for r in rows])
    print(f"DPO pairs: {len(ds)} from {data_path}")

    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    peft_config = LoraConfig(
        r=rank, lora_alpha=rank * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    cfg = DPOConfig(
        output_dir=str(out),
        num_train_epochs=epochs,
        learning_rate=lr,
        beta=beta,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        max_length=1536,
        max_prompt_length=1024,
        report_to=[],
    )
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft_config)
    trainer.train()

    adapter = out / "adapter_final"
    trainer.model.save_pretrained(adapter)
    print("saved adapter:", adapter)

    # Merge for GGUF export: base (already the SFT merge) + DPO adapter.
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
