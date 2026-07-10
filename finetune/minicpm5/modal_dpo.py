"""Modal runner for the assist-lane DPO calibration pass.

Usage:
  uvx modal run finetune/minicpm5/modal_dpo.py --base-run assist-e3-r32-v6 --tag v8 \
      --pairs-file assist_dpo_pairs_sentiment.jsonl

Trains a DPO LoRA on top of <base-run>/merged_hf using judge-archive
preference pairs, gates it with the assist behaviour eval, and leaves
<out-run>/merged_hf ready for modal_export_gguf.py.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "yassai-minicpm5-dpo"
REMOTE_ROOT = Path("/root/yassai")
CKPT_ROOT = Path("/checkpoints")

app = modal.App(APP_NAME)
ckpt_volume = modal.Volume.from_name("yassai-minicpm5-checkpoints", create_if_missing=False)
hf_cache = modal.Volume.from_name("yassai-hf-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "transformers==4.57.3",
        "trl==0.20.0",
        "peft==0.11.1",
        "datasets==3.6.0",
        "accelerate==1.11.0",
        "safetensors==0.6.2",
        "sentencepiece==0.2.1",
        "protobuf==6.33.0",
    )
    .add_local_file("finetune/minicpm5/train_dpo.py", remote_path=str(REMOTE_ROOT / "finetune/minicpm5/train_dpo.py"), copy=True)
    .add_local_file("finetune/minicpm5/eval_assist_behavior.py", remote_path=str(REMOTE_ROOT / "finetune/minicpm5/eval_assist_behavior.py"), copy=True)
    .add_local_file("finetune/minicpm5/data/assist_dpo_pairs.jsonl", remote_path=str(REMOTE_ROOT / "finetune/minicpm5/data/assist_dpo_pairs.jsonl"), copy=True)
    .add_local_file("finetune/minicpm5/data/assist_dpo_pairs_sentiment.jsonl", remote_path=str(REMOTE_ROOT / "finetune/minicpm5/data/assist_dpo_pairs_sentiment.jsonl"), copy=True)
    # Heldout for the post-DPO behaviour gate (built locally, judge-free checks).
    .add_local_file("scripts/build_minicpm5_assist_data.py", remote_path=str(REMOTE_ROOT / "scripts/build_minicpm5_assist_data.py"), copy=True)
    .add_local_file("finetune/minicpm5/data/assist_teacher_raw.jsonl", remote_path=str(REMOTE_ROOT / "finetune/minicpm5/data/assist_teacher_raw.jsonl"), copy=True)
    .add_local_file("finetune/minicpm5/data/assist_claude_authored.jsonl", remote_path=str(REMOTE_ROOT / "finetune/minicpm5/data/assist_claude_authored.jsonl"), copy=True)
)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 2,
    volumes={"/checkpoints": ckpt_volume, "/cache/huggingface": hf_cache},
)
def dpo(base_run: str = "assist-e3-r32-v6", tag: str = "v7", epochs: float = 2.0,
        lr: float = 5e-6, beta: float = 0.1, rank: int = 16,
        pairs_file: str = "assist_dpo_pairs.jsonl") -> str:
    base = CKPT_ROOT / base_run / "merged_hf"
    if not base.exists():
        raise SystemExit(f"base merged model missing: {base}")
    out_dir = CKPT_ROOT / f"assist-dpo-{tag}"

    env = {
        "BASE_MODEL": str(base),
        "DATA": str(REMOTE_ROOT / "finetune/minicpm5/data" / pairs_file),
        "OUTPUT_DIR": str(out_dir),
        "EPOCHS": str(epochs),
        "LR": str(lr),
        "BETA": str(beta),
        "LORA_RANK": str(rank),
        "HF_HOME": "/cache/huggingface",
    }
    subprocess.run(["python", str(REMOTE_ROOT / "finetune/minicpm5/train_dpo.py")], check=True, env={**os.environ, **env})

    # Behaviour gate on held-out prompts: DPO must not regress formats.
    heldout = REMOTE_ROOT / "finetune/minicpm5/data/assist_heldout.jsonl"
    subprocess.run(
        ["python", str(REMOTE_ROOT / "scripts/build_minicpm5_assist_data.py"),
         "--teacher", str(REMOTE_ROOT / "finetune/minicpm5/data/assist_teacher_raw.jsonl"),
         "--claude", str(REMOTE_ROOT / "finetune/minicpm5/data/assist_claude_authored.jsonl"),
         "--out", str(heldout), "--seed", "20269998", "--ner", "24",
         "--teacher-split", "heldout"],
        check=True,
    )
    eval_env = {
        **env,
        "BASE_MODEL": str(base),
        "ADAPTER": str(out_dir / "adapter_final"),
        "DATA": str(heldout),
    }
    subprocess.run(["python", str(REMOTE_ROOT / "finetune/minicpm5/eval_assist_behavior.py")], check=True, env={**os.environ, **eval_env})

    ckpt_volume.commit()
    hf_cache.commit()
    return str(out_dir)


@app.local_entrypoint()
def main(base_run: str = "assist-e3-r32-v6", tag: str = "v7", epochs: float = 2.0,
         lr: float = 5e-6, beta: float = 0.1, rank: int = 16,
         pairs_file: str = "assist_dpo_pairs.jsonl"):
    print(dpo.remote(base_run=base_run, tag=tag, epochs=epochs, lr=lr, beta=beta, rank=rank, pairs_file=pairs_file))
