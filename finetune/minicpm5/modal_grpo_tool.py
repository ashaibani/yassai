"""Modal runner for tool-lane clue-encoding GRPO (MiniCPM5 base).

Deliberately separate from modal_grpo.py: that image is pinned for Qwen3.5
(transformers 5.13 / trl 0.24, whose grpo_trainer hard-imports mergekit and
conflicts with 5.13 at resolve time). MiniCPM5 trains fine on the original
proven stack below, and the tool lane must not block on the Qwen session's
pin churn.

Usage:
  uvx modal run --detach finetune/minicpm5/modal_grpo_tool.py \
      --base-run v2-e3-r32 --tag cluev1 --epochs 3 --num-generations 6
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "yassai-minicpm5-grpo-tool"
REMOTE_ROOT = Path("/root/yassai")
CKPT_ROOT = Path("/checkpoints")

app = modal.App(APP_NAME)
ckpt_volume = modal.Volume.from_name("yassai-minicpm5-checkpoints", create_if_missing=False)
hf_cache = modal.Volume.from_name("yassai-hf-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        # The MiniCPM5-proven stack (same as the SFT/DPO images that produced
        # v2e3b and assist-v6): trl 0.20's GRPOTrainer imports cleanly.
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
    .add_local_file("finetune/minicpm5/train_grpo.py", remote_path=str(REMOTE_ROOT / "finetune/minicpm5/train_grpo.py"), copy=True)
    .add_local_file("finetune/minicpm5/rewards_rlvr.py", remote_path=str(REMOTE_ROOT / "finetune/minicpm5/rewards_rlvr.py"), copy=True)
    .add_local_file("finetune/minicpm5/data/logic_grpo_pool.jsonl", remote_path=str(REMOTE_ROOT / "finetune/minicpm5/data/logic_grpo_pool.jsonl"), copy=True)
    .add_local_file("scripts/build_minicpm5_sft_data_v2.py", remote_path=str(REMOTE_ROOT / "scripts/build_minicpm5_sft_data_v2.py"), copy=True)
    .add_local_file("finetune/minicpm5/eval_tool_behavior.py", remote_path=str(REMOTE_ROOT / "finetune/minicpm5/eval_tool_behavior.py"), copy=True)
)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 4,
    volumes={"/checkpoints": ckpt_volume, "/cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def grpo_tool(base_run: str = "v2-e3-r32", tag: str = "cluev1", epochs: float = 3.0,
              lr: float = 5e-6, rank: int = 16, num_generations: int = 6) -> str:
    base = CKPT_ROOT / base_run / "merged_hf"
    if not base.exists():
        raise SystemExit(f"base merged model missing: {base}")
    data_dir = REMOTE_ROOT / "finetune/minicpm5/data"
    out_dir = CKPT_ROOT / f"tool-grpo-{tag}"

    env = {
        "BASE_MODEL": str(base),
        "DATA": str(data_dir / "logic_grpo_pool.jsonl"),
        "OUTPUT_DIR": str(out_dir),
        "EPOCHS": str(epochs),
        "LR": str(lr),
        "LORA_RANK": str(rank),
        "NUM_GENERATIONS": str(num_generations),
        "HF_HOME": "/cache/huggingface",
        "PYTHONPATH": str(REMOTE_ROOT / "finetune/minicpm5"),
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    }
    subprocess.run(["python", str(REMOTE_ROOT / "finetune/minicpm5/train_grpo.py")], check=True, env={**os.environ, **env})

    # Behaviour gate on a fresh held-out v2 split: GRPO must not regress
    # tool-call formats or the maths families (the toolv3 lesson).
    heldout = data_dir / "minicpm5_yassai_v2_heldout.jsonl"
    subprocess.run(
        ["python", str(REMOTE_ROOT / "scripts/build_minicpm5_sft_data_v2.py"),
         "--out", str(heldout), "--seed", "20269999"],
        check=True,
    )
    eval_env = {**env, "ADAPTER": str(out_dir / "adapter_final"), "DATA": str(heldout)}
    subprocess.run(["python", str(REMOTE_ROOT / "finetune/minicpm5/eval_tool_behavior.py")], check=True, env={**os.environ, **eval_env})

    ckpt_volume.commit()
    hf_cache.commit()
    return str(out_dir)


@app.local_entrypoint()
def main(base_run: str = "v2-e3-r32", tag: str = "cluev1", epochs: float = 3.0,
         lr: float = 5e-6, rank: int = 16, num_generations: int = 6):
    print(grpo_tool.remote(base_run=base_run, tag=tag, epochs=epochs, lr=lr,
                           rank=rank, num_generations=num_generations))
