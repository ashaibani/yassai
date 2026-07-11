"""Re-run the tool behaviour gate on an existing SFT checkpoint (no retrain).

Usage:
  uvx modal run finetune/minicpm5/modal_tool_gate.py --run-name v2-e3-r32-q35-tool-v1 --base-model Qwen/Qwen3.5-2B
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "yassai-tool-behavior-gate"
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
        "transformers==5.13.0",
        "trl==0.24.0",
        "peft==0.19.1",
        "datasets==3.6.0",
        "accelerate>=1.11.0",
        "safetensors>=0.6.2",
        "sentencepiece==0.2.1",
        "protobuf==6.33.0",
        "huggingface_hub>=0.34",
    )
    .add_local_file("scripts/build_minicpm5_sft_data_v2.py", remote_path=str(REMOTE_ROOT / "scripts/build_minicpm5_sft_data_v2.py"), copy=True)
    .add_local_file("finetune/minicpm5/eval_tool_behavior.py", remote_path=str(REMOTE_ROOT / "finetune/minicpm5/eval_tool_behavior.py"), copy=True)
)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes={"/checkpoints": ckpt_volume, "/cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def gate(run_name: str = "v2-e3-r32-q35-tool-v1", base_model: str = "Qwen/Qwen3.5-2B") -> str:
    adapter = CKPT_ROOT / run_name / "adapter_final"
    if not adapter.exists():
        raise FileNotFoundError(adapter)
    data_dir = REMOTE_ROOT / "finetune/minicpm5/data"
    data_dir.mkdir(parents=True, exist_ok=True)
    heldout = data_dir / "minicpm5_yassai_v2_heldout.jsonl"
    subprocess.run(
        ["python", str(REMOTE_ROOT / "scripts/build_minicpm5_sft_data_v2.py"),
         "--out", str(heldout), "--seed", "20269999"],
        check=True,
    )
    env = {
        **os.environ,
        "BASE_MODEL": base_model,
        "ADAPTER": str(adapter),
        "DATA": str(heldout),
        "HF_HOME": "/cache/huggingface",
        "TRANSFORMERS_CACHE": "/cache/huggingface",
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "HUGGING_FACE_HUB_TOKEN": os.environ.get("HF_TOKEN", ""),
    }
    subprocess.run(
        ["python", str(REMOTE_ROOT / "finetune/minicpm5/eval_tool_behavior.py")],
        check=True,
        env=env,
    )
    return f"PASS {run_name} adapter={adapter}"


@app.local_entrypoint()
def main(run_name: str = "v2-e3-r32-q35-tool-v1", base_model: str = "Qwen/Qwen3.5-2B") -> None:
    print(gate.remote(run_name=run_name, base_model=base_model))
