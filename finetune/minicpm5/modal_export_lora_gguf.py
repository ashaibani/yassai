"""Convert a PEFT LoRA adapter to GGUF (llama.cpp) for serving with --lora.

Usage:
  uvx modal run finetune/minicpm5/modal_export_lora_gguf.py \
      --run-name assist-e3-r32-q35-2b --base-model Qwen/Qwen3.5-2B
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "yassai-lora-gguf-export"
CKPT_ROOT = Path("/checkpoints")
LLAMA_CPP_VERSION = "b9948"

app = modal.App(APP_NAME)
ckpt_volume = modal.Volume.from_name("yassai-minicpm5-checkpoints", create_if_missing=False)
hf_cache = modal.Volume.from_name("yassai-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "git")
    .pip_install(
        "torch==2.7.1",
        "transformers==5.13.0",
        "peft==0.19.1",
        "safetensors>=0.6.2",
        "sentencepiece==0.2.1",
        "protobuf==6.33.0",
        "numpy==2.4.6",
        "huggingface_hub>=0.34",
    )
    .run_commands(
        "mkdir -p /opt/llama.cpp",
        (
            "curl -fsSL -o /tmp/llama-src.tgz "
            f"https://github.com/ggml-org/llama.cpp/archive/refs/tags/{LLAMA_CPP_VERSION}.tar.gz"
        ),
        "tar xzf /tmp/llama-src.tgz -C /opt/llama.cpp --strip-components=1",
        "pip install /opt/llama.cpp/gguf-py",
    )
)


@app.function(
    image=image,
    timeout=60 * 60,
    volumes={"/checkpoints": ckpt_volume, "/cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def export_lora(run_name: str = "assist-e3-r32-q35-2b", base_model: str = "Qwen/Qwen3.5-2B") -> str:
    adapter = CKPT_ROOT / run_name / "adapter_final"
    out_dir = CKPT_ROOT / run_name / "gguf"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"yassai-{run_name}-lora-f16.gguf"
    if not adapter.exists():
        raise FileNotFoundError(adapter)

    env = {
        **os.environ,
        "HF_HOME": "/cache/huggingface",
        "TRANSFORMERS_CACHE": "/cache/huggingface",
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "HUGGING_FACE_HUB_TOKEN": os.environ.get("HF_TOKEN", ""),
    }

    # convert_lora_to_gguf expects a LOCAL base directory, not a Hub id.
    # Volume paths (adapters trained on an in-volume SFT merge, e.g.
    # /checkpoints/v2-e3-r32/merged_hf) pass straight through.
    if base_model.startswith("/"):
        base_dir = base_model
    else:
        from huggingface_hub import snapshot_download

        base_dir = snapshot_download(
            repo_id=base_model,
            token=os.environ.get("HF_TOKEN") or None,
            cache_dir="/cache/huggingface",
        )
    print("base_dir", base_dir)

    conv = Path("/opt/llama.cpp/convert_lora_to_gguf.py")
    if not conv.exists():
        candidates = list(Path("/opt/llama.cpp").rglob("convert_lora_to_gguf.py"))
        if not candidates:
            raise FileNotFoundError("convert_lora_to_gguf.py not found")
        conv = candidates[0]

    cmd = [
        "python",
        str(conv),
        str(adapter),
        "--outfile",
        str(out),
        "--outtype",
        "f16",
        "--base",
        base_dir,
    ]
    print("running", cmd)
    subprocess.run(cmd, check=True, env=env)
    ckpt_volume.commit()
    hf_cache.commit()
    print("wrote", out, "size", out.stat().st_size)
    return str(out)


@app.local_entrypoint()
def main(run_name: str = "assist-e3-r32-q35-2b", base_model: str = "Qwen/Qwen3.5-2B") -> None:
    print(export_lora.remote(run_name=run_name, base_model=base_model))
