"""Convert the trained MiniCPM5 merged HF checkpoint in Modal to GGUF.

Usage:
  uvx modal run finetune/minicpm5/modal_export_gguf.py
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "yassai-minicpm5-gguf-export"
CKPT_ROOT = Path("/checkpoints")
# Converter, quantizer and the gguf python package must come from ONE llama.cpp
# tree, matched to the runtime libs the agent ships (Dockerfile LLAMA_VERSION).
# An unpinned PyPI `gguf` once drifted ahead of the b9620 converter and wrote
# scrambled tokenizer metadata (models emitted <|fim_middle|>/stripped JSON).
LLAMA_CPP_VERSION = "b9946"

app = modal.App(APP_NAME)
ckpt_volume = modal.Volume.from_name("yassai-minicpm5-checkpoints", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "git")
    .pip_install(
        "torch==2.7.1",
        "transformers==4.57.3",
        "safetensors==0.6.2",
        "sentencepiece==0.2.1",
        "protobuf==6.33.0",
        "numpy==2.4.6",
        "huggingface_hub>=0.34",
    )
    .run_commands(
        "mkdir -p /opt/llama.cpp /opt/llama-bin",
        (
            "curl -fsSL -o /tmp/llama.tgz "
            f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_VERSION}/"
            f"llama-{LLAMA_CPP_VERSION}-bin-ubuntu-x64.tar.gz"
        ),
        "tar xzf /tmp/llama.tgz -C /opt/llama-bin --strip-components=1",
        (
            "curl -fsSL -o /tmp/llama-src.tgz "
            f"https://github.com/ggml-org/llama.cpp/archive/refs/tags/{LLAMA_CPP_VERSION}.tar.gz"
        ),
        "tar xzf /tmp/llama-src.tgz -C /opt/llama.cpp --strip-components=1",
        # the version-matched gguf writer from the SAME tree as the converter
        "pip install /opt/llama.cpp/gguf-py",
    )
)


@app.function(
    image=image,
    timeout=60 * 60,
    volumes={"/checkpoints": ckpt_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],  # HF_TOKEN for --hf-repo pushes
)
def export(run_name: str = "exact-e14-r32", quant: str = "Q4_K_M", hf_repo: str = "") -> str:
    merged = CKPT_ROOT / run_name / "merged_hf"
    out_dir = CKPT_ROOT / run_name / "gguf"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (merged / "model.safetensors").exists():
        raise FileNotFoundError(f"missing merged model: {merged}")

    f16 = out_dir / "MiniCPM5-yassai-F16.gguf"
    qout = out_dir / f"MiniCPM5-yassai-{quant}.gguf"

    env = {**os.environ, "LD_LIBRARY_PATH": "/opt/llama-bin:" + os.environ.get("LD_LIBRARY_PATH", "")}
    subprocess.run(
        [
            "python",
            "/opt/llama.cpp/convert_hf_to_gguf.py",
            str(merged),
            "--outfile",
            str(f16),
            "--outtype",
            "f16",
        ],
        check=True,
        env=env,
    )
    subprocess.run(
        [
            "/opt/llama-bin/llama-quantize",
            str(f16),
            str(qout),
            quant,
        ],
        check=True,
        env=env,
    )

    ckpt_volume.commit()

    if hf_repo:
        # Push straight to the Hub (HF_TOKEN from the huggingface-secret) so
        # the CI image bake needs no manual download/re-upload hop.
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(hf_repo, repo_type="model", private=True, exist_ok=True)
        for path, suffix in ((qout, quant), (f16, "F16")):
            dest = f"MiniCPM5-yassai-{run_name}-{suffix}.gguf"
            api.upload_file(path_or_fileobj=str(path), path_in_repo=dest, repo_id=hf_repo)
            print(f"pushed to https://huggingface.co/{hf_repo}/resolve/main/{dest}")

    return str(qout)


@app.local_entrypoint()
def main(run_name: str = "exact-e14-r32", quant: str = "Q4_K_M", hf_repo: str = "") -> None:
    print(export.remote(run_name=run_name, quant=quant, hf_repo=hf_repo))
