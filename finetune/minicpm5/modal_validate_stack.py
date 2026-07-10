"""Validate the in-container local-model stack on NATIVE amd64 (Modal CPU).

The judging VM is linux/amd64; this Mac can only emulate amd64 (QEMU lies for
compute) and its arm64 container run of the ubuntu-arm64 llama libs decodes
garbage. Modal CPU workers are native amd64, so this is the truthful preflight:
same llama.cpp release, same runtime libs (libgomp1/libstdc++6), same GGUF the
CI image bakes, driven by the cross-compiled localmodeleval harness.

Usage:
  CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o finetune/minicpm5/localmodeleval-linux-amd64 ./cmd/localmodeleval
  uvx modal run finetune/minicpm5/modal_validate_stack.py
"""
from __future__ import annotations

import subprocess

import modal

APP_NAME = "yassai-localstack-validate"
LLAMA_VERSION = "b9946"
DEFAULT_GGUF_URL = "https://huggingface.co/ashaibani/yassai-minicpm5-local/resolve/main/MiniCPM5-yassai-v2e3b-Q4_K_M.gguf"

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("python:3.12-slim", add_python=None)
    .apt_install("curl", "ca-certificates", "libstdc++6", "libgomp1")
    .run_commands(
        "mkdir -p /llama-dist /opt/llama",
        (
            "curl -fsSL -o /tmp/llama.tgz "
            f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_VERSION}/"
            f"llama-{LLAMA_VERSION}-bin-ubuntu-x64.tar.gz"
        ),
        "tar xzf /tmp/llama.tgz -C /llama-dist --strip-components=1",
        "find /llama-dist -name '*.so*' -exec cp -a {} /opt/llama/ \\;",
        "find /llama-dist -name 'llama-server' -exec cp -a {} /opt/llama/ \\;",
    )
    .add_local_file("finetune/minicpm5/localmodeleval-linux-amd64", "/localmodeleval", copy=True)
    .run_commands("chmod +x /localmodeleval")
)


@app.function(
    image=image,
    cpu=8,
    memory=8192,
    timeout=30 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def validate(gguf_url: str = DEFAULT_GGUF_URL) -> str:
    import os

    token = os.environ["HF_TOKEN"]
    subprocess.run(
        ["curl", "-fsSL", "-H", f"Authorization: Bearer {token}",
         "-o", "/model.gguf", gguf_url],
        check=True,
    )
    proc = subprocess.run(
        ["/localmodeleval", "-lib", "/opt/llama", "-ft", "/model.gguf", "-threads", "8"],
        capture_output=True, text=True, timeout=25 * 60,
    )
    out = proc.stdout + proc.stderr
    print(out)
    # Healthy = the boot canary decoded coherently and at least one task was
    # accepted end-to-end (the binary exits non-zero when nothing passes).
    verdict = "HEALTHY" if proc.returncode == 0 and "boot canary passed" in out else "BROKEN"
    print(f"AMD64_STACK_VERDICT: {verdict}")
    return verdict


@app.local_entrypoint()
def main(gguf_url: str = DEFAULT_GGUF_URL) -> None:
    print(validate.remote(gguf_url=gguf_url))
