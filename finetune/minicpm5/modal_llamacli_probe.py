"""Discriminator: llama.cpp's own llama-cli vs our yzma harness on amd64.

If llama-cli decodes our GGUF coherently where the yzma harness emits garbage,
the defect is in the yzma usage path on Linux; if llama-cli is also garbage,
the GGUF x linux-llama.cpp combination is at fault. Also probes the BASE
MiniCPM5 GGUF as a control (our conversion vs any conversion).
"""
from __future__ import annotations

import subprocess

import modal

APP_NAME = "yassai-llamacli-probe"
LLAMA_VERSION = "b9946"
FT_URL = "https://huggingface.co/ashaibani/yassai-minicpm5-local/resolve/main/MiniCPM5-yassai-v2e3b-Q4_K_M.gguf"

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("python:3.12-slim", add_python=None)
    .apt_install("curl", "ca-certificates", "libstdc++6", "libgomp1")
    .run_commands(
        "mkdir -p /opt/llama-bin",
        (
            "curl -fsSL -o /tmp/llama.tgz "
            f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_VERSION}/"
            f"llama-{LLAMA_VERSION}-bin-ubuntu-x64.tar.gz"
        ),
        "tar xzf /tmp/llama.tgz -C /opt/llama-bin --strip-components=1",
    )
)

PROMPT = ("<|im_start|>system\nYou are yassai-local, a small local specialist for math and logic tasks.\n"
          "Use run_python for arithmetic.\n<|im_end|>\n<|im_start|>user\nWhat is 17 * 23? "
          "Return only the number.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


@app.function(
    image=image,
    cpu=8,
    memory=8192,
    timeout=20 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def probe() -> None:
    import os

    token = os.environ["HF_TOKEN"]
    subprocess.run(
        ["curl", "-fsSL", "-H", f"Authorization: Bearer {token}", "-o", "/ft.gguf", FT_URL],
        check=True,
    )
    env = {**os.environ, "LD_LIBRARY_PATH": "/opt/llama-bin/lib:/opt/llama-bin"}
    cli = None
    for cand in ("/opt/llama-bin/bin/llama-completion", "/opt/llama-bin/llama-completion"):
        if os.path.exists(cand):
            cli = cand
            break
    print("llama-cli at:", cli)
    cmd = [cli, "-m", "/ft.gguf", "-p", PROMPT, "-n", "200", "--temp", "0",
           "--special", "-no-cnv", "-st", "--no-display-prompt", "-t", "8"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=6 * 60,
                              env=env, stdin=subprocess.DEVNULL)
        out, err = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = (exc.stderr or b"").decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        print("TIMED OUT - partial output follows")
    print("=== llama-cli stdout ===")
    print(out[:1200])
    print("=== stderr tail ===")
    print(err[-800:])


@app.local_entrypoint()
def main() -> None:
    probe.remote()
