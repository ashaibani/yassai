"""Smoke-test the exported MiniCPM5 GGUF with llama.cpp on Modal CPU."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "yassai-minicpm5-gguf-smoke"
LLAMA_CPP_VERSION = "b9620"
MODEL_PATH = Path("/checkpoints/exact-e14-r32/gguf/MiniCPM5-yassai-Q4_K_M.gguf")

app = modal.App(APP_NAME)
ckpt_volume = modal.Volume.from_name("yassai-minicpm5-checkpoints", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates")
    .run_commands(
        "mkdir -p /opt/llama-bin",
        (
            "curl -fsSL -o /tmp/llama.tgz "
            f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_VERSION}/"
            f"llama-{LLAMA_CPP_VERSION}-bin-ubuntu-x64.tar.gz"
        ),
        "tar xzf /tmp/llama.tgz -C /opt/llama-bin --strip-components=1",
    )
)


@app.function(image=image, timeout=20 * 60, volumes={"/checkpoints": ckpt_volume})
def smoke() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    prompt = (
        "<s><|im_start|>system\n"
        "You are yassai-local, a small local specialist for math and logic tasks.\n"
        "Use run_python for arithmetic, percentages, schedules, projections, combinatorics, and constraint checks.\n"
        "Never do these calculations in your head.\n"
        "When a tool is needed, respond with exactly one tool call and no prose:\n"
        '<tool_call>{"name":"run_python","arguments":{"code":"..."}}</tool_call>\n'
        "The Python must compute from named variables and print concise final values.\n"
        "After receiving a run_python result, return only the final answer requested by the user."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. "
        "In Q2 it restocks 800 units. In Q3 it sells 640 units. "
        "How many units remain at the end of Q3?"
        "<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    env = {**os.environ, "LD_LIBRARY_PATH": "/opt/llama-bin:" + os.environ.get("LD_LIBRARY_PATH", "")}
    proc = subprocess.run(
        [
            "/opt/llama-bin/llama-cli",
            "-m",
            str(MODEL_PATH),
            "-p",
            prompt,
            "-n",
            "96",
            "-no-cnv",
            "--no-display-prompt",
            "--temp",
            "0",
            "--top-p",
            "1",
            "-c",
            "2048",
        ],
        check=True,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
    )
    output = proc.stdout
    ok = "<tool_call>" in output and "</tool_call>" in output and "run_python" in output and "2400" in output
    return {"ok": ok, "output_tail": output[-2000:]}


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(smoke.remote(), ensure_ascii=False, indent=2))
