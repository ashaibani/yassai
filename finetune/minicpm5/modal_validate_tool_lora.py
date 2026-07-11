"""Validate the tool lane's stock+LoRA serve path on NATIVE amd64.

Same trap as modal_validate_assist_lora.py / modal_validate_stack.py: a GGUF
that decodes cleanly on macOS can emit garbage on Linux CPU kernels, and the
Qwen3.5 tool-lane switch adds the serve-time LoRA variable. This spins up
llama-server with the EXACT production arguments from internal/localllm New()
(reasoning off, flash-attn on, q8_0 KV, -ub 256, --lora) on a Modal amd64 CPU
worker and runs a maths tool-call canary through /v1/chat/completions.

Usage:
  uvx modal run finetune/minicpm5/modal_validate_tool_lora.py \
      --lora-url https://huggingface.co/.../yassai-v2-e3-r32-q35-tool-v1-lora-f16.gguf
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request

import modal

APP_NAME = "yassai-tool-lora-validate"
LLAMA_VERSION = "b9948"
REPO = "https://huggingface.co/ashaibani/yassai-minicpm5-local/resolve/main"
DEFAULT_MODEL_URL = f"{REPO}/Qwen_Qwen3.5-2B-Q4_K_M.gguf"
# Filled after export; pass --lora-url for a not-yet-promoted artefact.
DEFAULT_LORA_URL = f"{REPO}/yassai-v2-e3-r32-q35-tool-v1-lora-f16.gguf"

# Byte-identical to internal/localllm systemPrompt / build_minicpm5_sft_data_v2.SYSTEM.
TOOL_SYSTEM = (
    "You are yassai-local, a small local specialist for math and logic tasks.\n"
    "Use run_python for arithmetic, percentages, schedules, projections, combinatorics, and constraint checks.\n"
    "Never do these calculations in your head.\n"
    "When a tool is needed, respond with exactly one tool call and no prose:\n"
    '<tool_call>{"name":"run_python","arguments":{"code":"..."}}</tool_call>\n'
    "The Python must compute from named variables and print concise final values.\n"
    "After receiving a run_python result, return only the final answer requested by the user."
)

MATH_CANARY = (
    "A depot starts with 500 units. It ships 20% of stock, then receives 50 units. "
    "How many units remain?"
)

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
)


def _chat(base: str, system: str, user: str, max_tok: int = 512) -> str:
    body = json.dumps({
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max_tok,
    }).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        out = json.load(resp)
    return (out["choices"][0]["message"].get("content") or "").strip()


def _health_ok(url: str) -> bool:
    """llama-server returns 503-JSON while loading; curl exits 0 on it, so
    require HTTP 200 (body often contains \"ok\")."""
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


@app.function(image=image, cpu=2, memory=4096, timeout=20 * 60,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def validate(model_url: str = DEFAULT_MODEL_URL, lora_url: str = DEFAULT_LORA_URL) -> str:
    import os

    token = os.environ["HF_TOKEN"]
    for url, dest in [(model_url, "/model.gguf"), (lora_url, "/lora.gguf")]:
        subprocess.run(["curl", "-fsSL", "-H", f"Authorization: Bearer {token}",
                        "-o", dest, url], check=True)

    # Production serve args (internal/localllm New) at judge-VM scale.
    srv = subprocess.Popen(
        ["/opt/llama/llama-server", "-m", "/model.gguf", "--lora", "/lora.gguf",
         "--host", "127.0.0.1", "--port", "18098", "-t", "2", "-c", "2048",
         "-np", "1", "--no-webui", "--reasoning", "off",
         "--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
         "-ub", "256"],
        env={**os.environ, "LD_LIBRARY_PATH": "/opt/llama"},
    )
    base = "http://127.0.0.1:18098"
    try:
        for _ in range(180):
            if _health_ok(base + "/health"):
                break
            if srv.poll() is not None:
                print("AMD64_TOOL_VERDICT: BROKEN (server exited)")
                return "BROKEN"
            time.sleep(1)
        else:
            print("AMD64_TOOL_VERDICT: BROKEN (never healthy)")
            return "BROKEN"

        t0 = time.time()
        canary = _chat(base, TOOL_SYSTEM, MATH_CANARY, max_tok=512)
        t_canary = time.time() - t0
        print(f"math canary ({t_canary:.1f}s): {canary[:400]!r}")

        # Contract checks: must emit a parseable run_python tool call, not
        # degenerate prose/backtick spam. Correctness is owned by the agent
        # gates; here we only prove the engine + LoRA decode coherently.
        has_tool = "<tool_call>" in canary and "run_python" in canary
        has_code = "print" in canary or "units" in canary.lower() or "stock" in canary.lower()
        degenerate = (
            not canary
            or len(canary) > 1600
            or canary.count("```python") > 2
            or '"\\n"\\n"' in canary
        )
        ok = has_tool and has_code and not degenerate
        verdict = "HEALTHY" if ok else "BROKEN"
        print(f"checks: tool={has_tool} codeish={has_code} degenerate={degenerate}")
        print(f"gen seconds: canary={t_canary:.1f} (2 threads, judge-VM scale)")
        print(f"AMD64_TOOL_VERDICT: {verdict}")
        return verdict
    finally:
        srv.kill()
        srv.wait()


@app.local_entrypoint()
def main(model_url: str = DEFAULT_MODEL_URL, lora_url: str = DEFAULT_LORA_URL) -> None:
    print(validate.remote(model_url=model_url, lora_url=lora_url))
