"""Validate the assist lane's stock+LoRA serve path on NATIVE amd64.

Same trap as modal_validate_stack.py: a GGUF that decodes cleanly on macOS
can emit garbage on Linux CPU kernels, and the Qwen3.5 switch adds a second
variable - the serve-time LoRA adapter. This spins up llama-server with the
EXACT production arguments from internal/localllm NewDirect (reasoning off,
flash-attn on, q8_0 KV, -ub 256, --lora) on a Modal amd64 CPU worker and
runs the boot canary plus a sentiment smoke through /v1/chat/completions.

Usage:
  uvx modal run finetune/minicpm5/modal_validate_assist_lora.py
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request

import modal

APP_NAME = "yassai-assist-lora-validate"
LLAMA_VERSION = "b9948"
REPO = "https://huggingface.co/ashaibani/yassai-minicpm5-local/resolve/main"
DEFAULT_MODEL_URL = f"{REPO}/Qwen_Qwen3.5-2B-Q4_K_M.gguf"
DEFAULT_LORA_URL = f"{REPO}/yassai-assist-e3-r32-q35-2b-lora-f16.gguf"

app = modal.App(APP_NAME)
ckpt_volume = modal.Volume.from_name("yassai-minicpm5-checkpoints", create_if_missing=False)

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


def _chat(base: str, system: str, user: str, max_tok: int = 160) -> str:
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


@app.function(image=image, cpu=2, memory=4096, timeout=20 * 60,
              volumes={"/checkpoints": ckpt_volume},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def validate(model_url: str = DEFAULT_MODEL_URL, lora_url: str = DEFAULT_LORA_URL,
             lora_volume_path: str = "") -> str:
    import os

    token = os.environ["HF_TOKEN"]
    for url, dest in [(model_url, "/model.gguf")]:
        subprocess.run(["curl", "-fsSL", "-H", f"Authorization: Bearer {token}",
                        "-o", dest, url], check=True)
    if lora_volume_path:
        subprocess.run(["cp", lora_volume_path, "/lora.gguf"], check=True)
    else:
        subprocess.run(["curl", "-fsSL", "-H", f"Authorization: Bearer {token}",
                        "-o", "/lora.gguf", lora_url], check=True)

    # Production serve args (internal/localllm NewDirect) at judge-VM scale:
    # 2 threads, ctx 2048.
    srv = subprocess.Popen(
        ["/opt/llama/llama-server", "-m", "/model.gguf", "--lora", "/lora.gguf",
         "--host", "127.0.0.1", "--port", "18099", "-t", "2", "-c", "2048",
         "-np", "1", "--no-webui", "--reasoning", "off",
         "--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
         "-ub", "256"],
        env={**os.environ, "LD_LIBRARY_PATH": "/opt/llama"},
    )
    base = "http://127.0.0.1:18099"
    try:
        for _ in range(180):
            try:
                urllib.request.urlopen(base + "/health", timeout=2)
                break
            except Exception:
                if srv.poll() is not None:
                    print("AMD64_ASSIST_VERDICT: BROKEN (server exited)")
                    return "BROKEN"
                time.sleep(1)
        else:
            print("AMD64_ASSIST_VERDICT: BROKEN (never healthy)")
            return "BROKEN"

        t0 = time.time()
        canary = _chat(base,
                       "Write exactly the code requested, nothing extra. Code only, minimal comments, no explanation or demos unless asked.",
                       "Write a Python function called add_two that returns its argument plus two.")
        t_canary = time.time() - t0
        print(f"canary ({t_canary:.1f}s): {canary[:200]!r}")

        t0 = time.time()
        sent = _chat(base,
                     "Classify the sentiment and justify it in one accurate sentence. Never contradict the text.",
                     "Classify the sentiment of this customer review as Positive, Negative, or Neutral and give a one-sentence reason:\n\n'Best pair of walking boots I have owned in thirty years - waterproof and comfortable from day one.'")
        t_sent = time.time() - t0
        print(f"sentiment ({t_sent:.1f}s): {sent[:200]!r}")

        ok = "def add_two" in canary and sent.startswith("Positive")
        verdict = "HEALTHY" if ok else "BROKEN"
        print(f"gen seconds: canary={t_canary:.1f} sentiment={t_sent:.1f} (2 threads, judge-VM scale)")
        print(f"AMD64_ASSIST_VERDICT: {verdict}")
        return verdict
    finally:
        srv.kill()
        srv.wait()


@app.local_entrypoint()
def main(model_url: str = DEFAULT_MODEL_URL, lora_url: str = DEFAULT_LORA_URL,
         lora_volume_path: str = "") -> None:
    print(validate.remote(model_url=model_url, lora_url=lora_url,
                          lora_volume_path=lora_volume_path))
