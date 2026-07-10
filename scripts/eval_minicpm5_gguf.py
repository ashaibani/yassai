#!/usr/bin/env python3
"""Compare base and yassai fine-tuned MiniCPM5 GGUFs with llama.cpp."""

from __future__ import annotations

import json
import contextlib
import io
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "base": ROOT / "models/minicpm5/MiniCPM5-1B-base-Q4_K_M.gguf",
    "finetuned": ROOT / "models/minicpm5/MiniCPM5-yassai-Q4_K_M.gguf",
}

SYSTEM = """You are yassai-local, a small local specialist for math and logic tasks.
Use run_python for arithmetic, percentages, schedules, projections, combinatorics, and constraint checks.
Never do these calculations in your head.
When a tool is needed, respond with exactly one tool call and no prose:
<tool_call>{"name":"run_python","arguments":{"code":"..."}}</tool_call>
The Python must compute from named variables and print concise final values.
After receiving a run_python result, return only the final answer requested by the user."""

CASES = [
    {
        "name": "warehouse_exact_tool",
        "kind": "tool",
        "prompt": (
            "A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. "
            "In Q2 it restocks 800 units. In Q3 it sells 640 units. "
            "How many units remain at the end of Q3?"
        ),
        "want": ["run_python", "2400", "0.37", "800", "640", "print"],
        "observation": "1672",
    },
    {
        "name": "train_exact_tool",
        "kind": "tool",
        "prompt": (
            "A train leaves City A at 08:00 travelling toward City B at 90 km/h. "
            "A second train leaves City B at 09:30 travelling toward City A at 110 km/h. "
            "The distance between the cities is 450 km. At what time do the trains meet, "
            "and how far from City A is the meeting point?"
        ),
        "want": ["run_python", "datetime", "450", "90", "110", "print"],
        "observation": "11:04:30\n276.75 km from City A",
    },
    {
        "name": "pet_logic_tool",
        "kind": "tool",
        "prompt": (
            "Emma, Liam, and Priya each own one pet: cat, dog, or parrot. "
            "Liam does not own a furred pet. Emma does not own the parrot. "
            "Priya does not own the dog. Who owns each pet?"
        ),
        "want": ["run_python", "itertools", "Emma", "Liam", "Priya", "print"],
        "observation": "Emma: dog; Liam: parrot; Priya: cat",
    },
    {
        "name": "warehouse_final",
        "kind": "final",
        "prompt": (
            "A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. "
            "In Q2 it restocks 800 units. In Q3 it sells 640 units. "
            "How many units remain at the end of Q3?\n\n"
            "run_python result:\n1672\n\nReturn the final answer only."
        ),
        "want": ["1672"],
    },
]

TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def render_prompt(user: str) -> str:
    return (
        "<s><|im_start|>system\n"
        + SYSTEM
        + "<|im_end|>\n<|im_start|>user\n"
        + user
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def run_model(model: Path, user: str, max_tokens: int = 192) -> tuple[str, float]:
    cmd = [
        "llama-completion",
        "-m",
        str(model),
        "-p",
        render_prompt(user),
        "-n",
        str(max_tokens),
        "-no-cnv",
        "--special",
        "--no-display-prompt",
        "--temp",
        "0",
        "--top-p",
        "1",
        "-c",
        "2048",
        "--device",
        "none",
        "--no-kv-offload",
        "--no-op-offload",
        "--fit",
        "off",
        "-ngl",
        "0",
    ]
    start = time.time()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    elapsed = time.time() - start
    text = proc.stdout.strip()
    if proc.returncode != 0:
        text += f"\n[stderr]\n{proc.stderr[-1000:]}"
    return text, elapsed


def execute_code(code: str) -> tuple[bool, str]:
    buf = io.StringIO()
    ns: dict[str, object] = {}
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, ns, ns)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, buf.getvalue().strip()


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def score(case: dict, text: str) -> tuple[bool, str]:
    if case["kind"] == "tool":
        m = TOOL_RE.search(text)
        if not m:
            return False, "missing wrapped tool call"
        try:
            obj = json.loads(m.group(1))
        except Exception as exc:
            return False, f"bad tool JSON: {exc}"
        if obj.get("name") != "run_python":
            return False, f"wrong tool {obj.get('name')!r}"
        code = obj.get("arguments", {}).get("code", "")
        missing = [w for w in case["want"] if w not in code and w not in text]
        if missing:
            return False, "missing " + ", ".join(missing)
        ran, observed = execute_code(code)
        if not ran:
            return False, f"code error: {observed}"
        if normalize(observed) != normalize(case["observation"]):
            return False, f"wrong observation: {observed!r}"
        return True, "ok"

    normalized = re.sub(r"\s+", " ", text)
    missing = [w for w in case["want"] if w not in normalized]
    return not missing, "ok" if not missing else "missing " + ", ".join(missing)


def main() -> None:
    for label, model in MODELS.items():
        if not model.exists():
            raise SystemExit(f"missing model: {model}")
        print(f"\n=== {label}: {model.name} ===")
        passed = 0
        for case in CASES:
            text, elapsed = run_model(model, case["prompt"], 96 if case["kind"] == "final" else 224)
            ok, reason = score(case, text)
            passed += int(ok)
            print(json.dumps(
                {
                    "case": case["name"],
                    "ok": ok,
                    "reason": reason,
                    "seconds": round(elapsed, 2),
                    "output": text[:800],
                },
                ensure_ascii=False,
            ))
        print(json.dumps({"model": label, "passed": passed, "total": len(CASES)}))


if __name__ == "__main__":
    main()
