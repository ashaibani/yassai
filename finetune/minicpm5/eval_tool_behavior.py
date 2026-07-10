#!/usr/bin/env python3
"""Evaluate MiniCPM5 adapter on yassai tool-call and final-answer behavior."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_tool_target(row: dict) -> bool:
    return "<tool_call>" in row["messages"][-1]["content"]


def prompt_messages(row: dict) -> list[dict]:
    return row["messages"][:-1]


def generate(model, tok, messages: list[dict], max_new_tokens: int = 512) -> str:
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    eos_token_id = [tok.eos_token_id]
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_token_id:
        eos_token_id.append(im_end_id)
    inputs = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(model.device)
    out = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tok.eos_token_id,
        eos_token_id=eos_token_id,
    )
    # MiniCPM5 marks <tool_call> and </tool_call> as special tokens, so
    # skip_special_tokens=True erases the exact wrapper we need to evaluate.
    text = tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=False).strip()
    return text.removesuffix("<|im_end|>").strip()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def extract_tool_json(got: str) -> tuple[dict | None, str]:
    m = TOOL_RE.search(got)
    raw = m.group(1) if m else got.strip()
    try:
        return json.loads(raw), "wrapped" if m else "raw"
    except Exception as exc:
        return None, f"bad tool JSON: {exc}"


def score_tool(expected: str, got: str) -> tuple[bool, str]:
    obj, source = extract_tool_json(got)
    if obj is None:
        return False, source
    if obj.get("name") != "run_python":
        return False, f"wrong tool: {obj.get('name')}"
    code = obj.get("arguments", {}).get("code", "")
    if not isinstance(code, str) or not code.strip():
        return False, "missing code"
    if "print" not in code:
        return False, "code does not print"
    return True, f"ok-{source}"


def score_final(expected: str, got: str) -> tuple[bool, str]:
    if norm(expected) == norm(got):
        return True, "exact"
    # The judge is semantic, but for this local gate require all key numbers and
    # names from the target to survive.
    tokens = re.findall(r"[A-Za-z]+|\$?\d+(?:,\d{3})*(?:\.\d+)?%?", expected)
    missing = [t for t in tokens if t not in got]
    if len(missing) <= 1:
        return True, "key-token"
    return False, "missing " + ", ".join(missing[:6])


def main() -> None:
    base = os.environ.get("BASE_MODEL", "openbmb/MiniCPM5-1B")
    adapter = os.environ.get("ADAPTER")
    data = Path(os.environ["DATA"])
    if not adapter:
        raise SystemExit("ADAPTER is required")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter).eval()

    rows = load_rows(data)
    passed = 0
    failures = []
    for i, row in enumerate(rows):
        expected = row["messages"][-1]["content"]
        got = generate(model, tok, prompt_messages(row), max_new_tokens=640)
        if is_tool_target(row):
            ok, reason = score_tool(expected, got)
        else:
            ok, reason = score_final(expected, got)
        passed += int(ok)
        if not ok:
            failures.append({"index": i, "reason": reason, "expected": expected, "got": got})
        print(json.dumps({"index": i, "ok": ok, "reason": reason, "got": got[:500]}, ensure_ascii=False))

    print(json.dumps({"passed": passed, "total": len(rows), "failures": failures}, ensure_ascii=False, indent=2))
    if passed != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
