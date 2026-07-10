#!/usr/bin/env python3
"""Evaluate the assistant-lane adapter on VERIFIABLE behaviour per family.

Runs the tuned model over a held-out assist dataset (built with a different
seed, so prompts are unseen) and checks each family programmatically - no
judge, so the gate is deterministic and free:

  ner              every gold span appears in the output
  code_generation  output parses, defines the requested function, and passes
                   the prompt's own worked example
  sentiment        output starts with the gold label
  summarisation    bullet/sentence count equals the prompt's stated N
  factual          non-empty, at most 4 sentences (content is teacher/judge
                   verified upstream; this only guards format collapse)

Env (same contract as eval_tool_behavior.py): BASE_MODEL, ADAPTER, DATA.
Exits 1 when any family lands under its floor - the export step must not run
on a regressed adapter.
"""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

FLOORS = {"ner": 0.80, "code_generation": 0.70, "sentiment": 0.75, "summarisation": 0.60, "factual": 0.90, "code_fix": 0.70}
PER_FAMILY = int(os.environ.get("EVAL_PER_FAMILY", "12"))


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def generate(model, tok, messages: list[dict], max_new_tokens: int = 384) -> str:
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    eos_token_id = [tok.eos_token_id]
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_token_id:
        eos_token_id.append(im_end_id)
    # transformers>=5 may return a BatchEncoding (dict-like) even with
    # return_tensors="pt"; normalise to input_ids tensor for generate().
    try:
        rendered = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
    except TypeError:
        rendered = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    if hasattr(rendered, "input_ids"):
        input_ids = rendered["input_ids"]
        attn = rendered.get("attention_mask") if hasattr(rendered, "get") else rendered["attention_mask"] if "attention_mask" in rendered else None
    elif isinstance(rendered, dict):
        input_ids = rendered["input_ids"]
        attn = rendered.get("attention_mask")
    else:
        input_ids = rendered
        attn = None
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
        eos_token_id=eos_token_id,
    )
    if attn is not None:
        gen_kwargs["attention_mask"] = attn.to(device)
    out = model.generate(**gen_kwargs)
    text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    return text.strip()


CALLED_RE = re.compile(r"called\s+`?([A-Za-z_]\w*)")
EXAMPLE_RE = re.compile(r"For example, (\w+\(.*?\)) should return (.+?)\.(?:\s|$)")
BULLET_SHAPE_RE = re.compile(r"exactly (two|three|four|five|\d+) (sentences|bullet points)")
WORDNUM = {"two": 2, "three": 3, "four": 4, "five": 5}


def check_ner(prompt: str, gold: str, got: str) -> bool:
    spans = [part.split(":", 1)[1].strip() for part in gold.split(";") if ":" in part]
    return all(s in got for s in spans)


def check_cg(prompt: str, gold: str, got: str) -> bool:
    m = CALLED_RE.search(prompt)
    if not m or f"def {m.group(1)}" not in got:
        return False
    code = got
    fence = re.search(r"```(?:python)?\s*\n(.*?)\n?```", got, re.S)
    if fence:
        code = fence.group(1)
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    ex = EXAMPLE_RE.search(prompt)
    if not ex:
        return True
    ns: dict = {}
    try:
        exec(code, ns)  # noqa: S102 - sandboxed enough for our own eval box
        got_val = eval(ex.group(1), ns)  # noqa: S307
        want_val = ast.literal_eval(ex.group(2))
    except Exception:
        return False

    def deep(x):
        return [deep(i) for i in x] if isinstance(x, (list, tuple)) else x

    return deep(got_val) == deep(want_val)


def check_sentiment(prompt: str, gold: str, got: str) -> bool:
    label = gold.split(".")[0].strip()
    return got.strip().startswith(label)


def check_summarisation(prompt: str, gold: str, got: str) -> bool:
    m = BULLET_SHAPE_RE.search(prompt)
    if not m:
        return bool(got.strip())
    n = WORDNUM.get(m.group(1), 0) or int(m.group(1))
    if m.group(2) == "bullet points":
        count = sum(1 for line in got.splitlines() if line.strip().startswith(("-", "•", "*")))
    else:
        count = len([s for s in re.split(r"(?<=[.!?])\s+", got.strip()) if s])
    return count == n


def check_factual(prompt: str, gold: str, got: str) -> bool:
    if not got.strip():
        return False
    return len([s for s in re.split(r"(?<=[.!?])\s+", got.strip()) if s]) <= 4


def check_code_fix(prompt: str, gold: str, got: str) -> bool:
    m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", prompt)
    if not m or f"def {m.group(1)}" not in got:
        return False
    def_idx = got.find("def ")
    if def_idx <= 0 or not got[:def_idx].strip():
        return False  # cause line must precede the code
    try:
        ast.parse(got[def_idx:])
    except SyntaxError:
        return False
    return " ".join(got[def_idx:].split()) != " ".join(prompt[prompt.find("def "):].split())


CHECKS = {
    "ner": check_ner,
    "code_fix": check_code_fix,
    "code_generation": check_cg,
    "sentiment": check_sentiment,
    "summarisation": check_summarisation,
    "factual": check_factual,
}


def main() -> None:
    base = os.environ.get("BASE_MODEL", "openbmb/MiniCPM5-1B")
    adapter = os.environ.get("ADAPTER")
    data = Path(os.environ["DATA"])
    if not adapter:
        raise SystemExit("ADAPTER is required")

    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    by_family: dict[str, list[dict]] = {}
    for row in load_rows(data):
        fam = row.get("family", "unknown")
        if fam in CHECKS and len(by_family.setdefault(fam, [])) < PER_FAMILY:
            by_family[fam].append(row)

    failed = False
    for fam, rows in sorted(by_family.items()):
        passed = 0
        for row in rows:
            gold = row["messages"][-1]["content"]
            prompt = row["messages"][1]["content"]
            got = generate(model, tok, row["messages"][:-1])
            ok = CHECKS[fam](prompt, gold, got)
            passed += ok
            if not ok:
                print(f"  FAIL {fam}: {prompt[:70]!r} -> {got[:90]!r}")
        rate = passed / len(rows) if rows else 0.0
        floor = FLOORS[fam]
        marker = "OK " if rate >= floor else "LOW"
        print(f"{marker} {fam}: {passed}/{len(rows)} ({rate:.0%}, floor {floor:.0%})")
        if rate < floor:
            failed = True
    if failed:
        raise SystemExit("assist behaviour eval under floor")


if __name__ == "__main__":
    main()
