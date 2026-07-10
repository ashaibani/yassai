#!/usr/bin/env python3
"""Verifiable rewards for GRPO/RLVR over the assist-lane families.

These are Python ports of the evidence gates in internal/localllm/direct.go
(and the held-out checks in eval_assist_behavior.py). They intentionally score
ONLY properties we can check without a judge:

  code_generation  function name present + ast.parse + worked-example exec
  code_fix          cause-line present + corrected def parses
  ner              high-precision entity spans recalled + label-shape sanity
  summarisation    bullet/sentence count matches the prompt's stated N
  sentiment        leading label is one of {Positive,Negative,Neutral,Mixed}
  factual          non-empty, <=4 sentences (content needs a judge component)

Reward hacking defence: every family mixes a *content* check with a *format*
check. Pure format gaming (e.g. always emitting 'Positive. ok') scores at most
the format half. For sentiment/factual, call sites should add a small
judge-scored component (see judge_bonus) under the Fireworks concurrency cap.

HARD RULE: never score against sample/live golden prompts. Build the RL prompt
pool from scripts/build_minicpm5_assist_data.py heldout/train splits only.
"""

from __future__ import annotations

import ast
import re
from typing import Callable

CALLED_RE = re.compile(r"called\s+`?([A-Za-z_]\w*)")
EXAMPLE_RE = re.compile(r"For example, (\w+\(.*?\)) should return (.+?)\.(?:\s|$)")
BULLET_SHAPE_RE = re.compile(r"exactly (two|three|four|five|\d+) (sentences|bullet points)")
WORDNUM = {"two": 2, "three": 3, "four": 4, "five": 5}
SENT_LABELS = ("Positive", "Negative", "Neutral", "Mixed")
ENTITY_LINE_RE = re.compile(r"([A-Za-z_ ]+):\s*([^;]+)")


def _code_body(answer: str) -> str:
    fence = re.search(r"```(?:python)?\s*\n(.*?)\n?```", answer, re.S)
    return fence.group(1) if fence else answer


def reward_code_generation(prompt: str, answer: str) -> float:
    m = CALLED_RE.search(prompt)
    if not m:
        return 0.0
    name = m.group(1)
    code = _code_body(answer)
    score = 0.0
    if f"def {name}" in code:
        score += 0.25
    try:
        ast.parse(code)
        score += 0.25
    except SyntaxError:
        return score
    ex = EXAMPLE_RE.search(prompt)
    if not ex:
        return score + 0.25  # parse+name only; no example to ground
    ns: dict = {}
    try:
        exec(code, ns)  # noqa: S102 - training-time reward on our own prompts
        got = eval(ex.group(1), ns)  # noqa: S307
        want = ast.literal_eval(ex.group(2))

        def deep(x):
            return [deep(i) for i in x] if isinstance(x, (list, tuple)) else x

        if deep(got) == deep(want):
            score += 0.5
    except Exception:
        pass
    return min(score, 1.0)


def reward_code_fix(prompt: str, answer: str) -> float:
    # Cause line = first non-empty line that is not a def/import/fence.
    lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    score = 0.0
    cause = next((ln for ln in lines if not ln.startswith(("def ", "import ", "from ", "```", "#"))), "")
    if cause and len(cause.split()) >= 4 and "def " not in cause:
        score += 0.35
    code = _code_body(answer)
    if "def " in code:
        score += 0.15
        try:
            ast.parse(code)
            score += 0.25
        except SyntaxError:
            return score
    # Bonus if the prompt's buggy symptom is mentioned in the cause line.
    for kw in ("off-by", "index", "range", "return", "None", "empty", "zero", "==", "!="):
        if kw.lower() in cause.lower():
            score += 0.1
            break
    return min(score, 1.0)


def reward_ner(prompt: str, answer: str) -> float:
    # Recall high-precision Titlecase multi-word / acronym spans from the prompt.
    # Mirrors gateNER's conservative signals, not a full NER gold set.
    candidates = set(re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", prompt))
    candidates |= set(re.findall(r"\b([A-Z]{2,}(?:\s+[A-Z]{2,})?)\b", prompt))
    # Drop common sentence starters that look like entities.
    candidates = {c for c in candidates if c.lower() not in {"the", "a", "an"} and len(c) > 2}
    if not candidates:
        return 0.5 if ":" in answer else 0.0
    hit = sum(1 for c in candidates if c in answer)
    recall = hit / max(len(candidates), 1)
    # Label-shape: at least one "Type: span" pair.
    shape = 1.0 if ENTITY_LINE_RE.search(answer) else 0.0
    return 0.7 * recall + 0.3 * shape


def reward_summarisation(prompt: str, answer: str) -> float:
    m = BULLET_SHAPE_RE.search(prompt)
    if not m:
        return 0.5 if answer.strip() else 0.0
    n = WORDNUM.get(m.group(1), 0) or int(m.group(1))
    if m.group(2) == "bullet points":
        count = sum(1 for ln in answer.splitlines() if re.match(r"^\s*([-*•]|\d+[.)])\s+", ln))
    else:
        count = len([s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()])
    if count == n:
        shape = 1.0
    elif abs(count - n) == 1:
        shape = 0.4
    else:
        shape = 0.0
    # Light content signal: answer should not be mostly the prompt echoed.
    overlap = len(set(answer.lower().split()) & set(prompt.lower().split()))
    content = 0.5 if overlap < max(8, len(prompt.split()) // 2) and len(answer.split()) >= n * 3 else 0.2
    return 0.7 * shape + 0.3 * content


def reward_sentiment(prompt: str, answer: str) -> float:
    ans = answer.strip()
    if not ans:
        return 0.0
    label_ok = any(ans.startswith(lab) for lab in SENT_LABELS)
    has_reason = "." in ans and len(ans.split()) >= 6
    # Format half only - content half needs a judge or gold label.
    return (0.4 if label_ok else 0.0) + (0.2 if has_reason else 0.0)


def reward_factual(prompt: str, answer: str) -> float:
    ans = answer.strip()
    if not ans:
        return 0.0
    sents = [s for s in re.split(r"(?<=[.!?])\s+", ans) if s.strip()]
    if not sents:
        return 0.0
    score = 0.3  # non-empty
    if len(sents) <= 4:
        score += 0.3
    if len(ans.split()) <= 120:
        score += 0.1
    return score  # content needs judge_bonus


REWARD_BY_FAMILY: dict[str, Callable[[str, str], float]] = {
    "code_generation": reward_code_generation,
    "code_fix": reward_code_fix,
    "ner": reward_ner,
    "summarisation": reward_summarisation,
    "sentiment": reward_sentiment,
    "factual": reward_factual,
}


# --- tool-lane logic reward (clue encoding) ---
# The zero-token gap: the tool lane mis-encodes semantic clues into its
# enumeration code and ships a confident wrong assignment. Reward = execute
# the model's emitted run_python code and check the printed assignment against
# the puzzle's derived truth. Anti-hack: a person scores only if EVERY
# "Name: item" occurrence in stdout names their true item, so printing all
# candidate assignments scores zero for contradicted people.

TOOLCALL_CODE_RE = re.compile(r'"code"\s*:\s*"((?:[^"\\]|\\.)*)"', re.S)


def _extract_tool_code(answer: str) -> str:
    i = answer.find("<tool_call>")
    seg = answer[i:] if i >= 0 else answer
    m = TOOLCALL_CODE_RE.search(seg)
    if not m:
        return ""
    try:
        import json as _json
        return _json.loads('"' + m.group(1) + '"')
    except Exception:
        return ""


def reward_logic_tool(prompt: str, answer: str, truth: dict | None) -> float:
    if not truth:
        return 0.0
    code = _extract_tool_code(answer)
    if not code.strip():
        return 0.0
    score = 0.1  # parseable tool call in the trained contract
    import subprocess
    try:
        proc = subprocess.run(["python3", "-"], input=code, capture_output=True,
                              text=True, timeout=5)
    except Exception:
        return score
    out = proc.stdout.strip()
    if proc.returncode != 0 or not out:
        return score
    score += 0.15  # executes cleanly and prints something
    correct = 0
    for person, item in truth.items():
        hits = re.findall(re.escape(person) + r"\s*[:\-]\s*([A-Za-z0-9 ]+)", out)
        vals = {h.strip().strip('.').lower() for h in hits}
        if vals and vals == {str(item).strip().lower()}:
            correct += 1
    score += 0.75 * (correct / max(1, len(truth)))
    return score


def reward_for(family: str, prompt: str, answer: str, judge_bonus: float = 0.0,
               truth: dict | None = None) -> float:
    """Composite reward in [0, 1+]. judge_bonus is optional (0..0.4 recommended)."""
    if family == "logic_tool":
        return reward_logic_tool(prompt, answer, truth)
    fn = REWARD_BY_FAMILY.get(family)
    base = fn(prompt, answer) if fn else 0.0
    # Cap judge influence so format/content gates still dominate.
    return min(base + max(0.0, min(judge_bonus, 0.4)), 1.4)


def self_test() -> None:
    cg_prompt = (
        "Write a Python function called `add` that returns the sum of two numbers. "
        "For example, add(2, 3) should return 5."
    )
    assert reward_code_generation(cg_prompt, "def add(a, b):\n    return a + b\n") >= 0.9
    assert reward_code_generation(cg_prompt, "def add(a, b):\n    return a - b\n") < 0.6
    assert reward_sentiment("x", "Positive. The review praises the product.") >= 0.5
    assert reward_summarisation("Summarise in exactly three bullet points.", "- a\n- b\n- c") >= 0.7
    print("rewards_rlvr self_test OK")


if __name__ == "__main__":
    self_test()
