#!/usr/bin/env python3
"""Teacher-generated SFT data for the assistant-lane families that cannot be
derived mechanically: sentiment, summarisation, factual.

Recipe (rejection-sampling distillation, the same shape as the reference
repo's data/label_dataset.py): a SPEC fixes the intended answer before any
text exists (sentiment label + aspect mix; summary themes + shape constraint;
curated factual topic), the teacher writes the source text and the answer,
and the glm-5p2 judge - the SAME judge realeval uses - filters. Factual
answers additionally need a second model to agree (the judge has no reference
to grade against, so cross-model agreement stands in for one).

Judge-rejected rows are dropped, not retried: with spec-fixed targets a
rejection means the teacher text drifted from the spec, and fresh sampling is
cheaper than repair.

Output rows: {"family", "prompt", "answer", "rubric"} - one JSON per line.
scripts/build_minicpm5_assist_data.py merges this cache with the parametric
families (ner, code_generation) into the final messages JSONL.

Usage:
  FIREWORKS_API_KEY=... python3 scripts/gen_assist_teacher_data.py \
      --out finetune/minicpm5/data/assist_teacher_raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
TEACHER = "accounts/fireworks/models/minimax-m3"
JUDGE = "accounts/fireworks/models/glm-5p2"
CHECKER = "accounts/fireworks/models/glm-5p2"  # factual cross-check

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def chat(model: str, prompt: str, max_tokens: int, temperature: float) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "reasoning_effort": "none",
    }).encode()
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ["FIREWORKS_API_KEY"],
            # Fireworks' WAF 403s the default Python-urllib agent.
            "User-Agent": "yassai-datagen/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.load(resp)
    return out["choices"][0]["message"]["content"].strip()


JUDGE_PROMPT = """You are a fast, decisive grader. Do not deliberate at length, do not count words, do not second-guess yourself. Grade the candidate in one short sentence, then stop.

Question: {q}

Known correct answer or rubric: {rubric}

Candidate: {cand}

Decide if the candidate correctly and adequately answers the question given the known correct answer/rubric above. Minor wording differences are fine; factual or logical errors are not. Give your verdict immediately - do not show detailed reasoning.

Respond in this exact format and nothing else:
<one short sentence> -> <CORRECT or INCORRECT>
JSON: {{"correct": true or false}}"""


def judge_pass(question: str, rubric: str, candidate: str) -> bool:
    reply = chat(JUDGE, JUDGE_PROMPT.format(q=question, rubric=rubric, cand=candidate), 300, 0.0)
    m = re.search(r"\{.*\}", reply, re.DOTALL)
    if not m:
        return False
    try:
        return bool(json.loads(m.group(0)).get("correct"))
    except json.JSONDecodeError:
        return False


# --------------------------------------------------------------------- specs

PRODUCTS = [
    "wireless earbuds", "an air fryer", "a standing desk", "a fitness tracker",
    "a robot vacuum", "an electric kettle", "a mechanical keyboard", "a baby monitor",
    "hiking boots", "a coffee grinder", "a graphics tablet", "an office chair",
    "a soundbar", "a dash cam", "an e-reader", "a food processor",
    "a bike light set", "a portable projector", "noise-cancelling headphones", "a smart doorbell",
]
SERVICES = [
    "a hotel stay", "a food delivery order", "a plumbing call-out", "an airline flight",
    "a car service appointment", "a broadband installation", "a streaming subscription",
    "an online course", "a gym membership", "a photo printing service",
]
POS_ASPECTS = ["build quality", "battery life", "customer support", "ease of setup", "value for money", "speed", "comfort", "sound quality", "packaging", "documentation"]
NEG_ASPECTS = ["late delivery", "confusing app", "flimsy hinge", "short cable", "noisy motor", "misleading listing photos", "long hold times", "firmware bugs", "scratchy fabric", "hidden fees"]

SUM_TOPICS = [
    "urban vertical farming", "community wind co-operatives", "hospital tele-triage systems",
    "regenerative agriculture", "modular housing construction", "public bike-share schemes",
    "school breakfast programmes", "port automation", "citizen science air-quality networks",
    "digital identity wallets", "textile recycling", "rural drone deliveries",
    "heat-pump retrofits", "open data in local government", "night-train revivals",
    "coral reef restoration", "warehouse robotics", "public library maker spaces",
    "car-free city centres", "grid-scale battery storage",
]

FACT_TOPICS = [
    "the difference between RAM and ROM", "why the sky appears blue",
    "the difference between TCP and UDP", "the three states of matter and their transitions",
    "the difference between a compiler and an interpreter", "how vaccines produce immunity",
    "the difference between weather and climate", "why leap years exist",
    "the difference between mass and weight", "how a rainbow forms",
    "the difference between series and parallel circuits", "what causes ocean tides",
    "the difference between viruses and bacteria", "how GPS positioning works",
    "the difference between HTTP and HTTPS", "why ice floats on water",
    "the difference between renewable and non-renewable energy", "how noise-cancelling headphones work",
    "the difference between RAM and storage", "what causes the seasons",
    "the difference between analogue and digital signals", "how a lithium-ion battery degrades",
    "the difference between bandwidth and latency", "why metals conduct electricity",
    "the difference between open-source and proprietary software", "how solar panels generate electricity",
    "the difference between a hurricane and a tornado", "why the Moon shows phases",
    "the difference between speed and velocity", "how yeast makes bread rise",
]


def sentiment_specs(rng: random.Random, n: int) -> list[dict]:
    specs = []
    kinds = ["Positive", "Negative", "Neutral", "mixed-positive", "mixed-negative", "sarcastic-negative"]
    for i in range(n):
        kind = kinds[i % len(kinds)]
        subject = rng.choice(PRODUCTS + SERVICES)
        pos = rng.sample(POS_ASPECTS, 2)
        neg = rng.sample(NEG_ASPECTS, 2)
        label = {"Positive": "Positive", "Negative": "Negative", "Neutral": "Neutral",
                 "mixed-positive": "Positive", "mixed-negative": "Negative",
                 "sarcastic-negative": "Negative"}[kind]
        specs.append({
            "family": "sentiment", "kind": kind, "subject": subject,
            "pos": pos, "neg": neg, "label": label,
            "container": rng.choice(["customer review", "tweet", "forum comment"]),
        })
    return specs


def summarisation_specs(rng: random.Random, n: int) -> list[dict]:
    specs = []
    for i in range(n):
        topic = SUM_TOPICS[i % len(SUM_TOPICS)]
        shape = rng.choice([
            ("exactly two sentences", 2, "sentences"),
            ("exactly three sentences", 3, "sentences"),
            ("exactly three bullet points", 3, "bullets"),
            ("exactly four bullet points", 4, "bullets"),
        ])
        specs.append({
            "family": "summarisation", "topic": topic,
            "shape_text": shape[0], "shape_n": shape[1], "shape_kind": shape[2],
            "themes": ["benefits or drivers", "problems or challenges", "responses, countermeasures or outlook"],
        })
    return specs


def factual_specs(rng: random.Random, n: int) -> list[dict]:
    return [{"family": "factual", "topic": FACT_TOPICS[i % len(FACT_TOPICS)],
             "style": rng.choice(["explain", "name-and-explain", "compare"])}
            for i in range(n)]


# ------------------------------------------------------------------ builders

def build_sentiment(spec: dict) -> dict | None:
    tone = {
        "Positive": "clearly positive overall",
        "Negative": "clearly negative overall",
        "Neutral": "genuinely balanced/neutral overall (praise and criticism in equal measure, no verdict)",
        "mixed-positive": f"positive OVERALL despite genuine complaints about {spec['neg'][0]}",
        "mixed-negative": f"negative OVERALL despite genuine praise for {spec['pos'][0]}",
        "sarcastic-negative": "negative, expressed through sarcasm (surface praise that is obviously mocking)",
    }[spec["kind"]]
    text = chat(TEACHER, (
        f"Write a realistic {spec['container']} about {spec['subject']} (2-4 sentences, first person, no headings). "
        f"The sentiment must be {tone}. Where natural, mention {spec['pos'][0]} and {spec['neg'][1]}. "
        "Output only the review text."
    ), 300, 0.9)
    prompt = (
        f"Classify the sentiment of this {spec['container']} as Positive, Negative, or Neutral, "
        f"and briefly explain your reasoning in one sentence.\n\n{text}"
    )
    answer = chat(TEACHER, (
        f"{prompt}\n\nThe correct label is {spec['label']}. Answer in exactly this shape: "
        f"the label, a full stop, then ONE sentence of reasoning that accurately reflects the text. "
        "Do not mention that you were told the label."
    ), 150, 0.2)
    rubric = f"{spec['label']}. The reasoning must accurately reflect the text's content ({tone})."
    if not answer.startswith(spec["label"]):
        return None
    if not judge_pass(prompt, rubric, answer):
        return None
    return {"family": "sentiment", "prompt": prompt, "answer": answer, "rubric": rubric}


def build_summarisation(spec: dict) -> dict | None:
    passage = chat(TEACHER, (
        f"Write a dense 150-190 word news-style passage about {spec['topic']}. It must clearly contain "
        f"three distinct strands: (1) {spec['themes'][0]}, (2) {spec['themes'][1]}, (3) {spec['themes'][2]}. "
        "No headings, no bullets. Output only the passage."
    ), 400, 0.9)
    prompt = f"Summarize the following passage in {spec['shape_text']}:\n\n{passage}"
    answer = chat(TEACHER, (
        f"{prompt}\n\nHard requirements: output {spec['shape_text']} and NOTHING else "
        f"({'one bullet per line starting with - ' if spec['shape_kind'] == 'bullets' else 'plain prose'}); "
        f"between them they must cover all three strands (benefits/drivers, problems/challenges, responses/outlook); "
        "never exceed the stated count."
    ), 260, 0.2)
    n = spec["shape_n"]
    if spec["shape_kind"] == "bullets":
        count = sum(1 for line in answer.splitlines() if line.strip().startswith(("-", "•", "*")))
    else:
        count = len([s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s])
    if count != n:
        return None
    rubric = (f"A summary in {spec['shape_text']} covering the passage's benefits/drivers, "
              f"problems/challenges, and responses/outlook. Wrong count = fail.")
    if not judge_pass(prompt, rubric, answer):
        return None
    return {"family": "summarisation", "prompt": prompt, "answer": answer, "rubric": rubric}


def build_factual(spec: dict) -> dict | None:
    style = {
        "explain": f"Explain {spec['topic']} in 2-3 sentences.",
        "name-and-explain": f"Briefly explain {spec['topic']}. Name the key terms involved, then explain in 2-3 sentences.",
        "compare": f"What is {spec['topic']}? Answer in 2-3 sentences.",
    }[spec["style"]]
    prompt = style
    answer = chat(TEACHER, (
        f"{prompt}\n\nAnswer directly and concisely (max 3 sentences), no preamble, "
        "only well-established textbook facts, no speculative claims or dates."
    ), 220, 0.2)
    check = chat(CHECKER, (
        f"Question: {prompt}\n\nAnswer: {answer}\n\n"
        "Is this answer factually correct and free of errors? Reply with exactly YES or NO followed by one sentence."
    ), 120, 0.0)
    if not check.upper().startswith("YES"):
        return None
    rubric = "A factually correct, concise explanation. Factual errors = fail."
    if not judge_pass(prompt, rubric, answer):
        return None
    return {"family": "factual", "prompt": prompt, "answer": answer, "rubric": rubric}


BUILDERS = {"sentiment": build_sentiment, "summarisation": build_summarisation, "factual": build_factual}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="finetune/minicpm5/data/assist_teacher_raw.jsonl")
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--sentiment", type=int, default=90)
    ap.add_argument("--summarisation", type=int, default=60)
    ap.add_argument("--factual", type=int, default=60)
    args = ap.parse_args()

    if not os.environ.get("FIREWORKS_API_KEY"):
        sys.exit("FIREWORKS_API_KEY is required")
    rng = random.Random(args.seed)
    specs = (sentiment_specs(rng, args.sentiment)
             + summarisation_specs(rng, args.summarisation)
             + factual_specs(rng, args.factual))
    rng.shuffle(specs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = kept = 0
    write_lock = threading.Lock()

    def run(spec: dict) -> None:
        nonlocal done, kept
        try:
            row = BUILDERS[spec["family"]](spec)
        except Exception as e:  # noqa: BLE001 - drop and continue, generation is cheap
            log(f"ERR {spec['family']}: {e}")
            row = None
        with write_lock:
            done += 1
            if row is not None:
                kept += 1
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if done % 10 == 0:
                log(f"{done}/{len(specs)} done, {kept} kept")

    # umans cap: max 3 concurrent Fireworks calls. Each builder makes its calls
    # sequentially, so 3 workers = at most 3 in flight.
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(run, specs))
    log(f"FINISHED: kept {kept}/{len(specs)} -> {out_path}")


if __name__ == "__main__":
    main()
