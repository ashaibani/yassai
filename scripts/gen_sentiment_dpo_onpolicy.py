#!/usr/bin/env python3
"""Build ON-POLICY sentiment DPO pairs from the assist model's own samples.

Judge-archive mining (mine_dpo_pairs.py) gave pairs on only 4 distinct
prompts and the resulting DPO (v7) memorised them without generalising
(fresh-holdout probe: v6 5/8, v7 5/8, identical failures). This generator
attacks the measured failure modes directly:

  - British understatement ('not bad at all' -> Positive)
  - double negation ("can't say I'm disappointed" -> Positive)
  - mixed-balanced reviews -> Neutral, not a confident verdict
  - sarcasm -> Negative (without over-firing on genuine surprise-praise)

For ~30 authored prompts with gold verdicts, we sample the CURRENT assist
model (temp 0 plus several temp 0.9 draws), keep its wrong-verdict answers
as `rejected` and its own correct answers as `chosen` (authored fallback
where it never gets one right). Verdicts are parsed locally - zero
Fireworks tokens.

HARD RULE: prompts are fresh authored content. Asserted against the golden
sample set, the variant set, AND the holdout eval set (the holdout must
stay eval-only or its verdict is meaningless).

Usage:
  python3 scripts/gen_sentiment_dpo_onpolicy.py \
      --model models/minicpm5/MiniCPM5-yassai-assist-v6-Q4_K_M.gguf \
      --out finetune/minicpm5/data/assist_dpo_pairs_sentiment.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

# Byte-identical to internal/localllm/direct.go directInstructions.
SYSTEM = "Classify the sentiment and justify it in one accurate sentence. Never contradict the text."

PROMPT_TMPL = (
    "Classify the sentiment of this customer review as Positive, Negative, or "
    "Neutral and give a one-sentence reason:\n\n'{text}'"
)

# (gold verdict, review text, authored fallback chosen answer)
CASES = [
    # --- British understatement -> Positive ---
    ("Positive", "Not a bad little kettle at all. Boils fast, doesn't drip, and the lid doesn't fight back. I'd buy it again without thinking twice.",
     "Positive. 'Not a bad ... at all' is understated approval, backed by fast boiling, no dripping, and the reviewer saying they would buy it again."),
    ("Positive", "I must say, this printer is really not bad. First one I've owned that just prints when asked. Quietly impressed.",
     "Positive. 'Really not bad' and 'quietly impressed' are understated praise for a printer that finally works reliably."),
    ("Positive", "Can't complain about this mattress, to be fair. Back pain gone within a week and the edges don't sag. Does what it says on the tin.",
     "Positive. 'Can't complain' and 'does what it says on the tin' are understated satisfaction, supported by the back pain disappearing."),
    ("Positive", "The soundbar is nothing to sniff at for the price. Dialogue is finally audible and the bass doesn't rattle the shelves. Quite pleased overall.",
     "Positive. 'Nothing to sniff at' and 'quite pleased' express approval of the clear dialogue and controlled bass."),
    ("Positive", "Honestly, not the worst decision I've made this year. The standing desk motor is smooth, remembers my heights, and my back thanks me daily.",
     "Positive. The wry understatement gives way to concrete praise: smooth motor, height memory, and daily relief for the reviewer's back."),
    ("Positive", "This rain jacket is far from useless. Kept me bone dry through a two-hour downpour and packs down to nothing. No complaints here.",
     "Positive. 'Far from useless' and 'no complaints' understate clear satisfaction: it kept the reviewer dry for two hours and packs small."),
    # --- double negation -> Positive ---
    ("Positive", "Ordered the meal kit expecting another letdown, but I can't say I was let down - portions generous, veg fresh, instructions actually made sense.",
     "Positive. 'Can't say I was let down' is a double negative meaning satisfied, backed by generous portions, fresh vegetables, and clear instructions."),
    ("Positive", "It wouldn't be true to say the headphones disappointed me. Battery outlasts my commute week and the hiss I feared never appeared.",
     "Positive. Saying it 'wouldn't be true' that they disappointed means the reviewer is pleased: long battery life and no hiss."),
    ("Positive", "I don't regret buying this blender for a second. Crushes ice without drama and cleans itself in thirty seconds.",
     "Positive. 'Don't regret ... for a second' expresses clear satisfaction, supported by effortless ice crushing and quick cleaning."),
    ("Positive", "Nobody could accuse this bike light of being dim. Drivers actually give me room now and the battery warning is impossible to miss.",
     "Positive. The negated accusation means the light is bright, and the reviewer credits it with safer passing distance and a clear battery warning."),
    ("Positive", "Can't fault the customer service, try as I might. Replacement arrived next day with a handwritten apology and a free filter.",
     "Positive. 'Can't fault' means the service was excellent: next-day replacement, an apology, and a free filter."),
    # --- mixed, genuinely balanced -> Neutral ---
    ("Neutral", "The e-reader's screen is gorgeous in sunlight and the battery lasts weeks, but the software freezes weekly and the store keeps logging me out.",
     "Neutral. Genuine praise for the screen and battery is balanced by equally serious software freezes and login problems, with no overall verdict given."),
    ("Neutral", "Lovely staff and the pastries are outstanding; shame the queue eats half your lunch break and half the tables wobble.",
     "Neutral. Strong praise for the staff and pastries is offset by long queues and wobbly tables, leaving the review balanced."),
    ("Neutral", "The campsite showers were spotless and hot, the pitches generous. The road noise, though, kept us up both nights and the shop shuts at four.",
     "Neutral. Clean hot showers and generous pitches are weighed against sleepless nights from road noise and a shop that shuts early - a balanced verdict."),
    ("Neutral", "Brilliant zoom lens for the money and autofocus locks instantly, but it weighs a tonne and hunts badly in low light.",
     "Neutral. The value and fast autofocus are balanced by the heavy weight and poor low-light performance, so the review is mixed."),
    ("Neutral", "The course content is genuinely excellent and the tutor knows her stuff; the platform, however, crashes on every quiz and support never replies.",
     "Neutral. Excellent content and tutoring are cancelled out by a crashing platform and unresponsive support, leaving the review balanced."),
    ("Neutral", "Gorgeous colour and the paint covered in one coat. It also took three days to dry and the smell lingered for a week.",
     "Neutral. One-coat coverage and lovely colour are balanced against the very slow drying and lingering smell, with no overall judgement stated."),
    # --- hedged / low-expectation openings that end Positive ---
    ("Positive", "I only bought this budget tripod as a stopgap, yet three shoots later the 'proper' one is still in its box. Steady, light, and the head is silk.",
     "Positive. The stopgap framing turns into genuine endorsement: the reviewer keeps choosing it over the expensive tripod for its steadiness and smooth head."),
    ("Positive", "Wasn't expecting much from a supermarket own-brand olive oil, but it's peppery, fresh, and half the price of my usual. Converted.",
     "Positive. Low expectations are overturned - 'converted' plus praise for the flavour and price makes this clearly positive."),
    ("Positive", "Bought the app under duress for a work trip. A month on I've planned two holidays with it and paid for premium voluntarily.",
     "Positive. Despite the reluctant start, voluntarily paying for premium and continued use show the reviewer is won over."),
    ("Positive", "Sceptical about robot mowers, and this one still surprised me: stripes like a groundsman, finds its dock every time, and the dog ignores it.",
     "Positive. The stated scepticism is overturned by admiring specifics - neat stripes, reliable docking, and no trouble with the dog."),
    # --- sarcasm -> Negative ---
    ("Negative", "What a triumph. The 'waterproof' speaker lasted one poolside afternoon before becoming a very ugly paperweight. Money well spent.",
     "Negative. 'What a triumph' and 'money well spent' are sarcastic: the supposedly waterproof speaker died after one afternoon."),
    ("Negative", "Ten out of ten for the diet app that logged me out every time I logged a meal. Really kept the calories off, since I gave up eating near my phone.",
     "Negative. The mock 'ten out of ten' is sarcasm about an app that constantly logged the reviewer out and made tracking impossible."),
    ("Negative", "Delighted to report the 'silent' dishwasher doubles as a jet engine. The neighbours now know our washing-up schedule. Marvellous.",
     "Negative. 'Delighted' and 'marvellous' are sarcastic complaints that the supposedly silent dishwasher is extremely loud."),
    ("Negative", "Thrilled with my new umbrella, which turned inside out at the first gust and speared a passing jogger. Sturdy British engineering at its finest.",
     "Negative. The mock thrill and 'engineering at its finest' are sarcasm about an umbrella that failed in the first gust."),
    # --- genuine negative with a token positive ---
    ("Negative", "The courier was friendly, I'll say that much. The wardrobe arrived with two panels cracked and the replacement came cracked as well.",
     "Negative. A friendly courier does not offset the product failing twice - both the original and the replacement arrived cracked."),
    ("Negative", "Nice packaging, shame about the contents. The candle tunnelled on the second burn and the 'sea breeze' scent is pure washing-up liquid.",
     "Negative. The compliment about packaging is a setup for the real verdict: the candle burns badly and smells wrong."),
    # --- plain positive / plain negative controls ---
    ("Positive", "Best pair of walking boots I've owned in thirty years. Waterproof, broken in from day one, and my ankles have never felt safer.",
     "Positive. Unqualified praise: the reviewer calls them the best in thirty years and lists comfort, waterproofing, and support."),
    ("Negative", "The subscription box was late three months running, two jars arrived smashed, and cancelling took four emails and a phone call.",
     "Negative. Repeated lateness, damaged items, and an obstructive cancellation process add up to an unambiguously bad experience."),
    ("Neutral", "The gym has brand-new machines and 24-hour access, but the changing rooms are grim and the induction was a hard sell for personal training.",
     "Neutral. New equipment and round-the-clock access are balanced by grim changing rooms and pushy sales tactics, so the review is mixed."),
]

VERDICT_RE = re.compile(r"\b(positive|negative|neutral)\b", re.IGNORECASE)


def parse_verdict(answer: str) -> str | None:
    m = VERDICT_RE.search(answer[:160])
    return m.group(1).capitalize() if m else None


def other_task_prompts() -> set[str]:
    texts: set[str] = set()
    for p in ["testdata/downloads_tasks_golden.json", "testdata/variant_tasks_golden.json",
              "testdata/sentiment_holdout_tasks.json"]:
        d = json.load(open(p))
        d = d if isinstance(d, list) else d["tasks"]
        for t in d:
            texts.add(t["prompt"])
    return texts


def chat(base: str, prompt: str, temp: float, seed: int, max_tok: int = 160) -> str:
    body = json.dumps({
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
        "temperature": temp,
        "seed": seed,
        "max_tokens": max_tok,
    }).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.load(resp)
    return (out["choices"][0]["message"].get("content") or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/minicpm5/MiniCPM5-yassai-assist-v6-Q4_K_M.gguf")
    ap.add_argument("--out", default="finetune/minicpm5/data/assist_dpo_pairs_sentiment.jsonl")
    ap.add_argument("--port", type=int, default=18093)
    ap.add_argument("--lib", default=os.environ.get("YZMA_LIB", os.path.expanduser("~/opt/llama")))
    ap.add_argument("--samples", type=int, default=5, help="temp-0.9 draws per prompt (plus one temp-0)")
    ap.add_argument("--max-rejected", type=int, default=2, help="max rejected per prompt")
    args = ap.parse_args()

    forbidden = other_task_prompts()

    srv = subprocess.Popen(
        [str(Path(args.lib) / "llama-server"), "-m", args.model,
         "--host", "127.0.0.1", "--port", str(args.port),
         "-c", "4096", "--threads", "6", "-ngl", "99", "--reasoning", "off"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{args.port}"
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(base + "/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        else:
            raise SystemExit("llama-server never became healthy")

        pairs, stats = [], {"always_right": 0, "never_right": 0, "split": 0}
        for i, (gold, text, fallback_chosen) in enumerate(CASES):
            prompt = PROMPT_TMPL.format(text=text)
            assert prompt not in forbidden, f"case {i} collides with an eval/sample task"
            draws = [(0.0, chat(base, prompt, 0.0, 7))]
            for k in range(args.samples):
                draws.append((0.9, chat(base, prompt, 0.9, 1000 + 31 * i + k)))

            correct = [a for _, a in draws if parse_verdict(a) == gold]
            wrong = []
            for _, a in draws:
                v = parse_verdict(a)
                if a and v is not None and v != gold and a not in wrong:
                    wrong.append(a)

            if not wrong:
                stats["always_right"] += 1
                continue
            if correct:
                chosen = min(correct, key=len)  # on-policy chosen
                stats["split"] += 1
            else:
                chosen = fallback_chosen  # authored chosen for the hard patterns
                stats["never_right"] += 1
            for rej in wrong[: args.max_rejected]:
                pairs.append({
                    "family": "sentiment", "system": SYSTEM, "prompt": prompt,
                    "chosen": chosen, "rejected": rej, "task_id": f"op_s{i:02d}",
                })
            print(f"[{i:02d}] gold={gold:8s} correct={len(correct)}/{len(draws)} pairs+={min(len(wrong), args.max_rejected)}")
    finally:
        srv.kill()
        srv.wait()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(pairs)} on-policy pairs -> {out}")
    print("prompt buckets:", json.dumps(stats))


if __name__ == "__main__":
    main()
