#!/usr/bin/env python3
"""Build the GRPO prompt pool for the tool lane's clue-encoding gap.

The zero-token runs showed the tool lane confidently mis-encoding semantic
clues ("allergic to fur") into its enumeration code - a gate-passing, judge-
failing answer. GRPO with a verifiable reward attacks exactly that skill:
every pool row carries a parametric puzzle with a DERIVED unique solution,
and the reward executes the model's emitted run_python code and checks the
printed assignment against that truth (finetune/minicpm5/rewards_rlvr.py).

Semantic-clue density is deliberately high (~75% of rows) because plain
elimination clues are already solid SFT territory. Clue semantics are chosen
to be world-knowledge-clean (no arguable edge cases).

HARD RULE: fresh parametric content only - asserted against the golden,
variant, and wildcard task files.

Usage:
  python3 scripts/build_logic_grpo_pool.py \
      --out finetune/minicpm5/data/logic_grpo_pool.jsonl --n 56
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

# Byte-identical to internal/localllm/localllm.go systemPrompt (the serving
# and SFT contract for the tool lane).
SYSTEM = """You are yassai-local, a small local specialist for math and logic tasks.
Use run_python for arithmetic, percentages, schedules, projections, combinatorics, and constraint checks.
Never do these calculations in your head.
When a tool is needed, respond with exactly one tool call and no prose:
<tool_call>{"name":"run_python","arguments":{"code":"..."}}</tool_call>
The Python must compute from named variables and print concise final values.
After receiving a run_python result, return only the final answer requested by the user."""

FIRST_NAMES = ["Amira", "Bruno", "Chiara", "Dev", "Elin", "Farid", "Greta", "Hugo",
               "Ines", "Jonas", "Katya", "Liam", "Mona", "Nils", "Oona", "Pavel",
               "Quinn", "Rosa", "Stefan", "Tara", "Umar", "Vera", "Wren", "Yusuf"]

FURRY = {"cat", "dog", "hamster", "rabbit"}
FLYING = {"parrot", "canary"}
BULKY_INSTRUMENTS = {"piano", "drums"}
CAFFEINATED = {"coffee", "tea"}
CONTACT_SPORTS = {"boxing", "karate"}

# (domain, item pool, [(clue template, banned-or-required set builder, kind)])
DOMAINS = [
    ("pet", ["cat", "dog", "parrot", "hamster", "goldfish", "rabbit", "canary"], [
        ("{p} is allergic to fur.", lambda items: {i for i in items if i in FURRY}, "ban"),
        ("{p} wants a pet that can fly.", lambda items: {i for i in items if i in FLYING}, "require"),
    ]),
    ("instrument", ["guitar", "piano", "violin", "drums", "flute", "trumpet"], [
        ("{p} can only carry a small instrument on the bus.", lambda items: {i for i in items if i in BULKY_INSTRUMENTS}, "ban"),
    ]),
    ("beverage", ["coffee", "tea", "water", "juice", "lemonade"], [
        ("{p} is strictly caffeine-free.", lambda items: {i for i in items if i in CAFFEINATED}, "ban"),
    ]),
    ("sport", ["tennis", "rowing", "cycling", "boxing", "karate", "archery"], [
        ("{p} only does non-contact sports.", lambda items: {i for i in items if i in CONTACT_SPORTS}, "ban"),
    ]),
]


def gen_assignment(rng: random.Random, want_semantic: bool) -> dict | None:
    domain, pool, semantic_clues = rng.choice(DOMAINS)
    n = rng.choice([3, 4])
    people = rng.sample(FIRST_NAMES, n)
    items = rng.sample(pool, n)
    solution = dict(zip(people, rng.sample(items, n)))

    clue_texts: list[str] = []
    preds = []
    used_semantic = False

    fixed = rng.choice(people)
    verb = "drinks" if domain == "beverage" else ("plays" if domain in ("instrument", "sport") else "has")
    clue_texts.append(f"{fixed} {verb} the {solution[fixed]}." if domain != "beverage"
                      else f"{fixed} drinks {solution[fixed]}.")
    preds.append(lambda a, p=fixed, it=solution[fixed]: a[p] == it)

    for p in people:
        if p == fixed:
            continue
        placed = False
        if want_semantic and not used_semantic:
            rng.shuffle(semantic_clues)
            for tmpl, setof, kind in semantic_clues:
                affected = setof(items)
                if kind == "ban" and solution[p] not in affected and affected:
                    clue_texts.append(tmpl.format(p=p))
                    preds.append(lambda a, p=p, banned=frozenset(affected): a[p] not in banned)
                    used_semantic = placed = True
                    break
                if kind == "require" and solution[p] in affected and len(affected) < len(items):
                    clue_texts.append(tmpl.format(p=p))
                    preds.append(lambda a, p=p, req=frozenset(affected): a[p] in req)
                    used_semantic = placed = True
                    break
        if placed:
            continue
        wrongs = [i for i in items if i != solution[p]]
        k = rng.choice([1, 2]) if n == 4 else 1
        picked = rng.sample(wrongs, min(k, len(wrongs)))
        if len(picked) == 2:
            clue_texts.append(f"{p} has neither the {picked[0]} nor the {picked[1]}.")
        else:
            clue_texts.append(f"{p} does not have the {picked[0]}.")
        preds.append(lambda a, p=p, banned=frozenset(picked): a[p] not in banned)

    if want_semantic and not used_semantic:
        return None
    sols = [dict(zip(people, perm)) for perm in itertools.permutations(items)
            if all(pr(dict(zip(people, perm))) for pr in preds)]
    if len(sols) != 1 or sols[0] != solution:
        return None

    people_txt = ", ".join(people[:-1]) + f", and {people[-1]}"
    items_txt = ", ".join(items[:-1]) + f", or {items[-1]}"
    clues = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(clue_texts))
    count_word = {3: "Three", 4: "Four"}[n]
    prompt = (f"{count_word} friends — {people_txt} — each have a different "
              f"{domain}: {items_txt}. Use the following clues to determine who has what:\n{clues}\n"
              f"State each person's {domain}.")
    return {"family": "logic_tool", "system": SYSTEM, "prompt": prompt,
            "truth": json.dumps(solution, sort_keys=True)}


def gen_seating(rng: random.Random) -> dict | None:
    n = 4
    people = rng.sample(FIRST_NAMES, n)
    order = rng.sample(people, n)  # left -> right, positions 1..n
    pos = {p: i + 1 for i, p in enumerate(order)}

    clue_texts = []
    preds = []
    # 'immediately left of' is the positional-semantics skill (index diff == 1).
    a, b = order[0], order[1]
    clue_texts.append(f"{a} sits immediately to the left of {b}.")
    preds.append(lambda q, a=a, b=b: q[a] + 1 == q[b])
    mid = order[rng.choice([1, 2])]
    clue_texts.append(f"{mid} does not sit at either end.")
    preds.append(lambda q, m=mid: q[m] not in (1, n))
    anchor = rng.choice([p for p in people if p not in (a, b, mid)] or [order[-1]])
    clue_texts.append(f"{anchor} sits in seat {pos[anchor]}.")
    preds.append(lambda q, p=anchor, s=pos[anchor]: q[p] == s)

    sols = []
    for perm in itertools.permutations(people):
        q = {p: i + 1 for i, p in enumerate(perm)}
        if all(pr(q) for pr in preds):
            sols.append(q)
    if len(sols) != 1 or sols[0] != pos:
        return None

    people_txt = ", ".join(people[:-1]) + f", and {people[-1]}"
    clues = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(clue_texts))
    prompt = (f"Four friends — {people_txt} — sit in a row of four seats numbered 1 to 4 "
              f"from left to right. Use the clues to work out the seating:\n{clues}\n"
              f"State each person's seat number.")
    truth = {p: str(pos[p]) for p in people}
    return {"family": "logic_tool", "system": SYSTEM, "prompt": prompt,
            "truth": json.dumps(truth, sort_keys=True)}


def other_task_prompts() -> set[str]:
    texts = set()
    for path in ["testdata/downloads_tasks_golden.json", "testdata/variant_tasks_golden.json",
                 "testdata/wildcard_tasks.json", "testdata/sentiment_holdout_tasks.json"]:
        try:
            d = json.load(open(path))
        except FileNotFoundError:
            continue
        d = d if isinstance(d, list) else d["tasks"]
        texts.update(t["prompt"] for t in d)
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="finetune/minicpm5/data/logic_grpo_pool.jsonl")
    ap.add_argument("--n", type=int, default=56)
    ap.add_argument("--seed", type=int, default=20269997)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    forbidden = other_task_prompts()
    rows, seen = [], set()
    semantic_target = int(args.n * 0.75)
    seat_target = max(4, args.n // 8)

    while len(rows) < args.n:
        remaining = args.n - len(rows)
        n_sem = sum(1 for r in rows if "allergic" in r["prompt"] or "caffeine" in r["prompt"]
                    or "can fly" in r["prompt"] or "small instrument" in r["prompt"]
                    or "non-contact" in r["prompt"])
        n_seat = sum(1 for r in rows if "seat" in r["prompt"])
        if n_seat < seat_target and remaining > 0:
            row = gen_seating(rng)
        else:
            row = gen_assignment(rng, want_semantic=n_sem < semantic_target)
        if row is None or row["prompt"] in seen:
            continue
        assert row["prompt"] not in forbidden, "pool row collides with an eval/sample task"
        seen.add(row["prompt"])
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_sem = sum(1 for r in rows if any(k in r["prompt"] for k in
                ("allergic", "caffeine", "can fly", "small instrument", "non-contact")))
    n_seat = sum(1 for r in rows if "seat" in r["prompt"])
    print(f"wrote {len(rows)} pool rows -> {out} (semantic={n_sem}, seating={n_seat})")


if __name__ == "__main__":
    main()
