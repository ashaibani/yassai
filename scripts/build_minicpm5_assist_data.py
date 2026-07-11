#!/usr/bin/env python3
"""Build the assistant-lane SFT dataset (direct answers, no tool contract) for
the second local model: ner + code_generation derived mechanically here, and
sentiment/summarisation/factual merged from the judge-filtered teacher cache
(scripts/gen_assist_teacher_data.py).

Every parametric row is verified while building: NER spans are asserted to
appear exactly once in their sentence, and every code_generation reference is
executed against its tests. The system prompts MUST stay byte-identical to
internal/localllm/direct.go's directInstructions - the model is trained on the
exact render it will be served with.

Usage:
  python3 scripts/build_minicpm5_assist_data.py \
      --teacher finetune/minicpm5/data/assist_teacher_raw.jsonl \
      --out finetune/minicpm5/data/minicpm5_yassai_assist.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

# Keep byte-identical to internal/localllm/direct.go directInstructions.
SYSTEM = {
    "code_generation": "Write exactly the code requested, nothing extra. Code only, minimal comments, no explanation or demos unless asked.",
    "ner": "Extract the entities in exactly the requested format, nothing else. Include every entity present; omissions are failures.",
    "sentiment": "Classify the sentiment and justify it in one accurate sentence. Never contradict the text.",
    "summarisation": "Obey the stated length and format limits exactly; cover every major theme; no preamble.",
    "factual": "Answer directly and concisely; explanations max 3 short sentences.",
    "code_fix": "State the one-line cause of the bug (what the buggy line actually does), then provide the minimal corrected function. Plain code, no fences, change only the bug.",
}


def forbidden_eval_material() -> tuple[set[str], set[str]]:
    """Collect every eval task id and prompt without consuming any answers."""
    ids: set[str] = set()
    prompts: set[str] = set()
    for path in Path("testdata").glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        tasks = data if isinstance(data, list) else data.get("tasks", [])
        for task in tasks:
            if not isinstance(task, dict):
                continue
            if isinstance(task.get("task_id"), str):
                ids.add(task["task_id"])
            if isinstance(task.get("prompt"), str):
                prompts.add(task["prompt"])
    return ids, prompts

# ------------------------------------------------------------------------ ner

PERSONS = [
    "Amara Okafor", "Kenji Watanabe", "Dr Lena Fischer", "Ravi Menon", "Sofia Alvarez",
    "Piotr Kowalski", "Mei-Ling Chen", "Prof Daniel Osei", "Ingrid Larsen", "Tariq Haddad",
    "Nadia Petrova", "Liam O'Sullivan", "Fatima Al-Rashid", "Marcus Webb", "Yuki Tanaka",
]
ORGS_WORD = [
    "Siemens", "Vodafone", "Barclays", "Novartis", "Airbus", "Spotify",
    "Deutsche Bank", "Standard Chartered", "Rolls-Royce", "Unilever",
]
ORGS_ACRONYM = [
    "NASA", "IBM", "WHO", "UNESCO", "MIT", "CERN", "OPEC", "NATO",
    "ETH Zurich", "UCL", "KPMG", "BBC",
]
ORGS_MULTI = [
    "the World Health Organization", "Stanford University", "the European Space Agency",
    "the International Monetary Fund", "Kyoto University", "the Red Cross",
]
LOCATIONS = [
    "Nairobi", "Copenhagen", "São Paulo", "Vancouver", "Singapore", "Marseille",
    "New Delhi", "Cape Town", "South Korea", "New Zealand", "Buenos Aires", "Osaka",
    "Lake Geneva", "the Gulf of Finland", "Edinburgh", "Rotterdam",
]
DATES = [
    "March 15, 2025", "3 June 2026", "January 2024", "October 7, 2023",
    "14 February 2026", "August 2025", "November 30, 2024", "2019",
    "April 1, 2026", "September 2023",
]

# Prominent named things that fit NO inventory label (programmes, missions,
# products). Reference answers note them separately instead of dropping or
# mislabelling them - the model must learn that convention (T05b Artemis-class).
PROGRAMMES = [
    "Aurora", "Helios", "Pathfinder", "Odyssey", "Horizon", "Vanguard",
    "Meridian", "Polaris", "Cascade", "Zephyr",
]
# Descriptor acronyms that are NOT entities; they appear in the text and must
# NOT appear in the entity list (models over-extract them otherwise).
NOISE = ["AI", "ML", "API", "GPU", "LLM"]

NER_TEMPLATES = [
    ("On {d}, {p} of {o} announced a new research campus in {l}.",
     [("p", "PERSON"), ("o", "ORGANIZATION"), ("l", "LOCATION"), ("d", "DATE")]),
    ("{p} joined {o} in {l} after leaving {o2} on {d}.",
     [("p", "PERSON"), ("o", "ORGANIZATION"), ("l", "LOCATION"), ("o2", "ORGANIZATION"), ("d", "DATE")]),
    ("{o} signed a partnership with {o2} in {l} on {d}, brokered by {p}.",
     [("o", "ORGANIZATION"), ("o2", "ORGANIZATION"), ("l", "LOCATION"), ("d", "DATE"), ("p", "PERSON")]),
    ("Speaking in {l} on {d}, {p} said {o} would double its investment in {l2}.",
     [("l", "LOCATION"), ("d", "DATE"), ("p", "PERSON"), ("o", "ORGANIZATION"), ("l2", "LOCATION")]),
    ("{o} opened its {l} office on {d}; {p} will lead the site.",
     [("o", "ORGANIZATION"), ("l", "LOCATION"), ("d", "DATE"), ("p", "PERSON")]),
    ("A report published on {d} by {o} ranked {l} ahead of {l2} for renewable adoption, according to {p}.",
     [("d", "DATE"), ("o", "ORGANIZATION"), ("l", "LOCATION"), ("l2", "LOCATION"), ("p", "PERSON")]),
    ("{p} and {p2} presented {o}'s findings at a conference in {l} on {d}.",
     [("p", "PERSON"), ("p2", "PERSON"), ("o", "ORGANIZATION"), ("l", "LOCATION"), ("d", "DATE")]),
    ("After talks in {l}, {o} confirmed on {d} that {p} will chair its {l2} board.",
     [("l", "LOCATION"), ("o", "ORGANIZATION"), ("d", "DATE"), ("p", "PERSON"), ("l2", "LOCATION")]),
]


def build_ner(rng: random.Random, n: int) -> list[dict]:
    rows = []
    # Acronym orgs are weighted up: the ETH-Zurich-in-partnership shape keeps
    # producing mislabels/omissions at eval, so it needs more gradient share.
    orgs_all = ORGS_WORD + ORGS_ACRONYM + ORGS_ACRONYM + ORGS_MULTI
    attempts = 0
    while len(rows) < n and attempts < n * 20:
        attempts += 1
        template, slots = NER_TEMPLATES[attempts % len(NER_TEMPLATES)]
        vals: dict[str, str] = {}
        persons = rng.sample(PERSONS, 2)
        orgs = rng.sample(orgs_all, 2)
        locs = rng.sample(LOCATIONS, 2)
        pool = {"p": persons[0], "p2": persons[1], "o": orgs[0], "o2": orgs[1],
                "l": locs[0], "l2": locs[1], "d": rng.choice(DATES)}
        for slot, _typ in slots:
            vals[slot] = pool[slot]
        sentence = template.format(**vals)
        entities = []
        ok = True
        for slot, typ in slots:
            span = vals[slot]
            display = span[4:] if span.startswith("the ") else span
            if sentence.count(display) != 1:
                ok = False
                break
            entities.append((display, typ))
        if not ok:
            continue
        extra = ""
        # 1 in 3: a noise-acronym descriptor appears in the text but must NOT
        # be extracted (models over-extract "AI"/"GPU" otherwise).
        if attempts % 3 == 0:
            noise = rng.choice(NOISE)
            sentence = sentence.rstrip(".") + f", expanding its {noise} research capacity."
        # 1 in 4: a programme name that fits no label - the answer notes it
        # separately (the reference-answer convention for T05b/Artemis-class).
        if attempts % 4 == 0:
            prog = rng.choice(PROGRAMMES)
            sentence = sentence.rstrip(".") + f" as part of the {prog} programme."
            extra = f"; Outside the inventory: {prog} programme"
        # 1 in 5: a trailing bare-year DATE ("targeted for 2028") that must be
        # extracted alongside the primary date.
        if attempts % 5 == 0:
            year = str(rng.choice([2027, 2028, 2029, 2030]))
            sentence = sentence.rstrip(".") + f", with completion targeted for {year}."
            entities.append((year, "DATE"))
        prompt = ("Extract all named entities from the following text and label each as "
                  "PERSON, ORGANIZATION, LOCATION, or DATE:\n\n'" + sentence + "'")
        answer = "; ".join(f"{t}: {s}" for s, t in entities) + extra
        rows.append({"family": "ner", "prompt": prompt, "answer": answer})
    assert len(rows) == n, f"ner: only built {len(rows)}/{n}"
    return rows


# ------------------------------------------------------------- code_generation

# (name, spec sentence, edge-case sentence, reference, tests[(args, expected)])
CG_FUNCS = [
    ("count_vowels", "takes a string and returns the number of vowels (a, e, i, o, u, case-insensitive)",
     "Handle an empty string (return 0).",
     "def count_vowels(s):\n    \"\"\"Return the number of vowels in s (case-insensitive).\"\"\"\n    return sum(1 for c in s.lower() if c in 'aeiou')",
     [(("Hello World",), 3), (("",), 0), (("XYZ",), 0)]),
    ("reverse_words", "takes a sentence string and returns it with the word order reversed",
     "Handle multiple spaces by collapsing them to single spaces.",
     "def reverse_words(s):\n    \"\"\"Return s with word order reversed, single-spaced.\"\"\"\n    return ' '.join(s.split()[::-1])",
     [(("hello brave world",), "world brave hello"), (("one",), "one")]),
    ("chunk_list", "takes a list and an integer n and returns the list split into consecutive chunks of size n",
     "The final chunk may be shorter; handle an empty list (return []).",
     "def chunk_list(lst, n):\n    \"\"\"Split lst into consecutive chunks of size n.\"\"\"\n    return [lst[i:i+n] for i in range(0, len(lst), n)]",
     [(([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]]), (([], 3), [])]),
    ("rotate_left", "takes a list and an integer k and returns the list rotated left by k positions",
     "Handle k larger than the list length and an empty list.",
     "def rotate_left(lst, k):\n    \"\"\"Return lst rotated left by k positions.\"\"\"\n    if not lst:\n        return []\n    k %= len(lst)\n    return lst[k:] + lst[:k]",
     [(([1, 2, 3, 4, 5], 2), [3, 4, 5, 1, 2]), (([1, 2], 5), [2, 1]), (([], 3), [])]),
    ("second_smallest", "takes a list of numbers and returns the second smallest DISTINCT value",
     "Return None when fewer than two distinct values exist.",
     "def second_smallest(nums):\n    \"\"\"Return the second smallest distinct value, or None.\"\"\"\n    distinct = sorted(set(nums))\n    return distinct[1] if len(distinct) >= 2 else None",
     [(([5, 1, 3, 1, 4],), 3), (([2, 2],), None)]),
    ("is_anagram", "takes two strings and returns True when they are anagrams, ignoring case and spaces",
     "Handle empty strings (two empty strings are anagrams).",
     "def is_anagram(a, b):\n    \"\"\"True when a and b are anagrams (case/space-insensitive).\"\"\"\n    norm = lambda s: sorted(s.replace(' ', '').lower())\n    return norm(a) == norm(b)",
     [(("Listen", "Silent"), True), (("hello", "world"), False)]),
    ("sum_digits", "takes a non-negative integer and returns the sum of its digits",
     "Handle 0 (return 0).",
     "def sum_digits(n):\n    \"\"\"Return the sum of n's digits.\"\"\"\n    return sum(int(d) for d in str(n))",
     [((493,), 16), ((0,), 0)]),
    ("unique_preserve_order", "takes a list and returns the distinct values in first-seen order",
     "Handle an empty list.",
     "def unique_preserve_order(lst):\n    \"\"\"Return distinct values of lst in first-seen order.\"\"\"\n    seen = set()\n    out = []\n    for x in lst:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out",
     [(([3, 1, 3, 2, 1],), [3, 1, 2]), (([],), [])]),
    ("merge_sum_dicts", "takes two dicts with numeric values and returns one dict where shared keys have their values summed",
     "Keys present in only one dict keep their value.",
     "def merge_sum_dicts(d1, d2):\n    \"\"\"Merge two numeric dicts, summing values on shared keys.\"\"\"\n    out = dict(d1)\n    for k, v in d2.items():\n        out[k] = out.get(k, 0) + v\n    return out",
     [(({"a": 1, "b": 2}, {"b": 3, "c": 4}), {"a": 1, "b": 5, "c": 4})]),
    ("fizzbuzz_list", "takes an integer n and returns the FizzBuzz sequence from 1 to n as a list of strings",
     "Multiples of 3 become 'Fizz', of 5 'Buzz', of both 'FizzBuzz'; other numbers appear as their string form.",
     "def fizzbuzz_list(n):\n    \"\"\"Return FizzBuzz strings for 1..n.\"\"\"\n    out = []\n    for i in range(1, n + 1):\n        s = ('Fizz' if i % 3 == 0 else '') + ('Buzz' if i % 5 == 0 else '')\n        out.append(s or str(i))\n    return out",
     [((5,), ["1", "2", "Fizz", "4", "Buzz"])]),
    ("longest_word", "takes a sentence string and returns its longest word, breaking ties by first occurrence",
     "Strip simple punctuation (.,!?) before comparing; handle an empty string (return '').",
     "def longest_word(s):\n    \"\"\"Return the longest word in s (ties: first occurrence).\"\"\"\n    words = [w.strip('.,!?') for w in s.split()]\n    return max(words, key=len) if words else ''",
     [(("The quick brown foxes jumped.",), "jumped"), (("",), "")]),
    ("count_occurrences", "takes a list and a target value and returns how many times the target appears",
     "Handle an empty list (return 0).",
     "def count_occurrences(lst, x):\n    \"\"\"Return how many times x appears in lst.\"\"\"\n    return sum(1 for v in lst if v == x)",
     [(([1, 2, 1, 1], 1), 3), (([], 5), 0)]),
    ("median_of", "takes a non-empty list of numbers and returns the median (mean of the middle two for even lengths)",
     "The input list must not be modified.",
     "def median_of(nums):\n    \"\"\"Return the median of nums without modifying it.\"\"\"\n    s = sorted(nums)\n    m = len(s) // 2\n    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2",
     [(([3, 1, 2],), 2), (([4, 1, 3, 2],), 2.5)]),
    ("running_total", "takes a list of numbers and returns the list of running totals",
     "Handle an empty list.",
     "def running_total(nums):\n    \"\"\"Return cumulative sums of nums.\"\"\"\n    out = []\n    total = 0\n    for n in nums:\n        total += n\n        out.append(total)\n    return out",
     [(([1, 2, 3],), [1, 3, 6]), (([],), [])]),
]

CG_PHRASINGS = [
    "Write a Python function called {name} that {spec}. For example, {example}. {edge}",
    "Implement a Python function called {name} which {spec}. For example, {example}. {edge}",
    "Write a function called {name} in Python that {spec}. For example, {example}. {edge}",
]


def verify_cg() -> None:
    for name, _spec, _edge, ref, tests in CG_FUNCS:
        ns: dict = {}
        exec(ref, ns)  # noqa: S102 - our own reference implementations
        fn = ns[name]
        for args, expected in tests:
            got = fn(*args)
            assert got == expected, f"{name}{args} -> {got!r}, want {expected!r}"


# Functions reserved for the heldout split - never trained on, so the
# post-train cg check measures generalisation to unseen specs.
CG_HELDOUT_FUNCS = {"median_of", "rotate_left", "is_anagram"}


def build_cg(rng: random.Random, split: str) -> list[dict]:
    verify_cg()
    rows = []
    for name, spec, edge, ref, tests in CG_FUNCS:
        if split == "train" and name in CG_HELDOUT_FUNCS:
            continue
        if split == "heldout" and name not in CG_HELDOUT_FUNCS:
            continue
        args, expected = tests[0]
        example = f"{name}({', '.join(repr(a) for a in args)}) should return {expected!r}"
        for phrasing in CG_PHRASINGS:
            prompt = phrasing.format(name=name, spec=spec, example=example, edge=edge)
            rows.append({"family": "code_generation", "prompt": prompt, "answer": ref})
    rng.shuffle(rows)
    return rows


# ---------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="finetune/minicpm5/data/assist_teacher_raw.jsonl")
    ap.add_argument("--claude", default="finetune/minicpm5/data/assist_claude_authored.jsonl",
                    help="Claude-authored sentiment/code_fix rows (gen_assist_claude_data.py)")
    ap.add_argument("--out", default="finetune/minicpm5/data/minicpm5_yassai_assist.jsonl")
    ap.add_argument("--v2-summaries", default="finetune/minicpm5/data/assist_v2_summaries.jsonl")
    ap.add_argument("--v2-sentiment", default="finetune/minicpm5/data/assist_v2_sentiment.jsonl")
    ap.add_argument("--seed", type=int, default=20260711)
    ap.add_argument("--ner", type=int, default=120)
    # Teacher rows are a fixed cache, so train/heldout must partition them -
    # reseeding only reshuffles the parametric families. Split by prompt hash:
    # ~1 in 7 rows to heldout. eval-on-train masked the v2-e5 overfit once
    # already; never again.
    ap.add_argument("--teacher-split", choices=["train", "heldout", "all"], default="train")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    split = args.teacher_split if args.teacher_split != "all" else "train"
    # cg appears three times: the v2 compound-summary expansion otherwise
    # diluted this execution-verified slice and regressed median handling.
    cg = build_cg(rng, split)
    rows = build_ner(rng, args.ner) + cg + cg + cg

    n_teacher = 0
    for cache in (Path(args.teacher), Path(args.claude), Path(args.v2_summaries), Path(args.v2_sentiment)):
        if not cache.exists():
            print(f"WARNING: cache {cache} missing - skipping")
            continue
        for line in cache.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            bucket = int(hashlib.md5(r["prompt"].encode()).hexdigest(), 16) % 7 == 0
            if args.teacher_split == "heldout" and not bucket:
                continue
            if args.teacher_split == "train" and bucket:
                continue
            rows.append({"family": r["family"], "prompt": r["prompt"], "answer": r["answer"]})
            n_teacher += 1

    forbidden_ids, forbidden_prompts = forbidden_eval_material()
    for row in rows:
        assert row.get("task_id") not in forbidden_ids, "training row id collides with eval material"
        assert row["prompt"] not in forbidden_prompts, "training prompt collides with eval material"

    rng.shuffle(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            fam = r["family"]
            msgs = [
                {"role": "system", "content": SYSTEM[fam]},
                {"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["answer"]},
            ]
            f.write(json.dumps({"messages": msgs, "family": fam}, ensure_ascii=False) + "\n")
    fams: dict[str, int] = {}
    for r in rows:
        fams[r["family"]] = fams.get(r["family"], 0) + 1
    print(f"wrote {len(rows)} rows ({n_teacher} from teacher cache) -> {out}")
    print("per family:", json.dumps(fams, sort_keys=True))


if __name__ == "__main__":
    main()
