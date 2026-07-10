#!/usr/bin/env python3
"""Mine DPO preference pairs from the judge archives.

Every localprobe run stored (task, answer, judge verdict) per model; several
models answered the SAME prompts. Where one answer PASSED the glm-5p2 judge
and another FAILED, that is a ready-made preference pair - real judge signal
on real task shapes, targeting exactly what SFT plateaued on (sentiment
verdict calibration, cause-line accuracy, rubric coverage).

The pair prompt is rendered with the SERVING system prompt for the family, so
DPO optimises the same distribution production samples from.

HARD RULE: prompts come ONLY from our own authored variant tasks. The golden
sample set (testdata/downloads_tasks_golden.json) is the published sample =
live prompt set - training on it is a rules breach, so it is excluded from
the prompt pool AND asserted against at the end.

Output: {"family", "system", "prompt", "chosen", "rejected"} JSONL.
"""

from __future__ import annotations

import argparse
import glob
import json
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
FAMILY_OF_CAT = {
    "factual": "factual", "sentiment": "sentiment", "summarisation": "summarisation",
    "ner": "ner", "code_generation": "code_generation", "code_debugging": "code_fix",
}


def load_tasks(path: str) -> dict[str, str]:
    d = json.load(open(path))
    d = d if isinstance(d, list) else d["tasks"]
    return {t["task_id"]: t["prompt"] for t in d}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="finetune/minicpm5/data/assist_dpo_pairs.jsonl")
    ap.add_argument("--max-len", type=int, default=1400, help="max answer chars")
    args = ap.parse_args()

    prompts = load_tasks("testdata/variant_tasks_golden.json")
    forbidden = set(load_tasks("testdata/downloads_tasks_golden.json"))  # sample/live ids - never train on these

    # bucket: (task_id) -> {"pass": [answers], "fail": [answers]}, plus family
    buckets: dict[str, dict] = {}
    for f in sorted(glob.glob("eval-results/localprobe-*.json")):
        rows = json.load(open(f))
        for r in rows:
            tid, ans, cat = r.get("ID"), (r.get("Answer") or "").strip(), r.get("Cat", "")
            fam = FAMILY_OF_CAT.get(cat)
            if not tid or not ans or tid not in prompts or fam is None:
                continue
            if len(ans) > args.max_len:
                continue
            b = buckets.setdefault(tid, {"family": fam, "pass": [], "fail": []})
            key = "pass" if r.get("Pass") else "fail"
            if ans not in b[key]:
                b[key].append(ans)

    pairs = []
    for tid, b in sorted(buckets.items()):
        for chosen in b["pass"]:
            for rejected in b["fail"]:
                if chosen == rejected:
                    continue
                pairs.append({
                    "family": b["family"],
                    "system": SYSTEM[b["family"]],
                    "prompt": prompts[tid],
                    "chosen": chosen,
                    "rejected": rejected,
                    "task_id": tid,
                })

    golden_prompts = set(load_tasks("testdata/downloads_tasks_golden.json").values())
    for p in pairs:
        assert p["task_id"] not in forbidden, f"sample-task id in pairs: {p['task_id']}"
        assert p["prompt"] not in golden_prompts, f"sample-task prompt in pairs: {p['task_id']}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    per: dict[str, int] = {}
    for p in pairs:
        per[p["family"]] = per.get(p["family"], 0) + 1
    print(f"wrote {len(pairs)} pairs -> {out}")
    print("per family:", json.dumps(per, sort_keys=True))


if __name__ == "__main__":
    main()
