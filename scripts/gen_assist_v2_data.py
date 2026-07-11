#!/usr/bin/env python3
"""Generate verified compound-summary SFT rows and sentiment counterweights.

This generator is deterministic and derives both passages and answers from
theme banks. It never reads eval answers or uses eval prompts as exemplars.
Before writing, every prompt is checked against every JSON task file under
testdata, and every summary is checked for shape, word caps, theme coverage,
and eight-word source echoing.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


BANKS = {
    "urban cooling": [
        ("Councils are planting shaded corridors beside busy pedestrian routes.", "Shaded walking routes cool busy streets", ["shaded", "routes"]),
        ("Shopkeepers report that cooler pavements keep customers in shopping areas longer.", "Cooler pavements support local shops", ["pavements", "shops"]),
        ("Young trees often fail because watering teams cannot cover prolonged dry spells.", "Water shortages threaten young trees", ["water", "trees"]),
        ("Neighbourhood groups will adopt planting beds and report damaged irrigation pipes.", "Residents will monitor beds and irrigation", ["residents", "irrigation"]),
        ("A monitoring trial recorded a 34% fall in afternoon surface temperatures.", "Monitoring found temperatures fell 34%", ["34%", "temperatures"]),
    ],
    "rural health": [
        ("Mobile clinics bring routine screening to villages without frequent bus services.", "Mobile clinics expand village screening", ["clinics", "screening"]),
        ("Patients save long journeys and identify treatable conditions earlier.", "Patients travel less and receive earlier diagnoses", ["travel", "earlier"]),
        ("Patchy signals prevent staff from accessing records during some appointments.", "Weak connectivity blocks medical records", ["connectivity", "records"]),
        ("Health boards are adding offline software and rotating specialist nurses.", "Offline tools and rotating specialists address gaps", ["offline", "specialists"]),
        ("Attendance rose by 42% during the first six months.", "Attendance increased 42% in six months", ["42%", "attendance"]),
    ],
    "reuse schemes": [
        ("Repair hubs collect unwanted appliances and restore safe items for resale.", "Repair hubs return appliances to use", ["repair", "appliances"]),
        ("Low-income households gain affordable equipment while councils reduce bulky waste.", "Affordable equipment also reduces council waste", ["affordable", "waste"]),
        ("Missing spare parts and uncertain safety histories slow many repairs.", "Parts shortages and safety doubts delay work", ["parts", "safety"]),
        ("Manufacturers will share manuals while colleges train volunteer technicians.", "Shared manuals and training strengthen repairs", ["manuals", "training"]),
        ("The pilot diverted 57% of donated machines from disposal.", "The pilot reused 57% of donations", ["57%", "reused"]),
    ],
    "local energy": [
        ("Schools are installing rooftop solar arrays through neighbourhood investment clubs.", "Community finance expands school solar power", ["community", "solar"]),
        ("Members receive modest returns and schools spend less on daytime electricity.", "Members earn returns while schools cut bills", ["returns", "bills"]),
        ("Old roofs and delayed grid connections prevent some sites from joining.", "Roof problems and grid delays restrict participation", ["roof", "grid"]),
        ("Engineers will assess buildings early and bundle connection applications.", "Early surveys and grouped applications reduce delays", ["surveys", "applications"]),
        ("Participating schools cut grid demand by 31% last term.", "Schools reduced grid demand by 31%", ["31%", "demand"]),
    ],
    "skills training": [
        ("Libraries host evening workshops on spreadsheets, online forms and video interviews.", "Libraries teach practical digital job skills", ["libraries", "skills"]),
        ("Jobseekers value free equipment and support outside normal working hours.", "Free evening support improves access", ["free", "access"]),
        ("Volunteer numbers fluctuate and older laptops struggle with current software.", "Volunteer gaps and ageing laptops limit sessions", ["volunteer", "laptops"]),
        ("Businesses will lend devices and release staff as monthly mentors.", "Businesses will provide devices and mentors", ["devices", "mentors"]),
        ("Completed job applications increased by 39% during the trial.", "Completed applications rose 39%", ["39%", "applications"]),
    ],
}

FILLER = [
    "The programme began after a year of consultation with residents.",
    "Organisers publish monthly updates and invite independent feedback.",
    "Several nearby districts are watching the early results closely.",
]

SENTIMENT = [
    ("The camera is light and the controls are clear, but focus hunts indoors and the battery is merely average. I have not decided whether to keep it.", "Neutral. Clear controls and low weight are balanced by unreliable indoor focus and ordinary battery life, without an overall verdict."),
    ("The room was quiet and spotless, although breakfast was limited and the shower pressure weak. Overall it did the job, nothing more.", "Neutral. Cleanliness and quiet are offset by weak facilities, and the reviewer gives a deliberately middling overall judgement."),
    ("Delivery took longer than promised, but the desk is sturdy, accurately finished and excellent value. I would buy it again.", "Positive. The delayed delivery is outweighed by quality, value and an explicit willingness to repurchase."),
    ("The staff were warm and the starter was excellent, yet both main courses arrived cold and the bill was wrong twice. We will not return.", "Negative. Friendly service and one good dish cannot outweigh cold meals, repeated billing errors and the decision not to return."),
    ("The app imports files quickly but its search is clumsy; exporting works well, though notifications are unreliable. A mixed update overall.", "Neutral. Useful importing and exporting are balanced by poor search and unreliable notifications, matching the stated mixed verdict."),
    ("These boots needed two days to soften, then stayed comfortable and dry through a week of rain. The brief break-in was worth it.", "Positive. A short break-in period is a minor drawback against lasting comfort, waterproofing and an explicit positive judgement."),
    ("The museum had thoughtful displays and helpful guides, but half the galleries were closed at full price. Interesting, yet poor value today.", "Negative. Strong displays and guides are outweighed by extensive closures and the reviewer's negative value judgement."),
    ("Setup was straightforward and printing is sharp; ink costs more than expected and the tray feels flimsy. It is adequate for occasional use.", "Neutral. Easy setup and print quality balance high ink costs and weak construction, producing an adequate rather than enthusiastic verdict."),
    ("The train was crowded, but it arrived exactly on time and the conductor found us seats after York. I was pleasantly surprised.", "Positive. Crowding is outweighed by punctuality, helpful service and the reviewer's explicit pleasant surprise."),
    ("The garden looked lovely and the cafe coffee was good, although paths were poorly signed and several toilets were shut. Neither good nor bad overall.", "Neutral. Attractive grounds and good coffee are balanced by navigation and facility problems, with an explicitly neutral conclusion."),
    ("The course covered useful basics, yet examples were dated and questions were rushed. I learned something, but would not recommend it at this price.", "Negative. Some useful teaching is outweighed by dated material, rushed support and an explicit refusal to recommend at the price."),
    ("Calls sound clear and the headset is comfortable; the microphone occasionally clips words, but colleagues still understand me. Solid, not exceptional.", "Neutral. Comfort and clear audio balance occasional microphone clipping, and the closing verdict is deliberately moderate."),
]


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9%]+(?:['’][A-Za-z0-9]+)?", text)


def norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def echoes(source: str, summary: str, width: int = 8) -> bool:
    src = norm_words(source)
    out = norm_words(summary)
    shingles = {tuple(src[i:i + width]) for i in range(len(src) - width + 1)}
    return any(tuple(out[i:i + width]) in shingles for i in range(len(out) - width + 1))


def forbidden_prompts() -> set[str]:
    prompts: set[str] = set()
    for path in Path("testdata").glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        rows = data if isinstance(data, list) else data.get("tasks", [])
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("prompt"), str):
                prompts.add(row["prompt"])
    return prompts


def summary_rows(rng: random.Random, count: int) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    forbidden = forbidden_prompts()
    attempts = 0
    while len(rows) < count:
        attempts += 1
        domain = rng.choice(list(BANKS))
        unit_count = rng.choice([2, 3, 3, 4])
        chosen = rng.sample(BANKS[domain], unit_count)
        require_number = attempts % 4 == 0
        if require_number and not any("%" in item[1] for item in chosen):
            chosen[-1] = BANKS[domain][-1]
        rng.shuffle(chosen)
        passage_parts = [item[0] for item in chosen]
        passage_parts.insert(rng.randrange(len(passage_parts) + 1), rng.choice(FILLER))
        passage = " ".join(passage_parts)
        kind = "bullet points" if attempts % 2 else "sentences"
        max_target = max(len(words(item[1])) for item in chosen)
        cap = rng.randint(max(10, max_target), 20)
        number_clause = " The final unit must mention the percentage." if require_number else ""
        prompt = (f"Summarise the following passage in exactly {unit_count} {kind}, each no longer than "
                  f"{cap} words, using your own words.{number_clause}\n\n'{passage}'")
        units = [item[1] if len(words(item[1])) >= 6 else item[1] + " in practice" for item in chosen]
        if kind == "bullet points":
            answer = "\n".join(f"- {unit}" for unit in units)
        else:
            answer = " ".join(unit.rstrip(".") + "." for unit in units)
        assert len(units) == unit_count
        assert all(6 <= len(words(unit)) <= cap for unit in units)
        assert not echoes(passage, answer)
        assert all(all(keyword.lower() in answer.lower() for keyword in item[2]) for item in chosen)
        if prompt in seen or prompt in forbidden:
            continue
        seen.add(prompt)
        rows.append({"family": "summarisation", "prompt": prompt, "answer": answer,
                     "meta": {"themes": [item[2] for item in chosen], "count": unit_count,
                              "cap": cap, "kind": kind}})
    return rows


def sentiment_rows(count: int) -> list[dict]:
    forbidden = forbidden_prompts()
    rows = []
    for i in range(count):
        review, answer = SENTIMENT[i % len(SENTIMENT)]
        cycle = i // len(SENTIMENT)
        if cycle:
            review += f" This assessment follows {cycle + 1} weeks of use."
        prompt = ("Classify the sentiment of this customer review as Positive, Negative, or Neutral and give a one-sentence reason:\n\n'"
                  + review + "'")
        assert prompt not in forbidden
        rows.append({"family": "sentiment", "prompt": prompt, "answer": answer})
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries-out", default="finetune/minicpm5/data/assist_v2_summaries.jsonl")
    ap.add_argument("--sentiment-out", default="finetune/minicpm5/data/assist_v2_sentiment.jsonl")
    ap.add_argument("--summaries", type=int, default=120)
    ap.add_argument("--sentiment", type=int, default=36)
    ap.add_argument("--seed", type=int, default=20260712)
    args = ap.parse_args()
    summaries = summary_rows(random.Random(args.seed), args.summaries)
    sentiment = sentiment_rows(args.sentiment)
    write_jsonl(Path(args.summaries_out), summaries)
    write_jsonl(Path(args.sentiment_out), sentiment)
    print(f"wrote {len(summaries)} verified summaries -> {args.summaries_out}")
    print(f"wrote {len(sentiment)} sentiment counterweights -> {args.sentiment_out}")


if __name__ == "__main__":
    main()
