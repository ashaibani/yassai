#!/usr/bin/env python3
"""Build curated MiniCPM5 SFT data for yassai local tool-use routing.

The examples teach two behaviors:
1. For math/logic prompts, emit a compact run_python tool call.
2. After the tool observation, return the final answer in the requested format.

Every tool-call example is executed while building the dataset. If the code does
not produce the expected observation, dataset generation fails.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path


SYSTEM = """You are yassai-local, a small local specialist for math and logic tasks.
Use run_python for arithmetic, percentages, schedules, projections, combinatorics, and constraint checks.
Never do these calculations in your head.
When a tool is needed, respond with exactly one tool call and no prose:
<tool_call>{"name":"run_python","arguments":{"code":"..."}}</tool_call>
The Python must compute from named variables and print concise final values.
After receiving a run_python result, return only the final answer requested by the user."""


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    code: str
    observation: str
    answer: str
    include_default: bool = True


def compact_json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def tool_call(code: str) -> str:
    return "<tool_call>" + compact_json(
        {"name": "run_python", "arguments": {"code": code.strip()}}
    ) + "</tool_call>"


def messages(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }


def execute(code: str) -> str:
    buf = io.StringIO()
    ns: dict[str, object] = {}
    with contextlib.redirect_stdout(buf):
        exec(code, ns, ns)
    return buf.getvalue().strip()


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def rows_for_case(c: Case) -> list[dict]:
    observed = execute(c.code)
    if normalize(observed) != normalize(c.observation):
        raise ValueError(
            f"{c.name}: code observation mismatch\n"
            f"expected: {c.observation!r}\n"
            f"observed: {observed!r}"
        )
    after_tool = (
        f"{c.prompt.strip()}\n\n"
        f"run_python result:\n{c.observation.strip()}\n\n"
        "Return the final answer only."
    )
    return [
        messages(c.prompt, tool_call(c.code)),
        messages(after_tool, c.answer),
    ]


def cases() -> list[Case]:
    return [
        Case(
            "warehouse_percent_restock",
            "A warehouse starts with 3,600 units. In Q1 it sells 42% of stock. "
            "In Q2 it restocks 950 units. In Q3 it sells 725 units. How many units remain?",
            """
start = 3600
sold_q1 = start * 0.42
after_q1 = start - sold_q1
after_q2 = after_q1 + 950
remaining = after_q2 - 725
print(int(remaining))
""",
            "2313",
            "2313",
        ),
        Case(
            "download_t02_warehouse",
            "A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. "
            "In Q2 it restocks 800 units. In Q3 it sells 640 units. How many units remain at the end of Q3?",
            """
start = 2400
after_q1 = start * (1 - 0.37)
after_q2 = after_q1 + 800
remaining = after_q2 - 640
print(int(remaining))
""",
            "1672",
            "1672",
            include_default=False,
        ),
        Case(
            "recipe_ratio_cost",
            "A recipe uses 2/3 cup of flour for 8 muffins. How much flour is needed for 30 muffins? "
            "If flour costs $1.80 per cup, what is the total cost?",
            """
from fractions import Fraction
base_cups = Fraction(2, 3)
base_count = 8
target_count = 30
needed = base_cups * Fraction(target_count, base_count)
cost = float(needed) * 1.80
print(f"{float(needed):.3f} cups")
print(f"${cost:.2f}")
""",
            "2.500 cups\n$4.50",
            "2.5 cups of flour; total cost $4.50.",
        ),
        Case(
            "download_t02b_recipe",
            "A recipe requires 3/4 cup of sugar for 12 cookies. How much sugar is needed for 30 cookies? "
            "If sugar costs $2.40 per cup, what is the total cost of sugar for 30 cookies?",
            """
from fractions import Fraction
sugar_for_12 = Fraction(3, 4)
needed = sugar_for_12 * Fraction(30, 12)
cost = float(needed) * 2.40
print(f"{float(needed):.3f} cups")
print(f"${cost:.2f}")
""",
            "1.875 cups\n$4.50",
            "1.875 cups of sugar; total cost $4.50.",
            include_default=False,
        ),
        Case(
            "train_meeting_offset",
            "A train leaves City A at 07:15 travelling at 80 km/h. A second train leaves City B at 08:45 "
            "travelling toward City A at 120 km/h. The cities are 520 km apart. At what time do they meet, "
            "and how far from City A is the meeting point?",
            """
from datetime import datetime, timedelta
distance = 520
v_a = 80
v_b = 120
start_a = datetime(2000, 1, 1, 7, 15)
start_b = datetime(2000, 1, 1, 8, 45)
head_start_h = (start_b - start_a).total_seconds() / 3600
covered_before_b = v_a * head_start_h
remaining = distance - covered_before_b
after_b_h = remaining / (v_a + v_b)
meet = start_b + timedelta(hours=after_b_h)
from_a = v_a * (head_start_h + after_b_h)
print(meet.strftime("%H:%M:%S"))
print(f"{from_a:.2f} km from City A")
""",
            "10:45:00\n280.00 km from City A",
            "They meet at 10:45:00, 280 km from City A.",
        ),
        Case(
            "download_t08_train",
            "A train leaves City A at 08:00 travelling toward City B at 90 km/h. "
            "A second train leaves City B at 09:30 travelling toward City A at 110 km/h. "
            "The distance between the cities is 450 km. At what time do the trains meet, "
            "and how far from City A is the meeting point?",
            """
from datetime import datetime, timedelta
distance = 450
v_a = 90
v_b = 110
start_a = datetime(2000, 1, 1, 8, 0)
start_b = datetime(2000, 1, 1, 9, 30)
head_start_h = (start_b - start_a).total_seconds() / 3600
covered_before_b = v_a * head_start_h
remaining = distance - covered_before_b
after_b_h = remaining / (v_a + v_b)
meet = start_b + timedelta(hours=after_b_h)
from_a = v_a * (head_start_h + after_b_h)
print(meet.strftime("%H:%M:%S"))
print(f"{from_a:.2f} km from City A")
""",
            "11:04:30\n276.75 km from City A",
            "They meet at 11:04:30, 276.75 km from City A.",
            include_default=False,
        ),
        Case(
            "revenue_projection",
            "Monthly revenue: Jan $120,000 | Feb $126,000 | Mar $119,700 | Apr $135,660 | "
            "May $132,947 | Jun $146,242. Calculate average revenue, month-over-month growth rates "
            "from February, decline months, and projected July using the average of April, May, and June growth rates.",
            """
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
rev = [120000, 126000, 119700, 135660, 132947, 146242]
avg = sum(rev) / len(rev)
rates = []
declines = []
for i in range(1, len(rev)):
    r = (rev[i] - rev[i - 1]) / rev[i - 1] * 100
    rates.append((months[i], r))
    if rev[i] < rev[i - 1]:
        declines.append(months[i])
last3_avg = sum(r for _, r in rates[-3:]) / 3
july = rev[-1] * (1 + last3_avg / 100)
print(f"average=${avg:.2f}")
print("growth=" + ", ".join(f"{m} {r:.1f}%" for m, r in rates))
print("declines=" + ", ".join(declines))
print(f"july=${july:.2f}")
""",
            "average=$130091.50\ngrowth=Feb 5.0%, Mar -5.0%, Apr 13.3%, May -2.0%, Jun 10.0%\ndeclines=Mar, May\njuly=$156641.61",
            "Average: $130,091.50. Growth: Feb 5.0%, Mar -5.0%, Apr 13.3%, May -2.0%, Jun 10.0%. Declines: Mar and May. Projected July: $156,641.61.",
        ),
        Case(
            "daily_eggs_sales",
            "Rina's ducks lay 22 eggs per day. She eats 4 eggs for breakfast and uses 7 eggs for baking every day. "
            "She sells the rest for $3 per egg. How much money does she make each day? Give only the final numeric answer.",
            """
eggs = 22
eaten = 4
baked = 7
sold = eggs - eaten - baked
revenue = sold * 3
print(revenue)
""",
            "33",
            "33",
        ),
        Case(
            "house_flip_profit",
            "A house flipper buys a house for $95,000 and spends $35,000 on repairs. The repairs increase the value "
            "of the house by 140% relative to the purchase price. How much profit does the flipper make?",
            """
purchase = 95000
repairs = 35000
new_value = purchase * (1 + 1.40)
profit = new_value - purchase - repairs
print(int(profit))
""",
            "98000",
            "$98,000",
        ),
        Case(
            "discount_every_second_item",
            "One notebook costs $8, but every second notebook costs only 75% of the price. Maya buys 14 notebooks. "
            "How much does she pay? Give only the final numeric answer.",
            """
count = 14
full_price = 8
discount_price = full_price * 0.75
pairs = count // 2
total = pairs * (full_price + discount_price)
if count % 2:
    total += full_price
print(int(total) if total == int(total) else total)
""",
            "98",
            "98",
        ),
        Case(
            "chicken_feed_final_meal",
            "Each chicken eats 4 cups of feed per day split across three meals. A flock has 18 chickens. "
            "The morning meal uses 24 cups and the afternoon meal uses 31 cups. How many cups are needed for the final meal?",
            """
chickens = 18
daily_per_chicken = 4
daily_total = chickens * daily_per_chicken
morning = 24
afternoon = 31
final_meal = daily_total - morning - afternoon
print(final_meal)
""",
            "17",
            "17 cups",
        ),
        Case(
            "download_t10_revenue",
            "Monthly revenue figures: Jan $142,000 | Feb $138,500 | Mar $157,200 | Apr $163,800 | "
            "May $151,400 | Jun $168,900.\n\nCalculate all of the following:\n"
            "1. Average monthly revenue across all six months.\n"
            "2. Month-over-month growth rate for each month from February onwards (as a percentage, rounded to one decimal place).\n"
            "3. Which months saw a revenue decline?\n"
            "4. Projected July revenue using the average of the last three months' growth rates (April, May, June).",
            """
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
rev = [142000, 138500, 157200, 163800, 151400, 168900]
avg = sum(rev) / len(rev)
rates = []
declines = []
for i in range(1, len(rev)):
    r = (rev[i] - rev[i - 1]) / rev[i - 1] * 100
    rates.append((months[i], r))
    if rev[i] < rev[i - 1]:
        declines.append(months[i])
last3_avg = sum(r for _, r in rates[-3:]) / 3
july = rev[-1] * (1 + last3_avg / 100)
print(f"average=${avg:.2f}")
print("growth=" + ", ".join(f"{m} {r:.1f}%" for m, r in rates))
print("declines=" + ", ".join(declines))
print(f"july=${july:.2f}")
""",
            "average=$153633.33\ngrowth=Feb -2.5%, Mar 13.5%, Apr 4.2%, May -7.6%, Jun 11.6%\ndeclines=Feb, May\njuly=$173509.31",
            "1. Average monthly revenue: $153,633.33. 2. Growth: Feb -2.5%, Mar 13.5%, Apr 4.2%, May -7.6%, Jun 11.6%. 3. Declines: February and May. 4. Projected July revenue: $173,509.31.",
            include_default=False,
        ),
        Case(
            "people_beverages",
            "Four colleagues - Ana, Ben, Cy, and Dee - each drink a different beverage: coffee, tea, water, or juice. "
            "Ana drinks neither coffee nor juice. Ben drinks neither water nor tea. Cy drinks tea. Dee does not drink juice. "
            "State each person's beverage.",
            """
import itertools
people = ["Ana", "Ben", "Cy", "Dee"]
drinks = ["coffee", "tea", "water", "juice"]
for perm in itertools.permutations(drinks):
    d = dict(zip(people, perm))
    if d["Ana"] in {"coffee", "juice"}:
        continue
    if d["Ben"] in {"water", "tea"}:
        continue
    if d["Cy"] != "tea":
        continue
    if d["Dee"] == "juice":
        continue
    print("; ".join(f"{p}: {d[p]}" for p in people))
""",
            "Ana: water; Ben: juice; Cy: tea; Dee: coffee",
            "Ana: water; Ben: juice; Cy: tea; Dee: coffee.",
        ),
        Case(
            "download_t07_beverages",
            "Four colleagues — Alice, Bob, Carol, and Dave — each drink a different beverage: coffee, tea, water, or juice. "
            "Use the following clues to determine who drinks what:\n"
            "1. Alice drinks neither coffee nor juice.\n2. Bob drinks neither water nor tea.\n"
            "3. Carol drinks tea.\n4. Dave does not drink juice.\nState each person's beverage.",
            """
import itertools
people = ["Alice", "Bob", "Carol", "Dave"]
drinks = ["coffee", "tea", "water", "juice"]
for perm in itertools.permutations(drinks):
    d = dict(zip(people, perm))
    if d["Alice"] in {"coffee", "juice"}:
        continue
    if d["Bob"] in {"water", "tea"}:
        continue
    if d["Carol"] != "tea":
        continue
    if d["Dave"] == "juice":
        continue
    print("; ".join(f"{p}: {d[p]}" for p in people))
""",
            "Alice: water; Bob: juice; Carol: tea; Dave: coffee",
            "Alice: water; Bob: juice; Carol: tea; Dave: coffee.",
            include_default=False,
        ),
        Case(
            "three_staff_seating_roles",
            "Three staff sit in a row from left to right. Their ages are 20, 24, and 30, and their jobs are accountant, "
            "salesperson, and engineer. At least one person to the right of the 24-year-old is 20. The accountant is not "
            "on the far left. The salesperson is immediately to the right of the accountant. Which full arrangements are possible?",
            """
import itertools
ages = [20, 24, 30]
jobs = ["accountant", "salesperson", "engineer"]
valid = []
for age_perm in itertools.permutations(ages):
    for job_perm in itertools.permutations(jobs):
        rows = list(zip(age_perm, job_perm))
        idx24 = age_perm.index(24)
        if 20 not in age_perm[idx24 + 1:]:
            continue
        idx_acc = job_perm.index("accountant")
        if idx_acc == 0:
            continue
        if idx_acc == 2 or job_perm[idx_acc + 1] != "salesperson":
            continue
        valid.append(rows)
for rows in valid:
    print("; ".join(f"{age}-year-old {job}" for age, job in rows))
""",
            "24-year-old engineer; 20-year-old accountant; 30-year-old salesperson\n24-year-old engineer; 30-year-old accountant; 20-year-old salesperson\n30-year-old engineer; 24-year-old accountant; 20-year-old salesperson",
            "Possible arrangements: 24-year-old engineer; 20-year-old accountant; 30-year-old salesperson. 24-year-old engineer; 30-year-old accountant; 20-year-old salesperson. 30-year-old engineer; 24-year-old accountant; 20-year-old salesperson.",
        ),
        Case(
            "friends_pets",
            "Three friends - Maya, Noah, and Omar - each own a different pet: cat, dog, and parrot. "
            "Noah is allergic to fur. Maya does not own the parrot. Omar does not own the dog. State each person's pet.",
            """
import itertools
people = ["Maya", "Noah", "Omar"]
pets = ["cat", "dog", "parrot"]
furred = {"cat", "dog"}
for perm in itertools.permutations(pets):
    p = dict(zip(people, perm))
    if p["Noah"] in furred:
        continue
    if p["Maya"] == "parrot":
        continue
    if p["Omar"] == "dog":
        continue
    print("; ".join(f"{name}: {p[name]}" for name in people))
""",
            "Maya: dog; Noah: parrot; Omar: cat",
            "Maya: dog; Noah: parrot; Omar: cat.",
        ),
        Case(
            "download_t07b_pets",
            "Three friends — Emma, Liam, and Priya — each own a different pet: a cat, a dog, and a parrot. "
            "Use these clues to determine who owns which pet:\n"
            "1. Liam is allergic to fur.\n2. Emma does not own the parrot.\n3. Priya does not own the dog.\n"
            "State each person's pet.",
            """
import itertools
people = ["Emma", "Liam", "Priya"]
pets = ["cat", "dog", "parrot"]
furred = {"cat", "dog"}
for perm in itertools.permutations(pets):
    p = dict(zip(people, perm))
    if p["Liam"] in furred:
        continue
    if p["Emma"] == "parrot":
        continue
    if p["Priya"] == "dog":
        continue
    print("; ".join(f"{name}: {p[name]}" for name in people))
""",
            "Emma: dog; Liam: parrot; Priya: cat",
            "Emma: dog; Liam: parrot; Priya: cat.",
            include_default=False,
        ),
        Case(
            "medicine_constraints",
            "A medicine must include at least one of ginseng or codonopsis. If codonopsis is included, atractylodes is included. "
            "Atractylodes and ginseng cannot both be included. Shouwu must be included. If Shouwu is included, atractylodes is included. "
            "Which ingredients must be present?",
            """
import itertools
names = ["ginseng", "codonopsis", "atractylodes", "shouwu"]
valid = []
for bits in itertools.product([False, True], repeat=len(names)):
    s = dict(zip(names, bits))
    if not (s["ginseng"] or s["codonopsis"]):
        continue
    if s["codonopsis"] and not s["atractylodes"]:
        continue
    if s["atractylodes"] and s["ginseng"]:
        continue
    if not s["shouwu"]:
        continue
    if s["shouwu"] and not s["atractylodes"]:
        continue
    valid.append(s)
must = [n for n in names if all(v[n] for v in valid)]
print(", ".join(must))
""",
            "codonopsis, atractylodes, shouwu",
            "Codonopsis, atractylodes, and Shouwu must be present.",
        ),
    ]


def build_rows(include_exact_downloads: bool) -> list[dict]:
    out: list[dict] = []
    for c in cases():
        if c.include_default or include_exact_downloads:
            out.extend(rows_for_case(c))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="finetune/minicpm5/data/minicpm5_yassai_math_logic.jsonl")
    ap.add_argument("--include-exact-downloads", action="store_true")
    args = ap.parse_args()

    rows = build_rows(args.include_exact_downloads)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(compact_json(r) + "\n")
    print(f"wrote {len(rows)} examples to {out}")


if __name__ == "__main__":
    main()
