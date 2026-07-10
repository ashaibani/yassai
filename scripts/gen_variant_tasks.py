#!/usr/bin/env python3
"""Generate the variant eval set - unseen-variant analogues of the 19 sample tasks.

The leaderboard scores UNSEEN PROMPT VARIANTS (Participant Guide), so this set -
not the memorised samples - is the local fitness function. Ids are prefix-coded
for cmd/realeval's categoryOf: f_* factual, m_* maths, s_* sentiment,
su_* summarisation, n_* ner, cd_* code_debugging, cg_* code_generation, l_* logic.

Maths/logic/NER/code answers are DERIVED (computed or solved here, generation
fails on any inconsistency). Factual/sentiment/summarisation rubrics are
hand-authored; --verify sends them to teacher models and reports disagreements.
"""

from __future__ import annotations

import argparse
import itertools
import json
from fractions import Fraction
from pathlib import Path

OUT_DEFAULT = Path(__file__).parent.parent / "testdata/variant_tasks_golden.json"


def task(tid: str, prompt: str, expected: str) -> dict:
    return {"task_id": tid, "prompt": prompt.strip(), "expected": expected.strip(), "validate": "llm"}


# --------------------------------------------------------------------------- maths


def warehouse(tid: str, start: int, q1_pct: int, restock: int, q3_sold: int) -> dict:
    after_q1 = start * (100 - q1_pct)
    assert after_q1 % 100 == 0, f"{tid}: q1 leaves fractional stock"
    after_q1 //= 100
    remaining = after_q1 + restock - q3_sold
    assert remaining > 0
    return task(
        tid,
        f"A distribution centre starts with {start:,} units. In Q1 it ships {q1_pct}% of stock. "
        f"In Q2 it receives {restock} new units. In Q3 it ships {q3_sold} units. "
        f"How many units remain at the end of Q3?",
        f"{remaining} units (after Q1 {after_q1}; after Q2 {after_q1 + restock}; after Q3 {remaining}).",
    )


def recipe(tid: str, num: int, den: int, base_count: int, target_count: int, price: float, ingredient: str, item: str) -> dict:
    needed = Fraction(num, den) * Fraction(target_count, base_count)
    assert 10_000 % needed.denominator == 0, f"{tid}: {needed} is not a terminating decimal"
    assert needed != 1 and needed.denominator != 1, f"{tid}: scaling too trivial ({needed})"
    cost = float(needed) * price
    needed_f = float(needed)
    assert abs(cost - round(cost, 2)) < 1e-9, f"{tid}: cost not clean 2dp"
    return task(
        tid,
        f"A recipe requires {num}/{den} cup of {ingredient} for {base_count} {item}. "
        f"How much {ingredient} is needed for {target_count} {item}? "
        f"If {ingredient} costs ${price:.2f} per cup, what is the total cost of {ingredient} for {target_count} {item}?",
        f"{needed_f:g} cups of {ingredient}; total cost ${cost:.2f}.",
    )


def trains(tid: str, v_a: int, v_b: int, dist: int, dep_a: str, dep_b: str) -> dict:
    ha, ma = map(int, dep_a.split(":"))
    hb, mb = map(int, dep_b.split(":"))
    head_start_h = (hb * 60 + mb - ha * 60 - ma) / 60
    covered = v_a * head_start_h
    remaining = dist - covered
    assert remaining > 0
    secs_total = remaining * 3600 / (v_a + v_b)
    assert abs(secs_total - round(secs_total)) < 1e-9, f"{tid}: meet time not whole seconds ({secs_total})"
    secs_total = round(secs_total)
    meet_min_total = hb * 3600 + mb * 60 + secs_total
    h, rem = divmod(meet_min_total, 3600)
    m, s = divmod(rem, 60)
    from_a = v_a * (head_start_h + secs_total / 3600)
    from_a_r = round(from_a, 2)
    assert abs(from_a - round(from_a, 3)) < 1e-9, f"{tid}: distance from A not exact to 3dp ({from_a})"
    meet = f"{h:02d}:{m:02d}:{s:02d}" if s else f"{h:02d}:{m:02d}"
    exact = f"{from_a:g}"
    shown = f"{from_a_r:g}" if f"{from_a_r:g}" == exact else f"{from_a_r:g} (exactly {exact})"
    return task(
        tid,
        f"A train leaves Station A at {dep_a} travelling toward Station B at {v_a} km/h. "
        f"A second train leaves Station B at {dep_b} travelling toward Station A at {v_b} km/h. "
        f"The distance between the stations is {dist} km. At what time do the trains meet, "
        f"and how far from Station A is the meeting point?",
        f"They meet at {meet}, {shown} km from Station A.",
    )


def revenue(tid: str, months: list[str], figures: list[int]) -> dict:
    avg = sum(figures) / len(figures)
    rates, declines = [], []
    for i in range(1, len(figures)):
        r = (figures[i] - figures[i - 1]) / figures[i - 1] * 100
        rates.append((months[i], r))
        if figures[i] < figures[i - 1]:
            declines.append(months[i])
    last3 = [r for _, r in rates[-3:]]
    nxt = figures[-1] * (1 + sum(last3) / 3 / 100)
    growth_str = ", ".join(f"{m} {r:.1f}%" for m, r in rates)
    figs = " | ".join(f"{m} ${v:,}" for m, v in zip(months, figures))
    next_month = "Jul" if months[-1] == "Jun" else "Jan" if months[-1] == "Dec" else "next month"
    return task(
        tid,
        f"Monthly revenue figures: {figs}.\n\nCalculate all of the following:\n"
        f"1. Average monthly revenue across all six months.\n"
        f"2. Month-over-month growth rate for each month from {months[1]} onwards "
        f"(as a percentage, rounded to one decimal place).\n"
        f"3. Which months saw a revenue decline?\n"
        f"4. Projected {next_month} revenue using the average of the last three months' growth rates.",
        f"1) Average ${avg:,.2f}. 2) Growth: {growth_str}. 3) Declines: {', '.join(declines) or 'none'}. "
        f"4) Projected {next_month}: ${nxt:,.2f} (from the raw average of the last three growth rates).",
    )


# --------------------------------------------------------------------------- logic


def solve_assignment(people: list[str], items: list[str], predicates) -> dict[str, str]:
    sols = []
    for perm in itertools.permutations(items):
        assignment = dict(zip(people, perm))
        if all(p(assignment) for p in predicates):
            sols.append(assignment)
    assert len(sols) == 1, f"logic puzzle has {len(sols)} solutions, need exactly 1"
    return sols[0]


def logic_puzzle(tid: str, intro: str, people: list[str], items: list[str], clue_texts: list[str], predicates) -> dict:
    sol = solve_assignment(people, items, predicates)
    clues = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(clue_texts))
    answer = "; ".join(f"{p}: {sol[p]}" for p in people)
    return task(
        tid,
        f"{intro}\nUse the following clues to determine the assignment:\n{clues}\nState each person's item.",
        f"{answer}.",
    )


# --------------------------------------------------------------------------- ner


def ner_task(tid: str, sentence: str, entities: list[tuple[str, str]]) -> dict:
    for span, _ in entities:
        assert span in sentence, f"{tid}: span {span!r} not in sentence"
    inventory = "PERSON, ORGANIZATION, LOCATION, or DATE"
    listing = "; ".join(f"{t}: {s}" for s, t in entities)
    return task(
        tid,
        f"Extract all named entities from the following text and label each as {inventory}:\n\n'{sentence}'",
        f"Exactly these entities: {listing}.",
    )


# --------------------------------------------------------------------------- code


def check_fn(src: str, name: str, tests: list[tuple], expect_pass: bool, tid: str) -> None:
    ns: dict = {}
    exec(src, ns, ns)  # noqa: S102 - our own generated code
    fn = ns[name]
    ok = True
    for args, want in tests:
        try:
            got = fn(*[json.loads(json.dumps(a)) for a in args])  # deep copy args
        except Exception:  # noqa: BLE001
            ok = False
            break
        if got != want:
            ok = False
            break
    assert ok == expect_pass, f"{tid}: expected {'pass' if expect_pass else 'fail'} for\n{src}"


def cbug(tid: str, description: str, buggy: str, fixed: str, name: str, tests: list[tuple], cause: str) -> dict:
    check_fn(buggy, name, tests, expect_pass=False, tid=tid)
    check_fn(fixed, name, tests, expect_pass=True, tid=tid)
    return task(
        tid,
        f"The following Python function is supposed to {description} but contains a bug. "
        f"Identify the bug and provide the corrected function.\n\n{buggy}",
        f"Cause: {cause} Corrected function:\n{fixed}",
    )


def cgen(tid: str, spec: str, reference: str, name: str, tests: list[tuple]) -> dict:
    check_fn(reference, name, tests, expect_pass=True, tid=tid)
    return task(tid, spec, f"A correct implementation, e.g.:\n{reference}")


# --------------------------------------------------------------------------- build


def build() -> list[dict]:
    out: list[dict] = []

    # maths (8): 2 x warehouse, 2 x recipe, 2 x trains, 2 x revenue
    out.append(warehouse("m_v01", 3200, 45, 650, 940, ))
    out.append(warehouse("m_v02", 5400, 35, 1200, 2115))
    out.append(recipe("m_v03", 3, 5, 6, 21, 1.50, "flour", "muffins"))
    out.append(recipe("m_v04", 5, 8, 10, 26, 3.20, "cocoa", "brownies"))
    out.append(trains("m_v05", 80, 100, 390, "07:30", "09:00"))
    out.append(trains("m_v06", 75, 105, 510, "06:45", "08:15"))
    out.append(revenue("m_v07", ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
                       [96000, 104000, 99500, 112400, 108900, 121600]))
    out.append(revenue("m_v08", ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                       [204000, 197800, 215600, 224100, 209300, 231800]))

    # logic (4)
    out.append(logic_puzzle(
        "l_v01",
        "Four flatmates — Nina, Omar, Petra, and Quinn — each play a different instrument: guitar, piano, violin, or drums.",
        ["Nina", "Omar", "Petra", "Quinn"], ["guitar", "piano", "violin", "drums"],
        ["Nina plays neither the guitar nor the drums.",
         "Omar plays neither the piano nor the violin.",
         "Petra plays the piano.",
         "Quinn does not play the drums."],
        [lambda a: a["Nina"] not in {"guitar", "drums"},
         lambda a: a["Omar"] not in {"piano", "violin"},
         lambda a: a["Petra"] == "piano",
         lambda a: a["Quinn"] != "drums"],
    ))
    out.append(logic_puzzle(
        "l_v02",
        "Three colleagues — Ravi, Sofia, and Tara — each commute by a different mode: bicycle, bus, or car.",
        ["Ravi", "Sofia", "Tara"], ["bicycle", "bus", "car"],
        ["Ravi refuses to sit in traffic, so he neither drives nor takes the bus.",
         "Sofia never takes public transport.",
         "Tara does not cycle."],
        [lambda a: a["Ravi"] not in {"car", "bus"},
         lambda a: a["Sofia"] != "bus",
         lambda a: a["Tara"] != "bicycle"],
    ))
    out.append(logic_puzzle(
        "l_v03",
        "Four analysts — Uma, Viktor, Wren, and Xia — each cover a different sector: energy, retail, tech, or healthcare.",
        ["Uma", "Viktor", "Wren", "Xia"], ["energy", "retail", "tech", "healthcare"],
        ["Uma covers neither energy nor retail.",
         "Viktor does not cover tech or healthcare.",
         "Wren covers healthcare.",
         "Xia does not cover energy."],
        [lambda a: a["Uma"] not in {"energy", "retail"},
         lambda a: a["Viktor"] not in {"tech", "healthcare"},
         lambda a: a["Wren"] == "healthcare",
         lambda a: a["Xia"] != "energy"],
    ))
    out.append(logic_puzzle(
        "l_v04",
        "Three siblings — Yara, Zane, and Alba — each adopted a different animal: a rabbit, a hamster, and a canary.",
        ["Yara", "Zane", "Alba"], ["rabbit", "hamster", "canary"],
        ["Zane is allergic to fur.",
         "Yara did not adopt the canary.",
         "Alba did not adopt the rabbit."],
        [lambda a: a["Zane"] not in {"rabbit", "hamster"},
         lambda a: a["Yara"] != "canary",
         lambda a: a["Alba"] != "rabbit"],
    ))

    # ner (3)
    out.append(ner_task(
        "n_v01",
        "On 4 February 2024, Lisa Su announced that AMD would open a research campus in Austin "
        "in partnership with the University of Texas.",
        [("4 February 2024", "DATE"), ("Lisa Su", "PERSON"), ("AMD", "ORGANIZATION"),
         ("Austin", "LOCATION"), ("University of Texas", "ORGANIZATION")],
    ))
    out.append(ner_task(
        "n_v02",
        "Satya Nadella met engineers from OpenAI in Seattle on 12 June 2025 before Microsoft "
        "confirmed the partnership.",
        [("Satya Nadella", "PERSON"), ("OpenAI", "ORGANIZATION"), ("Seattle", "LOCATION"),
         ("12 June 2025", "DATE"), ("Microsoft", "ORGANIZATION")],
    ))
    out.append(ner_task(
        "n_v03",
        "In October 2019, Margrethe Vestager warned that the European Commission would examine "
        "Amazon's marketplace practices across Europe.",
        [("October 2019", "DATE"), ("Margrethe Vestager", "PERSON"),
         ("European Commission", "ORGANIZATION"), ("Amazon", "ORGANIZATION"), ("Europe", "LOCATION")],
    ))

    # code_debugging (3) - different bug archetypes from the samples
    out.append(cbug(
        "cd_v01",
        "return the sum of the even numbers in a list",
        "def sum_evens(nums):\n    total = 0\n    for n in nums:\n        if n % 2 == 1:\n            total += n\n    return total",
        "def sum_evens(nums):\n    total = 0\n    for n in nums:\n        if n % 2 == 0:\n            total += n\n    return total",
        "sum_evens",
        [(([1, 2, 3, 4, 5, 6],), 12), (([7, 9],), 0), (([],), 0), (([2, -4],), -2)],
        "the parity test is inverted: n % 2 == 1 selects odd numbers, so the function sums the odds instead of the evens.",
    ))
    out.append(cbug(
        "cd_v02",
        "count how many times a target value appears in a list",
        "def count_target(nums, target):\n    count = 0\n    for i in range(1, len(nums)):\n        if nums[i] == target:\n            count += 1\n    return count",
        "def count_target(nums, target):\n    count = 0\n    for i in range(len(nums)):\n        if nums[i] == target:\n            count += 1\n    return count",
        "count_target",
        [(([5, 1, 5, 5], 5), 3), (([5], 5), 1), (([1, 2, 3], 9), 0), (([],  1), 0)],
        "the loop starts at index 1 and never inspects the first element, so a match at index 0 is missed (off-by-one).",
    ))
    out.append(cbug(
        "cd_v03",
        "return the average of a list of numbers",
        "def average(nums):\n    total = 0\n    for n in nums:\n        total += n\n    return total / (len(nums) - 1)",
        "def average(nums):\n    total = 0\n    for n in nums:\n        total += n\n    return total / len(nums)",
        "average",
        [(([2, 4, 6],), 4.0), (([10],), 10.0), (([1, 2],), 1.5)],
        "the sum is divided by len(nums) - 1 instead of len(nums), inflating the result (and crashing on single-element lists).",
    ))

    # code_generation (3)
    out.append(cgen(
        "cg_v01",
        "Write a Python function called run_length_encode that takes a string and returns a list of "
        "(character, count) tuples for consecutive runs. For example, run_length_encode('aaabbc') "
        "should return [('a', 3), ('b', 2), ('c', 1)]. It should handle an empty string (return []) "
        "and a string with no repeated characters.",
        "def run_length_encode(s):\n    \"\"\"Return (char, count) tuples for consecutive runs in s.\"\"\"\n"
        "    result = []\n    for ch in s:\n        if result and result[-1][0] == ch:\n"
        "            result[-1] = (ch, result[-1][1] + 1)\n        else:\n            result.append((ch, 1))\n    return result",
        "run_length_encode",
        [(("aaabbc",), [("a", 3), ("b", 2), ("c", 1)]), (("",), []), (("abc",), [("a", 1), ("b", 1), ("c", 1)]),
         (("aa",), [("a", 2)])],
    ))
    out.append(cgen(
        "cg_v02",
        "Write a Python function called interval_intersection that takes two lists of closed intervals "
        "(each interval is a [start, end] pair, each list sorted and non-overlapping) and returns the "
        "list of their intersections. For example, interval_intersection([[1,4],[7,10]], [[3,8]]) should "
        "return [[3,4],[7,8]]. Handle empty inputs and touching endpoints ([1,3] and [3,5] intersect at [3,3]).",
        "def interval_intersection(a, b):\n    \"\"\"Return intersections of two sorted non-overlapping interval lists.\"\"\"\n"
        "    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n"
        "        lo = max(a[i][0], b[j][0])\n        hi = min(a[i][1], b[j][1])\n"
        "        if lo <= hi:\n            out.append([lo, hi])\n"
        "        if a[i][1] < b[j][1]:\n            i += 1\n        else:\n            j += 1\n    return out",
        "interval_intersection",
        [(([[1, 4], [7, 10]], [[3, 8]]), [[3, 4], [7, 8]]), (([], [[1, 2]]), []),
         (([[1, 3]], [[3, 5]]), [[3, 3]]), (([[0, 2], [5, 10]], [[1, 5], [8, 12]]), [[1, 2], [5, 5], [8, 10]])],
    ))
    out.append(cgen(
        "cg_v03",
        "Write a Python function called top_k_frequent that takes a list of strings and an integer k, and "
        "returns the k most frequent strings ordered by descending frequency, breaking ties alphabetically. "
        "For example, top_k_frequent(['b','a','b','c','a','b'], 2) should return ['b', 'a']. Handle k larger "
        "than the number of distinct strings.",
        "def top_k_frequent(words, k):\n    \"\"\"Return the k most frequent words, ties broken alphabetically.\"\"\"\n"
        "    from collections import Counter\n    counts = Counter(words)\n"
        "    ordered = sorted(counts, key=lambda w: (-counts[w], w))\n    return ordered[:k]",
        "top_k_frequent",
        [((["b", "a", "b", "c", "a", "b"], 2), ["b", "a"]), (((["x"]), 5), ["x"]),
         ((["z", "y", "z", "y"], 2), ["y", "z"])],
    ))

    # factual (3) - authored rubrics
    out.append(task(
        "f_v01",
        "Name the three primary colours in the CMY subtractive colour model and briefly explain why "
        "printers use CMY (plus black) instead of RGB.",
        "Cyan, magenta, and yellow. Printed ink absorbs (subtracts) light reflected off paper, so printing "
        "uses the subtractive CMY model; RGB is additive and only works for light-emitting displays. Black (K) "
        "is added because mixing CMY inks yields muddy dark brown, wastes ink, and pure black text needs a dedicated ink.",
    ))
    out.append(task(
        "f_v02",
        "What is the difference between HTTP and HTTPS? Briefly explain what the extra layer provides.",
        "HTTP transfers data in plaintext; HTTPS is HTTP running over TLS (SSL) encryption. TLS provides "
        "confidentiality (traffic cannot be read in transit), integrity (tampering is detected), and server "
        "authentication via certificates.",
    ))
    out.append(task(
        "f_v03",
        "Explain the difference between a compiler and an interpreter. Give one typical example language for each.",
        "A compiler translates the whole source program into machine code (or another target) before execution; "
        "an interpreter executes source code statement by statement at runtime. Typical examples: C or Go are "
        "compiled; Python or JavaScript are commonly interpreted. (Hybrids like JIT/bytecode exist.)",
    ))

    # sentiment (4) - authored, covering labels the samples never test
    out.append(task(
        "s_v01",
        "Classify the sentiment of this customer review as Positive, Negative, or Neutral and give a "
        "one-sentence reason:\n\n'The app crashed twice during setup and the tutorial skipped steps, but "
        "once running it synced my data flawlessly and support replied within minutes.'",
        "Positive - the reviewer's final impression (flawless syncing, fast support) outweighs the setup problems. "
        "A Neutral verdict with a reason acknowledging the balance is a defensible alternative; Negative is wrong.",
    ))
    out.append(task(
        "s_v02",
        "Classify the sentiment of this customer review as Positive, Negative, or Neutral and give a "
        "one-sentence reason:\n\n'Beautiful packaging and quick delivery, but the blender's motor burned out "
        "in the first week and the refund process has dragged on for a month.'",
        "Negative - the product failed and the refund experience is poor; the positive delivery/packaging notes do "
        "not offset a broken product and unresolved refund.",
    ))
    out.append(task(
        "s_v03",
        "Classify the sentiment of this tweet as Positive, Negative, or Neutral and give a one-sentence "
        "reason:\n\n'New phone arrived today. Battery lasts longer than my old one, screen scratches way too "
        "easily. Jury's still out.'",
        "Neutral - the author explicitly withholds judgement ('jury's still out') and balances one clear positive "
        "(battery) against one clear negative (screen).",
    ))
    out.append(task(
        "s_v04",
        "Classify the sentiment of this review as Positive, Negative, or Neutral and give a one-sentence "
        "reason:\n\n'Honestly did not expect much at this price, yet the headphones sound crisp, the fit is "
        "comfortable for hours, and the case feels premium.'",
        "Positive - every concrete point (sound, comfort, build) is praise; the low-expectation opener only "
        "amplifies the positive surprise.",
    ))

    # summarisation (2) - authored passages with strict shape constraints
    out.append(task(
        "su_v01",
        "Summarize the following passage in exactly two sentences:\n\n'Electric vehicles are moving from "
        "early-adopter novelty to mainstream transport as battery costs fall and charging networks expand. "
        "Fleet operators cite lower running costs and simpler maintenance, while city governments see a route "
        "to cleaner air. Yet challenges remain: raw material supply chains concentrate risk in a few countries, "
        "grid capacity must grow to absorb peak charging demand, and drivers in rural areas still face sparse "
        "charging coverage. Manufacturers are responding with cheaper chemistries, faster-charging designs, and "
        "partnerships with utilities to manage load.'",
        "Exactly two sentences. First covers adoption drivers (falling battery costs, expanding charging, lower "
        "running costs/cleaner air); second covers the challenges (material supply concentration, grid capacity, "
        "rural coverage) and the manufacturers' response (cheaper chemistries, faster charging, utility partnerships).",
    ))
    out.append(task(
        "su_v02",
        "Summarize the following passage in exactly three bullet points, each no longer than 15 words:\n\n"
        "'Open-source software now underpins most commercial technology stacks, from operating systems to "
        "machine-learning frameworks. Companies benefit from faster development, shared maintenance, and the "
        "ability to audit code they depend on. However, maintainer burnout, unfunded critical projects, and "
        "supply-chain attacks on package registries have exposed fragility at the ecosystem's core. In response, "
        "large firms are funding foundations, security teams now scan dependencies continuously, and governments "
        "have begun to treat key open-source projects as critical infrastructure.'",
        "Exactly three bullets, each 15 words or fewer, covering: (1) open source underpins commercial stacks "
        "with development/maintenance/audit benefits, (2) fragility - maintainer burnout, unfunded projects, "
        "supply-chain attacks, (3) responses - corporate funding, dependency scanning, treating projects as "
        "critical infrastructure.",
    ))

    out.extend(adversarial())
    ids = [t["task_id"] for t in out]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    return out


def eval_snippet(code: str, names: list[str]) -> dict:
    ns: dict = {}
    exec(code, ns, ns)  # noqa: S102
    return {n: ns[n] for n in names}


def adversarial() -> list[dict]:
    """Harder tier mirroring the author's generate_adversarial.py patterns:
    multi-step order-sensitive maths, dense positional logic, semantic Python
    gotchas ('what does this evaluate to'), algorithmic codegen, ambiguous
    sentiment, strict multi-constraint summarisation."""
    out: list[dict] = []

    # maths: order-sensitive flows and percent traps (derived inline)
    tank = 520 - 9 * 12 + 14 * 15 - 6 * 20
    assert tank == 502
    out.append(task(
        "m_a01",
        "A reservoir starts with 520 litres. It drains at 9 litres per minute for 12 minutes, is then "
        "refilled at 14 litres per minute for 15 minutes, then drains at 6 litres per minute for 20 "
        "minutes. How many litres are in the reservoir now?",
        "502 litres (520 - 108 + 210 - 120).",
    ))
    pool_h = 1 / (1 / 8 + 1 / 6)
    assert abs(pool_h - 24 / 7) < 1e-12
    out.append(task(
        "m_a02",
        "Pump A can fill a tank in 8 hours. Pump B can fill the same tank in 6 hours. If both pumps run "
        "together, how many hours does filling take? Answer as a decimal rounded to 2 decimal places.",
        "3.43 hours (24/7 = 3.4285..., combined rate 1/8 + 1/6 = 7/24 per hour).",
    ))
    price = 400 * 1.25 * 0.75 * 1.08
    assert abs(price - 405.0) < 1e-9
    out.append(task(
        "m_a03",
        "A price is increased by 25%, then decreased by 25%, then increased by 8%. If the original price "
        "was $400, what is the final price?",
        "$405.00 (400 x 1.25 x 0.75 x 1.08; note the increase and decrease do not cancel).",
    ))

    # logic: positional seating + arithmetic weights, solver-verified
    seat_sols = []
    people5 = ["Ana", "Bram", "Cleo", "Dev", "Esme"]
    for perm in itertools.permutations(people5):
        # seats 1..5 left to right = perm order
        if perm[0] != "Ana" or perm[4] != "Esme":
            continue
        # Cleo sits immediately left of Dev
        ci, di = perm.index("Cleo"), perm.index("Dev")
        if di - ci != 1:
            continue
        # Bram does not sit next to Esme
        bi = perm.index("Bram")
        if abs(bi - 4) == 1:
            continue
        seat_sols.append(perm)
    assert len(seat_sols) == 1, f"seating puzzle has {len(seat_sols)} solutions"
    out.append(task(
        "l_a01",
        "Five colleagues — Ana, Bram, Cleo, Dev, and Esme — sit in five seats numbered 1 to 5 in a row "
        "(seat 1 is the far left). Ana sits in seat 1. Esme sits in seat 5. Cleo sits immediately to the "
        "left of Dev. Bram does not sit next to Esme. What is the seating order from seat 1 to seat 5?",
        ", ".join(seat_sols[0]) + ".",
    ))

    weight_sols = []
    boxes = ["P", "Q", "R", "S", "T"]
    for perm in itertools.permutations([2, 4, 6, 8, 10]):
        w = dict(zip(boxes, perm))
        if w["P"] + w["Q"] != 6:
            continue
        if not w["R"] > w["S"]:
            continue
        if w["T"] != 10:
            continue
        if abs(w["S"] - w["Q"]) != 2:
            continue
        weight_sols.append(w)
    assert len(weight_sols) == 1, f"weights puzzle has {len(weight_sols)} solutions"
    out.append(task(
        "l_a02",
        "Five boxes — P, Q, R, S, and T — contain weights of 2, 4, 6, 8, and 10 kg, one weight per box. "
        "P and Q together weigh 6 kg. R is heavier than S. T is the heaviest box. S and Q differ by "
        "exactly 2 kg. What is the weight of each box?",
        "; ".join(f"{b}: {weight_sols[0][b]} kg" for b in boxes) + ".",
    ))

    # code_debugging (semantic 'what does this evaluate to' - execution-verified)
    r = eval_snippet(
        "def make_adders():\n    return [lambda x: x + i for i in range(1, 4)]\n"
        "adders = make_adders()\nresults = [a(10) for a in adders]",
        ["results"])
    assert r["results"] == [13, 13, 13]
    out.append(task(
        "cd_a01",
        "What does `results` evaluate to, and why, given this code?\n\n"
        "def make_adders():\n    return [lambda x: x + i for i in range(1, 4)]\n\n"
        "adders = make_adders()\nresults = [a(10) for a in adders]",
        "results = [13, 13, 13], not [11, 12, 13]: the lambdas close over the loop variable i by "
        "reference (late binding), and i is 3 for all of them after the comprehension finishes.",
    ))
    r = eval_snippet(
        "def append_item(item, bucket=[]):\n    bucket.append(item)\n    return bucket\n"
        "first = append_item(1)\nsecond = append_item(2)",
        ["first", "second"])
    assert r["first"] == [1, 2] and r["second"] == [1, 2] and r["first"] is r["second"]
    out.append(task(
        "cd_a02",
        "What do `first` and `second` evaluate to, and why, given this code?\n\n"
        "def append_item(item, bucket=[]):\n    bucket.append(item)\n    return bucket\n\n"
        "first = append_item(1)\nsecond = append_item(2)",
        "Both are the SAME list [1, 2]: default argument values are evaluated once at function "
        "definition, so every call without bucket shares one mutable list (first is second is True).",
    ))
    r = eval_snippet(
        "nums = [1, 2, 3, 4, 5, 6]\nevens = (n for n in nums if n % 2 == 0)\n"
        "total1 = sum(evens)\ntotal2 = sum(evens)",
        ["total1", "total2"])
    assert r["total1"] == 12 and r["total2"] == 0
    out.append(task(
        "cd_a03",
        "What do `total1` and `total2` evaluate to, and why, given this code?\n\n"
        "nums = [1, 2, 3, 4, 5, 6]\nevens = (n for n in nums if n % 2 == 0)\n"
        "total1 = sum(evens)\ntotal2 = sum(evens)",
        "total1 = 12 and total2 = 0: the generator is exhausted by the first sum(), so the second "
        "sum() iterates nothing.",
    ))

    # code_generation (algorithmic, reference verified against tests)
    out.append(cgen(
        "cg_a01",
        "Write a Python function called edit_distance that takes two strings a and b and returns the "
        "minimum number of single-character insertions, deletions, or substitutions required to "
        "transform a into b (Levenshtein distance). For example, edit_distance('kitten', 'sitting') "
        "should return 3.",
        "def edit_distance(a, b):\n    \"\"\"Return the Levenshtein distance between a and b.\"\"\"\n"
        "    m, n = len(a), len(b)\n    dp = list(range(n + 1))\n"
        "    for i in range(1, m + 1):\n        prev, dp[0] = dp[0], i\n"
        "        for j in range(1, n + 1):\n            cur = dp[j]\n"
        "            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))\n"
        "            prev = cur\n    return dp[n]",
        "edit_distance",
        [(("kitten", "sitting"), 3), (("", "abc"), 3), (("abc", "abc"), 0), (("flaw", "lawn"), 2)],
    ))
    out.append(cgen(
        "cg_a02",
        "Write a Python function called lis_length that takes a list of integers and returns the length "
        "of the longest strictly increasing subsequence. For example, lis_length([10, 9, 2, 5, 3, 7, "
        "101, 18]) should return 4. Handle an empty list (return 0).",
        "def lis_length(nums):\n    \"\"\"Return the length of the longest strictly increasing subsequence.\"\"\"\n"
        "    import bisect\n    tails = []\n    for n in nums:\n"
        "        i = bisect.bisect_left(tails, n)\n"
        "        if i == len(tails):\n            tails.append(n)\n        else:\n            tails[i] = n\n"
        "    return len(tails)",
        "lis_length",
        [(([10, 9, 2, 5, 3, 7, 101, 18],), 4), (([],), 0), (([5, 5, 5],), 1), (([1, 2, 3],), 3)],
    ))

    # ambiguous sentiment (sarcasm + backhanded praise)
    out.append(task(
        "s_a01",
        "Classify the sentiment of this app review as Positive, Negative, or Neutral and give a "
        "one-sentence reason:\n\n'Oh fantastic, another update that resets all my settings and hides "
        "the export button. Five stars for consistency, I guess.'",
        "Negative - the praise is sarcastic ('oh fantastic', 'five stars for consistency, I guess'); "
        "the reviewer is complaining about settings resets and a hidden feature.",
    ))
    out.append(task(
        "s_a02",
        "Classify the sentiment of this restaurant review as Positive, Negative, or Neutral and give a "
        "one-sentence reason:\n\n'For a place with plastic chairs and a hand-written menu, the food was "
        "shockingly good - honestly the best noodles I've had all year.'",
        "Positive - despite the dismissive framing of the decor, the verdict on the food ('shockingly "
        "good', 'best noodles all year') is emphatic praise.",
    ))

    # strict multi-constraint summarisation
    out.append(task(
        "su_a01",
        "Summarize the following passage in exactly three bullet points. Each bullet must be no longer "
        "than 12 words, and the second bullet must mention the percentage figure:\n\n'Global "
        "smartphone shipments fell for the eighth consecutive quarter as consumers held on to devices "
        "longer. Analysts attribute 61% of the decline to lengthening replacement cycles rather than "
        "economic pressure. Manufacturers are responding by shifting marketing budgets toward trade-in "
        "programmes, on-device AI features, and extended software support windows designed to make "
        "upgrades feel worthwhile again.'",
        "Exactly three bullets, each 12 words or fewer: (1) shipments fell for an eighth straight "
        "quarter as users keep devices longer; (2) must mention 61% attributed to longer replacement "
        "cycles; (3) makers respond with trade-ins, on-device AI, longer software support.",
    ))

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()
    rows = build()
    Path(args.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    by_cat: dict[str, int] = {}
    for r in rows:
        by_cat[r["task_id"].split("_")[0]] = by_cat.get(r["task_id"].split("_")[0], 0) + 1
    print(f"wrote {len(rows)} variant tasks to {args.out}")
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(by_cat.items())))


if __name__ == "__main__":
    main()
