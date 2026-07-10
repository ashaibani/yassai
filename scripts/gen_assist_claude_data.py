#!/usr/bin/env python3
"""Claude-authored SFT data for the assist lane's two weakest families.

Unlike gen_assist_teacher_data.py (spec -> teacher model -> judge filter),
every example here was written directly by Claude in one authoring pass,
targeting the EXACT failure modes observed on the unseen-variant probes:

  sentiment  - sarcasm read literally, positive idioms read as negative
               ("the jury's still out", "can't complain"), mixed reviews where
               the overall verdict must win, true neutrals, and rationales
               that never contradict the text (the judge fails
               right-label-wrong-reason answers).
  code_fix   - correction-style code_debugging ("supposed to X but contains a
               bug"): one-line cause naming what the buggy line actually does,
               then the minimal corrected function. Every pair is EXECUTED at
               build time: the buggy version must fail its tests and the fixed
               version must pass, so no un-verified example can ship.

Output rows: {"family", "prompt", "answer"} - one JSON per line, merged by
scripts/build_minicpm5_assist_data.py like the teacher cache.

Usage:
  python3 scripts/gen_assist_claude_data.py \
      --out finetune/minicpm5/data/assist_claude_authored.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ------------------------------------------------------------------ sentiment
# (container, text, label, rationale)
SENTIMENT: list[tuple[str, str, str, str]] = [
    # --- sarcasm: surface praise, real verdict negative ---
    ("customer review",
     "Oh brilliant, another firmware update that turns my doorbell into a very expensive wall ornament. Truly the innovation we were promised.",
     "Negative",
     "The praise is sarcastic - the update actually made the doorbell stop working."),
    ("tweet",
     "Love how my 'waterproof' fitness tracker died in light drizzle. 10/10 engineering, would drown again.",
     "Negative",
     "The mock rating and quotes around waterproof signal sarcasm about the tracker failing in rain."),
    ("forum comment",
     "Fantastic hold music, really. I got to enjoy all forty-five minutes of it before being disconnected.",
     "Negative",
     "Calling the hold music fantastic is ironic - the complaint is a 45-minute wait ending in a disconnection."),
    ("customer review",
     "What a bargain: pay premium prices AND assemble the wardrobe yourself with instructions written in hieroglyphics.",
     "Negative",
     "The word bargain is used mockingly about high prices and incomprehensible instructions."),
    ("tweet",
     "So glad my food arrived cold. Saved me the trouble of letting it cool down. #blessed",
     "Negative",
     "The gratitude and #blessed are sarcastic - the complaint is cold food."),
    ("customer review",
     "The kettle takes ten minutes to boil, which is great because I really needed more waiting in my life.",
     "Negative",
     "The claimed benefit is ironic - the reviewer is complaining the kettle is far too slow."),
    ("forum comment",
     "A masterclass in design: the charging port is placed exactly where your hand grips the controller. Genius.",
     "Negative",
     "Masterclass and genius are sarcastic jabs at a charging port that blocks a normal grip."),
    ("tweet",
     "Big thanks to the airline for the surprise overnight stay in the terminal. Five-star floor, very firm.",
     "Negative",
     "The thanks and five-star rating mock a cancelled flight that stranded the writer overnight."),
    # --- positive idioms that models misread as negative ---
    ("customer review",
     "Honestly, I can't complain. The blender does everything the listing promised and the jug survived a drop onto tile.",
     "Positive",
     "\"Can't complain\" is an idiom of satisfaction, reinforced by the blender exceeding expectations."),
    ("forum comment",
     "Not too shabby at all. Setup took five minutes and the picture quality punches well above the price tag.",
     "Positive",
     "\"Not too shabby\" is understated praise, backed by easy setup and strong picture quality."),
    ("customer review",
     "It's nothing to write home about looks-wise, but it has run every single day for two years without a hiccup. I'd buy it again in a heartbeat.",
     "Positive",
     "Despite the mild dig at its looks, two years of flawless running and a repurchase pledge make the verdict positive."),
    ("tweet",
     "Two weeks with the new earbuds and the battery hype is real. The jury's still out on the case hinge, but so far so good.",
     "Positive",
     "\"So far so good\" and confirmed battery hype outweigh one undecided point about the hinge."),
    ("customer review",
     "For the price, you really can't go wrong. It isn't a flagship and doesn't pretend to be, but it nails the basics.",
     "Positive",
     "\"Can't go wrong\" plus nailing the basics is a clear recommendation at this price."),
    ("forum comment",
     "I was ready to send it back on day one, but it grew on me. Couldn't imagine my desk without it now.",
     "Positive",
     "The early doubt is resolved - the writer now considers it indispensable."),
    ("customer review",
     "Does what it says on the tin. No frills, no drama, just a solid little heater that warms the room fast.",
     "Positive",
     "\"Does what it says on the tin\" is approval, supported by fast, reliable heating."),
    # --- mixed but positive overall ---
    ("customer review",
     "Delivery took nearly three weeks and the box looked like it had been through a war. The desk inside, though, is rock solid and looks far more expensive than it was. Worth the wait.",
     "Positive",
     "Shipping complaints are outweighed by the explicit verdict that the desk was worth the wait."),
    ("tweet",
     "The app crashes if you rotate the screen, which is silly, but the actual banking features are the best I've used. Switching my main account over.",
     "Positive",
     "One bug is acknowledged, but calling the features the best and switching accounts is a positive verdict."),
    ("customer review",
     "I'll be honest: the manual is useless and the app nags too much. But the robot itself vacuums better than my old upright and the mapping is spookily accurate. Keeping it.",
     "Positive",
     "The complaints are secondary - superior cleaning, accurate mapping, and the decision to keep it carry the verdict."),
    ("forum comment",
     "Fan noise under load is noticeable and the RGB software is bloated. Still, frame rates jumped 40% and it has been rock stable for a month, so no regrets on the upgrade.",
     "Positive",
     "Noise and software gripes are minor next to a 40% performance gain, stability, and no regrets."),
    ("customer review",
     "Sizing runs small - order a size up. Once you get past that, the boots are waterproof as advertised and kept my feet warm through a week of Scottish hills.",
     "Positive",
     "The sizing caveat is advice, not a complaint about quality; the boots performed exactly as advertised."),
    # --- mixed but negative overall ---
    ("customer review",
     "The screen is gorgeous and the speakers surprisingly punchy. Shame the battery barely survives four hours and the hinge started creaking in week two. I'm returning it.",
     "Negative",
     "Praise for the screen and speakers is outweighed by battery and hinge failures and the decision to return it."),
    ("tweet",
     "Credit where due: support answered in minutes. Unfortunately they answered to tell me my two-month-old washer isn't covered. Never buying this brand again.",
     "Negative",
     "Fast support does not offset a denied warranty claim and the vow never to buy the brand again."),
    ("customer review",
     "Lovely packaging, premium feel, and the first espresso was excellent. Then the pump died on day nine and the replacement did the same. Twice burned, I'm out.",
     "Negative",
     "Early praise is overturned by two consecutive pump failures and the writer giving up on the product."),
    ("forum comment",
     "The keyboard feels fantastic to type on, I'll give it that. But three keys double-press out of the box on a so-called premium board? Unacceptable at this price.",
     "Negative",
     "One positive note about feel is outweighed by defective keys called unacceptable for the price."),
    ("customer review",
     "The staff were friendly and the room was clean, but the 'sea view' was a car park and the air conditioning ran all night like a jet engine. Wouldn't stay again.",
     "Negative",
     "Friendly staff cannot rescue a misleading view and sleepless nights - the verdict is not staying again."),
    # --- true neutral: balanced, no verdict ---
    ("customer review",
     "The tablet is fine. Screen is sharp but reflective; battery lasts a day but charges slowly. If you need something basic it does the job, if you want more, look elsewhere.",
     "Neutral",
     "Praise and criticism are balanced and the recommendation depends entirely on the buyer's needs."),
    ("forum comment",
     "Switched from the old model a month ago. Some things are better (quieter, lighter), some are worse (fewer ports, shorter cable). Overall it's a wash.",
     "Neutral",
     "The writer explicitly weighs improvements against regressions and calls it a wash."),
    ("customer review",
     "It photographs well in daylight and struggles at night, which is exactly what you'd expect at this price. Neither impressed nor disappointed.",
     "Neutral",
     "The performance matches expectations and the writer states they are neither impressed nor disappointed."),
    ("tweet",
     "New update moved all the menus around. Not better, not worse, just different. Muscle memory ruined for a week I guess.",
     "Neutral",
     "The writer explicitly judges the change as neither better nor worse, only different."),
    ("customer review",
     "Delivery was on time, the chair matches the photos, assembly was standard flat-pack fare. It's a chair. It does chair things.",
     "Neutral",
     "The review is factual and deadpan with no praise or complaint beyond expectations being met."),
    # --- clear positive ---
    ("customer review",
     "Three months in and this air fryer has genuinely changed how we cook. Crispy results, easy cleanup, and the basket coating shows zero wear.",
     "Positive",
     "The reviewer credits it with changing how they cook and praises results, cleanup, and durability."),
    ("tweet",
     "That moment when your budget earbuds outlast and outsound the flagship pair your mate paid triple for. Absolute steal.",
     "Positive",
     "Calling them an absolute steal that outperforms a flagship is unambiguous praise."),
    ("forum comment",
     "The mechanical keyboard arrived, and wow - the tactile feedback is everything the reviews promised. Typing feels addictive now.",
     "Positive",
     "The enthusiasm (wow, addictive) confirms the purchase exceeded expectations."),
    ("customer review",
     "Booked the plumber for a leaking valve; he arrived early, fixed it in twenty minutes, charged exactly the quote, and left the place spotless. Faultless service.",
     "Positive",
     "Early arrival, quick fix, honest pricing, and the word faultless make this clearly positive."),
    ("tweet",
     "This library's new API is so clean I deleted half my wrapper code. More of this energy please, open source.",
     "Positive",
     "Deleting wrapper code because the API is clean is strong developer praise."),
    # --- clear negative ---
    ("customer review",
     "The suitcase wheel snapped off in the airport on its first trip. The handle jams, the zip catches, and support wants receipts I already sent twice.",
     "Negative",
     "Multiple hardware failures on first use plus unhelpful support make this plainly negative."),
    ("tweet",
     "Day 3 of the broadband installation saga. Third engineer no-show. I have written novels shorter than my complaint thread.",
     "Negative",
     "Repeated missed appointments and a growing complaint thread express clear frustration."),
    ("forum comment",
     "GPU drivers crash on every second boot since the update. Rolled back, still crashing. This card is going back tomorrow.",
     "Negative",
     "Persistent crashes and the decision to return the card are unambiguously negative."),
    ("customer review",
     "The 'leather' sofa started peeling within six weeks. The store blames 'improper conditioning'. Of a sofa. In a living room.",
     "Negative",
     "A peeling sofa and a dismissive excuse from the store make the review scathing."),
    ("tweet",
     "Restaurant charged us for the 'chef's special' that never left the kitchen. Waiter shrugged. Manager shrugged harder.",
     "Negative",
     "Being charged for food that never arrived and indifferent staff is a clear complaint."),
    # --- v5 counterweight: complaints acknowledged, verdict still POSITIVE.
    # v4 over-learnt sarcasm and began flipping this exact shape to Negative
    # (observed live on two golden tasks); these anchor the overall-verdict
    # rule from the positive side. ---
    ("customer review",
     "The parcel arrived two days late and the box was crushed, but the monitor inside was flawless and support sorted a partial refund within the hour. Genuinely impressed.",
     "Positive",
     "Shipping problems are acknowledged, but a flawless product and fast, generous support leave the reviewer impressed."),
    ("tweet",
     "Order showed up late, manual was missing, and yet this little espresso machine still made the best flat white I've had at home. Keeper.",
     "Positive",
     "Despite delivery annoyances, the writer calls the results the best at home and is keeping it."),
    ("customer review",
     "Setup instructions were confusing and I needed a YouTube video to finish. Once running though, the projector is stunning for the price. Happy customer.",
     "Positive",
     "Setup friction is outweighed by a stunning result and the writer's explicit happiness."),
    ("forum comment",
     "Two dead pixels out of the box had me worried, but the replacement arrived in 48 hours and it's been perfect for a month. Credit to the support team.",
     "Positive",
     "An initial defect was resolved quickly and the writer now praises both product and support."),
    ("customer review",
     "The app nags about subscriptions and the packaging is wasteful. The scale itself, though, is accurate to the gram and syncs instantly. Recommended with those caveats.",
     "Positive",
     "Caveats are noted, but accuracy, reliable syncing, and an explicit recommendation set the verdict."),
    ("tweet",
     "Queue was out the door and the table wobbled, but that ramen was worth every minute of the wait. Going back next week.",
     "Positive",
     "The complaints are minor against food worth the wait and the plan to return."),
    ("customer review",
     "I'll grumble about the proprietary charger forever, but this trimmer has outlasted three of its predecessors and still holds a full week of charge.",
     "Positive",
     "One design gripe does not outweigh exceptional durability and battery life."),
    ("forum comment",
     "Driver install was clunky on Linux, not going to lie. After that? Buttery 165Hz, great colours, zero flicker. Would buy again.",
     "Positive",
     "The install complaint is followed by strong performance praise and an explicit would-buy-again."),
    ("customer review",
     "The hotel lift was out of order all weekend and our room faced the bins. Still, the staff upgraded our breakfast, the beds were dreamy, and we left smiling.",
     "Positive",
     "Real annoyances are conceded, but the upgrades, comfort, and leaving smiling carry the verdict."),
    ("tweet",
     "Honestly expected this budget phone to be rubbish. Camera's average at night but everything else runs smoother than my old flagship. Pleasantly shocked.",
     "Positive",
     "Low expectations were beaten - one average aspect does not dent the pleasant shock."),
    ("customer review",
     "Not perfect: the zip sticks occasionally and the straps could be thicker. But after six months of daily commuting the bag shows zero wear, and everything stays dry in proper rain.",
     "Positive",
     "Small flaws are listed honestly, yet durability and waterproofing over six months make it a positive review."),
    ("forum comment",
     "The firmware updater only runs on Windows, which is annoying at this price. That aside, the keyboard's build quality is superb and the switches feel incredible.",
     "Positive",
     "One software annoyance is outweighed by superb build quality and switch feel."),
    ("customer review",
     "Colour is slightly darker than the photos and one cushion seam was loose. The sofa is otherwise solid, deeply comfortable, and arrived a week early. We're delighted.",
     "Positive",
     "Minor cosmetic issues are conceded, but comfort, solidity, early delivery, and delight decide the verdict."),
    ("tweet",
     "The conference wifi died twice and lunch queues were chaos, but the talks were the best I've seen in years. Already booked for next year.",
     "Positive",
     "Logistics complaints are outweighed by the best talks in years and rebooking."),
    ("customer review",
     "Instructions in five languages, none of them helpful. Assembly took two hours of guesswork. And yet the finished wardrobe is sturdy, silent, and looks premium. Worth the swearing.",
     "Positive",
     "Assembly frustration is real, but 'worth the swearing' and premium results make it positive."),
    ("forum comment",
     "It ships with bloatware, which I removed in ten minutes. Underneath is the cleanest, fastest budget laptop I've tested this year.",
     "Positive",
     "The bloatware complaint is quickly resolved and the overall judgement is the year's best budget laptop."),
    ("customer review",
     "The subscription upsell on first launch nearly made me return it. Glad I didn't: the free tier does everything I need and the hardware is faultless.",
     "Positive",
     "The initial annoyance is admitted, but faultless hardware and a sufficient free tier win out."),
    ("tweet",
     "Landlord special paint job aside, this flat has been brilliant: quiet street, quick landlord, boiler that actually works. No plans to leave.",
     "Positive",
     "A cosmetic gripe is trivial next to quiet, responsiveness, reliability, and the intent to stay."),
    ("customer review",
     "Portion sizes could be bigger for the price, I'll say that. But every dish was cooked perfectly, service was warm without hovering, and the tiramisu alone justifies a return trip.",
     "Positive",
     "The price-portion gripe is outweighed by perfect cooking, warm service, and a promised return."),
    ("forum comment",
     "Cable management on this case is an afterthought and the manual is a single sheet. Airflow though? Best I've measured under £100. Recommended build.",
     "Positive",
     "Two design complaints are noted, but class-leading airflow and a recommendation settle it."),
    # --- v5 counterweight: unambiguous positives with enthusiasm ---
    ("customer review",
     "Bought this stand mixer refurbished with low expectations. It has since produced forty loaves, two birthday cakes, and zero complaints. Best kitchen money I've spent.",
     "Positive",
     "Heavy successful use and 'best kitchen money spent' are unambiguous praise."),
    ("tweet",
     "My plants have never looked happier. This grow light paid for itself in one basil season. Ten out of ten.",
     "Positive",
     "Thriving plants and a ten-out-of-ten rating are clear enthusiasm."),
    ("forum comment",
     "Three years, two house moves, one toddler, and this vacuum still runs like day one. Buy once, cry once.",
     "Positive",
     "Long-term durability through heavy use with an endorsement idiom is plainly positive."),
    ("customer review",
     "The tailor took one look at the jacket, pinned it in five minutes, and had it back to me the next morning fitting like it was made for me. Exceptional.",
     "Positive",
     "Fast, precise service ending in 'exceptional' is clearly positive."),
    # --- tricky: negation and faint praise ---
    ("customer review",
     "I wanted to hate the redesign, but I can't. Everything is where my thumb expects it and the dark mode is properly dark at last.",
     "Positive",
     "The writer admits their scepticism was defeated - the redesign works well for them."),
    ("forum comment",
     "It's not that the projector is bad, exactly. It's that my phone torch is brighter. Make of that what you will.",
     "Negative",
     "The comparison to a phone torch is a damning judgement of the projector's brightness despite the soft phrasing."),
    ("customer review",
     "Nothing about this laptop is exciting, and that's exactly why I love it. It boots, it works, it never surprises me. Perfect office machine.",
     "Positive",
     "Boring reliability is framed as the reason the writer loves it - the verdict is explicit."),
    ("tweet",
     "The update didn't break anything, which by this app's standards counts as a triumph, I suppose. Still waiting on the features promised in January though.",
     "Negative",
     "The faint praise is backhanded - the real complaint is months of unshipped promised features."),
    ("customer review",
     "Against all odds, the budget drone survived my son's piloting. Propellers everywhere, frame intact. Whoever engineered this deserves a raise.",
     "Positive",
     "Surviving repeated crashes and the call for the engineer's raise are enthusiastic praise."),
]


# ------------------------------------------------------ capped summarisation
# (passage, shape_text, bullets) - every bullet hand-counted under its cap.
SUM_CAPPED: list[tuple[str, str, list[str]]] = [
    ("City cycling schemes are expanding fast as councils add protected lanes and hire fleets. Ridership doubled in three years in several pilot cities. But growth brings friction: bike theft is rising, docking stations overflow at commuter peaks, and drivers complain about lost parking. Councils are responding with GPS-tagged fleets, dynamic rebalancing vans, and consultation forums before each new lane.",
     "exactly three bullet points, each no longer than 12 words",
     ["Protected lanes and hire fleets doubled ridership in pilot cities.",
      "Theft, overflowing docks, and parking loss are growing frictions.",
      "Councils deploy GPS tags, rebalancing vans, and public consultations."]),
    ("Community solar projects let households buy shares of a local array instead of installing rooftop panels. Subscribers typically save ten to fifteen percent on bills. Critics note long waiting lists, complex contracts, and uneven state regulation. Developers now offer simplified agreements and guaranteed buyout clauses, while several states are standardising rules to speed approvals.",
     "exactly three bullet points, each no longer than 12 words",
     ["Households buy shares of local arrays, saving ten to fifteen percent.",
      "Waiting lists, complex contracts, and patchy regulation draw criticism.",
      "Simpler agreements, buyouts, and standardised state rules are emerging."]),
    ("Hospital pharmacies are piloting robotic dispensing to cut medication errors. Early sites report error rates falling by half and pharmacists freed for patient consultations. The robots are expensive, demand strict inventory formats, and stall on damaged barcodes. Vendors are adding vision systems that read imperfect labels, and hospitals are pooling procurement to share costs.",
     "exactly three bullet points, each no longer than 12 words",
     ["Robotic dispensing halves medication errors and frees pharmacists for consultations.",
      "High costs, strict formats, and damaged barcodes limit adoption.",
      "Vision systems and pooled procurement are addressing the barriers."]),
    ("Regional airports are courting electric aviation startups with free landing slots and charging infrastructure. Short-hop routes under 300 kilometres could cut fares and emissions substantially. Battery weight still restricts payloads, certification timelines stretch for years, and insurers remain cautious. Manufacturers are lobbying for phased certification and demonstrating safety with cargo-only routes first.",
     "exactly two sentences",
     ["Regional airports are offering slots and charging to attract electric aviation, promising cheaper, cleaner short-hop routes. Battery weight, slow certification, and cautious insurers remain hurdles that cargo-first trials and phased approval aim to clear."]),
    ("School districts adopting four-day weeks report lower costs and easier teacher recruitment. Families gain flexibility but scramble for Friday childcare, and researchers see mixed effects on attainment. Districts are extending remaining days, adding optional Friday enrichment, and monitoring test scores before committing permanently.",
     "exactly two sentences",
     ["Four-day school weeks cut costs and help recruitment while straining childcare and showing mixed attainment results. Districts extend other days, offer Friday enrichment, and track scores before permanent adoption."]),
    ("Vertical farms near city centres promise year-round greens with far less water and no pesticides. Energy bills for lighting remain the dominant cost, and some early ventures folded when electricity prices spiked. Operators now sign long-term renewable power deals, switch to efficient LED spectra, and focus on premium herbs where margins absorb the power bill.",
     "exactly three bullet points, each no longer than 15 words",
     ["Urban vertical farms deliver year-round greens using less water and no pesticides.",
      "Lighting energy dominates costs and has already sunk several early ventures.",
      "Operators respond with renewable power deals, efficient LEDs, and premium herb lines."]),
    ("Public libraries are lending laptops and hotspots to bridge the digital divide. Demand outstrips supply in most branches, with waiting lists spanning weeks. Damage and loss rates run higher than for books, straining budgets. Libraries now bundle digital-skills classes with loans and partner with telecoms for discounted replacement hardware.",
     "exactly three bullet points, each no longer than 12 words",
     ["Libraries lend laptops and hotspots to bridge the digital divide.",
      "Demand exceeds supply; damage and loss strain limited budgets.",
      "Skills classes and telecom partnerships stretch the programmes further."]),
    ("Textile recyclers are scaling fibre-to-fibre processes that turn old garments into new yarn. The technology handles pure cotton well but struggles with blended fabrics, which dominate fast fashion. Sorting remains manual and expensive. Brands are trialling recycling-friendly designs, and automated near-infrared sorters are entering pilot lines.",
     "exactly two sentences",
     ["Fibre-to-fibre recycling now turns pure-cotton garments into new yarn, but blended fabrics and costly manual sorting hold it back. Recycling-friendly designs and near-infrared sorting pilots are the industry's response."]),
]

# --------------------------------------------------------------- code_fix
# (name, spec, buggy, fixed, cause, tests[(args, expected)])
CODE_FIX: list[tuple[str, str, str, str, str, list[tuple]]] = [
    ("sum_odds", "return the sum of the odd numbers in a list",
     "def sum_odds(nums):\n    total = 0\n    for n in nums:\n        if n % 2 == 0:\n            total += n\n    return total",
     "def sum_odds(nums):\n    total = 0\n    for n in nums:\n        if n % 2 == 1:\n            total += n\n    return total",
     "The condition n % 2 == 0 selects even numbers instead of odd ones.",
     [(([1, 2, 3, 4, 5],), 9), (([2, 4],), 0)]),
    ("find_min", "return the smallest value in a non-empty list",
     "def find_min(nums):\n    smallest = nums[0]\n    for n in nums:\n        if n > smallest:\n            smallest = n\n    return smallest",
     "def find_min(nums):\n    smallest = nums[0]\n    for n in nums:\n        if n < smallest:\n            smallest = n\n    return smallest",
     "The comparison n > smallest tracks the maximum instead of the minimum.",
     [(([5, 3, 8, 1],), 1), (([2],), 2)]),
    ("count_words", "return the number of words in a sentence string",
     "def count_words(s):\n    return len(s)",
     "def count_words(s):\n    return len(s.split())",
     "len(s) counts characters, not words - the string is never split.",
     [(("the quick brown fox",), 4), (("",), 0)]),
    ("last_element", "return the last element of a non-empty list",
     "def last_element(lst):\n    return lst[len(lst)]",
     "def last_element(lst):\n    return lst[-1]",
     "lst[len(lst)] is one past the end and raises IndexError; the last index is len(lst) - 1 or simply -1.",
     [(([1, 2, 3],), 3), ((["a"],), "a")]),
    ("average_of", "return the average of a non-empty list of numbers",
     "def average_of(nums):\n    return sum(nums) / (len(nums) - 1)",
     "def average_of(nums):\n    return sum(nums) / len(nums)",
     "Dividing by len(nums) - 1 skews the mean - the divisor must be the full count.",
     [(([2, 4, 6],), 4.0), (([5],), 5.0)]),
    ("is_palindrome", "return True when a string reads the same forwards and backwards, ignoring case",
     "def is_palindrome(s):\n    return s == s[::-1]",
     "def is_palindrome(s):\n    s = s.lower()\n    return s == s[::-1]",
     "The comparison is case-sensitive, so 'Level' is wrongly rejected - lower-case the string first.",
     [(("Level",), True), (("hello",), False)]),
    ("first_duplicate", "return the first value that appears twice in a list, or None",
     "def first_duplicate(lst):\n    seen = set()\n    for x in lst:\n        seen.add(x)\n        if x in seen:\n            return x\n    return None",
     "def first_duplicate(lst):\n    seen = set()\n    for x in lst:\n        if x in seen:\n            return x\n        seen.add(x)\n    return None",
     "The value is added to seen before the membership check, so the very first element is always 'found'.",
     [(([1, 2, 2, 3],), 2), (([1, 2, 3], ), None)]),
    ("clamp", "clamp a number into the inclusive range [lo, hi]",
     "def clamp(x, lo, hi):\n    if x < lo:\n        return hi\n    if x > hi:\n        return lo\n    return x",
     "def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x",
     "The bounds are swapped: values below lo must clamp to lo, and values above hi to hi.",
     [((5, 1, 10), 5), ((-3, 1, 10), 1), ((42, 1, 10), 10)]),
    ("factorial", "return n! for a non-negative integer n",
     "def factorial(n):\n    if n == 0:\n        return 0\n    return n * factorial(n - 1)",
     "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)",
     "The base case returns 0, which zeroes out every product - 0! must be 1.",
     [((0,), 1), ((5,), 120)]),
    ("merge_sorted", "merge two sorted lists into one sorted list",
     "def merge_sorted(a, b):\n    return a + b",
     "def merge_sorted(a, b):\n    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i])\n            i += 1\n        else:\n            out.append(b[j])\n            j += 1\n    out.extend(a[i:])\n    out.extend(b[j:])\n    return out",
     "Concatenation preserves the two runs but not global order - the lists must be interleaved by comparison.",
     [(([1, 3, 5], [2, 4]), [1, 2, 3, 4, 5]), (([], [1]), [1])]),
    ("count_vowels_fix", "return the number of vowels in a string, case-insensitive",
     "def count_vowels_fix(s):\n    count = 0\n    for c in s:\n        if c in 'aeiou':\n            count += 1\n    return count",
     "def count_vowels_fix(s):\n    count = 0\n    for c in s.lower():\n        if c in 'aeiou':\n            count += 1\n    return count",
     "Upper-case vowels are missed because the characters are never lower-cased.",
     [(("AeIoU",), 5), (("XYZ",), 0)]),
    ("running_max", "return the running maximum of a list",
     "def running_max(nums):\n    out = []\n    best = 0\n    for n in nums:\n        if n > best:\n            best = n\n        out.append(best)\n    return out",
     "def running_max(nums):\n    out = []\n    best = None\n    for n in nums:\n        if best is None or n > best:\n            best = n\n        out.append(best)\n    return out",
     "Initialising best to 0 breaks all-negative inputs - the first element must seed the maximum.",
     [(([-5, -2, -7],), [-5, -2, -2]), (([1, 3, 2],), [1, 3, 3])]),
    ("remove_target", "return the list with every occurrence of target removed, without mutating the input",
     "def remove_target(lst, target):\n    for x in lst:\n        if x == target:\n            lst.remove(x)\n    return lst",
     "def remove_target(lst, target):\n    return [x for x in lst if x != target]",
     "Removing from the list while iterating it skips the element after each removal (and mutates the input).",
     [(([1, 2, 2, 2, 3], 2), [1, 3]), (([2, 2], 2), [])]),
    ("safe_divide", "return a / b, or None when b is zero",
     "def safe_divide(a, b):\n    if a == 0:\n        return None\n    return a / b",
     "def safe_divide(a, b):\n    if b == 0:\n        return None\n    return a / b",
     "The guard tests the numerator instead of the divisor, so dividing by zero still crashes.",
     [((10, 2), 5.0), ((1, 0), None), ((0, 5), 0.0)]),
]

CD_PHRASINGS = [
    "The following Python function is supposed to {spec} but contains a bug. Identify the bug and provide the corrected function.\n\n{buggy}",
    "This Python function should {spec}, but it has a bug. Find the bug and give the corrected implementation.\n\n{buggy}",
    "The function below is meant to {spec}. It contains a bug - identify it and provide a corrected version.\n\n{buggy}",
]


def verify_code_fix() -> None:
    for name, _spec, buggy, fixed, _cause, tests in CODE_FIX:
        for src, expect_pass in ((buggy, False), (fixed, True)):
            ns: dict = {}
            exec(src, ns)  # noqa: S102 - our own curated pairs
            fn = ns[name]
            passed = True
            for args, expected in tests:
                try:
                    if fn(*args) != expected:
                        passed = False
                except Exception:
                    passed = False
            assert passed == expect_pass, f"{name}: buggy/fixed verification failed (expect_pass={expect_pass})"


def build_rows(rng: random.Random) -> list[dict]:
    rows = []
    for container, text, label, rationale in SENTIMENT:
        prompt = (f"Classify the sentiment of this {container} as Positive, Negative, or Neutral, "
                  f"and briefly explain your reasoning in one sentence.\n\n{text}")
        rows.append({"family": "sentiment", "prompt": prompt, "answer": f"{label}. {rationale}"})
    for passage, shape, units in SUM_CAPPED:
        prompt = f"Summarize the following passage in {shape}:\n\n{passage}"
        answer = "\n".join(f"- {u}" for u in units) if "bullet" in shape else " ".join(units)
        rows.append({"family": "summarisation", "prompt": prompt, "answer": answer})
    verify_code_fix()
    for name, spec, buggy, fixed, cause, _tests in CODE_FIX:
        for phrasing in CD_PHRASINGS:
            prompt = phrasing.format(spec=spec, buggy=buggy)
            rows.append({"family": "code_fix", "prompt": prompt, "answer": f"{cause}\n\n{fixed}"})
    rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="finetune/minicpm5/data/assist_claude_authored.jsonl")
    ap.add_argument("--seed", type=int, default=20260710)
    args = ap.parse_args()
    rows = build_rows(random.Random(args.seed))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fams: dict[str, int] = {}
    for r in rows:
        fams[r["family"]] = fams.get(r["family"], 0) + 1
    print(f"wrote {len(rows)} rows -> {out}")
    print("per family:", json.dumps(fams, sort_keys=True))


if __name__ == "__main__":
    main()
