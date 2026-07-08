# Category → model routing (working notes)

Candidates (from `ALLOWED_MODELS`, expected IDs):
- `accounts/fireworks/models/minimax-m3` — MiniMax-M3 (428B total / 23B active MoE, reasoning)
- `accounts/fireworks/models/kimi-k2p7-code` — Kimi K2.7 Code (1T total / 32B active MoE)

## Artificial Analysis prior (scraped 08 Jul 2026, AA Intelligence Index v4.1)

Higher is better unless noted. Source: RSC data on the AA eval pages.

| AA metric | → our category | MiniMax-M3 | Kimi K2.7 Code | Prior winner |
|---|---|---|---|---|
| AA-Omniscience (knowledge reliability) | factual_knowledge | **+1.37** | −10.7 | M3 (large) |
| GPQA Diamond | factual / sci reasoning | **0.929** | 0.896 | M3 |
| Humanity's Last Exam | factual / reasoning | **0.371** | 0.328 | M3 |
| CritPt (hard physics/reasoning) | logical_deductive | 0.037 | **0.100** | Kimi |
| SciCode | code_generation | 0.454 | **0.475** | Kimi (slight) |
| coding_index | code_generation/debug | 58.6 | **60.8** | Kimi (slight) |
| Terminal-Bench v2.1 | code_debugging | 0.652 | **0.674** | Kimi (slight) |
| Terminal-Bench hard | code_debugging | 0.424 | **0.447** | Kimi (slight) |
| AA-LCR (long context) | text_summarisation | **0.74** | 0.663 | M3 |
| IFBench (instruction following) | ALL (format compliance) | **0.829** | 0.631 | M3 (large) |
| τ²-bench (agentic tool use) | (agent loop) | 0.889 | **0.901** | Kimi (slight) |
| Intelligence Index v4.1 (overall) | — | **44.4** | 41.9 | M3 |
| Blended $/1M | — (cost) | **~£0.22** | ~£0.52 | M3 (3×) |
| Output tokens on AA suite | — (verbosity) | 512k (reasoning) | 256k | — |

## Prior map hypothesis (to validate empirically — NOT final)

- **MiniMax-M3 = default.** Wins factual/knowledge, long-context, instruction-following (format!), overall intelligence, and cost. Best all-rounder for our mostly-easy tasks.
- **Kimi K2.7 Code = route candidate for code_generation + code_debugging** (and maybe hard logical puzzles). Edge is small and measured on *frontier* coding.

## Caveats that make the empirical test mandatory

1. **Coding edge is tiny (~2 pts) and on frontier benchmarks.** Our coding tasks ("write a function from a spec", "fix a bug in a snippet") are far easier; the gap likely shrinks or vanishes → routing may not help.
2. **Kimi's IFBench is much weaker (0.63 vs 0.83).** Our output must be exact JSON with preserved `task_id`s and requested formats. Weak instruction-following is a real accuracy-gate risk *across every category*, including code — a perfect function inside malformed JSON still fails.
3. **The token metric counts reasoning tokens.** M3 is a reasoning model and emits ~2× the output tokens on AA's suite. Controlling reasoning-token emission (the concise-prompt work) likely moves the token rank MORE than model routing does. Measure tokens *including* reasoning on our tasks.

## Decision rule for the map (metric-aligned)

Per category, pick the model with the **fewest tokens among those that clear the accuracy gate** on our eval — not the highest raw benchmark score. Route away from the M3 default only where Kimi K2.7 Code both passes the gate and uses fewer tokens net of format-failure risk.

## Empirical probe result (08 Jul 2026) — `cmd/routeprobe` on golden 40

Per-category batches, classifier off, gate = golden validators.

| category | M3 acc / tok | K2.7 Code acc / tok |
|---|---|---|
| code_debugging | 5/5 / 7,849 | 5/5 / 7,743 |
| code_generation | 5/5 / 7,309 | 5/5 / 7,910 |
| factual | 4/5 / 4,389 | 5/5 / 4,350 |
| logical_reasoning | 5/5 / 5,654 | 5/5 / **14,243** |
| maths | 5/5 / 4,991 | 5/5 / 4,772 |
| ner | 5/5 / 9,310 | 5/5 / 9,531 |
| sentiment | 5/5 / 4,666 | 5/5 / 4,515 |
| summarisation | 5/5 / 9,242 | 5/5 / 9,322 |
| **TOTAL** | **39/40 / 53,410** | **40/40 / 62,386** |

**Verdict: do NOT build category → model routing.**
- Accuracy is tied at ~ceiling on our (easy) tasks — no accuracy-driven routing case. Per-category "winners" are decided by noise-level token deltas, except logical_reasoning where Kimi over-reasoned (14.2k vs 5.7k, 2 calls, 68s).
- MiniMax-M3 is the leaner single default: −14% tokens (53.4k vs 62.4k) at equal accuracy (its 1 miss is one factual task). Kimi emits more reasoning tokens across the board.

**Real token levers (ranked), from the probe:**
1. **Batch amortisation** — ~80% of each call's tokens are the fixed prompt (~4–7k input regardless of task). Fewer, bigger batches amortise it. Biggest, most certain win.
2. **System-prompt trim** — cuts the dominant per-call cost directly.
3. **Reasoning-token control** (concise prompt / low reasoning effort) — keep M3's reasoning lean; Kimi's logical blowup is the anti-pattern.
4. **Prompt caching** — CachedTokens was 0; enabling shared-prefix caching would slash repeated prompt cost.

Caveat: 40-task set, lenient heuristic validators (some auto-pass) — accuracy is a coarse proxy; the token findings are robust regardless.

## Batch-size sweep (08 Jul 2026) — M3 on golden 40

| batch | tokens | calls | accuracy | time |
|---|---|---|---|---|
| 1 | 78,865 | 42 | 38/40 | 77.8s |
| 5 | 62,209 | 9 | 40/40 | 52.0s |
| 10 | 36,797 | 4 | 40/40 | 19.2s |
| 20 | 25,356 | 2 | 40/40 | 15.6s |
| 40 | 16,481 | 1 | 40/40 | 37.7s |

**Batching is the dominant token lever: batch-40 = −79% tokens vs batch-1 (16.5k vs 78.9k), accuracy equal-or-better.** Current default MaxBatchSize=20 (25.4k) leaves ~35% vs batch-40. Bigger wins because tasks are tiny and the ~6k system prompt is then paid once.

Ceiling caveats (why not always max out the batch):
- **Output-token limit** — all answers come back in one response; long answers (code, summaries) can truncate a big batch.
- **Per-request latency** — batch-40 took 37.7s; the guide's "response time per request < 30s" may bite (batch-20 was 15.6s). Ambiguous whether it means per API call or the whole run (≤10 min); treat as a soft cap.
- **Accuracy at scale on *harder* tasks** — golden tasks are tiny; a real 40-task batch of long prompts may degrade.
→ Pack batches by **token budget** (not fixed count), output-aware, and validate the ceiling on harder/longer tasks.

## Crystallised token strategy (evidence-backed)
1. **One model: MiniMax-M3** (leaner, ties accuracy). No category router, no category grouping — simpler wins.
2. **Batch as large as the token budget + latency + accuracy safely allow** (biggest lever, up to ~5×).
3. **Trim the system prompt** (~6k fixed) — compounds at every batch size.
4. **Keep reasoning lean** (concise prompt / low reasoning effort).

## REVISION after re-run + variance check (08 Jul 2026)

Re-running batch-20 vs batch-100 on the same golden 40 overturned "bigger is always better":
- **batch-20 is stable**: ~25-27k tokens across runs, 40/40.
- **batch-40/100 swing wildly**: 16.5k (first sweep, one clean turn) vs 46.5k (re-run: turn-0 emitted 16k output → failed to parse → turn-1 added 14.7k reasoning). Same tasks, 3× swing. `tool_runs=0` — pure generation blow-up, not tool use.

Revised stance:
- The ~4k fixed per-call overhead is **injected default memory** (`MEMORY.md` + 4 seeded `memories/*.md`, ~16 KB), not the 412-token system prompt.
- Batching amortises that fixed INPUT, but beyond ~20 it destabilises OUTPUT/REASONING and can net-increase tokens. **Sweet spot ≈ batch 20.** Kept the adaptive token-budget packing (`planBatches`, splits long-task batches) but capped the default at 20.
- The dominant *reliable* lever is now **generation control**: terse output contract, sane `max_tokens`, suppress needless reasoning / second turns. This is where the concise-prompt work pays off.
- **Measurement must use repeats** — single runs swing ~3×.

## Correction after instrumenting the real prompt (08 Jul 2026)

Measured composition (`promptsize_test.go` + call records), which overturns the "trim the prompt" thesis:
- **Fixed text overhead is only ~376 tok** (system 208 + scaffold 167), NOT the ~4k earlier estimated. Removing the seeded dev-memory + tightening the system prompt saved ~1-2k — small (memory was ~700 tok, not 4k). Still worth doing (the seeded notes were off-topic/confusing).
- A 20-task batch's ~9.6k prompt is **mostly real task content** (summarisation passages, NER text) — a hard floor, not trimmable.
- **Reasoning is the dominant controllable cost**: the harder half of a batch emitted 7,646 reasoning tokens vs 1,285 on the easy half.

Net effect of this iteration (minimal memory + terse contract + cap-20 batching): **killed the catastrophic blow-ups** (46k → stable 22-27k across 3 repeats) at 39-40/40. The real win was **stability + correctness**, not a big token cut. calls=2 = two single-turn batches (the terse contract stopped the extra reasoning turn).

Revised lever ranking:
1. **Reasoning-token control** (`reasoning_effort=low` / brief-reasoning directive) — the real remaining headroom.
2. Prompt caching across batches within a run (stable prefix first) — minor.
3. Task content is a floor (it's the judge's input) — not reducible.

## Adaptive reasoning-effort experiment (08 Jul 2026) — adaptive LOSES, uniform low WINS

`reasoning_effort` is honoured by M3 (code_generation reason 550→88, summarisation 862→114, logical 13,391→7,150 high vs low). M3 over golden 40, per-category batches:

| effort | accuracy | total tokens |
|---|---|---|
| high | 40/40 | 70,580 |
| adaptive (hard=high, easy=low) | 40/40 | 63,305 |
| **low (uniform)** | 40/40 | **47,662** |

**Verdict: default `reasoning_effort=low`; do NOT do adaptive-by-category.** Higher effort buys zero accuracy here (all 40/40) and only burns tokens — worst on `logical_reasoning`, where high blew up to 28.9k tok / 5 calls / 94s vs low's clean 6.2k / 1 call. The "hard categories need more effort" premise is false for this difficulty: they're already correct at low; extra effort over-thinks and destabilises. Easy categories are unaffected by the param.

`low` also beats the no-param default (47.7k vs 53.4k, −11%) and fixed the 1 factual miss (now 40/40). Default set to `AGENT_REASONING_EFFORT=low`; `effortForBatch` keeps the adaptive path available but it is not the default. This is the real reliable token lever found in the whole investigation.

## Real-benchmark eval (08 Jul 2026) — `cmd/realeval` + structured umans judge

48 real tasks (TriviaQA, GSM8K, SST-2, CNN/DM, CoNLL-2003, HumanEvalFix, LogiQA, HumanEval), deployed config (M3, effort=low, batch≤20, minimal memory, terse). Scoring: `numeric` for maths; a **structured** umans-flash LLM judge (`{"verdict","reason"}` via `response_format:json_object`) for the other 7 categories.

| category | acc |
|---|---|
| factual / maths / sentiment / summarisation / code-gen / code-debug | 6/6 each |
| ner | 4/6 |
| logical | 3/6 |
| **OVERALL** | **43/48 (89.6%)**, 63.7k tokens, 4 calls, 107s |

- An earlier 18.8% was a **judge bug**: umans-flash is a reasoning model that returned `content=null` (all text in `reasoning_content`, truncated by `max_tokens`). Fixed with `max_tokens=2048` + `reasoning_effort=low` + a **structured JSON verdict** parsed like the agent's answers (not string-sniffing).
- Real weaknesses: **logical** (LogiQA constraint puzzles — genuine reasoning misses) and **NER** (partly reference-strictness: CoNLL tags MISC/nationality entities like "Uzbek"/"Soviet" that the prompt didn't request).
- Open question the easy golden set couldn't answer: does higher `reasoning_effort` recover **logical** accuracy on genuinely hard tasks?

## Adaptive effort (4-tier) + Kimi-code routing on real tasks (08 Jul 2026)

4-tier map: logical→xhigh, maths/code→high, summ/ner→medium, factual/sentiment→low. `max_tokens` scaled per tier so higher effort doesn't truncate.

| config | overall | logical | ner | code (gen/debug) | tokens | time |
|---|---|---|---|---|---|---|
| uniform low (M3) | 43/48 | 3/6 | 4/6 | 6/6 | 63.7k | 107s |
| adaptive 4-tier (M3) | 44/48 | 4/6 | 4/6 | 6/6 | 79.9k | 151s |
| adaptive + code→Kimi K2.7 Code | 46/48 | 5/6 | 5/6 | 6/6 | 64.5k | 113s |

Single runs — reasoning models swing ~3×, so treat small diffs as noise. Reads:
- **Kimi-for-code buys nothing:** code was 6/6 on M3 in all three; routing to Kimi can't beat ceiling and adds a 2nd model (cache fragmentation, weaker instruction-following, format risk). Skip.
- **Adaptive effort is marginal + noisy:** only `logical` plausibly benefits (xhigh), and 3/4/5 is within variance. The full ladder over-spends — maths/code/summ/ner were already 6/6 at `low`, so raising them cost tokens for nothing (+25% in the M3 run).
- The overall 43/44/46 spread is entirely logical+ner (both M3, both noisy), not the interventions.
- Metrics note: Kimi doesn't report reasoning tokens, so the client counts reasoning *chars* (inflated); `total_tokens` is the reliable figure.

Verdict: no clean win on this eval. Token-optimal + simplest = **uniform low**; at most **xhigh-for-logical-only** as an accuracy-gate hedge. Do **not** route code to Kimi. Settle the marginal logical gain with repeats if it matters for the gate.

## Guide update + practice-set eval (08 Jul 2026)

Updated participant guide adds: (1) an 8-task **practice set**, (2) explicit rule that **local model inference in the container is permitted and counts as ZERO tokens** (a `ZERO_API_CALLS` marker is "a valid strategy"), (3) grading env **4 GB / 2 vCPU** (2-3B 4-bit local models fit), (4) a failure-mode table (PULL_ERROR/amd64, RUNTIME_ERROR, TIMEOUT, OUTPUT_MISSING, INVALID_RESULTS_SCHEMA, MODEL_VIOLATION, IMAGE_TOO_LARGE, ACCURACY_GATE_FAILED).

Changes made:
- **Runtime image → `python:3.12-slim`** (was distroless): distroless has no shell, so `sh.run`/`python3` were dead. Now the code-execution action space works in the container (a zero-Fireworks-token compute lever).
- **Category technique-hints** (`categoryHints` / `categoryGuidance`, injected per batch by category): logical → solve constraint puzzles *with code* (enumerate + filter constraints; `python3`/itertools now available); NER → exhaustive typed extraction. `Config.Categories` lets eval drive these off true categories.
- Config: adaptive + code→Kimi (per user), `effortTier` = logical→xhigh only.

Practice-set eval (adaptive + Kimi-code + skills + xhigh judge): **8/8 (100%)**, 4,492 tokens, 3 calls, 9.6s. Validates the full pipeline; tasks are easy so they don't stress the skills (whose payoff is the hidden hard constraint puzzles).

Open: the **local-model strategy** now explicitly allowed (zero token score) is the strongest possible token rank if a small in-container model can clear the accuracy gate — the direction the earlier Gemma/Grug idea was aiming at.

## Classifier "not running" — RESOLVED (08 Jul 2026): QEMU emulation, not a bug

Symptom: in the amd64 image run via `docker run` on an arm64 Mac (QEMU), the classifier produced wrong/uniform predictions → `calls=1`, no adaptive tiering. Diagnosis:
- **Tokenisation is correct**: Go (daulet) IDs match the Python pipeline exactly, incl. `[CLS]`=50281/`[SEP]`=50282 and mask all 1s.
- **Inference is correct natively**: the taskclf integration test PASSES on native arm64 (macOS arm64 onnxruntime dylib) — all 7 categories right; and the full agent run natively **tiers (`calls=2`)** with correct answers (`practice-07` logical → "Sam").
- **Only QEMU-emulated amd64 onnxruntime (SIMD/int8 kernels) mis-computes.** Native amd64 (the judge VM) is unaffected.

Conclusion: the classifier is **not broken**; it works on the native amd64 judge, so adaptive + skills fire there. Earlier "real bug" call was wrong — it's QEMU. Kept improvement: `classifyTasks` now logs inference errors (no silent degradation). Caveat: the classifier-driven path can't be container-tested on an arm64 Mac; for 100% certainty run the amd64 container on the native-amd64 AMD box, but native-arm64 + Python-amd64 evidence make it near-certain.
