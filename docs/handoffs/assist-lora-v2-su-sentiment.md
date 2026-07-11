# Handoff: retrain the ASSIST LoRA (Qwen3.5-2B) - summarisation + sentiment

Self-contained brief - assume NO prior context, memories, or other sessions.
Everything you need is in this file and the repo at
`/Users/mohamed.ashaibani/Playground/AMDHackathon`.

## Mission context (read first)

- Project `yassai`: Go agent for the AMD Developer Hackathon Act 2, Track 1.
  Scoring: LLM-judge accuracy gate, then ascending Fireworks tokens; local
  in-container inference counts zero and `ZERO_API_CALLS` is valid. The live
  submission (image `sha-f80f152`, title 'yassai - fully local, zero API
  calls') is FULLY LOCAL: one stock `Qwen3.5-2B` Q4_K_M + TWO serve-time
  LoRAs, `AGENT_NO_REMOTE=1` baked, in-container 18/19 (0.947) @ 0 tokens.
  At 0 tokens the tiebreak is accuracy, and the final judging may use unseen
  prompts - generalisation beyond the sample set is the whole game now.
- Judge VM: 4 GB RAM, 2 vCPU, linux/amd64, 10-min cap. Lanes are
  sequential-resident llama-servers.
- Two lanes, two adapters on ONE shared base:
  - **Tool lane** (`q35-tool-v1` LoRA): maths + logic via a `run_python`
    contract. NOT yours - touch nothing tagged `q35-tool-*`, `cluev1`, or
    `v2-e3-r32`.
  - **Assist lane** (`yassai-assist-e3-r32-q35-2b` LoRA): direct answers for
    code generation, NER, sentiment, summarisation, factual, code fixes.
    **This adapter is what you retrain.**
- Your mission: produce ONE improved assist LoRA (suggested tag
  `q35-assist-v2`) that (A) learns compound-constraint summarisation and
  (B) fixes a sentiment regression versus stock - without regressing any
  other assist family. Deliver a results memo + artefacts; do NOT integrate
  (no CI, Dockerfile, or submission changes - that is the main session's
  decision).

## Hard rules (non-negotiable)

- **NEVER train on anything in `testdata/`** - `downloads_tasks_golden.json`
  is the live sample set; variant/wildcard/golden/holdout files are eval
  sets; all are radioactive for training, mining, and few-shot exemplars.
  Parametric generation with derived answers is the intended play. Every
  data builder asserts overlap against these files by id AND prompt text -
  keep the asserts and add them to anything new (a DPO miner once smuggled
  two sample tasks; the asserts exist because of that).
- API keys: `source ~/config/.env` (FIREWORKS_API_KEY, HF_TOKEN, GH_TOKEN).
  Never print values. Fireworks/umans concurrency <= 3.
- Unique `--tag` on every training run (an untagged rerun once overwrote a
  rollback path). Defaults that work: 3 epochs (not 5), LoRA r32, lr 1e-4.
- **Acceptance only at the served quant**: stock Q4_K_M base + your LoRA as
  an f16 GGUF through llama-server b9948 `--lora` `--reasoning off`. bf16
  transformers numbers do not count - a GRPO run once lifted in-distribution
  rewards 0.29 -> 0.66, passed the behaviour gate 846/846, and still
  REGRESSED the served fresh holdout; it was rejected. Merged GGUF export is
  BROKEN for Qwen3.5 on the pinned converter (MTP block `blk.24` missing) -
  serve-time LoRA is the only ship shape.
- Eval variance is +/-1-2 tasks per set; only deltas >= 2 are signal.
- UK English, no em or en dashes, in everything you write.

## Task A: compound-constraint summarisation (the prize, ~3-4 judged tasks)

**The evidence.** Audit of the training data (11 Jul 2026): the assist
data's summarisation slice is 55 judge-filtered teacher rows, of which
**ZERO contain a word cap and ZERO are compound-constraint**. The model has
never seen the skill it is failing. The failures at the served build:

- `T04b` (sample set): the SOLE golden miss - 'exactly three bullet points,
  each at most twelve words, own words'.
- `su_v02`, `su_a01` (variant set), `su_w01` (wildcard, theme-coverage
  miss). `su_v01` and `su_w02` pass.

**The failing skill** is satisfying four constraints TOGETHER: exactly N
units (bullets or sentences), each unit <= M words, own words (no verbatim
copying), full theme coverage. Measured lessons: single-axis fixes break
another axis, and LOWER word targets backfire into verbatim copying - do
not author 'easier' short-cap data. The serving path already retries with
an explicit rewrite nudge, a clause-aware trim, and a polish pass
(`internal/localllm/direct.go`: `summariseRewriteNudge`, `trimToWordCap`,
`gateSummarisation`, `summaryEchoesSource`) - your job is first-pass
competence so those retries stop being load-bearing.

**Attack 1 - parametric su SFT data (the banker).**

- Build a generator (new script under `scripts/`) that assembles passages
  from theme banks: pick a domain, pick 3-5 themes, render each theme as
  1-3 sentences plus filler, so the DERIVED THEMES AND KEYWORDS ARE EXACT
  at build time. Style prompts like the real family: single-quoted passage
  + 'Summarise ... in EXACTLY three bullet points ... each no longer than
  twelve words ... in your own words'. Vary: count 2-4, cap 10-20 words,
  bullets vs sentences, and 'must mention the percentage' variants.
- Machine-verify every assistant target before inclusion: exact unit
  count; per-unit word cap; no 8-or-more-word contiguous run shared with
  the passage (port `summaryEchoesSource` from
  `internal/localllm/direct.go` - normalise with lowercase + non-alnum
  split); every theme's keywords represented. Reject and regenerate until
  clean. Targets can come from a strong teacher via Fireworks
  (`scripts/gen_assist_teacher_data.py` shows the teacher + judge-filter
  pattern; note Fireworks WAF 403s the default python-urllib User-Agent -
  set a custom one) or be authored directly.
- Volume: ~80-150 verified rows, merged via
  `scripts/build_minicpm5_assist_data.py` (extend it; keep the other
  slices' proportions so nothing dilutes).
- **Prompt contract**: the per-family SYSTEM instruction in training must
  stay byte-identical to serving - for summarisation that is
  `"Obey the stated length and format limits exactly; cover every major
  theme; no preamble."` (see `familyInstruction` in
  `internal/localllm/direct.go`). The skill goes in the responses, never
  in a new system prompt.

**Attack 2 - GRPO (stretch, only if SFT leaves T04b-class still failing).**

- Summarisation is the best RLVR domain left: every reward component is
  programmatic. Port the gates into
  `finetune/minicpm5/rewards_rlvr.py` as a new reward: unit count, per-unit
  word cap, anti-echo 8-word shingle, theme-keyword recall (the pool
  carries derived themes in the truth column - `train_grpo.py` already
  passes truth through), small penalty for preamble/prose outside the
  units. Anti-hack: require a minimum word floor per unit AND coverage, so
  degenerate two-word bullets cannot win.
- Pool builder precedent: `scripts/build_logic_grpo_pool.py` (56-puzzle
  pool with derived truths and overlap asserts) - copy its shape.
- **KNOWN BLOCKER**: TRL GRPOTrainer's import graph on the Qwen image
  (transformers 5.13) reaches vllm/mergekit and neither installs cleanly.
  Options are written up in
  `docs/handoffs/qwen35-tool-lane-results.md` (GRPO status section): pin
  an import-clean TRL, or vendor a minimal trainer. Pick ONE, do not
  thrash. `train_grpo.py` gotchas: `gradient_accumulation_steps` MUST
  equal `num_generations`; Qwen LoRA targets are already in
  `finetune/minicpm5/train_trl.py` (hybrid Gated DeltaNet:
  `in_proj_{qkv,a,b,z}`, `out_proj`, etc.).
- GRPO acceptance is the same served-quant fresh-holdout law as SFT.

## Task B: sentiment rebalance (small, surgical)

**The evidence.** On the fresh 8-task sentiment holdout
(`testdata/sentiment_holdout_tasks.json`, never trained or mined): stock
Qwen3.5-2B scores 8/8; the current assist LoRA scores 7/8 - **our adapter
degrades stock**. The miss is a balanced/mixed review pushed to Negative.
Cause hypothesis: the sentiment slice (90 teacher rows + 69 Claude-authored
rows in `finetune/minicpm5/data/assist_claude_authored.jsonl`) was authored
against the 1B's sarcasm-blindness; the 2B already reads sarcasm, so the
sarcasm-heavy mix injects a negativity bias.

- Fix: audit the 159 rows' label distribution; add mixed-review ->
  Neutral/positive counterweight rows (precedent: 24 authored positive
  counterweights fixed the same failure class on the 1B) and/or downweight
  the sarcasm-heavy slice.
- DPO is NOT the lever - two controlled DPO passes on the 1B moved the
  holdout by exactly 0. This is data composition.
- Acceptance: holdout back to 8/8 AND the 6 sentiment tasks in the
  43-variant set held at 6/6. If a trade appears, the holdout wins (it is
  the unseen-generalisation proxy, and unseen prompts are what final
  judging may use).

## Asset inventory (all committed, all working)

| Asset | Path / invocation |
|---|---|
| Teacher data (203 rows: sent 90, fact 58, su 55) | `finetune/minicpm5/data/assist_teacher_raw.jsonl` |
| Authored data (111 rows: code_fix 42, sent 69) | `finetune/minicpm5/data/assist_claude_authored.jsonl` |
| Data builder (merges + parametric ner/cg, md5%7 teacher split, overlap asserts) | `scripts/build_minicpm5_assist_data.py --out finetune/minicpm5/data/minicpm5_yassai_assist.jsonl` |
| Trainer (H100, ~2 min) | `uvx modal run --detach finetune/minicpm5/modal_train.py --dataset assist --epochs 3 --tag q35-assist-v2 --base-model Qwen/Qwen3.5-2B` |
| Behaviour gate (held-out split; env `BASE_MODEL`/`ADAPTER`/`DATA`) | `finetune/minicpm5/eval_assist_behavior.py` (prior assist run: cg 9/9, fact 11/11, ner 12/12, sent 11/12, su 10/11) |
| LoRA GGUF export (f16, ~67 MB) | `finetune/minicpm5/modal_export_lora_gguf.py` |
| amd64 serve canary (prod args on Modal CPU) | `finetune/minicpm5/modal_validate_assist_lora.py` |
| Direct-lane probe (bypasses agent) | `cmd/localprobe -model models/qwen35/Qwen_Qwen3.5-2B-Q4_K_M.gguf -lora <adapter>.gguf -tasks testdata/<set>.json` |
| Full-agent eval + Fireworks judge | `cmd/realeval` (see README 'Reproducing the numbers'), `cmd/scorefile` to judge saved answers |
| Current adapter (your baseline) | `models/qwen35/yassai-assist-e3-r32-q35-2b-lora-f16.gguf` |
| Stock base Q4 | `models/qwen35/Qwen_Qwen3.5-2B-Q4_K_M.gguf` (also on HF `ashaibani/yassai-minicpm5-local`) |
| llama-server b9948 | `~/opt/llama` (NOT brew, NOT scratchpad) |
| Modal volume | `yassai-minicpm5-checkpoints` (fetch artefacts with `modal volume get` if the HF storage cap bites) |

## Ship-bar table (fill this in for the results memo)

Re-baseline the CURRENT adapter on every row before training - judge
variance is +/-1-2 and the numbers below are from single 11 Jul runs.

| Set | Current (11 Jul, served quant) | Ship bar |
|---|---|---|
| NEW su holdout (you author it - see below) | measure first | baseline + >= 2 |
| Sample T04 + T04b (realeval) | T04 pass, T04b FAIL | 2/2 (T04b is the prize) |
| Variant su (`su_v01/v02/a01`) | 1/3 | >= 2/3 |
| Wildcard su (`su_w01/w02`) | 1/2 | 2/2 |
| Sentiment holdout (8) | 7/8 (stock 8/8) | 8/8 |
| Variant sentiment (6 in the 43-set) | 6/6 | 6/6 held |
| Variants total (43) | 38/43 | >= 38, and no other family down >= 2 |
| Behaviour gate (held-out) | green | green |
| amd64 canary | HEALTHY | HEALTHY |

The internal 40-task practice set `testdata/golden.json` (`su1..su5`) is a
free extra eval signal - eval-only like everything else in `testdata/`.

## Verification order (each step gates the next)

1. **Re-baseline** the current adapter on the full table. Hygiene first:
   `pkill -f llama-server` (spare port 42343 - an unrelated embeddings
   server); health-wait must grep for `"ok"` (the server returns 503-JSON
   while loading and curl exits 0 on it).
2. **Author the su holdout**: ~10 fresh compound-constraint tasks ->
   `testdata/su_holdout_tasks.json`, eval-only forever, styled like the
   family but with novel passages. Baseline the current adapter on it.
3. **Build the data** (su expansion + sentiment rebalance) with overlap
   asserts against ALL `testdata/` files.
4. **SFT** with `--tag q35-assist-v2`; behaviour gate must be green.
5. **Export** the LoRA f16 GGUF; run the full served-quant table locally.
6. **amd64 canary** via `modal_validate_assist_lora.py`.
7. **GRPO stretch** only if T04b-class tasks still fail and time permits.
8. **Write the results memo** (see definition of done).

## Traps - each already paid for once, do not pay twice

- The served-quant law and the +/-2 noise floor (both above) dominate every
  decision. macOS Metal and amd64 CPU decode DIVERGE on borderline tasks -
  macOS is fine for iteration, but borderline calls are only truthful on
  amd64/in-container.
- Byte-identical SYSTEM contract between training rows and serving.
- `modal run --detach` always (ephemeral apps die with the client - a
  usage-limit disconnect once killed a live run); never launch duplicate
  training clients (a retry race once produced triple H100 jobs).
- Eval-on-train masks overfit - respect the builder's md5%7 teacher split;
  the behaviour gate runs on the held-out side.
- Do not 'simplify' su data to short word caps - lower caps measurably
  cause verbatim copying, which the anti-echo gate then rejects.
- transformers 5 returns BatchEncoding where old code expected tensors
  (`eval_tool_behavior.py` has the fixed pattern if you hit it).
- HF repo `ashaibani/yassai-minicpm5-local` sits near its storage cap;
  keep the CI-referenced GGUFs untouched and prefer `modal volume get`
  for moving artefacts.

## Definition of done

A memo at `docs/handoffs/assist-lora-v2-results.md` containing: the
ship-bar table filled in (baseline vs new, served quant), Modal app ids +
volume paths + the LoRA GGUF artefact location, the data changes (row
counts per slice, verification method), and a GO / NO-GO recommendation on
replacing `yassai-assist-e3-r32-q35-2b-lora-f16.gguf`. Do NOT change any
CI default, Dockerfile line, or the live submission - adoption is a
separate decision with the main session.
