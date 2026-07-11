# Handoff: scout small specialised models + training datasets per category

Self-contained brief - assume NO prior context, memories, or other sessions.
Everything you need is in this file and the repo at
`/Users/mohamed.ashaibani/Playground/AMDHackathon`. This is a RESEARCH
mission: survey, verify, prototype the best one or two - not a training run.

## Mission context (read first)

- Project `yassai`: Go agent for the AMD Developer Hackathon Act 2, Track 1.
  Scoring: LLM-judge accuracy gate, then ascending Fireworks tokens; LOCAL
  inference counts zero tokens. Final judging runs the submitted Docker image
  on a HIDDEN task set ('same format, difficulty' as the published samples)
  across eight categories: factual, maths, sentiment, summarisation, NER,
  code_debugging, logic, code_generation.
- Judge VM: **4 GB RAM, 2 vCPU, linux/amd64, 10-minute cap**. Hosts are often
  contended (2-3x slowdowns measured) - wall clock is a first-class
  constraint.
- Current architecture: ONE stock `Qwen3.5-2B` Q4_K_M served by llama.cpp
  b9948 llama-server with TWO serve-time LoRAs (tool lane: maths/logic via a
  `run_python` contract; assist lane: everything else), plus an optional
  MiniCPM5-1B tool-lane variant, an ONNX task classifier for routing
  (onnxruntime 1.27 ships in the image already), and deterministic evidence
  gates around every answer. Lanes are sequential-resident: one model in RAM
  at a time, each spawn costs 10-60s on a contended host.
- Hackathon deadline: 12 July 22:05 BST. Split every recommendation into
  'landable before the deadline' vs 'future work' - the research value
  extends beyond the event.

## Hard rules

- **NEVER train or mine on anything in `testdata/`** - those are the eval
  sets (sample tasks, 43 adversarial variants, 18 wildcards, per-family
  holdouts). Public external datasets are fine. Data builders in this repo
  assert overlap against `testdata/` by id and prompt text; keep that
  pattern for anything new.
- Keys: `source ~/config/.env` (FIREWORKS_API_KEY, HF_TOKEN). Never print
  values. Fireworks concurrency <= 3.
- Local models are unrestricted (the ALLOWED_MODELS rule applies only to
  Fireworks API calls). Licences still matter: prefer Apache-2.0/MIT.
- Acceptance law for ANY quality claim: measured at the SERVED quantisation
  on amd64 CPU, on eval-only sets, with the +/-2-task noise floor respected.
  bf16 GPU numbers do not count (a GRPO run once lifted bf16 rewards
  0.29 -> 0.66 and regressed the served holdout).
- UK English, no em or en dashes, in everything you write.

## Current per-category truth (the improvement targets)

Measured on unseen-style sets (wildcard/variants/holdouts, judge-aligned
scorer), current lanes:

| Category | State | Residual failure class |
|---|---|---|
| maths | strong: 11/11 variants; exact ratio splits now answered deterministically | word-problem phrasing diversity |
| logic | strong: 6/6 variants, clue holdout 7/8 | deep semantic-clue encoding ('allergic to fur' -> exclusion sets) |
| code_generation | strong: ~5/5 | edge-case specs (cg_v02/cg_a02 class) |
| sentiment | strong: holdout 7-8/8 | mixed-review balance nuance |
| NER | good | label wobble (rubric tolerates ONE mislabel, zero missing entities) |
| summarisation | WEAK: compound constraints + theme coverage (T04b/su_w01 class) | multi-theme coverage under count+word-cap limits |
| code_debugging | WEAK-ish: cd_w02 class | partial fixes (fixes one case of a bug, misses the sibling case) |
| factual | knowledge ceiling: f_w01 class | hallucinated proper nouns (invented lake names); 2B world-knowledge limit |

The published Judging FAQ (11 Jul) revealed the actual rubrics - design
toward them: sentiment accepts Mixed/Neutral/Positive but REQUIRES a
both-sides reason on two-sided reviews; NER tolerates one mislabel but no
missing entity; summarisation demands two-sided coverage plus exact counts
and word caps; maths tolerates rounding when the final value is right. The
aligned rubric texts live in `testdata/downloads_tasks_golden.json`
(expected fields).

## Constraints for ANY new model (verify each explicitly)

1. Servable by ONE of: llama.cpp **b9948** as GGUF (check that exact build
   supports the architecture - newer archs may not be), onnxruntime 1.27
   (already in the image - encoder models ride free via the
   `internal/taskclf` pattern with daulet/tokenizers Go BPE), or pure Go.
   No PyTorch in the image.
2. RAM: fits 4 GB alongside the agent. Generative: 2-3B at Q4 max (~1.6 GB
   resident). Encoders: 100-500 MB, can co-reside.
3. Speed: measure tok/s (or ms/inference for encoders) at 2 CPU threads on
   amd64 - Modal is the honest testbed (`scripts/modal_rehearse_image.py`
   shows the sandbox pattern; `finetune/minicpm5/modal_validate_assist_lora.py`
   shows CPU serving validation).
4. A third llama-server lane costs a spawn (10-60s contended) - prefer
   candidates that REPLACE an existing lane or ride the ONNX runtime.
5. Image <= 10 GB compressed (current ~4 GB; headroom exists).

## Research question 1: specialised models per category

Survey current best small models per weak category; verify constraints;
rank by expected gain / integration cost. Non-exhaustive starting points -
verify what is actually current and best rather than trusting this list:

- **NER: GLiNER family** (encoder-based span extraction, ONNX-exportable,
  ~150-500 MB). Likely the highest-value candidate: deterministic spans,
  no generation, CPU-fast, and the rubric's zero-missing-entities demand
  suits recall-tuned extraction. Integration path: ONNX alongside the task
  classifier; the 2B then only formats LABEL: span lines.
- **Sentiment: encoder classifiers** (DeBERTa-v3 / twitter-RoBERTa class).
  Caveat: the rubric needs a REASON sentence, so the pattern is encoder
  label + 2B-written reason. Assess whether that beats the current 7-8/8.
- **Code: Qwen2.5-Coder-1.5B/3B (or current equivalent)** for
  code_generation edge cases and especially code-FIX quality (cd_w02
  class). GGUF availability and b9948 support need checking. Could REPLACE
  the assist lane for the two code families via routing.
- **Maths: Qwen2.5-Math-1.5B class** - but our tool lane already scores
  11/11 by writing python; the bar is high. Probably not worth a lane.
- **Summarisation**: no obvious small specialist beats an instruction-tuned
  2B; the leverage is data (question 2) or constrained decoding. Survey
  anyway (e.g. small models fine-tuned for controllable summarisation).
- **Factual**: survey abstention/verification approaches that fit the
  budget (a tiny verifier that flags low-confidence proper nouns so the
  answer hedges rather than invents). A learned verifier (0.5-1B judge)
  scoring answer plausibility could also backstop several families.
- **Logic**: nothing small and specialised is likely to beat the
  tool-encode approach; deprioritise.

For the top 1-2 candidates: PROTOTYPE. `cmd/localprobe` probes a bare GGUF
against eval sets (`-assist -model X.gguf -tasks testdata/variant_tasks_golden.json`);
for ONNX encoders, a standalone Go or python probe against the NER/sentiment
slices of the variant/wildcard/holdout sets is enough evidence. Report
per-category deltas vs the current lane numbers above.

## Research question 2: datasets to improve the EXISTING models

Per weak category, shortlist public, licence-clean corpora (or parametric
generation recipes) that target the residual failure class. Check
contamination risk is nil (none of these will contain the hidden tasks, but
document the reasoning). Starting points to verify:

- summarisation: compound-constraint supervision barely exists publicly -
  the parametric generator `scripts/gen_assist_v2_data.py` is the base; a
  retrain was REJECTED once (read
  `docs/handoffs/assist-lora-v2-results.md` for why: coverage shifted,
  sentiment regressed, other families diluted). Its stated corrections:
  multi-theme compact units, broader targets, preserved gradient shares.
- code_debugging: bug-fix pair corpora (CodeXGLUE bug2fix, QuixBugs,
  mutation-generated pairs over MBPP-style functions) with the repo's
  execution-verification pattern (run buggy vs fixed on synthesised
  inputs).
- sentiment: sarcasm/irony sets (SemEval-2018 irony, DynaSent round 2) as
  counterweights - the v2c lesson says do not erase sarcasm competence
  when rebalancing.
- NER: CoNLL-2003 / OntoNotes / WNUT-17 reformatted to the serving
  LABEL: span contract for format-stability SFT.
- maths: GSM8K/SVAMP TRAIN splits rendered into the tool-call contract
  (model emits python) for phrasing diversity.
- logic: the repo's own `scripts/build_logic_grpo_pool.py` generates
  verified puzzles; survey LogiQA/proofwriter-class sets for diversity.
- factual: answer-or-abstain style tuning data (how to say 'commonly known
  as X' rather than inventing a proper noun) - survey what exists.

## What has been tried (do not repeat)

- v2c assist retrain (su + sentiment data): NO-GO, memo at
  `docs/handoffs/assist-lora-v2-results.md`.
- Sentiment DPO on the 1B: moved the holdout by exactly zero, twice.
- GRPO clue-encoding on the 1B: bf16 gains, served-quant regression -
  rejected. (TRL 0.24 GRPOTrainer also has an unresolved import conflict
  on the transformers-5.13 image.)
- Stock bigger base without fine-tune (Qwen2.5-3B): 28/43 variants - not a
  shortcut.
- MTP speculative decoding at b9948: slower on CPU, non-exact - no-go.

## Assets

- Eval-only sets (never train on them): `testdata/variant_tasks_golden.json`
  (43), `testdata/wildcard_tasks.json` (18), `testdata/su_holdout_tasks.json`,
  `testdata/sentiment_holdout_tasks.json`, `testdata/clue_holdout_tasks.json`,
  `testdata/golden.json` (40 practice).
- Scoring: `cmd/scorefile` (GOLDEN_PATH/RESULTS_PATH envs) - now aligned to
  the published rubrics; it matched the organisers' judge exactly (18/19 =
  0.94736844) after alignment.
- Full-agent eval: `cmd/realeval` (TASKS_PATH env + model-path envs; build
  with `CGO_ENABLED=1 CGO_LDFLAGS=-L$PWD/lib`).
- amd64 truth: Modal (`scripts/modal_rehearse_image.py` runs a published
  image; `finetune/minicpm5/modal_*.py` cover training/export/serve
  validation). macOS Metal decode DIVERGES from amd64 on borderline tasks -
  amd64 numbers are the honest ones.
- llama-server b9948 at `~/opt/llama`. HF repo
  `ashaibani/yassai-minicpm5-local` (private, near its storage cap - move
  artefacts via `modal volume get` when possible).
- ONNX integration reference: `internal/taskclf` (onnxruntime + Go BPE
  tokeniser) - the pattern any encoder candidate would follow.

## Definition of done

A memo at `docs/handoffs/specialised-models-research-results.md` with:
1. A ranked table: candidate (model or dataset), target category, expected
   gain (vs the numbers in this brief), integration cost, constraint
   check results (arch/licence/RAM/speed), and deadline-feasible yes/no.
2. Probe numbers for the top one or two candidates on the eval-only sets
   (served format, amd64 where possible).
3. A dataset shortlist with licences and contamination notes.
4. A clear GO/NO-GO recommendation per item.
Do NOT change CI, the Dockerfile, live submission settings, or any
production artefact - integration is the main session's decision.
