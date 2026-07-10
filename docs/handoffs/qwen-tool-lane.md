# Handoff: train a Qwen3.5-2B TOOL-LANE model to replace MiniCPM5-v2e3b

Self-contained brief - assume NO prior context, memories, or other sessions.
Everything you need is in this file, the repo at
`/Users/mohamed.ashaibani/Playground/AMDHackathon`, and git history (all
assets referenced here are committed as of `6c2f777`).

## Mission context (read first)

- Project `yassai`: Go agent for the AMD Developer Hackathon Act 2, Track 1.
  Currently **rank 1 live** (~1,45x Fireworks tokens @ 17/19). Scoring: LLM
  judge accuracy gate (floor ~16/19, 0.842 held rank 3 today), then ascending
  Fireworks tokens; **local in-container inference counts zero** and
  `ZERO_API_CALLS` is explicitly valid.
- Judge VM: 4 GB RAM, 2 vCPU, 10-min cap, linux/amd64. Lanes are
  **sequential-resident** (one llama-server at a time - agent.go closes each
  before the next spawns), so a 2B lane peaks ~1.6 GB. Fits.
- Architecture: two local lanes + one remote insurance call.
  - **Tool lane** (`MiniCPM5-yassai-v2e3b`, 1B): maths + logic via a trained
    `run_python` contract - model emits ONE JSON tool call, the agent
    executes the Python, the final answer interpolates computed values.
  - **Assist lane** (stock `Qwen3.5-2B` Q4 + our SFT as a serve-time LoRA):
    direct answers for codegen/NER/sentiment/summaries/code fixes. This
    replacement already proved the pattern you are asked to repeat.
- HARD RULE: never train on the sample tasks (`testdata/downloads_tasks_golden.json`
  = the live prompts), the eval variants, wildcards, or any captured live
  prompt. Parametric style-mirroring with derived answers is the intended
  play. Every generator in this repo asserts against those files - keep it
  that way.

## Why replace the tool lane's base

1. **The measured gap**: the tool lane mis-encodes SEMANTIC clues into its
   enumeration code ("allergic to fur" -> wrong exclusion set). On the fresh
   8-task clue holdout (`testdata/clue_holdout_tasks.json`, eval-only,
   baseline measured this session) it scores **5/8** - the three fails are
   `assert len(sols)==1` crashes. This is the last systematic blocker to a
   zero-token submission (current zero-token distribution 15-17/19; bar is
   >=17 twice).
2. **Capacity precedent**: the assist lane's sentiment gap looked like a
   data/calibration problem for six fine-tunes and two DPO passes; it was
   model capacity - stock Qwen3.5-2B solved 8/8 holdout patterns the 1B
   could never produce. Logic reasoning may be the same story: stock
   Qwen3.5-2B answers logic variants 5/6 DIRECT (no tool), vs the 1B's 2/6.
3. **Hedge**: a GRPO clue-encoding run on the 1B is in flight in another
   session (Modal tag `tool-grpo-cluev1`, base `v2-e3-r32`). Your job is the
   2B alternative, not that run - use distinct tags (`q35-tool-*`) and touch
   nothing named `cluev1` or `v2-e3-r32`.

## The task

Train Qwen3.5-2B on the tool-lane contract and beat the 1B where it is weak
without losing where it is strong:

| Metric (harness below) | 1B v2e3b today | Ship bar for 2B |
|---|---|---|
| Variants maths (realeval, gated) | 10/11 | >=10/11 |
| Variants logic (realeval, gated) | 6/6 | 6/6 |
| Clue holdout (no-remote, tool lane only) | 5/8 | >=7/8 |
| Golden maths+logic (realeval) | 6/6 | 6/6 |
| Wall clock, 19-task rehearsal | 214s total build | comfortably < 600s |

## Assets you reuse (all committed, all working)

- **SFT data**: `scripts/build_minicpm5_sft_data_v2.py` - parametric,
  execution-verified tool-call cases (maths families + assignment/seating
  logic incl. allergy clues x3 boost + eval-code traces). Rows are plain
  ChatML messages with the tool call embedded in assistant content - model-
  family agnostic. The SYSTEM prompt in it must stay byte-identical to
  `internal/localllm/localllm.go` `systemPrompt` (serving contract).
- **Trainer**: `finetune/minicpm5/modal_train.py` with `--base-model`
  (already parameterised; transformers 5.13 handles `qwen3_5`;
  `train_trl.py` has the multi-base template assert + expanded LoRA targets
  for Qwen's hybrid Gated DeltaNet: `in_proj_{qkv,a,b,z}`, `out_proj` etc.).
  3 epochs, LoRA r32, unique `--tag` ALWAYS (an untagged rerun once
  destroyed a rollback path).
- **GRPO after SFT** (this is where the clue skill lives):
  `scripts/build_logic_grpo_pool.py` (56 semantic-dense puzzles with derived
  unique solutions), `finetune/minicpm5/rewards_rlvr.py::reward_logic_tool`
  (executes the emitted code, checks the printed assignment against truth;
  anti-hack: contradiction spam is capped, truths unreachable without
  solving), `finetune/minicpm5/train_grpo.py` (note: accumulation is tied to
  num_generations - TRL divisibility rule; MiniCPM needed
  `tok.model_input_names=["input_ids","attention_mask"]` - Qwen may not),
  `finetune/minicpm5/modal_grpo.py --lane tool`. Point `--base-run` at YOUR
  SFT checkpoint's volume dir.
- **Behaviour gate**: `finetune/minicpm5/eval_tool_behavior.py` (BASE_MODEL/
  ADAPTER/DATA envs; DATA = a fresh `build_minicpm5_sft_data_v2.py --seed
  20269999` split). Must pass before export. Maths non-regression is the
  law: a previous allergy-boosted retrain (toolv3) fixed clues and broke
  trains-class maths - it was reverted.
- **Export**: merged GGUF is BROKEN for Qwen3.5 on the pinned converter
  (b9948 omits the MTP block -> `missing tensor blk.24...`). Ship
  **stock Q4 + LoRA-f16 GGUF** via `finetune/minicpm5/modal_export_lora_gguf.py`,
  exactly like the assist lane. Stock Q4 already on HF:
  `ashaibani/yassai-minicpm5-local/Qwen_Qwen3.5-2B-Q4_K_M.gguf`.
- **Eval harnesses**: `cmd/realeval` (full agent; env-driven - see README
  "Reproducing the numbers"), `cmd/localmodeleval` (tool-contract
  smoke), `cmd/localprobe -lora`. Auth: `source ~/config/.env`
  (FIREWORKS_API_KEY etc. - never print); Fireworks concurrency <= 3.

## New wiring you must add (small, mirrors the assist lane)

`internal/localllm` `New()` (the TOOL-lane spawn) does not yet pass
`--lora`: copy the `LoraPath` handling from `NewDirect()` (stat-guard ->
append `--lora`), plumb `LocalModelLora` through agent config/env
(`LOCAL_MODEL_LORA_PATH`), Dockerfile fetch + env, and ci.yml build arg -
the assist lane's `LOCAL_BASE_LORA_URL` pattern is the template. Serving
flags stay as-is (`--reasoning off` etc. - Qwen3.5 is also a think-capable
ChatML model and the empty-think render matched our training template on the
assist run).

## Verification order (each gates the next)

1. SFT on `--base-model Qwen/Qwen3.5-2B --dataset v2 --tag q35-tool-v1`;
   behaviour gate green.
2. GGUF tokenisation round-trip: `finetune/minicpm5/modal_smoke_gguf.py` /
   `cmd/localmodeleval` - the 1B had EOG-flagged tool-call token ids that
   silently broke serving; verify Qwen's `<tool_call>`/`</tool_call>`
   tokenise and STOP correctly through llama.cpp b9948 with `--lora`.
3. amd64 decode validation (macOS-clean Q4s have decoded garbage on Linux
   CPU): adapt `finetune/minicpm5/modal_validate_assist_lora.py` to the tool
   system prompt + a maths tool-call canary.
4. Local realeval: golden (maths 4/4 logic 2/2), variants (maths >=10/11,
   logic 6/6), clue holdout (>=7/8, vs the committed 5/8 baseline).
5. GRPO (`--lane tool`) on top if SFT alone doesn't clear the clue holdout;
   re-gate, re-probe. Reward-graph sanity: mean reward should start ~0.4-0.6
   (the 1B baseline solves most plain clues) and climb; flat 0.1 means the
   tool-call parse is failing - check the completion format first.
6. Wall-clock: 2B tool generation is ~2-3x the 1B per token on 2 vCPU and
   tool calls are long (~200-400 tok) - measure the 19-task pass in the CI
   dress rehearsal (push to a branch or main; rehearsal runs on every main
   push) before any adoption talk.

## Traps (paid for once already - do not pay twice)

- Byte-identical prompt contracts: serving `systemPrompt` == training SYSTEM.
- Judge/eval variance is +/-1-2 tasks; only deltas >=2 are signal. The Q4
  requant/decode lottery shifts borderline tasks between macOS and amd64 -
  only in-container numbers are truthful for borderline cases.
- 3 epochs, not 5. Unique --tag. Never pool golden prompts into training
  (the DPO miner once smuggled two sample tasks - every builder now asserts;
  keep the asserts).
- Reap zombie llama-servers before probes (`pkill -f llama-server`, spare
  port 42343 - an unrelated embeddings server).
- Health-wait on llama-server must check for `"ok"` (it returns 503-JSON
  while loading and curl exits 0 on it).
- HF repo `ashaibani/yassai-minicpm5-local` has ~20 GB free; keep the two
  CI-referenced GGUFs (`MiniCPM5-yassai-v2e3b-Q4_K_M.gguf`,
  `Qwen_Qwen3.5-2B-Q4_K_M.gguf` + assist LoRA) untouched.

## Definition of done

A memo in `docs/handoffs/` with the comparison table above filled in, the
tagged Modal checkpoints + HF LoRA artefact, and a GO/NO-GO on replacing the
tool lane - including the wall-clock number and whether a SINGLE Qwen base
GGUF + two LoRAs (tool + assist) can serve both lanes sequentially (the mmap
page cache makes the second spawn of the same base near-instant; that layout
halves image weight payload). Do NOT change any CI default or touch the live
submission - adoption is a separate decision with the main session.
