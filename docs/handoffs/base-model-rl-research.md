# Handoff: best base model + RL pipeline for the next assist/tool models

Self-contained brief - assume NO prior context, no memories, no other sessions.
Everything you need is in this file and the paths below.

## Mission context (read first)

- Project: `yassai`, an agent for the AMD Developer Hackathon Act 2, Track 1.
  Repo: `/Users/mohamed.ashaibani/Playground/AMDHackathon` (Go agent + Python
  training scripts). Currently **rank 1** on the live leaderboard.
- Scoring: submissions run in a judge container against 19 hidden tasks across
  8 families (factual, maths, sentiment, summarisation, NER, code debugging,
  logic, code generation). An LLM judge (glm-5p2 over Fireworks) gates
  accuracy (observed floor: 16/19); above the gate, rank = **ascending total
  Fireworks tokens**. **Local model inference inside the container counts as
  ZERO tokens** and a fully-local run (`ZERO_API_CALLS`) is explicitly valid.
  Participant guide (updated copy):
  https://drive.google.com/file/d/1UGpOZiGGGBqQhGQxX7g19QAA-Dq9hPKk/view
- Judge VM: **4 GB RAM, 2 vCPU, 10-minute cap, linux/amd64**. The guide says
  2B-3B 4-bit models fit; 7B does not leave room for agent code.
- HARD RULE: never train on the sample tasks, the evaluation tasks, or any
  captured live prompts. Style-mirroring with randomised content is the
  intended play ("genuinely capable, not hardcoded"). An overlap audit
  pattern exists - reuse it (exact + 8-gram shingle match of training prompts
  vs `testdata/*.json`); shipped datasets currently measure 0 exact overlaps.

## Current models and their measured ceilings (the reason for this task)

Two LoRA fine-tunes of `openbmb/MiniCPM5-1B` serve two in-container lanes via
llama-server (llama.cpp b9948, Q4_K_M GGUF):

| Model | Lane | Measured (43 adversarial variants, glm-5p2 judge) |
|---|---|---|
| `MiniCPM5-yassai-v2e3b` | tool lane: maths/logic via a trained `run_python` contract | maths 10/11, logic 6/6 (with gates) |
| `MiniCPM5-yassai-assist-v6` | direct answers: codegen, NER, summarisation, code-fix (+ sentiment/factual trained) | cg ~4/5, ner ~2/3, su ~2/3, sentiment 3-4/6, factual 2/3 |

Known 1B ceilings SFT could not fix (6 fine-tune iterations of evidence):
sentiment nuance (sarcasm/idioms - systematic bias, self-consistency cannot
catch consistent errors), codegen edge cases (empty-string handling), factual
knowledge on fresh topics, and explanation/cause-line quality that judges
grade. Ungated (no evidence gates) the 1B stack scores **12/19** - below the
accuracy floor; a **0-token submission needs a local model at >=17/19
ungated**. That is the prize this research serves.
Also measured: STOCK Qwen2.5-3B-Instruct Q4 probes only ~28/43 on our
variants - bigger-without-fine-tuning is not a shortcut. A DPO pass over v6
(judge-archive preference pairs) is in flight in another session - do not
touch it; your job is the NEXT generation.

## The task

1. **Base model selection.** Artificial Analysis claims MiniCPM5-1B is the
   best in our compute range (https://artificialanalysis.ai/models/open-source/tiny).
   Verify whether that claim holds FOR OUR WORKLOAD (their benchmarks are not
   our judge). Build a shortlist across the 1B-3B range and bake them off
   empirically. Candidates worth testing (check llama.cpp GGUF support,
   licence, and template quirks for each): MiniCPM5-1B (incumbent),
   Qwen2.5-1.5B/3B-Instruct, Qwen3-1.7B, Llama-3.2-1B/3B-Instruct,
   granite-3.x-2b-instruct, SmolLM2-1.7B-Instruct, Gemma-3-1b/4b-it,
   Phi-4-mini. Constraints that disqualify: no GGUF/llama.cpp support; Q4
   weights + KV that will not fit alongside a second model in 4 GB;
   **latency on 2 vCPU** (measure tok/s with `llama-bench -t 2` or the probe's
   GenMS column; the whole 19-task local pass must fit well inside ~7 min);
   restrictive licence for hackathon use.
2. **RL pipeline design.** Recommend and prototype the post-SFT stage:
   - DPO on judge archives already exists (`scripts/mine_dpo_pairs.py`,
     `finetune/minicpm5/train_dpo.py`, `modal_dpo.py`) - AND HAS BEEN RUN
     TWICE with a controlled negative result you must build on, not repeat:
     - **dpo-v7** (57 mined pairs): the probe "gain" was memorisation of the
       4-6 prompts per family the pairs covered; a fresh 8-task sentiment
       holdout (`testdata/sentiment_holdout_tasks.json`) showed v6 5/8 = v7
       5/8, identical failures. Also QUARANTINED (deleted from HF/volume):
       the miner originally pooled the golden sample set and 2 sample
       prompts leaked into training - the miner now excludes + asserts.
     - **dpo-v8** (36 ON-policy pairs: fresh authored prompts per failure
       mode, model's own temp-0.9 wrong answers as rejected, via
       `scripts/gen_sentiment_dpo_onpolicy.py`): holdout still 5/8. The
       model was 0/6-draws wrong on the failing patterns (understatement,
       double negation, balanced-mixed) - DPO reweights existing modes; it
       cannot create comprehension the policy lacks.
     - **CRITICAL EVAL TRAP - the Q4 requant noise floor**: v7 and v8
       (disjoint training data) produce BYTE-IDENTICAL temp-0 answers on
       several variant tasks, including two broken codegen answers that
       look like "DPO regressions" (cg 4/5 -> 2/5 in both). The
       adapter-merge -> re-save -> F16 -> Q4_K_M chain moves borderline
       behaviours to different quant lattice points (about +/-2 tasks per
       family, both directions); a small-LR DPO signal (reward margins
       ~0.015) is INVISIBLE under it. Any post-SFT RL you evaluate must
       either (a) be compared in bf16 merged form before quantisation, or
       (b) train with enough signal (higher lr/steps, verifiable rewards)
       to clear the noise floor on a NEVER-TRAINED holdout.
   - The bigger opportunity is **GRPO/RLVR with verifiable rewards**: this
     repo's evidence gates are literally programmatic reward functions -
     worked-example execution for codegen (`exampleCheck` in
     `internal/localllm/direct.go`), buggy-fails/fixed-passes tests for
     code-fix, bullet/sentence counts and per-bullet word caps for
     summarisation, entity-recall + label-shape checks for NER, grounding
     for maths. Design rewards from these + judge-in-the-loop sparingly
     (Fireworks concurrency cap: 3). Watch for reward hacking (format-reward
     gaming without content) - mix a small judge-scored component or KL
     anchor. TRL 0.20 is pinned in the Modal images; check GRPOTrainer
     availability there or pin what you need.
3. Deliverables:
   - A comparison table: per-candidate stock probe scores per family, tok/s
     at `-t 2`, RAM estimate, template/thinking quirks.
   - At least one full fine-tune of the best non-incumbent candidate through
     the existing pipeline, probed against the variants.
   - A written recommendation: base model + RL recipe + expected gain vs the
     current v6/DPO line, and whether it plausibly reaches >=17/19 ungated
     (the zero-token bar).

## The infrastructure you will reuse (all working, all versioned)

- **Probe harness** (the fitness function):
  `go run ./cmd/localprobe -assist -model <gguf> -tasks testdata/variant_tasks_golden.json`
  - spawns llama-server from `$YZMA_LIB` (`~/opt/llama`, b9948), judges every
  answer with the matched glm-5p2 judge. Needs `FIREWORKS_API_KEY` (source
  `~/config/.env` - never print values). Task sets: `downloads_tasks_golden.json`
  (19, sample-shaped smoke), `variant_tasks_golden.json` (43 adversarial -
  primary), `wildcard_tasks.json` (18 off-distribution robustness).
- **Training**: `uvx modal run finetune/minicpm5/modal_train.py --dataset assist --epochs 3 --tag <unique>`
  (H100, ~2 min). `BASE_MODEL` is currently hard-coded to `openbmb/MiniCPM5-1B`
  inside `modal_train.py` - parameterise it for the bake-off. Datasets build
  remotely from `scripts/build_minicpm5_assist_data.py` (+ committed teacher
  and Claude-authored caches in `finetune/minicpm5/data/`);
  `--dataset v2` builds the tool-lane data. **3 epochs, not 5** (5 overfits).
  ALWAYS pass a unique `--tag`: an untagged rerun once overwrote the previous
  checkpoint and destroyed the rollback path.
- **Behaviour gates** (must pass before export): `eval_assist_behavior.py` /
  `eval_tool_behavior.py` run automatically on a held-out split (different
  seed + reserved code functions + hash-split teacher rows).
- **Export**: `uvx modal run finetune/minicpm5/modal_export_gguf.py --run-name <run> --quant Q4_K_M --hf-repo ashaibani/yassai-minicpm5-local`
  - converter pinned to llama.cpp b9948 (MUST match the runtime; an unpinned
  converter once scrambled tokenizer metadata). NOTE: the converter/exporter
  and `train_trl.py`'s template assert are MiniCPM-specific in places - a new
  base model needs (a) its own chat-template render contract verified between
  HF training render and llama-server serving render, and (b) a conversion
  sanity probe (`finetune/minicpm5/modal_smoke_gguf.py` / `cmd/localmodeleval`
  for the tool contract).
- **Auth**: `source ~/config/.env` provides FIREWORKS_API_KEY, HF_TOKEN,
  GH_TOKEN; Modal is already authenticated. Keep Fireworks concurrency <= 3.

## Traps learned the hard way (do not relearn them)

- MiniCPM5 is a hybrid-THINKING model: serve/probe with llama-server
  `--reasoning off` or content comes back empty. Check every new candidate
  for thinking-mode behaviour.
- Serving system prompts live in `internal/localllm/direct.go`
  (`directInstructions`) and must stay byte-identical to the training data's
  system prompts (`SYSTEM` in `build_minicpm5_assist_data.py`).
- Fireworks' WAF 403s Python's default urllib User-Agent - set a custom UA.
- Authored training data must be balance-audited: a sarcasm-heavy sentiment
  set once taught the model to flip genuinely-positive reviews (live accuracy
  dropped below the gate). Counterweight positives are in the current data.
- Judge/model variance flips 1-2 borderline verdicts between identical runs -
  only deltas >= 2 tasks are signal.
- amd64 ONNX/ML under QEMU on Apple Silicon silently mis-computes - never
  benchmark emulated; use native arm64 locally or real amd64 in CI.

## Definition of done

A memo (markdown in `docs/handoffs/` or similar) with the comparison table,
the trained best-candidate probe results, the RL recommendation with a
prototype run's numbers, and a clear GO/NO-GO on replacing either lane's base
model - plus every artefact tagged and pushed to the HF repo so the main line
can adopt it with a one-line CI change.
