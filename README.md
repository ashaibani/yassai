# yassai

Yet Another Super Simple AI agent for the AMD Developer Hackathon Act 2,
Track 1. It reads a batch of tasks, answers as many as possible with
**fine-tuned local models running inside the container** (zero Fireworks
tokens), verifies every local answer with deterministic evidence gates, and
sends only what the gates reject to a single Fireworks call. Submissions are
ranked by **fewest Fireworks tokens, subject to an accuracy gate**, so the
design goal is: be correct first, then spend as few tokens as possible getting
there.

**Current standing: RANK 1** on the live Track-1 leaderboard - **1,559 tokens
at 0.895 accuracy (17/19)**, one API call, 10/19 tasks answered locally, zero
fallbacks; second place at time of writing: 1,797 tokens. Best observed live
run: 1,292 tokens at 0.947 (18/19). The current build (Qwen3.5-2B assist lane,
below) scores **19/19 on the sample-shaped local eval** with 12/19 local.

## How it works

The whole run is one pass (`internal/agent`), engineered to fit the judging
VM (4 GB RAM, 2 vCPU, 10-minute cap):

1. **Read** `/input/tasks.json`.
2. **Classify** each prompt into the 8 capability categories with a bundled
   fp32 ONNX classifier (`assets/taskclf`, Granite ModernBERT backbone). It
   runs ONCE up front, then its session is freed before any model server
   spawns - peak memory is the maximum of the phases, never their sum. A
   keyword heuristic covers classifier failure.
3. **Local-first lanes** (spawned lazily and sequentially via llama-server
   with q8_0 KV cache, flash attention, and quota-aware threads - at most one
   model resident at a time, so peak RAM is one lane, not the sum). A lane
   engages only when the classifier and the keyword heuristic AGREE on the
   category - disagreement is an uncertainty signal and those tasks go
   remote:
   - **Tool lane** (`MiniCPM5-yassai-v2e3b`, a MiniCPM5-1B fine-tune): maths
     and logic through a fine-tuned `run_python` tool contract - the model
     writes Python, the agent executes it, and the final answer must
     interpolate the computed values.
   - **Assist lane** (stock `Qwen3.5-2B` Q4 with our assist SFT applied as a
     **serve-time LoRA**): direct answers for code generation (only when the
     prompt carries an executable worked example), NER, sentiment (guarded by
     a self-consistency gate), capped summarisation, and correction-style
     code debugging; eval-style debugging is answered by executing the
     snippet and explaining with ground truth in hand.
4. **Evidence gates** hold every local answer to account: grounding (answer
   numbers must derive from executed output), physics and magnitude bounds,
   format compliance (bullet/sentence counts, per-bullet word caps), label
   sanity (acronym spans are not LOCATIONs, non-date spans are not DATEs),
   entity-recall checks, worked-example execution for generated code, and
   self-consistency across prompt phrasings. Bug-fix cause lines are
   **grounded in execution**: the agent runs the buggy function, and when it
   provably raises while the model's fix succeeds on the same input, the
   shipped cause line states the observed exception instead of the model's
   guess. Rejects get one gate-guided retry; still-rejected tasks go remote.
   **Gates are recall-biased: a wrongly rejected answer costs a few tokens, a
   wrongly accepted one costs accuracy.**
5. **One remote batch** (insurance): all gate rejects fold into a single
   Fireworks call at `reasoning_effort=none` with terse per-category recipes.
   Dedicated code batches get the `run_python` tool; mixed folded batches stay
   text-only single-turn (offered the tool, the model reaches for it on a coin
   flip and the second turn re-pays the whole context). Incomplete replies get
   one lean retry (no recipes). The deadline is anchored to **process start**
   so model loading counts against the 10-minute budget, and local solving
   always reserves the remote call's slot.
6. **Write** `/output/results.json`. If the remote path ever fails, the best
   gate-rejected local answer ships instead of an apology.

### Zero-token mode

`AGENT_NO_REMOTE=1` disables the Fireworks path entirely: gate rejects ship
their best local answer and the run scores `ZERO_API_CALLS` (explicitly valid
per the participant guide). It is **off by default**: the gates are
load-bearing (the 1B-era stack measured 12/19 ungated), and the flag waits
for a local stack that clears the accuracy gate on its own twice in a row.
The Qwen switch plus MTP speculative decoding (the model's own draft head,
supported by our pinned llama.cpp) is the active path towards it.

## The local models

Both lanes clear a held-out behaviour gate before export
(`eval_assist_behavior.py` / `eval_tool_behavior.py` - deterministic checks
on unseen prompts, including code functions never trained on), and training
never touches the sample or evaluation tasks: the guide forbids hardcoding,
and so do our builders (the DPO pair miner hard-asserts against sample ids
and prompt text).

| Lane | Model | Families served | Provenance |
|---|---|---|---|
| Tool | `MiniCPM5-yassai-v2e3b` (LoRA fine-tune of MiniCPM5-1B, merged Q4_K_M, converter pinned to the runtime's `b9948`) | maths, logic | Parameterised, execution-verified tool-call cases (`scripts/build_minicpm5_sft_data_v2.py`) - the observation is real stdout and the answer is interpolated from the executed namespace |
| Assist | stock `Qwen3.5-2B` Q4_K_M + `yassai-assist-e3-r32-q35-2b` LoRA at serve time (the pinned converter cannot merge-export Qwen3.5's MTP block, and speculative decoding needs those tensors anyway) | code gen, NER, sentiment, summarisation, code fix (factual trained but served remote) | Judge-filtered teacher distillation (`gen_assist_teacher_data.py`), Claude-authored adversarial examples (`gen_assist_claude_data.py` - sarcasm, idioms, word caps, execution-verified bug-fix pairs), and derived parametric NER/codegen (`build_minicpm5_assist_data.py`) |

Why Qwen3.5-2B for the assist lane: on the 43-variant adversarial probe it
scores **31/43 with the LoRA (v6-era 1B assist: 21/43)**, and on a fresh
8-task sentiment holdout - understatement, double negation, balanced-mixed -
it scores 7-8/8 where the 1B scored 5/8 at best. Those holdout patterns were
comprehension failures no amount of 1B fine-tuning or DPO moved (measured:
two controlled DPO passes, zero holdout transfer); model capacity was the
binding constraint. The stock+LoRA serve path is validated on native amd64
(`finetune/minicpm5/modal_validate_assist_lora.py`) because a Q4 that decodes
cleanly on macOS can still emit garbage on Linux CPU kernels.

## Reproducing the numbers

```bash
# local eval against the sample-shaped set (needs FIREWORKS_API_KEY for the judge)
USE_CLASSIFIER=1 TASKCLF_DIR=assets/taskclf ONNXRUNTIME_LIB=<libonnxruntime> \
LOCAL_MODEL_PATH=models/minicpm5/MiniCPM5-yassai-v2e3b.gguf \
LOCAL_BASE_MODEL_PATH=models/qwen35/Qwen_Qwen3.5-2B-Q4_K_M.gguf \
LOCAL_BASE_LORA_PATH=models/qwen35/yassai-assist-e3-r32-q35-2b-lora-f16.gguf \
LOCAL_BASE_EXTENDED=text_summarisation,code_fix,sentiment_classification \
YZMA_LIB=$HOME/opt/llama AGENT_REASONING_EFFORT=auto \
go run ./cmd/realeval

# capability probe of a bare GGUF (which families can it carry?)
go run ./cmd/localprobe -assist -model <model.gguf> [-lora <adapter.gguf>] \
  -tasks testdata/variant_tasks_golden.json
```

Three local task sets, in order of honesty:
`testdata/downloads_tasks_golden.json` (sample-shaped smoke),
`testdata/variant_tasks_golden.json` (43 adversarial unseen variants - the
fitness function), and `testdata/wildcard_tasks.json` (practice tasks plus
off-distribution phrasings that deliberately dodge the router - the
robustness audit). Every CI push to main also runs a **dress rehearsal**: the
freshly built image answers the full sample set inside `--memory 4g --cpus 2`
limits and reports to the telemetry worker - it predicted the live leaderboard
token count within 5 tokens.

## Shipping

CI builds `ghcr.io/ashaibani/yassai` (linux/amd64) with the tool GGUF, the
assist base GGUF, and the assist LoRA baked in from Hugging Face (versioned
artefacts - every training run has a unique tag), the fp32 classifier,
llama.cpp `b9948`, and a Python 3 runtime for the deterministic tools.
Submissions point at immutable `sha-*` tags. Runtime configuration is
env-only, per the guide: the harness injects `FIREWORKS_API_KEY`,
`FIREWORKS_BASE_URL`, and `ALLOWED_MODELS`; the image defaults everything
else.

## Telemetry

Every run (live, rehearsal, or dev) reports tasks, answers, metrics, batch
plans, and call traces to a Cloudflare Worker (source in `inspo/callback`).
That worker is how the lucky-timeout run, the tool-loop token spikes, and the
live-vs-rehearsal parity were all diagnosed.

## Docs

- `docs/model-routing.md` - category routing and batch planning
- `docs/action-space-and-finish.md` - the single code-execution surface
- `docs/memory-and-context.md` - context management
- `docs/text2img.md` - prompt-token render research (kept for reference)
- `docs/presentation.html` / `docs/presentation.pdf` - the submission deck
