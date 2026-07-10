# yassai

Yet Another Stupid Simple AI agent for the AMD Developer Hackathon Act 2,
Track 1. It reads a batch of tasks, answers as many as possible with
**fine-tuned local models running inside the container** (zero Fireworks
tokens), verifies every local answer with deterministic evidence gates, and
sends only what the gates reject to a single Fireworks call. Submissions are
ranked by **fewest Fireworks tokens, subject to an accuracy gate**, so the
design goal is: be correct first, then spend as few tokens as possible getting
there.

**Current result: RANK 1** on the live Track-1 leaderboard - **1,292 tokens at
0.947 accuracy (18/19)**, one API call, 13/19 tasks answered locally, zero
fallbacks. Second place at time of writing: 1,797 tokens.

## How it works

The whole run is one pass (`internal/agent`), engineered to fit the judging
VM (4 GB RAM, 2 vCPU, 10-minute cap):

1. **Read** `/input/tasks.json`.
2. **Classify** each prompt into the 8 capability categories with a bundled
   fp32 ONNX classifier (`assets/taskclf`, Granite ModernBERT backbone). It
   runs ONCE up front, then its session is freed before any model server
   spawns - peak memory is the maximum of the phases, never their sum. A
   keyword heuristic covers classifier failure.
3. **Local-first lanes** (both MiniCPM5-1B fine-tunes, spawned lazily and
   sequentially via llama-server with q8_0 KV cache, flash attention, and
   quota-aware threads). A lane engages only when the classifier and the
   keyword heuristic AGREE on the category - disagreement is an uncertainty
   signal and those tasks go remote:
   - **Tool lane** (`MiniCPM5-yassai-v2e3b`): maths and logic through a
     fine-tuned `run_python` tool contract - the model writes Python, the
     agent executes it, and the final answer must interpolate the computed
     values.
   - **Assist lane** (`MiniCPM5-yassai-assist-v6`): direct answers for
     code generation (only when the prompt carries an executable worked
     example), NER, capped summarisation, and correction-style code
     debugging; eval-style debugging is answered by executing the snippet
     and explaining with ground truth in hand.
4. **Evidence gates** hold every local answer to account: grounding (answer
   numbers must derive from executed output), physics and magnitude bounds,
   format compliance (bullet/sentence counts, per-bullet word caps), label
   sanity (acronym spans are not LOCATIONs, non-date spans are not DATEs),
   entity-recall checks, worked-example execution for generated code, and
   self-consistency across prompt phrasings. Rejects get one gate-guided
   retry; still-rejected tasks go remote. **Gates are recall-biased: a wrongly
   rejected answer costs a few tokens, a wrongly accepted one costs
   accuracy.**
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
per the participant guide). It is **off by default**: measured honestly, our
1B models pass 12/19 ungated - the gates are load-bearing. The flag exists for
the day a local model clears the accuracy gate on its own.

## The local models

Both are LoRA fine-tunes of `openbmb/MiniCPM5-1B`, trained on Modal (H100,
~2 minutes each), exported to Q4_K_M GGUF with a converter pinned to the same
llama.cpp build the image ships (`b9948`):

| Model | Families served | Training data |
|---|---|---|
| `v2e3b` (tool lane) | maths, logic | Parameterised, execution-verified tool-call cases (`scripts/build_minicpm5_sft_data_v2.py`) - the observation is real stdout and the answer is interpolated from the executed namespace |
| `assist-v6` | code gen, NER, summarisation, code fix (factual and sentiment trained but served remote) | Judge-filtered teacher distillation (`gen_assist_teacher_data.py`), Claude-authored adversarial examples (`gen_assist_claude_data.py` - sarcasm, idioms, word caps, execution-verified bug-fix pairs), and derived parametric NER/codegen (`build_minicpm5_assist_data.py`) |

Every training run must clear a held-out behaviour gate
(`eval_assist_behavior.py` / `eval_tool_behavior.py` - deterministic checks on
unseen prompts, including code functions never trained on) before export.
Training never touches the sample or evaluation tasks: the guide forbids
hardcoding, and so do our builders.

## Reproducing the numbers

```bash
# local eval against the sample-shaped set (needs FIREWORKS_API_KEY for the judge)
USE_CLASSIFIER=1 TASKCLF_DIR=assets/taskclf ONNXRUNTIME_LIB=<libonnxruntime> \
LOCAL_MODEL_PATH=models/minicpm5/MiniCPM5-yassai-v2e3b.gguf \
LOCAL_BASE_MODEL_PATH=models/minicpm5/MiniCPM5-yassai-assist-v6-Q4_K_M.gguf \
LOCAL_BASE_EXTENDED=text_summarisation,code_fix \
YZMA_LIB=$HOME/opt/llama AGENT_REASONING_EFFORT=auto \
go run ./cmd/realeval

# capability probe of a bare GGUF (which families can it carry?)
go run ./cmd/localprobe -assist -model <model.gguf> -tasks testdata/variant_tasks_golden.json
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

CI builds `ghcr.io/ashaibani/yassai` (linux/amd64) with both GGUFs baked in
from Hugging Face (versioned artefacts - every training run has a unique tag),
the fp32 classifier, llama.cpp `b9948`, and a Python 3 runtime for the
deterministic tools. Submissions point at immutable `sha-*` tags. Runtime
configuration is env-only, per the guide: the harness injects
`FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS`; the image
defaults everything else.

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
