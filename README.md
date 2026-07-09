# yassai

Yet Another Stupid Simple AI agent for the AMD Developer Hackathon Act 2, Track 1. It
reads a batch of tasks, answers them through Fireworks-hosted models, and writes
the results. Submissions are ranked by **fewest Fireworks tokens, subject to an
accuracy gate**, so the design goal is: be correct first, then spend as few
tokens as possible getting there.

**Current result:** 19/19 on the real Track-1 eval set at **4,826 Fireworks
tokens** (2 model calls, 0 fallbacks, ~10s), graded by a local judge matched to
the organisers' reference judge (`internal/judge`, default model `glm-5p2`).

## How it works

The whole run is one pass (`internal/agent`):

1. **Read** `/input/tasks.json`.
2. **Classify** each prompt into one of the 8 capability categories with a
   bundled in-process ONNX classifier (`assets/taskclf`; best-effort - falls back
   to a keyword heuristic if the ONNX runtime can't load). Local inference is
   zero Fireworks tokens.
3. **Plan two batches** by execution mode: a **direct** batch (factual, sentiment,
   summarisation, NER, code debugging, code generation - answered straight to
   JSON) and a **code** batch (maths + logic). Two batches means the fewest
   system-prompt copies, which dominates token cost at this scale.
4. **Solve** each batch in a single call at `reasoning_effort=none` - thinking
   tokens are the biggest cost lever, so code execution replaces reasoning where
   correctness needs it:
   - Direct batch: the model returns terse JSON answers for all its ids.
   - Code batch: the model writes **python3** (full stdlib), executed in a real
     subprocess (`internal/pyexec`), which calls a pre-defined `submit(answers=[...])`.
     Answer strings are **interpolated from the computed variables** (never
     hand-typed), so the executed result is authoritative - this is what makes
     maths/logic reliable at zero reasoning tokens, in one call.
5. **Write** `/output/results.json` and exit 0.

Resilience (a crash or malformed output means the submission does not qualify, so
every stage degrades instead of failing):

- The classifier is **best-effort**: if the model or ONNX runtime cannot load, or
  an inference errors, the agent logs one line to stderr and continues (tasks
  just get no category hint).
- An over-packed or failing batch **self-heals**: `solveBatchWithRecovery` halves
  it and retries.
- If the model returns raw JSON without calling `submit()`, backward-compatible
  parsing catches it as a fallback.
- The whole run is bounded by an internal 9m30s deadline, inside the harness's
  10-minute limit.

## Task classifier

Each prompt is tagged with its capability categories by a small ModernBERT
encoder exported to ONNX and run in-process on CPU (`internal/taskclf`,
`assets/taskclf/`). The predicted `categories` are added to each task object in
the batch prompt so the model knows the task type up front.

- **Backbone:** `ibm-granite/granite-embedding-small-english-r2` (Granite
  Embedding R2, ~47M params, ModernBERT architecture), fine-tuned as an 8-way
  multi-label head (sigmoid + per-label thresholds, uniform 0.5). Apache-2.0.
- **Tokeniser:** HF byte-level **BPE**, loaded by `daulet/tokenizers` (verified
  byte-identical to HuggingFace), with **head+tail** truncation to `max_len`
  (1024, head_frac 0.75) so a leading *or* trailing instruction always survives.
- **Footprint / speed:** int8 model **~48 MB**, a few milliseconds per task.
- **Accuracy:** held-out hard real-benchmark micro-F1 **0.977** at the deployed
  0.5 thresholds (the previous MiniLM/WordPiece v1 was 0.889).
- **Why it pays for itself:** adding the predicted categories to the prompt costs
  ~10 tokens per task of labels but nets **~12% fewer Fireworks tokens** overall,
  because the model reasons less when told the task type (agent A/B, `cmd/eval`).

The 8 categories, in logit order: `factual_knowledge`, `mathematical_reasoning`,
`sentiment_classification`, `text_summarisation`, `named_entity_recognition`,
`code_debugging`, `logical_deductive_reasoning`, `code_generation`. See
`assets/taskclf/README.md` for the full model card and inference contract.

> **Apple-Silicon note:** the classifier is verified correct on native amd64 (the
> judging VM). Running the amd64 image under QEMU emulation on an arm64 Mac makes
> onnxruntime's int8/SIMD kernels silently mis-compute, so the classifier appears
> to give wrong labels there only - it is an emulation artefact, not a code bug.
> Test ONNX changes natively (`ONNXRUNTIME_LIB=<arm64 dylib> go test ./internal/taskclf/`).

## Token efficiency

Because the ranking metric is total Fireworks tokens, the fixed per-call overhead
is minimised aggressively while keeping everything load-bearing for accuracy:

- **Terse system prompt** (~260 tokens): the action space, rules, and an example
  in one block. High signal, low noise.
- **Batching** amortises that fixed overhead over many tasks per call.
- **Category labels** in the prompt (net token win, above).
- **Category hints** (`categoryHints`) are injected into the system prompt **only
  for the categories present** in a batch, so they cost tokens only where they
  help. Currently: mathematical reasoning ("always compute with code"),
  logical/deductive puzzles ("solve programmatically"), and NER ("be exhaustive").
- **No paid reasoning tokens**: every category runs at `reasoning_effort=none`;
  maths and logic use executed Python instead.
- **Empty context is omitted**: the memory and skills blocks are dropped from the
  prompt entirely when they carry nothing (the default at runtime).
- **Text2img for genuinely long source passages** (`internal/textimg`): `auto`
  keeps task ids, instructions, format constraints, code, and arithmetic as text,
  but renders an individually labelled quoted passage once it reaches ~2,000
  characters (~500 text tokens). Live Fireworks measurements found 52% prompt
  savings at 2,000 characters and 62-69% for 5,000-40,000 characters. The real
  19-task set has no passage long enough, so `auto` correctly stays text-only and
  preserves the 19/19 accuracy floor. Aggressive `tasks`, `dense`, and `full`
  modes remain available for experiments.

## Action space (Python 3)

Maths and logic batches expose one native function tool: `run_python(code)`.
The host executes the code in a bounded Python 3 subprocess (`internal/pyexec`)
with the full standard library and a pre-defined `submit(answers=[...])`.
Direct batches do not expose a tool and return JSON immediately.

### submit(answers=[...])

The model calls the pre-defined function with every task id. It must interpolate
answers from computed variables rather than hand-type numbers or names; this
prevents a correct calculation from being copied incorrectly into the answer.

### Example

```python
product = 17 * 23
submit(answers=[
    {'task_id': 'm1', 'answer': str(product)},
])
```

## Model selection

`ALLOWED_MODELS` (injected by the harness) is the **single source of truth** for
permitted models; no model ID is ever hardcoded. The first entry is the default
(list order = priority). `AGENT_MODEL` can pin a specific choice but is honoured
only when it is in the allow-list; otherwise it is ignored. All inference goes
through `FIREWORKS_BASE_URL`.

## Reasoning effort

By default (`AGENT_REASONING_EFFORT=auto`) every deployed category resolves to
`none`. Maths and logic get deterministic work from `run_python`; the other
categories were already accurate without paid reasoning. Setting
`AGENT_REASONING_EFFORT` to `low`, `medium`, `high`, or `xhigh` remains available
for experiments but increases ranked token use.

## Memory and skills

- **Memory** is a file-backed markdown store (`internal/memory`). It starts as an
  empty clean-slate index and is injected into the prompt **only when it actually
  contains notes or selected `memories/*.md` docs**. None are bundled, and a
  single run writes none, so by default memory costs zero prompt tokens.
- **Skills** (`internal/skills`) is an optional loader for local skill
  directories. **No skills are bundled in the container**, so at runtime the
  skills block is empty and omitted.

## Configuration

All configuration is via environment variables (nothing is baked in):

| Variable | Default | Purpose |
|---|---|---|
| `FIREWORKS_API_KEY` | - | Injected by the harness. |
| `FIREWORKS_BASE_URL` | `https://api.fireworks.ai/inference/v1` | All model calls go through this. |
| `ALLOWED_MODELS` | - | Comma-separated allow-list; first = default. |
| `AGENT_MODEL` | - | Optional pin, used only if in `ALLOWED_MODELS`. |
| `AGENT_REASONING_EFFORT` | `auto` (all tiers currently `none`) | Override with `none`/`low`/`medium`/`high`/`xhigh`. |
| `AGENT_BATCH_SIZE` | `40` | Max tasks per batch. |
| `AGENT_BATCH_TOKENS` | `50000` | Max estimated prompt tokens per batch. |
| `AGENT_MAX_CONCURRENCY` | `1` | Max parallel batch solving. |
| `AGENT_CONTEXT_TOKENS` | `200000` | Assumed context window. |
| `AGENT_TEXTIMG` | `auto` | Compress quoted sources >=2,000 chars; also supports `off`, `tasks`, `dense`, and `full`. |
| `TASKCLF_DIR` | `assets/taskclf` | Classifier artefacts; set empty to disable. |
| `ONNXRUNTIME_LIB` | *(set in image)* | Path to `libonnxruntime`; if unset locally, the classifier is skipped. |
| `AGENT_MEMORY_ROOT` | `.` | Root for the memory store. |
| `AGENT_SKILL_ROOTS` | - | Comma-separated extra skill dirs. |
| `LLM_TIMEOUT_SECONDS` | `120` | Per-call timeout. |
| `TASKS_PATH` / `RESULTS_PATH` | `/input/tasks.json` / `/output/results.json` | I/O paths (for local testing). |

A `metrics.json` (tokens, latency, batch/tool counts, per-call records) is written
next to the results as a development aid.

## Local run

```bash
export FIREWORKS_API_KEY=...
export FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
export ALLOWED_MODELS=accounts/fireworks/models/minimax-m3
TASKS_PATH=testdata/practice.json RESULTS_PATH=/tmp/results.json go run ./cmd/agent
```

The classifier needs `libonnxruntime`; point `ONNXRUNTIME_LIB` at a local copy
(matching your native architecture) or leave it unset to run without the
classifier. On macOS (arm64):

```bash
# Download and install the macOS ONNX runtime dylib
ORT_VER=1.27.0
curl -fsSL -o /tmp/ort.tgz \
  "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-osx-arm64-${ORT_VER}.tgz"
tar xzf /tmp/ort.tgz -C /tmp
mkdir -p ~/lib/onnxruntime
cp /tmp/onnxruntime-osx-arm64-${ORT_VER}/lib/libonnxruntime.*.dylib ~/lib/onnxruntime/
ln -sf libonnxruntime.1.27.0.dylib ~/lib/onnxruntime/libonnxruntime.dylib

export ONNXRUNTIME_LIB=~/lib/onnxruntime/libonnxruntime.dylib
```

## Docker build

The judging VM is `linux/amd64`. The runtime image is `python:3.12-slim`, which
provides the interpreter used by the native `run_python` tool. The build is cgo
(`CGO_ENABLED=1`): it statically links `libtokenizers.a` and bundles
`libonnxruntime.so` plus the int8 `assets/taskclf/` artefacts. The result is far
under the 10 GB limit.

CI builds and publishes the image automatically (see [Images](#images)); to build
it by hand:

```bash
docker buildx build --platform linux/amd64 -t ghcr.io/<your-org>/yassai:latest --push .
```

The image must be public with an exact `registry/name:tag` reference - a private
image or a bad reference is the most common way to fail to qualify.

## Images

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) builds + tests on every
push to `main` and every pull request, then publishes to
`ghcr.io/<owner>/yassai`:

| Trigger | Tags |
|---|---|
| push to `main` | `latest`, `main`, `sha-<commit>` (immutable) |
| pull request | `pr-<number>`, `sha-<commit>` (immutable) |

`latest` tracks the newest main build; pull any historical build by its
`sha-<commit>`. **Make the GHCR package public once** (its Package settings ->
change visibility) or the judge cannot pull it.

## Evaluation

Three harnesses under `cmd/`:

- **`cmd/eval`** - compare batch sizes / strategies on a task set; writes
  per-strategy metrics, results, and a comparison JSON to `eval-results/`.
- **`cmd/realeval`** - run the agent on real benchmark tasks and score numeric
  answers exactly plus judged answers with an LLM judge.
- **`cmd/routeprobe`** - probe accuracy/tokens per (model, category) to build a
  routing preference map.

Fireworks-backed text2img tests are deliberately excluded from required CI so
provider variability cannot block an image build. Run them locally with
`TEXTIMG_LIVE_TESTS=1 FIREWORKS_API_KEY=... go test ./internal/textimg -v`, or
launch the manual **Text2img live experiments** workflow. Responses with
`content: null` are recorded as empty experimental results rather than panics.

### Test data (`testdata/`)

| file | what |
|---|---|
| `practice.json` | the guide's 8 practice tasks, with reference answers |
| `container-input.json` | the same 8 in the exact `/input/tasks.json` format |
| `real_tasks.json` | real benchmark tasks across the 8 categories |
| `tasks.comprehensive.json` / `tasks.batch-large.json` | 40 tasks, 5 per category |
| `tasks.sample.json` | 2 quick smoke tasks |
| `golden.json` | easy short-prompt golden set |

## Design notes

`docs/model-routing.md` is the working decision log, `docs/text2img.md` records
the image-tokenisation experiments, and `docs/action-space-and-finish.md`
documents native Python execution. `docs/phase1-design.md` preserves the
original system design and its superseded decisions.
