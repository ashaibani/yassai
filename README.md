# yassai

Yet Another Stupid Simple AI agent for the AMD Developer Hackathon Act 2, Track 1. It
reads a batch of tasks, answers them through Fireworks-hosted models, and writes
the results. Submissions are ranked by **fewest Fireworks tokens, subject to an
accuracy gate**, so the design goal is: be correct first, then spend as few
tokens as possible getting there.

**Current result:** 19/19 on the real Track-1 eval set at **~4,960 Fireworks
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
- **Adaptive reasoning effort** (below): high effort only where it changes the
  answer.
- **Empty context is omitted**: the memory and skills blocks are dropped from the
  prompt entirely when they carry nothing (the default at runtime).

## Action space (MicroPython)

Everything flows through the action space. The model emits fenced `micropy` or
`python` code blocks; the host extracts them, runs them in a MicroPython WASM
sandbox (wazero), and feeds observations back. The model calls `submit()` to
return final answers - there is no manual JSON parsing of the model's text
output as the primary path.

### submit(answers=[...])

The primary output mechanism. The model calls this with a list of
`{task_id, answer}` dicts to submit final answers for all tasks in the batch.
The host validates the submission and signals completion of the agent loop.

### Available tools (Python globals inside a code block)

- `submit(answers=[...])` - submit final answers
- `sh.run(command='...')` - run shell commands (python3 available in the image)
- `fs.read/write/edit/list` - file operations
- `web.fetch/search` - web access
- `vars` - persistent dict for storing intermediate results across code blocks
- `tools.help()` - list available tools

### Example

```micropy
# Compute answers for all tasks, then submit
r1 = sh.run(command='python3 -c "print(17*23)"')
vars['t1'] = r1['output'].strip()
vars['t2'] = 'Paris'
submit(answers=[
    {'task_id': 't1', 'answer': vars['t1']},
    {'task_id': 't2', 'answer': vars['t2']},
])
```

The runtime image ships a real shell and `python3`, so `sh.run` and
`python3 -c ...` work in the container.

## Model selection

`ALLOWED_MODELS` (injected by the harness) is the **single source of truth** for
permitted models; no model ID is ever hardcoded. The first entry is the default
(list order = priority). `AGENT_MODEL` can pin a specific choice but is honoured
only when it is in the allow-list; otherwise it is ignored. All inference goes
through `FIREWORKS_BASE_URL`.

## Reasoning effort

By default (`AGENT_REASONING_EFFORT` unset) the effort is **adaptive per category
tier**: only categories that empirically benefit from more reasoning are
elevated. On the real-task eval only logical/deductive puzzles improved with more
effort (they are routed to `xhigh`); every other category was already at ceiling
on `low`, so raising it only burned tokens. Batches are grouped by tier so one
`reasoning_effort` applies to the whole call, and `max_tokens` scales with the
tier. Setting `AGENT_REASONING_EFFORT` to a fixed value (`low`/`medium`/`high`/
`xhigh`) overrides the adaptivity.

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
| `AGENT_REASONING_EFFORT` | *(unset = adaptive)* | Fix effort to `low`/`medium`/`high`/`xhigh`. |
| `AGENT_BATCH_SIZE` | `20` | Max tasks per batch. |
| `AGENT_BATCH_TOKENS` | `12000` | Max estimated prompt tokens per batch. |
| `AGENT_MAX_CONCURRENCY` | `3` | Max parallel batch solving (1 = sequential). |
| `AGENT_CONTEXT_TOKENS` | `200000` | Assumed context window. |
| `TASKCLF_DIR` | `assets/taskclf` | Classifier artefacts; set empty to disable. |
| `ONNXRUNTIME_LIB` | *(set in image)* | Path to `libonnxruntime`; if unset locally, the classifier is skipped. |
| `AGENT_MEMORY_ROOT` | `.` | Root for the memory store. |
| `AGENT_SKILL_ROOTS` | - | Comma-separated extra skill dirs. |
| `LLM_TIMEOUT_SECONDS` | `180` | Per-call timeout. |
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

The judging VM is `linux/amd64`. The runtime image is `python:3.12-slim` (it
needs a real shell and `python3` for the `sh.run` action path; a distroless/static
image cannot run those). The build is cgo (`CGO_ENABLED=1`): it statically links
`libtokenizers.a` and bundles `libonnxruntime.so`, plus the `assets/taskclf/`
artefacts. The result is well under the 10 GB limit (~115 MB in the last build).

CI builds and publishes the image automatically (see [Images](#images)); to build
it by hand:

```bash
docker buildx build --platform linux/amd64 -t ghcr.io/<your-org>/yassai:latest --push .
```

The image must be public with an exact `registry/name:tag` reference - a private
image or a bad reference is the most common way to fail to qualify.

## Images

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) builds + tests on every
push to `main` and every pull request, then publishes to GHCR as
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

`docs/model-routing.md` is the working decision log. `docs/memory-and-context.md`
and `docs/phase1-design.md` cover the memory model and the original system design.
