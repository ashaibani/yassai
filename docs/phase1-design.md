# Phase 1 system design

Date: 07 July 2026

> **Status:** this is the original Phase-1 design record. The shipped agent has
> since evolved; for current behaviour see [`README.md`](../README.md) and
> [`model-routing.md`](model-routing.md). Points where the implementation diverged
> from this plan are flagged inline.

Major shipped divergences (09 July 2026):

| Phase-1 proposal | Shipped path |
|---|---|
| Fenced MicroPython/WASM action loop | Native `run_python` function tool backed by Python 3 |
| Tools for every task | Tool exposed only to maths and logic |
| Adaptive/high reasoning effort | `reasoning_effort=none` for every category |
| General context/memory assembly | Compact task JSON; memory and skills stay off the hot path |
| Fixed-size category batching | Two execution-mode batches (direct and code) |
| pxpipe as future research | Accuracy-gated text2img `auto` mode for >=2,000-character sources |

## Track 1 constraints

The Participant Guide requires a public `linux/amd64` Docker image that reads `/input/tasks.json`, writes valid `/output/results.json`, exits with code 0 on success, and completes inside 10 minutes. Track 1 is judged across factual knowledge, maths, sentiment, summarisation, named entity recognition, code debugging, logic, and code generation. All inference must use `FIREWORKS_BASE_URL`, `FIREWORKS_API_KEY`, and only models listed in `ALLOWED_MODELS`. Submissions first pass an accuracy gate, then rank by total recorded tokens.

## Architecture

1. Harness adapter
   - Loads tasks and environment variables.
   - Writes result JSON in the exact required schema.
   - Emits optional metrics for development only.

2. Model client
   - Uses the official OpenAI Go SDK.
   - Sends raw `chat/completions` requests so Fireworks-specific fields such as `top_k` can be passed without waiting for typed SDK support.
   - Selects the model from `ALLOWED_MODELS` (first entry = priority); no model ID is hardcoded. *(Superseded: the original plan preferred a specific model; the shipped `chooseModel` treats `ALLOWED_MODELS` as the single source of truth and honours `AGENT_MODEL` only when it is in the allow-list.)*

3. Context manager
   - Classifies each task into the eight Track 1 capability categories.
   - Builds a compact JSON task bundle rather than verbose conversational context.
   - Trims only extreme prompts, preserving head and tail to avoid losing summarisation or code details.

4. MicroPython WASM action space
   - Embeds `micropython-wasi.wasm` copied from the ScheduleAssurance `ai-sdk` implementation.
   - Runs under wazero with WebAssembly exception handling enabled.
   - Extracts model action blocks labelled `micropy`, executes them in a sandbox, and returns structured observations to the model.
   - Persists a small `vars` dictionary per batch session across action calls.
   - Installs tool surfaces into the MicroPython global namespace: `fs`, `sh`, `web`, `browser`, and `tools`.
   - Routes every tool request through the WASM host bridge. There is no native model function-calling path.

5. Action tools exposed inside MicroPython
   - Filesystem: `fs.read`, `fs.write`, `fs.edit`, and `fs.list`.
   - Terminal: `sh.run`, including `run`, `status`, `output`, `wait`, `cancel`, and `list` lifecycle operations for background work.
   - Web: `web.fetch` and `web.search` surfaces.
   - Browser-style control: `browser.drive` with navigation hooks in the lightweight server build.
   - Tool discovery: `tools.help` returns the installed action surface.

6. Memory
   - Provides a file-backed markdown memory store (`internal/memory`). *(Superseded: not the `/tmp` JSON store originally sketched.)*
   - Because hidden tasks are independent and the agent runs once, memory stays empty by default and is omitted from the prompt - see [`memory-and-context.md`](memory-and-context.md).

7. Recovery
   - Solves tasks in batches for token efficiency.
   - If a batch fails, it is split recursively to preserve accuracy.
   - Single-task failures receive explicit fallback answers rather than malformed output.

## Non-goals and ethics

The agent intentionally does not include hard-coded local solvers for benchmark categories. We should not reduce token usage through reward-hacky shortcuts such as deterministic sentiment or arithmetic classifiers. Token efficiency should come from architecture: compact context, batching, memory, and exact computation through the MicroPython action space when the model chooses to use it.

## Research links applied

- The Hitchhiker paper motivates an explicit perception, reasoning, action, and memory split.
- The code-as-harness survey motivates code blocks as the primary action interface rather than a large function-call catalogue.
- Grug-12B motivates compact, dense reasoning traces and short outputs. Phase 1 uses prompting only; fine-tuning is deferred until a measured baseline exists.
- pxpipe suggests token efficiency should be measured end-to-end, not only on transformed prompts. For Track 1 we therefore log prompt, output, and total usage when the provider returns it.

## Phase 1 baseline plan

1. Stabilise the Golang harness and Docker image.
2. Run smoke tests with Minimax M3 through Fireworks.
3. Benchmark hidden-style local fixtures for accuracy and token usage.
4. Tune batching, prompt size, and action-space reliability.
5. Only then start Phase 2 data generation and Grug-style fine-tuning.


## Memory and context (retained, off the submission hot path)

See [`memory-and-context.md`](memory-and-context.md) for the current design. In
brief:

- File-backed memory and skill-loading modules remain available for experiments,
  but `solveBatch` does not inject either into production requests.
- Category-specific help is the compiled-in `categoryRecipe` map and is emitted
  only for categories present in the current batch.
- The run-once agent does not maintain memory during a run.
