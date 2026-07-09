# Memory and context management

Updated: 09 July 2026

## Submission hot path

The validated submission sends a compact category-aware task bundle directly
from `internal/agent`. It does not inject memory or skills into Fireworks
requests. Hidden tasks are independent and the container runs once, so reading,
writing, or summarising memory would add tokens and turns without helping a
later request.

`internal/contextmgr`, `internal/memory`, and `internal/skills` remain available
for local experiments and earlier harness compatibility, but they are not used
by `solveBatch` in the production two-batch path.

## Prompt contents

Each request contains only:

1. a short JSON output contract;
2. recipes for categories present in that batch; and
3. the compact task array (`id`, optional short kind, and prompt).

Maths/logic batches additionally receive the `run_python` tool schema. Empty
blocks are not serialised.

The default batch token budget is 50,000 with a nominal 200,000-token context
window. Recovery halves a failing/over-packed batch rather than trimming the
middle of an individual prompt.

## Text2img context compression

`AGENT_TEXTIMG=auto` may move only a quoted source passage of at least 2,000
characters into a labelled PNG. The task id, category, instruction, and output
constraints stay in text. Code, maths, logic, and short prompts never leave the
text channel. See [`text2img.md`](text2img.md) for measurements and rejected
aggressive modes.

## Why the modules remain

The file-backed memory store and skill loader are useful outside the one-shot
hackathon workload, and removing them would erase working research code. They
are constructed best-effort and cost no Fireworks tokens unless a future caller
explicitly puts them back into request assembly.
