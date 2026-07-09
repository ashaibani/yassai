# Action space and terminal finish protocol

Updated: 09 July 2026

## Current design

yassai has two execution paths, selected by the local task classifier:

1. Direct tasks (factual, sentiment, summarisation, NER, code debugging, and
   code generation) receive no tools and return the answer JSON in one call.
2. Maths and logic tasks receive one native function tool,
   `run_python(code)`, backed by a bounded Python 3 subprocess.

The split is deliberate. Tool schemas and an observation turn cost tokens, so
they are exposed only where deterministic computation materially improves the
accuracy gate.

## Tool contract

The Python runtime pre-defines:

```python
submit(answers=[
    {"task_id": "T02", "answer": answer_text},
])
```

The generated program may import the full standard library. It must call
`submit` once with every id in the batch and must not redefine `submit`.
`internal/pyexec` validates ids, rejects empty/incomplete submissions, and
returns the submitted JSON to the agent.

## Computed-answer invariant

Every computed number and every selected logic value is interpolated from a
variable into the final answer. For example:

```python
remaining = start * (1 - sold_rate) + restock - sold_q3
submit(answers=[{
    "task_id": "T02",
    "answer": str(round(remaining)) + " units",
}])
```

This rule fixed the most damaging failure mode in the real task set: the model
could write correct Python but then hand-type a different number in `submit`.
The executed variables are now authoritative.

## Turn and recovery behaviour

- Code batches require a tool call on the first turn.
- After a successful execution, tools are disabled and the model may only
  format a final response if the submitted JSON was not already complete.
- Complete tool submissions terminate immediately, avoiding a formatting turn.
- Malformed or incomplete batches are split recursively; only missing tasks are
  retried.
- Direct JSON parsing remains a compatibility path, not a peer action channel.

## Why native function calling replaced fenced MicroPython

The original Phase-1 prototype used fenced `micropy` blocks in a WASM REPL.
Production moved to native function calling plus real Python 3 because it:

- has the full standard library (`itertools` is essential for logic puzzles);
- removes code-block extraction ambiguity;
- provides explicit tool-call ids and structured observations;
- makes `tool_choice=required` enforceable; and
- uses fewer turns on the validated 19-task workload.

The old MicroPython package remains in the repository as historical prototype
code, but it is not on the submission hot path.
