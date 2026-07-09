# Action space and terminal finish protocol

## Research synthesis

Sources:
- *The Hitchhiker’s Guide to Agentic AI* (Roitman, arXiv:2606.24937) - ReAct loops, action-space design, structured outputs, MCP-style standardised tools.
- *Code as Agent Harness* (arXiv:2605.18747) - code as the executable/inspectable/stateful interface for reason-act-verify; verification-driven tool use; plan-execute-verify control.
- λ-RLM / RLM REPL (`inspo/lambda-RLM`) - single REPL action medium; terminal `FINAL(...)` / `FINAL_VAR(...)` extracted from trajectory; variables as buffers before finish.
- Industry practice (OpenAI Agents SDK, ReAct derivatives) - explicit tool schemas; optional `tool_choice=required`; separate terminal action; validate structured outputs host-side.

## Design principles for yassai

1. **One medium for all work**: fenced Python/micropy code blocks are the only action channel (code-as-harness).
2. **One terminal action**: `finish(answers=[...])` (alias `submit`) is the only way a batch completes. Host validates coverage + non-empty answers against expected `task_id`s.
3. **Buffers then commit**: optional `answers(task_id=..., answer=...)` merges into a buffer; `finish` commits the full set (RLM `FINAL_VAR` pattern).
4. **No dual-path conflict**: do not ask the model to either emit free JSON *or* call a tool as peer options. If the model still emits bare JSON, the **host promotes** it into a synthetic `finish(...)` so validation stays unified.
5. **Structured observations**: tool results return as Observation messages (ReAct observe step).
6. **Deterministic verification**: maths/puzzles via `sh.run(python3)` inside the same action space (verification-driven tool use).

## Protocol the model sees

```
optional prose thought
```python
# tools: sh.run, fs.*, web.*, vars, answers, finish/submit
finish(answers=[{'task_id':'T01','answer':'...'}, ...])
```
→ Observation → (more acts) → finish

## Host validation

- `SetExpectedTasks(ids)` before each batch.
- `finish`/`submit` rejects unknown ids, empty answers, and incomplete coverage.
- Incomplete finish keeps partials and recovery splits only missing tasks.
- Bare complete JSON is host-promoted to `finish` (still runs through the executor).

## Why not native function-calling only?

Fireworks models vary; fenced code works across models, composes tools in one turn, and matches the code-as-harness survey. Native tools can be layered later without changing the finish contract.
