# Memory and context management

Date: 07 July 2026 (updated 08 July 2026)

## Context budget

The context manager (`internal/contextmgr`) builds one compact JSON bundle per
batch: the tasks (each with its predicted `categories`), and - only when
non-empty - a memory block and a skills block. It assumes a 200k-token window and
reserves 24k tokens for model output and tool observations, allocating the
remaining budget across task prompts (trimming only extreme ones, head+tail).

Answer-shaping rules (output schema, terseness, the MicroPython escape hatch) live
in the **system prompt**, not in a per-call instruction list, so they are stated
once and not duplicated in every user message.

## Memory

Memory is a file-backed markdown store (`internal/memory`): a `MEMORY.md` index
plus selective `memories/*.md` files chosen by keyword overlap with the batch.

Two things make it cost **zero tokens by default**:

- It starts as an empty clean-slate index (`store.go` `DefaultIndex`), and nothing
  is bundled. The store regenerates that empty index on demand; the generated
  `MEMORY.md` / `memories/` are git-ignored and never shipped in the image.
- The context manager injects the memory block **only when it actually has
  selected docs**. An empty store is omitted from the prompt entirely.

The agent runs once per evaluation (read tasks, answer, exit), so it is **not**
instructed to maintain memory - writing `memories/*.md` mid-run would spend tokens
and turns for no benefit when there is no later run to read them. The store and
the MicroPython `fs.*` tools remain available for local/experimental use, but the
shipped agent does not use them for memory upkeep.

## Category-specific help

The real per-category guidance is **not** loaded from files - it is the
compiled-in `categoryHints` map in `internal/agent`. Hints are appended to the
system prompt **only for the categories present in a batch**, so they cost tokens
only where they help (currently logical/deductive reasoning and NER).

## Skills

`internal/skills` is an optional loader for local skill directories
(`.agents/skills` in the working dir, `~/.agents/skills`, or `AGENT_SKILL_ROOTS`),
where each skill is a directory with a `SKILL.md`. **No skills are bundled in the
container**, so at runtime the skills block is empty and omitted. The loader
exists for local experimentation; category help in the submission comes from
`categoryHints` above.
