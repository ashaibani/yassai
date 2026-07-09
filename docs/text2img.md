# Text2img prompt compression

Updated: 09 July 2026

## Goal

Fireworks vision inputs are charged from image geometry rather than normal BPE
tokenisation. Dense text rendered into a small monochrome image can therefore
cost fewer prompt tokens than sending the same long passage as text.

The implementation lives in `internal/textimg` (renderer) and
`internal/agent/textimg.go` (routing and multimodal message construction).

## Accuracy-first `auto` mode

`AGENT_TEXTIMG=auto` is the production default. It keeps the following as text:

- task ids and task-type labels;
- the question and exact output constraints;
- code and function names;
- arithmetic and logic prompts; and
- short quoted source snippets.

Only a quoted source passage of at least 2,000 characters (roughly 500 BPE
tokens) is rendered. Each image gets a separate wire-level text label such as
`IMAGE:T04 source text` immediately before the `image_url` part. This prevents
the model from associating one passage with the next task.

Maths and logic always remain text because one OCR digit/name error can fail an
otherwise correct programmatic solution.

## Fireworks measurements (MiniMax-M3, scale 1)

| Source size | Text prompt tokens | Image prompt tokens | Saving |
|---:|---:|---:|---:|
| 500 chars | 262 | 216 | 18% |
| 2,000 chars | 645 | 312 | 52% |
| 5,000 chars | 1,390 | 528 | 62% |
| 10,000 chars | 2,640 | 888 | 66% |
| 20,000 chars | 5,141 | 1,632 | 68% |
| 40,000 chars | 10,144 | 3,096 | 69% |

A 19,250-character context retained 2/2 answer accuracy while reducing prompt
tokens from 4,312 to 1,421 (67%). Scale 1 is the only token-positive setting for
short content; scale 2 and 3 cost more than text in the 595-character probe.

## Real 19-task decision

An early threshold compressed the two ~450-character summarisation passages.
That run saved only 13 prompt tokens and scored 18/19 because OCR changed a key
summary detail. The change was rejected. At the 2,000-character threshold none
of the real tasks activates text2img, so the submission retains its 19/19,
4,826-token result while supporting future long-context tasks automatically.

This is why the threshold is based on the measured 500-token regime, not merely
on whether an image request is technically cheaper.

## Experimental modes

| `AGENT_TEXTIMG` | Behaviour |
|---|---|
| `off` | Plain text only. |
| `auto` / `hybrid` | Long quoted passages only; production default. |
| `tasks` | Full tasks grouped into one image per task type. |
| `dense` | All direct tasks on one dense page. Useful for stress testing only. |
| `full` | Recipes plus grouped task sheets as images. Most aggressive. |

The aggressive modes are intentionally opt-in. Dense pages showed task-boundary
drift (for example, T01 receiving T01b's answer), while grouped pages improved
alignment but still lost exact formatting instructions.

## Testing

Required CI runs deterministic renderer, message-wiring, fallback, and threshold
tests without a network key. Live model tests require both variables:

```bash
TEXTIMG_LIVE_TESTS=1 FIREWORKS_API_KEY=... \
  go test ./internal/textimg -v -count=1
```

The manual `Text2img live experiments` GitHub Actions workflow runs the same
token, scale, OCR, and long-context probes. Provider responses with
`content: null` are handled as empty results and logged rather than panicking.
