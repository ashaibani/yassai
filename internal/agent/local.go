package agent

import "context"

// trySolveLocal is intentionally a no-op.
// Prompt-specific canned solvers are reward-hacking risk on a judged track.
// Compute accuracy comes from the LLM (+ optional MicroPython action blocks).
func trySolveLocal(ctx context.Context, t Task, category string) (string, bool) {
	_ = ctx
	_ = t
	_ = category
	return "", false
}
