package agent

import "context"

// trySolveLocal intentionally does not answer benchmark prompts directly.
// Local execution is exposed through the generic run_python tool loop instead;
// that keeps the production path prompt-driven rather than cached-answer driven.
func trySolveLocal(ctx context.Context, t Task, category string) (string, bool) {
	_ = ctx
	_ = t
	_ = category
	return "", false
}
