package agent

import (
	"context"
	"time"
)

// Event is a streaming telemetry item for the local demo UI.
type Event struct {
	Type      string         `json:"type"`
	Timestamp time.Time      `json:"timestamp"`
	Data      map[string]any `json:"data,omitempty"`
}

// EventCallback receives progress events during SolveWithEvents.
type EventCallback func(Event)

// SolveWithEvents runs Solve and emits coarse start/batch/done events for the demo.
// The lean production path uses Solve directly; this exists for cmd/demo SSE.
func (a *Agent) SolveWithEvents(ctx context.Context, tasks []Task, cb EventCallback) ([]Result, Metrics, error) {
	emit := func(typ string, data map[string]any) {
		if cb == nil {
			return
		}
		cb(Event{Type: typ, Timestamp: time.Now(), Data: data})
	}
	emit("start", map[string]any{"task_count": len(tasks)})
	results, metrics, err := a.Solve(ctx, tasks)
	if err != nil {
		emit("error", map[string]any{"message": err.Error()})
		return results, metrics, err
	}
	emit("done", map[string]any{
		"results": len(results),
		"calls":   metrics.Calls,
		"tokens":  metrics.TotalTokens,
	})
	return results, metrics, nil
}
