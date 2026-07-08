package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/ashaibani/yassai/internal/llm"
	"github.com/ashaibani/yassai/internal/markdown"
)

// Event is emitted during SolveWithEvents for real-time observability.
type Event struct {
	Type      string    `json:"type"` // "classify_start", "classify_done", "batch_plan", "batch_start", "call", "tool", "batch_end", "result", "done", "error"
	Timestamp time.Time `json:"timestamp"`
	Data      any       `json:"data"`
}

// EventCallback is called for each event during SolveWithEvents.
type EventCallback func(Event)

// SolveWithEvents runs the full solve pipeline, calling cb for each step.
func (a *Agent) SolveWithEvents(ctx context.Context, tasks []Task, cb EventCallback) ([]Result, Metrics, error) {
	a.metrics = Metrics{Model: a.model, StartedAt: time.Now()}
	answers := make(map[string]string, len(tasks))

	if len(tasks) > 0 && a.llm == nil {
		return nil, a.metrics, fmt.Errorf("FIREWORKS_API_KEY is required")
	}

	// Step 1: Classify
	cb(Event{Type: "classify_start", Timestamp: time.Now(), Data: map[string]any{"task_count": len(tasks)}})
	if a.cfg.Categories != nil {
		a.categories = a.cfg.Categories
	} else {
		a.categories = a.classifyTasks(tasks)
	}
	if a.categories != nil {
		cb(Event{Type: "classify_done", Timestamp: time.Now(), Data: a.categories})
	} else {
		cb(Event{Type: "classify_skip", Timestamp: time.Now(), Data: map[string]any{"reason": "classifier unavailable"}})
	}

	// Step 2: Plan batches
	batches := a.planBatches(tasks)
	batchPlan := make([]map[string]any, len(batches))
	for i, b := range batches {
		batchPlan[i] = map[string]any{
			"index":    i,
			"task_ids": taskIDs(b),
			"size":     len(b),
			"effort":   a.effortForBatch(b),
			"model":    a.modelForTask(b[0]),
		}
	}
	cb(Event{Type: "batch_plan", Timestamp: time.Now(), Data: map[string]any{"batches": batchPlan}})

	// Step 3: Solve batches (optionally concurrent)
	maxConcurrent := a.cfg.MaxConcurrency
	if maxConcurrent <= 0 {
		maxConcurrent = 1
	}
	if maxConcurrent > len(batches) {
		maxConcurrent = len(batches)
	}
	type batchResult struct {
		index   int
		answers map[string]string
		run     BatchRun
	}
	sem := make(chan struct{}, maxConcurrent)
	resultsCh := make(chan batchResult, len(batches))
	var wg sync.WaitGroup
	for i, batch := range batches {
		wg.Add(1)
		go func(idx int, b []Task) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			cb(Event{Type: "batch_start", Timestamp: time.Now(), Data: map[string]any{
				"index":    idx,
				"task_ids": taskIDs(b),
				"size":     len(b),
				"effort":   a.effortForBatch(b),
				"model":    a.modelForTask(b[0]),
			}})
			batchAnswers, run := a.solveBatchWithRecoveryEvents(ctx, b, cb)
			cb(Event{Type: "batch_end", Timestamp: time.Now(), Data: map[string]any{
				"index":    idx,
				"task_ids": taskIDs(b),
				"calls":    run.Calls,
				"tools":    run.Tools,
				"error":    run.Error,
			}})
			resultsCh <- batchResult{index: idx, answers: batchAnswers, run: run}
		}(i, batch)
	}
	wg.Wait()
	close(resultsCh)
	orderedResults := make([]batchResult, len(batches))
	for br := range resultsCh {
		orderedResults[br.index] = br
	}
	for _, br := range orderedResults {
		a.metrics.BatchSummaries = append(a.metrics.BatchSummaries, br.run)
		for id, answer := range br.answers {
			answers[id] = answer
		}
	}

	// Step 4: Assemble results
	results := make([]Result, 0, len(tasks))
	for _, task := range tasks {
		answer := strings.TrimSpace(answers[task.TaskID])
		if answer == "" {
			answer = "I do not have enough information to answer confidently."
			a.metrics.Fallbacks++
		}
		results = append(results, Result{TaskID: task.TaskID, Answer: answer})
		cb(Event{Type: "result", Timestamp: time.Now(), Data: map[string]any{
			"task_id":    task.TaskID,
			"answer":     answer,
			"categories": a.categories[task.TaskID],
		}})
	}
	a.metrics.FinishedAt = time.Now()
	a.metrics.DurationMS = a.metrics.FinishedAt.Sub(a.metrics.StartedAt).Milliseconds()
	cb(Event{Type: "done", Timestamp: time.Now(), Data: a.metrics})
	return results, a.metrics, nil
}

// solveBatchWithRecoveryEvents mirrors solveBatchWithRecovery but emits events.
func (a *Agent) solveBatchWithRecoveryEvents(ctx context.Context, batch []Task, cb EventCallback) (map[string]string, BatchRun) {
	run := BatchRun{TaskIDs: taskIDs(batch)}
	answers, calls, tools, err := a.solveBatchEvents(ctx, batch, cb)
	run.Calls = calls
	run.Tools = tools
	if err == nil {
		return answers, run
	}
	run.Error = err.Error()
	if len(batch) == 1 {
		a.metrics.Fallbacks++
		return map[string]string{batch[0].TaskID: fallbackAnswer(batch[0], err)}, run
	}
	mid := len(batch) / 2
	left, leftRun := a.solveBatchWithRecoveryEvents(ctx, batch[:mid], cb)
	right, rightRun := a.solveBatchWithRecoveryEvents(ctx, batch[mid:], cb)
	merged := map[string]string{}
	for k, v := range left {
		merged[k] = v
	}
	for k, v := range right {
		merged[k] = v
	}
	run.Calls += leftRun.Calls + rightRun.Calls
	run.Tools += leftRun.Tools + rightRun.Tools
	return merged, run
}

// solveBatchEvents mirrors solveBatch but emits per-call and per-tool events.
func (a *Agent) solveBatchEvents(ctx context.Context, batch []Task, cb EventCallback) (map[string]string, int, int, error) {
	a.metrics.BatchCount++
	a.exec.ResetSubmit()
	sessionID := "batch-" + strings.Join(taskIDs(batch), "-")

	// Select model for this batch (may differ from default if routing is configured)
	batchModel := a.model
	if len(batch) > 0 {
		batchModel = a.modelForTask(batch[0])
	}
	llmClient := a.llm
	if batchModel != a.model && a.cfg.APIKey != "" {
		llmClient = llm.New(llm.Config{APIKey: a.cfg.APIKey, BaseURL: a.cfg.BaseURL, Model: batchModel, Timeout: a.cfg.Timeout})
	}
	sys := systemPrompt()
	if g := a.categoryGuidance(batch); g != "" {
		sys += "\n\nTechnique hints for these task types:\n" + g
	}
	taskJSON := a.ctx.BuildBatchPrompt(batch, a.memory.LoadForTasks(batch), a.skills.LoadForPrompt(a.cfg.MemoryRoot, taskText(batch), 8000), a.categories)
	messages := []llm.Message{
		{Role: "system", Content: sys},
		{Role: "user", Content: taskJSON},
	}
	var calls int
	var tools int
	var lastText string
	effort := a.effortForBatch(batch)

	for turn := 0; ; turn++ {
		maxTokens := maxTokensForBatch(batch, turn, effort)
		callStart := time.Now()
		text, usage, err := llmClient.Chat(ctx, messages, maxTokens, effort)
		callDur := time.Since(callStart)
		if err != nil {
			a.recordCall(batch, turn, callStart, callDur, usage, "", err)
			cb(Event{Type: "call", Timestamp: time.Now(), Data: map[string]any{
				"turn":       turn,
				"task_ids":   taskIDs(batch),
				"effort":     effort,
				"error":      err.Error(),
				"latency_ms": callDur.Milliseconds(),
			}})
			return nil, calls, tools, err
		}
		calls++
		a.mu.Lock()
		a.metrics.Calls++
		a.metrics.PromptTokens += usage.PromptTokens
		a.metrics.OutputTokens += usage.CompletionTokens
		a.metrics.TotalTokens += usage.TotalTokens
		a.metrics.CachedTokens += usage.CachedTokens
		a.metrics.ReasoningTokens += usage.ReasoningTokens
		a.mu.Unlock()
		lastText = strings.TrimSpace(text)
		a.recordCall(batch, turn, callStart, callDur, usage, lastText, nil)

		cb(Event{Type: "call", Timestamp: time.Now(), Data: map[string]any{
			"turn":              turn,
			"task_ids":          taskIDs(batch),
			"batch_size":        len(batch),
			"effort":            effort,
			"prompt_tokens":     usage.PromptTokens,
			"completion_tokens": usage.CompletionTokens,
			"total_tokens":      usage.TotalTokens,
			"cached_tokens":     usage.CachedTokens,
			"reasoning_tokens":  usage.ReasoningTokens,
			"latency_ms":        callDur.Milliseconds(),
			"output_preview":    truncateForEvent(lastText, 500),
			"output_full":       lastText,
		}})

		// Check if the model called submit() during code execution.
		if a.exec.IsSubmitted() {
			result := a.exec.SubmitResult()
			cb(Event{Type: "submit", Timestamp: time.Now(), Data: map[string]any{
				"turn":     turn,
				"task_ids": taskIDs(batch),
				"answers":  result,
			}})
			return result, calls, tools, nil
		}

		// If the model hit max_tokens without producing output, the batch is
		// too large for the token budget. Return an error so
		// solveBatchWithRecovery splits it in half.
		if lastText == "" && usage.CompletionTokens >= maxTokens*9/10 {
			return nil, calls, tools, fmt.Errorf("batch truncated at max_tokens=%d (completion=%d), needs split", maxTokens, usage.CompletionTokens)
		}

		blocks := markdown.ExtractActionBlocks(lastText)
		if len(blocks) == 0 {
			// No code blocks. Try to parse as JSON (backward compat).
			if answers, ok := parseAnswers(lastText, batch); ok {
				cb(Event{Type: "submit", Timestamp: time.Now(), Data: map[string]any{
					"turn":     turn,
					"task_ids": taskIDs(batch),
					"answers":  answers,
					"via":      "json_fallback",
				}})
				return answers, calls, tools, nil
			}
			// Ask the model to use the action space.
			messages = append(messages,
				llm.Message{Role: "assistant", Content: compactAssistantEcho(lastText)},
				llm.Message{Role: "user", Content: "Use the action space. Emit a micropy code block that computes any needed answers and calls submit(answers=[...]). Do not return raw JSON."},
			)
			continue
		}

		observations := make([]map[string]any, 0, len(blocks))
		for i, block := range blocks {
			res, err := a.exec.Run(ctx, sessionID, block.Code, map[string]any{"tasks": batch})
			tools++
			a.mu.Lock()
			a.metrics.ToolRuns++
			a.mu.Unlock()
			obs := map[string]any{"index": i, "stdout": res.Stdout, "json": res.JSON, "value": res.Value}
			if err != nil {
				obs["error"] = err.Error()
			}
			observations = append(observations, obs)
			cb(Event{Type: "tool", Timestamp: time.Now(), Data: map[string]any{
				"turn":       turn,
				"task_ids":   taskIDs(batch),
				"tool_index": i,
				"code":       block.Code,
				"stdout":     res.Stdout,
				"json":       res.JSON,
				"error":      errString(err),
			}})
		}

		// Check if submit() was called during this round of code execution.
		if a.exec.IsSubmitted() {
			result := a.exec.SubmitResult()
			cb(Event{Type: "submit", Timestamp: time.Now(), Data: map[string]any{
				"turn":     turn,
				"task_ids": taskIDs(batch),
				"answers":  result,
			}})
			return result, calls, tools, nil
		}

		obsBytes, _ := json.Marshal(observations)
		messages = append(messages,
			llm.Message{Role: "assistant", Content: lastText},
			llm.Message{Role: "user", Content: "Observation: " + string(obsBytes) + "\nIf you have all answers, call submit(answers=[{...}]). Otherwise emit more code."},
		)
	}
}

func truncateForEvent(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max/2] + "\n...[truncated]...\n" + s[len(s)-max/2:]
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}
