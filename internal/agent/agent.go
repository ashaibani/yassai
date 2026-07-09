package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ashaibani/yassai/internal/contextmgr"
	"github.com/ashaibani/yassai/internal/llm"
	"github.com/ashaibani/yassai/internal/markdown"
	"github.com/ashaibani/yassai/internal/memory"
	"github.com/ashaibani/yassai/internal/micropy"
	"github.com/ashaibani/yassai/internal/skills"
	"github.com/ashaibani/yassai/internal/taskclf"
)

type Agent struct {
	cfg        Config
	model      string
	llm        *llm.Client
	ctx        contextmgr.Manager
	memory     *memory.Store
	skills     *skills.Loader
	clf        *taskclf.Classifier
	categories map[string][]string
	metrics    Metrics
	mu         sync.Mutex
}

func New(cfg Config) (*Agent, error) {
	if cfg.MaxBatchSize <= 0 {
		cfg.MaxBatchSize = 20
	}
	if cfg.MaxTurns <= 0 {
		cfg.MaxTurns = 4
	}
	if cfg.MaxConcurrency <= 0 {
		cfg.MaxConcurrency = 3
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 90 * time.Second
	}
	model := chooseModel(cfg.AllowedModels, cfg.PreferredModel)
	ag := &Agent{
		cfg:    cfg,
		model:  model,
		ctx:    contextmgr.Manager{MaxContextTokens: cfg.MaxContextTokens, ReserveTokens: 24000},
		memory: memory.New(cfg.MemoryRoot),
		skills: skills.NewLoader(cfg.SkillRoots),
	}
	if strings.TrimSpace(cfg.APIKey) != "" {
		ag.llm = llm.New(llm.Config{APIKey: cfg.APIKey, BaseURL: cfg.BaseURL, Model: model, Timeout: cfg.Timeout})
	}
	if dir := strings.TrimSpace(cfg.ClassifierDir); dir != "" {
		if clf, cerr := taskclf.New(dir, "", cfg.ClassifierLib); cerr != nil {
			fmt.Fprintln(os.Stderr, "task classifier disabled:", cerr)
		} else {
			ag.clf = clf
		}
	}
	return ag, nil
}

func (a *Agent) classifyTasks(tasks []Task) map[string][]string {
	if a.clf == nil {
		return nil
	}
	out := make(map[string][]string, len(tasks))
	loggedErr := false
	for _, t := range tasks {
		preds, err := a.clf.Classify(t.Prompt)
		if err != nil {
			if !loggedErr {
				fmt.Fprintln(os.Stderr, "task classifier inference failed:", err)
				loggedErr = true
			}
			continue
		}
		cats := make([]string, len(preds))
		for i, p := range preds {
			cats[i] = p.Label
		}
		out[t.TaskID] = cats
	}
	return out
}

// Solve answers every task under the parent context deadline. Batches run with
// bounded concurrency; each batch has its own MicroPython executor and a local
// deadline so one slow batch cannot starve the rest of the 10-minute budget.
func (a *Agent) Solve(ctx context.Context, tasks []Task) ([]Result, Metrics, error) {
	a.metrics = Metrics{Model: a.model, StartedAt: time.Now()}
	answers := make(map[string]string, len(tasks))

	if len(tasks) == 0 {
		a.metrics.FinishedAt = time.Now()
		return nil, a.metrics, nil
	}
	if a.llm == nil {
		return nil, a.metrics, fmt.Errorf("FIREWORKS_API_KEY is required")
	}

	if a.cfg.Categories != nil {
		a.categories = a.cfg.Categories
	} else {
		a.categories = a.classifyTasks(tasks)
	}

	batches := a.planBatches(tasks)
	a.metrics.Classifications = a.categories
	a.metrics.BatchPlan = a.batchPlanRecords(batches)

	maxConcurrent := a.cfg.MaxConcurrency
	if maxConcurrent <= 0 {
		maxConcurrent = 3
	}
	if maxConcurrent > len(batches) {
		maxConcurrent = len(batches)
	}

	type batchResult struct {
		index   int
		answers map[string]string
		run     BatchRun
	}

	// Shared deadline budget: leave a small tail so results can still be written.
	deadline, hasDeadline := ctx.Deadline()
	globalRemaining := func() time.Duration {
		if !hasDeadline {
			return 8 * time.Minute
		}
		left := time.Until(deadline) - 20*time.Second
		if left < 0 {
			return 0
		}
		return left
	}

	var (
		wg      sync.WaitGroup
		sem     = make(chan struct{}, maxConcurrent)
		results = make([]batchResult, len(batches))
		stopped atomic.Bool
	)

	for i, batch := range batches {
		if stopped.Load() || ctx.Err() != nil {
			// Mark remaining batches as timed out without launching more work.
			for j := i; j < len(batches); j++ {
				results[j] = batchResult{
					index:   j,
					answers: map[string]string{},
					run:     BatchRun{TaskIDs: taskIDs(batches[j]), Error: "skipped: parent context deadline"},
				}
			}
			break
		}

		wg.Add(1)
		go func(idx int, b []Task) {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				results[idx] = batchResult{
					index:   idx,
					answers: map[string]string{},
					run:     BatchRun{TaskIDs: taskIDs(b), Error: ctx.Err().Error()},
				}
				return
			}
			defer func() { <-sem }()

			// Per-batch budget: share remaining time fairly, but never take more
			// than 3 minutes or less than 45s when anything remains.
			remaining := globalRemaining()
			if remaining <= 0 {
				stopped.Store(true)
				results[idx] = batchResult{
					index:   idx,
					answers: map[string]string{},
					run:     BatchRun{TaskIDs: taskIDs(b), Error: "batch skipped: no time remaining"},
				}
				return
			}
			batchBudget := remaining
			if maxConcurrent > 1 {
				// Rough share for in-flight work; still allow a healthy single-batch budget.
				share := remaining
				if nLeft := len(batches) - idx; nLeft > 0 {
					share = remaining * time.Duration(maxConcurrent) / time.Duration(nLeft)
				}
				if share < batchBudget {
					batchBudget = share
				}
			}
			if batchBudget > 3*time.Minute {
				batchBudget = 3 * time.Minute
			}
			if batchBudget < 45*time.Second && remaining >= 45*time.Second {
				batchBudget = 45 * time.Second
			}

			batchCtx, cancel := context.WithTimeout(ctx, batchBudget)
			defer cancel()

			exec, err := micropy.New(micropy.Config{Timeout: 15 * time.Second, MemoryBytes: 16 * 1024 * 1024})
			if err != nil {
				results[idx] = batchResult{
					index:   idx,
					answers: map[string]string{},
					run:     BatchRun{TaskIDs: taskIDs(b), Error: err.Error()},
				}
				return
			}
			exec.SetExpectedTasks(taskIDs(b))

			batchAnswers, run := a.solveBatchWithRecovery(batchCtx, exec, b)
			results[idx] = batchResult{index: idx, answers: batchAnswers, run: run}
		}(i, batch)
	}
	wg.Wait()

	for _, br := range results {
		a.metrics.BatchSummaries = append(a.metrics.BatchSummaries, br.run)
		for id, answer := range br.answers {
			if strings.TrimSpace(answer) != "" {
				answers[id] = answer
			}
		}
	}

	out := make([]Result, 0, len(tasks))
	for _, task := range tasks {
		answer := strings.TrimSpace(answers[task.TaskID])
		if answer == "" {
			answer = "Unable to produce a confident answer within the time budget."
			a.metrics.Fallbacks++
		}
		out = append(out, Result{TaskID: task.TaskID, Answer: answer})
	}
	a.metrics.FinishedAt = time.Now()
	a.metrics.DurationMS = a.metrics.FinishedAt.Sub(a.metrics.StartedAt).Milliseconds()
	return out, a.metrics, nil
}

func (a *Agent) solveBatchWithRecovery(ctx context.Context, exec *micropy.Executor, batch []Task) (map[string]string, BatchRun) {
	run := BatchRun{TaskIDs: taskIDs(batch)}
	if ctx.Err() != nil {
		run.Error = ctx.Err().Error()
		return map[string]string{}, run
	}

	answers, calls, tools, err := a.solveBatch(ctx, exec, batch)
	run.Calls = calls
	run.Tools = tools
	if err == nil && completeAnswers(answers, batch) {
		return answers, run
	}
	if err != nil {
		run.Error = err.Error()
	} else {
		run.Error = fmt.Sprintf("incomplete batch answers (%d/%d)", len(answers), len(batch))
	}

	// Keep any partial answers already produced.
	merged := cloneAnswerMap(answers)

	if len(batch) == 1 {
		if strings.TrimSpace(merged[batch[0].TaskID]) == "" {
			a.mu.Lock()
			a.metrics.Fallbacks++
			a.mu.Unlock()
			merged[batch[0].TaskID] = fallbackAnswer(batch[0], err)
		}
		return merged, run
	}

	// Prefer splitting only the missing tasks so partial work is not discarded.
	missing := missingTasks(batch, merged)
	if len(missing) == 0 {
		return merged, run
	}
	if len(missing) == 1 || ctx.Err() != nil {
		// Single missing task or no time: solve missing only (or fall back).
		sub, subRun := a.solveBatchWithRecovery(ctx, exec, missing)
		for k, v := range sub {
			if strings.TrimSpace(v) != "" {
				merged[k] = v
			}
		}
		run.Calls += subRun.Calls
		run.Tools += subRun.Tools
		if subRun.Error != "" {
			run.Error = subRun.Error
		}
		return merged, run
	}

	mid := len(missing) / 2
	left, leftRun := a.solveBatchWithRecovery(ctx, exec, missing[:mid])
	right, rightRun := a.solveBatchWithRecovery(ctx, exec, missing[mid:])
	for k, v := range left {
		if strings.TrimSpace(v) != "" {
			merged[k] = v
		}
	}
	for k, v := range right {
		if strings.TrimSpace(v) != "" {
			merged[k] = v
		}
	}
	run.Calls += leftRun.Calls + rightRun.Calls
	run.Tools += leftRun.Tools + rightRun.Tools
	return merged, run
}

func (a *Agent) solveBatch(ctx context.Context, exec *micropy.Executor, batch []Task) (map[string]string, int, int, error) {
	if err := ctx.Err(); err != nil {
		return nil, 0, 0, err
	}
	a.mu.Lock()
	a.metrics.BatchCount++
	a.mu.Unlock()

	exec.ResetSubmit()
	exec.SetExpectedTasks(taskIDs(batch))
	sessionID := "batch-" + strings.Join(taskIDs(batch), "-")

	sys := systemPrompt(batch)
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
	var partial map[string]string
	effort := a.effortForBatch(batch)
	batchModel := a.model
	if len(batch) > 0 {
		batchModel = a.modelForTask(batch[0])
	}
	llmClient := a.llm
	if batchModel != a.model && a.cfg.APIKey != "" {
		llmClient = llm.New(llm.Config{APIKey: a.cfg.APIKey, BaseURL: a.cfg.BaseURL, Model: batchModel, Timeout: a.cfg.Timeout})
	}

	maxTurns := a.cfg.MaxTurns
	if maxTurns <= 0 {
		maxTurns = 4
	}

	for turn := 0; turn < maxTurns; turn++ {
		if err := ctx.Err(); err != nil {
			if completeAnswers(partial, batch) {
				return partial, calls, tools, nil
			}
			return partial, calls, tools, err
		}

		maxTokens := maxTokensForBatch(batch, turn, effort)
		callStart := time.Now()
		chat, err := llmClient.ChatDetailed(ctx, messages, maxTokens, effort)
		text, usage := chat.Content, chat.Usage
		callDur := time.Since(callStart)
		if err != nil {
			a.recordCall(batch, turn, callStart, callDur, usage, "", chat.ReasoningContent, batchModel, effort, maxTokens, messages, err)
			if completeAnswers(partial, batch) {
				return partial, calls, tools, nil
			}
			return partial, calls, tools, err
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
		a.recordCall(batch, turn, callStart, callDur, usage, lastText, chat.ReasoningContent, batchModel, effort, maxTokens, messages, nil)

		// Prefer code-action path first (submit / tools).
		blocks := markdown.ExtractActionBlocks(lastText)
		if len(blocks) > 0 {
			observations := make([]map[string]any, 0, len(blocks))
			toolTraces := make([]ToolTrace, 0, len(blocks))
			for i, block := range blocks {
				if err := ctx.Err(); err != nil {
					break
				}
				res, runErr := exec.Run(ctx, sessionID, block.Code, map[string]any{
					"tasks":    tasksAsMaps(batch),
					"task_ids": taskIDs(batch),
				})
				tools++
				a.mu.Lock()
				a.metrics.ToolRuns++
				a.mu.Unlock()
				obs := map[string]any{"index": i, "stdout": res.Stdout, "json": res.JSON, "value": res.Value}
				if runErr != nil {
					obs["error"] = runErr.Error()
				}
				observations = append(observations, obs)
				tt := ToolTrace{Index: i, Code: block.Code, Stdout: res.Stdout, JSON: res.JSON}
				if runErr != nil {
					tt.Error = runErr.Error()
				}
				toolTraces = append(toolTraces, tt)

				if exec.IsSubmitted() {
					submitted := exec.SubmitResult()
					partial = mergeAnswers(partial, submitted)
					a.appendToolTraces(batch, turn, toolTraces)
					if completeAnswers(submitted, batch) {
						return submitted, calls, tools, nil
					}
					// Incomplete submit: keep partials and continue / split later.
					return submitted, calls, tools, fmt.Errorf("submit incomplete (%d/%d)", len(submitted), len(batch))
				}
			}
			a.appendToolTraces(batch, turn, toolTraces)

			// No submit: try to salvage answers from assistant text / observations.
			if answers, ok := parseAnswers(lastText, batch); ok {
				partial = mergeAnswers(partial, answers)
				if completeAnswers(answers, batch) {
					return answers, calls, tools, nil
				}
			}

			if turn == maxTurns-1 {
				if completeAnswers(partial, batch) {
					return partial, calls, tools, nil
				}
				if len(batch) == 1 && lastText != "" {
					return map[string]string{batch[0].TaskID: lastText}, calls, tools, nil
				}
				return partial, calls, tools, fmt.Errorf("max turns reached without complete answers")
			}

			obsBytes, _ := json.Marshal(observations)
			messages = append(messages,
				llm.Message{Role: "assistant", Content: lastText},
				llm.Message{Role: "user", Content: "Observation: " + string(obsBytes) + "\n" + nextStepPrompt(batch)},
			)
			continue
		}

		// No code blocks: host-promote complete structured JSON into the action space
		// by synthesising a finish() call (keeps ONE terminal protocol).
		if answers, ok := parseAnswers(lastText, batch); ok {
			partial = mergeAnswers(partial, answers)
			if completeAnswers(answers, batch) {
				code := synthesiseFinishCode(answers, batch)
				res, runErr := exec.Run(ctx, sessionID, code, map[string]any{
					"tasks":    tasksAsMaps(batch),
					"task_ids": taskIDs(batch),
				})
				tools++
				a.mu.Lock()
				a.metrics.ToolRuns++
				a.mu.Unlock()
				tt := []ToolTrace{{Index: 0, Code: code, Stdout: res.Stdout, JSON: res.JSON}}
				if runErr != nil {
					tt[0].Error = runErr.Error()
				}
				a.appendToolTraces(batch, turn, tt)
				if exec.IsSubmitted() {
					submitted := exec.SubmitResult()
					if completeAnswers(submitted, batch) {
						return submitted, calls, tools, nil
					}
					return submitted, calls, tools, fmt.Errorf("promoted finish incomplete (%d/%d)", len(submitted), len(batch))
				}
				// Validation failed: feed error and continue.
				messages = append(messages,
					llm.Message{Role: "assistant", Content: compactAssistantEcho(lastText)},
					llm.Message{Role: "user", Content: "Host tried finish() from your JSON but validation failed: " + errString(runErr) + ". " + nextStepPrompt(batch)},
				)
				continue
			}
			if len(batch) > 1 {
				return answers, calls, tools, fmt.Errorf("partial answers (%d/%d)", len(answers), len(batch))
			}
		}

		if lastText == "" && usage.CompletionTokens >= maxTokens*9/10 {
			return partial, calls, tools, fmt.Errorf("batch truncated at max_tokens=%d (completion=%d), needs split", maxTokens, usage.CompletionTokens)
		}

		if turn == maxTurns-1 {
			if completeAnswers(partial, batch) {
				// Last-turn host promote of partials that became complete.
				return partial, calls, tools, nil
			}
			return partial, calls, tools, fmt.Errorf("max turns reached without finish()")
		}

		messages = append(messages,
			llm.Message{Role: "assistant", Content: compactAssistantEcho(lastText)},
			llm.Message{Role: "user", Content: nextStepPrompt(batch)},
		)
	}
	return partial, calls, tools, fmt.Errorf("max turns reached without complete answers")
}

func nextStepPrompt(batch []Task) string {
	ids := strings.Join(taskIDs(batch), ", ")
	return "Continue in the action space. Emit ONE python/micropy fenced code block. " +
		"When every task is solved, end that block with finish(answers=[...]) covering task_ids: " + ids + ". " +
		"You may buffer partials with answers(task_id=..., answer=...) then finish(...). " +
		"Do not reply with prose-only or raw JSON outside a code block."
}

func needsCode(batch []Task) bool {
	// Heuristic: multi-step numeric / code / puzzle wording benefits from tools.
	for _, t := range batch {
		p := strings.ToLower(t.Prompt)
		if strings.Contains(p, "python") || strings.Contains(p, "function") ||
			strings.Contains(p, "calculate") || strings.Contains(p, "how many") ||
			strings.Contains(p, "bug") || strings.Contains(p, "clues") ||
			strings.Contains(p, "write a") {
			return true
		}
	}
	return false
}

func tasksAsMaps(batch []Task) []map[string]string {
	out := make([]map[string]string, len(batch))
	for i, t := range batch {
		out[i] = map[string]string{"task_id": t.TaskID, "prompt": t.Prompt}
	}
	return out
}

func (a *Agent) recordCall(batch []Task, turn int, start time.Time, dur time.Duration, usage llm.Usage, output, reasoning, model, effort string, maxTokens int, messages []llm.Message, err error) {
	rec := CallRecord{
		Turn:             turn,
		Timestamp:        start,
		LatencyMS:        dur.Milliseconds(),
		PromptTokens:     usage.PromptTokens,
		CompletionTokens: usage.CompletionTokens,
		TotalTokens:      usage.TotalTokens,
		CachedTokens:     usage.CachedTokens,
		ReasoningTokens:  usage.ReasoningTokens,
		BatchSize:        len(batch),
		TaskIDs:          taskIDs(batch),
		OutputChars:      len(output),
		Model:            model,
		Effort:           effort,
		MaxTokens:        maxTokens,
		AssistantMessage: truncateTrace(output, 32000),
		ReasoningContent: truncateTrace(reasoning, 32000),
	}
	if err != nil {
		rec.Error = err.Error()
	}
	if a.cfg.TraceMessages {
		rec.Messages = cloneMessages(messages)
	}
	a.mu.Lock()
	a.metrics.CallRecords = append(a.metrics.CallRecords, rec)
	a.mu.Unlock()
}

func cloneMessages(messages []llm.Message) []llm.Message {
	out := make([]llm.Message, len(messages))
	copy(out, messages)
	return out
}

func (a *Agent) appendToolTraces(batch []Task, turn int, traces []ToolTrace) {
	if len(traces) == 0 {
		return
	}
	ids := taskIDs(batch)
	a.mu.Lock()
	defer a.mu.Unlock()
	for i := len(a.metrics.CallRecords) - 1; i >= 0; i-- {
		rec := &a.metrics.CallRecords[i]
		if rec.Turn == turn && stringSliceEqual(rec.TaskIDs, ids) {
			rec.ToolTraces = traces
			return
		}
	}
}

func stringSliceEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func taskText(tasks []Task) string {
	var b strings.Builder
	for _, t := range tasks {
		b.WriteString(t.TaskID)
		b.WriteByte(' ')
		b.WriteString(t.Prompt)
		b.WriteByte('\n')
	}
	return b.String()
}

func systemPrompt(batch []Task) string {
	ids := strings.Join(taskIDs(batch), ", ")
	return strings.Join([]string{
		"You are a precise multi-task agent. Everything you do goes through the CODE ACTION SPACE.",
		"",
		"PROTOCOL (ReAct-style, code-as-harness):",
		"1. Optional short reasoning in prose.",
		"2. Act by emitting exactly one fenced code block tagged python, python3, py, or micropy.",
		"3. The harness runs the block and returns an Observation.",
		"4. Repeat until you call the terminal action finish(...).",
		"",
		"TERMINAL ACTION (only way to complete the batch):",
		"  finish(answers=[",
		"    {'task_id': '<id>', 'answer': '<non-empty string>'},",
		"    ...",
		"  ])",
		"Aliases: submit(...) is identical to finish(...).",
		"Partial helper: answers(task_id=..., answer=...) buffers one task; still call finish when all ready.",
		"Batch task_ids (use ONLY these): " + ids,
		"",
		"TOOLS inside code blocks:",
		"  finish / submit / answers",
		"  sh.run(command='python3 -c \"...\"')  # compute and verify",
		"  fs.read, fs.write, fs.edit, fs.list",
		"  web.fetch, web.search",
		"  vars  # persists across code blocks in this batch",
		"  tools.help()",
		"",
		"RULES:",
		"- Never finish with prose-only or bare JSON. Always end in a code block that calls finish(...).",
		"- Maths, projections, constraint puzzles: compute via sh.run python3; do not guess.",
		"- Keep answers short; honour format constraints (bullet counts, labels, etc.).",
		"- Cover every task_id in one finish call.",
		"",
		"EXAMPLES:",
		"```python",
		"finish(answers=[{'task_id': 'T01', 'answer': 'Red, green, blue (additive light model).'}])",
		"```",
		"```python",
		"r = sh.run(command='python3 -c \"print(2400 - int(2400*0.37) + 800 - 640)\"')",
		"finish(answers=[{'task_id': 'T02', 'answer': r['output'].strip()}])",
		"```",
	}, "\n")
}

var categoryHints = map[string]string{
	"logical_deductive_reasoning": "Logical puzzles: enumerate assignments in python3 (itertools) and filter by every constraint, then submit the unique solution.",
	"mathematical_reasoning":      "Math: compute with python3 via sh.run; print exact numbers; then submit.",
	"named_entity_recognition":    "NER: list every PERSON/ORGANIZATION/LOCATION/DATE in the text; be exhaustive; no invented entities.",
	"code_debugging":              "Code debug: state the bug briefly, then provide a complete corrected function.",
	"code_generation":             "Code gen: provide a complete function with docstring matching the examples/edge cases.",
}

func (a *Agent) categoryGuidance(batch []Task) string {
	if a.cfg.DisableHints {
		return ""
	}
	present := map[string]bool{}
	for _, t := range batch {
		for _, c := range a.categories[t.TaskID] {
			present[c] = true
		}
	}
	var b strings.Builder
	for _, c := range []string{
		"mathematical_reasoning",
		"logical_deductive_reasoning",
		"named_entity_recognition",
		"code_debugging",
		"code_generation",
	} {
		if present[c] {
			if b.Len() > 0 {
				b.WriteByte('\n')
			}
			b.WriteString(categoryHints[c])
		}
	}
	return b.String()
}

func chooseModel(allowed []string, preferred string) string {
	if len(allowed) == 0 {
		return preferred
	}
	if preferred != "" {
		for _, m := range allowed {
			if m == preferred {
				return preferred
			}
		}
	}
	for _, m := range allowed {
		if m == "accounts/fireworks/models/minimax-m3" {
			return m
		}
	}
	return allowed[0]
}

func (a *Agent) batchPlanRecords(batches [][]Task) []BatchPlanRecord {
	out := make([]BatchPlanRecord, len(batches))
	for i, b := range batches {
		model := ""
		if len(b) > 0 {
			model = a.modelForTask(b[0])
		}
		out[i] = BatchPlanRecord{Index: i, TaskIDs: taskIDs(b), Size: len(b), Effort: a.effortForBatch(b), Model: model}
	}
	return out
}

func (a *Agent) planBatches(tasks []Task) [][]Task {
	maxTok := a.cfg.MaxBatchTokens
	if maxTok <= 0 {
		maxTok = 8000
	}
	maxN := a.cfg.MaxBatchSize
	if maxN <= 0 {
		maxN = 20
	}

	byGroup := map[string][]Task{}
	var groups []string
	for _, t := range tasks {
		key := a.effortForTask(t) + "|" + a.modelForTask(t)
		if _, ok := byGroup[key]; !ok {
			groups = append(groups, key)
		}
		byGroup[key] = append(byGroup[key], t)
	}
	var batches [][]Task
	for _, key := range groups {
		var cur []Task
		curTok := 0
		for _, t := range byGroup[key] {
			tk := estTokens(t.Prompt) + 16
			if len(cur) > 0 && (len(cur) >= maxN || curTok+tk > maxTok) {
				batches = append(batches, cur)
				cur, curTok = nil, 0
			}
			cur = append(cur, t)
			curTok += tk
		}
		if len(cur) > 0 {
			batches = append(batches, cur)
		}
	}
	return batches
}

func estTokens(s string) int { return len(s)/4 + 1 }

var effortTier = map[string]string{
	"logical_deductive_reasoning": "high",
	"code_debugging":              "medium",
	"code_generation":             "medium",
	"mathematical_reasoning":      "medium",
}

var tierRank = map[string]int{"low": 0, "medium": 1, "high": 2, "xhigh": 3}

func (a *Agent) effortForTask(t Task) string {
	if a.cfg.ReasoningEffort != "" {
		return a.cfg.ReasoningEffort
	}
	tiers := a.cfg.EffortTierMap
	if tiers == nil {
		tiers = effortTier
	}
	best := "low"
	for _, c := range a.categories[t.TaskID] {
		if e, ok := tiers[c]; ok && tierRank[e] > tierRank[best] {
			best = e
		}
	}
	return best
}

func (a *Agent) modelForTask(t Task) string {
	if len(a.cfg.ModelRouteMap) == 0 {
		return a.model
	}
	for _, c := range a.categories[t.TaskID] {
		if m, ok := a.cfg.ModelRouteMap[c]; ok && m != "" {
			return m
		}
	}
	return a.model
}

func (a *Agent) effortForBatch(batch []Task) string {
	if len(batch) == 0 {
		return a.cfg.ReasoningEffort
	}
	return a.effortForTask(batch[0])
}

func maxTokensForBatch(batch []Task, turn int, effort string) int {
	perTask, ceil := 1000, 24000
	switch effort {
	case "medium":
		perTask, ceil = 1400, 32000
	case "high":
		perTask, ceil = 2000, 40000
	case "xhigh":
		perTask, ceil = 2800, 48000
	}
	base := 800 + len(batch)*perTask
	if turn > 0 {
		base = base * 5 / 4
	}
	if base < 2048 {
		base = 2048
	}
	if base > ceil {
		base = ceil
	}
	return base
}

func parseAnswers(text string, batch []Task) (map[string]string, bool) {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil, false
	}
	candidates := []string{text}
	if j := extractJSON(text); j != "" && j != text {
		candidates = append([]string{j}, candidates...)
	}
	for _, block := range markdown.ExtractBlocks(text) {
		info := strings.ToLower(block.Info)
		if info == "" || strings.Contains(info, "json") {
			candidates = append(candidates, block.Code)
		}
	}
	ids := map[string]bool{}
	for _, t := range batch {
		ids[t.TaskID] = true
	}
	var best map[string]string
	for _, candidate := range candidates {
		if answers, ok := parseAnswerJSON(candidate, ids); ok {
			if len(answers) == len(ids) {
				return answers, true
			}
			if len(answers) > len(best) {
				best = answers
			}
		}
	}
	if len(best) > 0 {
		return best, true
	}
	if len(batch) == 1 && !strings.Contains(text, "```") {
		return map[string]string{batch[0].TaskID: strings.Trim(text, "` \n\t")}, true
	}
	return nil, false
}

func parseAnswerJSON(s string, ids map[string]bool) (map[string]string, bool) {
	var wrapped struct {
		Answers []struct {
			TaskID string `json:"task_id"`
			Answer any    `json:"answer"`
		} `json:"answers"`
	}
	if err := json.Unmarshal([]byte(s), &wrapped); err == nil && len(wrapped.Answers) > 0 {
		out := map[string]string{}
		for _, item := range wrapped.Answers {
			if !ids[item.TaskID] {
				continue
			}
			ans := stringifyAnswer(item.Answer)
			if ans == "" {
				continue
			}
			out[item.TaskID] = ans
		}
		if len(out) > 0 {
			return out, true
		}
	}

	var arr []struct {
		TaskID string `json:"task_id"`
		Answer any    `json:"answer"`
	}
	if err := json.Unmarshal([]byte(s), &arr); err == nil && len(arr) > 0 {
		out := map[string]string{}
		for _, item := range arr {
			if !ids[item.TaskID] {
				continue
			}
			ans := stringifyAnswer(item.Answer)
			if ans == "" {
				continue
			}
			out[item.TaskID] = ans
		}
		if len(out) > 0 {
			return out, true
		}
	}

	var obj map[string]any
	if err := json.Unmarshal([]byte(s), &obj); err == nil && len(obj) > 0 {
		out := map[string]string{}
		for id := range ids {
			if v, ok := obj[id]; ok {
				ans := stringifyAnswer(v)
				if ans != "" {
					out[id] = ans
				}
			}
		}
		if len(out) > 0 {
			return out, true
		}
	}
	return nil, false
}

func stringifyAnswer(v any) string {
	switch x := v.(type) {
	case string:
		return strings.TrimSpace(x)
	case nil:
		return ""
	default:
		b, _ := json.Marshal(x)
		return string(b)
	}
}

func extractJSON(s string) string {
	startObj := strings.Index(s, "{")
	startArr := strings.Index(s, "[")
	start := -1
	if startObj >= 0 && startArr >= 0 {
		if startObj < startArr {
			start = startObj
		} else {
			start = startArr
		}
	} else if startObj >= 0 {
		start = startObj
	} else {
		start = startArr
	}
	if start < 0 {
		return ""
	}
	endObj := strings.LastIndex(s, "}")
	endArr := strings.LastIndex(s, "]")
	end := endObj
	if endArr > end {
		end = endArr
	}
	if end <= start {
		return ""
	}
	return strings.TrimSpace(s[start : end+1])
}

func truncateTrace(s string, max int) string {
	s = strings.TrimSpace(s)
	if max <= 0 || len(s) <= max {
		return s
	}
	return s[:max/2] + "\n...\n" + s[len(s)-max/2:]
}

func completeAnswers(answers map[string]string, batch []Task) bool {
	if answers == nil {
		return false
	}
	for _, t := range batch {
		if strings.TrimSpace(answers[t.TaskID]) == "" {
			return false
		}
	}
	return true
}

func missingTasks(batch []Task, answers map[string]string) []Task {
	var out []Task
	for _, t := range batch {
		if strings.TrimSpace(answers[t.TaskID]) == "" {
			out = append(out, t)
		}
	}
	return out
}

func mergeAnswers(dst, src map[string]string) map[string]string {
	if dst == nil {
		dst = map[string]string{}
	}
	for k, v := range src {
		if strings.TrimSpace(v) != "" {
			dst[k] = v
		}
	}
	return dst
}

func cloneAnswerMap(in map[string]string) map[string]string {
	out := map[string]string{}
	for k, v := range in {
		out[k] = v
	}
	return out
}

func synthesiseFinishCode(answers map[string]string, batch []Task) string {
	var b strings.Builder
	b.WriteString("finish(answers=[\n")
	for _, t := range batch {
		ans := answers[t.TaskID]
		esc := strings.ReplaceAll(ans, "\\", "\\\\")
		esc = strings.ReplaceAll(esc, "'", "\\'")
		esc = strings.ReplaceAll(esc, "\n", "\\n")
		fmt.Fprintf(&b, "  {'task_id': '%s', 'answer': '%s'},\n", t.TaskID, esc)
	}
	b.WriteString("])\n")
	return b.String()
}

func errString(err error) string {
	if err == nil {
		return "unknown error"
	}
	return err.Error()
}

func compactAssistantEcho(s string) string {
	s = strings.TrimSpace(s)
	if len(s) > 1200 {
		return s[:600] + "\n...\n" + s[len(s)-600:]
	}
	return s
}

func taskIDs(tasks []Task) []string {
	ids := make([]string, len(tasks))
	for i, task := range tasks {
		ids[i] = task.TaskID
	}
	return ids
}

func fallbackAnswer(task Task, err error) string {
	if err != nil {
		return "Unable to answer confidently: " + err.Error()
	}
	return "Unable to answer confidently for task " + task.TaskID
}
