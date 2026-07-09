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
		cfg.MaxBatchSize = 40
	}
	if cfg.MaxTurns <= 0 {
		cfg.MaxTurns = 1 // single-pass default for token efficiency
	}
	if cfg.MaxConcurrency <= 0 {
		cfg.MaxConcurrency = 1
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 120 * time.Second
	}
	// Force low effort unless caller overrides - reasoning tokens dominate cost.
	if cfg.ReasoningEffort == "" {
		cfg.ReasoningEffort = "low"
	}
	model := chooseModel(cfg.AllowedModels, cfg.PreferredModel)
	ag := &Agent{
		cfg:    cfg,
		model:  model,
		ctx:    contextmgr.Manager{MaxContextTokens: cfg.MaxContextTokens, ReserveTokens: 8000},
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
	out := make(map[string][]string, len(tasks))
	loggedErr := false
	for _, t := range tasks {
		if a.clf != nil {
			preds, err := a.clf.Classify(t.Prompt)
			if err != nil {
				if !loggedErr {
					fmt.Fprintln(os.Stderr, "task classifier inference failed:", err)
					loggedErr = true
				}
			} else if len(preds) > 0 {
				cats := make([]string, len(preds))
				for i, p := range preds {
					cats[i] = p.Label
				}
				out[t.TaskID] = cats
				continue
			}
		}
		// Local/heuristic fallback when ONNX is unavailable or empty.
		out[t.TaskID] = []string{heuristicCategory(t.Prompt)}
	}
	return out
}

// heuristicCategory is a cheap, general keyword router aligned to the 8 benchmark
// capability labels. It is not task-id specific and works on prompt variants.
func heuristicCategory(prompt string) string {
	p := strings.ToLower(prompt)
	switch {
	case strings.Contains(p, "sentiment") || strings.Contains(p, "positive, negative, or neutral") || strings.Contains(p, "positive/negative"):
		return "sentiment_classification"
	case strings.Contains(p, "summar") || strings.Contains(p, "bullet point") || strings.Contains(p, "in exactly"):
		return "text_summarisation"
	case strings.Contains(p, "named entit") || strings.Contains(p, "person, organization") || strings.Contains(p, "label each as person"):
		return "named_entity_recognition"
	case strings.Contains(p, "bug") || strings.Contains(p, "contains a bug") || strings.Contains(p, "corrected function") || strings.Contains(p, "identify the bug"):
		return "code_debugging"
	case strings.Contains(p, "write a python function") || strings.Contains(p, "write a function") || strings.Contains(p, "implement a") || strings.Contains(p, "called merge_") || strings.Contains(p, "called flatten"):
		return "code_generation"
	case strings.Contains(p, "clue") || strings.Contains(p, "who drinks") || strings.Contains(p, "who owns") || strings.Contains(p, "each own a different") || strings.Contains(p, "determine who"):
		return "logical_deductive_reasoning"
	case strings.Contains(p, "calculate") || strings.Contains(p, "how many") || strings.Contains(p, "how much") || strings.Contains(p, "%") || strings.Contains(p, "km/h") || strings.Contains(p, "revenue") || strings.Contains(p, "growth rate") || strings.Contains(p, "restock") || strings.Contains(p, "units remain"):
		return "mathematical_reasoning"
	default:
		return "factual_knowledge"
	}
}

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

	// Skip classifier in lean mode unless Categories already supplied - saves latency/tokens noise.
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
	deadline, hasDeadline := ctx.Deadline()
	globalRemaining := func() time.Duration {
		if !hasDeadline {
			return 8 * time.Minute
		}
		left := time.Until(deadline) - 15*time.Second
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
			for j := i; j < len(batches); j++ {
				results[j] = batchResult{index: j, answers: map[string]string{}, run: BatchRun{TaskIDs: taskIDs(batches[j]), Error: "skipped: deadline"}}
			}
			break
		}
		wg.Add(1)
		go func(idx int, b []Task) {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				results[idx] = batchResult{index: idx, answers: map[string]string{}, run: BatchRun{TaskIDs: taskIDs(b), Error: ctx.Err().Error()}}
				return
			}
			defer func() { <-sem }()

			remaining := globalRemaining()
			if remaining <= 0 {
				stopped.Store(true)
				results[idx] = batchResult{index: idx, answers: map[string]string{}, run: BatchRun{TaskIDs: taskIDs(b), Error: "no time"}}
				return
			}
			budget := remaining
			if budget > 3*time.Minute {
				budget = 3 * time.Minute
			}
			batchCtx, cancel := context.WithTimeout(ctx, budget)
			defer cancel()

			exec, err := micropy.New(micropy.Config{Timeout: 12 * time.Second, MemoryBytes: 8 * 1024 * 1024})
			if err != nil {
				results[idx] = batchResult{index: idx, answers: map[string]string{}, run: BatchRun{TaskIDs: taskIDs(b), Error: err.Error()}}
				return
			}
			exec.SetExpectedTasks(taskIDs(b))
			ans, run := a.solveBatchWithRecovery(batchCtx, exec, b)
			results[idx] = batchResult{index: idx, answers: ans, run: run}
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
			answer = "Unable to answer."
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
		run.Error = fmt.Sprintf("incomplete (%d/%d)", len(answers), len(batch))
	}
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
	missing := missingTasks(batch, merged)
	if len(missing) == 0 || ctx.Err() != nil {
		return merged, run
	}
	// One recovery call for all missing ids (avoid binary-split token blowups).
	sub, calls, tools, _ := a.solveBatch(ctx, exec, missing)
	run.Calls += calls
	run.Tools += tools
	for k, v := range sub {
		if strings.TrimSpace(v) != "" {
			merged[k] = v
		}
	}
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
	sessionID := "b-" + strings.Join(taskIDs(batch), "-")

	sys := systemPrompt(batch, a.categories)
	user := buildLeanUserPrompt(batch, a.categories)
	messages := []llm.Message{
		{Role: "system", Content: sys},
		{Role: "user", Content: user},
	}

	var calls, tools int
	var lastText string
	var partial map[string]string
	effort := a.cfg.ReasoningEffort
	if effort == "" {
		effort = "low"
	}
	batchModel := a.model
	llmClient := a.llm

	maxTurns := a.cfg.MaxTurns
	if maxTurns <= 0 {
		maxTurns = 1
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

		// 1) Code path: optional compute + finish/submit
		blocks := markdown.ExtractActionBlocks(lastText)
		if len(blocks) > 0 {
			toolTraces := make([]ToolTrace, 0, len(blocks))
			for i, block := range blocks {
				res, runErr := exec.Run(ctx, sessionID, block.Code, map[string]any{"tasks": tasksAsMaps(batch), "task_ids": taskIDs(batch)})
				tools++
				a.mu.Lock()
				a.metrics.ToolRuns++
				a.mu.Unlock()
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
					return submitted, calls, tools, fmt.Errorf("finish incomplete (%d/%d)", len(submitted), len(batch))
				}
			}
			a.appendToolTraces(batch, turn, toolTraces)
			// Harvest JSON printed by optional compute blocks.
			for _, tt := range toolTraces {
				for _, blob := range []string{tt.Stdout, tt.JSON} {
					if answers, ok := parseAnswers(blob, batch); ok {
						partial = mergeAnswers(partial, answers)
						if completeAnswers(answers, batch) {
							return answers, calls, tools, nil
						}
					}
				}
			}
		}

		// 2) Direct JSON / prose answers (primary token-efficient path)
		if answers, ok := parseAnswers(lastText, batch); ok {
			partial = mergeAnswers(partial, answers)
			if completeAnswers(answers, batch) {
				return answers, calls, tools, nil
			}
			if len(batch) > 1 {
				return answers, calls, tools, fmt.Errorf("partial answers (%d/%d)", len(answers), len(batch))
			}
		}

		if turn == maxTurns-1 {
			if completeAnswers(partial, batch) {
				return partial, calls, tools, nil
			}
			if len(batch) == 1 && lastText != "" {
				return map[string]string{batch[0].TaskID: lastText}, calls, tools, nil
			}
			return partial, calls, tools, fmt.Errorf("incomplete after %d turn(s)", maxTurns)
		}

		// Rare multi-turn: ask only for missing JSON (no long tool lecture).
		missing := missingTasks(batch, partial)
		messages = append(messages,
			llm.Message{Role: "assistant", Content: compactAssistantEcho(lastText)},
			llm.Message{Role: "user", Content: retryPrompt(missing)},
		)
	}
	return partial, calls, tools, fmt.Errorf("max turns")
}

// buildLeanUserPrompt is intentionally tiny: task_id + prompt only, no memory/skills/categories.
func buildLeanUserPrompt(batch []Task, categories map[string][]string) string {
	type item struct {
		ID     string `json:"id"`
		Kind   string `json:"k,omitempty"`
		Prompt string `json:"q"`
	}
	items := make([]item, len(batch))
	for i, t := range batch {
		it := item{ID: t.TaskID, Prompt: t.Prompt}
		// Hybrid: tag only high-error categories so model focuses recipe there.
		if cats := categories[t.TaskID]; len(cats) > 0 {
			if k := shortKindHighError(cats[0]); k != "" {
				it.Kind = k
			}
		}
		items[i] = it
	}
	b, _ := json.Marshal(items)
	return string(b)
}

// highErrorCategories are the ones that historically cost accuracy/tokens on this track.
// Recipes inject only for these; other categories rely on base caveman rules.
var highErrorCategories = map[string]bool{
	"mathematical_reasoning":   true,
	"named_entity_recognition": true,
	"code_debugging":           true,
}

func shortKindHighError(cat string) string {
	if !highErrorCategories[cat] {
		return ""
	}
	switch cat {
	case "mathematical_reasoning":
		return "math"
	case "named_entity_recognition":
		return "ner"
	case "code_debugging":
		return "cfix"
	default:
		return ""
	}
}

// Caveman-compressed recipes (drop articles/filler; keep technical terms exact).
var categoryRecipe = map[string]string{
	"mathematical_reasoning":   "math: lead final number plain digits no commas; multi-part use 1. 2. 3.; skip long derivation",
	"named_entity_recognition": "ner: Entity:TYPE only; types PERSON ORGANIZATION LOCATION DATE; exhaust text; skip invented names",
	"code_debugging":           "cfix: one-line root cause then full fixed function only",
}

func systemPrompt(batch []Task, categories map[string][]string) string {
	// Caveman base: same substance, fewer tokens (caveman skill pattern).
	var b strings.Builder
	b.WriteString("One reply. JSON only. No markdown no fence no fluff.\n")
	b.WriteString(`{"answers":[{"task_id":"...","answer":"..."},...]}`)
	b.WriteString("\nEvery task_id. Answers short. Facts exact. Code keep exact.\n")
	b.WriteString("Default: fact=1-3 short sentences; sent=Positive|Negative|Neutral + brief reason; sum=obey sentence/bullet limits; logic=Person:value all people; cgen=full function+docstring.\n")

	present := map[string]bool{}
	for _, t := range batch {
		for _, c := range categories[t.TaskID] {
			if highErrorCategories[c] {
				present[c] = true
			}
		}
	}
	for _, c := range []string{"mathematical_reasoning", "named_entity_recognition", "code_debugging"} {
		if present[c] {
			b.WriteString(categoryRecipe[c])
			b.WriteByte('\n')
		}
	}
	// If no labels, still inject high-error card once (cheap insurance).
	if len(present) == 0 && len(categories) == 0 {
		for _, c := range []string{"mathematical_reasoning", "named_entity_recognition", "code_debugging"} {
			b.WriteString(categoryRecipe[c])
			b.WriteByte('\n')
		}
	}
	return strings.TrimSpace(b.String())
}

func retryPrompt(missing []Task) string {
	return "Missing: " + strings.Join(taskIDs(missing), ",") + `. JSON only {"answers":[...]} for those ids.`
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
		AssistantMessage: truncateTrace(output, 16000),
		ReasoningContent: truncateTrace(reasoning, 8000),
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
		out[i] = BatchPlanRecord{Index: i, TaskIDs: taskIDs(b), Size: len(b), Effort: a.cfg.ReasoningEffort, Model: a.model}
	}
	return out
}

func (a *Agent) planBatches(tasks []Task) [][]Task {
	maxN := a.cfg.MaxBatchSize
	if maxN <= 0 {
		maxN = 40
	}
	// Prefer one big batch for fixed prompt amortisation (leaderboard is total tokens).
	var batches [][]Task
	for i := 0; i < len(tasks); i += maxN {
		j := i + maxN
		if j > len(tasks) {
			j = len(tasks)
		}
		batches = append(batches, tasks[i:j])
	}
	return batches
}

func maxTokensForBatch(batch []Task, turn int, effort string) int {
	// Enough room for all short answers in one JSON object.
	base := 600 + len(batch)*220
	if turn > 0 {
		base = base * 5 / 4
	}
	if base < 2048 {
		base = 2048
	}
	if base > 6000 {
		base = 6000
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
		if info == "" || strings.Contains(info, "json") || strings.Contains(info, "python") || strings.Contains(info, "py") {
			candidates = append(candidates, block.Code)
			if j := extractJSON(block.Code); j != "" {
				candidates = append(candidates, j)
			}
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

func compactAssistantEcho(s string) string {
	s = strings.TrimSpace(s)
	if len(s) > 800 {
		return s[:400] + "\n...\n" + s[len(s)-400:]
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
		return "Unable to answer: " + err.Error()
	}
	return "Unable to answer " + task.TaskID
}
