package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
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
	exec       *micropy.Executor
	clf        *taskclf.Classifier // task-type classifier; nil if unavailable
	categories map[string][]string // task_id -> predicted categories (best-effort)
	metrics    Metrics
	mu         sync.Mutex // protects metrics during concurrent batch solving
}

func New(cfg Config) (*Agent, error) {
	if cfg.MaxBatchSize <= 0 {
		cfg.MaxBatchSize = 20
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 120 * time.Second
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
	exec, err := micropy.New(micropy.Config{Timeout: 20 * time.Second, MemoryBytes: 16 * 1024 * 1024})
	if err != nil {
		return nil, err
	}
	ag.exec = exec

	// Task-type classifier is best-effort: if it can't load (missing model or
	// ONNX runtime), log and continue so the agent behaves exactly as before.
	if dir := strings.TrimSpace(cfg.ClassifierDir); dir != "" {
		if clf, cerr := taskclf.New(dir, "", cfg.ClassifierLib); cerr != nil {
			fmt.Fprintln(os.Stderr, "task classifier disabled:", cerr)
		} else {
			ag.clf = clf
		}
	}
	return ag, nil
}

// classifyTasks predicts capability categories for each task (best-effort).
// Returns nil if the classifier is unavailable; per-task failures are skipped.
func (a *Agent) classifyTasks(tasks []Task) map[string][]string {
	if a.clf == nil {
		return nil
	}
	out := make(map[string][]string, len(tasks))
	loggedErr := false
	for _, t := range tasks {
		preds, err := a.clf.Classify(t.Prompt)
		if err != nil {
			if !loggedErr { // surface inference failures instead of silently degrading to no categories
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

func (a *Agent) Solve(ctx context.Context, tasks []Task) ([]Result, Metrics, error) {
	a.metrics = Metrics{Model: a.model, StartedAt: time.Now()}
	answers := make(map[string]string, len(tasks))
	pending := append([]Task(nil), tasks...)

	if len(pending) > 0 && a.llm == nil {
		return nil, a.metrics, fmt.Errorf("FIREWORKS_API_KEY is required")
	}

	// Categories drive adaptive reasoning + technique hints. Use supplied
	// categories if provided (eval), else classify once up front (best-effort).
	if a.cfg.Categories != nil {
		a.categories = a.cfg.Categories
	} else {
		a.categories = a.classifyTasks(pending)
	}

	batches := a.planBatches(pending)
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
			batchAnswers, run := a.solveBatchWithRecovery(ctx, b)
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

	results := make([]Result, 0, len(tasks))
	for _, task := range tasks {
		answer := strings.TrimSpace(answers[task.TaskID])
		if answer == "" {
			answer = "I do not have enough information to answer confidently."
			a.metrics.Fallbacks++
		}
		results = append(results, Result{TaskID: task.TaskID, Answer: answer})
	}
	a.metrics.FinishedAt = time.Now()
	a.metrics.DurationMS = a.metrics.FinishedAt.Sub(a.metrics.StartedAt).Milliseconds()
	return results, a.metrics, nil
}

func (a *Agent) solveBatchWithRecovery(ctx context.Context, batch []Task) (map[string]string, BatchRun) {
	run := BatchRun{TaskIDs: taskIDs(batch)}
	answers, calls, tools, err := a.solveBatch(ctx, batch)
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
	left, leftRun := a.solveBatchWithRecovery(ctx, batch[:mid])
	right, rightRun := a.solveBatchWithRecovery(ctx, batch[mid:])
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

func (a *Agent) solveBatch(ctx context.Context, batch []Task) (map[string]string, int, int, error) {
	a.metrics.BatchCount++
	sys := systemPrompt()
	if g := a.categoryGuidance(batch); g != "" {
		sys += "\n\nTechnique hints for these task types:\n" + g
	}
	messages := []llm.Message{
		{Role: "system", Content: sys},
		{Role: "user", Content: a.ctx.BuildBatchPrompt(batch, a.memory.LoadForTasks(batch), a.skills.LoadForPrompt(a.cfg.MemoryRoot, taskText(batch), 8000), a.categories)},
	}
	var calls int
	var tools int
	var lastText string
	effort := a.effortForBatch(batch)

	for turn := 0; turn < 3; turn++ {
		maxTokens := maxTokensForBatch(batch, turn, effort)
		callStart := time.Now()
		text, usage, err := a.llm.Chat(ctx, messages, maxTokens, effort)
		callDur := time.Since(callStart)
		if err != nil {
			a.recordCall(batch, turn, callStart, callDur, usage, "", err)
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

		if answers, ok := parseAnswers(lastText, batch); ok {
			return answers, calls, tools, nil
		}

		// If the model hit max_tokens without producing parseable output,
		// the batch is too large for the token budget. Return an error so
		// solveBatchWithRecovery splits it in half (much cheaper than retrying).
		if lastText == "" && usage.CompletionTokens >= maxTokens*9/10 {
			return nil, calls, tools, fmt.Errorf("batch truncated at max_tokens=%d (completion=%d), needs split", maxTokens, usage.CompletionTokens)
		}

		blocks := markdown.ExtractActionBlocks(lastText)
		if len(blocks) == 0 {
			messages = append(messages,
				llm.Message{Role: "assistant", Content: compactAssistantEcho(lastText)},
				llm.Message{Role: "user", Content: "Return only valid JSON in the required schema. Do not include markdown."},
			)
			continue
		}

		observations := make([]map[string]any, 0, len(blocks))
		for i, block := range blocks {
			res, err := a.exec.Run(ctx, "batch-"+strings.Join(taskIDs(batch), "-"), block.Code, map[string]any{"tasks": batch})
			tools++
			a.mu.Lock()
			a.metrics.ToolRuns++
			a.mu.Unlock()
			obs := map[string]any{"index": i, "stdout": res.Stdout, "json": res.JSON, "value": res.Value}
			if err != nil {
				obs["error"] = err.Error()
			}
			observations = append(observations, obs)
		}
		obsBytes, _ := json.Marshal(observations)
		messages = append(messages,
			llm.Message{Role: "assistant", Content: lastText},
			llm.Message{Role: "user", Content: "MicroPython observation: " + string(obsBytes) + "\nNow return the final JSON object only."},
		)
	}

	if answers, ok := parseAnswers(lastText, batch); ok {
		return answers, calls, tools, nil
	}
	return nil, calls, tools, fmt.Errorf("model did not return parseable answers")
}

func (a *Agent) recordCall(batch []Task, turn int, start time.Time, dur time.Duration, usage llm.Usage, output string, _ any) {
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
	}
	a.mu.Lock()
	a.metrics.CallRecords = append(a.metrics.CallRecords, rec)
	a.mu.Unlock()
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

func systemPrompt() string {
	parts := []string{
		"You are a precise, terse task-solving agent.",
		"Output ONLY this JSON, nothing else: {\"answers\":[{\"task_id\":\"...\",\"answer\":\"...\"}]}. Keep every task_id.",
		"Give the shortest fully-correct answer and match the exact format each task asks for. No preamble, explanation, or thinking in the output.",
		"Answer directly. Only when a task needs exact computation, code, files, or the web, emit ONE fenced micropy block (MicroPython WASM; tools are Python globals sh.run, fs.read/write/edit/list, web.fetch/search), e.g. sh.run(command='python3 -c \"print(6*7)\"'); set result to a JSON value, then use the returned observation to output the final JSON.",
	}
	return strings.Join(parts, "\n")
}

// categoryHints are concise, category-specific techniques injected into the
// system prompt only when a batch contains that category (so they cost tokens
// only where they help). They favour LOCAL code execution (zero Fireworks
// tokens) over prose where that is more reliable.
var categoryHints = map[string]string{
	"logical_deductive_reasoning": "Logical / constraint puzzles: solve them PROGRAMMATICALLY, not by prose (prose is error-prone). Emit a micropy block that enumerates the candidate assignments and keeps only those satisfying EVERY stated constraint - e.g. sh.run(command='python3 -c \"import itertools, json; ...\"'). Set result to the unique surviving assignment, then output the answer it proves.",
	"named_entity_recognition":    "Named-entity recognition: be EXHAUSTIVE for the requested types (person, organisation, location, date). Read the whole text; do not miss trailing or adjectival entities. Label each entity with its type; never invent entities absent from the text; return exactly the requested format.",
}

// categoryGuidance concatenates the hints for the distinct categories present in
// a batch (per the classifier / supplied categories).
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
	for _, c := range []string{"logical_deductive_reasoning", "named_entity_recognition"} {
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
	// ALLOWED_MODELS (injected by the harness at runtime) is the single source
	// of truth for permitted models; we never hardcode a model ID. AGENT_MODEL
	// pins a specific choice, honoured only when it is one of the permitted
	// models; otherwise the first permitted model wins (list order is priority).
	if len(allowed) == 0 {
		return preferred // no allow-list (local dev only); "" if unset
	}
	if preferred != "" {
		for _, m := range allowed {
			if m == preferred {
				return preferred
			}
		}
	}
	// Prefer minimax-m3 as the default general-purpose model if present.
	for _, m := range allowed {
		if m == "accounts/fireworks/models/minimax-m3" {
			return m
		}
	}
	return allowed[0]
}

// planBatches packs tasks into batches by estimated token budget so the large,
// fixed per-call prompt overhead (system prompt + injected memory) is amortised
// over as many tasks as safely fit, capped by MaxBatchSize. The prompt is ~90%+
// of each call's tokens, so fewer, fuller batches cut total tokens sharply (see
// docs/model-routing.md). solveBatchWithRecovery still halves a batch on failure,
// so an over-packed batch self-heals.
func (a *Agent) planBatches(tasks []Task) [][]Task {
	maxTok := a.cfg.MaxBatchTokens
	if maxTok <= 0 {
		maxTok = 12000
	}
	maxN := a.cfg.MaxBatchSize
	if maxN <= 0 {
		maxN = 40
	}
	// Group by reasoning-effort tier so each batch is tier-homogeneous (one
	// reasoning_effort fits all its tasks), then token-pack within each tier.
	// With a fixed Config.ReasoningEffort every task shares one tier, so this
	// collapses to a single group and packs exactly as before.
	// Group by (effort, model) so each batch is both tier-homogeneous and
	// model-homogeneous. With no model routing this collapses to effort-only grouping.
	byGroup := map[string][]Task{}
	var groups []string
	for _, t := range tasks {
		e := a.effortForTask(t)
		m := a.modelForTask(t)
		key := e + "|" + m
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
			tk := estTokens(t.Prompt) + 16 // task text + per-task JSON overhead
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

// estTokens is a cheap chars/4 token estimate used only for batch packing.
func estTokens(s string) int { return len(s)/4 + 1 }

// effortTier elevates only the categories that empirically benefit from more
// reasoning. On the real-task eval only logical (constraint puzzles) improved
// with higher effort; every other category was already at ceiling on "low", so
// raising them only burned tokens. Categories not listed default to "low".
// See docs/model-routing.md.
var effortTier = map[string]string{
	"logical_deductive_reasoning": "xhigh",
}

var tierRank = map[string]int{"low": 0, "medium": 1, "high": 2, "xhigh": 3}

// effortForTask returns reasoning_effort for one task: an explicit
// Config.ReasoningEffort override wins; otherwise the highest tier among the
// task's predicted categories (default low).
func (a *Agent) effortForTask(t Task) string {
	if a.cfg.ReasoningEffort != "" {
		return a.cfg.ReasoningEffort
	}
	tiers := a.cfg.EffortTierMap
	if tiers == nil {
		tiers = effortTier // built-in default
	}
	best := "low"
	for _, c := range a.categories[t.TaskID] {
		if e, ok := tiers[c]; ok && tierRank[e] > tierRank[best] {
			best = e
		}
	}
	return best
}

// modelForTask returns the model to use for a task based on its categories.
// If no routing map is set or no category matches, returns the default model.
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

// effortForBatch returns the batch's reasoning_effort. planBatches makes batches
// tier-homogeneous, so the first task's tier applies to the whole batch.
func (a *Agent) effortForBatch(batch []Task) string {
	if len(batch) == 0 {
		return a.cfg.ReasoningEffort
	}
	return a.effortForTask(batch[0])
}

// maxTokensForBatch sizes the output budget by batch size AND reasoning tier:
// higher effort needs far more room or the reasoning truncates and the batch
// fails. Bounded per tier so a batch can't run away.
func maxTokensForBatch(batch []Task, turn int, effort string) int {
	perTask, ceil := 1200, 32000
	switch effort {
	case "medium":
		perTask, ceil = 1600, 40000
	case "high":
		perTask, ceil = 2500, 48000
	case "xhigh":
		perTask, ceil = 4000, 64000
	}
	base := 1200 + len(batch)*perTask
	// On retry (empty output), INCREASE the budget rather than shrinking it.
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
	ids := map[string]bool{}
	for _, t := range batch {
		ids[t.TaskID] = true
	}
	for _, candidate := range candidates {
		if answers, ok := parseAnswerJSON(candidate, ids); ok {
			return answers, true
		}
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
			out[item.TaskID] = stringifyAnswer(item.Answer)
		}
		return out, len(out) == len(ids)
	}

	var arr []struct {
		TaskID string `json:"task_id"`
		Answer any    `json:"answer"`
	}
	if err := json.Unmarshal([]byte(s), &arr); err == nil && len(arr) > 0 {
		out := map[string]string{}
		for _, item := range arr {
			if ids[item.TaskID] {
				out[item.TaskID] = stringifyAnswer(item.Answer)
			}
		}
		return out, len(out) == len(ids)
	}

	var obj map[string]any
	if err := json.Unmarshal([]byte(s), &obj); err == nil && len(obj) > 0 {
		out := map[string]string{}
		for id := range ids {
			if v, ok := obj[id]; ok {
				out[id] = stringifyAnswer(v)
			}
		}
		return out, len(out) == len(ids)
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
	return "Unable to answer confidently: " + err.Error()
}
