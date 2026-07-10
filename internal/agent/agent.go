package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ashaibani/yassai/internal/contextmgr"
	"github.com/ashaibani/yassai/internal/llm"
	"github.com/ashaibani/yassai/internal/localllm"
	"github.com/ashaibani/yassai/internal/markdown"
	"github.com/ashaibani/yassai/internal/memory"
	"github.com/ashaibani/yassai/internal/pyexec"
	"github.com/ashaibani/yassai/internal/skills"
	"github.com/ashaibani/yassai/internal/taskclf"
)

type Agent struct {
	cfg          Config
	model        string
	llm          *llm.Client
	ctx          contextmgr.Manager
	memory       *memory.Store
	skills       *skills.Loader
	clf          *taskclf.Classifier
	localRejects map[string]string // gate-rejected local answers, kept as last-ditch fallbacks
	categories   map[string][]string
	metrics      Metrics
	mu           sync.Mutex
}

func New(cfg Config) (*Agent, error) {
	if cfg.MaxBatchSize <= 0 {
		cfg.MaxBatchSize = 40
	}
	// MathBatchSize 0 means inherit MaxBatchSize (status quo).
	if cfg.MaxTurns <= 0 {
		cfg.MaxTurns = 1 // single-pass default for token efficiency
	}
	if cfg.MaxConcurrency <= 0 {
		cfg.MaxConcurrency = 1
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 120 * time.Second
	}
	// Empty ReasoningEffort => adaptive per-category tiers (see LeanEffortTiers).
	// Explicit values ("none","low","medium",...) override all categories.
	if cfg.EffortTierMap == nil {
		cfg.EffortTierMap = LeanEffortTiers()
	}
	model := chooseModel(cfg.AllowedModels, cfg.PreferredModel)
	ag := &Agent{
		cfg:          cfg,
		model:        model,
		ctx:          contextmgr.Manager{MaxContextTokens: cfg.MaxContextTokens, ReserveTokens: 8000},
		memory:       memory.New(cfg.MemoryRoot),
		skills:       skills.NewLoader(cfg.SkillRoots),
		localRejects: map[string]string{},
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
	// Local solvers spawn LAZILY inside localFirstPass, one lane at a time,
	// and are closed as soon as their lane finishes: on the 4 GB judge VM the
	// classifier arena, the assist server, and the tool server never coexist.
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
	case strings.Contains(p, "bug") || strings.Contains(p, "contains a bug") || strings.Contains(p, "corrected function") || strings.Contains(p, "identify the bug") || evalStyleCode(p):
		return "code_debugging"
	case strings.Contains(p, "write a python function") || strings.Contains(p, "write a function") || strings.Contains(p, "implement a") || strings.Contains(p, "called merge_") || strings.Contains(p, "called flatten"):
		return "code_generation"
	case strings.Contains(p, "clue") || strings.Contains(p, "who drinks") || strings.Contains(p, "who owns") || strings.Contains(p, "each own a different") || strings.Contains(p, "determine who") ||
		strings.Contains(p, "seating order") || strings.Contains(p, "sits in seat") || strings.Contains(p, "seats numbered") || strings.Contains(p, "in a row") ||
		strings.Contains(p, "one weight per") || strings.Contains(p, "the heaviest") || strings.Contains(p, "each have a different") || strings.Contains(p, "each drink") ||
		(strings.Contains(p, "if all ") && strings.Contains(p, "are all ")) || strings.Contains(p, "does it follow") || strings.Contains(p, "syllog"):
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

	if a.cfg.Categories != nil {
		a.categories = a.cfg.Categories
	} else {
		a.categories = a.classifyTasks(tasks)
	}
	// The classifier's one job is done: free its ONNX arena (fp32 is ~0.5 GB)
	// before any llama-server spawns. Peak memory on the 4 GB judge VM is the
	// max of the phases, not their sum - so classify, free, then infer.
	if a.clf != nil {
		a.clf.Close()
		a.clf = nil
	}

	pending := make([]Task, 0, len(tasks))
	for _, task := range tasks {
		category := heuristicCategory(task.Prompt)
		if cats := a.categories[task.TaskID]; len(cats) > 0 {
			category = cats[0]
		}
		if answer, ok := trySolveLocal(ctx, task, category); ok {
			answers[task.TaskID] = answer
			a.metrics.LocalAnswers++
			continue
		}
		pending = append(pending, task)
	}
	// Model lanes run BEFORE batch planning so batching only ever sees truly
	// remote work: a gate reject then joins an existing batch instead of
	// stranding as a solo call that pays the whole per-batch scaffold again
	// (observed: one leftover logic task = 1,318 tokens, 40% of the run).
	pending = a.localFirstPass(ctx, pending, answers)
	// AGENT_NO_REMOTE=1: the zero-token mode. Nothing ever reaches Fireworks;
	// gate-rejected local answers ship as-is (recall-biased gates bounce many
	// correct answers) and the run scores ZERO_API_CALLS. Rank is tokens
	// ascending, so matching the 0-token leaders requires exactly this.
	if os.Getenv("AGENT_NO_REMOTE") == "1" {
		pending = nil
	}
	if len(pending) > 0 && a.llm == nil {
		return nil, a.metrics, fmt.Errorf("FIREWORKS_API_KEY is required for %d unsolved task(s)", len(pending))
	}

	batches := a.planBatches(pending)
	a.metrics.Classifications = a.categories
	a.metrics.BatchPlan = a.batchPlanRecords(batches)
	if len(batches) == 0 {
		out := make([]Result, 0, len(tasks))
		for _, task := range tasks {
			answer := strings.TrimSpace(answers[task.TaskID])
			if answer == "" {
				if rej := strings.TrimSpace(a.localRejects[task.TaskID]); rej != "" {
					fmt.Fprintf(os.Stderr, "fallback %s: shipping gate-rejected local answer\n", task.TaskID)
					answer = rej
				} else {
					answer = "Unable to answer."
				}
				a.metrics.Fallbacks++
			}
			out = append(out, Result{TaskID: task.TaskID, Answer: answer})
		}
		a.metrics.FinishedAt = time.Now()
		a.metrics.DurationMS = a.metrics.FinishedAt.Sub(a.metrics.StartedAt).Milliseconds()
		return out, a.metrics, nil
	}

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

			exec, err := pyexec.New(pyexec.Config{Timeout: 20 * time.Second})
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
			// A gate-rejected local answer is a real attempt the recall-biased
			// gates bounced; when the remote path produced nothing (deadline,
			// provider error) it is strictly better than an admission.
			if rej := strings.TrimSpace(a.localRejects[task.TaskID]); rej != "" {
				fmt.Fprintf(os.Stderr, "fallback %s: shipping gate-rejected local answer\n", task.TaskID)
				answer = rej
			} else {
				answer = "Unable to answer."
			}
			a.metrics.Fallbacks++
		}
		out = append(out, Result{TaskID: task.TaskID, Answer: answer})
	}
	a.metrics.FinishedAt = time.Now()
	a.metrics.DurationMS = a.metrics.FinishedAt.Sub(a.metrics.StartedAt).Milliseconds()
	return out, a.metrics, nil
}

// localFirstPass answers what the in-container models can before any batch is
// planned: zero Fireworks tokens per accepted answer, and gate rejects join a
// shared remote batch instead of stranding in their own scaffolded call.
//
// The budget is deadline-aware, not a fixed slice: on the judging VM the 1B
// decodes on shared CPU at 10-20x below dev-machine speed, and a fixed budget
// let local attempts starve the remote batch entirely (observed live: the
// maths batch died on "context deadline exceeded" with zero calls and two
// dice-roll fallback answers). Local solving must always leave the remote
// path its reserve.
func (a *Agent) localFirstPass(ctx context.Context, pending []Task, answers map[string]string) []Task {
	baseCfgd := strings.TrimSpace(a.cfg.LocalBaseModelPath) != ""
	toolCfgd := strings.TrimSpace(a.cfg.LocalModelPath) != ""
	if !baseCfgd && !toolCfgd {
		return pending
	}
	// Reserve enough of the context for the remote batches + response
	// assembly. Everything before deadline-reserve belongs to local solving,
	// capped at 4 minutes so dev runs without deadlines stay bounded too.
	reserve := remoteReserve
	if os.Getenv("AGENT_NO_REMOTE") == "1" {
		// No remote phase to protect - keep only enough slack to assemble and
		// write results, and let local solving use the rest of the window.
		reserve = 30 * time.Second
	}
	localBudget := time.Now().Add(4 * time.Minute)
	if deadline, ok := ctx.Deadline(); ok {
		if cut := deadline.Add(-reserve); cut.Before(localBudget) {
			localBudget = cut
		}
	}
	if reserve != remoteReserve {
		// Zero-token runs on the 2 vCPU judge VM need the whole window: 19
		// tasks with retries can exceed the 4-minute dev default.
		if deadline, ok := ctx.Deadline(); ok {
			localBudget = deadline.Add(-reserve)
		}
	}

	// Partition by lane so each server lives only for its own pass: the
	// assist server closes before the tool server spawns, keeping one model
	// resident at a time on the 4 GB judge VM.
	type assistJob struct {
		task   Task
		family string // "" => code trace
	}
	var assistJobs []assistJob
	var toolTasks []Task
	remaining := make([]Task, 0, len(pending))
	for _, t := range pending {
		fam := heuristicCategory(t.Prompt)
		// Off-distribution phrasings are where the keyword heuristic misroutes
		// ("Split £2,760 in the ratio 5:4:3" has no maths keyword and landed in
		// the factual lane - wrong by a mile). The classifier reads semantics;
		// a local lane only engages when BOTH agree. Disagreement is exactly
		// the uncertainty signal the remote insurance call exists for.
		if cats := a.categories[t.TaskID]; len(cats) > 0 && cats[0] != fam {
			fmt.Fprintf(os.Stderr, "local skip %s: classifier says %s, heuristic says %s\n", t.TaskID, cats[0], fam)
			remaining = append(remaining, t)
			continue
		}
		switch {
		case baseCfgd && (fam == localllm.FamilyCodeGen || fam == localllm.FamilyNER):
			assistJobs = append(assistJobs, assistJob{t, fam})
		case baseCfgd && a.cfg.LocalBaseExtended != "" &&
			(fam == localllm.FamilySentiment || fam == localllm.FamilySummarisation || fam == localllm.FamilyFactual):
			assistJobs = append(assistJobs, assistJob{t, fam})
		case baseCfgd && fam == "code_debugging" && evalStyleCode(strings.ToLower(t.Prompt)):
			assistJobs = append(assistJobs, assistJob{t, ""})
		case baseCfgd && fam == "code_debugging":
			assistJobs = append(assistJobs, assistJob{t, localllm.FamilyCodeFix})
		case toolCfgd && localTrainedFamily(t.Prompt) && a.taskUsesCode(t):
			toolTasks = append(toolTasks, t)
		default:
			remaining = append(remaining, t)
		}
	}

	record := func(t Task, res localllm.Result) {
		if res.OK {
			answers[t.TaskID] = res.Answer
			a.metrics.LocalAnswers++
			return
		}
		fmt.Fprintf(os.Stderr, "local reject %s: %s\n", t.TaskID, res.Reason)
		if strings.TrimSpace(res.Answer) != "" {
			// Recall-biased gates reject plenty of correct answers. Keep
			// them: if the remote path dies (timeout, provider error), a
			// rejected local answer beats shipping nothing.
			a.localRejects[t.TaskID] = res.Answer
		}
		remaining = append(remaining, t)
	}

	if len(assistJobs) > 0 {
		solver, err := localllm.NewDirect(localllm.Config{ModelPath: a.cfg.LocalBaseModelPath, LibPath: a.cfg.LocalLibPath, Extended: a.cfg.LocalBaseExtended})
		if err != nil {
			// Never fatal: the Fireworks path is the accuracy baseline.
			fmt.Fprintln(os.Stderr, "local base model disabled:", err)
			for _, j := range assistJobs {
				remaining = append(remaining, j.task)
			}
		} else {
			for _, j := range assistJobs {
				if ctx.Err() != nil || time.Now().After(localBudget) {
					remaining = append(remaining, j.task)
					continue
				}
				if j.family == "" {
					record(j.task, solver.SolveCodeTrace(ctx, j.task.Prompt))
				} else {
					record(j.task, solver.SolveTask(ctx, j.task.Prompt, j.family))
				}
			}
			solver.Close()
		}
	}
	if len(toolTasks) > 0 {
		solver, err := localllm.New(localllm.Config{ModelPath: a.cfg.LocalModelPath, LibPath: a.cfg.LocalLibPath})
		if err != nil {
			fmt.Fprintln(os.Stderr, "local model disabled:", err)
			remaining = append(remaining, toolTasks...)
		} else {
			for _, t := range toolTasks {
				if ctx.Err() != nil || time.Now().After(localBudget) {
					remaining = append(remaining, t)
					continue
				}
				record(t, solver.SolveTask(ctx, t.Prompt))
			}
			solver.Close()
		}
	}
	return remaining
}

// remoteReserve is the context slice localFirstPass must leave for the remote
// batches: one folded batch call plus recovery and output writing.
const remoteReserve = 3 * time.Minute

func (a *Agent) solveBatchWithRecovery(ctx context.Context, exec *pyexec.Executor, batch []Task) (map[string]string, BatchRun) {
	run := BatchRun{TaskIDs: taskIDs(batch)}
	if ctx.Err() != nil {
		run.Error = ctx.Err().Error()
		return map[string]string{}, run
	}
	answers, calls, tools, err := a.solveBatch(ctx, exec, batch, false)
	// Physics screen on remote answers: an impossible two-vehicle meeting time
	// (03:04.5 for a 09:30 departure - observed from Fireworks) is dropped so
	// the standard missing-ids recovery earns one retry at it. Applied only to
	// this first pass - a retry that still fails the bound ships as-is rather
	// than looping.
	for _, t := range batch {
		if ans, ok := answers[t.TaskID]; ok {
			if reason := localllm.MeetingTimeBound(t.Prompt, ans); reason != "" {
				fmt.Fprintf(os.Stderr, "remote answer rejected %s: %s\n", t.TaskID, reason)
				delete(answers, t.TaskID)
			}
		}
	}
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
	sub, calls, tools, _ := a.solveBatch(ctx, exec, missing, true)
	run.Calls += calls
	run.Tools += tools
	for k, v := range sub {
		if strings.TrimSpace(v) != "" {
			merged[k] = v
		}
	}
	return merged, run
}

func (a *Agent) solveBatch(ctx context.Context, exec *pyexec.Executor, batch []Task, lean bool) (map[string]string, int, int, error) {
	if err := ctx.Err(); err != nil {
		return nil, 0, 0, err
	}
	a.mu.Lock()
	a.metrics.BatchCount++
	a.mu.Unlock()
	exec.ResetSubmit()
	exec.SetExpectedTasks(taskIDs(batch))
	sessionID := "b-" + strings.Join(taskIDs(batch), "-")

	allowCode := a.batchAllowsCode(batch)
	messages := a.buildBatchMessages(batch, allowCode, lean)

	var calls, tools int
	var lastText string
	var partial map[string]string
	hadSuccessfulTool := false
	effort := a.effortForBatch(batch)
	wiredEffort := wireEffort(effort)
	batchModel := a.model
	llmClient := a.llm

	maxTurns := a.cfg.MaxTurns
	if maxTurns <= 0 {
		maxTurns = 1
	}
	// Math batches need room for tool call + observation + final JSON.
	if allowCode && maxTurns < 3 {
		maxTurns = 3
	}

	for turn := 0; turn < maxTurns; turn++ {
		if err := ctx.Err(); err != nil {
			if completeAnswers(partial, batch) {
				return partial, calls, tools, nil
			}
			return partial, calls, tools, err
		}
		maxTokens := maxTokensForBatch(batch, turn, effort, allowCode)
		opts := llm.ChatOptions{MaxTokens: maxTokens, ReasoningEffort: wiredEffort}
		if allowCode {
			opts.Tools = []llm.ToolDef{llm.PythonExecTool()}
			if hadSuccessfulTool {
				opts.ToolChoice = "none"
			} else {
				opts.ToolChoice = "required"
			}
		}
		callStart := time.Now()
		chat, err := llmClient.ChatWithOptions(ctx, messages, opts)
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

		// 1) Native tool calls (preferred): run_python on math batches.
		if allowCode && len(chat.ToolCalls) > 0 {
			toolTraces := make([]ToolTrace, 0, len(chat.ToolCalls))
			messages = append(messages, llm.Message{Role: "assistant", Content: lastText, ToolCalls: normaliseToolCalls(chat.ToolCalls)})
			for i, tc := range chat.ToolCalls {
				code := toolCallCode(tc)
				tt := ToolTrace{Index: i, Code: code}
				if code == "" {
					tt.Error = "empty run_python code"
					toolTraces = append(toolTraces, tt)
					messages = append(messages, llm.Message{
						Role: "tool", ToolCallID: tc.ID, Name: tc.Function.Name,
						Content: `{"error":"empty code"}`,
					})
					continue
				}
				res, runErr := exec.Run(ctx, sessionID, code, map[string]any{"tasks": tasksAsMaps(batch), "task_ids": taskIDs(batch)})
				tools++
				a.mu.Lock()
				a.metrics.ToolRuns++
				a.mu.Unlock()
				tt.Stdout, tt.JSON = res.Stdout, res.JSON
				if runErr != nil {
					tt.Error = runErr.Error()
				} else if strings.TrimSpace(res.Stdout) != "" || strings.TrimSpace(res.JSON) != "" {
					hadSuccessfulTool = true
				}
				toolTraces = append(toolTraces, tt)
				observation, _ := json.Marshal(map[string]any{
					"stdout": res.Stdout,
					"json":   res.JSON,
					"error":  tt.Error,
				})
				messages = append(messages, llm.Message{
					Role: "tool", ToolCallID: tc.ID, Name: tc.Function.Name,
					Content: string(observation),
				})
				if exec.IsSubmitted() {
					submitted := exec.SubmitResult()
					partial = mergeAnswers(partial, submitted)
					a.appendToolTraces(batch, turn, toolTraces)
					if completeAnswers(submitted, batch) {
						return submitted, calls, tools, nil
					}
				}
				for _, blob := range []string{res.Stdout, res.JSON} {
					if answers, ok := parseAnswers(blob, batch); ok {
						partial = mergeAnswers(partial, answers)
						if completeAnswers(answers, batch) {
							a.appendToolTraces(batch, turn, toolTraces)
							return answers, calls, tools, nil
						}
					}
				}
			}
			a.appendToolTraces(batch, turn, toolTraces)
			if hadSuccessfulTool {
				messages = append(messages, llm.Message{Role: "user", Content: "Use the run_python stdout/json above to return final JSON answers for all ids now. Do not call tools again. Copy checked values exactly: last/exactly K averages must show K items; projection answers must include the raw average used; keep seconds in time answers when present."})
			}
			// Continue so the model can emit final JSON from tool observations.
			continue
		}

		// 2) Legacy markdown ```python``` act blocks (compat if model ignores tools).
		var blocks []markdown.Block
		if allowCode {
			blocks = markdown.ExtractActionBlocks(lastText)
		}
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

		// 3) Direct JSON / prose answers (primary path for non-math).
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

		// Rare multi-turn: ask only for missing JSON.
		missing := missingTasks(batch, partial)
		messages = append(messages,
			llm.Message{Role: "assistant", Content: compactAssistantEcho(lastText)},
			llm.Message{Role: "user", Content: retryPrompt(missing)},
		)
	}
	return partial, calls, tools, fmt.Errorf("max turns")
}

// toolCallCode extracts MicroPython source from a native tool call.
func toolCallCode(tc llm.ToolCall) string {
	args := strings.TrimSpace(tc.Function.Arguments)
	if args == "" {
		return ""
	}
	var obj struct {
		Code   string `json:"code"`
		Source string `json:"source"`
		Python string `json:"python"`
		Script string `json:"script"`
	}
	if err := json.Unmarshal([]byte(args), &obj); err == nil {
		for _, c := range []string{obj.Code, obj.Source, obj.Python, obj.Script} {
			if strings.TrimSpace(c) != "" {
				return c
			}
		}
	}
	if strings.HasPrefix(args, "{") || strings.HasPrefix(args, "[") {
		return ""
	}
	// Arguments sometimes arrive as a raw code string.
	if strings.Contains(args, "print") || strings.Contains(args, "=") {
		return strings.Trim(args, "\"")
	}
	return ""
}

func normaliseToolCalls(calls []llm.ToolCall) []llm.ToolCall {
	out := make([]llm.ToolCall, len(calls))
	copy(out, calls)
	for i := range out {
		code := toolCallCode(out[i])
		payload, _ := json.Marshal(map[string]string{"code": code})
		out[i].Function.Arguments = string(payload)
	}
	return out
}
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

// highErrorCategories get a one-line format recipe (generic, never answer content).
var highErrorCategories = map[string]bool{
	"mathematical_reasoning":      true,
	"named_entity_recognition":    true,
	"code_debugging":              true,
	"code_generation":             true,
	"logical_deductive_reasoning": true,
	"text_summarisation":          true,
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
	case "code_generation":
		return "cgen"
	case "logical_deductive_reasoning":
		return "logic"
	case "text_summarisation":
		return "sum"
	default:
		return ""
	}
}

// Generic format recipes only - no task-specific content.
// Tuned for harsh LLM-judge survival + native run_python tool on math batches.
var categoryRecipe = map[string]string{
	"mathematical_reasoning":      "math: call run_python once; write STRAIGHT-LINE code (no assert, no try) and build the submit answer by INTERPOLATING the computed variables (str(round(total))+' units'), NEVER hand-type a computed number; answer in the exact shape asked (numbered parts if the task numbers them); requested units/currency; ASCII signs; preserve exact times incl seconds (offset departures: head start covered first, then closing speed); chained maths (growth rates, projections) MUST reuse the raw variables - idiom: rates=[(v[i]-v[i-1])/v[i-1]*100 ...]; proj=last*(1+sum(rates[-3:])/3/100); round(r,1) is for DISPLAY only, never feed a rounded value into later maths; for 'last K' slices use the final K entries (v[-K:]); sequential percent changes compound in order",
	"named_entity_recognition":    "ner: read the label inventory from the prompt and use its exact names/abbreviations (for example PERSON/ORGANIZATION/LOCATION/DATE or PER/ORG/LOC/MISC); preserve exact source spans and order of appearance; do not invent labels or entities; label only what the inventory permits - and when a prominent named entity (programme, mission, product) fits NO permitted label, note it separately as outside the inventory instead of mislabelling or silently dropping it",
	"code_debugging":              "cbug: if asked to FIX: one-line cause, then the minimal corrected fn - change only the bug, plain code no fences, preserve the stated semantics. If asked what code EVALUATES to or does: when run_python is available RUN the snippet, print the values and read the REAL result - never hand-trace; state the exact output/behaviour first, then the one-line mechanism (late binding, shared mutable default, exhausted iterator, float representation, aliasing, wrong base case); infinite recursion/hangs: reason it out instead of running",
	"code_generation":             "cgen: plain code no fences; full fn + 1-line docstring; handle the edge cases the spec states; implement EXACTLY the spec - add no operation it does not require (no stray reverse/sort/dedup); trace the given example input to verify output shape and order before answering",
	"logical_deductive_reasoning": "logic: solve via run_python - brute-force enumerate (itertools.permutations/product) over the candidates; encode EACH clue as an explicit boolean condition - including world-knowledge clues (allergic to fur => not in {cat, dog, hamster, rabbit}; 'immediately left of' => index difference exactly 1) - and collect assignments satisfying ALL of them; exactly ONE should survive - submit from sols[0] only when len(sols)==1, otherwise a clue is mis-encoded: re-read the clues and fix the conditions (never assert/crash, never guess); build the answer by INTERPOLATING the winning variables, NEVER hand-type names or values; answer in the exact shape asked (assignment list, seating order left to right, or single value); multiple-choice: the exact option text; never refuse",
	"text_summarisation":          "sum: output EXACTLY the N sentences or N bullets the task states (count them - never more, never fewer); obey every stated constraint (per-bullet word caps, required figures or terms); map each sentence/bullet to a DISTINCT major theme and cover ALL major themes - benefits/drivers, problems/challenges, AND any responses/countermeasures/outlook the source describes (the response theme is the one most often dropped - never drop it); if themes outnumber bullets, pack two related themes into one bullet rather than omitting one; the COUNT is the hardest constraint of all: N+1 bullets is an automatic fail even with perfect coverage - merge, never add",
}

// categoryRecipeNoTools replaces the run_python-centric recipes when the batch
// has no code-exec surface (folded gate rejects in the direct batch). Same
// answer-shape discipline, no references to a tool that is not there.
var categoryRecipeNoTools = map[string]string{
	"mathematical_reasoning":      "math: compute carefully step by step, double-check each arithmetic step, and answer in the exact shape asked (requested units/currency, numbered parts if the task numbers them).",
	"logical_deductive_reasoning": "logic: deduce systematically - test each candidate against EVERY clue (including world-knowledge clues like allergies) and give only the final assignment in the exact shape asked; multiple-choice: the exact option text; never refuse.",
	"code_debugging":              "cbug: if asked to FIX: one-line cause, then the minimal corrected fn - change only the bug, plain code no fences, preserve the stated semantics. State the cause ONLY in terms of what the buggy line actually does (wrong operator, wrong index, wrong method for that TYPE - check the type first); never assert a mechanism you are not certain applies. If asked what code EVALUATES to: trace it carefully step by step; state the exact output/behaviour first, then the one-line mechanism.",
}

func systemPrompt(batch []Task, categories map[string][]string, allowCode bool) string {
	header, recipes := systemPromptParts(batch, categories, allowCode)
	if recipes == "" {
		return header
	}
	return header + "\n" + recipes
}

// systemPromptParts splits the system prompt into the JSON output contract
// (header - always sent as text so parseAnswers never depends on OCR) and the
// per-category recipes (movable into a textimg render in TextImg "full" mode).
func systemPromptParts(batch []Task, categories map[string][]string, allowCode bool) (string, string) {
	// Caveman-tight header (inspo/caveman: drop filler, keep every constraint
	// byte-exact) - and family clauses ship only when their family is in the
	// batch. The fact/sent/sum guidance used to ride along unconditionally.
	var b strings.Builder
	b.WriteString("JSON only:\n")
	b.WriteString(`{"answers":[{"task_id":"...","answer":"..."},...]}`)
	b.WriteString("\nAll ids. Plain English strings, no preamble. Terse but complete: every asked part, label, unit, reason, docstring, edge case.\n")
	inBatch := map[string]bool{}
	for _, t := range batch {
		for _, c := range categories[t.TaskID] {
			inBatch[c] = true
		}
		inBatch[heuristicCategory(t.Prompt)] = true
	}
	if inBatch["factual_knowledge"] {
		b.WriteString("fact: COUNT the parts asked - incl. sub-elements in passing or brackets ('plus X') - answer EVERY one, 1-2 short sentences each, no extra background.\n")
	}
	if inBatch["sentiment_classification"] {
		b.WriteString("sent: Positive|Negative|Neutral + 1 short reason; judge the writer's OVERALL verdict - sarcasm means the opposite of the surface words.\n")
	}
	if inBatch["text_summarisation"] {
		b.WriteString("sum: EXACTLY the stated number of sentences/bullets - count them - and RECOUNT each bullet against any word cap before answering.\n")
	}
	var r strings.Builder
	present := map[string]bool{}
	for _, t := range batch {
		for _, c := range categories[t.TaskID] {
			if highErrorCategories[c] {
				present[c] = true
			}
		}
		if hc := heuristicCategory(t.Prompt); usesCodeExec(hc) {
			present[hc] = true
		}
	}
	for _, c := range []string{"mathematical_reasoning", "named_entity_recognition", "code_debugging", "code_generation", "logical_deductive_reasoning", "text_summarisation"} {
		if present[c] {
			recipe := categoryRecipe[c]
			// Folded batches have no run_python: the tool-centric recipes are
			// misleading there (and ~350 tokens of dead weight). Local lanes
			// answer most of these families anyway; the folded remnants are
			// gate rejects that need the short no-tools guidance instead.
			if !allowCode {
				if alt, ok := categoryRecipeNoTools[c]; ok {
					recipe = alt
				}
			}
			r.WriteString(recipe)
			r.WriteByte('\n')
		}
	}
	if allowCode {
		// Native tool surface: call run_python instead of fencing markdown. Maths
		// and logic both run here; the executed code (not the model) is authoritative,
		// and calling submit() ends the task with no extra formatting call.
		r.WriteString("tool run_python(code): Python 3 with the FULL stdlib (import freely - itertools, math, etc.). submit(answers=[{\"task_id\":..,\"answer\":..}, ...]) is ALREADY PRE-DEFINED by the runtime: call it once with ALL ids; NEVER write 'def submit'. CRITICAL: every number and name inside a submit answer MUST be interpolated from a computed VARIABLE (e.g. \"Alice drinks \"+A, or str(round(after_q3))+\" units\") - NEVER hand-type a value your code computed, because hand-typed values are usually wrong even when the code is right. maths: keep raw floats, round() only for display. logic: itertools.permutations over the candidates, keep the assignment satisfying EVERY clue, build the answer from the winning variables. 'what does this code evaluate to' tasks: PASTE the snippet into your code, capture its variables, and interpolate the REAL values into that answer - never trace by hand. Write terse code, NO comments, and do NOT print() anything - only call submit() (you get one run and never see stdout, so prints are wasted tokens). Trivia may skip the tool.\n")
	}
	return strings.TrimSpace(b.String()), strings.TrimSpace(r.String())
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
	// Telemetry payloads must stay KV-sized: swap base64 PNG data URIs for a
	// size note.
	for i := range out {
		if len(out[i].ImageURLs) == 0 {
			continue
		}
		ph := make([]string, len(out[i].ImageURLs))
		for j, u := range out[i].ImageURLs {
			ph[j] = fmt.Sprintf("png data URI omitted (%d bytes)", len(u))
		}
		out[i].ImageURLs = ph
	}
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
		out[i] = BatchPlanRecord{Index: i, TaskIDs: taskIDs(b), Size: len(b), Effort: a.effortForBatch(b), Model: a.model}
	}
	return out
}

// planBatches groups by effort and optionally isolates categories. Math must be
// isolated whenever it uses run_python; the other focus categories are
// selectable because batching is the dominant token-efficiency tradeoff.
// MathBatchSize (when >0) chunks only mathematical_reasoning groups independently of MaxBatchSize,
// so non-math can stay large (prompt amortisation) while multi-step math can run smaller.
func (a *Agent) planBatches(tasks []Task) [][]Task {
	maxN := a.cfg.MaxBatchSize
	if maxN <= 0 {
		maxN = 40
	}
	mathN := a.cfg.MathBatchSize
	if mathN <= 0 {
		mathN = maxN
	}
	type key struct {
		effort string
		code   bool
		focus  string
	}
	order := make([]key, 0)
	groups := map[key][]Task{}
	for _, t := range tasks {
		cat := a.primaryCat(t)
		focus := batchIsolationFocus(a.cfg.BatchIsolation, cat)
		k := key{effort: a.effortForTask(t), code: usesCodeExec(cat), focus: focus}
		if _, ok := groups[k]; !ok {
			order = append(order, k)
		}
		groups[k] = append(groups[k], t)
	}
	// Fold a tiny code group into a direct group: with the model lanes running
	// before batching, a code group is usually just 1-2 gate rejects, and a
	// dedicated batch pays ~1.1k tokens of code-exec scaffold for them. Losing
	// the run_python tool on 1-2 tasks costs less than that scaffold.
	fold := 2
	if v, err := strconv.Atoi(os.Getenv("AGENT_FOLD_CODE_REMAINDER")); err == nil {
		fold = v
	}
	if fold > 0 {
		var directKey *key
		for i := range order {
			if !order[i].code {
				directKey = &order[i]
				break
			}
		}
		if directKey != nil {
			kept := order[:0]
			for _, k := range order {
				if k.code && len(groups[k]) <= fold {
					groups[*directKey] = append(groups[*directKey], groups[k]...)
					delete(groups, k)
					continue
				}
				kept = append(kept, k)
			}
			order = kept
		}
	}
	var batches [][]Task
	for _, k := range order {
		g := groups[k]
		chunk := maxN
		if k.code {
			chunk = mathN
		}
		for i := 0; i < len(g); i += chunk {
			j := i + chunk
			if j > len(g) {
				j = len(g)
			}
			batches = append(batches, g[i:j])
		}
	}
	return batches
}

func batchIsolationFocus(mode, cat string) string {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "none", "off", "false", "0":
		return ""
	case "math", "maths":
		if cat == "mathematical_reasoning" {
			return cat
		}
		return ""
	default: // focus (default): isolate only logic, so it forms its own code-exec
		// batch separate from maths. NER/debugging now run reliably at effort
		// "none" via python3, so they merge into the main direct batch to save the
		// per-batch system-prompt overhead.
		switch cat {
		case "logical_deductive_reasoning":
			return cat
		default:
			return ""
		}
	}
}

func (a *Agent) primaryCat(t Task) string {
	if cs := a.categories[t.TaskID]; len(cs) > 0 {
		return cs[0]
	}
	return heuristicCategory(t.Prompt)
}

// usesCodeExec marks the categories solved via a run_python tool call: maths
// (arithmetic/projection) and logic (itertools constraint enumeration). For
// these the executed code does the reasoning, so the model runs at effort
// "none" and the tool output is authoritative - this removes the logic
// reasoning-token hog and is the main token saving.
func usesCodeExec(cat string) bool {
	if os.Getenv("AGENT_NO_CODE") != "" {
		return false // eval knob: answer everything directly (no run_python)
	}
	return cat == "mathematical_reasoning" || cat == "logical_deductive_reasoning"
}

// evalStyleCode detects 'what does this code evaluate to / what happens when'
// questions. Hand-tracing Python semantics gotchas (late binding, shared
// mutable defaults, iterator exhaustion) at effort none is unreliable -
// running the snippet is deterministic truth, so these join the code batch.
func evalStyleCode(lowerPrompt string) bool {
	if !strings.Contains(lowerPrompt, "code") && !strings.Contains(lowerPrompt, "def ") {
		return false
	}
	return strings.Contains(lowerPrompt, "evaluate to") ||
		strings.Contains(lowerPrompt, "what happens when") ||
		strings.Contains(lowerPrompt, "what does") || strings.Contains(lowerPrompt, "what do")
}

func (a *Agent) taskUsesCode(t Task) bool {
	if usesCodeExec(a.primaryCat(t)) {
		return true
	}
	if os.Getenv("AGENT_NO_CODE") == "" && evalStyleCode(strings.ToLower(t.Prompt)) {
		return true
	}
	// The classifier's first label over-fires (code_debugging noise) and can
	// bury or drop maths/logic entirely, which would silently route a
	// computation task away from code-exec - the accuracy-critical path. The
	// keyword heuristic is precise for these two categories (clues/km/h/%/...)
	// so it routes to code-exec on its own; a rare false positive just means
	// one extra task solved in the code batch, which the recipes handle.
	return usesCodeExec(heuristicCategory(t.Prompt))
}

// localTrainedFamily reports whether the prompt sits inside the local model's
// SFT coverage (maths/logic word problems). Code tasks pass the grounding gate
// whenever their snippet executes, but the weights were never tuned on that
// family and unseen-variant code_debugging answers judge at ~1/3 - those stay
// on the API. heuristicCategory checks code_debugging before maths/logic, so
// code-shaped prompts cannot leak in via a stray '%' or 'how many'.
func localTrainedFamily(prompt string) bool {
	switch heuristicCategory(prompt) {
	case "mathematical_reasoning", "logical_deductive_reasoning":
		return true
	}
	return false
}

// batchAllowsCode attaches the run_python surface only to DEDICATED code
// batches (every task code-eligible). A folded batch - direct tasks plus 1-2
// gate rejects - must stay text-only: offered the tool, the model reaches for
// it on a coin flip, and the second turn re-pays the whole grown context
// (observed twice: +~2,600 tokens for one logic straggler). The no-tools
// recipes carry those remnants instead.
func (a *Agent) batchAllowsCode(batch []Task) bool {
	if len(batch) == 0 {
		return false
	}
	for _, t := range batch {
		if !a.taskUsesCode(t) {
			return false
		}
	}
	return true
}

// effortForTask resolves reasoning effort for one task.
// Global cfg.ReasoningEffort overrides when set (including "none"/"off").
func (a *Agent) effortForTask(t Task) string {
	if e := strings.TrimSpace(a.cfg.ReasoningEffort); e != "" && !strings.EqualFold(e, "auto") {
		return normaliseEffort(e)
	}
	cat := a.primaryCat(t)
	if a.cfg.EffortTierMap != nil {
		if e, ok := a.cfg.EffortTierMap[cat]; ok {
			return normaliseEffort(e)
		}
	}
	return "none"
}

func (a *Agent) effortForBatch(batch []Task) string {
	if len(batch) == 0 {
		return "none"
	}
	return a.effortForTask(batch[0])
}

func normaliseEffort(e string) string {
	e = strings.ToLower(strings.TrimSpace(e))
	switch e {
	case "", "off", "none", "0", "false", "disabled":
		return "none"
	case "low", "medium", "high", "xhigh":
		return e
	default:
		return "low"
	}
}

// wireEffort converts internal effort to Fireworks reasoning_effort.
// Fireworks requires the literal value "none" to disable thinking; omitting
// the field leaves adaptive/default thinking on (still burns reasoning tokens).
func wireEffort(effort string) string {
	if effort == "" {
		return "none"
	}
	return effort
}

func maxTokensForBatch(batch []Task, turn int, effort string, allowCode bool) int {
	// Enough room for all short answers in one JSON object.
	base := 600 + len(batch)*220
	switch normaliseEffort(effort) {
	case "medium":
		if base < 3200 {
			base = 3200
		}
	case "high":
		if base < 4500 {
			base = 4500
		}
	case "xhigh":
		// Reasoning tokens consume the completion budget before the final JSON.
		// A multi-question deduction batch otherwise truncates into the generic
		// fallback even when the model has solved the questions internally.
		if base < 6000 {
			base = 6000
		}
	}
	if turn > 0 {
		base = base * 5 / 4
	}
	if allowCode && turn == 0 && base < 3072 {
		base = 3072
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
