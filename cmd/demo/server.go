package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/ashaibani/yassai/internal/agent"
	"github.com/ashaibani/yassai/internal/judge"
	"github.com/ashaibani/yassai/internal/validate"
)

type Server struct {
	mu            sync.Mutex
	runs          map[string]*RunState
	apiKey        string
	baseURL       string
	effortTierMap map[string]string
	modelRouteMap map[string]string
}

type RunState struct {
	ID           string                 `json:"id"`
	Status       string                 `json:"status"`
	StartedAt    time.Time              `json:"started_at"`
	Events       []agent.Event          `json:"events"`
	Results      []agent.Result         `json:"results,omitempty"`
	Metrics      agent.Metrics          `json:"metrics,omitempty"`
	Config       RunConfig              `json:"config"`
	Tasks        []agent.Task           `json:"tasks"`
	JudgeResults map[string]JudgeResult `json:"judge_results,omitempty"`
	mu           sync.Mutex
}

type RunConfig struct {
	Model           string `json:"model"`
	BatchSize       int    `json:"batch_size"`
	BatchTokens     int    `json:"batch_tokens"`
	MaxConcurrency  int    `json:"max_concurrency"`
	ReasoningEffort string `json:"reasoning_effort"`
	DisableHints    bool   `json:"disable_hints"`
	EnableJudge     bool   `json:"enable_judge"`
	JudgeModel      string `json:"judge_model"`
	JudgeEffort     string `json:"judge_effort"`
}

type JudgeResult struct {
	Pass   bool   `json:"pass"`
	Reason string `json:"reason"`
	Via    string `json:"via"`
}

type runRequest struct {
	TasksFile        string            `json:"tasks_file"`
	Model            string            `json:"model"`
	BatchSize        int               `json:"batch_size"`
	BatchTokens      int               `json:"batch_tokens"`
	MaxConcurrency   int               `json:"max_concurrency"`
	ReasoningEffort  string            `json:"reasoning_effort"`
	DisableHints     bool              `json:"disable_hints"`
	EnableClassifier bool              `json:"enable_classifier"`
	EnableJudge      bool              `json:"enable_judge"`
	JudgeModel       string            `json:"judge_model"`
	JudgeEffort      string            `json:"judge_effort"`
	EffortTierMap    map[string]string `json:"effort_tier_map"`
	ModelRouteMap    map[string]string `json:"model_route_map"`
}

func main() {
	loadEnv()
	allowedModels := splitAllowedModels(os.Getenv("ALLOWED_MODELS"))
	srv := &Server{
		runs:          make(map[string]*RunState),
		apiKey:        os.Getenv("FIREWORKS_API_KEY"),
		baseURL:       getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"),
		effortTierMap: agent.DefaultEffortTier(),
		modelRouteMap: agent.DefaultModelRouteMap(allowedModels),
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/", srv.handleIndex)
	mux.HandleFunc("/api/tasks", srv.handleTasks)
	mux.HandleFunc("/api/run", srv.handleRun)
	mux.HandleFunc("/api/run/", srv.handleRunDetail)
	mux.HandleFunc("/api/classify", srv.handleClassify)
	mux.HandleFunc("/api/routing", srv.handleRouting)
	mux.HandleFunc("/api/models", srv.handleModels)
	mux.HandleFunc("/app.js", srv.handleAppJS)
	port := getenv("PORT", "7070")
	fmt.Printf("yassai demo server on http://localhost:%s\n", port)
	if err := http.ListenAndServe(":"+port, mux); err != nil {
		fmt.Fprintln(os.Stderr, "server error:", err)
		os.Exit(1)
	}
}

func (s *Server) handleAppJS(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/javascript")
	w.Write([]byte(appJS))
}

func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte(indexHTML))
}

func (s *Server) handleTasks(w http.ResponseWriter, r *http.Request) {
	td := getenv("TASKS_PATH", "testdata")
	files, _ := filepath.Glob(filepath.Join(td, "*.json"))
	type taskSet struct {
		File        string `json:"file"`
		Name        string `json:"name"`
		Count       int    `json:"count"`
		HasExpected bool   `json:"has_expected"`
	}
	var sets []taskSet
	for _, f := range files {
		base := filepath.Base(f)
		if base == "golden.json" {
			continue
		}
		b, _ := os.ReadFile(f)
		var arr []map[string]any
		json.Unmarshal(b, &arr)
		count := len(arr)
		hasExp := false
		if count > 0 {
			_, hasExp = arr[0]["expected"]
		}
		sets = append(sets, taskSet{File: base, Name: strings.TrimSuffix(base, ".json"), Count: count, HasExpected: hasExp})
	}
	writeJSON(w, sets)
}

func (s *Server) handleRun(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST required", 405)
		return
	}
	var req runRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	tasksPath := filepath.Join(getenv("TASKS_PATH", "testdata"), req.TasksFile)
	tasks, cases, err := loadTasksWithCases(tasksPath)
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	if len(tasks) == 0 {
		http.Error(w, "no tasks", 400)
		return
	}
	model := req.Model
	if model == "" {
		model = s.defaultModel()
	}
	if req.BatchSize <= 0 {
		req.BatchSize = 20
	}
	if req.BatchTokens <= 0 {
		req.BatchTokens = 12000
	}
	classifierDir := ""
	classifierLib := getenv("ONNXRUNTIME_LIB", "")
	if req.EnableClassifier {
		classifierDir = getenv("TASKCLF_DIR", "assets/taskclf")
	}
	runID := fmt.Sprintf("run-%d", time.Now().UnixMilli())
	rs := &RunState{
		ID:        runID,
		Status:    "running",
		StartedAt: time.Now(),
		Config: RunConfig{
			Model:           model,
			BatchSize:       req.BatchSize,
			BatchTokens:     req.BatchTokens,
			MaxConcurrency:  req.MaxConcurrency,
			ReasoningEffort: req.ReasoningEffort,
			DisableHints:    req.DisableHints,
			EnableJudge:     req.EnableJudge,
			JudgeModel:      req.JudgeModel,
			JudgeEffort:     req.JudgeEffort,
		},
		Tasks: tasks,
	}
	s.mu.Lock()
	s.runs[runID] = rs
	s.mu.Unlock()
	// Merge server-level default routing with request-level overrides
	s.mu.Lock()
	if req.EffortTierMap == nil {
		req.EffortTierMap = s.effortTierMap
	}
	if req.ModelRouteMap == nil {
		req.ModelRouteMap = s.modelRouteMap
	}
	s.mu.Unlock()
	go s.executeRun(runID, rs, tasks, model, req, cases, classifierDir, classifierLib)
	writeJSON(w, map[string]any{"run_id": runID, "task_count": len(tasks)})
}

func (s *Server) executeRun(runID string, rs *RunState, tasks []agent.Task, model string, req runRequest, cases []validate.Case, classifierDir, classifierLib string) {
	defer func() {
		if r := recover(); r != nil {
			rs.mu.Lock()
			rs.Status = "error"
			rs.mu.Unlock()
			fmt.Fprintf(os.Stderr, "run %s panic: %v\n", runID, r)
		}
	}()
	// Collect all allowed models from both the default and routing map
	allowedModels := []string{model}
	for _, m := range req.ModelRouteMap {
		found := false
		for _, existing := range allowedModels {
			if existing == m {
				found = true
				break
			}
		}
		if !found {
			allowedModels = append(allowedModels, m)
		}
	}
	cfg := agent.Config{
		APIKey:           s.apiKey,
		BaseURL:          s.baseURL,
		AllowedModels:    allowedModels,
		PreferredModel:   model,
		MaxBatchSize:     req.BatchSize,
		MaxBatchTokens:   req.BatchTokens,
		MaxConcurrency:   req.MaxConcurrency,
		ReasoningEffort:  req.ReasoningEffort,
		DisableHints:     req.DisableHints,
		MaxContextTokens: 200000,
		MemoryRoot:       ".",
		Timeout:          180 * time.Second,
		ClassifierDir:    classifierDir,
		ClassifierLib:    classifierLib,
		EffortTierMap:    req.EffortTierMap,
		ModelRouteMap:    req.ModelRouteMap,
	}
	ag, err := agent.New(cfg)
	if err != nil {
		rs.mu.Lock()
		rs.Status = "error"
		rs.Events = append(rs.Events, agent.Event{Type: "error", Timestamp: time.Now(), Data: map[string]any{"message": err.Error()}})
		rs.mu.Unlock()
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 9*time.Minute+30*time.Second)
	defer cancel()
	cb := func(ev agent.Event) {
		rs.mu.Lock()
		rs.Events = append(rs.Events, ev)
		rs.mu.Unlock()
	}
	results, metrics, err := ag.SolveWithEvents(ctx, tasks, cb)
	rs.mu.Lock()
	rs.Results = results
	rs.Metrics = metrics
	if err != nil {
		rs.Status = "error"
		rs.Events = append(rs.Events, agent.Event{Type: "error", Timestamp: time.Now(), Data: map[string]any{"message": err.Error()}})
	} else if len(cases) > 0 && rs.Config.EnableJudge {
		rs.Status = "judging"
	} else {
		rs.Status = "done"
	}
	rs.mu.Unlock()
	if len(cases) > 0 && err == nil && rs.Config.EnableJudge {
		s.autoValidate(runID, results, cases)
		rs.mu.Lock()
		rs.Status = "done"
		rs.mu.Unlock()
	}
}

func (s *Server) autoValidate(runID string, results []agent.Result, cases []validate.Case) {
	rs := s.runs[runID]
	if rs == nil {
		return
	}
	if !rs.Config.EnableJudge {
		return
	}
	caseByID := map[string]validate.Case{}
	for _, c := range cases {
		caseByID[c.TaskID] = c
	}
	judgeResults := make(map[string]JudgeResult)
	// Use the same Fireworks API key + base URL for the judge (no umans dependency).
	// The judge model defaults to the agent model if not specified.
	judgeKey := s.apiKey
	judgeBase := s.baseURL
	judgeModel := rs.Config.JudgeModel
	if judgeModel == "" {
		judgeModel = rs.Config.Model
	}
	judgeEffort := rs.Config.JudgeEffort
	if judgeEffort == "" {
		judgeEffort = "xhigh"
	}
	jd := judge.New(judgeKey, judgeBase, judgeModel, judgeEffort)
	sem := make(chan struct{}, 3)
	var wg sync.WaitGroup
	var mu sync.Mutex
	for _, r := range results {
		c, ok := caseByID[r.TaskID]
		if !ok {
			continue
		}
		if c.Validate == "llm" {
			wg.Add(1)
			go func(r agent.Result, c validate.Case) {
				defer wg.Done()
				sem <- struct{}{}
				defer func() { <-sem }()
				pass, verdict, jerr := jd.Grade(context.Background(), c.Prompt, c.Expected, r.Answer)
				jr := JudgeResult{Via: "judge"}
				if jerr != nil {
					jr.Reason = "ERR: " + jerr.Error()
				} else {
					jr.Pass, jr.Reason = pass, verdict
				}
				mu.Lock()
				judgeResults[r.TaskID] = jr
				mu.Unlock()
				rs.mu.Lock()
				rs.JudgeResults = judgeResults
				rs.mu.Unlock()
			}(r, c)
		} else {
			res := validate.Check(r.Answer, c)
			judgeResults[r.TaskID] = JudgeResult{Pass: res.Pass, Reason: res.Reason, Via: "check"}
			rs.mu.Lock()
			rs.JudgeResults = judgeResults
			rs.mu.Unlock()
		}
	}
	wg.Wait()
}

func (s *Server) handleRunDetail(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/run/"), "/")
	runID := parts[0]
	s.mu.Lock()
	rs, ok := s.runs[runID]
	s.mu.Unlock()
	if !ok {
		http.Error(w, "run not found", 404)
		return
	}
	if len(parts) > 1 && parts[1] == "events" {
		s.handleSSE(w, r, rs)
		return
	}
	rs.mu.Lock()
	defer rs.mu.Unlock()
	writeJSON(w, rs)
}

func (s *Server) handleSSE(w http.ResponseWriter, r *http.Request, rs *RunState) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", 500)
		return
	}
	sent := 0
	for {
		rs.mu.Lock()
		total := len(rs.Events)
		status := rs.Status
		rs.mu.Unlock()
		for i := sent; i < total; i++ {
			rs.mu.Lock()
			ev := rs.Events[i]
			rs.mu.Unlock()
			data, _ := json.Marshal(ev)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}
		sent = total
		if status != "running" && status != "judging" {
			rs.mu.Lock()
			final := map[string]any{"type": "final", "status": status, "results": rs.Results, "metrics": rs.Metrics, "judge_results": rs.JudgeResults}
			rs.mu.Unlock()
			data, _ := json.Marshal(final)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
			return
		}
		// While judging, send periodic judge progress updates
		if status == "judging" {
			rs.mu.Lock()
			jrCount := len(rs.JudgeResults)
			jrCopy := make(map[string]JudgeResult, jrCount)
			for k, v := range rs.JudgeResults {
				jrCopy[k] = v
			}
			rs.mu.Unlock()
			if jrCount > 0 {
				update := map[string]any{"type": "judge_progress", "judge_results": jrCopy, "count": jrCount}
				data, _ := json.Marshal(update)
				fmt.Fprintf(w, "data: %s\n\n", data)
				flusher.Flush()
			}
		}
		select {
		case <-r.Context().Done():
			return
		case <-time.After(200 * time.Millisecond):
		}
	}
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	models := s.allowedModels()
	writeJSON(w, map[string]any{"models": models, "default": s.defaultModel()})
}

func splitAllowedModels(raw string) []string {
	if raw == "" {
		return []string{
			"accounts/fireworks/models/minimax-m3",
			"accounts/fireworks/models/kimi-k2p7-code",
		}
	}
	var out []string
	for _, m := range strings.Split(raw, ",") {
		m = strings.TrimSpace(m)
		if m != "" {
			out = append(out, m)
		}
	}
	return out
}

func (s *Server) allowedModels() []string {
	return splitAllowedModels(os.Getenv("ALLOWED_MODELS"))
}

func (s *Server) defaultModel() string {
	models := s.allowedModels()
	// Prefer minimax-m3 if present; otherwise fall back to the first available.
	for _, m := range models {
		if m == "accounts/fireworks/models/minimax-m3" {
			return m
		}
	}
	if len(models) > 0 {
		return models[0]
	}
	return "accounts/fireworks/models/minimax-m3"
}

func (s *Server) handleClassify(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST required", 405)
		return
	}
	var req struct {
		Prompt string `json:"prompt"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	if req.Prompt == "" {
		http.Error(w, "prompt required", 400)
		return
	}
	clf, err := newClassifier(getenv("TASKCLF_DIR", "assets/taskclf"), getenv("ONNXRUNTIME_LIB", ""))
	if err != nil {
		writeJSON(w, map[string]any{"error": err.Error(), "labels": []string{}})
		return
	}
	preds, _ := clf.Classify(req.Prompt)
	type predOut struct {
		Label string  `json:"label"`
		Score float64 `json:"score"`
	}
	out := make([]predOut, len(preds))
	for i, p := range preds {
		out[i] = predOut{Label: p.Label, Score: float64(p.Score)}
	}
	writeJSON(w, map[string]any{"predictions": out})
}

func loadTasksWithCases(path string) ([]agent.Task, []validate.Case, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	var raw []map[string]any
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, nil, err
	}
	var tasks []agent.Task
	var cases []validate.Case
	for _, item := range raw {
		id, _ := item["task_id"].(string)
		if id == "" {
			continue
		}
		prompt, _ := item["prompt"].(string)
		tasks = append(tasks, agent.Task{TaskID: id, Prompt: prompt})
		if exp, ok := item["expected"].(string); ok && exp != "" {
			val, _ := item["validate"].(string)
			if val == "" {
				val = "contains_ci"
			}
			cases = append(cases, validate.Case{TaskID: id, Prompt: prompt, Expected: exp, Validate: val})
		}
	}
	return tasks, cases, nil
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

func getenv(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

func loadEnv() {
	paths := []string{".env", filepath.Join(os.Getenv("HOME"), "config", ".env")}
	for _, path := range paths {
		b, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		for _, line := range strings.Split(string(b), "\n") {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "#") || !strings.Contains(line, "=") {
				continue
			}
			parts := strings.SplitN(line, "=", 2)
			key := strings.TrimSpace(parts[0])
			value := strings.Trim(strings.TrimSpace(parts[1]), "'\"")
			if key != "" && os.Getenv(key) == "" {
				os.Setenv(key, value)
			}
		}
	}
}

func (s *Server) handleRouting(w http.ResponseWriter, r *http.Request) {
	if r.Method == "POST" {
		var req struct {
			EffortTierMap map[string]string `json:"effort_tier_map"`
			ModelRouteMap map[string]string `json:"model_route_map"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		s.mu.Lock()
		if req.EffortTierMap != nil {
			s.effortTierMap = req.EffortTierMap
		}
		if req.ModelRouteMap != nil {
			s.modelRouteMap = req.ModelRouteMap
		}
		s.mu.Unlock()
	}
	s.mu.Lock()
	resp := map[string]any{
		"effort_tier_map": s.effortTierMap,
		"model_route_map": s.modelRouteMap,
		"categories": []string{
			"factual_knowledge", "mathematical_reasoning", "sentiment_classification",
			"text_summarisation", "named_entity_recognition", "code_debugging",
			"logical_deductive_reasoning", "code_generation",
		},
		"effort_levels": []string{"low", "medium", "high", "xhigh"},
		"models":        s.allowedModels(),
	}
	s.mu.Unlock()
	writeJSON(w, resp)
}
