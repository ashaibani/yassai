package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ashaibani/yassai/internal/agent"
	"github.com/ashaibani/yassai/internal/callback"
)

const (
	defaultInputPath  = "/input/tasks.json"
	defaultOutputPath = "/output/results.json"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run() error {
	loadLocalEnv()

	inputPath := getenv("TASKS_PATH", defaultInputPath)
	outputPath := getenv("RESULTS_PATH", defaultOutputPath)

	rawTasks, err := os.ReadFile(inputPath)
	if err != nil {
		return fmt.Errorf("read tasks %s: %w", inputPath, err)
	}
	tasks, err := parseTasks(rawTasks)
	if err != nil {
		return err
	}
	if len(tasks) == 0 {
		if err := writeJSON(outputPath, []agent.Result{}); err != nil {
			return err
		}
		callback.Report(rawTasks, []agent.Result{}, nil, callbackLabel())
		return nil
	}

	cfg := agent.Config{
		APIKey:           os.Getenv("FIREWORKS_API_KEY"),
		BaseURL:          normaliseBaseURL(getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")),
		AllowedModels:    splitCSV(os.Getenv("ALLOWED_MODELS")),
		PreferredModel:   os.Getenv("AGENT_MODEL"),
		MaxBatchSize:     getenvInt("AGENT_BATCH_SIZE", 20),
		MaxBatchTokens:   getenvInt("AGENT_BATCH_TOKENS", 12000),
		MaxConcurrency:   getenvInt("AGENT_MAX_CONCURRENCY", 3),
		ReasoningEffort:  os.Getenv("AGENT_REASONING_EFFORT"), // "" = adaptive reasoning by category tier (effortForTask)
		MaxContextTokens: getenvInt("AGENT_CONTEXT_TOKENS", 200000),
		MemoryRoot:       getenv("AGENT_MEMORY_ROOT", "."),
		SkillRoots:       splitCSV(os.Getenv("AGENT_SKILL_ROOTS")),
		Timeout:          time.Duration(getenvInt("LLM_TIMEOUT_SECONDS", 180)) * time.Second,
		ClassifierDir:    getenv("TASKCLF_DIR", "assets/taskclf"),
		ClassifierLib:    os.Getenv("ONNXRUNTIME_LIB"),
		TraceMessages:    envBool("AGENT_TRACE_MESSAGES", true),
		DisableLocal:     envBool("AGENT_DISABLE_LOCAL", false),
	}

	ag, err := agent.New(cfg)
	if err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 9*time.Minute+30*time.Second)
	defer cancel()

	results, metrics, err := ag.Solve(ctx, tasks)
	if err != nil {
		// Still report the input tasks so we can see what was sent, even on failure.
		callback.Report(rawTasks, results, metrics, callbackLabel())
		return err
	}
	if err := writeJSON(outputPath, results); err != nil {
		return err
	}
	metricsPath := filepath.Join(filepath.Dir(outputPath), "metrics.json")
	_ = writeJSON(metricsPath, metrics)

	// Best-effort: report input tasks + output results + metrics to the
	// benchmark-callback worker so we can evaluate without re-submitting.
	// Controlled by CALLBACK_URL env var (default = deployed worker; set to
	// "off" to disable). Never fails the run.
	callback.Report(rawTasks, results, metrics, callbackLabel())

	return nil
}

func parseTasks(raw []byte) ([]agent.Task, error) {
	var tasks []agent.Task
	if err := json.Unmarshal(raw, &tasks); err != nil {
		return nil, fmt.Errorf("parse tasks: %w", err)
	}
	for i := range tasks {
		tasks[i].TaskID = strings.TrimSpace(tasks[i].TaskID)
		if tasks[i].TaskID == "" {
			return nil, fmt.Errorf("task at index %d has empty task_id", i)
		}
	}
	return tasks, nil
}

func writeJSON(path string, v any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	return os.WriteFile(path, b, 0o644)
}

func defaultConfigDir() string {
	if v := os.Getenv("YASSAI_HOME"); v != "" {
		return v
	}
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".yassai")
	}
	return ".yassai"
}

func loadLocalEnv() {
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
			key = strings.TrimSpace(strings.TrimPrefix(key, "export ")) // support `export KEY=val` .env lines
			value := strings.Trim(strings.TrimSpace(parts[1]), "'\"")
			if key != "" && os.Getenv(key) == "" {
				_ = os.Setenv(key, value)
			}
		}
	}
}

func getenv(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

func envBool(key string, fallback bool) bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv(key)))
	if v == "" {
		return fallback
	}
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func getenvInt(key string, fallback int) int {
	var v int
	if _, err := fmt.Sscanf(os.Getenv(key), "%d", &v); err == nil && v > 0 {
		return v
	}
	return fallback
}

func splitCSV(s string) []string {
	var out []string
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func normaliseBaseURL(base string) string {
	base = strings.TrimRight(strings.TrimSpace(base), "/")
	for _, suffix := range []string{"/chat/completions", "/completions"} {
		if strings.HasSuffix(base, suffix) {
			return strings.TrimRight(strings.TrimSuffix(base, suffix), "/")
		}
	}
	return base
}

// callbackLabel builds a human-readable label for the callback dashboard.
func callbackLabel() string {
	if l := os.Getenv("CALLBACK_LABEL"); l != "" {
		return l
	}
	host, _ := os.Hostname()
	model := os.Getenv("AGENT_MODEL")
	if model == "" {
		model = "default"
	}
	if host == "" {
		return model
	}
	return model + " @ " + host
}
