package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/ashaibani/yassai/internal/agent"
	"github.com/ashaibani/yassai/internal/validate"
)

type EvalResult struct {
	Strategy  string         `json:"strategy"`
	BatchSize int            `json:"batch_size"`
	Metrics   agent.Metrics  `json:"metrics"`
	Results   []agent.Result `json:"results"`
	DurationS float64        `json:"duration_s"`
}

type Comparison struct {
	Model      string       `json:"model"`
	Results    []EvalResult `json:"results"`
	BestTokens string       `json:"best_tokens"`
	BestSpeed  string       `json:"best_speed"`
}

func main() {
	tasksPath := getenv("TASKS_PATH", "testdata/tasks.comprehensive.json")
	outputDir := getenv("EVAL_OUTPUT_DIR", "./eval-results")
	apiKey := os.Getenv("FIREWORKS_API_KEY")
	baseURL := getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
	models := splitCSV(getenv("ALLOWED_MODELS", "accounts/fireworks/models/minimax-m3"))
	if len(models) == 0 {
		models = []string{"accounts/fireworks/models/minimax-m3"}
	}
	batchSizes := []int{1, 5, 10, 20}
	if v := os.Getenv("BATCH_SIZES"); v != "" {
		batchSizes = parseIntSlice(v)
	}

	tasks, err := readTasks(tasksPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	if len(tasks) == 0 {
		fmt.Fprintln(os.Stderr, "no tasks found")
		os.Exit(1)
	}

	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "FIREWORKS_API_KEY is required")
		os.Exit(1)
	}

	_ = os.MkdirAll(outputDir, 0o755)

	for _, model := range models {
		comp := Comparison{Model: model}
		for _, bs := range batchSizes {
			strategy := fmt.Sprintf("batch-%d", bs)
			cfg := agent.Config{
				APIKey:           apiKey,
				BaseURL:          baseURL,
				AllowedModels:    []string{model},
				PreferredModel:   model,
				MaxBatchSize:     bs,
				MaxContextTokens: 200000,
				MemoryRoot:       getenv("AGENT_MEMORY_ROOT", defaultConfigDir()),
				Timeout:          120 * time.Second,
			}
			ag, err := agent.New(cfg)
			if err != nil {
				fmt.Fprintf(os.Stderr, "agent.New error for %s batch %d: %v\n", model, bs, err)
				continue
			}
			ctx, cancel := context.WithTimeout(context.Background(), 9*time.Minute+30*time.Second)
			start := time.Now()
			results, metrics, err := ag.Solve(ctx, tasks)
			dur := time.Since(start).Seconds()
			cancel()
			if err != nil {
				fmt.Fprintf(os.Stderr, "Solve error for %s batch %d: %v\n", model, bs, err)
			}
			er := EvalResult{
				Strategy:  strategy,
				BatchSize: bs,
				Metrics:   metrics,
				Results:   results,
				DurationS: dur,
			}
			comp.Results = append(comp.Results, er)
			// Print summary line
			fmt.Printf("%-35s batch=%-3d tokens=%-8d cached=%-8d calls=%-3d time=%.1fs\n",
				model, bs, metrics.TotalTokens, metrics.CachedTokens, metrics.Calls, dur)
			// Validate against golden answers
			goldenPath := getenv("GOLDEN_PATH", "testdata/golden.json")
			cases, gerr := validate.LoadCases(goldenPath)
			if gerr != nil {
				fmt.Fprintf(os.Stderr, "load golden: %v\n", gerr)
			} else {
				valResults := make([]validate.Result, len(results))
				for i, r := range results {
					for _, c := range cases {
						if c.TaskID == r.TaskID {
							valResults[i] = validate.Check(r.Answer, c)
							break
						}
					}
				}
				report := validate.Score(valResults)
				fmt.Printf("    accuracy: %d/%d (%.1f%%)", report.Passed, report.Total, report.PassRate*100)
				for cat, cs := range report.ByCategory {
					fmt.Printf("  %s=%d/%d", cat, cs.Passed, cs.Total)
				}
				fmt.Println()
				valFile := filepath.Join(outputDir, fmt.Sprintf("%s-batch-%d-validation.json", sanitize(model), bs))
				_ = writeJSON(valFile, report)
			}

			// Save per-strategy metrics
			metricsFile := filepath.Join(outputDir, fmt.Sprintf("%s-batch-%d-metrics.json", sanitize(model), bs))
			_ = writeJSON(metricsFile, metrics)
			resultsFile := filepath.Join(outputDir, fmt.Sprintf("%s-batch-%d-results.json", sanitize(model), bs))
			_ = writeJSON(resultsFile, results)
		}
		// Determine best by tokens and by speed
		if len(comp.Results) > 0 {
			byTokens := append([]EvalResult(nil), comp.Results...)
			sort.Slice(byTokens, func(i, j int) bool { return byTokens[i].Metrics.TotalTokens < byTokens[j].Metrics.TotalTokens })
			comp.BestTokens = fmt.Sprintf("batch-%d (%d tokens)", byTokens[0].BatchSize, byTokens[0].Metrics.TotalTokens)
			bySpeed := append([]EvalResult(nil), comp.Results...)
			sort.Slice(bySpeed, func(i, j int) bool { return bySpeed[i].DurationS < bySpeed[j].DurationS })
			comp.BestSpeed = fmt.Sprintf("batch-%d (%.1fs)", bySpeed[0].BatchSize, bySpeed[0].DurationS)
			compFile := filepath.Join(outputDir, fmt.Sprintf("%s-comparison.json", sanitize(model)))
			_ = writeJSON(compFile, comp)
			fmt.Printf("\nBest by tokens: %s | Best by speed: %s\n\n", comp.BestTokens, comp.BestSpeed)
		}
	}
}

func readTasks(path string) ([]agent.Task, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tasks %s: %w", path, err)
	}
	var tasks []agent.Task
	if err := json.Unmarshal(b, &tasks); err != nil {
		return nil, fmt.Errorf("parse tasks %s: %w", path, err)
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

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func splitCSV(s string) []string {
	if s == "" {
		return nil
	}
	var out []string
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func parseIntSlice(s string) []int {
	var out []int
	for _, p := range strings.Split(s, ",") {
		var n int
		fmt.Sscanf(strings.TrimSpace(p), "%d", &n)
		if n > 0 {
			out = append(out, n)
		}
	}
	return out
}

func sanitize(s string) string {
	r := strings.NewReplacer("/", "-", "\\", "-", ":", "-")
	return r.Replace(s)
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
