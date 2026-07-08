// Command routeprobe measures per-(model, category) accuracy and token cost on
// the golden set, to decide whether category-aware model routing is worth it.
//
// For each model in ALLOWED_MODELS it groups the golden tasks by their true
// category (task_id prefix), runs each category group as a single batch, and
// records gate-proxy accuracy + tokens (total/output/reasoning). The classifier
// is intentionally disabled so we measure raw model behaviour per category.
//
// Usage:
//
//	FIREWORKS_API_KEY=... ALLOWED_MODELS="a,b" go run ./cmd/routeprobe
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/ashaibani/yassai/internal/agent"
	"github.com/ashaibani/yassai/internal/validate"
)

func categoryOf(id string) string {
	switch {
	case strings.HasPrefix(id, "su"):
		return "summarisation"
	case strings.HasPrefix(id, "cd"):
		return "code_debugging"
	case strings.HasPrefix(id, "cg"):
		return "code_generation"
	case strings.HasPrefix(id, "f"):
		return "factual"
	case strings.HasPrefix(id, "m"):
		return "maths"
	case strings.HasPrefix(id, "s"):
		return "sentiment"
	case strings.HasPrefix(id, "n"):
		return "ner"
	case strings.HasPrefix(id, "l"):
		return "logical_reasoning"
	default:
		return "unknown"
	}
}

var highEffortCats = map[string]bool{"maths": true, "logical_reasoning": true, "code_debugging": true, "code_generation": true}

// tier maps a (probe) category to its reasoning tier.
func tier(cat string) string {
	if highEffortCats[cat] {
		return "high"
	}
	return "low"
}

type cell struct {
	Model           string  `json:"model"`
	Category        string  `json:"category"`
	Effort          string  `json:"effort"`
	Pass            int     `json:"pass"`
	Total           int     `json:"total"`
	TotalTokens     int     `json:"total_tokens"`
	OutputTokens    int     `json:"output_tokens"`
	ReasoningTokens int     `json:"reasoning_tokens"`
	PromptTokens    int     `json:"prompt_tokens"`
	Calls           int     `json:"calls"`
	DurationS       float64 `json:"duration_s"`
	Err             string  `json:"err,omitempty"`
}

func main() {
	goldenPath := getenv("GOLDEN_PATH", "testdata/golden.json")
	apiKey := os.Getenv("FIREWORKS_API_KEY")
	baseURL := getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
	models := splitCSV(getenv("ALLOWED_MODELS",
		"accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code"))
	mode := getenv("EFFORT_MODE", "") // "", "low", "medium", "high", "adaptive"
	fmt.Printf("effort mode: %q\n", mode)
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "FIREWORKS_API_KEY is required")
		os.Exit(1)
	}

	cases, err := validate.LoadCases(goldenPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	caseByID := map[string]validate.Case{}
	groups := map[string][]agent.Task{}
	var order []string
	seen := map[string]bool{}
	for _, c := range cases {
		caseByID[c.TaskID] = c
		cat := categoryOf(c.TaskID)
		if !seen[cat] {
			seen[cat] = true
			order = append(order, cat)
		}
		groups[cat] = append(groups[cat], agent.Task{TaskID: c.TaskID, Prompt: c.Prompt})
	}
	sort.Strings(order)

	tmpMem, _ := os.MkdirTemp("", "routeprobe-mem-")
	defer os.RemoveAll(tmpMem)

	var grid []cell
	for _, model := range models {
		fmt.Printf("\n=== %s ===\n", short(model))
		for _, cat := range order {
			tasks := groups[cat]
			eff := ""
			switch mode {
			case "adaptive":
				eff = tier(cat)
			case "low", "medium", "high":
				eff = mode
			}
			cfg := agent.Config{
				APIKey:           apiKey,
				BaseURL:          baseURL,
				AllowedModels:    []string{model},
				PreferredModel:   model,
				MaxBatchSize:     len(tasks), // whole category as one batch
				MaxContextTokens: 200000,
				MemoryRoot:       tmpMem,
				Timeout:          120 * time.Second,
				ReasoningEffort:  eff,
			}
			c := cell{Model: model, Category: cat, Total: len(tasks), Effort: eff}
			ag, nerr := agent.New(cfg)
			if nerr != nil {
				c.Err = nerr.Error()
				grid = append(grid, c)
				fmt.Printf("  %-18s NEW ERROR: %v\n", cat, nerr)
				continue
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
			start := time.Now()
			results, metrics, serr := ag.Solve(ctx, tasks)
			c.DurationS = time.Since(start).Seconds()
			cancel()
			if serr != nil {
				c.Err = serr.Error()
			}
			for _, r := range results {
				if validate.Check(r.Answer, caseByID[r.TaskID]).Pass {
					c.Pass++
				}
			}
			c.TotalTokens, c.OutputTokens, c.ReasoningTokens, c.PromptTokens, c.Calls =
				metrics.TotalTokens, metrics.OutputTokens, metrics.ReasoningTokens, metrics.PromptTokens, metrics.Calls
			grid = append(grid, c)
			fmt.Printf("  %-18s eff=%-4s acc=%d/%d  tok=%-6d out=%-6d reason=%-6d calls=%d  %.1fs%s\n",
				cat, iff(c.Effort == "", "def", c.Effort), c.Pass, c.Total, c.TotalTokens, c.OutputTokens, c.ReasoningTokens, c.Calls, c.DurationS,
				iff(c.Err != "", "  ERR:"+c.Err, ""))
		}
	}

	_ = os.MkdirAll("eval-results", 0o755)
	b, _ := json.MarshalIndent(grid, "", "  ")
	_ = os.WriteFile("eval-results/routeprobe.json", append(b, '\n'), 0o644)

	// Per-category winner (accuracy first, then fewer tokens) and totals.
	fmt.Printf("\n=== ROUTING MAP (winner = higher acc, then fewer tokens) ===\n")
	byCat := map[string][]cell{}
	for _, c := range grid {
		byCat[c.Category] = append(byCat[c.Category], c)
	}
	totTok := map[string]int{}
	totPass := map[string]int{}
	totN := map[string]int{}
	for _, cat := range order {
		cs := byCat[cat]
		sort.Slice(cs, func(i, j int) bool {
			if cs[i].Pass != cs[j].Pass {
				return cs[i].Pass > cs[j].Pass
			}
			return cs[i].TotalTokens < cs[j].TotalTokens
		})
		win := cs[0]
		fmt.Printf("  %-18s -> %-14s (acc %d/%d, %d tok)\n", cat, short(win.Model), win.Pass, win.Total, win.TotalTokens)
		for _, c := range grid {
			if c.Category == cat {
				totTok[c.Model] += c.TotalTokens
				totPass[c.Model] += c.Pass
				totN[c.Model] += c.Total
			}
		}
	}
	fmt.Printf("\n=== PER-MODEL TOTALS (single-model baseline) ===\n")
	for _, m := range models {
		fmt.Printf("  %-14s acc=%d/%d  total_tokens=%d\n", short(m), totPass[m], totN[m], totTok[m])
	}
	fmt.Println("\nsaved: eval-results/routeprobe.json")
}

func short(m string) string {
	if i := strings.LastIndex(m, "/"); i >= 0 {
		return m[i+1:]
	}
	return m
}

func iff(cond bool, a, b string) string {
	if cond {
		return a
	}
	return b
}

func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func splitCSV(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}
