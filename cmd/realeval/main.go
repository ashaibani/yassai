// Command realeval runs the agent on a REAL benchmark task set (testdata/
// real_tasks.json) with the deployed config, and scores each answer: heuristic
// validators for numeric/ner/contains, and the reference-repo LLM judge
// (internal/judge, default model glm-5p2 over Fireworks) for the open-ended
// "llm" tasks. Reports per-category accuracy + tokens — the honest "does the
// agent actually work on real tasks" number.
//
// Usage:
//
//	FIREWORKS_API_KEY=... go run ./cmd/realeval
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/ashaibani/yassai/internal/agent"
	"github.com/ashaibani/yassai/internal/judge"
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
		return "logical"
	default:
		return "unknown"
	}
}

func getenv(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

// realevalTier maps a (realeval) category to its reasoning-effort tier, matching
// the agent's effortTier difficulty ranking.
func canonicalCat(short string) string {
	switch short {
	case "factual":
		return "factual_knowledge"
	case "maths":
		return "mathematical_reasoning"
	case "sentiment":
		return "sentiment_classification"
	case "summarisation":
		return "text_summarisation"
	case "ner":
		return "named_entity_recognition"
	case "logical":
		return "logical_deductive_reasoning"
	default:
		return short // code_debugging, code_generation already canonical
	}
}

func realevalTier(cat string) string {
	if cat == "logical" {
		return "xhigh" // only category that benefited from more reasoning; rest stay low
	}
	return "low"
}

type scored struct {
	id, cat string
	pass    bool
	via     string // "judge" or "check"
	reason  string
}

func main() {
	tasksPath := getenv("TASKS_PATH", "testdata/real_tasks.json")
	apiKey := os.Getenv("FIREWORKS_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "FIREWORKS_API_KEY is required")
		os.Exit(1)
	}
	model := getenv("ALLOWED_MODELS", "accounts/fireworks/models/minimax-m3")
	model = strings.Split(model, ",")[0]

	cases, err := validate.LoadCases(tasksPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	caseByID := map[string]validate.Case{}
	tasks := make([]agent.Task, 0, len(cases))
	for _, c := range cases {
		caseByID[c.TaskID] = c
		tasks = append(tasks, agent.Task{TaskID: c.TaskID, Prompt: c.Prompt})
	}

	tmpMem, _ := os.MkdirTemp("", "realeval-mem-")
	defer os.RemoveAll(tmpMem)
	baseCfg := agent.Config{
		APIKey:           apiKey,
		BaseURL:          getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"),
		DisableHints:     getenv("SKILLS", "on") == "off",
		MaxBatchSize:     20,
		MaxBatchTokens:   12000,
		MaxContextTokens: 200000,
		MemoryRoot:       tmpMem,
		Timeout:          180 * time.Second,
	}
	mode := getenv("EFFORT_MODE", "") // "adaptive" = per-category tier; else uniform AGENT_REASONING_EFFORT
	ansByID := map[string]string{}
	var metrics agent.Metrics
	start := time.Now()
	run := func(ts []agent.Task, mdl, effort string) {
		cfg := baseCfg
		cfg.AllowedModels = []string{mdl}
		cfg.PreferredModel = mdl
		cfg.ReasoningEffort = effort
		cats := map[string][]string{}
		for _, t := range ts {
			cats[t.TaskID] = []string{canonicalCat(categoryOf(t.TaskID))}
		}
		cfg.Categories = cats // true categories → drives technique hints (classifier not needed in eval)
		ag, nerr := agent.New(cfg)
		if nerr != nil {
			fmt.Fprintln(os.Stderr, "agent.New:", nerr)
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), 9*time.Minute+30*time.Second)
		res, m, serr := ag.Solve(ctx, ts)
		cancel()
		if serr != nil {
			fmt.Fprintln(os.Stderr, "solve:", serr)
		}
		for _, r := range res {
			ansByID[r.TaskID] = r.Answer
		}
		metrics.TotalTokens += m.TotalTokens
		metrics.PromptTokens += m.PromptTokens
		metrics.OutputTokens += m.OutputTokens
		metrics.ReasoningTokens += m.ReasoningTokens
		metrics.Calls += m.Calls
	}
	if mode == "adaptive" {
		codeModel := getenv("CODE_MODEL", "") // route code_debugging/code_generation here if set
		type grp struct{ model, effort string }
		groups := map[grp][]agent.Task{}
		var order []grp
		for _, t := range tasks {
			cat := categoryOf(t.TaskID)
			m := model
			if codeModel != "" && (cat == "code_debugging" || cat == "code_generation") {
				m = codeModel
			}
			g := grp{m, realevalTier(cat)}
			if _, ok := groups[g]; !ok {
				order = append(order, g)
			}
			groups[g] = append(groups[g], t)
		}
		fmt.Printf("running %d real tasks ADAPTIVE (base=%s code_model=%q)...\n", len(tasks), categoryShort(model), codeModel)
		for _, g := range order {
			fmt.Printf("  %-16s effort=%-6s: %d tasks\n", categoryShort(g.model), g.effort, len(groups[g]))
			run(groups[g], g.model, g.effort)
		}
	} else {
		eff := getenv("AGENT_REASONING_EFFORT", "low")
		fmt.Printf("running %d real tasks on %s (effort=%s, batch<=20)...\n", len(tasks), categoryShort(model), eff)
		run(tasks, model, eff)
	}
	dur := time.Since(start).Seconds()

	jd := judge.New(apiKey, baseCfg.BaseURL, getenv("MODEL_JUDGE", ""))
	out := make([]scored, len(cases))
	sem := make(chan struct{}, 3) // umans concurrency cap
	var wg sync.WaitGroup
	for i, c := range cases {
		i, c := i, c
		ans := ansByID[c.TaskID]
		s := scored{id: c.TaskID, cat: categoryOf(c.TaskID)}
		if c.Validate == "llm" {
			wg.Add(1)
			go func() {
				defer wg.Done()
				sem <- struct{}{}
				defer func() { <-sem }()
				pass, verdict, jerr := jd.Grade(context.Background(), c.Prompt, c.Expected, ans)
				s.via = "judge"
				if jerr != nil {
					s.reason = "ERR: " + jerr.Error()
				} else {
					s.pass, s.reason = pass, firstLine(verdict)
				}
				out[i] = s
			}()
		} else {
			res := validate.Check(ans, c)
			s.via, s.pass, s.reason = "check", res.Pass, res.Reason
			out[i] = s
		}
	}
	wg.Wait()

	// aggregate
	type agg struct{ pass, total int }
	byCat := map[string]*agg{}
	var pass, total int
	for _, s := range out {
		if byCat[s.cat] == nil {
			byCat[s.cat] = &agg{}
		}
		byCat[s.cat].total++
		total++
		if s.pass {
			byCat[s.cat].pass++
			pass++
		}
	}
	fmt.Printf("\n=== REAL-TASK EVAL (%s) ===\n", tasksPath)
	cats := make([]string, 0, len(byCat))
	for k := range byCat {
		cats = append(cats, k)
	}
	sort.Strings(cats)
	for _, c := range cats {
		fmt.Printf("  %-16s %d/%d\n", c, byCat[c].pass, byCat[c].total)
	}
	fmt.Printf("\nOVERALL: %d/%d (%.1f%%)  tokens=%d (prompt=%d out=%d reason=%d)  calls=%d  %.1fs\n",
		pass, total, 100*float64(pass)/float64(total),
		metrics.TotalTokens, metrics.PromptTokens, metrics.OutputTokens, metrics.ReasoningTokens, metrics.Calls, dur)

	// failures detail
	fmt.Println("\nfailures:")
	for _, s := range out {
		if !s.pass {
			fmt.Printf("  [%s|%s|%s] %s\n", s.id, s.cat, s.via, firstLine(s.reason))
		}
	}

	_ = os.MkdirAll("eval-results", 0o755)
	b, _ := json.MarshalIndent(map[string]any{
		"overall_pass": pass, "overall_total": total, "metrics": metrics,
	}, "", "  ")
	_ = os.WriteFile("eval-results/realeval.json", append(b, '\n'), 0o644)
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

func categoryShort(m string) string {
	if i := strings.LastIndex(m, "/"); i >= 0 {
		return m[i+1:]
	}
	return m
}
