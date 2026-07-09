// Score agent results against golden cases with heuristic checks + LLM judge.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/ashaibani/yassai/internal/judge"
	"github.com/ashaibani/yassai/internal/validate"
)

func main() {
	golden := getenv("GOLDEN_PATH", "testdata/downloads_tasks_golden.json")
	resultsPath := getenv("RESULTS_PATH", "/tmp/yassai_results.json")
	umansKey := os.Getenv("UMANS_API_KEY")
	if umansKey == "" {
		umansKey = os.Getenv("FIREWORKS_API_KEY")
	}
	cases, err := validate.LoadCases(golden)
	if err != nil {
		fmt.Fprintln(os.Stderr, "golden:", err)
		os.Exit(1)
	}
	b, err := os.ReadFile(resultsPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "results:", err)
		os.Exit(1)
	}
	var results []struct {
		TaskID string `json:"task_id"`
		Answer string `json:"answer"`
	}
	if err := json.Unmarshal(b, &results); err != nil {
		fmt.Fprintln(os.Stderr, "parse results:", err)
		os.Exit(1)
	}
	ans := map[string]string{}
	for _, r := range results {
		ans[r.TaskID] = r.Answer
	}
	mode := getenv("JUDGE_MODE", "lenient") // lenient | strict | noref
	jd := judge.NewWithMode(umansKey, getenv("UMANS_BASE_URL", ""), getenv("UMANS_JUDGE_MODEL", "accounts/fireworks/models/minimax-m3"), getenv("UMANS_JUDGE_EFFORT", "xhigh"), mode)
	// Official leaderboard grades every task with an LLM judge. When JUDGE_ALL=1
	// (or mode is strict/noref), skip deterministic validators and route every
	// case through the judge so local scores match the gate shape.
	judgeAll := envTruthy("JUDGE_ALL") || mode == "strict" || mode == "noref" || mode == "harsh" || mode == "leaderboard"
	type row struct {
		id, via string
		pass    bool
		reason  string
	}
	out := make([]row, len(cases))
	sem := make(chan struct{}, 3)
	var wg sync.WaitGroup
	for i, c := range cases {
		i, c := i, c
		answer := ans[c.TaskID]
		useJudge := judgeAll || c.Validate == "llm"
		if useJudge {
			if umansKey == "" {
				out[i] = row{c.TaskID, "judge", false, "SKIP (no API key)"}
				continue
			}
			wg.Add(1)
			go func() {
				defer wg.Done()
				sem <- struct{}{}
				defer func() { <-sem }()
				pass, verdict, jerr := jd.Grade(context.Background(), c.Prompt, c.Expected, answer)
				r := row{id: c.TaskID, via: "judge/" + mode}
				if jerr != nil {
					r.reason = "ERR: " + jerr.Error()
				} else {
					r.pass, r.reason = pass, firstLine(verdict)
				}
				out[i] = r
			}()
		} else {
			res := validate.Check(answer, c)
			out[i] = row{c.TaskID, "check", res.Pass, res.Reason}
		}
	}
	wg.Wait()
	pass, total := 0, len(out)
	fmt.Printf("=== SCORE mode=%s judge_all=%v (%s) ===\n", mode, judgeAll, golden)
	for _, r := range out {
		mark := "FAIL"
		if r.pass {
			mark = "PASS"
			pass++
		}
		fmt.Printf("  %s [%s|%s] %s\n", mark, r.id, r.via, trunc(r.reason, 120))
	}
	fmt.Printf("\nOVERALL: %d/%d (%.1f%%)\n", pass, total, 100*float64(pass)/float64(total))
	_ = os.MkdirAll("eval-results", 0o755)
	rep, _ := json.MarshalIndent(map[string]any{"pass": pass, "total": total, "mode": mode, "judge_all": judgeAll, "rows": out}, "", "  ")
	_ = os.WriteFile("eval-results/downloads_tasks_scored.json", append(rep, '\n'), 0o644)
}

func getenv(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

func envTruthy(k string) bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv(k)))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}
func trunc(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
