// Command localprobe maps which task families a bare local GGUF can carry
// WITHOUT the agent scaffolding: it answers every task in a set via
// llama-server's /v1/chat/completions (the model's own chat template) and
// scores answers exactly like cmd/realeval (heuristic validators + the
// reference glm-5p2 judge). The per-category pass rates say which families are
// safe to route local-first before any fine-tune or gate work.
//
// EVAL-ONLY: not part of the deployed agent.
//
// Usage:
//
//	FIREWORKS_API_KEY=... go run ./cmd/localprobe \
//	  -model models/minicpm5/MiniCPM5-1B-base-Q4_K_M.gguf \
//	  -tasks testdata/downloads_tasks_golden.json
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/ashaibani/yassai/internal/judge"
	"github.com/ashaibani/yassai/internal/validate"
)

func main() {
	model := flag.String("model", "models/minicpm5/MiniCPM5-1B-base-Q4_K_M.gguf", "GGUF to probe")
	tasksPath := flag.String("tasks", "testdata/downloads_tasks_golden.json", "task set")
	lib := flag.String("lib", firstNonEmpty(os.Getenv("YZMA_LIB"), os.Getenv("HOME")+"/opt/llama"), "dir with llama-server")
	port := flag.Int("port", 18089, "llama-server port")
	maxTok := flag.Int("maxtok", 768, "max generation tokens per task")
	outPath := flag.String("out", "", "answers JSON (default eval-results/localprobe-<model>.json)")
	flag.Parse()

	apiKey := os.Getenv("FIREWORKS_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "FIREWORKS_API_KEY is required (judge)")
		os.Exit(1)
	}
	cases, err := validate.LoadCases(*tasksPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}

	// --reasoning off: MiniCPM5 is a hybrid-thinking model; left on, the 1B
	// spends the whole token budget inside <think> and content comes back empty.
	srv := exec.Command(filepath.Join(*lib, "llama-server"),
		"-m", *model, "--host", "127.0.0.1", "--port", fmt.Sprint(*port),
		"-c", "4096", "--threads", "6", "-ngl", "99", "--reasoning", "off")
	srv.Stdout, srv.Stderr = nil, nil
	if err := srv.Start(); err != nil {
		fmt.Fprintln(os.Stderr, "llama-server:", err)
		os.Exit(1)
	}
	defer func() { _ = srv.Process.Kill(); _, _ = srv.Process.Wait() }()
	base := fmt.Sprintf("http://127.0.0.1:%d", *port)
	if err := waitReady(base, 120*time.Second); err != nil {
		fmt.Fprintln(os.Stderr, "server never became ready:", err)
		os.Exit(1)
	}

	type row struct {
		ID, Cat, Answer, Via, Reason string
		Pass                         bool
		GenTokens                    int
		GenMS                        int64
	}
	rows := make([]row, len(cases))
	fmt.Printf("probing %d tasks with %s...\n", len(cases), filepath.Base(*model))
	for i, c := range cases {
		t0 := time.Now()
		ans, ntok, gerr := chat(base, c.Prompt, *maxTok)
		rows[i] = row{ID: c.TaskID, Cat: categoryOf(c.TaskID), Answer: ans,
			GenTokens: ntok, GenMS: time.Since(t0).Milliseconds()}
		if gerr != nil {
			rows[i].Reason = "GEN ERR: " + gerr.Error()
		}
	}

	jd := judge.New(apiKey, firstNonEmpty(os.Getenv("FIREWORKS_BASE_URL"), ""), os.Getenv("MODEL_JUDGE"))
	sem := make(chan struct{}, 3) // umans concurrency cap
	var wg sync.WaitGroup
	for i, c := range cases {
		if rows[i].Reason != "" { // generation failed; auto-fail
			rows[i].Via = "gen"
			continue
		}
		if c.Validate == "llm" {
			i, c := i, c
			wg.Add(1)
			go func() {
				defer wg.Done()
				sem <- struct{}{}
				defer func() { <-sem }()
				pass, verdict, jerr := jd.Grade(context.Background(), c.Prompt, c.Expected, rows[i].Answer)
				rows[i].Via = "judge"
				if jerr != nil {
					rows[i].Reason = "ERR: " + jerr.Error()
				} else {
					rows[i].Pass, rows[i].Reason = pass, firstLine(verdict)
				}
			}()
		} else {
			res := validate.Check(rows[i].Answer, c)
			rows[i].Via, rows[i].Pass, rows[i].Reason = "check", res.Pass, res.Reason
		}
	}
	wg.Wait()

	type agg struct{ pass, total, tok int }
	byCat := map[string]*agg{}
	var pass, tok int
	for _, r := range rows {
		if byCat[r.Cat] == nil {
			byCat[r.Cat] = &agg{}
		}
		byCat[r.Cat].total++
		byCat[r.Cat].tok += r.GenTokens
		tok += r.GenTokens
		if r.Pass {
			byCat[r.Cat].pass++
			pass++
		}
	}
	fmt.Printf("\n=== LOCAL PROBE (%s on %s) ===\n", filepath.Base(*model), *tasksPath)
	cats := make([]string, 0, len(byCat))
	for k := range byCat {
		cats = append(cats, k)
	}
	sort.Strings(cats)
	for _, c := range cats {
		fmt.Printf("  %-16s %d/%d  (gen tok %d)\n", c, byCat[c].pass, byCat[c].total, byCat[c].tok)
	}
	fmt.Printf("\nOVERALL: %d/%d (%.1f%%)  local gen tokens=%d (leaderboard-free)\n",
		pass, len(rows), 100*float64(pass)/float64(len(rows)), tok)
	fmt.Println("\nfailures:")
	for _, r := range rows {
		if !r.Pass {
			fmt.Printf("  [%s|%s|%s] %s\n", r.ID, r.Cat, r.Via, firstLine(r.Reason))
		}
	}

	out := *outPath
	if out == "" {
		stem := strings.TrimSuffix(filepath.Base(*model), ".gguf")
		out = "eval-results/localprobe-" + stem + ".json"
	}
	_ = os.MkdirAll(filepath.Dir(out), 0o755)
	b, _ := json.MarshalIndent(rows, "", "  ")
	_ = os.WriteFile(out, append(b, '\n'), 0o644)
	fmt.Println("\nanswers ->", out)
}

func chat(base, prompt string, maxTok int) (string, int, error) {
	body, _ := json.Marshal(map[string]any{
		"messages":    []map[string]string{{"role": "user", "content": prompt}},
		"temperature": 0,
		"max_tokens":  maxTok,
	})
	req, _ := http.NewRequest("POST", base+"/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	hc := &http.Client{Timeout: 180 * time.Second}
	resp, err := hc.Do(req)
	if err != nil {
		return "", 0, err
	}
	defer resp.Body.Close()
	var r struct {
		Choices []struct {
			Message struct {
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&r); err != nil {
		return "", 0, err
	}
	if len(r.Choices) == 0 {
		return "", 0, fmt.Errorf("no choices (status %s)", resp.Status)
	}
	ans := strings.TrimSpace(r.Choices[0].Message.Content)
	if ans == "" { // thinking never closed: salvage whatever went into the think stream
		ans = strings.TrimSpace(r.Choices[0].Message.ReasoningContent)
	}
	return ans, r.Usage.CompletionTokens, nil
}

func waitReady(base string, d time.Duration) error {
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		resp, err := http.Get(base + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				return nil
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("timeout after %s", d)
}

// categoryOf mirrors cmd/realeval's id->category mapping.
func categoryOf(id string) string {
	if len(id) >= 3 && id[0] == 'T' {
		switch id[:3] {
		case "T01":
			return "factual"
		case "T02", "T08", "T10":
			return "maths"
		case "T03":
			return "sentiment"
		case "T04":
			return "summarisation"
		case "T05":
			return "ner"
		case "T06":
			return "code_debugging"
		case "T07":
			return "logical"
		case "T09":
			return "code_generation"
		}
	}
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

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}
