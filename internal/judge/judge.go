// Package judge implements the accuracy grader used to score agent answers in
// local evaluations (the "accuracy gate" proxy). It matches the AMD query-router
// reference repo (inspo/fine-tune-llm-query-router-amd, built by the event
// organisers), so a local pass/fail predicts the leaderboard gate:
//
//   - code_generation -> objective grading by executing the generated function
//     against known test cases (RunTests, a port of the repo's code_exec.py).
//   - everything else -> a fast, decisive, reference-based LLM judge (Grade),
//     using the repo's JUDGE_PROMPT and its default judge model (glm-5p2) over
//     the Fireworks OpenAI-compatible API.
//
// EVAL-ONLY: this is not part of the deployed agent.
package judge

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// DefaultModel mirrors the reference repo's MODEL_JUDGE (.env.example).
const DefaultModel = "accounts/fireworks/models/glm-5p2"

const defaultBase = "https://api.fireworks.ai/inference/v1"

type Judge struct {
	key, base, model string
	hc               *http.Client
}

// New builds a judge. Empty base/model fall back to the Fireworks endpoint and
// the reference repo's default judge model (glm-5p2). key is the bearer token
// for base (the Fireworks API key when using the default endpoint).
func New(key, base, model string) *Judge {
	if strings.TrimSpace(base) == "" {
		base = defaultBase
	}
	if strings.TrimSpace(model) == "" {
		model = DefaultModel
	}
	return &Judge{
		key:   key,
		base:  strings.TrimRight(base, "/"),
		model: model,
		hc:    &http.Client{Timeout: 120 * time.Second},
	}
}

// judgePrompt is the reference repo's JUDGE_PROMPT (data/label_dataset.py),
// adapted from grading two candidates (A/B) to grading a single candidate.
// Deliberately fast and decisive; reference-based; lenient on wording but strict
// on factual/logical errors - matching how the organisers grade.
const judgePrompt = `You are a fast, decisive grader. Do not deliberate at length, do not count words, do not second-guess yourself. Grade the candidate in one short sentence, then stop.

Question: %s

Known correct answer or rubric: %s

Candidate: %s

Decide if the candidate correctly and adequately answers the question given the known correct answer/rubric above. Minor wording differences are fine; factual or logical errors are not. Give your verdict immediately - do not show detailed reasoning.

Respond in this exact format and nothing else:
<one short sentence> -> <CORRECT or INCORRECT>
JSON: {"correct": true or false}`

// jsonObjRe matches the JSON object in a judge reply (greedy + DOTALL), mirroring
// label_dataset.parse_judge_json's re.search(r"\{.*\}", text, re.DOTALL).
var jsonObjRe = regexp.MustCompile(`(?s)\{.*\}`)

// Grade grades a single candidate answer against a reference (the "everything
// else" branch of the reference repo's grade()). Returns (correct, verdictLine,
// err). Callers should bound concurrency.
func (j *Judge) Grade(ctx context.Context, task, reference, candidate string) (bool, string, error) {
	prompt := fmt.Sprintf(judgePrompt, task, reference, candidate)
	payload, _ := json.Marshal(map[string]any{
		"model":       j.model,
		"messages":    []map[string]string{{"role": "user", "content": prompt}},
		"max_tokens":  1200,
		"temperature": 0,
	})
	req, err := http.NewRequestWithContext(ctx, "POST", j.base+"/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return false, "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+j.key)
	resp, err := j.hc.Do(req)
	if err != nil {
		return false, "", err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		snip := string(raw)
		if len(snip) > 200 {
			snip = snip[:200]
		}
		return false, "", fmt.Errorf("judge http %d: %s", resp.StatusCode, snip)
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return false, "", fmt.Errorf("judge parse: %w", err)
	}
	if len(out.Choices) == 0 {
		return false, "", fmt.Errorf("judge returned no choices")
	}
	return parseVerdict(strings.TrimSpace(out.Choices[0].Message.Content))
}

// parseVerdict extracts the boolean verdict from a judge reply: first the JSON
// {"correct": ...}, then the "-> CORRECT / -> INCORRECT" marker as a fallback.
func parseVerdict(text string) (bool, string, error) {
	if text == "" {
		return false, "", fmt.Errorf("judge returned empty content")
	}
	if m := jsonObjRe.FindString(text); m != "" {
		var v struct {
			Correct *bool `json:"correct"`
		}
		if err := json.Unmarshal([]byte(m), &v); err == nil && v.Correct != nil {
			return *v.Correct, firstLine(text), nil
		}
	}
	u := strings.ToUpper(text)
	switch {
	case strings.Contains(u, "-> INCORRECT"), strings.Contains(u, "->INCORRECT"):
		return false, firstLine(text), nil
	case strings.Contains(u, "-> CORRECT"), strings.Contains(u, "->CORRECT"):
		return true, firstLine(text), nil
	case strings.Contains(u, "INCORRECT"):
		return false, firstLine(text), nil
	case strings.Contains(strings.ReplaceAll(u, "INCORRECT", ""), "CORRECT"):
		return true, firstLine(text), nil
	}
	return false, firstLine(text), fmt.Errorf("no verdict in judge reply: %q", firstLine(text))
}

// Test is one input/output case for objective code_generation grading.
type Test struct {
	Args     []any `json:"args"`
	Expected any   `json:"expected"`
}

var codeBlockRe = regexp.MustCompile("(?s)```(?:python)?\\s*\\n(.*?)```")

// extractCode returns the first fenced python block, or the whole text if none.
func extractCode(answer string) string {
	if m := codeBlockRe.FindStringSubmatch(answer); m != nil {
		return m[1]
	}
	return answer
}

// RunTests grades a code_generation answer objectively by executing the extracted
// function against known cases - a port of the reference repo's code_exec.run_tests.
// Returns true only if the function is defined and every test passes. Requires
// python3 on PATH. EVAL-ONLY: this runs model-generated code with only a timeout
// for isolation; never do this with untrusted input in production.
func RunTests(ctx context.Context, answer, functionName string, tests []Test, timeout time.Duration) (bool, error) {
	if timeout <= 0 {
		timeout = 10 * time.Second
	}
	dir, err := os.MkdirTemp("", "judge-code-")
	if err != nil {
		return false, err
	}
	defer os.RemoveAll(dir)

	testsJSON, err := json.Marshal(tests)
	if err != nil {
		return false, err
	}
	testsPath := filepath.Join(dir, "tests.json")
	if err := os.WriteFile(testsPath, testsJSON, 0o644); err != nil {
		return false, err
	}
	harness := extractCode(answer) + "\n\n" +
		"import json, sys\n" +
		"_tests = json.load(open(sys.argv[1]))\n" +
		"_fn = " + functionName + "\n" +
		"_ok = True\n" +
		"for _t in _tests:\n" +
		"    try:\n" +
		"        if _fn(*_t['args']) != _t['expected']:\n" +
		"            _ok = False\n" +
		"    except Exception:\n" +
		"        _ok = False\n" +
		"print('PASS' if _ok else 'FAIL')\n"
	harnessPath := filepath.Join(dir, "harness.py")
	if err := os.WriteFile(harnessPath, []byte(harness), 0o644); err != nil {
		return false, err
	}

	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	outBytes, _ := exec.CommandContext(cctx, "python3", harnessPath, testsPath).Output()
	return strings.HasSuffix(strings.TrimSpace(string(outBytes)), "PASS"), nil
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}
