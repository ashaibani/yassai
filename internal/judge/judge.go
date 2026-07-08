// Package judge implements an LLM-based accuracy grader (the "accuracy gate"
// proxy) for open-ended answers that heuristic validators can't score, e.g.
// summarisation and code. It mirrors the hackathon's LLM-Judge: decide whether a
// candidate answer satisfies the task intent, given a reference. Grading uses an
// OpenAI-compatible endpoint (default: umans-flash) and is EVAL-ONLY — it is not
// part of the deployed agent.
package judge

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type Judge struct {
	key, base, model, effort string
	hc                       *http.Client
}

func New(key, base, model, effort string) *Judge {
	if base == "" {
		base = "https://api.fireworks.ai/inference/v1"
	}
	if model == "" {
		model = "accounts/fireworks/models/minimax-m3"
	}
	if effort == "" {
		effort = "xhigh" // grade carefully; the judge is eval-only, so accuracy > speed
	}
	return &Judge{key: key, base: strings.TrimRight(base, "/"), model: model, effort: effort,
		hc: &http.Client{Timeout: 180 * time.Second}}
}

const sysPrompt = `You grade an AI agent's answer for a capability benchmark. Given a TASK, a REFERENCE (ground-truth answer or solution), and a CANDIDATE answer, decide whether the candidate correctly satisfies the task.
Rules:
- Accept different wording, order, or formatting, and a brief added explanation, as long as the substance is correct.
- Summaries: require faithfulness to the source and respect of any stated length/format constraint.
- Code: judge functional correctness against the reference behaviour, not textual identity.
- Be strict about factual/logical correctness.
Return ONLY a JSON object of the form {"verdict":"PASS","reason":"<short>"} where verdict is exactly "PASS" or "FAIL". Output nothing outside the JSON.`

// Grade returns (pass, verdictText, err). Callers should bound concurrency.
func (j *Judge) Grade(ctx context.Context, task, reference, candidate string) (bool, string, error) {
	user := fmt.Sprintf("TASK:\n%s\n\nREFERENCE:\n%s\n\nCANDIDATE:\n%s\n\nVerdict:", task, reference, candidate)
	payload, _ := json.Marshal(map[string]any{
		"model": j.model,
		"messages": []map[string]string{
			{"role": "system", "content": sysPrompt},
			{"role": "user", "content": user},
		},
		"max_tokens":       8192, // room for xhigh reasoning to finish AND emit the verdict
		"temperature":      0,
		"reasoning_effort": j.effort,
		"response_format":  map[string]any{"type": "json_object"}, // force a parseable verdict
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
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return false, "", fmt.Errorf("judge parse: %w", err)
	}
	if len(out.Choices) == 0 {
		return false, "", fmt.Errorf("judge returned no choices")
	}
	msg := out.Choices[0].Message
	body := strings.TrimSpace(msg.Content)
	if body == "" {
		body = strings.TrimSpace(msg.ReasoningContent) // reasoning model: content can be empty on truncation
	}
	if body == "" {
		return false, "", fmt.Errorf("judge returned empty content and reasoning")
	}
	// Structured verdict — parse a JSON object like the agent parses its answers,
	// instead of sniffing the text for "PASS".
	var v struct {
		Verdict string `json:"verdict"`
		Reason  string `json:"reason"`
	}
	if obj := extractJSONObject(body); obj != "" {
		if err := json.Unmarshal([]byte(obj), &v); err == nil && v.Verdict != "" {
			return strings.EqualFold(strings.TrimSpace(v.Verdict), "PASS"), v.Reason, nil
		}
	}
	// Fallback (truncated / non-JSON): sniff a decisive token.
	return verdictPass(body), firstLine(body), nil
}

// extractJSONObject returns the first balanced {...} substring, or "".
func extractJSONObject(s string) string {
	i := strings.IndexByte(s, '{')
	if i < 0 {
		return ""
	}
	depth := 0
	for j := i; j < len(s); j++ {
		switch s[j] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return s[i : j+1]
			}
		}
	}
	return ""
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

// verdictPass reads PASS/FAIL from a judge reply, tolerating a leading token or a
// concluding decision at the end of a reasoning trace.
func verdictPass(s string) bool {
	u := strings.ToUpper(strings.TrimSpace(s))
	switch {
	case strings.HasPrefix(u, "PASS"):
		return true
	case strings.HasPrefix(u, "FAIL"):
		return false
	default:
		return strings.LastIndex(u, "PASS") > strings.LastIndex(u, "FAIL")
	}
}
