// Package pyexec runs model-generated code in a real python3 subprocess. The
// runtime image ships python3 (python:3.12-slim), so unlike the MicroPython WASM
// sandbox the full stdlib is available (itertools, json, str.zfill/format, ...).
// That matters for tokens: in MicroPython the model's first-attempt maths/logic
// code repeatedly hit feature gaps (no itertools, no format(), no str.zfill),
// erroring and forcing multi-turn retries that re-send the whole history. With
// real python3 the first attempt runs, so code-exec batches finish in one call.
//
// A submit() is pre-defined via a preamble; it prints a sentinel line that this
// host parses back into the final answers. This executes the agent's OWN
// generated code, bounded by a timeout - the same approach as the organisers'
// reference code_exec.py. It is not a hardened sandbox; do not feed it untrusted
// third-party input.
package pyexec

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
	"time"
)

const submitSentinel = "__YASSAI_SUBMIT__"

// preamble pre-defines submit() so the model calls it directly. It writes a
// sentinel-prefixed JSON line that Run parses. python3 has json, so this is
// robust regardless of how the model formats its answers.
const preamble = "import json as __j, sys as __s\n" +
	"def submit(answers=None, **__kw):\n" +
	"    _a = answers if answers is not None else __kw.get('answers')\n" +
	"    __s.stdout.write('" + submitSentinel + "'+__j.dumps(_a)+'\\n')\n"

// Result mirrors the fields the agent loop reads from a tool run.
type Result struct {
	Stdout string
	JSON   string
	Value  any
}

// Config keeps a micropy-compatible shape; only Timeout is used.
type Config struct {
	Timeout     time.Duration
	MemoryBytes int
}

type Executor struct {
	timeout   time.Duration
	expected  []string
	submitted bool
	result    map[string]string
}

func New(cfg Config) (*Executor, error) {
	t := cfg.Timeout
	if t <= 0 {
		t = 15 * time.Second
	}
	return &Executor{timeout: t}, nil
}

func (e *Executor) SetExpectedTasks(ids []string)   { e.expected = append([]string(nil), ids...) }
func (e *Executor) ResetSubmit()                    { e.submitted = false; e.result = nil }
func (e *Executor) IsSubmitted() bool               { return e.submitted }
func (e *Executor) SubmitResult() map[string]string { return e.result }

// Run executes the model's code in python3 with the submit() preamble prepended.
func (e *Executor) Run(ctx context.Context, sessionID, code string, input map[string]any) (Result, error) {
	cctx, cancel := context.WithTimeout(ctx, e.timeout)
	defer cancel()
	// -I: isolated mode (ignore env/user site) for reproducibility.
	cmd := exec.CommandContext(cctx, "python3", "-I", "-c", preamble+"\n"+code)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	runErr := cmd.Run()

	out := stdout.String()
	var submitJSON string
	if i := strings.LastIndex(out, submitSentinel); i >= 0 {
		rest := out[i+len(submitSentinel):]
		if nl := strings.IndexByte(rest, '\n'); nl >= 0 {
			rest = rest[:nl]
		}
		submitJSON = strings.TrimSpace(rest)
		if m, ok := parseSubmit(submitJSON); ok {
			e.result = m
			e.submitted = true
		}
		out = out[:i] // hide the sentinel line from the observation
	}

	res := Result{Stdout: strings.TrimSpace(out), JSON: submitJSON}
	if runErr != nil && !e.submitted {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = runErr.Error()
		}
		return res, fmt.Errorf("python3: %s", lastLine(msg))
	}
	return res, nil
}

func parseSubmit(s string) (map[string]string, bool) {
	var arr []struct {
		TaskID string `json:"task_id"`
		Answer any    `json:"answer"`
	}
	if err := json.Unmarshal([]byte(s), &arr); err == nil && len(arr) > 0 {
		m := map[string]string{}
		for _, it := range arr {
			if strings.TrimSpace(it.TaskID) == "" {
				continue
			}
			m[it.TaskID] = stringify(it.Answer)
		}
		if len(m) > 0 {
			return m, true
		}
	}
	return nil, false
}

func stringify(v any) string {
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

func lastLine(s string) string {
	s = strings.TrimRight(s, "\n")
	if i := strings.LastIndexByte(s, '\n'); i >= 0 {
		return strings.TrimSpace(s[i+1:])
	}
	return s
}
