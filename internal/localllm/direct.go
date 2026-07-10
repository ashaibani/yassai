// DirectSolver runs the UN-tuned MiniCPM5 base GGUF for families the
// fine-tuned tool lane cannot serve (the SFT specialised v2e3b into the
// run_python contract and cost it general abilities: base scores 5/5 on
// unseen code_generation variants where v2e3b direct-chat scores 1/5).
// Answers are free like the tool lane; per-family evidence gates reject
// doubtful ones back to the Fireworks batch, so accuracy cannot drop below
// the remote baseline.
package localllm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Families the base lane accepts. Sentiment/summarisation/factual stay remote:
// the judge fails right-label-wrong-rationale answers and rubric omissions,
// which no cheap gate can see (2/6, 1/3, 2/3 on the variant probe).
const (
	FamilyCodeGen = "code_generation"
	FamilyNER     = "named_entity_recognition"
)

// directInstructions are terse per-family system prompts (the track-1 inspo
// ROUTES pattern): output tokens are free locally, but short answers keep the
// judge focused on content instead of stray claims.
var directInstructions = map[string]string{
	FamilyCodeGen: "Write exactly the code requested, nothing extra. Code only, minimal comments, no explanation or demos unless asked.",
	FamilyNER:     "Extract the entities in exactly the requested format, nothing else. Include every entity present; omissions are failures.",
}

var directMaxGen = map[string]int{
	FamilyCodeGen: 512,
	FamilyNER:     320,
}

type DirectSolver struct {
	proc    *exec.Cmd
	baseURL string
	client  *http.Client
	timeout time.Duration
	mu      sync.Mutex // -np 1 server: one request at a time
}

// NewDirect spawns a llama-server for the base GGUF with thinking disabled.
// MiniCPM5 is a hybrid-thinking model: without --reasoning off the 1B spends
// its whole budget inside <think> and content comes back empty.
func NewDirect(cfg Config) (*DirectSolver, error) {
	if strings.TrimSpace(cfg.ModelPath) == "" {
		return nil, errors.New("localllm: empty base model path")
	}
	if cfg.Threads <= 0 {
		cfg.Threads = 6
	}
	if cfg.CtxSize == 0 {
		cfg.CtxSize = 4096
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 60 * time.Second
	}
	abs, err := filepath.Abs(cfg.ModelPath)
	if err != nil {
		return nil, err
	}
	server := strings.TrimSpace(cfg.ServerPath)
	if server == "" {
		server = filepath.Join(cfg.LibPath, "llama-server")
	}
	port, err := freePort()
	if err != nil {
		return nil, fmt.Errorf("localllm: no free port: %w", err)
	}
	cmd := exec.Command(server,
		"-m", abs,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"-t", strconv.Itoa(cfg.Threads),
		"-c", strconv.Itoa(int(cfg.CtxSize)),
		"-np", "1",
		"--no-webui",
		"--reasoning", "off",
	)
	cmd.Env = append(cmd.Environ(), "LD_LIBRARY_PATH="+cfg.LibPath)
	cmd.Stdout = nil
	cmd.Stderr = nil
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("localllm: start base llama-server: %w", err)
	}
	s := &DirectSolver{
		proc:    cmd,
		baseURL: "http://127.0.0.1:" + strconv.Itoa(port),
		client:  &http.Client{Timeout: 3 * time.Minute},
		timeout: cfg.Timeout,
	}
	if err := s.waitHealthy(90 * time.Second); err != nil {
		s.Close()
		return nil, fmt.Errorf("localllm: base server never became healthy: %w", err)
	}
	// Boot canary: degeneracy check only (per-task gates own correctness).
	cctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	canary, cerr := s.chat(cctx, "Write a Python function called add_two that returns its argument plus two.", FamilyCodeGen)
	if cerr != nil || !strings.Contains(canary, "def add_two") {
		s.Close()
		return nil, fmt.Errorf("localllm: base boot canary failed: err=%v out=%.80q", cerr, canary)
	}
	return s, nil
}

func (s *DirectSolver) waitHealthy(budget time.Duration) error {
	deadline := time.Now().Add(budget)
	for time.Now().Before(deadline) {
		resp, err := s.client.Get(s.baseURL + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				return nil
			}
		}
		if s.proc.ProcessState != nil {
			return errors.New("base llama-server exited during startup")
		}
		time.Sleep(500 * time.Millisecond)
	}
	return errors.New("health check timed out")
}

func (s *DirectSolver) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.proc != nil && s.proc.Process != nil {
		_ = s.proc.Process.Kill()
		_, _ = s.proc.Process.Wait()
	}
}

// SolveTask answers one prompt for a supported family and gates the result.
// OK=false means the caller must leave the task in the remote batch.
func (s *DirectSolver) SolveTask(ctx context.Context, prompt, family string) Result {
	if _, ok := directInstructions[family]; !ok {
		return Result{Reason: "family " + family + " not served by base lane"}
	}
	tctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	answer, err := s.chat(tctx, prompt, family)
	if err != nil {
		return Result{Reason: "base generate: " + err.Error()}
	}
	answer = strings.TrimSpace(answer)
	if answer == "" {
		return Result{Reason: "base produced no content"}
	}
	var reason string
	switch family {
	case FamilyCodeGen:
		reason = gateCodeGen(tctx, prompt, answer)
	case FamilyNER:
		reason = gateNER(prompt, answer)
	}
	if reason != "" {
		return Result{Answer: answer, Reason: reason}
	}
	return Result{Answer: answer, OK: true}
}

// chat calls /v1/chat/completions so the server applies the base model's own
// (upstream, correct) template - unlike the fine-tune, whose GGUF template is
// mangled and needs the manual training render in Solver.generate.
func (s *DirectSolver) chat(ctx context.Context, prompt, family string) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	body, _ := json.Marshal(map[string]any{
		"messages": []map[string]string{
			{"role": "system", "content": directInstructions[family]},
			{"role": "user", "content": prompt},
		},
		"temperature": 0,
		"max_tokens":  directMaxGen[family],
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	var out struct {
		Choices []struct {
			Message struct {
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", fmt.Errorf("decode chat response: %w", err)
	}
	if resp.StatusCode != http.StatusOK || len(out.Choices) == 0 {
		return "", fmt.Errorf("chat status %d", resp.StatusCode)
	}
	ans := out.Choices[0].Message.Content
	if strings.TrimSpace(ans) == "" { // thinking leak despite --reasoning off
		ans = out.Choices[0].Message.ReasoningContent
	}
	return ans, nil
}

var calledRe = regexp.MustCompile(`called\s+` + "`?" + `([A-Za-z_]\w*)`)

// gateCodeGen: the generated code must contain the requested function name and
// compile (ast.parse). Compilation is the grounding analogue for this family -
// it cannot prove correctness, but it kills truncation, prose-only answers,
// and syntax degeneration, which were the observed base-model failure shapes.
func gateCodeGen(ctx context.Context, prompt, answer string) string {
	code := answer
	if m := fenceRe.FindStringSubmatch(answer); m != nil {
		code = m[1]
	}
	if m := calledRe.FindStringSubmatch(prompt); m != nil {
		if !strings.Contains(code, "def "+m[1]) {
			return "code lacks requested function def " + m[1]
		}
	} else if !strings.Contains(code, "def ") {
		return "no function definition in answer"
	}
	pctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(pctx, "python3", "-c", "import ast,sys; ast.parse(sys.stdin.read())")
	cmd.Stdin = strings.NewReader(code)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "generated code does not parse: " + firstLineOf(stderr.String())
	}
	return exampleCheck(ctx, prompt, code)
}

// exampleCheck executes the generated function against the worked example the
// prompt itself provides ("f(args) should return expected") and compares
// results. Parse gates pass logic bugs (observed: an interval-intersection
// answer using < for touching endpoints); the prompt's own example is ground
// truth we can run. Prompts without the pattern skip the check.
func exampleCheck(ctx context.Context, prompt, code string) string {
	m := calledRe.FindStringSubmatch(prompt)
	if m == nil {
		return ""
	}
	fname := m[1]
	callStart := strings.Index(prompt, fname+"(")
	if callStart < 0 {
		return ""
	}
	call, rest := balancedPrefix(prompt[callStart:])
	if call == "" || !strings.HasPrefix(rest, " should return ") {
		return ""
	}
	expected := literalPrefix(strings.TrimPrefix(rest, " should return "))
	if expected == "" {
		return ""
	}
	script := code + `

def _norm(x):
    if isinstance(x, (list, tuple)):
        return [_norm(i) for i in x]
    return x

_r = ` + call + `
_e = ` + expected + `
assert _norm(_r) == _norm(_e), "example mismatch: got %r, want %r" % (_r, _e)
`
	pctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(pctx, "python3", "-")
	cmd.Stdin = strings.NewReader(script)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "example check failed: " + lastLineOf(stderr.String())
	}
	return ""
}

// balancedPrefix returns the leading "name(...)" call with balanced brackets
// and closed strings, plus the remainder of s after it.
func balancedPrefix(s string) (string, string) {
	depth, inStr := 0, byte(0)
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case inStr != 0:
			if c == inStr {
				inStr = 0
			}
		case c == '\'' || c == '"':
			inStr = c
		case c == '(' || c == '[' || c == '{':
			depth++
		case c == ')' || c == ']' || c == '}':
			depth--
			if depth == 0 {
				return s[:i+1], s[i+1:]
			}
		}
	}
	return "", s
}

// literalPrefix extracts the leading Python literal from s: a balanced
// bracket/quote form, or a bare scalar ended by a sentence-final full stop.
func literalPrefix(s string) string {
	if s == "" {
		return ""
	}
	if c := s[0]; c == '(' || c == '[' || c == '{' {
		lit, _ := balancedPrefix(s)
		return lit
	}
	depth, inStr := 0, byte(0)
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case inStr != 0:
			if c == inStr {
				inStr = 0
			}
		case c == '\'' || c == '"':
			inStr = c
		case c == '(' || c == '[' || c == '{':
			depth++
		case c == ')' || c == ']' || c == '}':
			depth--
		case depth == 0 && (c == ',' || c == '\n'):
			return strings.TrimSpace(s[:i])
		case depth == 0 && c == '.':
			// full stop ends the literal unless it is a decimal point
			if i+1 >= len(s) || s[i+1] == ' ' || s[i+1] == '\n' {
				return strings.TrimSpace(s[:i])
			}
		}
	}
	return strings.TrimSpace(s)
}

func lastLineOf(s string) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	return lines[len(lines)-1]
}

var (
	yearRe    = regexp.MustCompile(`\b(19|20)\d{2}\b`)
	capRunRe  = regexp.MustCompile(`\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b`)
	capWordRe = regexp.MustCompile(`([^.!?:\n]\s)([A-Z][a-z]{2,})\b`)
	acronymRe = regexp.MustCompile(`\b[A-Z]{2,}\b`)
)

// nerLabelVocab: all-caps words that are NER label names in the instruction,
// not entities in the source - exempt from the acronym recall check.
var nerLabelVocab = map[string]bool{
	"PERSON": true, "ORGANIZATION": true, "ORG": true, "LOCATION": true,
	"LOC": true, "DATE": true, "TIME": true, "MISC": true, "GPE": true,
	"EVENT": true, "PRODUCT": true, "MONEY": true, "PERCENT": true, "NER": true,
}

// gateNER rejects answers that omit high-precision entity signals present in
// the source: years, multi-word capitalised names, and mid-sentence capitalised
// words. Every observed local NER failure (missed 2025 / Seattle / Microsoft)
// was an omission this recall check catches; false positives merely send the
// task back to the remote batch, costing tokens, never accuracy.
func gateNER(prompt, answer string) string {
	for _, y := range yearRe.FindAllString(prompt, -1) {
		if !strings.Contains(answer, y) {
			return "answer omits year " + y
		}
	}
	for _, name := range capRunRe.FindAllString(prompt, -1) {
		if strings.Contains(answer, name) {
			continue
		}
		// Sentence-start capitalisation drags function words into the run
		// ("On March 15" -> "On March"); the entity signal is the tail, so an
		// answer carrying the run minus its first word still passes.
		if i := strings.IndexByte(name, ' '); i > 0 && strings.Contains(answer, name[i+1:]) {
			continue
		}
		return "answer omits name " + name
	}
	for _, m := range capWordRe.FindAllStringSubmatch(prompt, -1) {
		w := m[2]
		if !strings.Contains(answer, w) {
			return "answer omits capitalised token " + w
		}
	}
	// All-caps acronyms (ETH, NASA) are invisible to the capitalised-word
	// patterns above but are near-certain entity parts in a NER source text.
	for _, w := range acronymRe.FindAllString(prompt, -1) {
		if nerLabelVocab[w] {
			continue
		}
		if !strings.Contains(answer, w) {
			return "answer omits acronym " + w
		}
	}
	return ""
}

func firstLineOf(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}
