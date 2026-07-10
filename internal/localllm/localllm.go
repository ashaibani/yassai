// Package localllm runs the fine-tuned MiniCPM5 GGUF inside the container for
// maths/logic tasks: prompt -> one run_python tool call -> execute -> final
// answer. The Participant Guide permits local inference (counts toward
// accuracy, NOT the token score), so every task answered here is free.
// A validation gate rejects anything doubtful - rejected tasks fall back to
// the normal Fireworks batch, so accuracy can never drop below that baseline.
//
// Engine: a spawned llama-server driven over localhost HTTP. The previous
// in-process yzma (purego) bindings decoded GARBAGE on Linux (both amd64 and
// arm64 - backtick/quote spam) while llama.cpp's own C++ binaries decoded the
// same GGUF coherently on the same hosts, so the C++ server owns inference and
// Go owns orchestration.
package localllm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/hybridgroup/yzma/pkg/message"
)

// systemPrompt is the exact contract the SFT data was trained on
// (scripts/build_minicpm5_sft_data_v2.py) - do not drift.
// The tool contract is the JSON tool call MiniCPM5 was pretrained on - the
// fenced-Python alternative was tried and REGRESSED the 1B LoRA (syntax errors
// + repeat-loop degeneration even at F16), so JSON stays. The GGUF-mangled
// </tool_call> tag is handled downstream by extractToolCode's balanced-brace
// parser, not by changing the contract. Must stay byte-identical to
// scripts/build_minicpm5_sft_data_v2.py.
const systemPrompt = `You are yassai-local, a small local specialist for math and logic tasks.
Use run_python for arithmetic, percentages, schedules, projections, combinatorics, and constraint checks.
Never do these calculations in your head.
When a tool is needed, respond with exactly one tool call and no prose:
<tool_call>{"name":"run_python","arguments":{"code":"..."}}</tool_call>
The Python must compute from named variables and print concise final values.
After receiving a run_python result, return only the final answer requested by the user.`

type Config struct {
	ModelPath  string // fine-tuned GGUF
	LibPath    string // directory containing llama-server and its shared libraries
	ServerPath string // llama-server binary; default LibPath/llama-server
	Threads    int
	CtxSize    uint32
	MaxGen     int32         // max generated tokens per turn
	Timeout    time.Duration // per-task wall clock budget
	// Extended (DirectSolver only): which assist-tuned families to unlock -
	// "all"/"1" for every extended family, or a CSV subset (e.g.
	// "factual_knowledge,text_summarisation"). Empty disables them all.
	Extended string
}

type Solver struct {
	cfg     Config
	proc    *exec.Cmd
	baseURL string
	client  *http.Client
	mu      sync.Mutex // one request at a time (-np 1 server)
}

// Result is one task attempt. OK=false means the caller must fall back to the
// remote path; Reason says why (telemetry).
type Result struct {
	Answer string
	Code   string
	Stdout string
	OK     bool
	Reason string
}

func New(cfg Config) (*Solver, error) {
	if strings.TrimSpace(cfg.ModelPath) == "" {
		return nil, errors.New("localllm: empty model path")
	}
	if cfg.Threads <= 0 {
		cfg.Threads = 6
	}
	if cfg.CtxSize == 0 {
		cfg.CtxSize = 4096
	}
	if cfg.MaxGen <= 0 {
		// Logic/revenue tool calls run ~200-300 tokens of code inside JSON;
		// leave generous headroom - truncation mid-JSON wastes the whole task.
		cfg.MaxGen = 1024
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 75 * time.Second
	}
	abs, err := filepath.Abs(cfg.ModelPath)
	if err != nil {
		return nil, err
	}
	if _, err := os.Stat(abs); err != nil {
		return nil, fmt.Errorf("localllm: model file: %w", err)
	}
	server := strings.TrimSpace(cfg.ServerPath)
	if server == "" {
		server = filepath.Join(cfg.LibPath, "llama-server")
	}
	if _, err := os.Stat(server); err != nil {
		return nil, fmt.Errorf("localllm: llama-server binary: %w", err)
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
	)
	cmd.Env = append(os.Environ(), "LD_LIBRARY_PATH="+cfg.LibPath)
	cmd.Stdout = nil
	cmd.Stderr = nil
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("localllm: start llama-server: %w", err)
	}
	s := &Solver{
		cfg:     cfg,
		proc:    cmd,
		baseURL: "http://127.0.0.1:" + strconv.Itoa(port),
		client:  &http.Client{Timeout: 3 * time.Minute},
	}
	if err := s.waitHealthy(90 * time.Second); err != nil {
		s.Close()
		return nil, fmt.Errorf("localllm: server never became healthy: %w", err)
	}
	// Boot canary: a stack that loads fine can still decode garbage (seen with
	// the in-process yzma path on Linux: backtick/quote spam). One in-family
	// generation proves the engine computes coherently; a degenerate result
	// disables local solving for the run so the Fireworks baseline takes over
	// instead of burning the 10-minute budget on 18 doomed generations. The
	// check targets DEGENERACY, not correctness - the per-task gates own that.
	cctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	canary, cerr := s.generate(cctx, "A depot starts with 500 units. It ships 20% of stock, then receives 50 units. How many units remain?")
	degenerate := cerr != nil || strings.TrimSpace(canary) == "" || len(canary) > 1600 ||
		strings.Contains(canary, "``````") || strings.Contains(canary, "\"\n\"\n\"") ||
		strings.Count(canary, "```python") > 2
	if degenerate {
		s.Close()
		return nil, fmt.Errorf("localllm: boot canary failed (degenerate or erroring decode): err=%v out=%.80q", cerr, canary)
	}
	return s, nil
}

func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

func (s *Solver) waitHealthy(budget time.Duration) error {
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
			return errors.New("llama-server exited during startup")
		}
		time.Sleep(500 * time.Millisecond)
	}
	return errors.New("health check timed out")
}

func (s *Solver) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.proc != nil && s.proc.Process != nil {
		_ = s.proc.Process.Kill()
		_, _ = s.proc.Process.Wait()
	}
}

// semanticClueMarkers flag logic clues that need world knowledge to encode.
// Families not yet covered by the SFT data get skipped locally - encoding
// them wrong yields unique-looking wrong solutions that grounding cannot
// catch. 'allergic' stays listed: the v2e3b weights still mis-encode the
// fur-allergy exclusion sometimes (seen on amd64 validation); prune only
// after a weights release passes allergy variants through the gates.
var semanticClueMarkers = []string{"allergic", "vegetarian", "afraid of"}

// SolveTask runs the two-turn tool contract for a single task prompt, with
// one gate-guided retry: greedy decoding is deterministic, so appending the
// rejection reason to the prompt is the cheapest way to get a different -
// and usually corrected - attempt for zero Fireworks tokens.
func (s *Solver) SolveTask(ctx context.Context, prompt string) Result {
	lp := strings.ToLower(prompt)
	for _, m := range semanticClueMarkers {
		if strings.Contains(lp, m) {
			return Result{Reason: "semantic clue (" + m + ") outside trained families"}
		}
	}
	res := s.solveOnce(ctx, prompt)
	if res.OK || res.Reason == "" || ctx.Err() != nil {
		return res
	}
	retry := s.solveOnce(ctx, prompt+"\n\nCheck carefully before answering: "+res.Reason+".")
	// Degeneracy guard: a retry that computed LESS than the first attempt
	// (fewer values in stdout) dodged the gate rather than fixing the code -
	// e.g. printing a scalar where the first attempt produced the asked list.
	if retry.OK && len(numRe.FindAllString(retry.Stdout, -1)) >= len(numRe.FindAllString(res.Stdout, -1)) {
		return retry
	}
	return res
}

func (s *Solver) solveOnce(ctx context.Context, prompt string) Result {
	deadline := time.Now().Add(s.cfg.Timeout)
	tctx, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()

	first, err := s.generate(tctx, prompt)
	if err != nil {
		return Result{Reason: "generate tool turn: " + err.Error()}
	}
	code, perr := extractToolCode(first)
	if perr != "" {
		return Result{Reason: perr}
	}
	stdout, err := runPython(tctx, code)
	if err != nil {
		return Result{Code: code, Reason: "python: " + err.Error()}
	}
	if strings.TrimSpace(stdout) == "" {
		return Result{Code: code, Reason: "python produced no output"}
	}
	after := fmt.Sprintf("%s\n\nrun_python result:\n%s\n\nReturn the final answer only.",
		strings.TrimSpace(prompt), strings.TrimSpace(stdout))
	answer, err := s.generate(tctx, after)
	if err != nil {
		return Result{Code: code, Stdout: stdout, Reason: "generate final turn: " + err.Error()}
	}
	answer = strings.TrimSpace(answer)
	if answer == "" || strings.Contains(answer, "<tool_call>") || strings.Contains(answer, "```") {
		return Result{Code: code, Stdout: stdout, Reason: "final turn not a plain answer"}
	}
	if reason := answerGroundedIn(answer, stdout); reason != "" {
		return Result{Answer: answer, Code: code, Stdout: stdout, Reason: reason}
	}
	if reason := answerPlausible(prompt, answer); reason != "" {
		return Result{Answer: answer, Code: code, Stdout: stdout, Reason: reason}
	}
	if reason := listEcho(answer, stdout); reason != "" {
		return Result{Answer: answer, Code: code, Stdout: stdout, Reason: reason}
	}
	return Result{Answer: answer, Code: code, Stdout: stdout, OK: true}
}

// answerPlausible rejects answers that are grounded in the tool output but
// obviously wrong or incomplete for the question shape. Grounding proves the
// answer copied the code's result; these checks catch the code being wrong.
func answerPlausible(prompt, answer string) string {
	lp := strings.ToLower(prompt)
	countQuestion := strings.Contains(lp, "how many") || strings.Contains(lp, "how much") ||
		strings.Contains(lp, "remain") || strings.Contains(lp, "total cost") || strings.Contains(lp, "how far")
	if countQuestion {
		if m := numRe.FindString(strings.ReplaceAll(answer, ",", "")); m != "" {
			idx := strings.Index(strings.ReplaceAll(answer, ",", ""), m)
			if idx > 0 && strings.ReplaceAll(answer, ",", "")[idx-1] == '-' {
				return "negative quantity for a count/amount question"
			}
		}
	}
	if strings.Contains(lp, "total cost") && !strings.Contains(answer, "$") && !strings.Contains(strings.ToLower(answer), "cost") {
		return "cost part missing from answer"
	}
	// Multi-question prompts need multi-part answers: a numeric answer must
	// carry at least two numbers when the prompt asks two questions.
	if strings.Count(prompt, "?") >= 2 {
		nums := numRe.FindAllString(strings.ReplaceAll(answer, ",", ""), -1)
		if len(nums) == 1 {
			return "single value answering a multi-question prompt"
		}
	}
	if reason := meetingTimeBound(prompt, answer); reason != "" {
		return reason
	}
	if reason := magnitudeBound(prompt, answer); reason != "" {
		return reason
	}
	return ""
}

// magnitudeBound rejects answers whose largest value sits orders of magnitude
// below every large figure in the prompt - the classic dropped-thousands /
// wrong-scale code error (projecting $238,049 as $2,380).
func magnitudeBound(prompt, answer string) string {
	maxOf := func(s string) float64 {
		best := 0.0
		for _, t := range numRe.FindAllString(strings.ReplaceAll(s, ",", ""), -1) {
			if v, err := strconv.ParseFloat(strings.TrimRight(t, "."), 64); err == nil && v > best {
				best = v
			}
		}
		return best
	}
	pmax, amax := maxOf(prompt), maxOf(answer)
	if pmax >= 10_000 && amax > 0 && amax < pmax/20 {
		return fmt.Sprintf("answer magnitude %.0f far below prompt scale %.0f", amax, pmax)
	}
	// Upper bound: quantity answers cannot dwarf every figure in the prompt
	// (a remaining-stock answer of 16,224 against a 3,200-unit prompt is a
	// wrong-code artefact, not arithmetic).
	if pmax >= 100 && amax > pmax*5 {
		return fmt.Sprintf("answer magnitude %.0f far above prompt scale %.0f", amax, pmax)
	}
	return ""
}

var listRe = regexp.MustCompile(`\[[^\[\]]{1,120}\]`)

// listEcho: when the tool output's first line is a list literal, the answer
// must reproduce it - collapsing [13, 13, 13] to '13' loses the asked shape.
func listEcho(answer, stdout string) string {
	first := strings.TrimSpace(strings.Split(strings.TrimSpace(stdout), "\n")[0])
	m := listRe.FindString(first)
	if m == "" {
		return ""
	}
	squash := func(s string) string { return strings.ReplaceAll(strings.ReplaceAll(s, " ", ""), "'", "\"") }
	if !strings.Contains(squash(answer), squash(m)) {
		return fmt.Sprintf("answer does not echo the list %s from tool output", m)
	}
	return ""
}

var (
	speedRe = regexp.MustCompile(`(\d+)\s*km/h`)
	depRe   = regexp.MustCompile(`at (\d{1,2}):(\d{2})`)
	clockRe = regexp.MustCompile(`\b(\d{1,2}):(\d{2})(?::\d{2})?\b`)
	distRe  = regexp.MustCompile(`(\d+)\s*km\b`)
)

// meetingTimeBound sanity-checks two-vehicle meeting answers with pure
// physics: the meeting happens strictly before either vehicle alone could
// cover the full distance from its own departure. Applies only when the
// prompt parses cleanly as that shape; any parse doubt skips the check.
func meetingTimeBound(prompt, answer string) string {
	speeds := speedRe.FindAllStringSubmatch(prompt, -1)
	deps := depRe.FindAllStringSubmatch(prompt, -1)
	if len(speeds) != 2 || len(deps) != 2 {
		return ""
	}
	var dist int
	for _, m := range distRe.FindAllStringSubmatch(prompt, -1) {
		v := atoiSafe(m[1])
		if v != atoiSafe(speeds[0][1]) && v != atoiSafe(speeds[1][1]) && v > dist {
			dist = v
		}
	}
	if dist == 0 {
		return ""
	}
	ans := clockRe.FindStringSubmatch(answer)
	if ans == nil {
		// The prompt parsed as a two-vehicle meeting question, so a meeting
		// answer without any parseable clock time is itself disqualifying -
		// skipping here would wave nonsense times straight through.
		return "meeting answer lacks a parseable clock time"
	}
	if atoiSafe(ans[1]) >= 24 || atoiSafe(ans[2]) >= 60 {
		return fmt.Sprintf("malformed clock time %s", ans[0])
	}
	meet := atoiSafe(ans[1])*60 + atoiSafe(ans[2])
	latestDep, bound := 0, 1<<30
	for i := range deps {
		depMin := atoiSafe(deps[i][1])*60 + atoiSafe(deps[i][2])
		if depMin > latestDep {
			latestDep = depMin
		}
		solo := depMin + int(float64(dist)/float64(atoiSafe(speeds[i][1]))*60)
		if solo < bound {
			bound = solo
		}
	}
	if meet <= latestDep || meet >= bound {
		return fmt.Sprintf("meeting time %s outside physical bounds (%02d:%02d..%02d:%02d)",
			ans[0], latestDep/60, latestDep%60, bound/60, bound%60)
	}
	// Self-consistency: the claimed distance from A must equal v_A x elapsed_A
	// for the claimed meeting time (2% tolerance). Speeds and departures pair
	// positionally: the first speed belongs to the first departure (train A).
	depA := atoiSafe(deps[0][1])*60 + atoiSafe(deps[0][2])
	vA := float64(atoiSafe(speeds[0][1]))
	expected := vA * float64(meet-depA) / 60
	tol := 0.02 * float64(dist)
	consistent := false
	for _, t := range numRe.FindAllString(strings.ReplaceAll(answer, ",", ""), -1) {
		v, err := strconv.ParseFloat(strings.TrimRight(t, "."), 64)
		if err != nil {
			continue
		}
		if math.Abs(v-expected) <= tol {
			consistent = true
			break
		}
	}
	if !consistent {
		return fmt.Sprintf("no distance in answer consistent with meeting time (expect ~%.1f km)", expected)
	}
	return ""
}

func atoiSafe(s string) int {
	v, _ := strconv.Atoi(s)
	return v
}

var fenceRe = regexp.MustCompile("(?s)```(?:python)?[ \t]*\n(.*?)\n?```")

// extractToolCode pulls the Python out of a tool turn. The current contract is
// a fenced block (raw code lines, nothing to escape); the legacy <tool_call>
// JSON wrapper is still recognised - strictly, then via balanced-brace
// extraction, because GGUF conversion mangled its </tool_call> special token.
func extractToolCode(turn string) (string, string) {
	if m := fenceRe.FindStringSubmatch(turn); m != nil {
		if code := strings.TrimSpace(m[1]); code != "" {
			return code, ""
		}
		return "", "empty fenced block"
	}
	if calls := message.ParseToolCalls(turn); len(calls) == 1 && calls[0].Function.Name == "run_python" {
		if code := strings.TrimSpace(calls[0].Function.Arguments["code"]); code != "" {
			return code, ""
		}
		return "", "empty code argument"
	}
	open := strings.Index(turn, "<tool_call>")
	if open < 0 {
		return "", "no fenced block or tool call in turn"
	}
	rest := turn[open+len("<tool_call>"):]
	start := strings.IndexByte(rest, '{')
	if start < 0 {
		return "", "tool call has no JSON payload"
	}
	depth, end, inStr, esc := 0, -1, false, false
	for i := start; i < len(rest); i++ {
		c := rest[i]
		switch {
		case esc:
			esc = false
		case inStr && c == '\\':
			esc = true
		case c == '"':
			inStr = !inStr
		case !inStr && c == '{':
			depth++
		case !inStr && c == '}':
			depth--
			if depth == 0 {
				end = i
			}
		}
		if end >= 0 {
			break
		}
	}
	if end < 0 {
		return "", "tool call JSON not brace-balanced (truncated?)"
	}
	var payload struct {
		Name      string `json:"name"`
		Arguments struct {
			Code string `json:"code"`
		} `json:"arguments"`
	}
	if err := jsonUnmarshal(rest[start:end+1], &payload); err != nil {
		return "", "tool call JSON invalid: " + err.Error()
	}
	if payload.Name != "run_python" || strings.TrimSpace(payload.Arguments.Code) == "" {
		return "", "tool call is not a run_python with code"
	}
	return strings.TrimSpace(payload.Arguments.Code), ""
}

var numRe = regexp.MustCompile(`\d[\d,]*\.?\d*`)

// answerGroundedIn is the hallucination firewall: every number in the final
// answer must be derivable from the tool stdout - equal to a stdout value, or
// a display-rounding of one (so nothing was invented) - and at least one
// stdout number must be used (so the tool result was not ignored). Non-numeric
// answers (logic assignments) must echo the stdout. Returns "" when grounded,
// else the rejection reason.
func answerGroundedIn(answer, stdout string) string {
	norm := func(s string) string { return strings.ReplaceAll(s, ",", "") }
	ansTok := numRe.FindAllString(norm(answer), -1)
	outTok := numRe.FindAllString(norm(stdout), -1)
	if len(ansTok) > 0 {
		outVals := make([]float64, 0, len(outTok))
		for _, t := range outTok {
			if v, err := strconv.ParseFloat(strings.TrimRight(t, "."), 64); err == nil {
				outVals = append(outVals, v)
			}
		}
		outDecimals := make([]int, len(outVals))
		for i, t := range outTok {
			t = strings.TrimRight(t, ".")
			if j := strings.IndexByte(t, '.'); j >= 0 {
				outDecimals[i] = len(t) - j - 1
			}
		}
		used := false
		for _, t := range ansTok {
			t = strings.TrimRight(t, ".")
			a, err := strconv.ParseFloat(t, 64)
			if err != nil {
				continue
			}
			decimals := 0
			if i := strings.IndexByte(t, '.'); i >= 0 {
				decimals = len(t) - i - 1
			}
			scale := math.Pow(10, float64(decimals))
			matched := false
			for i, o := range outVals {
				if math.Abs(o-a) < 1e-9 {
					matched = true
					break
				}
				// Rounding is only legitimate against raw float tails; stdout
				// values already formatted to <=3 decimals are display-final,
				// and re-rounding them (1.625 -> 1.62) loses required precision.
				if outDecimals[i] > 3 && math.Abs(math.Round(o*scale)/scale-a) < 1e-9 {
					matched = true
					break
				}
			}
			if matched {
				used = true
				continue
			}
			if len(strings.TrimLeft(t, "0")) >= 2 { // lone digits (part labels, list positions) are formatting
				return fmt.Sprintf("answer number %q not derivable from tool output", t)
			}
		}
		if !used && len(outVals) > 0 {
			return "answer ignores tool output numbers"
		}
		return ""
	}
	// no numbers: logic-style answer - require some literal overlap with stdout
	firstLine := strings.TrimSpace(strings.Split(strings.TrimSpace(stdout), "\n")[0])
	probe := firstLine[:min(12, len(firstLine))]
	if probe != "" && !strings.Contains(strings.ToLower(answer), strings.ToLower(probe)) {
		return "answer does not echo tool output"
	}
	return ""
}

func (s *Solver) generate(ctx context.Context, userPrompt string) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	// Build the prompt in the EXACT training render (verified by
	// assert_minicpm5_template_contract in finetune/minicpm5/train_trl.py):
	// the official MiniCPM5 no-think format, <s> BOS included, thinking block
	// empty. llama-server tokenises string prompts with special-token parsing,
	// so the ChatML markers become their reserved ids exactly as in training.
	prompt := "<s><|im_start|>system\n" + systemPrompt + "<|im_end|>\n" +
		"<|im_start|>user\n" + strings.TrimSpace(userPrompt) + "<|im_end|>\n" +
		"<|im_start|>assistant\n<think>\n\n</think>\n\n"
	body, _ := json.Marshal(map[string]any{
		"prompt":       prompt,
		"n_predict":    s.cfg.MaxGen,
		"temperature":  0,
		"cache_prompt": false,
		"stop":         []string{"<|im_end|>"},
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.baseURL+"/completion", bytes.NewReader(body))
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
		Content string `json:"content"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", fmt.Errorf("decode /completion response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("/completion status %d", resp.StatusCode)
	}
	return strings.TrimSpace(out.Content), nil
}

func runPython(ctx context.Context, code string) (string, error) {
	pctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(pctx, "python3", "-c", code)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	if pctx.Err() == context.DeadlineExceeded {
		return "", pctx.Err()
	}
	if err != nil {
		return strings.TrimSpace(stdout.String()), fmt.Errorf("%w: %s", err, strings.TrimSpace(stderr.String()))
	}
	return strings.TrimSpace(stdout.String()), nil
}

func jsonUnmarshal(s string, v any) error { return json.Unmarshal([]byte(s), v) }

// MeetingTimeBound exposes the two-vehicle physics check for remote answers
// too: the Fireworks path occasionally ships impossible meeting times
// (03:04.5 for a 09:30 departure), and the bound is engine-agnostic.
func MeetingTimeBound(prompt, answer string) string {
	return meetingTimeBound(prompt, answer)
}
