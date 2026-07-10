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
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Families the base lane accepts. With the PLAIN base model only ner and
// code_generation are safe (sentiment/summarisation/factual judge at 2/6, 1/3,
// 2/3 on the variant probe - rationale and rubric failures no cheap gate can
// see). The v3 assist fine-tune (judge-filtered teacher SFT) unlocks the other
// three; they activate only when Config.Extended is set.
const (
	FamilyCodeGen       = "code_generation"
	FamilyNER           = "named_entity_recognition"
	FamilySentiment     = "sentiment_classification"
	FamilySummarisation = "text_summarisation"
	FamilyFactual       = "factual_knowledge"
	// FamilyCodeFix is correction-style code_debugging ("supposed to X but
	// contains a bug"); eval-style tracing is FamilyCodeTrace.
	FamilyCodeFix = "code_fix"
)

// directInstructions are terse per-family system prompts (the track-1 inspo
// ROUTES pattern): output tokens are free locally, but short answers keep the
// judge focused on content instead of stray claims.
// directInstructions must stay byte-identical to the SYSTEM prompts in
// scripts/build_minicpm5_assist_data.py: the assist model is trained on the
// exact render it is served with.
var directInstructions = map[string]string{
	FamilyCodeGen:       "Write exactly the code requested, nothing extra. Code only, minimal comments, no explanation or demos unless asked.",
	FamilyNER:           "Extract the entities in exactly the requested format, nothing else. Include every entity present; omissions are failures.",
	FamilyCodeTrace:     "State the exact evaluated values first (copy the ground-truth values verbatim), then explain the mechanism in one or two sentences (late binding, shared mutable default, iterator exhaustion, aliasing, wrong base case). Be brief.",
	FamilySentiment:     "Classify the sentiment and justify it in one accurate sentence. Never contradict the text.",
	FamilySummarisation: "Obey the stated length and format limits exactly; cover every major theme; no preamble.",
	FamilyFactual:       "Answer directly and concisely; explanations max 3 short sentences.",
	FamilyCodeFix:       "State the one-line cause of the bug (what the buggy line actually does), then provide the minimal corrected function. Plain code, no fences, change only the bug.",
}

var directMaxGen = map[string]int{
	FamilyCodeGen:       512,
	FamilyNER:           320,
	FamilyCodeTrace:     350,
	FamilySentiment:     120,
	FamilySummarisation: 260,
	FamilyFactual:       200,
	FamilyCodeFix:       420,
}

// extendedFamilies are served only by the assist fine-tune (Config.Extended).
var extendedFamilies = map[string]bool{
	FamilySentiment:     true,
	FamilySummarisation: true,
	FamilyFactual:       true,
	FamilyCodeFix:       true,
}

// DirectInstruction exposes the serving system prompt for a family ("" when
// the base lane does not serve it) so eval harnesses probe with the exact
// render production uses.
func DirectInstruction(family string) string { return directInstructions[family] }

type DirectSolver struct {
	proc     *exec.Cmd
	baseURL  string
	client   *http.Client
	timeout  time.Duration
	extended map[string]bool // assist-tuned families unlocked for this model
	mu       sync.Mutex      // -np 1 server: one request at a time
}

// parseExtended expands Config.Extended into the family set it unlocks.
func parseExtended(spec string) map[string]bool {
	out := map[string]bool{}
	spec = strings.TrimSpace(spec)
	if spec == "" || spec == "0" {
		return out
	}
	if spec == "1" || strings.EqualFold(spec, "all") {
		for f := range extendedFamilies {
			out[f] = true
		}
		return out
	}
	for _, f := range strings.Split(spec, ",") {
		f = strings.TrimSpace(f)
		if extendedFamilies[f] {
			out[f] = true
		}
	}
	return out
}

// NewDirect spawns a llama-server for the base GGUF with thinking disabled.
// MiniCPM5 is a hybrid-thinking model: without --reasoning off the 1B spends
// its whole budget inside <think> and content comes back empty.
func NewDirect(cfg Config) (*DirectSolver, error) {
	if strings.TrimSpace(cfg.ModelPath) == "" {
		return nil, errors.New("localllm: empty base model path")
	}
	if cfg.Threads <= 0 {
		cfg.Threads = runtime.GOMAXPROCS(0) // container-quota-aware; judge VM is 2 vCPU
		if cfg.Threads > 6 {
			cfg.Threads = 6
		}
	}
	if cfg.CtxSize == 0 {
		cfg.CtxSize = 2048 // assist prompts+gen fit; halves KV on the 4 GB judge VM
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
		// 4 GB judge VM: q8_0 KV halves cache memory (needs flash-attn for V),
		// and a small microbatch shrinks compute buffers at negligible speed
		// cost for our short prompts.
		"--flash-attn", "on",
		"--cache-type-k", "q8_0",
		"--cache-type-v", "q8_0",
		"-ub", "256",
	)
	cmd.Env = append(cmd.Environ(), "LD_LIBRARY_PATH="+cfg.LibPath)
	cmd.Stdout = nil
	cmd.Stderr = nil
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("localllm: start base llama-server: %w", err)
	}
	s := &DirectSolver{
		proc:     cmd,
		baseURL:  "http://127.0.0.1:" + strconv.Itoa(port),
		client:   &http.Client{Timeout: 3 * time.Minute},
		timeout:  cfg.Timeout,
		extended: parseExtended(cfg.Extended),
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

// SolveTask answers one prompt for a supported family and gates the result,
// with one gate-guided retry: greedy decoding is deterministic, so appending
// the rejection reason is the cheapest way to get a different - and usually
// corrected - attempt for zero Fireworks tokens (same pattern as the tool
// lane). Observed recoverable rejects: sentence counts, per-bullet word caps,
// label rules, entity omissions, code that fails to parse.
// OK=false means the caller must leave the task in the remote batch.
func (s *DirectSolver) SolveTask(ctx context.Context, prompt, family string) Result {
	if _, ok := directInstructions[family]; !ok {
		return Result{Reason: "family " + family + " not served by base lane"}
	}
	if extendedFamilies[family] && !s.extended[family] {
		return Result{Reason: "family " + family + " not unlocked (LOCAL_BASE_EXTENDED)"}
	}
	res := s.solveGated(ctx, prompt, family)
	if res.OK && family == FamilySentiment {
		// Self-consistency: sentiment labels are unverifiable by any cheap
		// gate, and the observed failure is a clean flip (a mixed-positive
		// review labelled Negative). A second, verdict-focused phrasing takes
		// a different greedy path; agreement is strong evidence, disagreement
		// goes remote. Costs one local generation, zero tokens.
		if reason := s.sentimentConsistent(ctx, prompt, res.Answer); reason != "" {
			return Result{Answer: res.Answer, Reason: reason}
		}
		return res
	}
	if res.OK && family == FamilyNER {
		// NER assignments wobble across runs (ETH Zurich has been labelled
		// LOCATION, DATE, and omitted outright). A second phrasing must agree
		// on the label:span pairs; disagreement goes remote.
		if reason := s.nerConsistent(ctx, prompt, res.Answer); reason != "" {
			return Result{Answer: res.Answer, Reason: reason}
		}
		return res
	}
	if res.OK || res.Reason == "" || ctx.Err() != nil {
		return res
	}
	retry := s.solveGated(ctx, prompt+"\n\nYour previous answer was rejected because: "+res.Reason+". Fix exactly that and answer again in full.", family)
	if retry.OK {
		return retry
	}
	// Keep the first attempt as the fallback candidate: it failed one check,
	// the retry failed too, and the first is the less-prompted of the two.
	return res
}

// nerConsistent re-extracts with a second phrasing and demands the identical
// LABEL:span pair set. Wobble across phrasings is exactly how the observed
// mislabels and omissions present; stable extractions agree.
func (s *DirectSolver) nerConsistent(ctx context.Context, prompt, answer string) string {
	tctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	check, err := s.chat(tctx, prompt+"\n\nList every entity again, one per line, exactly as LABEL: span. Include all of them.", FamilyNER)
	if err != nil {
		return "ner consistency check failed: " + err.Error()
	}
	pairs := func(text string) map[string]bool {
		out := map[string]bool{}
		for _, m := range nerPairRe.FindAllStringSubmatch(text, -1) {
			out[m[1]+"|"+strings.TrimSpace(m[2])] = true
		}
		return out
	}
	first, second := pairs(answer), pairs(check)
	if len(first) == 0 {
		return "ner: no LABEL: span pairs found in answer"
	}
	if len(first) != len(second) {
		return fmt.Sprintf("ner extractions disagree across phrasings (%d vs %d pairs)", len(first), len(second))
	}
	for k := range first {
		if !second[k] {
			return "ner extractions disagree across phrasings on " + k
		}
	}
	return ""
}

var sentLabelRe = regexp.MustCompile(`\b(Positive|Negative|Neutral)\b`)

func (s *DirectSolver) sentimentConsistent(ctx context.Context, prompt, answer string) string {
	first := sentLabelRe.FindString(answer)
	if first == "" {
		return "sentiment: no label found in answer"
	}
	tctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	check, err := s.chat(tctx, prompt+"\n\nConsider only the writer's OVERALL verdict, weighing every praise and complaint. Answer with exactly one word: Positive, Negative, or Neutral.", FamilySentiment)
	if err != nil {
		return "sentiment consistency check failed: " + err.Error()
	}
	second := sentLabelRe.FindString(check)
	if second == "" || second != first {
		return fmt.Sprintf("sentiment labels disagree across phrasings (%s vs %s)", first, second)
	}
	return ""
}

func (s *DirectSolver) solveGated(ctx context.Context, prompt, family string) Result {
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
	if family == FamilyNER {
		// The outside-the-inventory training convention backfired at the
		// judge ("fabricates a nonexistent LOCATION: Outside the inventory"):
		// live reference answers want pure entity lists. Strip the note.
		if i := strings.Index(answer, "Outside the inventory"); i >= 0 {
			answer = strings.TrimSpace(strings.TrimRight(strings.TrimSpace(answer[:i]), ";,-"))
		}
	}
	var reason string
	switch family {
	case FamilyCodeGen:
		reason = gateCodeGen(tctx, prompt, answer)
	case FamilyNER:
		reason = gateNER(prompt, answer)
	case FamilySentiment:
		reason = gateSentiment(answer)
	case FamilySummarisation:
		reason = gateSummarisation(prompt, answer)
	case FamilyFactual:
		reason = gateFactual(answer)
	case FamilyCodeFix:
		reason = gateCodeFix(tctx, prompt, answer)
	}
	if reason != "" {
		return Result{Answer: answer, Reason: reason}
	}
	return Result{Answer: answer, OK: true}
}

var shapeRe = regexp.MustCompile(`exactly (two|three|four|five|\d+) (sentences|bullet points)`)

var wordNum = map[string]int{"two": 2, "three": 3, "four": 4, "five": 5}

// gateSentiment: format only - the label must lead and a justification must
// follow. Label correctness rests on the assist SFT; a malformed answer goes
// remote rather than gambling the judge.
func gateSentiment(answer string) string {
	for _, label := range []string{"Positive", "Negative", "Neutral"} {
		if strings.HasPrefix(answer, label) {
			if len(strings.TrimSpace(strings.TrimPrefix(strings.TrimPrefix(answer, label), "."))) < 15 {
				return "sentiment answer lacks a justification sentence"
			}
			return ""
		}
	}
	return "sentiment answer does not start with a Positive/Negative/Neutral label"
}

var wordCapRe = regexp.MustCompile(`(?i)(?:at most|no more than|no longer than|max(?:imum)?(?: of)?|within|under) (\d+|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty) words`)

var wordNumExt = map[string]int{
	"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
	"nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
}

// gateSummarisation enforces the prompt's own stated shape - the constraints
// the judge fails most often: the item count and any per-item word cap. Both
// are deterministic, so a violation goes remote instead of gambling the judge.
func gateSummarisation(prompt, answer string) string {
	m := shapeRe.FindStringSubmatch(prompt)
	if m == nil {
		return ""
	}
	n := wordNum[m[1]]
	if n == 0 {
		if v, err := strconv.Atoi(m[1]); err == nil {
			n = v
		}
	}
	var units []string
	if m[2] == "bullet points" {
		for _, line := range strings.Split(answer, "\n") {
			t := strings.TrimSpace(line)
			if strings.HasPrefix(t, "-") || strings.HasPrefix(t, "•") || strings.HasPrefix(t, "*") {
				units = append(units, strings.TrimLeft(t, "-•* "))
			}
		}
	} else {
		units = sentenceRe.Split(strings.TrimSpace(answer), -1)
	}
	if len(units) != n {
		return fmt.Sprintf("summary has %d %s, prompt demands exactly %d", len(units), m[2], n)
	}
	if wc := wordCapRe.FindStringSubmatch(prompt); wc != nil {
		cap := wordNumExt[strings.ToLower(wc[1])]
		if cap == 0 {
			if v, err := strconv.Atoi(wc[1]); err == nil {
				cap = v
			}
		}
		for i, u := range units {
			if words := len(strings.Fields(u)); cap > 0 && words > cap {
				return fmt.Sprintf("item %d has %d words, prompt caps at %d", i+1, words, cap)
			}
		}
	}
	if strings.Contains(strings.ToLower(prompt), "must mention the percentage") && !strings.Contains(answer, "%") {
		return "prompt demands the percentage figure and the answer has none"
	}
	return ""
}

var sentenceRe = regexp.MustCompile(`[.!?]\s+`)

func gateFactual(answer string) string {
	if len(sentenceRe.Split(strings.TrimSpace(answer), -1)) > 4 {
		return "factual answer exceeds 4 sentences"
	}
	low := strings.ToLower(answer)
	if strings.Contains(low, "i don't know") || strings.Contains(low, "i cannot") {
		return "factual answer refuses"
	}
	return ""
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

// FamilyCodeTrace is the eval-style code_debugging subset ("what does X
// evaluate to, and why"). The snippet is executed for ground truth before the
// model explains it - hand-tracing Python gotchas is exactly what 1B models
// (and, ~2/6 runs, the remote model at effort none) get wrong.
const FamilyCodeTrace = "code_trace"

var backtickIdentRe = regexp.MustCompile("`([A-Za-z_]\\w*)`")

// SolveCodeTrace answers an eval-style code_debugging task: run the prompt's
// snippet, capture the asked identifiers' real values, generate a short
// explanation with those values in hand, and gate on the answer echoing every
// value. Not-applicable prompts (no backticked identifiers, no code after the
// question) reject to the remote batch.
func (s *DirectSolver) SolveCodeTrace(ctx context.Context, prompt string) Result {
	q := strings.LastIndexByte(prompt, '?')
	if q < 0 || q == len(prompt)-1 {
		return Result{Reason: "code_trace: no question/code split"}
	}
	question, code := prompt[:q+1], strings.TrimSpace(prompt[q+1:])
	idents := uniqueStrings(backtickIdentRe.FindAllStringSubmatch(question, -1))
	if len(idents) == 0 || code == "" {
		return Result{Reason: "code_trace: no backticked identifiers or empty snippet"}
	}
	var sb strings.Builder
	sb.WriteString(code)
	sb.WriteString("\n\n")
	for _, id := range idents {
		fmt.Fprintf(&sb, "print(%q, repr(%s))\n", id+" =", id)
	}
	truth, err := runPython(ctx, sb.String())
	if err != nil || strings.TrimSpace(truth) == "" {
		return Result{Reason: fmt.Sprintf("code_trace: snippet execution failed: %v", err)}
	}
	if hint := mechanismHint(code); hint != "" {
		truth += "\nMechanism: " + hint
	}
	tctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	answer, err := s.chat(tctx, prompt+"\n\nGround truth from ACTUALLY executing the code (trust these values over any hand-trace):\n"+truth, FamilyCodeTrace)
	if err != nil {
		return Result{Reason: "code_trace generate: " + err.Error()}
	}
	answer = strings.TrimSpace(answer)
	squash := func(v string) string { return strings.ReplaceAll(strings.ReplaceAll(v, " ", ""), "'", "\"") }
	sqAns := squash(answer)
	for _, line := range strings.Split(truth, "\n") {
		ident, val, ok := strings.Cut(line, " = ")
		if !ok || strings.HasPrefix(ident, "Mechanism:") {
			continue
		}
		val = squash(strings.TrimSpace(val))
		// The value must appear NEAR its identifier, not merely anywhere: a
		// trace answer that says "total1 is 12" and later "both evaluate to 12"
		// contains every value yet contradicts itself, and the judge fails it.
		sqIdent := squash(strings.TrimSpace(ident))
		found, named := false, false
		for idx := strings.Index(sqAns, sqIdent); idx >= 0; {
			named = true
			window := sqAns[idx:]
			if len(window) > 90+len(val) {
				window = window[:90+len(val)]
			}
			if strings.Contains(window, val) {
				found = true
				break
			}
			next := strings.Index(sqAns[idx+1:], sqIdent)
			if next < 0 {
				break
			}
			idx += 1 + next
		}
		if !named {
			return Result{Answer: answer, Reason: "code_trace: answer never names " + strings.TrimSpace(ident)}
		}
		if !found {
			return Result{Answer: answer, Reason: fmt.Sprintf("code_trace: value %s not stated next to %s", val, strings.TrimSpace(ident))}
		}
	}
	return Result{Answer: answer, OK: true}
}

// mechanismHint names the Python gotcha the snippet exhibits, when one is
// syntactically detectable. Judges fail trace answers that state the right
// value but not the mechanism, and the 1B names it reliably only when told.
func mechanismHint(code string) string {
	switch {
	case strings.Contains(code, "lambda") && (strings.Contains(code, "for ") || strings.Contains(code, " in range")):
		return "late binding - the lambdas capture the loop VARIABLE, not its value, so all of them see its final value"
	case regexp.MustCompile(`def \w+\([^)]*=\s*(\[\]|\{\})`).MatchString(code):
		return "shared mutable default argument - the same default object persists across calls"
	case strings.Contains(code, "(n for") || strings.Contains(code, "(x for") || regexp.MustCompile(`=\s*\(.+for .+in .+\)`).MatchString(code):
		return "generator exhaustion - a generator can be consumed only once; the second consumption sees nothing"
	}
	return ""
}

func uniqueStrings(matches [][]string) []string {
	seen := map[string]bool{}
	var out []string
	for _, m := range matches {
		if !seen[m[1]] {
			seen[m[1]] = true
			out = append(out, m[1])
		}
	}
	return out
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

var defNameRe = regexp.MustCompile(`def\s+([A-Za-z_]\w*)\s*\(`)

// gateCodeFix: the answer must open with a prose cause line, then a corrected
// function that parses, keeps the buggy function's name, and actually differs
// from the buggy source. Correctness beyond that rests on the SFT (every
// training pair is execution-verified); these checks kill the observed
// failure shapes - echoing the buggy code back, code-only answers with no
// cause, and truncated code.
func gateCodeFix(ctx context.Context, prompt, answer string) string {
	m := defNameRe.FindStringSubmatch(prompt)
	if m == nil {
		return "code_fix: prompt has no function definition"
	}
	name := m[1]
	defIdx := strings.Index(answer, "def ")
	if defIdx < 0 {
		return "code_fix: no corrected function in answer"
	}
	if strings.TrimSpace(answer[:defIdx]) == "" {
		return "code_fix: missing cause line before the corrected function"
	}
	code := answer[defIdx:]
	if fence := fenceRe.FindStringSubmatch(answer); fence != nil {
		code = fence[1]
	}
	if !strings.Contains(code, "def "+name) {
		return "code_fix: corrected function renames " + name
	}
	squash := func(v string) string { return strings.Join(strings.Fields(v), " ") }
	buggyIdx := strings.Index(prompt, "def ")
	if buggyIdx >= 0 && squash(code) == squash(prompt[buggyIdx:]) {
		return "code_fix: corrected function is identical to the buggy one"
	}
	pctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(pctx, "python3", "-c", "import ast,sys; ast.parse(sys.stdin.read())")
	cmd.Stdin = strings.NewReader(code)
	if err := cmd.Run(); err != nil {
		return "code_fix: corrected function does not parse"
	}
	return ""
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
	// Common descriptor acronyms (an "AI research lab") are exempt: reference
	// answers do not list them, so demanding them only false-rejects.
	for _, w := range acronymRe.FindAllString(prompt, -1) {
		if nerLabelVocab[w] || noiseAcronyms[w] {
			continue
		}
		if !strings.Contains(answer, w) {
			return "answer omits acronym " + w
		}
	}
	return gateNERLabels(answer)
}

var noiseAcronyms = map[string]bool{
	"AI": true, "ML": true, "IT": true, "LLM": true, "API": true, "GPU": true,
	"CPU": true, "USB": true, "CEO": true, "CTO": true, "CFO": true, "COO": true,
	"GDP": true, "HR": true, "PR": true, "USD": true, "GBP": true, "EUR": true,
}

var (
	nerPairRe    = regexp.MustCompile(`(PERSON|ORGANIZATION|LOCATION|DATE)\s*:\s*([^;\n]+)`)
	acronymLedRe = regexp.MustCompile(`^[A-Z]{2,}\s+[A-Z]`)
)

// gateNERLabels rejects label assignments that are near-certainly wrong on
// surface form alone: acronym spans ("ETH Zurich", "NASA") are organisations,
// not locations, and programme/mission names are not people or places -
// observed live as NASA->LOCATION and Artemis->PERSON, both judge-fatal.
func gateNERLabels(answer string) string {
	for _, m := range nerPairRe.FindAllStringSubmatch(answer, -1) {
		span := strings.TrimSpace(m[2])
		if m[1] == "LOCATION" && (acronymLedRe.MatchString(span) || allCapsRe.MatchString(span)) {
			return "acronym span labelled LOCATION: " + span
		}
		low := strings.ToLower(span)
		if (strings.Contains(low, "programme") || strings.Contains(low, "program") || strings.Contains(low, "mission")) && m[1] != "ORGANIZATION" {
			return "programme/mission span labelled " + m[1] + ": " + span
		}
		// A DATE span must look like a date: digits or a month name. Observed
		// live: "ETH Zurich" labelled DATE.
		if m[1] == "DATE" && !dateShapeRe.MatchString(span) {
			return "non-date span labelled DATE: " + span
		}
	}
	// The outside-the-inventory note is for programmes/missions that fit no
	// label - never for organisations. The v5 weights learnt to relegate
	// acronym-led orgs there ("Outside the inventory: ETH Zurich"), which the
	// judge scores as an omission.
	if i := strings.Index(answer, "Outside the inventory"); i >= 0 {
		note := answer[i:]
		if sp := acronymLedRe.FindString(note); sp != "" || allCapsRe.MatchString(strings.TrimSpace(note)) {
			return "organisation-pattern span relegated outside the inventory"
		}
		for _, w := range acronymRe.FindAllString(note, -1) {
			if !nerLabelVocab[w] && !noiseAcronyms[w] {
				return "acronym " + w + " relegated outside the inventory"
			}
		}
	}
	return ""
}

var dateShapeRe = regexp.MustCompile(`(?i)\d|january|february|march|april|may|june|july|august|september|october|november|december`)

var allCapsRe = regexp.MustCompile(`^[A-Z]{2,}$`)

func firstLineOf(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}
