package micropy

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
	"github.com/tetratelabs/wazero/experimental"
	wasi "github.com/tetratelabs/wazero/imports/wasi_snapshot_preview1"
	"github.com/tetratelabs/wazero/sys"
)

//go:embed micropython-wasi.wasm
var wasmBytes []byte

const hostResultCap = 512 * 1024

type Config struct {
	Timeout     time.Duration
	MemoryBytes int
	WorkingDir  string
}

type Executor struct {
	timeout  time.Duration
	pages    uint32
	cwd      string
	once     sync.Once
	initErr  error
	runtime  wazero.Runtime
	compiled wazero.CompiledModule
	varsMu   sync.Mutex
	vars     map[string]map[string]any
	tools    *ToolHost
}

type Result struct {
	JSON   string         `json:"json"`
	Value  any            `json:"value,omitempty"`
	Vars   map[string]any `json:"vars,omitempty"`
	Stdout string         `json:"stdout,omitempty"`
}

type captured struct {
	JSON  string         `json:"json"`
	Vars  map[string]any `json:"vars"`
	Error string         `json:"error,omitempty"`
}

func New(cfg Config) (*Executor, error) {
	if cfg.Timeout <= 0 {
		cfg.Timeout = 20 * time.Second
	}
	pages := uint32(256)
	if cfg.MemoryBytes > 0 {
		pages = uint32((cfg.MemoryBytes + 65535) / 65536)
	}
	cwd := cfg.WorkingDir
	if cwd == "" {
		cwd, _ = os.Getwd()
	}
	return &Executor{timeout: cfg.Timeout, pages: pages, cwd: cwd, vars: map[string]map[string]any{}, tools: NewToolHost(cwd)}, nil
}

func (e *Executor) Run(ctx context.Context, sessionID, code string, input map[string]any) (Result, error) {
	if strings.TrimSpace(code) == "" {
		return Result{}, fmt.Errorf("micropython code is empty")
	}
	if err := e.load(ctx); err != nil {
		return Result{}, err
	}
	runCtx, cancel := context.WithTimeout(ctx, e.timeout)
	defer cancel()
	host := &hostState{tools: e.tools}
	runCtx = context.WithValue(runCtx, hostKey{}, host)
	transcript, err := e.buildTranscript(sessionID, code, input)
	if err != nil {
		return Result{}, err
	}
	var stdout, stderr strings.Builder
	cfg := wazero.NewModuleConfig().WithName("").WithArgs("micropython", "-c", transcript).WithStdout(&stdout).WithStderr(&stderr)
	mod, err := e.runtime.InstantiateModule(runCtx, e.compiled, cfg)
	if mod != nil {
		_ = mod.Close(runCtx)
	}
	if err != nil {
		if exit, ok := err.(*sys.ExitError); !ok || exit.ExitCode() != 0 {
			if msg := strings.TrimSpace(stderr.String()); msg != "" {
				return Result{Stdout: stdout.String()}, fmt.Errorf("micropython: %s", lastLine(msg))
			}
			return Result{Stdout: stdout.String()}, fmt.Errorf("micropython execution failed: %w", err)
		}
	}
	if host.result == nil {
		if msg := strings.TrimSpace(stderr.String()); msg != "" {
			return Result{Stdout: stdout.String()}, fmt.Errorf("micropython: %s", lastLine(msg))
		}
		return Result{Stdout: stdout.String()}, fmt.Errorf("micropython produced no result")
	}
	var cap captured
	if err := json.Unmarshal(host.result, &cap); err != nil {
		return Result{Stdout: stdout.String()}, err
	}
	if strings.TrimSpace(cap.Error) != "" {
		return Result{JSON: cap.JSON, Vars: cap.Vars, Stdout: stdout.String()}, fmt.Errorf("micropython: %s", cap.Error)
	}
	e.storeVars(sessionID, cap.Vars)
	var value any
	if strings.TrimSpace(cap.JSON) != "" {
		_ = json.Unmarshal([]byte(cap.JSON), &value)
	}
	return Result{JSON: cap.JSON, Value: value, Vars: cap.Vars, Stdout: strings.TrimSpace(stdout.String())}, nil
}

// IsSubmitted returns true if the model called submit() during the last Run.
func (e *Executor) IsSubmitted() bool {
	return e.tools.IsSubmitted()
}

// SubmitResult returns the submitted answers.
func (e *Executor) SubmitResult() map[string]string {
	return e.tools.SubmitResult()
}

// ResetSubmit clears submit state for a new batch.
func (e *Executor) ResetSubmit() {
	e.tools.ResetSubmit()
}

// SetExpectedTasks configures submit() validation for the next runs.
func (e *Executor) SetExpectedTasks(ids []string) {
	e.tools.SetExpectedTasks(ids)
}

func (e *Executor) load(ctx context.Context) error {
	e.once.Do(func() {
		rt := wazero.NewRuntimeWithConfig(ctx, wazero.NewRuntimeConfig().WithCoreFeatures(api.CoreFeaturesV2|experimental.CoreFeaturesExceptionHandling).WithCloseOnContextDone(true).WithMemoryLimitPages(e.pages))
		if _, err := wasi.Instantiate(ctx, rt); err != nil {
			e.initErr = fmt.Errorf("instantiate wasi: %w", err)
			return
		}
		if _, err := rt.NewHostModuleBuilder("micropython_wasm").NewFunctionBuilder().WithFunc(hostResultCapFn).Export("host_result_cap").NewFunctionBuilder().WithFunc(hostCall).Export("host_call").Instantiate(ctx); err != nil {
			e.initErr = fmt.Errorf("instantiate micropython host bridge: %w", err)
			return
		}
		compiled, err := rt.CompileModule(ctx, wasmBytes)
		if err != nil {
			e.initErr = fmt.Errorf("compile micropython wasm: %w", err)
			return
		}
		e.runtime = rt
		e.compiled = compiled
	})
	return e.initErr
}

func (e *Executor) buildTranscript(sessionID, code string, input map[string]any) (string, error) {
	enc := func(v any) (string, error) {
		b, err := json.Marshal(v)
		if err != nil {
			return "", err
		}
		return base64.StdEncoding.EncodeToString(b), nil
	}
	if input == nil {
		input = map[string]any{}
	}
	inputB64, err := enc(input)
	if err != nil {
		return "", err
	}
	varsB64, err := enc(e.loadVars(sessionID))
	if err != nil {
		return "", err
	}
	toolsB64, err := enc(toolSurfaceBindings())
	if err != nil {
		return "", err
	}
	cwdB64, err := enc(e.cwd)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(prelude)
	fmt.Fprintf(&b, "_install_surfaces(_ld(%q))\n", toolsB64)
	fmt.Fprintf(&b, "input = _ld(%q)\n", inputB64)
	fmt.Fprintf(&b, "vars = _ld(%q)\n", varsB64)
	fmt.Fprintf(&b, "cwd = _ld(%q)\n", cwdB64)
	b.WriteString("result = None\n_err = None\ntry:\n")
	for _, line := range strings.Split(code, "\n") {
		b.WriteString("    ")
		b.WriteString(line)
		b.WriteByte('\n')
	}
	b.WriteString("    pass\nexcept Exception as _e:\n    _err = type(_e).__name__ + ': ' + str(_e)\n")
	b.WriteString(epilogue)
	return b.String(), nil
}

func (e *Executor) loadVars(sessionID string) map[string]any {
	out := map[string]any{}
	if sessionID == "" {
		return out
	}
	e.varsMu.Lock()
	defer e.varsMu.Unlock()
	for k, v := range e.vars[sessionID] {
		out[k] = v
	}
	return out
}

func (e *Executor) storeVars(sessionID string, vars map[string]any) {
	if sessionID == "" {
		return
	}
	e.varsMu.Lock()
	defer e.varsMu.Unlock()
	if len(vars) == 0 {
		delete(e.vars, sessionID)
		return
	}
	cloned := make(map[string]any, len(vars))
	for k, v := range vars {
		cloned[k] = v
	}
	e.vars[sessionID] = cloned
}

type hostKey struct{}

type hostState struct {
	result []byte
	tools  *ToolHost
}

func hostResultCapFn(ctx context.Context, mod api.Module) int32 { return hostResultCap }

func hostCall(ctx context.Context, mod api.Module, namePtr, nameLen, payloadPtr, payloadLen, resultPtr, resultCap uint32) int32 {
	host, _ := ctx.Value(hostKey{}).(*hostState)
	if host == nil {
		return -1
	}
	mem := mod.Memory()
	nameBytes, ok := mem.Read(namePtr, nameLen)
	if !ok {
		return -1
	}
	payloadBytes, ok := mem.Read(payloadPtr, payloadLen)
	if !ok {
		return -1
	}
	name := string(nameBytes)
	var resp []byte
	if name == "__result__" {
		host.result = append([]byte(nil), payloadBytes...)
		resp = []byte("{}")
	} else {
		value, err := host.tools.Dispatch(ctx, name, payloadBytes)
		if err != nil {
			resp, _ = json.Marshal(map[string]any{"error": err.Error()})
		} else {
			resp, _ = json.Marshal(value)
		}
	}
	if uint32(len(resp)) > resultCap {
		return int32(len(resp))
	}
	if !mem.Write(resultPtr, resp) {
		return -1
	}
	return int32(len(resp))
}

type ToolHost struct {
	cwd   string
	mu    sync.Mutex
	terms map[string]*terminalRun
	pages map[string]string
	next  int

	// submit captures the model's final answers when it calls submit().
	submitMu        sync.Mutex
	submitted       bool
	submitResult    map[string]string // task_id -> answer
	submitError     string
	expectedTaskIDs map[string]bool // set by agent before Run; used for validation
}

type terminalRun struct {
	cmd     *exec.Cmd
	out     bytes.Buffer
	started time.Time
	ended   time.Time
	exit    *int
	err     error
}

func NewToolHost(cwd string) *ToolHost {
	return &ToolHost{cwd: cwd, terms: map[string]*terminalRun{}, pages: map[string]string{}}
}

// doSubmit is called when the model calls submit() in MicroPython.
// It validates that every task_id in the batch has an answer and stores
// the result. The agent loop checks IsSubmitted() after each code execution.
func (h *ToolHost) doSubmit(args map[string]any) (any, error) {
	answersRaw, ok := args["answers"]
	if !ok {
		return nil, fmt.Errorf("submit() requires an 'answers' argument: a list of {task_id, answer} dicts")
	}
	arr, ok := answersRaw.([]any)
	if !ok {
		return nil, fmt.Errorf("'answers' must be a list of dicts")
	}
	h.submitMu.Lock()
	taskIDs := map[string]bool{}
	for id := range h.expectedTaskIDs {
		taskIDs[id] = true
	}
	h.submitMu.Unlock()
	if len(taskIDs) == 0 {
		// Fallback: accept tasks list from call args if agent did not pre-set IDs.
		if tasksRaw, ok := args["tasks"].([]any); ok {
			for _, item := range tasksRaw {
				if m, ok := item.(map[string]any); ok {
					if id, _ := m["task_id"].(string); id != "" {
						taskIDs[id] = true
					}
				}
			}
		}
	}
	result := map[string]string{}
	for i, item := range arr {
		m, ok := item.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("answers[%d] is not a dict", i)
		}
		taskID, _ := m["task_id"].(string)
		if taskID == "" {
			return nil, fmt.Errorf("answers[%d] has no task_id", i)
		}
		if len(taskIDs) > 0 && !taskIDs[taskID] {
			return nil, fmt.Errorf("answers[%d] has unknown task_id %q", i, taskID)
		}
		answer := strings.TrimSpace(stringifyForSubmit(m["answer"]))
		if answer == "" {
			return nil, fmt.Errorf("answers for task_id %q is empty", taskID)
		}
		result[taskID] = answer
	}
	if len(taskIDs) > 0 && len(result) != len(taskIDs) {
		missing := make([]string, 0, len(taskIDs))
		for id := range taskIDs {
			if _, ok := result[id]; !ok {
				missing = append(missing, id)
			}
		}
		return nil, fmt.Errorf("submit() missing answers for task_id(s): %s", strings.Join(missing, ","))
	}
	h.submitMu.Lock()
	h.submitted = true
	h.submitResult = result
	h.submitError = ""
	h.submitMu.Unlock()
	return map[string]any{"ok": true, "count": len(result)}, nil
}

// IsSubmitted returns true if the model has called submit() with valid answers.
func (h *ToolHost) IsSubmitted() bool {
	h.submitMu.Lock()
	defer h.submitMu.Unlock()
	return h.submitted
}

// SubmitResult returns the submitted answers (task_id -> answer).
func (h *ToolHost) SubmitResult() map[string]string {
	h.submitMu.Lock()
	defer h.submitMu.Unlock()
	return h.submitResult
}

// ResetSubmit clears the submit state for a new batch.
func (h *ToolHost) ResetSubmit() {
	h.submitMu.Lock()
	h.submitted = false
	h.submitResult = nil
	h.submitError = ""
	h.submitMu.Unlock()
}

// SetExpectedTasks configures which task_ids submit()/finish() must cover.
func (h *ToolHost) SetExpectedTasks(ids []string) {
	h.submitMu.Lock()
	defer h.submitMu.Unlock()
	if len(ids) == 0 {
		h.expectedTaskIDs = nil
		return
	}
	h.expectedTaskIDs = make(map[string]bool, len(ids))
	for _, id := range ids {
		if id != "" {
			h.expectedTaskIDs[id] = true
		}
	}
}

// doAnswer merges a single task answer into the submit buffer without completing.
// Useful for multi-step batches: answers(task_id=..., answer=...) then finish().
func (h *ToolHost) doAnswer(args map[string]any) (any, error) {
	taskID, _ := args["task_id"].(string)
	if taskID == "" {
		return nil, fmt.Errorf("answers(task_id=..., answer=...) requires task_id")
	}
	h.submitMu.Lock()
	if len(h.expectedTaskIDs) > 0 && !h.expectedTaskIDs[taskID] {
		h.submitMu.Unlock()
		return nil, fmt.Errorf("unknown task_id %q", taskID)
	}
	h.submitMu.Unlock()
	answer := strings.TrimSpace(stringifyForSubmit(args["answer"]))
	if answer == "" {
		return nil, fmt.Errorf("answer for task_id %q is empty", taskID)
	}
	h.submitMu.Lock()
	if h.submitResult == nil {
		h.submitResult = map[string]string{}
	}
	h.submitResult[taskID] = answer
	// Only mark submitted if all expected ids are present.
	complete := len(h.expectedTaskIDs) > 0
	if complete {
		for id := range h.expectedTaskIDs {
			if strings.TrimSpace(h.submitResult[id]) == "" {
				complete = false
				break
			}
		}
	}
	if complete {
		h.submitted = true
	}
	count := len(h.submitResult)
	h.submitMu.Unlock()
	return map[string]any{"ok": true, "task_id": taskID, "buffered": count, "complete": complete}, nil
}

func stringifyForSubmit(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case nil:
		return ""
	case float64:
		if x == float64(int64(x)) {
			return fmt.Sprintf("%d", int64(x))
		}
		return fmt.Sprintf("%v", x)
	case bool:
		if x {
			return "true"
		}
		return "false"
	default:
		b, _ := json.Marshal(x)
		return string(b)
	}
}

func (h *ToolHost) Dispatch(ctx context.Context, name string, payload []byte) (any, error) {
	args := map[string]any{}
	if len(bytes.TrimSpace(payload)) > 0 {
		_ = json.Unmarshal(payload, &args)
	}
	switch name {
	case "fs.read":
		return h.fsRead(args)
	case "fs.write":
		return h.fsWrite(args)
	case "fs.edit":
		return h.fsEdit(args)
	case "fs.list":
		return h.fsList(args)
	case "sh.run":
		return h.shRun(ctx, args)
	case "web.fetch":
		return h.webFetch(ctx, args)
	case "web.search":
		return h.webSearch(ctx, args)
	case "browser.drive":
		return h.browserDrive(ctx, args)
	case "submit", "finish":
		// finish is the preferred terminal action; submit is kept as an alias.
		return h.doSubmit(args)
	case "answers":
		// answers(task_id=..., answer=...) is a one-task helper that merges into
		// the current submit buffer (does not complete the batch alone).
		return h.doAnswer(args)
	case "tools.help":
		return map[string]any{"tools": toolSurfaceBindings()}, nil
	default:
		return nil, fmt.Errorf("unknown MicroPython action-space tool %q", name)
	}
}

func toolSurfaceBindings() []map[string]any {
	return []map[string]any{
		{"tool": "fs.read", "namespace": []string{"fs"}, "member": "read", "args": "path string, optional offset int bytes, limit int bytes", "example": "fs.read(path='README.md', limit=4000)"},
		{"tool": "fs.write", "namespace": []string{"fs"}, "member": "write", "args": "path string, content string", "example": "fs.write(path='/tmp/note.txt', content='hello')"},
		{"tool": "fs.edit", "namespace": []string{"fs"}, "member": "edit", "args": "path string, oldText string that appears exactly once, newText string", "example": "fs.edit(path='x.txt', oldText='old', newText='new')"},
		{"tool": "fs.list", "namespace": []string{"fs"}, "member": "list", "args": "optional path string, limit int", "example": "fs.list(path='.', limit=50)"},
		{"tool": "sh.run", "namespace": []string{"sh"}, "member": "run", "args": "action run|status|output|wait|cancel|list, command string for run, async bool, terminal_id string, timeout int seconds, offset int, limit int", "example": "t=sh.run(command='python3 -c \"print(6*7)\"'); result=t['output'].strip()"},
		{"tool": "web.fetch", "namespace": []string{"web"}, "member": "fetch", "args": "url string, optional limit int bytes", "example": "web.fetch(url='https://example.com', limit=20000)"},
		{"tool": "web.search", "namespace": []string{"web"}, "member": "search", "args": "query string", "example": "web.search(query='AMD Developer Hackathon')"},
		{"tool": "browser.drive", "namespace": []string{"browser"}, "member": "drive", "args": "action string, url string for navigate/new_page", "example": "browser.drive(action='navigate', url='https://example.com')"},
		{"tool": "finish", "namespace": []string{}, "member": "finish", "args": "answers list of {task_id, answer} dicts covering EVERY batch task", "example": "finish(answers=[{'task_id':'t1','answer':'42'},{'task_id':'t2','answer':'Paris'}])"},
		{"tool": "submit", "namespace": []string{}, "member": "submit", "args": "alias of finish(answers=[...])", "example": "submit(answers=[{'task_id':'t1','answer':'42'}])"},
		{"tool": "answers", "namespace": []string{}, "member": "answers", "args": "task_id string, answer any - buffer one answer; call finish when all ready", "example": "answers(task_id='t1', answer='42')"},
		{"tool": "tools.help", "namespace": []string{"tools"}, "member": "help", "args": "none", "example": "tools.help()"},
	}
}

func (h *ToolHost) cleanPath(p string) (string, error) {
	if p == "" {
		return "", fmt.Errorf("path is required")
	}
	if !filepath.IsAbs(p) {
		p = filepath.Join(h.cwd, p)
	}
	return filepath.Clean(p), nil
}

func stringArg(m map[string]any, key string) string { v, _ := m[key].(string); return v }
func intArg(m map[string]any, key string, def int) int {
	if f, ok := m[key].(float64); ok && f > 0 {
		return int(f)
	}
	return def
}
func boolArg(m map[string]any, key string) bool { v, _ := m[key].(bool); return v }

func (h *ToolHost) fsRead(args map[string]any) (any, error) {
	p, err := h.cleanPath(stringArg(args, "path"))
	if err != nil {
		return nil, err
	}
	b, err := os.ReadFile(p)
	if err != nil {
		return nil, err
	}
	limit := intArg(args, "limit", 65536)
	off := intArg(args, "offset", 0)
	if off > len(b) {
		off = len(b)
	}
	end := off + limit
	if end > len(b) {
		end = len(b)
	}
	return map[string]any{"path": p, "content": string(b[off:end]), "bytes_total": len(b), "offset": off, "complete": end == len(b), "next_offset": end}, nil
}

func (h *ToolHost) fsWrite(args map[string]any) (any, error) {
	p, err := h.cleanPath(stringArg(args, "path"))
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return nil, err
	}
	content := stringArg(args, "content")
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		return nil, err
	}
	return map[string]any{"path": p, "bytes": len(content)}, nil
}

func (h *ToolHost) fsEdit(args map[string]any) (any, error) {
	p, err := h.cleanPath(stringArg(args, "path"))
	if err != nil {
		return nil, err
	}
	oldText := stringArg(args, "oldText")
	newText := stringArg(args, "newText")
	b, err := os.ReadFile(p)
	if err != nil {
		return nil, err
	}
	s := string(b)
	count := strings.Count(s, oldText)
	if oldText == "" || count != 1 {
		return nil, fmt.Errorf("oldText must appear exactly once, got %d", count)
	}
	s = strings.Replace(s, oldText, newText, 1)
	if err := os.WriteFile(p, []byte(s), 0o644); err != nil {
		return nil, err
	}
	return map[string]any{"path": p, "edits_applied": 1}, nil
}

func (h *ToolHost) fsList(args map[string]any) (any, error) {
	p := stringArg(args, "path")
	if p == "" {
		p = "."
	}
	p, err := h.cleanPath(p)
	if err != nil {
		return nil, err
	}
	ents, err := os.ReadDir(p)
	if err != nil {
		return nil, err
	}
	limit := intArg(args, "limit", 200)
	out := []map[string]any{}
	for i, ent := range ents {
		if i >= limit {
			break
		}
		info, _ := ent.Info()
		size := int64(0)
		if info != nil {
			size = info.Size()
		}
		out = append(out, map[string]any{"name": ent.Name(), "is_dir": ent.IsDir(), "size": size})
	}
	return map[string]any{"path": p, "entries": out}, nil
}

func (h *ToolHost) shRun(ctx context.Context, args map[string]any) (any, error) {
	action := stringArg(args, "action")
	if action == "" {
		action = "run"
	}
	id := stringArg(args, "terminal_id")
	switch action {
	case "run":
		command := stringArg(args, "command")
		if command == "" {
			return nil, fmt.Errorf("command is required")
		}
		cmdCtx := ctx
		if !boolArg(args, "async") {
			var cancel context.CancelFunc
			cmdCtx, cancel = context.WithTimeout(ctx, time.Duration(intArg(args, "timeout", 30))*time.Second)
			defer cancel()
		}
		cmd := exec.CommandContext(cmdCtx, "/bin/sh", "-c", command)
		cmd.Dir = h.cwd
		r := &terminalRun{cmd: cmd, started: time.Now()}
		cmd.Stdout = &r.out
		cmd.Stderr = &r.out
		h.mu.Lock()
		h.next++
		termID := fmt.Sprintf("term_%d", h.next)
		h.terms[termID] = r
		h.mu.Unlock()
		if boolArg(args, "async") {
			if err := cmd.Start(); err != nil {
				return nil, err
			}
			go h.waitTerm(r)
			return h.termStatus(termID, r), nil
		}
		err := cmd.Run()
		code := cmd.ProcessState.ExitCode()
		r.exit = &code
		r.err = err
		r.ended = time.Now()
		return h.termOutput(termID, r, args), nil
	case "status":
		h.mu.Lock()
		r := h.terms[id]
		h.mu.Unlock()
		if r == nil {
			return nil, fmt.Errorf("unknown terminal_id")
		}
		return h.termStatus(id, r), nil
	case "output":
		h.mu.Lock()
		r := h.terms[id]
		h.mu.Unlock()
		if r == nil {
			return nil, fmt.Errorf("unknown terminal_id")
		}
		return h.termOutput(id, r, args), nil
	case "wait":
		h.mu.Lock()
		r := h.terms[id]
		h.mu.Unlock()
		if r == nil {
			return nil, fmt.Errorf("unknown terminal_id")
		}
		deadline := time.Now().Add(time.Duration(intArg(args, "timeout", 30)) * time.Second)
		for time.Now().Before(deadline) {
			if r.exit != nil {
				return h.termOutput(id, r, args), nil
			}
			time.Sleep(100 * time.Millisecond)
		}
		return h.termStatus(id, r), nil
	case "cancel":
		h.mu.Lock()
		r := h.terms[id]
		h.mu.Unlock()
		if r == nil {
			return nil, fmt.Errorf("unknown terminal_id")
		}
		if r.cmd.Process != nil {
			_ = r.cmd.Process.Kill()
		}
		return h.termStatus(id, r), nil
	case "list":
		h.mu.Lock()
		defer h.mu.Unlock()
		out := []any{}
		for k, r := range h.terms {
			out = append(out, h.termStatus(k, r))
		}
		return map[string]any{"terminals": out}, nil
	default:
		return nil, fmt.Errorf("unknown sh action %q", action)
	}
}

func (h *ToolHost) waitTerm(r *terminalRun) {
	err := r.cmd.Wait()
	code := 0
	if r.cmd.ProcessState != nil {
		code = r.cmd.ProcessState.ExitCode()
	}
	r.exit = &code
	r.err = err
	r.ended = time.Now()
}
func (h *ToolHost) termStatus(id string, r *terminalRun) map[string]any {
	status := "running"
	var code any
	if r.exit != nil {
		status = "exited"
		code = *r.exit
	}
	return map[string]any{"terminal_id": id, "status": status, "exit_code": code, "started_at": r.started.Format(time.RFC3339), "ended_at": r.ended.Format(time.RFC3339)}
}
func (h *ToolHost) termOutput(id string, r *terminalRun, args map[string]any) map[string]any {
	s := r.out.String()
	off := intArg(args, "offset", 0)
	if off > len(s) {
		off = len(s)
	}
	limit := intArg(args, "limit", 65536)
	end := off + limit
	if end > len(s) {
		end = len(s)
	}
	out := h.termStatus(id, r)
	out["output"] = s[off:end]
	out["size"] = len(s)
	out["offset"] = off
	out["next_offset"] = end
	return out
}

func (h *ToolHost) webFetch(ctx context.Context, args map[string]any) (any, error) {
	url := stringArg(args, "url")
	if url == "" {
		return nil, fmt.Errorf("url is required")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "yassai/0.1")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	limit := int64(intArg(args, "limit", 200000))
	b, _ := io.ReadAll(io.LimitReader(res.Body, limit))
	return map[string]any{"url": url, "status": res.StatusCode, "content_type": res.Header.Get("Content-Type"), "text": string(b)}, nil
}

func (h *ToolHost) webSearch(ctx context.Context, args map[string]any) (any, error) {
	q := stringArg(args, "query")
	if q == "" {
		return nil, fmt.Errorf("query is required")
	}
	return map[string]any{"query": q, "results": []any{}, "note": "web.search is exposed through the MicroPython action space; configure a search backend in ToolHost.Dispatch for live search."}, nil
}

func (h *ToolHost) browserDrive(ctx context.Context, args map[string]any) (any, error) {
	action := stringArg(args, "action")
	if action == "navigate" || action == "new_page" {
		fetched, err := h.webFetch(ctx, args)
		if err != nil {
			return nil, err
		}
		h.mu.Lock()
		h.next++
		id := fmt.Sprintf("page_%d", h.next)
		h.pages[id] = stringArg(args, "url")
		h.mu.Unlock()
		return map[string]any{"page": id, "fetch": fetched}, nil
	}
	if action == "pages" {
		h.mu.Lock()
		defer h.mu.Unlock()
		return map[string]any{"pages": h.pages}, nil
	}
	return nil, fmt.Errorf("browser action %q is not implemented in the lightweight server build", action)
}

const prelude = `import json as _j
import binascii as _b
import host as _host

def _ld(s):
    return _j.loads(_b.a2b_base64(s))

def _call(name, args):
    resp = _j.loads(_host.call(name, _j.dumps(args if args is not None else {})))
    if isinstance(resp, dict) and resp.get("error"):
        raise RuntimeError(resp.get("error"))
    return resp

def _tool_args(name, args, kwargs):
    if len(args) > 1:
        raise TypeError("%s() takes at most one positional dict argument" % name)
    if len(args) == 1 and kwargs:
        raise TypeError("%s() takes either a dict or keyword arguments, not both" % name)
    if len(args) == 1:
        if args[0] is None:
            return {}
        if not isinstance(args[0], dict):
            raise TypeError("%s() positional argument must be a dict" % name)
        return args[0]
    return kwargs

def _mk_tool(n):
    def _tool(*args, **kwargs):
        payload = _tool_args(n, args, kwargs)
        if n == "submit":
            payload = dict(payload)
            payload["tasks"] = input.get("tasks", []) if isinstance(input, dict) else []
        return _call(n, payload)
    return _tool

class _Namespace:
    pass

def _install_surfaces(bindings):
    g = globals()
    for b in bindings:
        ns = b.get("namespace") or []
        member = b.get("member")
        tool = b.get("tool")
        if not member or not tool:
            continue
        if not ns:
            g[member] = _mk_tool(tool)
            continue
        root = g.get(ns[0])
        if root is None:
            root = _Namespace(); g[ns[0]] = root
        target = root
        for part in ns[1:]:
            child = getattr(target, part, None)
            if child is None:
                child = _Namespace(); setattr(target, part, child)
            target = child
        setattr(target, member, _mk_tool(tool))
`

const epilogue = `_host.call("__result__", _j.dumps({"json": (_j.dumps(result) if result is not None else ""), "vars": vars, "error": _err}))
`

func lastLine(s string) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	if len(lines) == 0 {
		return s
	}
	return lines[len(lines)-1]
}
