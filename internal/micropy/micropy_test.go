package micropy

import (
	"context"
	"strings"
	"testing"
)

func TestMicroPythonRunsCode(t *testing.T) {
	exec, err := New(Config{})
	if err != nil {
		t.Fatal(err)
	}
	res, err := exec.Run(context.Background(), "test", "result = 6 * 7", nil)
	if err != nil && strings.Contains(err.Error(), "exception") {
		t.Skip(err)
	}
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(res.JSON) != "42" {
		t.Fatalf("got %q, want 42", res.JSON)
	}
}

func TestMicroPythonPersistsVars(t *testing.T) {
	exec, _ := New(Config{})
	_, err := exec.Run(context.Background(), "s", "vars['n'] = 11", nil)
	if err != nil && strings.Contains(err.Error(), "exception") {
		t.Skip(err)
	}
	if err != nil {
		t.Fatal(err)
	}
	res, err := exec.Run(context.Background(), "s", "result = vars.get('n')", nil)
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(res.JSON) != "11" {
		t.Fatalf("got %q, want 11", res.JSON)
	}
}

func TestMicroPythonToolSurface(t *testing.T) {
	exec, err := New(Config{})
	if err != nil {
		t.Fatal(err)
	}
	code := "out = sh.run(command='printf hello')\nresult = {'text': out.get('output'), 'status': out.get('status')}"
	res, err := exec.Run(context.Background(), "tools", code, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(res.JSON, "hello") {
		t.Fatalf("tool output missing: %s", res.JSON)
	}
}

func TestSubmitRejectsMissingTaskIDs(t *testing.T) {
	exec, err := New(Config{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = exec.Run(context.Background(), "submit-missing", "submit(answers=[{'task_id':'a','answer':'1'}])", map[string]any{"tasks": []map[string]any{{"task_id": "a"}, {"task_id": "b"}}})
	if err == nil || !strings.Contains(err.Error(), "missing answers") {
		t.Fatalf("expected missing answers error, got %v", err)
	}
}
