package pyexec

import (
	"context"
	"os/exec"
	"testing"
	"time"
)

func skipIfNoPython(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}
}

// Answers must be captured from the pre-defined submit(), built from computed
// variables (the pattern the agent relies on).
func TestRunSubmit(t *testing.T) {
	skipIfNoPython(t)
	e, _ := New(Config{Timeout: 10 * time.Second})
	e.SetExpectedTasks([]string{"T1", "T2"})
	code := "x = 2400*(1-0.37) + 800 - 640\n" +
		"submit(answers=[{\"task_id\":\"T1\",\"answer\":str(round(x))+\" units\"},{\"task_id\":\"T2\",\"answer\":\"ok\"}])\n"
	_, err := e.Run(context.Background(), "s", code, nil)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if !e.IsSubmitted() {
		t.Fatal("expected submitted")
	}
	r := e.SubmitResult()
	if r["T1"] != "1672 units" {
		t.Errorf("T1=%q want %q", r["T1"], "1672 units")
	}
	if r["T2"] != "ok" {
		t.Errorf("T2=%q want ok", r["T2"])
	}
}

// The point of python3 over MicroPython: full stdlib imports work first-try
// (itertools was one of the feature gaps that forced multi-turn retries).
func TestImportsWork(t *testing.T) {
	skipIfNoPython(t)
	e, _ := New(Config{Timeout: 10 * time.Second})
	code := "import itertools\n" +
		"perms = list(itertools.permutations([1,2,3]))\n" +
		"submit(answers=[{\"task_id\":\"L\",\"answer\":str(len(perms))}])\n"
	if _, err := e.Run(context.Background(), "s", code, nil); err != nil {
		t.Fatalf("imports should work under python3: %v", err)
	}
	if got := e.SubmitResult()["L"]; got != "6" {
		t.Errorf("permutation count = %q, want 6", got)
	}
}

// A runtime error with no submit surfaces as an error (so the agent can recover).
func TestRunErrorSurfaces(t *testing.T) {
	skipIfNoPython(t)
	e, _ := New(Config{Timeout: 10 * time.Second})
	if _, err := e.Run(context.Background(), "s", "raise ValueError('boom')\n", nil); err == nil {
		t.Fatal("expected an error from failing code")
	}
	if e.IsSubmitted() {
		t.Fatal("should not be submitted after an error")
	}
}
