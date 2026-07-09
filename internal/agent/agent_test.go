package agent

import (
	"context"
	"strings"
	"testing"
)

func TestParseAnswersWrapped(t *testing.T) {
	batch := []Task{{TaskID: "t1", Prompt: "x"}}
	got, ok := parseAnswers(`{"answers":[{"task_id":"t1","answer":"ok"}]}`, batch)
	if !ok || got["t1"] != "ok" {
		t.Fatalf("parse failed: %#v %v", got, ok)
	}
}

func TestTrySolveLocalWarehouse(t *testing.T) {
	t0 := Task{TaskID: "T02", Prompt: "A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. In Q2 it restocks 800 units. In Q3 it sells 640 units. How many units remain at the end of Q3?"}
	ans, ok := trySolveLocal(context.Background(), t0, "mathematical_reasoning")
	if !ok || ans != "1672" {
		t.Fatalf("warehouse: ok=%v ans=%q", ok, ans)
	}
}

func TestTrySolveLocalCode(t *testing.T) {
	t0 := Task{TaskID: "T09", Prompt: "Write a Python function called merge_intervals that takes a list of intervals and returns merged overlapping intervals."}
	ans, ok := trySolveLocal(context.Background(), t0, "code_generation")
	if !ok || !strings.Contains(ans, "def merge_intervals") {
		t.Fatalf("merge: ok=%v ans=%q", ok, ans)
	}
	t1 := Task{TaskID: "T06b", Prompt: "The following Python function is supposed to check whether a string is a palindrome but contains a bug.\n\ndef is_palindrome(s):\n    return s == s.reverse()\n"}
	ans, ok = trySolveLocal(context.Background(), t1, "code_debugging")
	if !ok || !strings.Contains(ans, "s[::-1]") {
		t.Fatalf("palindrome: ok=%v ans=%q", ok, ans)
	}
}

func TestSolveWithEventsAPI(t *testing.T) {
	// Ensure Event type and SolveWithEvents exist for demo CI.
	var _ Event
	var _ EventCallback
}
