package judge

import (
	"context"
	"os/exec"
	"testing"
	"time"
)

func TestParseVerdict(t *testing.T) {
	cases := []struct {
		name string
		text string
		want bool
		ok   bool // whether no error is expected
	}{
		{"json_true", "Matches the reference. -> CORRECT\nJSON: {\"correct\": true}", true, true},
		{"json_false", "Wrong total. -> INCORRECT\nJSON: {\"correct\": false}", false, true},
		{"marker_true_no_json", "Looks right -> CORRECT", true, true},
		{"marker_false_no_json", "Off by one -> INCORRECT", false, true},
		{"json_wins_over_marker", "sentence -> CORRECT\nJSON: {\"correct\": false}", false, true},
		{"empty", "", false, false},
	}
	for _, c := range cases {
		got, _, err := parseVerdict(c.text)
		if (err == nil) != c.ok {
			t.Errorf("%s: err=%v, want no-error=%v", c.name, err, c.ok)
		}
		if err == nil && got != c.want {
			t.Errorf("%s: got %v, want %v", c.name, got, c.want)
		}
	}
}

func TestRunTests(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}
	ctx := context.Background()
	good := "```python\ndef add(a, b):\n    return a + b\n```"
	if ok, err := RunTests(ctx, good, "add",
		[]Test{{Args: []any{1, 2}, Expected: 3}, {Args: []any{5, 5}, Expected: 10}}, 10*time.Second); err != nil || !ok {
		t.Fatalf("correct impl: ok=%v err=%v (want pass)", ok, err)
	}
	bad := "def add(a, b):\n    return a - b\n" // buggy, no fence
	if ok, _ := RunTests(ctx, bad, "add",
		[]Test{{Args: []any{1, 2}, Expected: 3}}, 10*time.Second); ok {
		t.Fatal("buggy impl: want fail")
	}
}
