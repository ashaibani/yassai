package localllm

import (
	"context"
	"os"
	"testing"
	"time"
)

// TestSolveTaskLive exercises the full local pipeline against a real GGUF.
// Env-gated: set LOCALLLM_LIVE=1, LOCAL_MODEL_PATH and YZMA_LIB.
func TestSolveTaskLive(t *testing.T) {
	if os.Getenv("LOCALLLM_LIVE") != "1" {
		t.Skip("set LOCALLLM_LIVE=1 to run")
	}
	s, err := New(Config{
		ModelPath: os.Getenv("LOCAL_MODEL_PATH"),
		LibPath:   os.Getenv("YZMA_LIB"),
		Timeout:   120 * time.Second,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()
	prompts := []string{
		"A depot starts with 3,000 units. In Q1 it sells 40% of stock. In Q2 it restocks 700 units. In Q3 it sells 900 units. How many units remain at the end of Q3?",
		"Four flatmates — Nora, Otto, Pia, and Quill — each play a different instrument: guitar, piano, violin, or drums. Nora plays the piano. Otto plays neither the guitar nor the drums. Pia does not play the drums. State each person's instrument.",
	}
	for _, p := range prompts {
		res := s.SolveTask(context.Background(), p)
		t.Logf("OK=%v reason=%q\ncode:\n%s\nstdout: %s\nanswer: %s", res.OK, res.Reason, res.Code, res.Stdout, res.Answer)
	}
}
