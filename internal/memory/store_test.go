package memory_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ashaibani/yassai/internal/memory"
)

func TestEnsureDefaultsCreatesMinimalIndex(t *testing.T) {
	root := t.TempDir()
	s := memory.New(root)
	b := s.LoadForTasks(nil)
	if !strings.Contains(b.Index, "MEMORY.md") {
		t.Fatalf("expected a MEMORY.md index, got %q", b.Index)
	}
	if _, err := os.Stat(filepath.Join(root, "MEMORY.md")); err != nil {
		t.Fatalf("MEMORY.md not created: %v", err)
	}
}

// The default seed must NOT inject any memory docs — that was pure per-call
// token overhead and off-topic for task-solving. Docs only appear if the agent
// creates them at runtime.
func TestNoSeededDocsByDefault(t *testing.T) {
	root := t.TempDir()
	s := memory.New(root)
	b := s.LoadForTasks(nil)
	if len(b.Docs) != 0 {
		t.Fatalf("expected no seeded memory docs by default, got %d", len(b.Docs))
	}
}
