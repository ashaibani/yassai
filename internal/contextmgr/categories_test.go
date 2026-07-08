package contextmgr_test

import (
	"strings"
	"testing"

	"github.com/ashaibani/yassai/internal/agenttypes"
	"github.com/ashaibani/yassai/internal/contextmgr"
	"github.com/ashaibani/yassai/internal/memory"
	"github.com/ashaibani/yassai/internal/skills"
)

// Verifies the classifier's labels reach the batch prompt (labels-only design).
func TestBuildBatchPromptIncludesCategories(t *testing.T) {
	tasks := []agenttypes.Task{{TaskID: "m1", Prompt: "What is 17 * 23?"}}
	cats := map[string][]string{"m1": {"mathematical_reasoning"}}
	mgr := contextmgr.Manager{MaxContextTokens: 200000, ReserveTokens: 24000}

	p := mgr.BuildBatchPrompt(tasks, memory.Bundle{}, skills.Bundle{}, cats)
	if !strings.Contains(p, `"categories"`) || !strings.Contains(p, "mathematical_reasoning") {
		t.Fatalf("prompt missing categories; got: %s", p)
	}

	// And with no categories (nil map) the field is omitted, not empty noise.
	p2 := mgr.BuildBatchPrompt(tasks, memory.Bundle{}, skills.Bundle{}, nil)
	if strings.Contains(p2, `"categories"`) {
		t.Fatalf("expected no categories field when map is nil; got: %s", p2)
	}
}
