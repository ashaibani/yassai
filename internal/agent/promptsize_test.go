package agent

import (
	"testing"

	"github.com/ashaibani/yassai/internal/contextmgr"
	"github.com/ashaibani/yassai/internal/memory"
	"github.com/ashaibani/yassai/internal/skills"
)

// TestPromptSizeBreakdown prints the exact per-call prompt composition so we can
// see the true fixed overhead (system prompt + batch scaffolding) versus task
// content. Run: go test -run TestPromptSizeBreakdown -v ./internal/agent/
func TestPromptSizeBreakdown(t *testing.T) {
	sys := systemPrompt([]Task{{TaskID: "t1", Prompt: "x"}}, map[string][]string{"t1": {"factual_knowledge"}})
	mgr := contextmgr.Manager{MaxContextTokens: 200000, ReserveTokens: 24000}
	one := []Task{{TaskID: "f1", Prompt: "What is the capital of France? Return only the name."}}
	bp1 := mgr.BuildBatchPrompt(one, memory.Bundle{}, skills.Bundle{}, nil)
	t.Logf("systemPrompt              = %5d chars (~%d tok)", len(sys), len(sys)/4)
	t.Logf("batchPrompt(1 tiny task)  = %5d chars (~%d tok)", len(bp1), len(bp1)/4)
	t.Logf("fixed overhead (sys+scaffold, empty mem/skills) ~= %d tok", (len(sys)+len(bp1))/4)
	t.Logf("---- batchPrompt(1 tiny task) ----\n%s\n----", bp1)
}
