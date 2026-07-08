package contextmgr_test

import (
	"fmt"
	"testing"

	"github.com/ashaibani/yassai/internal/agenttypes"
	"github.com/ashaibani/yassai/internal/contextmgr"
	"github.com/ashaibani/yassai/internal/memory"
	"github.com/ashaibani/yassai/internal/skills"
)

func TestPromptSize(t *testing.T) {
	tasks := []agenttypes.Task{
		{TaskID: "s1", Prompt: "Classify the sentiment of I loved this excellent product. Return only the label."},
		{TaskID: "m1", Prompt: "What is 17 * 23? Return only the number."},
	}
	mem := memory.New(t.TempDir()) // temp root: don't litter the source tree with a regenerated MEMORY.md
	memBundle := mem.LoadForTasks(tasks)
	t.Logf("memory index size: %d, docs: %d", len(memBundle.Index), len(memBundle.Docs))
	for _, d := range memBundle.Docs {
		t.Logf("  doc: %s size: %d", d.Path, len(d.Content))
	}
	sk := skills.NewLoader(nil)
	skillBundle := sk.LoadForPrompt(".", tasks[0].Prompt+" "+tasks[1].Prompt, 8000)
	t.Logf("skills index: %d, loaded: %d", len(skillBundle.Index), len(skillBundle.Loaded))
	for _, s := range skillBundle.Index {
		fmt.Printf("  idx: %s (%d desc)\n", s.Name, len(s.Description))
	}
	for _, s := range skillBundle.Loaded {
		t.Logf("  loaded: %s content_len: %d", s.Name, len(s.Content))
	}
	mgr := contextmgr.Manager{MaxContextTokens: 200000, ReserveTokens: 24000}
	prompt := mgr.BuildBatchPrompt(tasks, memBundle, skillBundle, nil)
	t.Logf("prompt total size: %d chars (~%d tokens)", len(prompt), len(prompt)/4)
	if len(prompt) > 10000 {
		t.Errorf("prompt too large: %d chars", len(prompt))
	}
}
