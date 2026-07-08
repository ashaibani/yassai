package contextmgr_test

import (
	"fmt"
	"testing"

	"github.com/ashaibani/yassai/internal/agenttypes"
	"github.com/ashaibani/yassai/internal/contextmgr"
	"github.com/ashaibani/yassai/internal/memory"
	"github.com/ashaibani/yassai/internal/skills"
)

func TestPromptSizeNoClassifier(t *testing.T) {
	root := t.TempDir() // temp root: don't litter the source tree with a regenerated MEMORY.md
	tasks := []agenttypes.Task{
		{TaskID: "f1", Prompt: "What is the capital of France? Return only the name."},
		{TaskID: "m1", Prompt: "What is 17 * 23? Return only the number."},
		{TaskID: "s1", Prompt: "Classify the sentiment of \"I loved this excellent product\". Return only the label."},
		{TaskID: "su1", Prompt: "Summarise the following text in one sentence: The new software update includes several security patches, a redesigned user interface, and performance improvements that reduce load times by up to 30 percent."},
		{TaskID: "n1", Prompt: "Extract named entities from this text: \"Tim Cook announced that Apple Inc. will open a new office in London by January 2025.\" Return as JSON with types."},
		{TaskID: "cd1", Prompt: "Debug this Python function: def add(a, b): return a - b. Return the corrected code only."},
		{TaskID: "l1", Prompt: "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies? Return only Yes or No."},
		{TaskID: "cg1", Prompt: "Write a Python function called is_palindrome that takes a string and returns True if it reads the same forwards and backwards, ignoring case and spaces. Return only the code."},
	}
	mem := memory.New(root)
	memBundle := mem.LoadForTasks(tasks)
	sk := skills.NewLoader(nil)
	skillBundle := sk.LoadForPrompt(".", "", 8000)
	mgr := contextmgr.Manager{MaxContextTokens: 200000, ReserveTokens: 24000}

	// True classifier cost: the same batch prompt with vs without a category
	// label on each task.
	cats := map[string][]string{}
	for _, t2 := range tasks {
		cats[t2.TaskID] = []string{"mathematical_reasoning"}
	}
	promptWith := mgr.BuildBatchPrompt(tasks, memBundle, skillBundle, cats)
	promptWithout := mgr.BuildBatchPrompt(tasks, memBundle, skillBundle, nil)

	fmt.Printf("\n=== CLASSIFIER COST ANALYSIS ===\n")
	fmt.Printf("WITH classifier:    %d chars (~%d tokens)\n", len(promptWith), len(promptWith)/4)
	fmt.Printf("WITHOUT classifier: %d chars (~%d tokens)\n", len(promptWithout), len(promptWithout)/4)
	diff := len(promptWith) - len(promptWithout)
	fmt.Printf("Difference:         %d chars (~%d tokens)\n", diff, diff/4)
	if diff > 0 {
		pct := float64(diff) / float64(len(promptWithout)) * 100
		fmt.Printf("Overhead:           %.1f%% of prompt\n", pct)
	}
	fmt.Printf("\n")

	// The classifier adds a "category" field to each task object.
	// Each category string is ~10-20 chars.
	// For 8 tasks: ~8 * 22 chars (including JSON key overhead) = ~176 chars = ~44 tokens
	t.Logf("done")
}
