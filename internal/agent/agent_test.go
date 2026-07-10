package agent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/ashaibani/yassai/internal/llm"
)

func TestParseAnswersWrapped(t *testing.T) {
	batch := []Task{{TaskID: "t1", Prompt: "x"}}
	got, ok := parseAnswers(`{"answers":[{"task_id":"t1","answer":"ok"}]}`, batch)
	if !ok || got["t1"] != "ok" {
		t.Fatalf("parse failed: %#v %v", got, ok)
	}
}

func TestTrySolveLocalDoesNotAnswerBenchmarkPrompts(t *testing.T) {
	t0 := Task{TaskID: "T02", Prompt: "A warehouse starts with 2,400 units. In Q1 it sells 37% of stock."}
	if ans, ok := trySolveLocal(context.Background(), t0, "mathematical_reasoning"); ok || ans != "" {
		t.Fatalf("expected no local benchmark answer, got ok=%v ans=%q", ok, ans)
	}
}

func TestSolveWithEventsAPI(t *testing.T) {
	var _ Event
	var _ EventCallback
}

func TestPlanBatchesMathBatchSize(t *testing.T) {
	// Two non-math + four math-like prompts; force categories to avoid heuristic drift.
	tasks := []Task{
		{TaskID: "A", Prompt: "Name three primary colors."},
		{TaskID: "B", Prompt: "What is RAM vs ROM?"},
		{TaskID: "M1", Prompt: "Calculate remaining units after 37% sell."},
		{TaskID: "M2", Prompt: "How much sugar is needed for 30 cookies?"},
		{TaskID: "M3", Prompt: "Two trains at 90 km/h meet when?"},
		{TaskID: "M4", Prompt: "Project July revenue with growth rates."},
	}
	ag, err := New(Config{
		MaxBatchSize:    40,
		MathBatchSize:   1,
		ReasoningEffort: "none",
		Categories: map[string][]string{
			"A":  {"factual_knowledge"},
			"B":  {"factual_knowledge"},
			"M1": {"mathematical_reasoning"},
			"M2": {"mathematical_reasoning"},
			"M3": {"mathematical_reasoning"},
			"M4": {"mathematical_reasoning"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	ag.categories = ag.cfg.Categories
	batches := ag.planBatches(tasks)
	if len(batches) != 5 { // 1 non-math group + 4 singleton math
		t.Fatalf("want 5 batches, got %d %#v", len(batches), batches)
	}
	if len(batches[0]) != 2 {
		t.Fatalf("non-math batch size: got %d", len(batches[0]))
	}
	for i := 1; i < len(batches); i++ {
		if len(batches[i]) != 1 {
			t.Fatalf("math batch %d size want 1 got %d", i, len(batches[i]))
		}
	}

	ag2, err := New(Config{MaxBatchSize: 40, MathBatchSize: 0, ReasoningEffort: "none", Categories: ag.cfg.Categories})
	if err != nil {
		t.Fatal(err)
	}
	ag2.categories = ag2.cfg.Categories
	batches2 := ag2.planBatches(tasks)
	if len(batches2) != 2 {
		t.Fatalf("inherit MaxBatchSize: want 2 batches, got %d %#v", len(batches2), batches2)
	}
}

func TestPlanBatchesIsolatesLogic(t *testing.T) {
	// Focus isolation (the default) keeps logic in its own code-exec batch,
	// separate from the direct categories - UNLESS the code group is tiny:
	// with the model lanes running before batching, a 1-2 task code group is
	// gate rejects, and folding them into the direct batch saves the ~1.1k
	// token per-batch scaffold. AGENT_FOLD_CODE_REMAINDER=0 restores strict
	// isolation.
	tasks := []Task{
		{TaskID: "F", Prompt: "What is RAM?"},
		{TaskID: "L", Prompt: "Who owns which pet given these clues?"},
		{TaskID: "S", Prompt: "Summarize this."},
	}
	cats := map[string][]string{
		"F": {"factual_knowledge"},
		"L": {"logical_deductive_reasoning"},
		"S": {"text_summarisation"},
	}
	ag, err := New(Config{MaxBatchSize: 40, BatchIsolation: "focus", ReasoningEffort: "none", Categories: cats})
	if err != nil {
		t.Fatal(err)
	}
	ag.categories = cats

	// Default fold: the lone logic task joins the direct batch.
	batches := ag.planBatches(tasks)
	if len(batches) != 1 || len(batches[0]) != 3 {
		t.Fatalf("small code group must fold into the direct batch, got %#v", batches)
	}

	// Fold disabled: strict isolation as before.
	t.Setenv("AGENT_FOLD_CODE_REMAINDER", "0")
	batches = ag.planBatches(tasks)
	if len(batches) != 2 {
		t.Fatalf("fold=0: want 2 batches (direct + logic), got %d %#v", len(batches), batches)
	}
	if len(batches[1]) != 1 || batches[1][0].TaskID != "L" {
		t.Fatalf("fold=0: logic task should be isolated, got %#v", batches)
	}

	// A code group above the fold threshold keeps its own batch.
	t.Setenv("AGENT_FOLD_CODE_REMAINDER", "2")
	tasks = append(tasks,
		Task{TaskID: "L2", Prompt: "Five friends sit in a row given these clues, who sits where?"},
		Task{TaskID: "L3", Prompt: "Who drinks tea given these clues?"},
	)
	cats["L2"] = []string{"logical_deductive_reasoning"}
	cats["L3"] = []string{"logical_deductive_reasoning"}
	batches = ag.planBatches(tasks)
	if len(batches) != 2 || len(batches[1]) != 3 {
		t.Fatalf("3-task code group must stay isolated, got %#v", batches)
	}
}

func TestBatchIsolationModes(t *testing.T) {
	for _, tc := range []struct {
		mode, cat, want string
	}{
		{"focus", "logical_deductive_reasoning", "logical_deductive_reasoning"},
		{"focus", "code_debugging", ""},
		{"focus", "mathematical_reasoning", ""},
		{"math", "mathematical_reasoning", "mathematical_reasoning"},
		{"math", "named_entity_recognition", ""},
		{"none", "logical_deductive_reasoning", ""},
	} {
		if got := batchIsolationFocus(tc.mode, tc.cat); got != tc.want {
			t.Errorf("mode=%q cat=%q: got %q want %q", tc.mode, tc.cat, got, tc.want)
		}
	}
}

func TestLeanEffortMathIsNone(t *testing.T) {
	// Maths and logic are solved via a run_python tool call, so the executed
	// code does the reasoning and the model runs at effort "none" (reasoning
	// tokens are the biggest cost lever). Every category is "none".
	tiers := LeanEffortTiers()
	for _, cat := range []string{"mathematical_reasoning", "logical_deductive_reasoning", "factual_knowledge"} {
		if tiers[cat] != "none" {
			t.Fatalf("%s effort should be none, got %q", cat, tiers[cat])
		}
	}
}

func TestMathRecipeRequiresSubsetAndTimeChecks(t *testing.T) {
	recipe := categoryRecipe["mathematical_reasoning"]
	for _, want := range []string{"STRAIGHT-LINE code", "preserve exact times incl seconds", "never feed a rounded value into later maths", "INTERPOLATING the computed variables"} {
		if !strings.Contains(recipe, want) {
			t.Fatalf("math recipe missing %q: %s", want, recipe)
		}
	}
}

func TestReasoningEffortGetsCompletionRoom(t *testing.T) {
	batch := []Task{{TaskID: "l1"}, {TaskID: "l2"}, {TaskID: "l3"}}
	if got := maxTokensForBatch(batch, 0, "xhigh", false); got != 6000 {
		t.Fatalf("xhigh batch budget: got %d want 6000", got)
	}
	if got := maxTokensForBatch(batch, 0, "none", false); got != 2048 {
		t.Fatalf("none batch budget: got %d want 2048", got)
	}
}

func TestNormaliseToolCallsWrapsMalformedArguments(t *testing.T) {
	calls := []llm.ToolCall{{
		ID:   "call_1",
		Type: "function",
		Function: llm.FunctionCall{
			Name:      "run_python",
			Arguments: `{"code":"print(1)`,
		},
	}}
	if got := toolCallCode(calls[0]); got != "" {
		t.Fatalf("truncated JSON must not be executed as code, got %q", got)
	}
	norm := normaliseToolCalls(calls)
	var payload map[string]string
	if err := json.Unmarshal([]byte(norm[0].Function.Arguments), &payload); err != nil {
		t.Fatalf("normalised arguments are not valid JSON: %v", err)
	}
	if payload["code"] != "" {
		t.Fatalf("truncated JSON should normalise to empty code, got %q", payload["code"])
	}
}

// TestHeuristicRoutingContract pins the keyword-heuristic mitigations for the
// classifier v4 defects seen in production telemetry and CI: code_debugging
// over-fires and can swallow NER/logic entirely (near-threshold int8 logits
// also flip between arm64 and amd64). The agent must route correctly from the
// prompt text alone, whatever the classifier says.
func TestHeuristicRoutingContract(t *testing.T) {
	cases := []struct{ prompt, want string }{
		{`Extract named entities from: "Tim Cook announced Apple Inc. will open a London office in 2025."`, "named_entity_recognition"},
		{"If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies?", "logical_deductive_reasoning"},
		{"Premise one: does it follow that some Lazzies are Bloops?", "logical_deductive_reasoning"},
	}
	for _, c := range cases {
		if got := heuristicCategory(c.prompt); got != c.want {
			t.Errorf("heuristicCategory(%.50q) = %q, want %q", c.prompt, got, c.want)
		}
	}

	// Even with the classifier reporting only noise labels, syllogism-shaped
	// logic must reach the code-exec batch.
	a := &Agent{categories: map[string][]string{"t1": {"code_debugging"}}}
	task := Task{TaskID: "t1", Prompt: cases[1].prompt}
	if !a.taskUsesCode(task) {
		t.Error("syllogism prompt with noisy classifier labels must route to code-exec")
	}
}
