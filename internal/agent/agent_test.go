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

func TestPlanBatchesIsolatesCodeDebugging(t *testing.T) {
	tasks := []Task{
		{TaskID: "F", Prompt: "What is RAM?"},
		{TaskID: "D", Prompt: "Find the bug in this function."},
		{TaskID: "S", Prompt: "Summarize this."},
	}
	cats := map[string][]string{
		"F": {"factual_knowledge"},
		"D": {"code_debugging"},
		"S": {"text_summarisation"},
	}
	ag, err := New(Config{MaxBatchSize: 40, ReasoningEffort: "none", Categories: cats})
	if err != nil {
		t.Fatal(err)
	}
	ag.categories = cats
	batches := ag.planBatches(tasks)
	if len(batches) != 2 {
		t.Fatalf("want 2 batches (general + code-debug), got %d %#v", len(batches), batches)
	}
	if len(batches[1]) != 1 || batches[1][0].TaskID != "D" {
		t.Fatalf("code-debug task should be isolated in its own batch, got %#v", batches)
	}
}

func TestBatchIsolationModes(t *testing.T) {
	for _, tc := range []struct {
		mode, cat, want string
	}{
		{"focus", "code_debugging", "code_debugging"},
		{"focus", "mathematical_reasoning", ""},
		{"math", "mathematical_reasoning", "mathematical_reasoning"},
		{"math", "named_entity_recognition", ""},
		{"none", "code_debugging", ""},
	} {
		if got := batchIsolationFocus(tc.mode, tc.cat); got != tc.want {
			t.Errorf("mode=%q cat=%q: got %q want %q", tc.mode, tc.cat, got, tc.want)
		}
	}
}

func TestLeanEffortKeepsMathOnLow(t *testing.T) {
	tiers := LeanEffortTiers()
	if tiers["mathematical_reasoning"] != "low" {
		t.Fatalf("math effort should stay low for reliable tool code, got %q", tiers["mathematical_reasoning"])
	}
	if tiers["factual_knowledge"] != "none" {
		t.Fatalf("factual effort should stay token-lean, got %q", tiers["factual_knowledge"])
	}
}

func TestMathRecipeRequiresSubsetAndTimeChecks(t *testing.T) {
	recipe := categoryRecipe["mathematical_reasoning"]
	for _, want := range []string{"assert len(subset)==K", "preserve exact times incl seconds", "NEVER store rounded rates", "include raw average used"} {
		if !strings.Contains(recipe, want) {
			t.Fatalf("math recipe missing %q: %s", want, recipe)
		}
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
