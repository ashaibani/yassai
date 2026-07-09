package agent

import (
	"context"
	"testing"
)

func TestParseAnswersWrapped(t *testing.T) {
	batch := []Task{{TaskID: "t1", Prompt: "x"}}
	got, ok := parseAnswers(`{"answers":[{"task_id":"t1","answer":"ok"}]}`, batch)
	if !ok || got["t1"] != "ok" {
		t.Fatalf("parse failed: %#v %v", got, ok)
	}
}

func TestTrySolveLocalIsNoop(t *testing.T) {
	t0 := Task{TaskID: "T02", Prompt: "A warehouse starts with 2,400 units."}
	if ans, ok := trySolveLocal(context.Background(), t0, "mathematical_reasoning"); ok || ans != "" {
		t.Fatalf("expected no-op local solver, got ok=%v ans=%q", ok, ans)
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
