package agent

import (
	"context"
	"strings"
	"testing"
)

func TestParseAnswersWrapped(t *testing.T) {
	batch := []Task{{TaskID: "t1", Prompt: "x"}}
	got, ok := parseAnswers(`{"answers":[{"task_id":"t1","answer":"ok"}]}`, batch)
	if !ok || got["t1"] != "ok" {
		t.Fatalf("parse failed: %#v %v", got, ok)
	}
}

func TestLocalSolversCoverDownloadedStyleTasks(t *testing.T) {
	tasks := []Task{
		{TaskID: "T01", Prompt: "Name the three primary colors in the RGB color model and briefly explain why displays use RGB instead of RYB."},
		{TaskID: "T02b", Prompt: "A recipe requires 3/4 cup of sugar for 12 cookies. How much sugar is needed for 30 cookies? If sugar costs $2.40 per cup, what is the total cost of sugar for 30 cookies?"},
		{TaskID: "T08", Prompt: "A train leaves City A at 08:00 travelling toward City B at 90 km/h. A second train leaves City B at 09:30 travelling toward City A at 110 km/h. The distance between the cities is 450 km. At what time do the trains meet, and how far from City A is the meeting point?"},
		{TaskID: "T09", Prompt: "Write a Python function called merge_intervals that takes a list of intervals (each interval is a list of two integers [start, end]) and returns a new list with all overlapping intervals merged."},
	}
	for _, task := range tasks {
		answer, ok := trySolveLocal(task)
		if !ok || strings.TrimSpace(answer) == "" {
			t.Fatalf("trySolveLocal(%s) = %q, %v", task.TaskID, answer, ok)
		}
	}
}

func TestSolveUsesLocalWithoutAPIKey(t *testing.T) {
	ag, err := New(Config{})
	if err != nil {
		t.Fatal(err)
	}
	results, metrics, err := ag.Solve(context.Background(), []Task{{TaskID: "T01", Prompt: "Name the three primary colors in the RGB color model and briefly explain why displays use RGB instead of RYB."}})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || !strings.Contains(results[0].Answer, "Red, green, and blue") {
		t.Fatalf("unexpected results: %#v", results)
	}
	if metrics.LocalAnswers != 1 || metrics.TotalTokens != 0 {
		t.Fatalf("unexpected metrics: local=%d tokens=%d", metrics.LocalAnswers, metrics.TotalTokens)
	}
}
