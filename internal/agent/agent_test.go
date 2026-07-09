package agent

import "testing"

func TestParseAnswersWrapped(t *testing.T) {
	batch := []Task{{TaskID: "t1", Prompt: "x"}}
	got, ok := parseAnswers(`{"answers":[{"task_id":"t1","answer":"ok"}]}`, batch)
	if !ok || got["t1"] != "ok" {
		t.Fatalf("parse failed: %#v %v", got, ok)
	}
}
