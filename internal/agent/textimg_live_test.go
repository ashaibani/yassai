package agent

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/ashaibani/yassai/internal/llm"
	"github.com/ashaibani/yassai/internal/validate"
)

// goldenDirectBatch loads the non-code tasks (the "direct" batch) from the
// real-benchmark golden file with their true categories.
func goldenDirectBatch(t *testing.T) ([]Task, map[string][]string) {
	t.Helper()
	cases, err := validate.LoadCases("../../testdata/downloads_tasks_golden.json")
	if err != nil {
		t.Skipf("golden tasks unavailable: %v", err)
	}
	catOf := func(id string) string {
		if len(id) < 3 || id[0] != 'T' {
			return ""
		}
		switch id[:3] {
		case "T01":
			return "factual_knowledge"
		case "T02", "T08", "T10":
			return "mathematical_reasoning"
		case "T03":
			return "sentiment_classification"
		case "T04":
			return "text_summarisation"
		case "T05":
			return "named_entity_recognition"
		case "T06":
			return "code_debugging"
		case "T07":
			return "logical_deductive_reasoning"
		case "T09":
			return "code_generation"
		}
		return ""
	}
	var batch []Task
	cats := map[string][]string{}
	for _, c := range cases {
		cat := catOf(c.TaskID)
		if usesCodeExec(cat) {
			continue
		}
		batch = append(batch, Task{TaskID: c.TaskID, Prompt: c.Prompt})
		cats[c.TaskID] = []string{cat}
	}
	return batch, cats
}

// TestTextImgDirectBatchLive sends the real direct batch as a textimg render
// and prints the raw model output - run when the image path misbehaves to see
// what the model actually returns. FIREWORKS_API_KEY required.
func TestTextImgDirectBatchLive(t *testing.T) {
	if os.Getenv("TEXTIMG_LIVE_TESTS") != "1" {
		t.Skip("set TEXTIMG_LIVE_TESTS=1 to run Fireworks network experiments")
	}
	apiKey := os.Getenv("FIREWORKS_API_KEY")
	if apiKey == "" {
		t.Skip("FIREWORKS_API_KEY not set")
	}
	batch, cats := goldenDirectBatch(t)
	if len(batch) == 0 {
		t.Skip("no direct-batch tasks in golden file")
	}
	mode := os.Getenv("TEXTIMG_LIVE_MODE")
	if mode == "" {
		mode = "tasks"
	}
	ag := &Agent{cfg: Config{TextImg: mode}, categories: cats}
	messages := ag.buildBatchMessages(batch, false)
	if len(messages) != 2 || len(messages[1].ImageURLs) == 0 {
		t.Fatalf("expected system+user-with-images, got %d messages (images=%d)",
			len(messages), len(messages[len(messages)-1].ImageURLs))
	}
	t.Logf("system prompt (%d chars):\n%s", len(messages[0].Content), messages[0].Content)
	t.Logf("user text: %s", messages[1].Content)
	t.Logf("images: %d (first %d b64 bytes)", len(messages[1].ImageURLs), len(messages[1].ImageURLs[0]))

	client := llm.New(llm.Config{
		APIKey:  apiKey,
		BaseURL: "https://api.fireworks.ai/inference/v1",
		Model:   "accounts/fireworks/models/minimax-m3",
		Timeout: 120 * time.Second,
	})
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Second)
	defer cancel()
	res, err := client.ChatWithOptions(ctx, messages, llm.ChatOptions{MaxTokens: 4000, ReasoningEffort: "none"})
	if err != nil {
		t.Fatalf("chat: %v", err)
	}
	t.Logf("usage: prompt=%d completion=%d reasoning=%d", res.Usage.PromptTokens, res.Usage.CompletionTokens, res.Usage.ReasoningTokens)
	t.Logf("finish=%s content (%d chars):\n%s", res.FinishReason, len(res.Content), res.Content)
	if strings.TrimSpace(res.ReasoningContent) != "" {
		t.Logf("reasoning_content (%d chars):\n%.2000s", len(res.ReasoningContent), res.ReasoningContent)
	}
	answers, ok := parseAnswers(res.Content, batch)
	t.Logf("parseAnswers ok=%v got=%d/%d ids", ok, len(answers), len(batch))
	for id, a := range answers {
		t.Logf("  %s: %.80s", id, a)
	}
}
