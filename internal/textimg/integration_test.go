package textimg

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestSystemPromptCompression(t *testing.T) {
	apiKey := requireLiveAPI(t)

	sysParts := []string{
		"You are a compact, high-accuracy AI agent.",
		"Return exactly this JSON object: answers array with task_id and answer fields.",
		"Preserve each task_id. Keep answers concise but complete.",
		"There are no native tool calls. Use MicroPython code blocks for computation.",
	}
	systemPrompt := strings.Join(sysParts, " ")

	taskContent := strings.Join([]string{
		"Task f1: What is the capital of France? Return only the name.",
		"Task m1: What is 17 * 23? Return only the number.",
		"Task s1: Classify sentiment of I loved this product. Return only the label.",
		"Task f5: What is the boiling point of water at sea level in Celsius? Return only the number.",
	}, "\n")

	textBody := map[string]any{
		"model": "accounts/fireworks/models/minimax-m3",
		"messages": []map[string]any{
			{"role": "system", "content": systemPrompt},
			{"role": "user", "content": taskContent + "\n\nReturn JSON answers for all tasks."},
		},
		"max_tokens": 500,
	}
	textResult, textPT, textCT := callAndParse2(t, apiKey, textBody)

	cfg := DefaultRenderConfig()
	cfg.Scale = 1
	images, err := RenderText(systemPrompt, cfg)
	if err != nil {
		t.Fatalf("RenderText: %v", err)
	}

	imgContent := []map[string]any{
		{"type": "text", "text": "The image contains your system instructions. Follow them."},
		{"type": "image_url", "image_url": map[string]string{"url": images[0].ToBase64DataURI()}},
		{"type": "text", "text": taskContent + "\n\nReturn JSON answers for all tasks."},
	}
	imgBody := map[string]any{
		"model":      "accounts/fireworks/models/minimax-m3",
		"messages":   []map[string]any{{"role": "user", "content": imgContent}},
		"max_tokens": 500,
	}
	imgResult, imgPT, imgCT := callAndParse2(t, apiKey, imgBody)

	expected := map[string]string{
		"f1": "Paris", "m1": "391", "s1": "positive", "f5": "100",
	}
	textCorrect := countCorrect2(textResult, expected)
	imgCorrect := countCorrect2(imgResult, expected)

	t.Logf("Text system prompt: prompt=%d completion=%d accuracy=%d/4", textPT, textCT, textCorrect)
	t.Logf("Img system prompt:  prompt=%d completion=%d accuracy=%d/4", imgPT, imgCT, imgCorrect)
	t.Logf("Savings: %d prompt tokens (%.0f%%)", textPT-imgPT, float64(textPT-imgPT)/float64(textPT)*100)
	t.Logf("  Text response: %s", textResult)
	t.Logf("  Img response:  %s", imgResult)
}

func TestLargePromptCompression(t *testing.T) {
	apiKey := requireLiveAPI(t)

	var sb strings.Builder
	for i := 0; i < 50; i++ {
		fmt.Fprintf(&sb, "Instruction: When processing task type %c, check the memory file for relevant context. ", 'A'+i%26)
		fmt.Fprintf(&sb, "Skill ai-sdk: Answer questions about AI SDK functions like generateText, streamText, ToolLoopAgent, embed, or tools. ")
		fmt.Fprintf(&sb, "Memory: Previously solved arithmetic task m1 with answer 391. Confidence 0.95. ")
		fmt.Fprintf(&sb, "Skill building-ai-agent: Build AI agents with state management, real-time WebSockets, scheduled tasks.\n")
	}
	largeContent := sb.String()

	taskContent := "Task m1: What is 17 * 23? Return only the number.\nTask f1: What is the capital of France?"

	textBody := map[string]any{
		"model":      "accounts/fireworks/models/minimax-m3",
		"messages":   []map[string]any{{"role": "user", "content": largeContent + "\n\n" + taskContent + "\n\nReturn JSON answers for all tasks."}},
		"max_tokens": 500,
	}
	textResult, textPT, textCT := callAndParse2(t, apiKey, textBody)

	cfg := DefaultRenderConfig()
	cfg.Scale = 1
	images, err := RenderText(largeContent, cfg)
	if err != nil {
		t.Fatalf("RenderText: %v", err)
	}

	totalH := 0
	for _, img := range images {
		totalH += img.H
	}

	imgParts := []map[string]any{
		{"type": "text", "text": "Context in images below. Then answer the tasks."},
	}
	for _, img := range images {
		imgParts = append(imgParts, map[string]any{
			"type":      "image_url",
			"image_url": map[string]string{"url": img.ToBase64DataURI()},
		})
	}
	imgParts = append(imgParts, map[string]any{
		"type": "text",
		"text": taskContent + "\n\nReturn JSON answers for all tasks.",
	})

	imgBody := map[string]any{
		"model":      "accounts/fireworks/models/minimax-m3",
		"messages":   []map[string]any{{"role": "user", "content": imgParts}},
		"max_tokens": 500,
	}
	imgResult, imgPT, imgCT := callAndParse2(t, apiKey, imgBody)

	expected := map[string]string{"m1": "391", "f1": "Paris"}
	textCorrect := countCorrect2(textResult, expected)
	imgCorrect := countCorrect2(imgResult, expected)

	t.Logf("Large prompt (%d chars):", len(largeContent))
	t.Logf("  Text: prompt=%d completion=%d accuracy=%d/2", textPT, textCT, textCorrect)
	t.Logf("  Img:  prompt=%d completion=%d accuracy=%d/2 (%dx%d, %d pages)", imgPT, imgCT, imgCorrect, images[0].W, totalH, len(images))
	t.Logf("  Savings: %d prompt tokens (%.0f%%)", textPT-imgPT, float64(textPT-imgPT)/float64(textPT)*100)
	t.Logf("  Text response: %s", textResult)
	t.Logf("  Img response:  %s", imgResult)
}

func callAndParse2(t *testing.T, apiKey string, body map[string]any) (string, int, int) {
	result := callLiveAPI(t, apiKey, body, 60*time.Second)
	return result.Content, result.PromptTokens, result.CompletionTokens
}

func countCorrect2(response string, expected map[string]string) int {
	correct := 0
	for _, exp := range expected {
		if strings.Contains(strings.ToLower(response), strings.ToLower(exp)) {
			correct++
		}
	}
	return correct
}
