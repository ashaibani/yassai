package textimg

import (
	"strings"
	"testing"
	"time"
)

func TestScaleComparison(t *testing.T) {
	apiKey := requireLiveAPI(t)

	systemPrompt := "You are a compact, high-accuracy AI agent for AMD Developer Hackathon Track 1. Return exactly this JSON object and nothing else: {\"answers\":[{\"task_id\":\"...\",\"answer\":\"...\"}]}. Preserve each task_id. Keep answers concise but complete."

	taskContent := "Tasks to solve:\n\nTask s1: What is the capital of France?\nTask s2: What is 15 * 17?\nTask s3: What is the chemical symbol for gold?\nTask s4: How many planets are in our solar system?\nTask s5: What is the boiling point of water in Celsius?\nTask s6: Who wrote Romeo and Juliet?\nTask s7: What is the square root of 144?\nTask s8: What is the largest ocean on Earth?"

	fullContent := systemPrompt + "\n\n" + taskContent

	// Measure text token cost
	textBody := map[string]any{
		"model":      "accounts/fireworks/models/minimax-m3",
		"messages":   []map[string]any{{"role": "user", "content": fullContent + "\n\nReturn only the JSON answers."}},
		"max_tokens": 10,
	}
	textPT := callLiveAPI(t, apiKey, textBody, 30*time.Second).PromptTokens
	t.Logf("Text only: prompt_tokens=%d (content=%d chars)", textPT, len(fullContent))

	scales := []int{1, 2, 3}
	for _, scale := range scales {
		cfg := DefaultRenderConfig()
		cfg.Scale = scale

		images, err := RenderText(fullContent, cfg)
		if err != nil {
			t.Fatalf("RenderText scale=%d: %v", scale, err)
		}

		totalW := images[0].W
		totalH := 0
		for _, img := range images {
			totalH += img.H
		}

		content := []map[string]any{
			{"type": "text", "text": "The image contains system prompt and tasks. Read the tasks and return JSON answers."},
		}
		for _, img := range images {
			content = append(content, map[string]any{
				"type":      "image_url",
				"image_url": map[string]string{"url": img.ToBase64DataURI()},
			})
		}

		imgBody := map[string]any{
			"model":      "accounts/fireworks/models/minimax-m3",
			"messages":   []map[string]any{{"role": "user", "content": content}},
			"max_tokens": 500,
		}

		result := callLiveAPI(t, apiKey, imgBody, 60*time.Second)
		pt, ct, response := result.PromptTokens, result.CompletionTokens, result.Content

		expected := []string{"Paris", "255", "Au", "8", "100", "Shakespeare", "12", "Pacific"}
		correct := 0
		for _, exp := range expected {
			if strings.Contains(strings.ToLower(response), strings.ToLower(exp)) {
				correct++
			}
		}

		savings := textPT - pt
		savingsPct := 0.0
		if textPT > 0 {
			savingsPct = float64(savings) / float64(textPT) * 100
		}

		t.Logf("Scale=%d: img=%dx%d (pages=%d) prompt_tokens=%d completion=%d | accuracy=%d/8 | savings=%d (%.0f%%)",
			scale, totalW, totalH, len(images), pt, ct, correct, savings, savingsPct)

		if correct < 6 {
			t.Logf("  Response: %s", response)
		}
	}
}
