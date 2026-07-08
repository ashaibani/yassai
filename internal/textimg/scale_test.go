package textimg

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"
)

func TestScaleComparison(t *testing.T) {
	apiKey := os.Getenv("FIREWORKS_API_KEY")
	if apiKey == "" {
		t.Skip("FIREWORKS_API_KEY not set")
	}

	systemPrompt := "You are a compact, high-accuracy AI agent for AMD Developer Hackathon Track 1. Return exactly this JSON object and nothing else: {\"answers\":[{\"task_id\":\"...\",\"answer\":\"...\"}]}. Preserve each task_id. Keep answers concise but complete."

	taskContent := "Tasks to solve:\n\nTask s1: What is the capital of France?\nTask s2: What is 15 * 17?\nTask s3: What is the chemical symbol for gold?\nTask s4: How many planets are in our solar system?\nTask s5: What is the boiling point of water in Celsius?\nTask s6: Who wrote Romeo and Juliet?\nTask s7: What is the square root of 144?\nTask s8: What is the largest ocean on Earth?"

	fullContent := systemPrompt + "\n\n" + taskContent

	// Measure text token cost
	textBody := map[string]any{
		"model":      "accounts/fireworks/models/minimax-m3",
		"messages":   []map[string]any{{"role": "user", "content": fullContent + "\n\nReturn only the JSON answers."}},
		"max_tokens": 10,
	}
	textPT := callAPI2(t, apiKey, textBody)
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

		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		jsonBody, _ := json.Marshal(imgBody)
		req, _ := http.NewRequestWithContext(ctx, "POST", "https://api.fireworks.ai/inference/v1/chat/completions", strings.NewReader(string(jsonBody)))
		req.Header.Set("Authorization", "Bearer "+apiKey)
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Errorf("scale=%d API call: %v", scale, err)
			continue
		}
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		var result map[string]any
		json.Unmarshal(raw, &result)
		usage := result["usage"].(map[string]any)
		pt := int(usage["prompt_tokens"].(float64))
		ct := int(usage["completion_tokens"].(float64))

		choices, _ := result["choices"].([]any)
		var response string
		if len(choices) > 0 {
			choice := choices[0].(map[string]any)
			msg := choice["message"].(map[string]any)
			response = msg["content"].(string)
		}

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

func callAPI2(t *testing.T, apiKey string, body map[string]any) int {
	jsonBody, _ := json.Marshal(body)
	req, _ := http.NewRequest("POST", "https://api.fireworks.ai/inference/v1/chat/completions", strings.NewReader(string(jsonBody)))
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("API call: %v", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var result map[string]any
	json.Unmarshal(raw, &result)
	usage := result["usage"].(map[string]any)
	return int(usage["prompt_tokens"].(float64))
}
