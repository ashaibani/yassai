package textimg

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"
)

// TestTokenSavingsBySize measures token savings at different content sizes.
// This determines the break-even point where image compression becomes worthwhile.
func TestTokenSavingsBySize(t *testing.T) {
	apiKey := os.Getenv("FIREWORKS_API_KEY")
	if apiKey == "" {
		t.Skip("FIREWORKS_API_KEY not set")
	}

	// Generate content at different sizes
	sizes := []int{500, 2000, 5000, 10000, 20000, 40000}

	for _, targetChars := range sizes {
		// Generate repetitive but realistic content
		content := generateContent(targetChars)

		// 1. Measure text token cost
		textBody := map[string]any{
			"model":      "accounts/fireworks/models/minimax-m3",
			"messages":   []map[string]any{{"role": "user", "content": "Answer: what is 2+2?\n\nContext:\n" + content}},
			"max_tokens": 10,
		}
		textPT := callAPI3(t, apiKey, textBody)

		// 2. Measure image token cost at scale=1 (densest)
		cfg := DefaultRenderConfig()
		cfg.Scale = 1
		images, err := RenderText(content, cfg)
		if err != nil {
			t.Fatalf("RenderText(%d): %v", targetChars, err)
		}

		totalImgH := 0
		for _, img := range images {
			totalImgH += img.H
		}

		content2 := []map[string]any{
			{"type": "text", "text": "Answer: what is 2+2? The context is in the image(s)."},
		}
		for _, img := range images {
			content2 = append(content2, map[string]any{
				"type":      "image_url",
				"image_url": map[string]string{"url": img.ToBase64DataURI()},
			})
		}

		imgBody := map[string]any{
			"model":      "accounts/fireworks/models/minimax-m3",
			"messages":   []map[string]any{{"role": "user", "content": content2}},
			"max_tokens": 10,
		}
		imgPT := callAPI3(t, apiKey, imgBody)

		savings := textPT - imgPT
		savingsPct := 0.0
		if textPT > 0 {
			savingsPct = float64(savings) / float64(textPT) * 100
		}

		t.Logf("chars=%-6d | text=%5d tok | img=%5d tok (%dx%d, %d pages) | savings=%5d (%.0f%%)",
			len(content), textPT, imgPT, images[0].W, totalImgH, len(images), savings, savingsPct)
	}
}

func generateContent(targetChars int) string {
	var sb strings.Builder
	templates := []string{
		"Task: What is the capital of %s? Answer: %s is the capital of %s.",
		"Fact: The %s was discovered in %d by %s. It has a molecular weight of %.2f.",
		"Code: def %s(%s): return %s + %s  # computes the %s of two values",
		"Memory: Previously solved %s for task %s. The answer was %s. Confidence: %.2f.",
		"Instruction: When solving %s tasks, always check %s first, then verify with %s.",
	}
	countries := []string{"France", "Japan", "Brazil", "Egypt", "Sweden", "India", "Canada", "Kenya"}
	capitals := []string{"Paris", "Tokyo", "Brasilia", "Cairo", "Stockholm", "New Delhi", "Ottawa", "Nairobi"}
	years := []int{1928, 1543, 1969, 1812, 1789, 1947, 1895, 1903}
	scientists := []string{"Fleming", "Copernicus", "Armstrong", "Davy", "Lavoisier", "Bardeen", "Rontgen", "Curie"}
	weights := []float64{409.3, 1.008, 22.99, 35.45, 12.011, 15.999, 14.007, 1.008}
	i := 0
	for sb.Len() < targetChars {
		tmpl := templates[i%len(templates)]
		switch i % len(templates) {
		case 0:
			c := countries[i%len(countries)]
			cap := capitals[i%len(capitals)]
			sb.WriteString(fmt.Sprintf(tmpl+"\n", c, cap, c))
		case 1:
			sb.WriteString(fmt.Sprintf(tmpl+"\n", "element", years[i%len(years)], scientists[i%len(scientists)], weights[i%len(weights)]))
		case 2:
			sb.WriteString(fmt.Sprintf(tmpl+"\n", "add", "a", "b", "a", "sum"))
		case 3:
			sb.WriteString(fmt.Sprintf(tmpl+"\n", "arithmetic", "t1", "42", 0.95))
		case 4:
			sb.WriteString(fmt.Sprintf(tmpl+"\n", "math", "MEMORY.md", "computation"))
		}
		i++
	}
	result := sb.String()
	if len(result) > targetChars {
		result = result[:targetChars]
	}
	return result
}

func callAPI3(t *testing.T, apiKey string, body map[string]any) int {
	jsonBody, _ := json.Marshal(body)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "POST", "https://api.fireworks.ai/inference/v1/chat/completions", strings.NewReader(string(jsonBody)))
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
	usage, ok := result["usage"].(map[string]any)
	if !ok {
		t.Fatalf("No usage in response: %s", string(raw))
	}
	return int(usage["prompt_tokens"].(float64))
}
