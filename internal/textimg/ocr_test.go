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

func TestOCRAccuracy(t *testing.T) {
	apiKey := os.Getenv("FIREWORKS_API_KEY")
	if apiKey == "" {
		t.Skip("FIREWORKS_API_KEY not set")
	}

	tests := []struct {
		name string
		text string
	}{
		{
			name: "simple_numbers",
			text: "The answer is 42. Pi is 3.14159. Speed of light is 299792458 m/s.",
		},
		{
			name: "json_structure",
			text: "\"task_id\":\"s1\",\"answer\":\"Paris\",\"confidence\":0.95}\n{\"task_id\":\"s2\",\"answer\":\"42\",\"confidence\":1.0}",
		},
		{
			name: "code_block",
			text: "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)\nprint(fibonacci(10))",
		},
		{
			name: "mixed_content",
			text: "Task: What is the capital of France?\nAnswer: Paris\n\nTask: What is 15 * 17?\nAnswer: 255\n\nTask: Name the largest planet.\nAnswer: Jupiter",
		},
	}

	cfg := DefaultRenderConfig()
	cfg.Scale = 2

	totalCorrect := 0
	totalTests := 0

	for _, tt := range tests {
		images, err := RenderText(tt.text, cfg)
		if err != nil {
			t.Fatalf("RenderText(%s): %v", tt.name, err)
		}
		if len(images) == 0 {
			t.Fatalf("RenderText(%s): no images", tt.name)
		}

		content := []map[string]any{
			{"type": "text", "text": "Read the text in this image and return it exactly. Return ONLY the text content, nothing else."},
			{"type": "image_url", "image_url": map[string]string{"url": images[0].ToBase64DataURI()}},
		}

		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		body := map[string]any{
			"model":      "accounts/fireworks/models/minimax-m3",
			"messages":   []map[string]any{{"role": "user", "content": content}},
			"max_tokens": 500,
		}
		jsonBody, _ := json.Marshal(body)
		req, _ := http.NewRequestWithContext(ctx, "POST", "https://api.fireworks.ai/inference/v1/chat/completions", strings.NewReader(string(jsonBody)))
		req.Header.Set("Authorization", "Bearer "+apiKey)
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Errorf("API call for %s: %v", tt.name, err)
			continue
		}
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		var result map[string]any
		json.Unmarshal(raw, &result)
		choices, ok := result["choices"].([]any)
		if !ok || len(choices) == 0 {
			t.Errorf("No choices for %s: %s", tt.name, string(raw))
			continue
		}
		choice := choices[0].(map[string]any)
		msg := choice["message"].(map[string]any)
		readback := msg["content"].(string)
		readback = strings.TrimSpace(readback)

		keyFacts := extractKeyFacts(tt.text)
		correctFacts := 0
		for _, fact := range keyFacts {
			if strings.Contains(readback, fact) {
				correctFacts++
			}
		}
		totalCorrect += correctFacts
		totalTests += len(keyFacts)

		pct := 0.0
		if len(keyFacts) > 0 {
			pct = float64(correctFacts) / float64(len(keyFacts)) * 100
		}

		usage := result["usage"].(map[string]any)
		pt := int(usage["prompt_tokens"].(float64))

		t.Logf("OCR %s: %d/%d facts (%.0f%%) | img=%dx%d prompt_tokens=%d",
			tt.name, correctFacts, len(keyFacts), pct,
			images[0].W, images[0].H, pt)

		if pct < 80 {
			t.Logf("  Original: %s", tt.text)
			t.Logf("  Readback: %s", readback)
		}
	}

	if totalTests > 0 {
		overallPct := float64(totalCorrect) / float64(totalTests) * 100
		t.Logf("\nOverall OCR accuracy: %d/%d (%.0f%%)", totalCorrect, totalTests, overallPct)
		if overallPct < 70 {
			t.Errorf("OCR accuracy too low (%.0f%%) - pxpipe approach not viable", overallPct)
		}
	}
}

func extractKeyFacts(s string) []string {
	var facts []string
	tokens := strings.Fields(s)
	for _, tok := range tokens {
		tok = strings.Trim(tok, "{}[]\",:;()")
		if tok == "" {
			continue
		}
		if isNumber(tok) || isCapitalized(tok) {
			facts = append(facts, tok)
		}
	}
	return facts
}

func isNumber(s string) bool {
	hasDigit := false
	for _, c := range s {
		if c >= '0' && c <= '9' {
			hasDigit = true
		} else if c != '.' && c != '-' {
			return false
		}
	}
	return hasDigit
}

func isCapitalized(s string) bool {
	if len(s) == 0 {
		return false
	}
	c := s[0]
	return c >= 'A' && c <= 'Z'
}

// unused but kept for reference
var _ = fmt.Sprintf
