package textimg

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestOCRAccuracy(t *testing.T) {
	apiKey := requireLiveAPI(t)

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

		body := map[string]any{
			"model":      "accounts/fireworks/models/minimax-m3",
			"messages":   []map[string]any{{"role": "user", "content": content}},
			"max_tokens": 500,
		}
		result := callLiveAPI(t, apiKey, body, 30*time.Second)
		readback := strings.TrimSpace(result.Content)

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

		t.Logf("OCR %s: %d/%d facts (%.0f%%) | img=%dx%d prompt_tokens=%d",
			tt.name, correctFacts, len(keyFacts), pct,
			images[0].W, images[0].H, result.PromptTokens)

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
