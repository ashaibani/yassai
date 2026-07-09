package textimg

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"testing"
	"time"
)

const liveTestEnv = "TEXTIMG_LIVE_TESTS"

type liveChatResult struct {
	Content          string
	ReasoningContent string
	FinishReason     string
	PromptTokens     int
	CompletionTokens int
	Raw              []byte
}

func requireLiveAPI(t *testing.T) string {
	t.Helper()
	if os.Getenv(liveTestEnv) != "1" {
		t.Skipf("set %s=1 to run Fireworks network experiments", liveTestEnv)
	}
	key := os.Getenv("FIREWORKS_API_KEY")
	if key == "" {
		t.Skip("FIREWORKS_API_KEY not set")
	}
	return key
}

// callLiveAPI is deliberately defensive: model responses may legally contain
// content:null (for example when the provider emits only reasoning/tool data).
// Live experiments should report that as an empty result, never panic CI.
func callLiveAPI(t *testing.T, apiKey string, body map[string]any, timeout time.Duration) liveChatResult {
	t.Helper()
	payload, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.fireworks.ai/inference/v1/chat/completions", bytes.NewReader(payload))
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("Fireworks API: %v", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read Fireworks response: %v", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		t.Fatalf("Fireworks status %d: %s", resp.StatusCode, raw)
	}
	var wire struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Content          *string `json:"content"`
				ReasoningContent string  `json:"reasoning_content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
		Error any `json:"error"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatalf("decode Fireworks response: %v: %s", err, raw)
	}
	if wire.Error != nil {
		t.Fatalf("Fireworks error: %v", wire.Error)
	}
	result := liveChatResult{
		PromptTokens:     wire.Usage.PromptTokens,
		CompletionTokens: wire.Usage.CompletionTokens,
		Raw:              raw,
	}
	if len(wire.Choices) == 0 {
		t.Logf("Fireworks returned no choices: %s", raw)
		return result
	}
	choice := wire.Choices[0]
	result.FinishReason = choice.FinishReason
	result.ReasoningContent = choice.Message.ReasoningContent
	if choice.Message.Content != nil {
		result.Content = *choice.Message.Content
	} else {
		t.Logf("Fireworks returned content:null (finish=%s, reasoning_chars=%d)", choice.FinishReason, len(choice.Message.ReasoningContent))
	}
	return result
}

func (r liveChatResult) String() string {
	return fmt.Sprintf("prompt=%d completion=%d finish=%s", r.PromptTokens, r.CompletionTokens, r.FinishReason)
}
