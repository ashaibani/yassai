package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestChatWiresImageLabelsImmediatelyBeforeImages(t *testing.T) {
	var requestBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request: %v", err)
		}
		if err := json.Unmarshal(body, &requestBody); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"finish_reason":"stop","message":{"content":"ok"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer server.Close()

	client := New(Config{APIKey: "test", BaseURL: server.URL, Model: "test-model", Timeout: time.Second})
	_, err := client.ChatWithOptions(context.Background(), []Message{{
		Role:        "user",
		Content:     "questions",
		ImageURLs:   []string{"data:image/png;base64,AAAA", "data:image/png;base64,BBBB"},
		ImageLabels: []string{"IMAGE:a source text", "IMAGE:b source text"},
	}}, ChatOptions{MaxTokens: 10})
	if err != nil {
		t.Fatal(err)
	}

	messages := requestBody["messages"].([]any)
	content := messages[0].(map[string]any)["content"].([]any)
	if len(content) != 5 {
		t.Fatalf("content parts=%d want 5: %#v", len(content), content)
	}
	wantTypes := []string{"text", "text", "image_url", "text", "image_url"}
	for i, want := range wantTypes {
		if got := content[i].(map[string]any)["type"]; got != want {
			t.Fatalf("part %d type=%v want %s", i, got, want)
		}
	}
	if got := content[1].(map[string]any)["text"]; got != "IMAGE:a source text" {
		t.Fatalf("first image label=%v", got)
	}
}
