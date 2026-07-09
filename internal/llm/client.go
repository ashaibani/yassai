package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	openai "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

type Config struct {
	APIKey  string
	BaseURL string
	Model   string
	Timeout time.Duration
}

type Client struct {
	client openai.Client
	model  string
}

type Message struct {
	Role    string
	Content string
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	CachedTokens     int `json:"cached_tokens"`
	ReasoningTokens  int `json:"reasoning_tokens"`
}

func New(cfg Config) *Client {
	if cfg.Timeout <= 0 {
		cfg.Timeout = 180 * time.Second
	}
	c := openai.NewClient(
		option.WithAPIKey(cfg.APIKey),
		option.WithBaseURL(strings.TrimRight(cfg.BaseURL, "/")),
		option.WithRequestTimeout(cfg.Timeout),
		option.WithMaxRetries(1),
	)
	return &Client{client: c, model: cfg.Model}
}

type ChatResult struct {
	Content          string
	ReasoningContent string
	Usage            Usage
}

func (c *Client) Chat(ctx context.Context, messages []Message, maxTokens int, reasoningEffort string) (string, Usage, error) {
	res, err := c.ChatDetailed(ctx, messages, maxTokens, reasoningEffort)
	return res.Content, res.Usage, err
}

func (c *Client) ChatDetailed(ctx context.Context, messages []Message, maxTokens int, reasoningEffort string) (ChatResult, error) {
	wireMessages := make([]map[string]string, len(messages))
	for i, m := range messages {
		wireMessages[i] = map[string]string{"role": m.Role, "content": m.Content}
	}
	body := map[string]any{
		"model":             c.model,
		"messages":          wireMessages,
		"max_tokens":        maxTokens,
		"temperature":       0,
		"top_k":             40,
		"presence_penalty":  0,
		"frequency_penalty": 0,
	}
	if reasoningEffort != "" {
		body["reasoning_effort"] = reasoningEffort
	}
	var raw json.RawMessage
	if err := c.client.Post(ctx, "chat/completions", body, &raw); err != nil {
		return ChatResult{}, err
	}
	var parsed chatResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return ChatResult{}, fmt.Errorf("parse chat response: %w: %s", err, string(raw))
	}
	if len(parsed.Choices) == 0 {
		return ChatResult{Usage: parsed.Usage.toUsage()}, fmt.Errorf("chat response had no choices: %s", string(raw))
	}
	content := parsed.Choices[0].Message.Content.String()
	if strings.TrimSpace(content) == "" && parsed.Choices[0].Text != "" {
		content = parsed.Choices[0].Text
	}
	reasoning := parsed.Choices[0].Message.ReasoningContent
	// Some Fireworks reasoning models return useful text only in
	// reasoning_content. Treat it as content so the agent can parse or follow up
	// instead of falling into empty-output retries and generic fallbacks.
	if strings.TrimSpace(content) == "" && strings.TrimSpace(reasoning) != "" {
		content = reasoning
	}
	usage := parsed.Usage.toUsage()
	if usage.ReasoningTokens == 0 && reasoning != "" {
		usage.ReasoningTokens = len(reasoning)
	}
	return ChatResult{Content: content, ReasoningContent: reasoning, Usage: usage}, nil
}

type apiUsage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
}

func (a apiUsage) toUsage() Usage {
	cached := a.PromptTokensDetails.CachedTokens
	reasoning := a.CompletionTokensDetails.ReasoningTokens
	return Usage{
		PromptTokens:     a.PromptTokens,
		CompletionTokens: a.CompletionTokens,
		TotalTokens:      a.TotalTokens,
		CachedTokens:     cached,
		ReasoningTokens:  reasoning,
	}
}

type chatResponse struct {
	Choices []struct {
		Text    string `json:"text,omitempty"`
		Message struct {
			Content          flexibleContent `json:"content"`
			ReasoningContent string          `json:"reasoning_content,omitempty"`
		} `json:"message"`
	} `json:"choices"`
	Usage apiUsage `json:"usage"`
}

type flexibleContent string

func (c *flexibleContent) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		*c = flexibleContent(s)
		return nil
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(b, &parts); err == nil {
		var out strings.Builder
		for _, p := range parts {
			if p.Text != "" {
				out.WriteString(p.Text)
			}
		}
		*c = flexibleContent(out.String())
		return nil
	}
	*c = flexibleContent(string(b))
	return nil
}

func (c flexibleContent) String() string { return string(c) }
