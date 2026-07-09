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
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
}

// ToolDef is an OpenAI-compatible function tool definition.
type ToolDef struct {
	Type     string         `json:"type"` // always "function"
	Function ToolFunction   `json:"function"`
}

type ToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// ToolCall is a model-requested function invocation.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
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
	ToolCalls        []ToolCall
	FinishReason     string
	Usage            Usage
}

type ChatOptions struct {
	MaxTokens       int
	ReasoningEffort string
	Tools           []ToolDef
	// ToolChoice: "", "auto", "none", or "required". Empty omits the field.
	ToolChoice string
}

func (c *Client) Chat(ctx context.Context, messages []Message, maxTokens int, reasoningEffort string) (string, Usage, error) {
	res, err := c.ChatDetailed(ctx, messages, maxTokens, reasoningEffort)
	return res.Content, res.Usage, err
}

func (c *Client) ChatDetailed(ctx context.Context, messages []Message, maxTokens int, reasoningEffort string) (ChatResult, error) {
	return c.ChatWithOptions(ctx, messages, ChatOptions{MaxTokens: maxTokens, ReasoningEffort: reasoningEffort})
}

func (c *Client) ChatWithOptions(ctx context.Context, messages []Message, opts ChatOptions) (ChatResult, error) {
	wireMessages := make([]map[string]any, len(messages))
	for i, m := range messages {
		msg := map[string]any{"role": m.Role}
		switch m.Role {
		case "tool":
			msg["content"] = m.Content
			if m.ToolCallID != "" {
				msg["tool_call_id"] = m.ToolCallID
			}
			if m.Name != "" {
				msg["name"] = m.Name
			}
		default:
			// Assistant messages that only carry tool_calls may have empty content.
			if m.Content != "" || len(m.ToolCalls) == 0 {
				msg["content"] = m.Content
			}
			if len(m.ToolCalls) > 0 {
				msg["tool_calls"] = m.ToolCalls
			}
		}
		wireMessages[i] = msg
	}
	body := map[string]any{
		"model":             c.model,
		"messages":          wireMessages,
		"max_tokens":        opts.MaxTokens,
		"temperature":       0,
		"top_k":             40,
		"presence_penalty":  0,
		"frequency_penalty": 0,
	}
	if opts.ReasoningEffort != "" {
		body["reasoning_effort"] = opts.ReasoningEffort
	}
	if len(opts.Tools) > 0 {
		body["tools"] = opts.Tools
		if opts.ToolChoice != "" {
			body["tool_choice"] = opts.ToolChoice
		}
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
	choice := parsed.Choices[0]
	content := choice.Message.Content.String()
	if strings.TrimSpace(content) == "" && choice.Text != "" {
		content = choice.Text
	}
	reasoning := choice.Message.ReasoningContent
	// Some Fireworks reasoning models return useful text only in
	// reasoning_content. Treat it as content so the agent can parse or follow up
	// instead of falling into empty-output retries and generic fallbacks.
	if strings.TrimSpace(content) == "" && strings.TrimSpace(reasoning) != "" && len(choice.Message.ToolCalls) == 0 {
		content = reasoning
	}
	usage := parsed.Usage.toUsage()
	if usage.ReasoningTokens == 0 && reasoning != "" {
		usage.ReasoningTokens = len(reasoning)
	}
	return ChatResult{
		Content:          content,
		ReasoningContent: reasoning,
		ToolCalls:        choice.Message.ToolCalls,
		FinishReason:     choice.FinishReason,
		Usage:            usage,
	}, nil
}

// PythonExecTool is the native function-tool schema for MicroPython execution.
func PythonExecTool() ToolDef {
	return ToolDef{
		Type: "function",
		Function: ToolFunction{
			Name:        "run_python",
			Description: "Execute MicroPython (stdlib only) and return stdout/json. Use for multi-step maths. Print the final answer(s) or a JSON answers object.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"code": map[string]any{
						"type":        "string",
						"description": "MicroPython source to run. Prefer print() of final numbers or a JSON object with answers.",
					},
				},
				"required": []string{"code"},
			},
		},
	}
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
		Text         string `json:"text,omitempty"`
		FinishReason string `json:"finish_reason,omitempty"`
		Message      struct {
			Content          flexibleContent `json:"content"`
			ReasoningContent string          `json:"reasoning_content,omitempty"`
			ToolCalls        []ToolCall      `json:"tool_calls,omitempty"`
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
