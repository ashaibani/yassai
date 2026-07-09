package llm

import (
	"encoding/json"
	"testing"
)

func TestPythonExecToolSchema(t *testing.T) {
	tool := PythonExecTool()
	if tool.Type != "function" || tool.Function.Name != "run_python" {
		t.Fatalf("unexpected tool: %+v", tool)
	}
	b, err := json.Marshal(tool)
	if err != nil {
		t.Fatal(err)
	}
	if !json.Valid(b) {
		t.Fatalf("invalid json: %s", b)
	}
}

func TestChatResponseToolCallsUnmarshal(t *testing.T) {
	raw := []byte(`{
  "choices": [{
    "finish_reason": "tool_calls",
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_1",
        "type": "function",
        "function": {"name": "run_python", "arguments": "{\"code\":\"print(1)\"}"}
      }]
    }
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
}`)
	var parsed chatResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatal(err)
	}
	if len(parsed.Choices) != 1 || len(parsed.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("tool calls: %+v", parsed.Choices)
	}
	tc := parsed.Choices[0].Message.ToolCalls[0]
	if tc.Function.Name != "run_python" || tc.ID != "call_1" {
		t.Fatalf("bad tool call: %+v", tc)
	}
}
