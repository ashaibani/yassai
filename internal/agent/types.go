package agent

import (
	"time"

	"github.com/ashaibani/yassai/internal/agenttypes"
	"github.com/ashaibani/yassai/internal/llm"
)

type Task = agenttypes.Task

type Result = agenttypes.Result

type Config struct {
	APIKey           string
	BaseURL          string
	AllowedModels    []string
	PreferredModel   string
	MaxBatchSize     int
	MathBatchSize    int                 // chunk size for mathematical_reasoning groups (0 = use MaxBatchSize)
	BatchIsolation   string              // "focus" (default), "math", or "none"
	MaxTurns         int                 // per-batch LLM turn limit (0 = default 4)
	MaxBatchTokens   int                 // per-batch task-content token budget (0 = default 8000)
	MaxConcurrency   int                 // max parallel batch workers (0 = default 3)
	ReasoningEffort  string              // "", "low", "medium", "high", "xhigh"; "" = adaptive by category (see effortForTask)
	Categories       map[string][]string // optional task_id -> categories override (eval); nil = classify at runtime
	DisableHints     bool                // if true, skip category technique-hint injection (for eval A/B)
	MaxContextTokens int
	MemoryRoot       string
	SkillRoots       []string
	Timeout          time.Duration
	ClassifierDir    string // artefact dir (onnx_config.json, vocab.txt, model); empty disables
	ClassifierLib    string // path to libonnxruntime (or via ONNXRUNTIME_LIB)

	// EffortTierMap overrides the default per-category reasoning effort.
	// nil = use the built-in map (only logical_deductive_reasoning -> xhigh).
	// Empty map = all categories use "low" effort.
	EffortTierMap map[string]string

	// ModelRouteMap routes specific categories to a different model.
	// nil/empty = use the single PreferredModel for all categories.
	// e.g. {"code_generation": "accounts/fireworks/models/kimi-k2p7-code"}
	ModelRouteMap map[string]string

	// TraceMessages stores full request messages, assistant outputs, and any
	// provider reasoning_content in metrics.call_records for callback telemetry.
	TraceMessages bool

	// TextImg renders non-code batch content as 1-bit PNGs (scale=1 bitmap font)
	// so Fireworks vision tokenisation (~1 token/32x32px) undercuts BPE text.
	// "" or "off" = plain text; "auto"/"hybrid" = quoted source passages only;
	// "tasks" = category-grouped full task sheets;
	// "dense" = one mixed task sheet (experimental); "full" = category recipes
	// plus grouped task sheets. The JSON output contract always stays as text.
	// Code-exec batches always stay text: maths/logic need digit-exact reads.
	TextImg string

	// LocalModelPath enables the in-container MiniCPM5 GGUF for maths/logic
	// tasks (local-first with verified fallback). Local answers count toward
	// accuracy but not the token score, so each accepted answer is free.
	// Empty disables. LocalLibPath is the llama.cpp shared-library directory.
	LocalModelPath string
	LocalLibPath   string

	// LocalBaseModelPath enables the UN-tuned MiniCPM5 base GGUF as a second
	// local lane for code_generation and gated NER - families the fine-tune
	// lost to tool-contract specialisation. Empty disables.
	LocalBaseModelPath string
}

type BatchPlanRecord struct {
	Index   int      `json:"index"`
	TaskIDs []string `json:"task_ids"`
	Size    int      `json:"size"`
	Effort  string   `json:"effort"`
	Model   string   `json:"model"`
}

type CallRecord struct {
	Turn             int           `json:"turn"`
	Timestamp        time.Time     `json:"timestamp"`
	LatencyMS        int64         `json:"latency_ms"`
	PromptTokens     int           `json:"prompt_tokens"`
	CompletionTokens int           `json:"completion_tokens"`
	TotalTokens      int           `json:"total_tokens"`
	CachedTokens     int           `json:"cached_tokens"`
	ReasoningTokens  int           `json:"reasoning_tokens"`
	BatchSize        int           `json:"batch_size"`
	OutputChars      int           `json:"output_chars"`
	TaskIDs          []string      `json:"task_ids"`
	Model            string        `json:"model,omitempty"`
	Effort           string        `json:"effort,omitempty"`
	MaxTokens        int           `json:"max_tokens,omitempty"`
	Messages         []llm.Message `json:"messages,omitempty"`
	AssistantMessage string        `json:"assistant_message,omitempty"`
	ReasoningContent string        `json:"reasoning_content,omitempty"`
	ToolTraces       []ToolTrace   `json:"tool_traces,omitempty"`
	Error            string        `json:"error,omitempty"`
}

// ToolTrace records a single code-execution observation within a batch turn.
type ToolTrace struct {
	Index  int    `json:"index"`
	Code   string `json:"code"`
	Stdout string `json:"stdout,omitempty"`
	JSON   string `json:"json,omitempty"`
	Error  string `json:"error,omitempty"`
}

type Metrics struct {
	Model           string              `json:"model"`
	Calls           int                 `json:"calls"`
	PromptTokens    int                 `json:"prompt_tokens"`
	OutputTokens    int                 `json:"output_tokens"`
	TotalTokens     int                 `json:"total_tokens"`
	CachedTokens    int                 `json:"cached_tokens"`
	ReasoningTokens int                 `json:"reasoning_tokens"`
	ToolRuns        int                 `json:"tool_runs"`
	LocalAnswers    int                 `json:"local_answers"`
	BatchCount      int                 `json:"batch_count"`
	Fallbacks       int                 `json:"fallbacks"`
	StartedAt       time.Time           `json:"started_at"`
	FinishedAt      time.Time           `json:"finished_at"`
	DurationMS      int64               `json:"duration_ms"`
	BatchSummaries  []BatchRun          `json:"batch_summaries,omitempty"`
	BatchPlan       []BatchPlanRecord   `json:"batch_plan,omitempty"`
	Classifications map[string][]string `json:"classifications,omitempty"`
	CallRecords     []CallRecord        `json:"call_records,omitempty"`
}

type BatchRun struct {
	TaskIDs []string `json:"task_ids"`
	Calls   int      `json:"calls"`
	Tools   int      `json:"tools"`
	Error   string   `json:"error,omitempty"`
}

// RoutingConfig is the serialisable view of the routing maps used by the demo API.
type RoutingConfig struct {
	EffortTierMap map[string]string `json:"effort_tier_map"`
	ModelRouteMap map[string]string `json:"model_route_map"`
	DefaultModel  string            `json:"default_model"`
}

// DefaultEffortTier is the demo-oriented high-accuracy map.
func DefaultEffortTier() map[string]string {
	return map[string]string{
		"logical_deductive_reasoning": "xhigh",
		"code_debugging":              "xhigh",
		"code_generation":             "xhigh",
	}
}

// LeanEffortTiers is the token-efficient leaderboard map. Reasoning_effort is
// "none" almost everywhere (Fireworks needs the literal value to disable
// thinking, which is the dominant token cost). Maths AND logic are "none"
// because they are solved with a run_python tool call - the executed code does
// the reasoning deterministically, so the model only has to translate the
// problem, not think through it (this kills the logic reasoning-token hog).
// code_debugging keeps a small "low" budget because the minimal-fix is easy to
// get subtly wrong; bump others only if live grading regresses.
func LeanEffortTiers() map[string]string {
	return map[string]string{
		"factual_knowledge":           "none",
		"sentiment_classification":    "none",
		"text_summarisation":          "none",
		"named_entity_recognition":    "none",
		"mathematical_reasoning":      "none",
		"logical_deductive_reasoning": "none",
		"code_debugging":              "none",
		"code_generation":             "none",
	}
}

// DefaultModelRouteMap returns the built-in default model routing map, filtered
// to only include models that are actually in the allowed list. If kimi-k2p7-code
// is available, code_debugging and code_generation are routed to it.
func DefaultModelRouteMap(allowedModels []string) map[string]string {
	hasKimi := false
	for _, m := range allowedModels {
		if m == "accounts/fireworks/models/kimi-k2p7-code" {
			hasKimi = true
			break
		}
	}
	if !hasKimi {
		return map[string]string{}
	}
	return map[string]string{
		"code_debugging":  "accounts/fireworks/models/kimi-k2p7-code",
		"code_generation": "accounts/fireworks/models/kimi-k2p7-code",
	}
}
