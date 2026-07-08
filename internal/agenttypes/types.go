package agenttypes

type Task struct {
	TaskID string `json:"task_id"`
	Prompt string `json:"prompt"`
}

type Result struct {
	TaskID string `json:"task_id"`
	Answer string `json:"answer"`
}
