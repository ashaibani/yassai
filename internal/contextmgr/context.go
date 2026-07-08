package contextmgr

import (
	"encoding/json"

	"github.com/ashaibani/yassai/internal/agenttypes"
	"github.com/ashaibani/yassai/internal/memory"
	"github.com/ashaibani/yassai/internal/skills"
)

const DefaultContextTokens = 200000

type Manager struct {
	MaxContextTokens int
	ReserveTokens    int
}
type taskView struct {
	TaskID     string   `json:"task_id"`
	Prompt     string   `json:"prompt"`
	Categories []string `json:"categories,omitempty"`
}
type promptPayload struct {
	Tasks        []taskView     `json:"tasks"`
	Memory       *memory.Bundle `json:"memory,omitempty"`
	Skills       *skills.Bundle `json:"skills,omitempty"`
	Instructions []string       `json:"instructions,omitempty"`
}

func (m Manager) BuildBatchPrompt(tasks []agenttypes.Task, mem memory.Bundle, skillBundle skills.Bundle, categories map[string][]string) string {
	budgetChars := m.promptBudgetChars()
	fixedBudget := len(mem.Index) + approxSkillsSize(skillBundle) + 12000
	remaining := budgetChars - fixedBudget
	if remaining < 8000 {
		remaining = 8000
	}
	perTask := remaining
	if len(tasks) > 0 {
		perTask = remaining / len(tasks)
		if perTask < 4000 {
			perTask = 4000
		}
	}
	views := make([]taskView, len(tasks))
	for i, task := range tasks {
		views[i] = taskView{TaskID: task.TaskID, Prompt: trimMiddle(task.Prompt, perTask), Categories: categories[task.TaskID]}
	}
	payload := promptPayload{Tasks: views}
	// Include memory only when it actually carries notes/docs. The default is an
	// empty clean-slate index and the agent runs once (it writes no memory), so
	// injecting it every call would be pure token overhead. Answer-shaping rules
	// live in the system prompt, not here.
	if trimmed := trimMemory(mem, budgetChars/3); len(trimmed.Docs) > 0 {
		payload.Memory = &trimmed
	}
	// Include skills only when relevant ones were actually loaded (none are
	// bundled in the container; this covers optional local skill dirs).
	if trimmed := trimSkills(skillBundle, budgetChars/5); len(trimmed.Loaded) > 0 {
		payload.Skills = &trimmed
		payload.Instructions = []string{"Apply any provided skills that fit the task."}
	}
	b, _ := json.Marshal(payload)
	return "Solve these tasks. Context follows as JSON. " + string(b)
}
func (m Manager) promptBudgetChars() int {
	tokens := m.MaxContextTokens
	if tokens <= 0 {
		tokens = DefaultContextTokens
	}
	reserve := m.ReserveTokens
	if reserve <= 0 {
		reserve = 24000
	}
	if tokens <= reserve {
		return tokens * 4
	}
	return (tokens - reserve) * 4
}
func approxSkillsSize(b skills.Bundle) int {
	n := 0
	for _, s := range b.Index {
		n += len(s.Name) + len(s.Description) + 64
	}
	for _, s := range b.Loaded {
		n += len(s.Content)
	}
	return n
}
func trimMemory(mem memory.Bundle, max int) memory.Bundle {
	if max <= 0 {
		return mem
	}
	used := len(mem.Index)
	out := memory.Bundle{Index: trimMiddle(mem.Index, max/2), Selected: mem.Selected}
	for _, d := range mem.Docs {
		left := max - used
		if left <= 0 {
			break
		}
		content := d.Content
		if len(content) > left {
			content = trimMiddle(content, left)
		}
		out.Docs = append(out.Docs, memory.Doc{Path: d.Path, Content: content})
		used += len(content)
	}
	return out
}
func trimSkills(bundle skills.Bundle, max int) skills.Bundle {
	if max <= 0 {
		return bundle
	}
	out := skills.Bundle{Index: bundle.Index}
	used := 0
	for _, s := range bundle.Loaded {
		if used >= max {
			break
		}
		content := s.Content
		if len(content) > max-used {
			content = trimMiddle(content, max-used)
		}
		s.Content = content
		out.Loaded = append(out.Loaded, s)
		used += len(content)
	}
	return out
}
func trimMiddle(s string, max int) string {
	if max <= 0 || len(s) <= max {
		return s
	}
	head := max * 2 / 3
	tail := max - head - 80
	if tail < 0 {
		tail = 0
	}
	return s[:head] + "\n[...middle omitted for context budget...]\n" + s[len(s)-tail:]
}
