package memory

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/ashaibani/yassai/internal/agenttypes"
)

func defaultYassaiHome() string {
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".yassai")
	}
	return ".yassai"
}

// DefaultIndex is intentionally minimal. This text is injected into EVERY task
// prompt, so it must carry no developer/project-management notes (that was pure
// token overhead and off-topic for task-solving). The agent may append durable
// cross-task facts here at runtime if any genuinely appear.
const DefaultIndex = `# MEMORY.md

Durable cross-task notes only; empty until something worth persisting appears.
`

type Store struct {
	root        string
	indexPath   string
	maxIndex    int
	maxSelected int
}
type Bundle struct {
	Index    string   `json:"index"`
	Docs     []Doc    `json:"docs,omitempty"`
	Selected []string `json:"selected,omitempty"`
}
type Doc struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

func New(root string) *Store {
	if strings.TrimSpace(root) == "" {
		root = "."
	}
	s := &Store{root: root, indexPath: filepath.Join(root, "MEMORY.md"), maxIndex: 32000, maxSelected: 160000}
	_ = s.EnsureDefaults()
	return s
}

func (s *Store) EnsureDefaults() error {
	if err := os.MkdirAll(filepath.Join(s.root, "memories"), 0o755); err != nil {
		return err
	}
	if _, err := os.Stat(s.indexPath); os.IsNotExist(err) {
		if err := os.WriteFile(s.indexPath, []byte(DefaultIndex), 0o644); err != nil {
			return err
		}
	}
	// No seeded memory docs: injecting dev/PM notes into every task prompt is
	// pure token overhead. The agent can create memory at runtime if warranted.
	defaults := map[string]string{}
	for rel, content := range defaults {
		path := filepath.Join(s.root, rel)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				return err
			}
			if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *Store) LoadForTasks(tasks []agenttypes.Task) Bundle {
	_ = s.EnsureDefaults()
	index := readTrimmed(s.indexPath, s.maxIndex)
	query := strings.ToLower(index + "\n" + taskText(tasks))
	candidates := s.indexedPaths(index)
	if len(candidates) == 0 {
		candidates = s.allMemoryPaths()
	}
	type scored struct {
		path  string
		score int
	}
	var scoredPaths []scored
	for _, rel := range candidates {
		content := readTrimmed(filepath.Join(s.root, rel), 4096)
		score := relevance(query, rel+"\n"+content)
		if strings.Contains(rel, "user-preferences") {
			score += 5
		}
		if score > 0 {
			scoredPaths = append(scoredPaths, scored{rel, score})
		}
	}
	sort.Slice(scoredPaths, func(i, j int) bool {
		if scoredPaths[i].score == scoredPaths[j].score {
			return scoredPaths[i].path < scoredPaths[j].path
		}
		return scoredPaths[i].score > scoredPaths[j].score
	})
	bundle := Bundle{Index: index}
	used := 0
	for _, sp := range scoredPaths {
		content := readTrimmed(filepath.Join(s.root, sp.path), s.maxSelected-used)
		if strings.TrimSpace(content) == "" {
			continue
		}
		used += len(content)
		bundle.Selected = append(bundle.Selected, sp.path)
		bundle.Docs = append(bundle.Docs, Doc{sp.path, content})
		if used >= s.maxSelected || len(bundle.Docs) >= 6 {
			break
		}
	}
	return bundle
}

func (s *Store) Snapshot() map[string]string {
	b := s.LoadForTasks(nil)
	out := map[string]string{"MEMORY.md": b.Index}
	for _, d := range b.Docs {
		out[d.Path] = d.Content
	}
	return out
}
func (s *Store) indexedPaths(index string) []string {
	seen := map[string]bool{}
	var out []string
	for _, line := range strings.Split(index, "\n") {
		for _, field := range strings.Split(line, "|") {
			field = strings.TrimSpace(field)
			if strings.HasPrefix(field, "memories/") && strings.HasSuffix(field, ".md") && !seen[field] {
				seen[field] = true
				out = append(out, field)
			}
		}
	}
	return out
}
func (s *Store) allMemoryPaths() []string {
	var out []string
	_ = filepath.WalkDir(filepath.Join(s.root, "memories"), func(path string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() && strings.HasSuffix(path, ".md") {
			if rel, err := filepath.Rel(s.root, path); err == nil {
				out = append(out, filepath.ToSlash(rel))
			}
		}
		return nil
	})
	sort.Strings(out)
	return out
}
func readTrimmed(path string, limit int) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	s := string(b)
	if limit > 0 && len(s) > limit {
		return s[:limit] + fmt.Sprintf("\n[...truncated at %d bytes on %s...]\n", limit, time.Now().Format("02 January 2006"))
	}
	return s
}
func taskText(tasks []agenttypes.Task) string {
	var b strings.Builder
	for _, t := range tasks {
		b.WriteString(t.TaskID)
		b.WriteByte(' ')
		b.WriteString(t.Prompt)
		b.WriteByte('\n')
	}
	return b.String()
}
func relevance(query, doc string) int {
	q := tokens(query)
	d := strings.ToLower(doc)
	score := 0
	for tok := range q {
		if len(tok) >= 4 && strings.Contains(d, tok) {
			score++
		}
	}
	return score
}
func tokens(s string) map[string]bool {
	out := map[string]bool{}
	for _, f := range strings.FieldsFunc(strings.ToLower(s), func(r rune) bool { return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '-') }) {
		if len(f) >= 4 {
			out[f] = true
		}
	}
	return out
}
