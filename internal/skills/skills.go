package skills

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type Loader struct{ roots []string }
type Skill struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Path        string `json:"path"`
	Scope       string `json:"scope,omitempty"`
	Content     string `json:"content,omitempty"`
}
type Bundle struct {
	Index  []Skill `json:"index,omitempty"`
	Loaded []Skill `json:"loaded,omitempty"`
}

func NewLoader(roots []string) *Loader {
	if len(roots) == 0 {
		if home, err := os.UserHomeDir(); err == nil {
			roots = append(roots, filepath.Join(home, ".agents", "skills"))
		}
	}
	return &Loader{roots: roots}
}
func (l *Loader) LoadForPrompt(workingDir, taskText string, maxBytes int) Bundle {
	skills := l.list(workingDir)
	query := strings.ToLower(taskText)
	type scored struct {
		skill Skill
		score int
	}
	var ss []scored
	for _, s := range skills {
		score := relevance(query, s.Name+" "+s.Description)
		if score >= 4 {
			ss = append(ss, scored{s, score})
		}
	}
	sort.Slice(ss, func(i, j int) bool {
		if ss[i].score == ss[j].score {
			return ss[i].skill.Name < ss[j].skill.Name
		}
		return ss[i].score > ss[j].score
	})
	bundle := Bundle{Index: make([]Skill, 0, len(ss))}
	for _, s := range ss {
		bundle.Index = append(bundle.Index, Skill{Name: s.skill.Name, Description: s.skill.Description, Path: s.skill.Path, Scope: s.skill.Scope})
	}
	used := 0
	for _, item := range ss {
		if maxBytes > 0 && used+len(item.skill.Content) > maxBytes {
			continue
		}
		bundle.Loaded = append(bundle.Loaded, item.skill)
		used += len(item.skill.Content)
		if len(bundle.Loaded) >= 2 {
			break
		}
	}
	return bundle
}

func (l *Loader) list(workingDir string) []Skill {
	roots := []string{}
	if strings.TrimSpace(workingDir) != "" {
		roots = append(roots, filepath.Join(workingDir, ".agents", "skills"))
	}
	roots = append(roots, l.roots...)
	seen := map[string]bool{}
	var out []Skill
	for _, root := range roots {
		entries, err := os.ReadDir(root)
		if err != nil {
			continue
		}
		for _, entry := range entries {
			if !entry.IsDir() || strings.HasPrefix(entry.Name(), ".") {
				continue
			}
			path := filepath.Join(root, entry.Name(), "SKILL.md")
			b, err := os.ReadFile(path)
			if err != nil {
				continue
			}
			name, desc := parseFrontmatter(string(b), entry.Name())
			if seen[name] {
				continue
			}
			seen[name] = true
			out = append(out, Skill{Name: name, Description: desc, Path: path, Scope: scope(root, workingDir), Content: string(b)})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}
func parseFrontmatter(content, fallback string) (string, string) {
	name := fallback
	desc := ""
	trim := strings.TrimPrefix(content, "\ufeff")
	if !strings.HasPrefix(trim, "---\n") {
		return name, desc
	}
	rest := strings.TrimPrefix(trim, "---\n")
	head, _, ok := strings.Cut(rest, "\n---")
	if !ok {
		return name, desc
	}
	for _, line := range strings.Split(head, "\n") {
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		v = strings.Trim(strings.TrimSpace(v), "'\"")
		switch strings.TrimSpace(k) {
		case "name":
			if v != "" {
				name = v
			}
		case "description":
			desc = v
		}
	}
	return name, desc
}
func scope(root, workingDir string) string {
	if workingDir != "" && filepath.Clean(root) == filepath.Clean(filepath.Join(workingDir, ".agents", "skills")) {
		return "workspace"
	}
	if home, err := os.UserHomeDir(); err == nil && strings.HasPrefix(filepath.Clean(root), filepath.Clean(home)) {
		return "global"
	}
	return "local"
}
func relevance(query, doc string) int {
	doc = strings.ToLower(doc)
	score := 0
	for _, tok := range strings.FieldsFunc(query, func(r rune) bool { return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '-') }) {
		if len(tok) >= 4 && strings.Contains(doc, tok) {
			score++
		}
	}
	return score
}
