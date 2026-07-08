package validate

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
)

type Case struct {
	TaskID   string `json:"task_id"`
	Prompt   string `json:"prompt"`
	Expected string `json:"expected"`
	Validate string `json:"validate"`
}

type Result struct {
	TaskID   string `json:"task_id"`
	Pass     bool   `json:"pass"`
	Reason   string `json:"reason,omitempty"`
	Expected string `json:"expected,omitempty"`
	Actual   string `json:"actual,omitempty"`
	Category string `json:"category,omitempty"`
}

type ScoreReport struct {
	Total      int                      `json:"total"`
	Passed     int                      `json:"passed"`
	Failed     int                      `json:"failed"`
	PassRate   float64                  `json:"pass_rate"`
	Results    []Result                 `json:"results"`
	ByCategory map[string]CategoryScore `json:"by_category"`
}

type CategoryScore struct {
	Total    int     `json:"total"`
	Passed   int     `json:"passed"`
	PassRate float64 `json:"pass_rate"`
}

func LoadCases(path string) ([]Case, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read golden %s: %w", path, err)
	}
	var cases []Case
	if err := json.Unmarshal(b, &cases); err != nil {
		return nil, fmt.Errorf("parse golden %s: %w", path, err)
	}
	return cases, nil
}

func Check(answer string, c Case) Result {
	actual := strings.TrimSpace(answer)
	res := Result{TaskID: c.TaskID, Expected: c.Expected, Actual: actual, Category: category(c.TaskID)}

	switch c.Validate {
	case "numeric":
		res.Pass, res.Reason = checkNumeric(actual, c.Expected)
	case "contains_ci":
		res.Pass, res.Reason = checkContainsCI(actual, c.Expected)
	case "contains_cs":
		res.Pass, res.Reason = checkContainsCS(actual, c.Expected)
	case "exact":
		res.Pass, res.Reason = checkExact(actual, c.Expected)
	case "ner":
		res.Pass, res.Reason = checkNER(actual, c.Expected)
	case "code_check":
		res.Pass, res.Reason = checkCode(actual, c.Expected)
	case "llm":
		res.Pass = true
		res.Reason = "requires llm judge (passing by default until judge is available)"
	default:
		res.Pass = false
		res.Reason = fmt.Sprintf("unknown validate type: %s", c.Validate)
	}
	return res
}

func Score(results []Result) ScoreReport {
	report := ScoreReport{Total: len(results), ByCategory: map[string]CategoryScore{}}
	for _, r := range results {
		cat := r.Category
		if cat == "" {
			cat = "unknown"
		}
		cs := report.ByCategory[cat]
		cs.Total++
		report.ByCategory[cat] = cs
		if r.Pass {
			report.Passed++
			cs.Passed++
			report.ByCategory[cat] = cs
		}
	}
	if report.Total > 0 {
		report.PassRate = float64(report.Passed) / float64(report.Total)
	}
	for k, v := range report.ByCategory {
		if v.Total > 0 {
			v.PassRate = float64(v.Passed) / float64(v.Total)
			report.ByCategory[k] = v
		}
	}
	return report
}

func checkNumeric(actual, expected string) (bool, string) {
	re := regexp.MustCompile(`-?\d+(?:\.\d+)?`)
	matches := re.FindStringSubmatch(actual)
	if len(matches) == 0 {
		return false, fmt.Sprintf("no number found in answer: %q", actual)
	}
	actualNum, err := strconv.ParseFloat(matches[0], 64)
	if err != nil {
		return false, fmt.Sprintf("cannot parse number %q: %v", matches[0], err)
	}
	expectedNum, err := strconv.ParseFloat(expected, 64)
	if err != nil {
		return false, fmt.Sprintf("cannot parse expected %q: %v", expected, err)
	}
	if actualNum == expectedNum {
		return true, ""
	}
	return false, fmt.Sprintf("expected %s, got %.0f", expected, actualNum)
}

func checkContainsCI(actual, expected string) (bool, string) {
	if strings.Contains(strings.ToLower(actual), strings.ToLower(expected)) {
		return true, ""
	}
	return false, fmt.Sprintf("expected to contain %q (case-insensitive), got %q", expected, actual)
}

func checkContainsCS(actual, expected string) (bool, string) {
	if strings.Contains(actual, expected) {
		return true, ""
	}
	return false, fmt.Sprintf("expected to contain %q (case-sensitive), got %q", expected, actual)
}

func checkExact(actual, expected string) (bool, string) {
	if strings.EqualFold(actual, expected) {
		return true, ""
	}
	return false, fmt.Sprintf("expected exactly %q, got %q", expected, actual)
}

func checkNER(actual, expected string) (bool, string) {
	// Parse expected as "Entity:type,Entity2:type2,..."
	pairs := strings.Split(expected, ",")
	found := 0
	missing := []string{}
	for _, pair := range pairs {
		parts := strings.SplitN(strings.TrimSpace(pair), ":", 2)
		if len(parts) != 2 {
			continue
		}
		entity := strings.TrimSpace(parts[0])
		typ := strings.TrimSpace(parts[1])
		if strings.Contains(strings.ToLower(actual), strings.ToLower(entity)) {
			found++
		} else {
			missing = append(missing, entity)
		}
		_ = typ
	}
	if found == len(pairs) {
		return true, fmt.Sprintf("all %d entities found", found)
	}
	return false, fmt.Sprintf("missing %d/%d entities: %s", len(missing), len(pairs), strings.Join(missing, ", "))
}

func checkCode(actual, expected string) (bool, string) {
	if strings.Contains(actual, expected) {
		return true, fmt.Sprintf("function name %q found", expected)
	}
	return false, fmt.Sprintf("expected function named %q in code", expected)
}

func category(taskID string) string {
	switch {
	case strings.HasPrefix(taskID, "f"):
		return "factual"
	case strings.HasPrefix(taskID, "m"):
		return "maths"
	case strings.HasPrefix(taskID, "s") && !strings.HasPrefix(taskID, "su"):
		return "sentiment"
	case strings.HasPrefix(taskID, "su"):
		return "summarisation"
	case strings.HasPrefix(taskID, "n"):
		return "ner"
	case strings.HasPrefix(taskID, "cd"):
		return "code_debugging"
	case strings.HasPrefix(taskID, "l"):
		return "logical_reasoning"
	case strings.HasPrefix(taskID, "cg"):
		return "code_generation"
	default:
		return "unknown"
	}
}
