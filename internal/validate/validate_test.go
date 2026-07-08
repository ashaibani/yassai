package validate_test

import (
	"testing"

	"github.com/ashaibani/yassai/internal/validate"
)

func TestNumeric(t *testing.T) {
	cases := []struct {
		actual   string
		expected string
		pass     bool
	}{
		{"391", "391", true},
		{"The answer is 391", "391", true},
		{"390", "391", false},
		{"fifty", "50", false},
		{"100", "100", true},
	}
	for _, c := range cases {
		res := validate.Check(c.actual, validate.Case{TaskID: "m1", Expected: c.expected, Validate: "numeric"})
		if res.Pass != c.pass {
			t.Errorf("numeric(%q, %q) = %v, want %v: %s", c.actual, c.expected, res.Pass, c.pass, res.Reason)
		}
	}
}

func TestContainsCI(t *testing.T) {
	cases := []struct {
		actual   string
		expected string
		pass     bool
	}{
		{"positive", "positive", true},
		{"POSITIVE", "positive", true},
		{"The sentiment is positive.", "positive", true},
		{"neutral", "positive", false},
	}
	for _, c := range cases {
		res := validate.Check(c.actual, validate.Case{TaskID: "s1", Expected: c.expected, Validate: "contains_ci"})
		if res.Pass != c.pass {
			t.Errorf("contains_ci(%q, %q) = %v, want %v: %s", c.actual, c.expected, res.Pass, c.pass, res.Reason)
		}
	}
}

func TestNER(t *testing.T) {
	res := validate.Check(
		`{"entities":[{"text":"Tim Cook","type":"PERSON"},{"text":"Apple Inc.","type":"ORG"},{"text":"London","type":"LOC"},{"text":"January 2025","type":"DATE"}]}`,
		validate.Case{TaskID: "n1", Expected: "Tim Cook:person,Apple:organisation,London:location,January 2025:date", Validate: "ner"},
	)
	if !res.Pass {
		t.Errorf("NER check failed: %s", res.Reason)
	}
}

func TestCodeCheck(t *testing.T) {
	res := validate.Check(
		"def is_palindrome(s):\n  s = s.lower().replace(' ', '')\n  return s == s[::-1]",
		validate.Case{TaskID: "cg1", Expected: "is_palindrome", Validate: "code_check"},
	)
	if !res.Pass {
		t.Errorf("code_check failed: %s", res.Reason)
	}
}

func TestScore(t *testing.T) {
	results := []validate.Result{
		{TaskID: "m1", Pass: true, Category: "maths"},
		{TaskID: "m2", Pass: false, Category: "maths"},
		{TaskID: "s1", Pass: true, Category: "sentiment"},
	}
	report := validate.Score(results)
	if report.Total != 3 || report.Passed != 2 {
		t.Fatalf("score wrong: %+v", report)
	}
	if report.ByCategory["maths"].PassRate != 0.5 {
		t.Errorf("maths pass rate wrong: %v", report.ByCategory["maths"])
	}
}
