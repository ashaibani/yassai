package localllm

import (
	"strings"
	"testing"
)

func TestAnswerGroundedIn(t *testing.T) {
	cases := []struct {
		name    string
		answer  string
		stdout  string
		grounded bool
	}{
		{"exact int", "1672 units", "1672", true},
		{"hallucinated int", "1344 units", "1672", false},
		{"comma formatting", "$173,509.31", "proj=$173509.31", true},
		{"round half up", "3.43 hours", "3.4285714285714284", true},
		{"round to 1dp rate", "Feb -2.5%, Mar 13.5%", "growth=Feb -2.5284%, Mar 13.5018%", true},
		{"invented rate", "Mar 99.9%", "growth=Mar 13.5018%", false},
		{"time tokens", "They meet at 11:04:30, 276.75 km from City A.", "11:04:30\n276.75 km from City A", true},
		{"ignores output", "The answer is 42", "1672", false},
		{"logic echo", "Alice: water; Bob: juice; Carol: tea; Dave: coffee.", "Alice: water; Bob: juice; Carol: tea; Dave: coffee", true},
		{"logic mismatch", "Everyone drinks tea.", "Alice: water; Bob: juice", false},
		{"part labels ok", "1. Average: $153,633.33. 2. Growth: Feb -2.5%.", "average=$153633.33\ngrowth=Feb -2.5%", true},
	}
	for _, c := range cases {
		reason := answerGroundedIn(c.answer, c.stdout)
		if (reason == "") != c.grounded {
			t.Errorf("%s: grounded=%v want %v (reason=%q)", c.name, reason == "", c.grounded, reason)
		}
	}
}

func TestAnswerPlausible(t *testing.T) {
	cases := []struct {
		name   string
		prompt string
		answer string
		ok     bool
	}{
		{"negative units", "How many units remain at the end of Q3?", "-3530 units", false},
		{"positive units", "How many units remain at the end of Q3?", "1470 units", true},
		{"missing cost", "How much cocoa is needed for 26 brownies? If cocoa costs $3.20 per cup, what is the total cost of cocoa for 26 brownies?", "1.625 cups of cocoa", false},
		{"full cost answer", "How much cocoa is needed? If cocoa costs $3.20 per cup, what is the total cost?", "1.625 cups of cocoa; total cost $5.20.", true},
		{"negative growth fine", "Which months saw a decline? Project July using growth rates.", "Declines: Feb, May. Growth Feb -2.5%. Projected July: $173,509.31.", true},
	}
	for _, c := range cases {
		reason := answerPlausible(c.prompt, c.answer)
		if (reason == "") != c.ok {
			t.Errorf("%s: ok=%v want %v (reason=%q)", c.name, reason == "", c.ok, reason)
		}
	}
}

func TestMeetingTimeBound(t *testing.T) {
	prompt := "A train leaves City A at 08:00 travelling toward City B at 90 km/h. A second train leaves City B at 09:30 travelling toward City A at 110 km/h. The distance between the cities is 450 km. At what time do the trains meet, and how far from City A is the meeting point?"
	if r := meetingTimeBound(prompt, "They meet at 15:30, 276.75 km from City A."); r == "" {
		t.Error("15:30 should be outside physical bounds")
	}
	if r := meetingTimeBound(prompt, "They meet at 11:04:30, 276.75 km from City A."); r != "" {
		t.Errorf("11:04:30 should pass: %s", r)
	}
	if r := meetingTimeBound(prompt, "They meet at 09:15, 276.75 km from City A."); r == "" {
		t.Error("09:15 (before later departure) should fail")
	}
	if r := meetingTimeBound("What is 2+2?", "4"); r != "" {
		t.Errorf("non-train prompt must skip the check: %s", r)
	}
}

func TestMagnitudeBound(t *testing.T) {
	prompt := "Monthly revenue figures: Jul $204,000 | Aug $197,800 | Sep $215,600. Projected January revenue?"
	if r := magnitudeBound(prompt, "Projected January revenue: $2,380.50."); r == "" {
		t.Error("100x-too-small projection should be rejected")
	}
	if r := magnitudeBound(prompt, "Projected January revenue: $238,049.65."); r != "" {
		t.Errorf("correct-scale projection should pass: %s", r)
	}
	if r := magnitudeBound("A depot starts with 3,200 units. How many remain?", "1470 units"); r != "" {
		t.Errorf("small-figure prompts skip the check: %s", r)
	}
}

func TestReRoundingRejected(t *testing.T) {
	if r := answerGroundedIn("1.62 cups of cocoa; total cost $5.20.", "1.625 cups\n$5.20"); r == "" {
		t.Error("re-rounding an already-formatted stdout value must be rejected")
	}
	if r := answerGroundedIn("1.625 cups of cocoa; total cost $5.20.", "1.625 cups\n$5.20"); r != "" {
		t.Errorf("exact match must pass: %s", r)
	}
	if r := answerGroundedIn("3.43 hours", "3.4285714285714284"); r != "" {
		t.Errorf("rounding a raw float tail must pass: %s", r)
	}
}

func TestExtractToolCodeLenient(t *testing.T) {
	full := `<tool_call>{"name":"run_python","arguments":{"code":"print(1)"}}</tool_call>`
	if c, e := extractToolCode(full); e != "" || c != "print(1)" {
		t.Errorf("strict parse failed: %q %q", c, e)
	}
	mangled := `<tool_call>{"name":"run_python","arguments":{"code":"start = 2400\nprint(start)"}}<|im_end|>`
	if c, e := extractToolCode(mangled); e != "" || !strings.Contains(c, "start = 2400") {
		t.Errorf("lenient parse failed: %q %q", c, e)
	}
	if _, e := extractToolCode("no call here"); e == "" {
		t.Error("missing tool call must error")
	}
	truncated := `<tool_call>{"name":"run_python","arguments":{"code":"print(`
	if _, e := extractToolCode(truncated); e == "" {
		t.Error("truncated JSON must error")
	}
}

func TestMeetingDistanceConsistency(t *testing.T) {
	prompt := "A train leaves City A at 08:00 travelling toward City B at 90 km/h. A second train leaves City B at 09:30 travelling toward City A at 110 km/h. The distance between the cities is 450 km. At what time do the trains meet, and how far from City A is the meeting point?"
	if r := meetingTimeBound(prompt, "They meet at 11:04:30, 276.75 km from City A."); r != "" {
		t.Errorf("consistent time+distance must pass: %s", r)
	}
	if r := meetingTimeBound(prompt, "They meet at 11:04:30, 976.75 km from City A."); r == "" {
		t.Error("wildly wrong distance must be rejected")
	}
}

func TestUpperMagnitudeAndListEcho(t *testing.T) {
	if r := magnitudeBound("A depot starts with 3,200 units. How many remain?", "16,224 units"); r == "" {
		t.Error("answer 5x above prompt scale must be rejected")
	}
	if r := magnitudeBound("A depot starts with 3,200 units. How many remain?", "1,470 units"); r != "" {
		t.Errorf("in-scale answer must pass: %s", r)
	}
	if r := listEcho("results = 13 - late binding", "results = [13, 13, 13]"); r == "" {
		t.Error("collapsed list must be rejected")
	}
	if r := listEcho("results = [13, 13, 13] - late binding", "results = [13, 13, 13]"); r != "" {
		t.Errorf("echoed list must pass: %s", r)
	}
}
