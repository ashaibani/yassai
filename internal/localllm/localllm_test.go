package localllm

import (
	"context"
	"os/exec"
	"strings"
	"testing"
)

func TestAnswerGroundedIn(t *testing.T) {
	cases := []struct {
		name     string
		answer   string
		stdout   string
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

func TestExtractToolCodeFence(t *testing.T) {
	fenced := "```python\nstart = 2400\nprint(start)\n```"
	if c, e := extractToolCode(fenced); e != "" || !strings.Contains(c, "start = 2400") {
		t.Errorf("fence parse failed: %q %q", c, e)
	}
	bare := "```\nprint(1)\n```"
	if c, e := extractToolCode(bare); e != "" || c != "print(1)" {
		t.Errorf("bare fence parse failed: %q %q", c, e)
	}
	// fenced code containing braces and quotes must survive untouched
	tricky := "```python\nprint('; '.join(f'{p}: {d[p]}' for p in people))\n```"
	if c, e := extractToolCode(tricky); e != "" || !strings.Contains(c, "{d[p]}") {
		t.Errorf("tricky fence parse failed: %q %q", c, e)
	}
	if _, e := extractToolCode("```python\nprint(1)"); e == "" {
		t.Error("unterminated fence must error")
	}
}

func TestExampleCheckGate(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}
	prompt := "Write a Python function called interval_intersection that takes two lists. " +
		"For example, interval_intersection([[1,4],[7,10]], [[3,8]]) should return [[3,4],[7,8]]. " +
		"Handle empty inputs and touching endpoints."
	good := "def interval_intersection(a, b):\n" +
		"    out, i, j = [], 0, 0\n" +
		"    while i < len(a) and j < len(b):\n" +
		"        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])\n" +
		"        if lo <= hi:\n            out.append([lo, hi])\n" +
		"        if a[i][1] < b[j][1]:\n            i += 1\n" +
		"        else:\n            j += 1\n" +
		"    return out\n"
	if reason := exampleCheck(context.Background(), prompt, good); reason != "" {
		t.Errorf("correct code must pass the example gate, got %q", reason)
	}
	bad := strings.Replace(good, "out.append([lo, hi])", "out.append([lo, hi - 1])", 1)
	if reason := exampleCheck(context.Background(), prompt, bad); reason == "" {
		t.Error("logic-bug code must fail the example gate")
	}
	// Prompts without an executable worked example must REJECT to remote:
	// parse-only gates pass wrong code (observed: a second-largest function
	// returning the minimum), so free-form specs are not local-safe.
	if reason := exampleCheck(context.Background(), "Write a Python function called flatten that flattens.", good); reason == "" {
		t.Error("no-example prompt must reject to remote")
	}
}

func TestGateNERSentenceStartTail(t *testing.T) {
	prompt := "Extract all named entities. On March 15, 2025, Satya Nadella visited Seattle for Microsoft."
	full := "PERSON: Satya Nadella; LOCATION: Seattle; ORGANIZATION: Microsoft; DATE: March 15, 2025"
	if reason := gateNER(prompt, full); reason != "" {
		t.Errorf("complete answer must pass, got %q", reason)
	}
	if reason := gateNER(prompt, strings.ReplaceAll(full, "Seattle", "")); reason == "" {
		t.Error("omitted mid-sentence entity must be rejected")
	}
	if reason := gateNER(prompt, strings.ReplaceAll(full, "2025", "2024")); reason == "" {
		t.Error("omitted year must be rejected")
	}
}

func TestGateNERAcronyms(t *testing.T) {
	prompt := "Extract entities and label each as PERSON, ORGANIZATION, LOCATION, or DATE. " +
		"Dr Chen joined ETH Zurich after leaving NASA."
	if reason := gateNER(prompt, "PERSON: Dr Chen; ORGANIZATION: ETH Zurich, NASA"); reason != "" {
		t.Errorf("complete answer must pass, got %q", reason)
	}
	if reason := gateNER(prompt, "PERSON: Dr Chen; ORGANIZATION: NASA; LOCATION: Zurich"); reason == "" {
		t.Error("omitted all-caps acronym (ETH) must be rejected")
	}
}

func TestGateNERLabelSanity(t *testing.T) {
	if r := gateNERLabels("ORGANIZATION: ETH Zurich; LOCATION: Zurich"); r != "" {
		t.Errorf("correct labels must pass, got %q", r)
	}
	if r := gateNERLabels("LOCATION: ETH Zurich; PERSON: Sundar Pichai"); r == "" {
		t.Error("acronym-led span labelled LOCATION must be rejected")
	}
}

func TestGroundCodeFixCauseRewritesProvenException(t *testing.T) {
	prompt := "This function should check whether a word reads the same forwards and backwards, " +
		"but it doesn't work. State the cause and provide the fix.\n\n" +
		"def is_mirror(word):\n    return word == word.reverse()"
	answer := "word.reverse() creates a reversed copy of the string, so the comparison is subtly wrong.\n\n" +
		"def is_mirror(word):\n    return word == word[::-1]"
	got := groundCodeFixCause(context.Background(), prompt, answer)
	if !strings.Contains(got, "AttributeError") {
		t.Fatalf("proven exception must replace the confabulated cause, got %q", got)
	}
	if !strings.Contains(got, "def is_mirror(word):\n    return word == word[::-1]") {
		t.Fatalf("the model's corrected function must be preserved verbatim, got %q", got)
	}
}

func TestGroundCodeFixCauseKeepsWrongOutputBugs(t *testing.T) {
	prompt := "This function should add two numbers but returns the wrong value. Fix it.\n\n" +
		"def add(a, b):\n    return a - b"
	answer := "The buggy line subtracts b instead of adding it.\n\ndef add(a, b):\n    return a + b"
	if got := groundCodeFixCause(context.Background(), prompt, answer); got != answer {
		t.Fatalf("wrong-output bugs must keep the model's cause line, got %q", got)
	}
}

func TestGroundCodeFixCauseSkipsUnattributable(t *testing.T) {
	// Both versions raise on the synthesized input: grounding must not fire.
	prompt := "This function should return the first item but crashes. Fix it.\n\n" +
		"def first_item(items):\n    return items[len(items)]"
	answer := "The index is out of range by one.\n\ndef first_item(items):\n    return items[missing_helper()]"
	if got := groundCodeFixCause(context.Background(), prompt, answer); got != answer {
		t.Fatalf("both-raise must be left untouched, got %q", got)
	}
}

func TestTrimToWordCap(t *testing.T) {
	prompt := "Summarise in exactly three bullet points, each no longer than 15 words."
	answer := "- one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen\n" +
		"- short bullet stays exactly as written\n" +
		"prose line is not a bullet and stays"
	got := trimToWordCap(prompt, answer)
	lines := strings.Split(got, "\n")
	if n := len(strings.Fields(strings.TrimPrefix(lines[0], "- "))); n != 15 {
		t.Fatalf("over-cap bullet must be trimmed to 15 words, got %d: %q", n, lines[0])
	}
	if lines[1] != "- short bullet stays exactly as written" {
		t.Fatalf("under-cap bullet must be untouched: %q", lines[1])
	}
	if lines[2] != "prose line is not a bullet and stays" {
		t.Fatalf("non-bullet line must be untouched: %q", lines[2])
	}
	if trimToWordCap("no cap stated here", answer) != answer {
		t.Fatal("prompts without a word cap must pass through unchanged")
	}
}

func TestExampleCheckShapeStrict(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}
	prompt := "Write a Python function called merge_intervals that merges overlapping intervals. " +
		"For example, merge_intervals([[1,3],[2,6],[8,10]]) should return [[1,6],[8,10]]."
	tuples := "def merge_intervals(intervals):\n" +
		"    intervals = sorted(intervals)\n" +
		"    out = []\n" +
		"    for s, e in intervals:\n" +
		"        if out and s <= out[-1][1]:\n" +
		"            out[-1] = (out[-1][0], max(out[-1][1], e))\n" +
		"        else:\n" +
		"            out.append((s, e))\n" +
		"    return out\n"
	reason := exampleCheck(context.Background(), prompt, tuples)
	if reason == "" {
		t.Fatal("tuples-for-lists must be rejected: the judge is shape-strict (T09)")
	}
	if !strings.Contains(reason, "got") || !strings.Contains(reason, "want") {
		t.Fatalf("reject reason must show the shape mismatch for the retry hint, got %q", reason)
	}
	lists := strings.NewReplacer("(out[-1][0], max(out[-1][1], e))", "[out[-1][0], max(out[-1][1], e)]",
		"out.append((s, e))", "out.append([s, e])").Replace(tuples)
	if reason := exampleCheck(context.Background(), prompt, lists); reason != "" {
		t.Fatalf("list-shaped correct code must pass, got %q", reason)
	}
}
