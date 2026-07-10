package agent

import (
	"strings"
	"testing"
)

func TestTextImgAutoFallsBackForShortQueries(t *testing.T) {
	task := Task{TaskID: "s1", Prompt: "Summarize exactly two sentences:\n\n'A short passage.'"}
	categories := map[string][]string{"s1": {"text_summarisation"}}
	ag := &Agent{cfg: Config{TextImg: "auto"}, categories: categories}
	messages := ag.buildBatchMessages([]Task{task}, false, false)
	if len(messages) != 2 || len(messages[1].ImageURLs) != 0 {
		t.Fatalf("short query should stay text, got %#v", messages)
	}
	if !strings.Contains(messages[1].Content, "A short passage") {
		t.Fatalf("short source disappeared from text: %s", messages[1].Content)
	}
}

func TestTextImgAutoRendersLongSourceWithWireLabel(t *testing.T) {
	source := strings.Repeat("Machine learning can support clinicians while privacy and bias still require oversight. ", 32)
	task := Task{TaskID: "s1", Prompt: "Summarize the following passage in exactly two sentences:\n\n'" + source + "'"}
	categories := map[string][]string{"s1": {"text_summarisation"}}
	ag := &Agent{cfg: Config{TextImg: "auto"}, categories: categories}
	messages := ag.buildBatchMessages([]Task{task}, false, false)
	if len(messages) != 2 || len(messages[1].ImageURLs) == 0 {
		t.Fatalf("long source should render as image, got %#v", messages)
	}
	if len(messages[1].ImageLabels) != len(messages[1].ImageURLs) || messages[1].ImageLabels[0] != "IMAGE:s1 source text" {
		t.Fatalf("image labels: %#v", messages[1].ImageLabels)
	}
	if !strings.Contains(messages[1].Content, "exactly two sentences") || !strings.Contains(messages[1].Content, "[IMAGE:s1]") {
		t.Fatalf("instruction/reference missing: %s", messages[1].Content)
	}
	if strings.Contains(messages[1].Content, source[:200]) {
		t.Fatal("long source should not remain duplicated in text")
	}
	if !strings.HasPrefix(messages[1].ImageURLs[0], "data:image/png;base64,") {
		t.Fatalf("unexpected image URL: %.32s", messages[1].ImageURLs[0])
	}
}

func TestTextImgNeverRendersCodeExecBatch(t *testing.T) {
	task := Task{TaskID: "m1", Prompt: strings.Repeat("Calculate 17 * 23. ", 200)}
	categories := map[string][]string{"m1": {"mathematical_reasoning"}}
	ag := &Agent{cfg: Config{TextImg: "full"}, categories: categories}
	messages := ag.buildBatchMessages([]Task{task}, true, false)
	if len(messages) != 2 || len(messages[1].ImageURLs) != 0 {
		t.Fatalf("code-exec batch must stay text, got %#v", messages)
	}
}

func TestSplitQuotedSource(t *testing.T) {
	instruction, source, ok := splitQuotedSource("Summarize exactly two sentences:\n\n'alpha beta' Keep the title.", "text_summarisation")
	if !ok || instruction != "Summarize exactly two sentences Keep the title." || source != "alpha beta" {
		t.Fatalf("got instruction=%q source=%q ok=%v", instruction, source, ok)
	}
	if _, _, ok := splitQuotedSource("Calculate '2+2'", "mathematical_reasoning"); ok {
		t.Fatal("math prompts must not be split into images")
	}
}

func TestTextImgASCIIFoldsAndMarksUnsupportedRunes(t *testing.T) {
	got := textImgASCII("smart — quotes “ok” café\tend")
	if got != `smart - quotes "ok" caf? end` {
		t.Fatalf("ASCII fold: %q", got)
	}
}

func TestTextImgManifestPreservesTypesAndIDs(t *testing.T) {
	batch := []Task{{TaskID: "f1"}, {TaskID: "s1"}, {TaskID: "sum1"}}
	categories := map[string][]string{
		"f1":   {"factual_knowledge"},
		"s1":   {"sentiment_classification"},
		"sum1": {"text_summarisation"},
	}
	got := textImgManifest(batch, categories)
	for _, want := range []string{"FACTUAL=[f1]", "SENTIMENT=[s1]", "SUMMARY=[sum1]"} {
		if !strings.Contains(got, want) {
			t.Fatalf("manifest missing %q: %s", want, got)
		}
	}
}
