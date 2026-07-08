package taskclf

import (
	"os"
	"path/filepath"
	"testing"
)

// TestClassifyIntegration exercises the bundled model end to end. It is skipped
// unless ONNXRUNTIME_LIB points at a libonnxruntime the host can load, so it is
// safe in CI without the native library.
func TestClassifyIntegration(t *testing.T) {
	lib := os.Getenv("ONNXRUNTIME_LIB")
	if lib == "" {
		t.Skip("set ONNXRUNTIME_LIB to run the ONNX integration test")
	}
	dir := filepath.Join("..", "..", "assets", "taskclf")
	if _, err := os.Stat(filepath.Join(dir, "model.int8.onnx")); err != nil {
		t.Skipf("model not present: %v", err)
	}

	clf, err := New(dir, "", lib)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer clf.Close()

	cases := []struct{ prompt, want string }{
		{"What is 17 * 23? Return only the number.", "mathematical_reasoning"},
		{"Write a Python function called is_palindrome that checks a string.", "code_generation"},
		{`Classify the sentiment of "I loved this excellent product". Return only the label.`, "sentiment_classification"},
		{`Extract named entities from: "Tim Cook announced Apple Inc. will open a London office in 2025."`, "named_entity_recognition"},
		{"If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies?", "logical_deductive_reasoning"},
		{"Debug this Python function: def add(a, b): return a - b. Return the corrected code.", "code_debugging"},
		{"Summarise the following in one sentence: the update adds security patches and faster load times.", "text_summarisation"},
	}
	for _, c := range cases {
		preds, err := clf.Classify(c.prompt)
		if err != nil {
			t.Fatalf("Classify(%.40q): %v", c.prompt, err)
		}
		got := map[string]bool{}
		for _, p := range preds {
			got[p.Label] = true
		}
		if !got[c.want] {
			t.Errorf("prompt %.50q: want %q, got %v", c.prompt, c.want, preds)
		}
	}
}
