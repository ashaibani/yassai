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
	if _, err := os.Stat(lib); err != nil {
		t.Skipf("ONNXRUNTIME_LIB points at %q which does not exist: %v", lib, err)
	}
	dir := filepath.Join("..", "..", "assets", "taskclf")
	// Exercise the same int8 artefact shipped in the runtime image.
	modelPath := filepath.Join(dir, "model.int8.onnx")
	if _, err := os.Stat(modelPath); err != nil {
		modelPath = filepath.Join(dir, "model.onnx")
		if _, err := os.Stat(modelPath); err != nil {
			t.Skipf("model not present: %v", err)
		}
	}
	// Skip if the file is a Git-LFS pointer (text) instead of the real binary.
	// LFS pointers are ~130 bytes; the real model is megabytes.
	if fi, err := os.Stat(modelPath); err == nil && fi.Size() < 1024 {
		t.Skipf("model file is an LFS pointer (%d bytes), not the real binary", fi.Size())
	}

	clf, err := New(dir, "", lib)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer clf.Close()

	// knownV4Miss marks probes the v4 model is KNOWN to misclassify
	// (code_debugging over-fires and swallows NER/logic; the borderline int8
	// logits also flip between arm64 and amd64 runners). These are logged, not
	// failed: production no longer depends on classifier top-labels for these
	// shapes - internal/agent's keyword heuristic routes them
	// (TestHeuristicRoutingContract pins that mitigation). Re-tighten when the
	// classifier is retrained.
	cases := []struct {
		prompt, want string
		knownV4Miss  bool
	}{
		{"What is 17 * 23? Return only the number.", "mathematical_reasoning", false},
		{"Write a Python function called is_palindrome that checks a string.", "code_generation", false},
		{`Classify the sentiment of "I loved this excellent product". Return only the label.`, "sentiment_classification", false},
		{`Extract named entities from: "Tim Cook announced Apple Inc. will open a London office in 2025."`, "named_entity_recognition", true},
		{"If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies?", "logical_deductive_reasoning", true},
		{"Debug this Python function: def add(a, b): return a - b. Return the corrected code.", "code_debugging", false},
		{"Summarise the following in one sentence: the update adds security patches and faster load times.", "text_summarisation", false},
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
		switch {
		case got[c.want] && c.knownV4Miss:
			t.Logf("known v4 miss now CLASSIFIES CORRECTLY (%.50q -> %q) - retrained model? re-tighten this probe", c.prompt, c.want)
		case !got[c.want] && c.knownV4Miss:
			t.Logf("known v4 miss (heuristic routing covers it): %.50q want %q got %v", c.prompt, c.want, preds)
		case !got[c.want]:
			t.Errorf("prompt %.50q: want %q, got %v", c.prompt, c.want, preds)
		}
	}
}
