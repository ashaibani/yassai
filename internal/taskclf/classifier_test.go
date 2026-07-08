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
	// Prefer the FP32 model (model.onnx); fall back to int8.
	modelPath := filepath.Join(dir, "model.onnx")
	if _, err := os.Stat(modelPath); err != nil {
		modelPath = filepath.Join(dir, "model.int8.onnx")
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
