package textimg

import (
	"bytes"
	"image/png"
	"strings"
	"testing"
)

func TestRenderTextProducesDecodablePagedPNG(t *testing.T) {
	results, err := RenderText(strings.Repeat("abcdef ghijkl mnopqr\n", 20), RenderConfig{MaxWidth: 128, MaxHeight: 64, Scale: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) < 2 {
		t.Fatalf("expected multiple pages, got %d", len(results))
	}
	for i, result := range results {
		img, err := png.Decode(bytes.NewReader(result.PNG))
		if err != nil {
			t.Fatalf("page %d is not a PNG: %v", i, err)
		}
		if img.Bounds().Dx() != result.W || img.Bounds().Dy() != result.H {
			t.Fatalf("page %d dimensions metadata=%dx%d decoded=%dx%d", i, result.W, result.H, img.Bounds().Dx(), img.Bounds().Dy())
		}
		if !strings.HasPrefix(result.ToBase64DataURI(), "data:image/png;base64,") {
			t.Fatalf("page %d data URI prefix", i)
		}
	}
}

func TestRenderTextEmpty(t *testing.T) {
	results, err := RenderText("", RenderConfig{MaxWidth: 128, MaxHeight: 64, Scale: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Chars != 0 {
		t.Fatalf("empty render: %#v", results)
	}
}
