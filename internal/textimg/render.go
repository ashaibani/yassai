package textimg

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/color"
	"image/png"
	"strings"
)

const (
	cellW = 6 // 5px glyph + 1px spacing
	cellH = 8 // 7px glyph + 1px spacing
	padX  = 4
	padY  = 4
)

// RenderConfig controls text-to-image rendering.
type RenderConfig struct {
	// MaxWidth in pixels (default 768 for OpenAI/Fireworks safe zone)
	MaxWidth int
	// MaxHeight in pixels (default 728 matching pxpipe's Anthropic page size)
	MaxHeight int
	// Scale factor for each pixel (1=native 5x7, 2=10x14, etc.)
	Scale int
}

// DefaultRenderConfig returns production-safe defaults.
func DefaultRenderConfig() RenderConfig {
	return RenderConfig{
		MaxWidth:  768,
		MaxHeight: 728,
		Scale:     2,
	}
}

// RenderResult holds a rendered PNG and its metadata.
type RenderResult struct {
	PNG   []byte
	W     int
	H     int
	Chars int
}

// RenderText renders text into one or more PNG images.
// Text is wrapped to fit within MaxWidth, and split into multiple images
// if it exceeds MaxHeight.
func RenderText(text string, cfg RenderConfig) ([]RenderResult, error) {
	if cfg.MaxWidth <= 0 {
		cfg = DefaultRenderConfig()
	}
	if cfg.Scale <= 0 {
		cfg.Scale = 1
	}

	scale := cfg.Scale
	// Calculate columns that fit in MaxWidth
	usableW := cfg.MaxWidth - 2*padX*scale
	cols := usableW / (cellW * scale)
	if cols < 20 {
		cols = 20
	}

	// Word-wrap text to cols
	lines := wrapText(text, cols)

	// Calculate rows per page
	usableH := cfg.MaxHeight - 2*padY*scale
	rowsPerPage := usableH / (cellH * scale)
	if rowsPerPage < 1 {
		rowsPerPage = 1
	}

	var results []RenderResult
	totalChars := 0

	for startLine := 0; startLine < len(lines); startLine += rowsPerPage {
		endLine := startLine + rowsPerPage
		if endLine > len(lines) {
			endLine = len(lines)
		}
		pageLines := lines[startLine:endLine]

		// Calculate actual image dimensions
		imgW := padX*scale*2 + cols*cellW*scale
		if imgW > cfg.MaxWidth {
			imgW = cfg.MaxWidth
		}
		imgH := padY*scale*2 + len(pageLines)*cellH*scale

		img := image.NewRGBA(image.Rect(0, 0, imgW, imgH))

		// Fill background white
		for y := 0; y < imgH; y++ {
			for x := 0; x < imgW; x++ {
				img.Set(x, y, color.RGBA{R: 255, G: 255, B: 255, A: 255})
			}
		}

		// Draw each line
		for lineIdx, line := range pageLines {
			yBase := padY*scale + lineIdx*cellH*scale
			for col, ch := range []byte(line) {
				if col >= cols {
					break
				}
				glyph, ok := fontAtlas[ch]
				if !ok {
					// Keep unsupported bytes visible rather than silently punching
					// holes in the rendered text. Production input is UTF-8 folded
					// before this point, but RenderText is also a public package API.
					glyph = fontAtlas['?']
				}
				for py := 0; py < 7; py++ {
					for px := 0; px < 5; px++ {
						if glyph[py]&(1<<(4-px)) != 0 {
							for sy := 0; sy < scale; sy++ {
								for sx := 0; sx < scale; sx++ {
									px2 := padX*scale + col*cellW*scale + px*scale + sx
									py2 := yBase + py*scale + sy
									if px2 < imgW && py2 < imgH {
										img.Set(px2, py2, color.RGBA{R: 0, G: 0, B: 0, A: 255})
									}
								}
							}
						}
					}
				}
				totalChars++
			}
		}

		var buf bytes.Buffer
		if err := png.Encode(&buf, img); err != nil {
			return nil, err
		}
		results = append(results, RenderResult{
			PNG:   buf.Bytes(),
			W:     imgW,
			H:     imgH,
			Chars: len(strings.Join(pageLines, "")),
		})
	}

	if len(results) == 0 {
		// Empty text: produce a 1x1 white pixel
		img := image.NewRGBA(image.Rect(0, 0, 1, 1))
		img.Set(0, 0, color.RGBA{R: 255, G: 255, B: 255, A: 255})
		var buf bytes.Buffer
		png.Encode(&buf, img)
		results = append(results, RenderResult{PNG: buf.Bytes(), W: 1, H: 1, Chars: 0})
	}

	return results, nil
}

// ToBase64DataURI converts a PNG to a data URI.
func (r RenderResult) ToBase64DataURI() string {
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(r.PNG)
}

// wrapText breaks text into lines no longer than cols characters.
// Respects existing newlines and wraps long lines at word boundaries.
func wrapText(text string, cols int) []string {
	if cols <= 0 {
		cols = 80
	}
	var result []string
	for _, paragraph := range strings.Split(text, "\n") {
		if len(paragraph) <= cols {
			result = append(result, paragraph)
			continue
		}
		// Word wrap
		words := strings.Fields(paragraph)
		if len(words) == 0 {
			result = append(result, "")
			continue
		}
		line := words[0]
		for _, w := range words[1:] {
			if len(line)+1+len(w) <= cols {
				line += " " + w
			} else {
				result = append(result, line)
				line = w
			}
		}
		result = append(result, line)
	}
	if len(result) == 0 {
		result = append(result, "")
	}
	return result
}
