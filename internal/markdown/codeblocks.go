package markdown

import "strings"

type Block struct {
	Info string
	Code string
}

func ExtractActionBlocks(text string) []Block {
	all := ExtractBlocks(text)
	var out []Block
	for _, block := range all {
		info := strings.ToLower(strings.TrimSpace(block.Info))
		first := firstLine(block.Code)
		// Accept bare code blocks (no info string) and common Python aliases
		// so the model can use whichever fence it naturally writes.
		if info == "" || strings.Contains(info, "micropy") || strings.Contains(info, "micropython") || strings.Contains(info, "python action") || strings.Contains(info, "python act") || strings.Contains(strings.ToLower(first), "# act") || info == "python" || info == "python3" || info == "py" || info == "py3" {
			out = append(out, block)
		}
	}
	return out
}

func ExtractBlocks(text string) []Block {
	var blocks []Block
	for {
		start := strings.Index(text, "```")
		if start < 0 {
			break
		}
		text = text[start+3:]
		lineEnd := strings.IndexByte(text, '\n')
		if lineEnd < 0 {
			break
		}
		info := strings.TrimSpace(text[:lineEnd])
		text = text[lineEnd+1:]
		end := strings.Index(text, "```")
		if end < 0 {
			break
		}
		blocks = append(blocks, Block{Info: info, Code: strings.TrimSpace(text[:end])})
		text = text[end+3:]
	}
	return blocks
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}
