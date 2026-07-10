package agent

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ashaibani/yassai/internal/llm"
	"github.com/ashaibani/yassai/internal/textimg"
)

// TextImg mode: Fireworks vision tokenisation charges by pixel area
// (~1 token per ~32x32px at 768 wide), so a scale=1 bitmap-font render packs
// ~9.5 chars/token vs ~3.6 chars/token for BPE text - measured 52-69% prompt
// savings beyond ~2,000 chars (internal/textimg live tests). Only non-code
// batches are rendered: maths/logic answers come from code interpolating
// digits read out of the prompt, and OCR misreads there fail the accuracy gate.

func (a *Agent) textImgMode() string {
	switch strings.TrimSpace(strings.ToLower(a.cfg.TextImg)) {
	case "hybrid", "contexts", "auto", "on", "1", "true":
		return "hybrid"
	case "tasks", "grouped":
		return "tasks"
	case "dense":
		return "dense"
	case "full":
		return "full"
	default:
		return ""
	}
}

// textImgMinChars: below this the per-image overhead (~60-100 tokens) eats the
// saving, so small batches stay text.
const textImgMinChars = 900

// Short snippets cost more as separate vision inputs than as BPE text. Keep
// them inline and reserve image tokenisation for passage-sized payloads.
const textImgSourceMinChars = 2000

// buildBatchMessages assembles the system+user opening messages for a batch,
// rendering the task sheet (and in "full" mode the category recipes) into
// scale=1 PNGs when TextImg is enabled for a non-code batch.
//
// lean drops the category recipes: recovery retries re-ask tasks the model
// just saw with the full scaffold, and paying the recipes twice was the
// biggest cost of an incomplete first reply (observed: a 9-task retry re-sent
// 1,998 prompt tokens, nearly doubling the run).
func (a *Agent) buildBatchMessages(batch []Task, allowCode, lean bool) []llm.Message {
	plain := func() []llm.Message {
		sys := systemPrompt(batch, a.categories)
		if lean {
			header, _ := systemPromptParts(batch, a.categories)
			sys = header + "\nAnswer ONLY the tasks listed. Every task_id must appear in the JSON."
		}
		return []llm.Message{
			{Role: "system", Content: sys},
			{Role: "user", Content: buildLeanUserPrompt(batch, a.categories)},
		}
	}
	mode := a.textImgMode()
	if mode == "" || allowCode || lean {
		return plain()
	}
	if mode == "hybrid" {
		if messages, ok := a.buildHybridMessages(batch); ok {
			return messages
		}
		return plain()
	}
	sheets := buildTaskSheets(batch, a.categories)
	sys := systemPrompt(batch, a.categories)
	manifest := textImgManifest(batch, a.categories)
	userText := "Each image is one task-type sheet. SOLVE the exact question under each task header; do not review, describe, or sentiment-classify a task unless its type below is SENTIMENT. " + manifest
	if mode == "dense" {
		sheets = []string{buildTaskSheet(batch, a.categories)}
		userText = "The image is a dense task sheet. SOLVE the exact question under each task header; do not review, describe, or sentiment-classify a task unless its type below is SENTIMENT. " + manifest
	}
	if mode == "full" {
		header, recipes := systemPromptParts(batch, a.categories)
		if recipes != "" {
			sys = header
			sheets = append([]string{"== ANSWER RECIPES ==\n" + recipes}, sheets...)
			userText = "The first image holds answer recipes; each remaining image is one task-type sheet. SOLVE the exact question under each task header; do not review, describe, or sentiment-classify a task unless its type below is SENTIMENT. " + manifest
		}
	}
	totalChars := 0
	for _, sheet := range sheets {
		totalChars += len(sheet)
	}
	if totalChars < textImgMinChars {
		return plain()
	}
	var urls []string
	for _, sheet := range sheets {
		pages, err := textimg.RenderText(textImgASCII(sheet), textimg.RenderConfig{MaxWidth: 768, MaxHeight: 728, Scale: 1})
		if err != nil || len(pages) == 0 {
			return plain()
		}
		for _, page := range pages {
			urls = append(urls, page.ToBase64DataURI())
		}
	}
	return []llm.Message{
		{Role: "system", Content: sys},
		{Role: "user", Content: userText, ImageURLs: urls},
	}
}

// buildHybridMessages leaves ids, instructions, format constraints, and short
// prompts as text. Only quoted source passages are rendered. This gives the
// model pixel-token savings without trusting OCR for "exactly three bullets",
// label inventories, function names, or arithmetic digits.
func (a *Agent) buildHybridMessages(batch []Task) ([]llm.Message, bool) {
	type item struct {
		ID     string `json:"id"`
		Kind   string `json:"k,omitempty"`
		Prompt string `json:"q"`
	}
	type sourceImage struct {
		id     string
		source string
	}
	items := make([]item, 0, len(batch))
	sources := make([]sourceImage, 0, 2)
	totalSourceChars := 0
	for _, task := range batch {
		category := ""
		if cats := a.categories[task.TaskID]; len(cats) > 0 {
			category = cats[0]
		}
		kind := textImgKind(category)
		prompt := task.Prompt
		if instruction, source, ok := splitQuotedSource(prompt, category); ok && len(source) >= textImgSourceMinChars {
			prompt = instruction + " Source text: [IMAGE:" + task.TaskID + "]."
			totalSourceChars += len(source)
			sources = append(sources, sourceImage{id: task.TaskID, source: source})
		}
		items = append(items, item{ID: task.TaskID, Kind: kind, Prompt: prompt})
	}
	if totalSourceChars < textImgMinChars {
		return nil, false
	}
	payload, err := json.Marshal(items)
	if err != nil {
		return nil, false
	}
	var urls, labels []string
	for _, source := range sources {
		pages, renderErr := textimg.RenderText(textImgASCII(source.source), textimg.RenderConfig{MaxWidth: 768, MaxHeight: 728, Scale: 1})
		if renderErr != nil || len(pages) == 0 {
			return nil, false
		}
		for pageIndex, page := range pages {
			urls = append(urls, page.ToBase64DataURI())
			label := "IMAGE:" + source.id + " source text"
			if len(pages) > 1 {
				label += fmt.Sprintf(" (page %d/%d)", pageIndex+1, len(pages))
			}
			labels = append(labels, label)
		}
	}
	return []llm.Message{
		{Role: "system", Content: systemPrompt(batch, a.categories)},
		{Role: "user", Content: string(payload), ImageURLs: urls, ImageLabels: labels},
	}, true
}

func splitQuotedSource(prompt, category string) (instruction, source string, ok bool) {
	switch category {
	case "sentiment_classification", "text_summarisation", "named_entity_recognition":
	default:
		return "", "", false
	}
	start := strings.Index(prompt, ": '")
	markerLen := len(": '")
	if start < 0 {
		start = strings.Index(prompt, ":\n\n'")
		markerLen = len(":\n\n'")
	}
	if start < 0 {
		return "", "", false
	}
	sourceStart := start + markerLen
	end := strings.LastIndex(prompt[sourceStart:], "'")
	if end < 0 {
		return "", "", false
	}
	end += sourceStart
	instruction = strings.TrimSpace(prompt[:start])
	if suffix := strings.TrimSpace(prompt[end+1:]); suffix != "" {
		instruction += " " + suffix
	}
	source = strings.TrimSpace(prompt[sourceStart:end])
	if instruction == "" || source == "" {
		return "", "", false
	}
	return instruction, source, true
}

// buildTaskSheets keeps task types on separate images. Dense mixed-category
// pages save a few image-header tokens but MiniMax can associate a prompt with
// the next id after a page transition; grouping prevents that boundary drift.
func buildTaskSheets(batch []Task, categories map[string][]string) []string {
	type group struct {
		kind  string
		tasks []Task
	}
	groups := make([]group, 0, 6)
	index := map[string]int{}
	for _, task := range batch {
		kind := "TASK"
		if cats := categories[task.TaskID]; len(cats) > 0 {
			if k := textImgKind(cats[0]); k != "" {
				kind = k
			}
		}
		i, ok := index[kind]
		if !ok {
			i = len(groups)
			index[kind] = i
			groups = append(groups, group{kind: kind})
		}
		groups[i].tasks = append(groups[i].tasks, task)
	}
	sheets := make([]string, 0, len(groups))
	for _, group := range groups {
		sheets = append(sheets, "TASK TYPE: "+group.kind+"\n"+buildTaskSheet(group.tasks, categories))
	}
	return sheets
}

// buildTaskSheet lays tasks out as plaintext blocks for OCR: an unambiguous
// '== id kind ==' header per task, prompt verbatim underneath.
func buildTaskSheet(batch []Task, categories map[string][]string) string {
	var b strings.Builder
	for _, t := range batch {
		b.WriteString("== ")
		b.WriteString(t.TaskID)
		if cats := categories[t.TaskID]; len(cats) > 0 {
			if k := textImgKind(cats[0]); k != "" {
				b.WriteByte(' ')
				b.WriteString(k)
			}
		}
		b.WriteString(" ==\n")
		b.WriteString(strings.TrimSpace(t.Prompt))
		b.WriteString("\n")
	}
	return b.String()
}

// textImgManifest keeps the identity and output type of every task in text.
// The exact prompt can be compressed into pixels, but ids and routing are too
// important to leave to OCR (T01 can otherwise become T1, and a summary can be
// mistaken for sentiment classification after an adjacent sentiment task).
func textImgManifest(batch []Task, categories map[string][]string) string {
	order := []string{"FACTUAL", "SENTIMENT", "SUMMARY", "NER", "CODE_DEBUG", "CODE_GENERATION"}
	grouped := make(map[string][]string, len(order))
	for _, task := range batch {
		kind := "TASK"
		if cats := categories[task.TaskID]; len(cats) > 0 {
			if k := textImgKind(cats[0]); k != "" {
				kind = k
			}
		}
		grouped[kind] = append(grouped[kind], task.TaskID)
	}
	parts := make([]string, 0, len(grouped))
	for _, kind := range order {
		if ids := grouped[kind]; len(ids) > 0 {
			parts = append(parts, fmt.Sprintf("%s=[%s]", kind, strings.Join(ids, ",")))
		}
	}
	if ids := grouped["TASK"]; len(ids) > 0 {
		parts = append(parts, fmt.Sprintf("TASK=[%s]", strings.Join(ids, ",")))
	}
	return "Task types and exact ids: " + strings.Join(parts, "; ") + ". Return exactly one answer for every listed id."
}

func textImgKind(category string) string {
	switch category {
	case "factual_knowledge":
		return "FACTUAL"
	case "sentiment_classification":
		return "SENTIMENT"
	case "text_summarisation":
		return "SUMMARY"
	case "named_entity_recognition":
		return "NER"
	case "code_debugging":
		return "CODE_DEBUG"
	case "code_generation":
		return "CODE_GENERATION"
	default:
		return ""
	}
}

// asciiFold maps typographic punctuation to ASCII: the 5x7 font atlas only
// covers ASCII and silently blanks other bytes, which would punch holes in
// dashes/quotes mid-sentence.
var asciiFold = strings.NewReplacer(
	"—", "-", "–", "-", "−", "-",
	"‘", "'", "’", "'",
	"“", `"`, "”", `"`,
	"…", "...", " ", " ",
	"×", "x", "÷", "/",
)

func textImgASCII(text string) string {
	text = asciiFold.Replace(text)
	var b strings.Builder
	b.Grow(len(text))
	for _, r := range text {
		switch {
		case r == '\n' || r == '\r':
			b.WriteRune(r)
		case r == '\t':
			b.WriteByte(' ')
		case r >= 32 && r <= 126:
			b.WriteRune(r)
		default:
			b.WriteByte('?')
		}
	}
	return b.String()
}
