package taskclf

import "github.com/daulet/tokenizers"

// Tokenizer wraps the HuggingFace byte-level BPE tokenizer (via daulet/tokenizers,
// which loads tokenizer.json through the same Rust lib HF uses - verified to match
// exactly) and applies head+tail truncation: the first `head` and last
// `maxLen-2-head` content tokens, so a leading OR trailing instruction survives.
type Tokenizer struct {
	tk           *tokenizers.Tokenizer
	clsID, sepID uint32
	maxLen, head int
}

// LoadTokenizer loads tokenizer.json. head is the number of content tokens kept
// from the front when truncating (tail = maxLen-2-head).
func LoadTokenizer(path string, maxLen, head int, clsID, sepID uint32) (*Tokenizer, error) {
	tk, err := tokenizers.FromFile(path)
	if err != nil {
		return nil, err
	}
	return &Tokenizer{tk: tk, clsID: clsID, sepID: sepID, maxLen: maxLen, head: head}, nil
}

// Encode returns input_ids ([CLS] + head+tail content + [SEP]) and attention_mask.
func (t *Tokenizer) Encode(text string) (ids, mask []int64) {
	content, _ := t.tk.Encode(text, false) // no special tokens; we add them below
	if cm := t.maxLen - 2; len(content) > cm {
		tail := cm - t.head
		merged := make([]uint32, 0, cm)
		merged = append(merged, content[:t.head]...)
		merged = append(merged, content[len(content)-tail:]...)
		content = merged
	}
	ids = make([]int64, 0, len(content)+2)
	ids = append(ids, int64(t.clsID))
	for _, id := range content {
		ids = append(ids, int64(id))
	}
	ids = append(ids, int64(t.sepID))
	mask = make([]int64, len(ids))
	for i := range mask {
		mask[i] = 1
	}
	return ids, mask
}

func (t *Tokenizer) Close() {
	if t.tk != nil {
		t.tk.Close()
	}
}
