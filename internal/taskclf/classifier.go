// Package taskclf runs the task capability classifier end to end in Go:
// byte-level BPE tokenisation (daulet/tokenizers) -> ONNX Runtime inference ->
// sigmoid + per-label thresholds -> the set of applicable capability labels.
//
// It depends on github.com/yalue/onnxruntime_go (ONNX Runtime shared library,
// via ONNXRUNTIME_LIB or the libPath arg) and github.com/daulet/tokenizers
// (links libtokenizers.a at build time).
package taskclf

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	ort "github.com/yalue/onnxruntime_go"
)

type onnxConfig struct {
	Labels        []string           `json:"labels"`
	Thresholds    map[string]float64 `json:"thresholds"`
	MaxLen        int                `json:"max_len"`
	HeadFrac      float64            `json:"head_frac"`
	TokenizerFile string             `json:"tokenizer_file"`
	ClsID         uint32             `json:"cls_id"`
	SepID         uint32             `json:"sep_id"`
	ModelInt8     string             `json:"model_int8"`
	ModelFP32     string             `json:"model_fp32"`
}

// Classifier is safe for sequential use. Call Close when done.
type Classifier struct {
	tok        *Tokenizer
	session    *ort.DynamicAdvancedSession
	labels     []string
	thresholds []float32
}

// Prediction is one applicable label with its sigmoid probability.
type Prediction struct {
	Label string  `json:"label"`
	Score float32 `json:"score"`
}

// New loads the classifier from an artefact dir (containing onnx_config.json,
// tokenizer.json and the .onnx model). onnxFile selects the model (e.g. the int8
// one from the config); pass "" to use the int8 model named in the config.
func New(dir, onnxFile, libPath string) (*Classifier, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "onnx_config.json"))
	if err != nil {
		return nil, fmt.Errorf("read onnx_config.json: %w", err)
	}
	var cfg onnxConfig
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("parse onnx_config.json: %w", err)
	}
	if onnxFile == "" {
		onnxFile = cfg.ModelInt8
	}

	head := int(float64(cfg.MaxLen-2) * cfg.HeadFrac)
	tok, err := LoadTokenizer(filepath.Join(dir, cfg.TokenizerFile), cfg.MaxLen, head, cfg.ClsID, cfg.SepID)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	if !ort.IsInitialized() {
		if libPath == "" {
			libPath = os.Getenv("ONNXRUNTIME_LIB")
		}
		if libPath != "" {
			ort.SetSharedLibraryPath(libPath)
		}
		if err := ort.InitializeEnvironment(); err != nil {
			return nil, fmt.Errorf("init onnxruntime (set ONNXRUNTIME_LIB): %w", err)
		}
	}

	session, err := ort.NewDynamicAdvancedSession(
		filepath.Join(dir, onnxFile),
		[]string{"input_ids", "attention_mask"},
		[]string{"logits"},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}

	thr := make([]float32, len(cfg.Labels))
	for i, l := range cfg.Labels {
		thr[i] = float32(cfg.Thresholds[l])
	}
	return &Classifier{tok: tok, session: session, labels: cfg.Labels, thresholds: thr}, nil
}

// Classify returns the applicable labels (prob >= threshold), highest first.
func (c *Classifier) Classify(text string) ([]Prediction, error) {
	scores, err := c.Scores(text)
	if err != nil {
		return nil, err
	}
	var out []Prediction
	for i, l := range c.labels {
		if scores[i] >= c.thresholds[i] {
			out = append(out, Prediction{Label: l, Score: scores[i]})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	return out, nil
}

// Scores returns the raw sigmoid probability for every label (index-aligned
// with the label map), regardless of threshold.
func (c *Classifier) Scores(text string) ([]float32, error) {
	ids, mask := c.tok.Encode(text)
	seq := int64(len(ids))
	shape := ort.NewShape(1, seq)

	inIDs, err := ort.NewTensor(shape, ids)
	if err != nil {
		return nil, err
	}
	defer inIDs.Destroy()
	inMask, err := ort.NewTensor(shape, mask)
	if err != nil {
		return nil, err
	}
	defer inMask.Destroy()
	out, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(len(c.labels))))
	if err != nil {
		return nil, err
	}
	defer out.Destroy()

	if err := c.session.Run([]ort.Value{inIDs, inMask}, []ort.Value{out}); err != nil {
		return nil, err
	}
	logits := out.GetData()
	scores := make([]float32, len(logits))
	for i, v := range logits {
		scores[i] = float32(1.0 / (1.0 + math.Exp(-float64(v))))
	}
	return scores, nil
}

// Labels returns the label map (index-aligned with Scores).
func (c *Classifier) Labels() []string { return c.labels }

// Close releases the tokenizer and ONNX session.
func (c *Classifier) Close() error {
	if c.tok != nil {
		c.tok.Close()
	}
	if c.session != nil {
		return c.session.Destroy()
	}
	return nil
}
