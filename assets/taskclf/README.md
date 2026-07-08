# Task capability classifier (model card - v2)

Multi-label classifier that tags a task prompt with one or more of the 8
capability categories. A small ModernBERT encoder exported to ONNX for fast,
in-process CPU inference from Go.

- **Backbone:** `ibm-granite/granite-embedding-small-english-r2` (Granite
  Embedding R2, ~47M params, ModernBERT architecture), fine-tuned as an 8-way
  multi-label head (sigmoid + thresholds). Apache-2.0.
- **Context:** native **8192 tokens** (RoPE); inference uses **head+tail**
  truncation to `max_len` so a leading *or* trailing instruction always survives.
- **Accuracy:** hard real-benchmark eval micro-F1 **0.977** (v1 MiniLM was 0.889);
  easy short-prompt set 38/40.
- **Footprint:** int8 model **~48 MB**.
- **Agent A/B:** adding the predicted categories to the batch prompt cut Fireworks
  tokens **~12%** (the hint reduces the model's reasoning) at a ~1-task accuracy
  cost - a net win under the token-efficiency ranking.

## Categories (output / logit order)

`0 factual_knowledge` · `1 mathematical_reasoning` · `2 sentiment_classification` ·
`3 text_summarisation` · `4 named_entity_recognition` · `5 code_debugging` ·
`6 logical_deductive_reasoning` · `7 code_generation`

## Files

| file | what |
|---|---|
| `model.int8.onnx` | int8 model (deploy this) |
| `tokenizer.json` | HF byte-level BPE tokeniser (loaded by daulet/tokenizers) |
| `label_map.json` | labels in output order |
| `thresholds.json` | per-label decision thresholds (uniform 0.5, recall-favouring) |
| `onnx_config.json` | contract: io names, max_len, head_frac, cls/sep ids, tokenizer type |

## Inference contract

- **Inputs:** `input_ids`, `attention_mask` (int64, `[batch, seq]`); no `token_type_ids`.
- **Output:** `logits` (float32, `[batch, 8]`).
- **Tokenisation:** ModernBERT byte-level BPE; `[CLS]` + head+tail(content, max_len) + `[SEP]`.
- **Decision (multi-label):** `sigmoid(logits[i]) >= thresholds[i]` → label `i`. Zero, one, or several may fire.

## How it was built

- **Data (mostly real):** benchmark **train** splits (GSM8K, SST-2, CNN/DailyMail,
  MBPP, SciQ, CoNLL-2003, LogiQA, TriviaQA) + a small premium-synthetic set for
  gaps (code_debugging, multi-label combos, factual/summarisation) + a diverse
  synthetic slice + short/terse examples for the easy distribution. Real vs eval
  splits kept disjoint (no leakage).
- **Training:** ModernBERT fine-tune, sigmoid + BCE with per-label `pos_weight`.
- **Export:** `optimum` ONNX export (handles ModernBERT RoPE) + int8 dynamic
  quantisation; PyTorch↔ONNX parity verified (max logit diff ~9e-6, int8 pred
  agreement ~98%).

## Integration

Used in-process by the agent (`internal/taskclf`): daulet/tokenizers loads
`tokenizer.json` (verified byte-identical to HuggingFace) → onnxruntime_go
inference → sigmoid + thresholds. **Best-effort**: if the ONNX runtime or model
is missing, the agent runs unchanged. Build is cgo (statically links
`libtokenizers.a`; dlopens `libonnxruntime.so` at runtime via `ONNXRUNTIME_LIB`);
runtime image is `python:3.12-slim` (a real shell + `python3` are needed for the
`sh.run` action path). See the Dockerfile - no manual steps.
