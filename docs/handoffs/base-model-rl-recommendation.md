# Base model + RL recommendation (yassai next assist/tool models)

Date: 10 July 2026  
Scope: restricted bake-off to **Qwen3.5-0.8B** and **Qwen3.5-2B** only (user direction; disk tight at ~8-13 GB free). Older candidates (Llama-3.2, SmolLM2, granite, Gemma-3, Phi-4-mini, Qwen2.5) intentionally not re-probed.

## Executive recommendation

| Decision | Verdict |
|---|---|
| Replace assist-lane **base** with Qwen3.5-2B | **GO (stock already beats MiniCPM5-1B)** |
| Replace tool-lane base with Qwen3.5-2B | **CONDITIONAL GO** (dual-lane RAM tight; prefer 2B assist + keep MiniCPM5-1B tool, or single-lane 2B) |
| Replace either lane with Qwen3.5-0.8B | **NO-GO** (stock 14/43, worse than MiniCPM5-1B) |
| Plausible zero-token bar (>=17/19 ungated) after SFT+RL | **PLAUSIBLE but not yet measured** on a loadable fine-tune GGUF |
| Next post-SFT stage | **GRPO/RLVR on verifiable gates first**, keep DPO as a thin sentiment/judge calibration pass |

**Headline:** stock Qwen3.5-2B Q4 scores **27/43** on the adversarial variant probe (+6 over MiniCPM5-1B stock assist 21/43 and assist-v6 21/43). After assist SFT (LoRA r=32, 3 epochs) served as **stock Q4 + LoRA GGUF**, the same probe scores **31/43** (+4 vs stock, **+10 vs v6**), with sentiment **6/6** and maths 7/11. Qwen3.5-0.8B remains a regression (14/43).

Artificial Analysis's "MiniCPM5-1B is best tiny" claim **does not hold on our judge workload** once Qwen3.5-2B is in range: their tiny leaderboard is not our glm-5p2 + family mix.

## Comparison table (stock, assist contract, variants)

Probe: `go run ./cmd/localprobe -assist -model <gguf> -tasks testdata/variant_tasks_golden.json -threads 2 -reasoning off`  
Judge: glm-5p2 (Fireworks). All gen_fail = 0 after zombie llama-server cleanup.

| Model | Overall | maths | logical | ner | code_debug | code_gen | factual | sentiment | summarisation | Q4 size | params | tg128 @ -t 2 (Metal*) | Licence | Template / thinking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| **Qwen3.5-2B stock** | **27/43** | 6/11 | **5/6** | **3/3** | 3/6 | 3/5 | 2/3 | 3/6 | **2/3** | 1.29 GiB | 1.94 B | 33.6 t/s | Apache-2.0 | ChatML + empty `<think>`; serve `--reasoning off` |
| MiniCPM5-1B stock assist | 21/43 | 4/11 | 2/6 | 2/3 | 2/6 | **4/5** | 2/3 | 4/6 | 1/3 | 0.64 GiB | 1.08 B | 89.6 t/s | MiniCPM licence | ChatML + empty think; `--reasoning off` |
| MiniCPM5 assist-v6 (prior FT) | 21/43 | 6/11 | 0/6 | 2/3 | 3/6 | 4/5 | 2/3 | 3/6 | 1/3 | 0.64 GiB | 1.08 B | (fast) | same | same |
| MiniCPM5 assist-dpo-v7 (prior) | 20/43 | 5/11 | 1/6 | 1/3 | 3/6 | 2/5 | 2/3 | **5/6** | 1/3 | 0.64 GiB | 1.08 B | (fast) | same | DPO helped sentiment, hurt cg/ner |
| **Qwen3.5-0.8B stock** | **14/43** | 3/11 | 3/6 | 0/3 | 2/6 | 1/5 | 1/3 | 4/6 | 0/3 | 0.54 GiB | 0.77 B | 81.9 t/s | Apache-2.0 | same ChatML/think |
| Qwen2.5-3B stock (prior, out of scope) | 25/43 | 8/11 | 2/6 | 2/3 | 3/6 | 4/5 | 2/3 | 4/6 | 0/3 | 1.8 GiB | ~3 B | slower / dual-unfit | Apache-2.0 | ChatML |

\* `llama-bench -t 2 -ngl 0` still used Metal on this M5 host; treat tok/s as **relative**, not judge-VM absolute. On 2 vCPU linux/amd64 expect roughly 3-8x slower; 2B still fits a 19-task pass inside ~7 min if avg completion stays ~300 tokens (budget: ~19 * 300 / 15 t/s ≈ 6 min worst case at 15 t/s).

### Family takeaways

- **Logic / NER / summarisation:** Qwen3.5-2B stock is clearly stronger than MiniCPM5-1B (logic 5/6 vs 2/6; NER 3/3; su 2/3).
- **Codegen:** MiniCPM5 still edges stock 2B (4/5 vs 3/5) - this is the main SFT target for Qwen.
- **Sentiment:** still the 1B/2B ceiling problem; DPO-v7 showed sentiment can move (5/6) while other families regress - do not DPO-only.
- **0.8B:** fails NER/summarisation/codegen systematically; only useful as a draft/speculative model, not a lane base.

### RAM / dual-lane fit (4 GB judge VM)

| Layout | Weights (Q4) | Fit guess |
|---|---:|---|
| MiniCPM5 tool 1B + MiniCPM5 assist 1B (current) | ~1.3 GiB | fits (proven) |
| MiniCPM5 tool 1B + **Qwen3.5-2B assist** | ~1.95 GiB | **tight but likely** with ctx 2-4k, n_parallel 1 |
| Qwen3.5-2B only (single lane for all families) | ~1.3 GiB | comfortable |
| Dual Qwen3.5-2B | ~2.6 GiB | **NO** once KV + agent + Micropython |
| Dual 0.8B | ~1.1 GiB | fits but accuracy NO-GO |

## Fine-tune of best non-incumbent (Qwen3.5-2B)

### What ran

```bash
uvx modal run finetune/minicpm5/modal_train.py \
  --dataset assist --epochs 3 --tag q35-2b \
  --base-model Qwen/Qwen3.5-2B
```

- Checkpoint: `/checkpoints/assist-e3-r32-q35-2b/{adapter_final,merged_hf}` on volume `yassai-minicpm5-checkpoints`
- Train: 450 assist rows, 3 epochs, LoRA r=32, ~9.4 min on H100, final train_loss ≈ 0.70, mean token accuracy ≈ 0.90
- Template contract: **`template_mode: train_chat_template`** (official no-think render matched our ChatML empty-think training template)
- LoRA targets expanded for hybrid Gated DeltaNet: `q/k/v/o_proj`, `gate/up/down_proj`, `in_proj_{qkv,a,b,z}`, `out_proj`

### Pipeline changes shipped in-repo

- `finetune/minicpm5/modal_train.py`: `--base-model`, HF secret, transformers **5.13.0** / peft **0.19.1** / trl **0.24.0** (Qwen3.5 = `model_type: qwen3_5` needs transformers>=5)
- `finetune/minicpm5/train_trl.py`: multi-base template assert, AutoModel fallback, expanded LoRA targets
- `finetune/minicpm5/modal_export_gguf.py`: run-name-scoped GGUF stems
- `finetune/minicpm5/eval_assist_behavior.py`: BatchEncoding-safe `generate()` for transformers 5
- `finetune/minicpm5/rewards_rlvr.py`, `train_grpo.py`, `modal_grpo.py`: GRPO/RLVR prototype
- `finetune/minicpm5/modal_export_lora_gguf.py`: LoRA-GGUF export path (for hybrid bases)

### SFT probe result (measured)

Serve path that works on llama.cpp b9948:

```bash
go run ./cmd/localprobe -assist \
  -model models/qwen35/Qwen_Qwen3.5-2B-Q4_K_M.gguf \
  -lora models/qwen35/yassai-assist-e3-r32-q35-2b-lora-f16.gguf \
  -tasks testdata/variant_tasks_golden.json
```

| Checkpoint | Overall | maths | logical | ner | code_debug | code_gen | factual | sentiment | summarisation | gen tok | avgMs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B stock | 27/43 | 6/11 | 5/6 | 3/3 | 3/6 | 3/5 | 2/3 | 3/6 | 2/3 | 12440 | 5785 |
| **Qwen3.5-2B + assist SFT LoRA** | **31/43** | **7/11** | 5/6 | 3/3 | 3/6 | 3/5 | 2/3 | **6/6** | 2/3 | **7215** | **3275** |
| MiniCPM5 assist-v6 | 21/43 | 6/11 | 0/6 | 2/3 | 3/6 | 4/5 | 2/3 | 3/6 | 1/3 | 8148 | 1261 |

SFT gains: **+4 overall**, perfect sentiment (6/6), +1 maths, shorter answers (7.2k vs 12.4k gen tokens). Codegen did not move (still 3/5) - GRPO with `exampleCheck` reward is the right next lever.

### Export / serve notes

1. **Merged GGUF incomplete on b9948:** `convert_hf_to_gguf` omits MTP/`blk.24` → `missing tensor 'blk.24.attn_norm.weight'`. Do not serve the merged Q4.
2. **LoRA GGUF works:** `modal_export_lora_gguf.py` produced a 64 MB f16 adapter; `llama-server -m stock.gguf --lora adapter.gguf` loads cleanly. `cmd/localprobe -lora` flag added for this path.
3. **HF private storage full:** 403 on push to `ashaibani/yassai-minicpm5-local`. Artefacts on Modal volume + local `models/qwen35/`.
4. **Behaviour eval crash:** fixed in `eval_assist_behavior.py` (BatchEncoding); re-run gate before treating adapter as CI-green.

## RL pipeline design

### Existing: DPO (do not block on it)

- Miner: `scripts/mine_dpo_pairs.py` from `eval-results/localprobe-*.json`
- Train: `modal_dpo.py` / `train_dpo.py` (TRL DPOTrainer)
- Evidence: DPO-v7 moved sentiment 3/6 → 5/6 but overall 21 → 20 and hurt codegen/NER. Treat DPO as a **narrow calibration** pass (sentiment/factual pairs only), never a full multi-family pass without behaviour gates.

### Recommended: GRPO / RLVR after SFT

TRL **0.20+ ships `GRPOTrainer`** (verified import). Prototype in-repo:

| File | Role |
|---|---|
| `finetune/minicpm5/rewards_rlvr.py` | Verifiable rewards ported from `internal/localllm/direct.go` gates + `eval_assist_behavior.py` |
| `finetune/minicpm5/train_grpo.py` | GRPOTrainer LoRA loop |
| `finetune/minicpm5/modal_grpo.py` | H100 runner; builds prompt pool from assist train/heldout only |

**Reward design (anti-hack):**

| Family | Content signal | Format signal | Notes |
|---|---|---|---|
| code_generation | worked-example `exec` match (0.5) | def name + ast.parse (0.5) | strongest RLVR |
| code_fix | cause-line keywords | def + parse | pair with buggy-fails tests later |
| ner | high-precision span recall | `Type: span` shape | conservative spans only |
| summarisation | light anti-echo | exact bullet/sentence count | mirrors gateSummarisation |
| sentiment | (optional judge_bonus ≤0.4) | label + reason length | format alone ≤0.6 |
| factual | (optional judge_bonus) | ≤4 sentences | never train on golden/live |

**Recipe (next session):**

1. SFT on Qwen3.5-2B assist (done) → fix GGUF/LoRA serve path → probe.
2. GRPO 1 epoch, `num_generations=4`, lr=5e-6, LoRA r=16, pool = heldout-first assist prompts (max 40/family), **no Fireworks in the loop**.
3. Behaviour gate must pass (`eval_assist_behavior.py`).
4. Optional: sentiment-only DPO with ≤30 pairs if sarcasm still flat.
5. Sparse judge-in-the-loop only for residual sentiment/factual (concurrency ≤3), mixed as `judge_bonus` ≤0.4 so format gaming cannot dominate.

**Expected gain vs v6/DPO line:**

| Stage | Variants (est.) | Notes |
|---|---:|---|
| MiniCPM5 assist-v6 | 21/43 | current |
| Qwen3.5-2B stock | 27/43 | measured |
| + SFT assist (LoRA serve) | **31/43** | **measured** (sentiment 6/6; codegen still 3/5) |
| + GRPO RLVR | 33-36/43 | target codegen via exampleCheck; su shape |
| + thin sentiment DPO | +0-2 sentiment | only if GRPO leaves sarcasm cold |
| Map to 19 golden ungated | ~14-17/19 | **17 needs both SFT packaging fix and RL**; stock 2B alone is not enough for zero-token |

Zero-token bar (>=17/19 ungated) is **plausible** if SFT recovers codegen edge cases and GRPO locks summarisation/NER shape, but **not proven** until a loadable fine-tune GGUF is probed. Do not drop remote fallback until golden-19 ungated ≥17 twice in a row (judge variance ±1-2).

## GO / NO-GO summary

| Lane | Base change | GO/NO-GO | Why |
|---|---|---|---|
| Assist | MiniCPM5-1B → **Qwen3.5-2B** | **GO** | +6 stock variants; Apache-2.0; GGUF+llama.cpp ok; template compatible |
| Tool (math/logic) | keep MiniCPM5-yassai-v2e3b **or** try Qwen3.5-2B later | **HOLD** | tool contract already strong; dual 2B RAM risk; re-evaluate after assist lands |
| Any lane | Qwen3.5-0.8B | **NO-GO** | 14/43 stock |
| Ship SFT via LoRA GGUF | stock Q4 + lora-f16 | **GO** | 31/43 measured; wire `--lora` in assist lane |
| Ship merged SFT GGUF | merged Q4 | **NO-GO** | missing blk.24/MTP on b9948 converter |
| RL | GRPO prototype | **GO to run** after loadable SFT | rewards + Modal runner ready; DPO stays secondary |

## Artefacts

| Artefact | Location |
|---|---|
| Stock probes | `eval-results/localprobe-qwen35-{0.8b,2b}-stock-variants.json`, `localprobe-minicpm5-1b-stock-assist-variants.json` |
| Stock GGUFs | `models/qwen35/Qwen_Qwen3.5-{0.8B,2B}-Q4_K_M.gguf` (bartowski) |
| SFT run | Modal volume `yassai-minicpm5-checkpoints/assist-e3-r32-q35-2b/` |
| Broken merged Q4 (local) | `models/qwen35/yassai-assist-e3-r32-q35-2b-Q4_K_M.gguf` (do not serve) |
| **Working LoRA GGUF (local)** | `models/qwen35/yassai-assist-e3-r32-q35-2b-lora-f16.gguf` (64 MB) |
| PEFT adapter (local) | `models/qwen35/adapter-q35-2b-sft/adapter_final/` |
| SFT LoRA probe | `eval-results/localprobe-qwen35-2b-sft-lora-variants.json` (**31/43**) |
| HF push | blocked (private storage limit) |

## Immediate next steps (one-line CI adoption path)

1. **Done:** LoRA GGUF export + variants probe → **31/43**.
2. Wire assist lane to `Qwen3.5-2B Q4 + yassai-assist-e3-r32-q35-2b-lora-f16.gguf`; keep tool lane MiniCPM5; measure dual RSS on amd64.
3. Probe golden-19 ungated (target signal toward 17/19).
4. Run `modal_grpo.py --base-run assist-e3-r32-q35-2b --tag grpo1` (codegen-heavy reward) after `eval_assist_behavior` green.
5. Free HF storage and push LoRA + stock pointers for CI one-line adopt.
6. Only then consider dropping remote for assist families (sentiment already local-safe at 6/6).

## Appendix: traps confirmed this run

- Dozens of leaked `llama-server` processes from prior sessions OOM'd probes; always reap before bake-offs.
- transformers 4.57.3 cannot load `qwen3_5` - need 5.x for train.
- Qwen3.5 is hybrid (Gated DeltaNet + attention + MTP); MiniCPM-centric merge export is insufficient.
- Judge variance still ±1-2 tasks; only treat deltas ≥2 as signal (2B stock +6 is solid).
