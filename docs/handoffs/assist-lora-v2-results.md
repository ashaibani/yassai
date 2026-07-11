# Assist LoRA v2 results

Date: 11 July 2026

Recommendation: **NO-GO. Do not replace `yassai-assist-e3-r32-q35-2b-lora-f16.gguf`.**

The candidate fixes sample task T04b and passes the native amd64 serve canary, but it fails the fresh summarisation improvement bar, regresses variant summarisation, and worsens the fresh sentiment holdout from 7/8 to 6/8. The full-agent variant result is also below the 38/43 ship bar. The artefact is valid for further research, not adoption.

## Ship-bar results

All model comparisons below use the stock `Qwen3.5-2B` Q4_K_M base plus an f16 serve-time LoRA through llama-server b9948 with reasoning disabled. The baseline was rerun in this session. Judge variance is approximately one or two tasks, so only deltas of at least two are treated as signal.

| Set | Current adapter re-baseline | v2c candidate | Ship bar | Result |
|---|---:|---:|---:|---|
| New compound summarisation holdout | 0/10 | 1/10 | baseline + at least 2 | FAIL |
| Sample T04 and T04b, direct served probe | 1/2 | 2/2 | 2/2 | PASS |
| Variant summarisation, `su_v01/v02/a01` | 2/3 | 1/3 | at least 2/3 | FAIL |
| Wildcard summarisation, `su_w01/w02` | 1/2 | 2/2 | 2/2 | PASS |
| Sentiment holdout | 7/8 | 6/8 | 8/8 | FAIL |
| Variant sentiment | 6/6 | 6/6 | 6/6 | PASS |
| Full-agent variants total | 38/43 recorded baseline | 34/43 | at least 38/43 | FAIL |
| Direct assist probe variants total | 31/43 | 29/43 | diagnostic only | REGRESSED |
| Behaviour gate | green | green | green | PASS |
| Native amd64 canary | HEALTHY | HEALTHY | HEALTHY | PASS |

The practice set moved from 34/40 to 31/40 on the direct assist probe. Its summarisation slice held at 4/5, but the failed item changed from a coverage omission to a factual and theme-coverage error.

The candidate's sample summarisation slice improved as intended: T04 and T04b both pass. This did not generalise reliably. On the new holdout, format adherence improved, but errors shifted to one-word cap overruns and omitted secondary themes. On variants, `su_v02` lost the response theme and `su_a01` omitted on-device AI. This is the exact trade the compound gate was designed to catch.

The sentiment rebalance also failed at the served quant. The current adapter misses only balanced task `sh_03`. The candidate still misses `sh_03` and newly misreads sarcastic `sh_01` as Positive, scoring 6/8. This outweighs the bf16 behaviour result of 12/12.

The full-agent 34/43 run used both production Qwen LoRAs, the ONNX classifier, `AGENT_NO_REMOTE=1`, and zero remote calls. Factual tasks were not unlocked in `LOCAL_BASE_EXTENDED` and therefore scored 0/3 in that local-only reproduction. This routing detail does not affect the NO-GO call because the assist-specific summarisation and sentiment gates already fail independently.

## Data changes

New deterministic generator: `scripts/gen_assist_v2_data.py`.

Generated caches:

- `finetune/minicpm5/data/assist_v2_summaries.jsonl`: 120 compound-constraint summarisation rows.
- `finetune/minicpm5/data/assist_v2_sentiment.jsonl`: 36 mixed-review counterweights.

The original sentiment audit was:

- Teacher cache: 90 rows, comprising 30 Positive, 45 Negative and 15 Neutral.
- Authored cache: 69 rows, comprising 44 Positive, 20 Negative and 5 Neutral.
- Combined: 159 rows, comprising 74 Positive, 65 Negative and 20 Neutral.

The final `v2c` hash-split training data has 623 rows:

| Family | Training rows |
|---|---:|
| Summarisation | 148 |
| Sentiment | 172 |
| NER | 120 |
| Code generation | 99 |
| Factual | 47 |
| Code fix | 37 |

The held-out behaviour dataset has 117 rows before the evaluator's 12-per-family cap. Code generation is weighted three times because the first candidate, `v2b`, diluted that slice and failed the gate at 8/12. The single composition correction recovered it to 10/12 in `v2c`.

Every generated summary is checked before inclusion for exact unit count, per-unit word cap, a six-word minimum per unit, absence of any shared contiguous eight-word source run, and presence of every derived theme keyword. The builder now loads every JSON file under `testdata/` and asserts that no training row collides by task id or exact prompt text. Eval answers are never read by the builder or generator.

The training system instruction remains byte-identical to serving:

`Obey the stated length and format limits exactly; cover every major theme; no preamble.`

## Training and artefacts

Accepted research checkpoint: `assist-e3-r32-q35-assist-v2c`.

- Training app: `ap-nHsdiCIWV9uomCVZcKfTH4`
- Modal checkpoint volume: `yassai-minicpm5-checkpoints`
- PEFT adapter: `/checkpoints/assist-e3-r32-q35-assist-v2c/adapter_final`
- Merged HF output: `/checkpoints/assist-e3-r32-q35-assist-v2c/merged_hf`, not a ship artefact
- Export app: `ap-5WTds8TAbKUZwh6sjO6dOp`
- LoRA GGUF on volume: `/checkpoints/assist-e3-r32-q35-assist-v2c/gguf/yassai-assist-e3-r32-q35-assist-v2c-lora-f16.gguf`
- Local LoRA GGUF: `models/qwen35/yassai-assist-e3-r32-q35-assist-v2c-lora-f16.gguf`
- LoRA size: 67,302,880 bytes
- amd64 canary app: `ap-MpBBf6GabYiZpATnPCI9Z6`
- amd64 verdict: `HEALTHY`

Training used `Qwen/Qwen3.5-2B`, three epochs, LoRA rank 32 and learning rate `1e-4`. Runtime was 962.9 seconds for 234 steps.

Behaviour gate for `v2c`:

| Family | Result | Floor |
|---|---:|---:|
| Code fix | 5/5 | 70% |
| Code generation | 10/12 | 70% |
| Factual | 11/11 | 90% |
| NER | 12/12 | 80% |
| Sentiment | 12/12 | 75% |
| Summarisation | 11/12 | 60% |

The earlier `v2b` run, app `ap-8ixbnjjHYN2kyVCr98AcVV`, is rejected because code generation scored 8/12, below its floor. An even earlier stopped `v2` app, `ap-ogFWiPbCyNfcDqAb2v0ijh`, skipped the new caches because of incorrect container-relative defaults and is invalid. Neither run was exported for acceptance.

## Evaluation artefacts

Baseline outputs use the `eval-results/assist-v2-baseline-*.json` prefix. Candidate outputs use `eval-results/assist-v2c-*.json`. The fresh eval-only set is `testdata/su_holdout_tasks.json`.

GRPO was not attempted. The SFT candidate already violates the sentiment and other-family acceptance gates, and summarisation-only RL cannot repair those regressions. The known TRL import blocker also remains unresolved. A future attempt should first rebalance data without erasing sarcasm competence, broaden summary targets so each compact unit carries multiple themes, and preserve the original NER and code-generation gradient shares from the start.

No CI defaults, Dockerfile lines, Hugging Face production artefacts, or live submission settings were changed.
