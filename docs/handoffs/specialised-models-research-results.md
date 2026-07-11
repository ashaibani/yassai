# Specialised models and datasets research results

Date: 11 July 2026

Scope: research and probes only. No training run was performed. No Dockerfile,
CI, live submission setting, or production model artefact was changed.

## Decision summary

1. Do not add Qwen2.5-Coder-1.5B as a replacement assist lane. It loads under
   llama.cpp b9948, but it gives no code-debugging gain, code generation is
   already strong, and the full served probe falls from the current direct
   assist diagnostic of 31/43 to 20/43.
2. Keep GLiNER small v2.1 ONNX as the highest-value future candidate for NER.
   Its structural probe found all 23 expected entity-label pairs in the five
   available NER tasks. It is not deadline-landable because the published
   wrapper pulls PyTorch and the ONNX graph needs a custom Go preprocessor and
   decoder before it can fit the current image.
3. Do not spend the remaining deadline on a new maths, sentiment, summary, or
   factual model. The best near-term gains are data or constrained-output work,
   and each needs a fresh served-quant acceptance run after training.

The quality results below are diagnostic. The GLiNER result is span and label
matching, not an LLM judge result. The Qwen Coder result is the repo's served
`cmd/localprobe` path with Fireworks judging, but it ran on the local arm64
machine. The amd64 Modal run is used for Qwen Coder loading, RSS, and speed.
No claim below treats bf16 or an unserved model as acceptance evidence.

## Ranked model shortlist

| Rank | Candidate and target | Expected gain versus current truth | Integration cost | Constraint check | Deadline | Decision |
|---|---|---|---|---|---|---|
| 1 | [GLiNER small v2.1 INT8 ONNX](https://huggingface.co/nexuswho/gliner-onnx-small), NER | High potential for recall and label stability. Probe: 5/5 tasks and 23/23 expected pairs at threshold 0.4. This is a structural +2-task comparison against the three-task direct Qwen NER slice, not a hidden-set claim. | High. Add a span extractor, label inventory mapping, word-boundary masks, span packing, thresholding, and a formatter. Then make the 2B produce only the required contract. | Apache-2.0. INT8 ONNX is 183.4 MB plus an 8.7 MB tokenizer. ORT 1.27 loads it. The graph requires `input_ids`, `attention_mask`, `words_mask`, `text_lengths`, `span_idx`, and `span_mask`, then returns span logits. The published GLiNER wrapper imports PyTorch, which is not allowed in the image. Local arm64 inference was 13.0 ms median at two ORT threads. No amd64 quality number was claimed because the wrapper dependency check was not production-clean. | No | GO for post-deadline engineering. NO-GO for the submission image. |
| 2 | [Qwen2.5-Coder-1.5B-Instruct GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF), code debugging and code generation | No useful net gain. Served direct probe: code debugging 3/6, code generation 5/5. Existing direct assist diagnostic: 3/6 and 3/5; project truth already calls code generation strong. | Medium if it replaces assist, high if it becomes a third lane. It needs a route change and a separate spawn. | Apache-2.0. Q4_K_M file is 1.117 GB. b9948 loads the Qwen2 architecture. Modal linux/amd64 at two threads: 1.85 GB RSS and 9.6 tokens/s on the two smoke prompts. Local arm64 full probe: 20/43 overall, with maths 5/11, logic 1/6, NER 0/3, sentiment 4/6, summarisation 0/3, and factual 2/3. | No | NO-GO. A code-only future lane is not justified while code generation is already strong and code debugging is unchanged. |
| 3 | [DistilBERT SST-2 ONNX](https://huggingface.co/optimum/distilbert-base-uncased-finetuned-sst-2-english), sentiment | Low. It supplies only Positive or Negative, cannot emit Neutral or Mixed, and still needs the 2B to write the required two-sided reason. | Low to medium. The ONNX classifier can ride ORT, but calibration and a reason handoff are required. | Apache-2.0. Ready ONNX artefact is about 269 MB. CPU compatible. The binary head does not match the published ternary rubric. | No | NO-GO. |
| 4 | [CardiffNLP Twitter RoBERTa sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), sentiment | Possible irony and social-text gain, but uncertain on product reviews and balanced reviews. | Medium. Export, add three-way calibration, and preserve the 2B reason path. | CC BY 4.0 rather than the preferred Apache-2.0 or MIT. No ready ORT artefact was verified in this pass. | No | NO-GO before the deadline. Revisit only with a matched review probe. |
| 5 | [SmolLM2 1.7B Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct), summary or general assist | Low and unproven. It is a general instruction model, not a compound-constraint summariser, and the current summary failure is mainly count, cap, and theme coverage. | Medium to high. It needs a generation loop and a verified b9948 GGUF or another supported runtime. | Apache-2.0 and an ONNX export exist, but no served b9948 probe was found. It does not remove the need for constraint-aware prompting or post-validation. | No | NO-GO. |
| 6 | [Qwen2.5-Math-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct), maths | Near-zero expected gain. The existing Python tool lane is 11/11 on the maths variants and already handles exact ratio splits. | Medium. A new Q4 export and a third lane would be needed, with no obvious quality headroom. | Apache-2.0, but the official Qwen page exposes BF16 safetensors rather than a verified official GGUF. b9948 and served Q4 were not validated. | No | NO-GO. |
| 7 | [NLI DeBERTa v3 small](https://huggingface.co/cross-encoder/nli-deberta-v3-small) or [MiniLM MS MARCO](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2), factual verifier | Near-zero without a retrieved premise. An NLI or relevance score cannot detect an invented lake name from the answer alone. | Medium. Add ORT pair scoring, a confidence policy, and a safe hedge path. | Apache-2.0. The quantized NLI ONNX is about 173 MB and the quantized MiniLM ONNX is about 23 MB, but both require a premise or candidate evidence. | No | NO-GO. A verifier becomes interesting only if retrieval or an answer-evidence contract is added. |

### Specialist probe evidence

#### GLiNER

The probe used `nexuswho/gliner-onnx-small`, `model_quantized.onnx`, ORT
1.27.0, threshold 0.4, and the exact text spans extracted from the NER
prompts. It did not read answers as training data.

| Set | Tasks | Expected pairs | Exact tasks | Pair recall | Median inference |
|---|---:|---:|---:|---:|---:|
| `variant_tasks_golden.json` NER slice | 3 | 15 | 3/3 | 15/15 | 13.0 ms |
| `wildcard_tasks.json` NER slice | 2 | 8 | 2/2 | 8/8 | 13.0 ms |
| Combined | 5 | 23 | 5/5 | 23/23 | 13.0 ms |

Threshold 0.3 also found 23/23 pairs but added a false positive, `research
campus`, on `n_v01`. Threshold 0.5 missed `Sequoia` on `n_w02`. Threshold 0.4
was the cleanest point on this tiny slice.

The result is promising because the rubric has a zero-missing-entity rule,
but it is not a ship result. The current agent still needs a formatter that
preserves source order, maps `organisation` to the prompt's exact label, and
does not omit entities when the requested inventory changes.

#### Qwen2.5-Coder-1.5B

The direct served probe used `cmd/localprobe -assist`, Q4_K_M, reasoning off,
two llama.cpp threads, 2,048 context tokens, and the repo's 43-task variant
set. The exact output was:

| Category | Qwen Coder | Current direct assist diagnostic |
|---|---:|---:|
| code debugging | 3/6 | 3/6 |
| code generation | 5/5 | 3/5 |
| factual | 2/3 | 2/3 |
| maths | 5/11 | 7/11 |
| logic | 1/6 | 5/6 |
| NER | 0/3 | 3/3 |
| sentiment | 4/6 | 6/6 |
| summarisation | 0/3 | 2/3 |
| total | 20/43 | 31/43 |

The direct baseline is the existing localprobe artefact, not the full routed
agent. It is enough to reject the replacement because the candidate does not
improve the target code-debugging slice and causes large regressions elsewhere.
The amd64 run loaded b9948 cleanly, which validates architecture support but
does not repair the quality result.

## Ranked dataset shortlist

The expected gains are deliberately modest. A dataset recommendation is not a
quality claim until a new served quantisation is evaluated on every relevant
eval-only set with the existing noise floor.

| Rank | Dataset or recipe | Target and residual failure | Licence and contamination note | Expected gain | Integration and deadline | Decision |
|---|---|---|---|---|---|---|
| 1 | Fresh parametric summary units based on `scripts/gen_assist_v2_data.py` | Summarisation: pack two or more themes into each unit while enforcing exact count, word cap, and response/outlook coverage. | Fresh authored or generated text. The existing builder checks prompts against every JSON file under `testdata` and checks source echoing, counts, caps, and theme keywords. Do not reuse any existing eval-shaped rows. | Best data bet, possibly +1 to +2 on a fresh compound holdout, but unverified. | Medium to high: rebuild the banks, train, export, then rerun served gates. The rejected v2c result shows that sample fixes do not generalise. Not deadline-landable. | GO for future research, NO-GO for a last-minute retrain. |
| 2 | [QuixBugs](https://github.com/jkoppel/QuixBugs) plus fresh mutation pairs | Code debugging: one-line fixes and sibling-case coverage. Its classic algorithms are useful for execution verification. | MIT in the upstream repository. The upstream set has 40 Python and 40 Java programs. Use Python only, retain the licence notice, and generate new mutations over clean functions. No repo task ids or prompts are used. | Possible +1 task on the six-task debugging slice after mutation augmentation. | Medium: render the existing `cbug` contract, run buggy and fixed code over generated inputs, reject partial fixes. Training and served validation are not deadline-landable. | GO for future research. |
| 3 | [DynaSent](https://huggingface.co/datasets/dynabench/dynasent), round 2 first | Sentiment: mixed-review balance, model-in-the-loop hard examples, and neutral decisions. | Dataset card states CC BY 4.0. Round 2 has 13,065 train examples and was collected with a model in the loop. It is external to this repository, so contamination risk is low by provenance. Still run exact prompt and id checks before writing SFT rows. | Possible +0 to +1 holdout task, with the main risk being loss of sarcasm competence during rebalancing. | Low to medium for data preparation, high for a safe retrain and gates. Not deadline-landable. | GO for a balanced counterweight, not a full sentiment retrain. |
| 4 | [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa) | Factual: answer-or-abstain style and resistance to common false beliefs. | Apache-2.0. It teaches truthfulness patterns, not hidden-task facts. It cannot provide missing evidence for a novel proper noun, so contamination risk is low but expected coverage is limited. | Possible +0 to +1 factual task if used for calibrated hedging, with no guarantee on invented names. | Low to medium data work, but needs a policy that distinguishes unknown from known answers. Not deadline-landable. | GO as a small future mixture, NO-GO as a standalone fix. |
| 5 | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) and [SVAMP](https://huggingface.co/datasets/ChilleD/SVAMP) train splits | Maths: phrasing diversity rendered into the existing `run_python` contract. | Both dataset cards state MIT. Use train only, render every answer from computed variables, and run exact prompt and id assertions. These are public external datasets and are not the hidden task set. | Near-zero expected gain because the current tool lane is already 11/11 on the maths variants. | Low to medium, but any retrain risks the already strong lane. Not deadline-landable. | NO-GO for this submission; retain as a future robustness pool. |
| 6 | [WNUT-17](https://huggingface.co/datasets/wnut_17) | NER: emerging and rare entities, improving recall beyond common person and organisation names. | CC BY 4.0. Label mapping is needed for the task's custom inventory. External provenance gives low contamination risk; run the standard exact checks. | Possible format and recall gain, but no direct evidence that it beats GLiNER on these prompts. | Medium for contract reformatting. Still needs a model or SFT change and served validation. Not deadline-landable. | GO for future NER tuning. |
| 7 | [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) with verified fresh mutations | Code debugging and edge-case code generation. Generate sibling-case bugs over clean Python functions and retain execution tests. | Dataset card states CC BY 4.0. Preserve attribution and do not use MBPP test rows as evaluation exemplars. Fresh mutations must be checked against every file under `testdata`. | Possible +1 to +2 edge-case fixes, but only if execution filtering rejects partial patches. | Medium to high. Training plus served validation is not deadline-landable. | GO for a controlled future data mixture. |
| 8 | [CodeXGLUE Bugs2Fix](https://github.com/microsoft/CodeXGLUE) | Code debugging: real bug-to-fix pairs at larger scale. | The repository says its code is MIT but its datasets follow the Computational Use of Data Agreement, not a simple MIT data licence. Legal and source-provenance review is required. | Possible +1 to +2, but uncertain transfer to the small Python sibling-case rubric. | High due data terms, cleaning, execution tests, and retraining. Not deadline-landable. | CONDITIONAL GO for research, NO-GO for immediate use. |
| 9 | [SemEval-2018 Task 3 irony](https://aclanthology.org/S18-1005/) | Sentiment: sarcasm and irony. The shared task has 3,834 training tweets and 784 test tweets. | The ACL paper describes the corpus and collection, but this pass did not verify a permissive dataset redistribution licence. Do not treat a mirrored copy as licence-clean. | Possible +0 to +1 on sarcastic reviews, with domain mismatch and legal risk. | Medium for data use, high for safe licensing and retraining. Not deadline-landable. | CONDITIONAL GO after rights review; otherwise NO-GO. |
| 10 | [CNN/DailyMail 1.0](https://huggingface.co/datasets/abisee/cnn_dailymail) | Summarisation: broad abstractive coverage and better paraphrasing. It does not directly teach exact bullet counts, word caps, or two-sided theme packing. | Dataset card states Apache-2.0 for version 1.0. Use only the stated version and preserve attribution. Do not assume it solves the compound rubric. | Low direct gain, perhaps useful as a small background mixture. | Medium to high because it needs a sequence-to-sequence or teacher-rendering path. Not deadline-landable. | NO-GO as the primary summary fix. |
| 11 | [Few-NERD](https://huggingface.co/datasets/DFKI-SLT/few-nerd) | NER: fine-grained entity variation. | CC BY-SA 4.0, which is less suitable than MIT or Apache-2.0 for this project and needs share-alike review. | Possible recall gain, but label remapping and legal terms reduce the value. | Medium. Not deadline-landable. | NO-GO unless the licence is explicitly accepted. |
| 12 | CoNLL-2003 and OntoNotes | NER: standard label formatting and common entities. | The available CoNLL card says `other` or unknown and points to Reuters organisational terms. OntoNotes is normally distributed under LDC terms. Neither is licence-clean for this mission. | Low to medium format gain, but no reason to accept the licence risk. | Medium. Not deadline-landable. | NO-GO. |
| 13 | LogiQA, ProofWriter, and the existing verified logic pool | Logic: semantic clue encoding and phrasing diversity. | LogiQA and ProofWriter licences were not verified as permissive in this pass. The repo's own pool is clean and already has overlap assertions, but the prior served-quant GRPO result regressed and is explicitly out of scope for repetition. | Possible +1 clue-holdout task from a clean SFT pool, but high served-quant risk. | High for training and validation. Not deadline-landable. | Keep the existing verified pool for future work. Do not repeat GRPO now. |

## Per-category recommendation

| Category | Model recommendation | Dataset recommendation | Immediate action |
|---|---|---|---|
| NER | Prototype GLiNER in a separate branch after implementing a pure ORT or Go span wrapper. | WNUT-17 or fresh contract-formatted authored examples. | Keep the current lane for the submission. |
| code_debugging | Do not route to Qwen Coder. | QuixBugs plus execution-verified fresh mutations, then MBPP mutations if licence attribution is acceptable. | Keep the current route and gates. |
| summarisation | No specialist model identified that fits both the runtime and compound rubric. | Fix the parametric generator with multi-theme units and preserve all family shares. | Do not retrain before a full served gate plan exists. |
| sentiment | No encoder beats the requirement without a reason writer and neutral class. | DynaSent round 2 first; use irony data only after rights review. | Keep the current adapter and avoid another sentiment-only DPO. |
| factual | No useful verifier without evidence. | TruthfulQA as a small calibration mixture, not a knowledge source. | Prefer conservative answer policy and do not add a verifier lane. |
| maths | No new model needed. | GSM8K and SVAMP are clean low-priority phrasing pools. | Keep the existing Python tool lane. |
| logic | No new specialist needed. | Retain the verified pool for later SFT experiments. | Do not repeat rejected GRPO. |
| code_generation | Qwen Coder has no required headroom because the category is already strong. | MBPP mutation and edge-case data are more relevant than a new base model. | Keep the current lane. |

## Contamination and acceptance notes

- The probes read `testdata` only as eval input. No task prompt, answer, id, or
  holdout row was used to build a training set, generate a teacher cache, or
  mine pairs in this mission.
- The existing repo pattern is the correct one: load all JSON files under
  `testdata`, reject exact prompt and task-id collisions, and keep the check in
  every future builder. External public data has low contamination risk by
  provenance, but provenance is not a substitute for the exact assertion.
- Any future quality claim must use the served quantisation, linux/amd64, two
  CPU threads, and the hidden-style eval-only sets. Local arm64 results are
  useful for ranking only. The Qwen Coder Modal result demonstrates the speed
  penalty that makes a third generative lane unattractive.
- A future GLiNER implementation should keep the encoder resident beside the
  task classifier, use a recall-favouring threshold, and let the 2B handle only
  label spelling and output formatting. The model itself is a better fit than
  another generative lane, but the wrapper must be made PyTorch-free first.

## Sources checked

- [GLiNER small ONNX model card](https://huggingface.co/nexuswho/gliner-onnx-small)
- [GLiNER small v2.1 base model](https://huggingface.co/urchade/gliner_small-v2.1)
- [Qwen2.5-Coder-1.5B-Instruct GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF)
- [DistilBERT SST-2 ONNX](https://huggingface.co/optimum/distilbert-base-uncased-finetuned-sst-2-english)
- [DynaSent dataset card](https://huggingface.co/datasets/dynabench/dynasent)
- [GSM8K dataset card](https://huggingface.co/datasets/openai/gsm8k)
- [SVAMP dataset card](https://huggingface.co/datasets/ChilleD/SVAMP)
- [TruthfulQA repository and licence](https://github.com/sylinrl/TruthfulQA)
- [CodeXGLUE repository and data terms](https://github.com/microsoft/CodeXGLUE)
- [QuixBugs repository and MIT licence](https://github.com/jkoppel/QuixBugs)
- [WNUT-17 dataset card](https://huggingface.co/datasets/wnut_17)
- [Few-NERD dataset card](https://huggingface.co/datasets/DFKI-SLT/few-nerd)
- [CNN/DailyMail dataset card](https://huggingface.co/datasets/abisee/cnn_dailymail)
- [SemEval-2018 Task 3 paper](https://aclanthology.org/S18-1005/)
