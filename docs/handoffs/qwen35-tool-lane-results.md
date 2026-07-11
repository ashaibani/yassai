# Qwen3.5-2B tool-lane training results

Date: 11 July 2026
Session tag: `q35-tool-v1`
Base: `Qwen/Qwen3.5-2B`
SFT checkpoint: `v2-e3-r32-q35-tool-v1`
Serve path: stock `Qwen_Qwen3.5-2B-Q4_K_M.gguf` + serve-time LoRA (merged GGUF broken on b9948 MTP)

## Artefacts

| Item | Location |
|---|---|
| Modal SFT volume dir | `yassai-minicpm5-checkpoints/v2-e3-r32-q35-tool-v1` (`adapter_final`, `merged_hf`, checkpoints 275/300/324) |
| LoRA GGUF (f16, 67.3 MB) | Modal `.../gguf/yassai-v2-e3-r32-q35-tool-v1-lora-f16.gguf` |
| Local copy | `models/qwen35/yassai-v2-e3-r32-q35-tool-v1-lora-f16.gguf` |
| HF artefact | `ashaibani/yassai-minicpm5-local/yassai-v2-e3-r32-q35-tool-v1-lora-f16.gguf` |
| SFT Modal app | `ap-UjkqKnKqE8qlw0RzNuOYqL` (train_loss 0.1432, 3 epochs, 324 steps) |
| Behaviour gate app | `ap-W3o1nuKjZcPKLF3EmtVW5r` |
| Export app | `ap-RNPmsEBtI34NX5ZZb2v3r4` |
| amd64 validate app | `ap-ekDXRJ0f6PMvG3onKEpPEP` |

## Wiring added (not adopted in CI defaults)

- `internal/localllm.New()` now accepts `LoraPath` + `--reasoning off` (mirrors assist `NewDirect`).
- Agent config/env: `LOCAL_MODEL_LORA_PATH`.
- Dockerfile: `LOCAL_MODEL_LORA_URL` -> `/assets/localmodel/tool-lora.gguf`.
- CI build-arg `LOCAL_MODEL_LORA_URL` defaults empty (live image unchanged).
- `cmd/localmodeleval -lora` flag.
- `finetune/minicpm5/modal_validate_tool_lora.py` (amd64 stock+LoRA canary).
- `finetune/minicpm5/modal_tool_gate.py` (re-gate without retrain).
- `eval_tool_behavior.py` fixed for transformers 5 BatchEncoding.
- GRPO image: transformers 5.13 + peft 0.19 + Qwen LoRA targets; TRL pin still unsettled for GRPO import graph.

## Ship-bar table (SFT only; GRPO not landed)

| Metric | 1B v2e3b baseline | Ship bar | Qwen3.5-2B tool SFT (`q35-tool-v1`) |
|---|---|---|---|
| Variants maths (realeval, gated) | 10/11 | >=10/11 | **11/11** (local 11; +1 summarisation misroute in set) |
| Variants logic (realeval, gated) | 6/6 | 6/6 | **6/6** (local 4; 1 classifier skip + 1 python reject fell remote) |
| Clue holdout (no-remote, tool only) | 5/8 | >=7/8 | **6/8** (fails `ch_01`, `ch_02` via `assert len(sols)==1`) |
| Golden maths+logic (realeval) | 6/6 | 6/6 | **6/6** (local 5; T07 print-join reject then remote) |
| Behaviour gate held-out v2 | n/a | pass | **846/846** |
| amd64 stock+LoRA canary | n/a | HEALTHY | **HEALTHY** (math tool call, 8.2s @ 2 threads) |
| localmodeleval smoke | n/a | framing ok | **2/3** (warehouse+trains PASS; pets_logic skipped by semantic-clue gate without allow flag) |
| Wall clock, 19-task rehearsal | 214s | <600s | **not measured** (no CI default change; adoption deferred) |

## Interpretation

- Maths non-regression cleared. Stronger than the 1B bar on variants.
- Plain logic variants cleared at 6/6, but local acceptance is not yet perfect (code bugs + classifier disagreement still cost local free answers).
- Clue holdout improved 5/8 -> 6/8 but misses the >=7/8 ship bar. Same failure mode as the 1B: semantic clue encoding yields non-unique enumeration (`assert len(sols)==1`).
- Capacity alone is not enough for the remaining two holdout fails; GRPO on the logic pool is still the intended next lever.

## GO / NO-GO (SFT checkpoint alone)

**NO-GO to replace the live tool lane yet.**

Reasons:
1. Clue holdout 6/8 < 7/8 ship bar (the stated last systematic blocker for zero-token).
2. No 19-task wall-clock rehearsal on the dual-Qwen layout yet.
3. Live CI defaults intentionally untouched; adoption is a separate main-session decision.
4. Tool-lane GRPO on the Qwen SFT merge did not land this session (packaging block below).

**Conditional GO after GRPO** if re-eval shows clue holdout >=7/8 without maths regression, plus an amd64 dress-rehearsal wall clock comfortably under 600s with a single Qwen base GGUF + tool LoRA + assist LoRA sequential serve.

## GRPO status (blocked this session)

Attempted: `modal_grpo.py --lane tool --base-run v2-e3-r32-q35-tool-v1 --tag q35-tool-grpo-v1`.

Blocker: TRL `GRPOTrainer` import graph on the Qwen3.5 image (transformers 5.13) optionally reaches `vllm` and `mergekit`. Those deps conflict with the SFT stack (`mergekit` pins `safetensors~=0.5`; full vLLM is heavy and unnecessary for our transformers-generate path). Shallow module stubs fail because importlib requires a real `__spec__`.

Working options for the next session (pick one, do not thrash):
1. Pin a known-good TRL that imports without vLLM/mergekit on transformers 5, or install a minimal real mergekit+deps set that does not downgrade safetensors.
2. Vendor a tiny local GRPOTrainer shim that only needs transformers+peft (drop TRL import).
3. Keep MiniCPM5 `tool-grpo-cluev1` (other session) as the 1B hedge; do not touch that run.

## Recommended next steps (main session)

1. Unblock Qwen tool GRPO packaging (option 1 or 2 above); train `q35-tool-grpo-v1` from `v2-e3-r32-q35-tool-v1/merged_hf`.
2. Re-gate with `modal_tool_gate.py`; re-run clue holdout + maths variants.
3. If holdout clears: export GRPO LoRA GGUF, validate amd64, measure 19-task rehearsal with `LOCAL_MODEL_PATH=Qwen_Qwen3.5-2B-Q4_K_M.gguf` + tool LoRA + existing assist LoRA (shared base mmap benefit).
4. Only then flip CI `LOCAL_MODEL_URL` / `LOCAL_MODEL_LORA_URL` away from MiniCPM5-v2e3b.

## Notes / traps paid this run

- Do not launch multiple identical Modal SFT clients; untagged/retry races produced triple H100 jobs.
- `eval_tool_behavior.py` bare-tensor generate path breaks on transformers 5 BatchEncoding (fixed).
- TRL 0.24 GRPOTrainer import pulls mergekit; TRL 0.20 pulls vLLM - neither is free on the transformers-5 image without packaging work.
- Health checks must require HTTP 200 (`ok`), not curl exit 0 on 503-while-loading.
- Shared-base layout (one Qwen Q4 + tool LoRA + assist LoRA) remains the right adoption shape once accuracy clears.
