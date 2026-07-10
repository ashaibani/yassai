# syntax=docker/dockerfile:1
# Judging VM is linux/amd64. Build with: docker buildx build --platform linux/amd64 ...

# --- fetch native libs for the target platform ---
# onnxruntime: dlopen'd at runtime by yalue/onnxruntime_go (needs >=1.23 for ORT API 26).
# libtokenizers: statically linked at BUILD time by daulet/tokenizers (HF BPE tokeniser).
FROM debian:bookworm-slim AS deps
ARG ORT_VERSION=1.27.0
ARG TOK_VERSION=1.27.0
# llama.cpp shared libs for the in-container MiniCPM5 offload (yzma dlopens
# them; purego, no cgo). yzma v1.18 needs >= b9946 (llama_model_n_layer_nextn);
# b9620 libs load-fail and silently disable the local model (seen in the
# sha-118fcd3 leaderboard image: GGUF baked, load failed, local_answers=0).
# b9948 = version verified locally (macOS arm64 + this Dockerfile's x64 dist):
# the base lane's `--reasoning off` flag and the yzma v1.18 symbols both present.
ARG LLAMA_VERSION=b9948
# Optional: URL of the fine-tuned GGUF. Left empty the image builds without a
# local model and the agent silently runs Fireworks-only. Private Hugging Face
# URLs authenticate via the BuildKit secret `hf_token` (never an ARG - ARGs
# persist in image history).
ARG LOCAL_MODEL_URL=""
# Optional: LoRA adapter GGUF for the tool lane (Qwen3.5 base + serve-time
# LoRA when merged export is incomplete). Missing file degrades to the
# base/merged GGUF alone.
ARG LOCAL_MODEL_LORA_URL=""
# Optional: URL of the assist-lane base GGUF (second local lane:
# code_generation + gated NER + LOCAL_BASE_EXTENDED families). Same auth
# rules as LOCAL_MODEL_URL.
ARG LOCAL_BASE_MODEL_URL=""
# Optional: LoRA adapter GGUF applied to the assist base at serve time
# (hybrid bases - Qwen3.5's MTP block - whose merged export the pinned
# converter cannot produce). Missing file degrades to stock serving.
ARG LOCAL_BASE_LORA_URL=""
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /tmp/ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
    && mkdir -p /ort && tar xzf /tmp/ort.tgz -C /ort --strip-components=1
RUN curl -fsSL -o /tmp/tok.tgz \
      "https://github.com/daulet/tokenizers/releases/download/v${TOK_VERSION}/libtokenizers.linux-amd64.tar.gz" \
    && mkdir -p /tok && tar xzf /tmp/tok.tgz -C /tok
RUN curl -fsSL -o /tmp/llama.tgz \
      "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_VERSION}/llama-${LLAMA_VERSION}-bin-ubuntu-x64.tar.gz" \
    && mkdir -p /llama-dist /llama && tar xzf /tmp/llama.tgz -C /llama-dist --strip-components=1 \
    && find /llama-dist -name '*.so*' -exec cp -a {} /llama/ \; \
    && find /llama-dist -name 'llama-server' -exec cp -a {} /llama/ \;
RUN --mount=type=secret,id=hf_token \
    mkdir -p /localmodel \
    && fetch() { \
         if [ -s /run/secrets/hf_token ]; then \
           curl -fsSL -H "Authorization: Bearer $(cat /run/secrets/hf_token)" -o "$1" "$2"; \
         else \
           curl -fsSL -o "$1" "$2"; \
         fi; \
       } \
    && if [ -n "${LOCAL_MODEL_URL}" ]; then \
         fetch /localmodel/minicpm5-yassai.gguf "${LOCAL_MODEL_URL}"; \
       fi \
    && if [ -n "${LOCAL_MODEL_LORA_URL}" ]; then \
         fetch /localmodel/tool-lora.gguf "${LOCAL_MODEL_LORA_URL}"; \
       fi \
    && if [ -n "${LOCAL_BASE_MODEL_URL}" ]; then \
         fetch /localmodel/assist-base.gguf "${LOCAL_BASE_MODEL_URL}"; \
       fi \
    && if [ -n "${LOCAL_BASE_LORA_URL}" ]; then \
         fetch /localmodel/assist-lora.gguf "${LOCAL_BASE_LORA_URL}"; \
       fi

# --- build the agent (cgo ON: both bindings are cgo; build on the target platform) ---
FROM golang:1.26-bookworm AS build
WORKDIR /src
COPY go.mod go.sum* ./
RUN go mod download
COPY . .
COPY --from=deps /tok/libtokenizers.a /libtok/libtokenizers.a
RUN CGO_ENABLED=1 CGO_LDFLAGS="-L/libtok" go build -trimpath -ldflags='-s -w' -o /out/yassai ./cmd/agent

# --- runtime: python-slim provides python3 for the native run_python tool.
# glibc + libstdc++ support the cgo binary and libonnxruntime; ca-certificates
# cover Fireworks HTTPS. Stays far under the 10GB limit. ---
FROM python:3.12-slim
# libgomp1: the llama.cpp CPU backend (libggml-cpu-*.so) links OpenMP; without
# it the dlopen fails and the local model silently disables.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /
COPY --from=build /out/yassai /yassai
COPY --from=deps /ort/lib/libonnxruntime.so* /opt/ort/
COPY --from=deps /llama/ /opt/llama/
COPY --from=deps /localmodel/ /assets/localmodel/
COPY assets/taskclf/ /assets/taskclf/
# LOCAL_MODEL_PATH points at the baked GGUF; if the build arg was empty the
# file is absent and the agent logs 'local model disabled' and continues.
# LOCAL_BASE_EXTENDED=1 only when LOCAL_BASE_MODEL_URL is the assist
# fine-tune: it unlocks sentiment/summarisation/factual in the base lane,
# which the un-tuned base model would fail.
ARG LOCAL_BASE_EXTENDED=""
# AGENT_NO_REMOTE=1 = zero-token mode: no Fireworks calls ever; gate rejects
# ship the best local answer. Rank is tokens ascending, so matching the
# 0-token leaders requires exactly this. AGENT_ALLOW_SEMANTIC_CLUES=1 lets the
# tool lane attempt world-knowledge clues (toolv3 weights trained on them).
ARG AGENT_NO_REMOTE=""
ARG AGENT_ALLOW_SEMANTIC_CLUES=""
ENV ONNXRUNTIME_LIB=/opt/ort/libonnxruntime.so \
    TASKCLF_DIR=/assets/taskclf \
    YZMA_LIB=/opt/llama \
    LOCAL_MODEL_PATH=/assets/localmodel/minicpm5-yassai.gguf \
    LOCAL_MODEL_LORA_PATH=/assets/localmodel/tool-lora.gguf \
    LOCAL_BASE_MODEL_PATH=/assets/localmodel/assist-base.gguf \
    LOCAL_BASE_LORA_PATH=/assets/localmodel/assist-lora.gguf \
    LOCAL_BASE_EXTENDED=${LOCAL_BASE_EXTENDED} \
    AGENT_NO_REMOTE=${AGENT_NO_REMOTE} \
    AGENT_ALLOW_SEMANTIC_CLUES=${AGENT_ALLOW_SEMANTIC_CLUES}
ENTRYPOINT ["/yassai"]
