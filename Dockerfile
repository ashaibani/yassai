# syntax=docker/dockerfile:1
# Judging VM is linux/amd64. Build with: docker buildx build --platform linux/amd64 ...

# --- fetch native libs for the target platform ---
# onnxruntime: dlopen'd at runtime by yalue/onnxruntime_go (needs >=1.23 for ORT API 26).
# libtokenizers: statically linked at BUILD time by daulet/tokenizers (HF BPE tokeniser).
FROM debian:bookworm-slim AS deps
ARG ORT_VERSION=1.27.0
ARG TOK_VERSION=1.27.0
# llama.cpp shared libs for the in-container MiniCPM5 offload (yzma dlopens
# them; purego, no cgo). Same pinned release the GGUF exporter uses.
ARG LLAMA_VERSION=b9620
# Optional: URL of the fine-tuned GGUF. Left empty the image builds without a
# local model and the agent silently runs Fireworks-only. Private Hugging Face
# URLs authenticate via the BuildKit secret `hf_token` (never an ARG - ARGs
# persist in image history).
ARG LOCAL_MODEL_URL=""
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
    && find /llama-dist -name '*.so*' -exec cp -a {} /llama/ \;
RUN --mount=type=secret,id=hf_token \
    mkdir -p /localmodel \
    && if [ -n "${LOCAL_MODEL_URL}" ]; then \
         if [ -s /run/secrets/hf_token ]; then \
           curl -fsSL -H "Authorization: Bearer $(cat /run/secrets/hf_token)" \
                -o /localmodel/minicpm5-yassai.gguf "${LOCAL_MODEL_URL}"; \
         else \
           curl -fsSL -o /localmodel/minicpm5-yassai.gguf "${LOCAL_MODEL_URL}"; \
         fi; \
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
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /
COPY --from=build /out/yassai /yassai
COPY --from=deps /ort/lib/libonnxruntime.so* /opt/ort/
COPY --from=deps /llama/ /opt/llama/
COPY --from=deps /localmodel/ /assets/localmodel/
COPY assets/taskclf/ /assets/taskclf/
# LOCAL_MODEL_PATH points at the baked GGUF; if the build arg was empty the
# file is absent and the agent logs 'local model disabled' and continues.
ENV ONNXRUNTIME_LIB=/opt/ort/libonnxruntime.so \
    TASKCLF_DIR=/assets/taskclf \
    YZMA_LIB=/opt/llama \
    LOCAL_MODEL_PATH=/assets/localmodel/minicpm5-yassai.gguf
ENTRYPOINT ["/yassai"]
