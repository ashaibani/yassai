# syntax=docker/dockerfile:1
# Judging VM is linux/amd64. Build with: docker buildx build --platform linux/amd64 ...

# --- fetch native libs for the target platform ---
# onnxruntime: dlopen'd at runtime by yalue/onnxruntime_go (needs >=1.23 for ORT API 26).
# libtokenizers: statically linked at BUILD time by daulet/tokenizers (HF BPE tokeniser).
FROM --platform=linux/amd64 debian:bookworm-slim AS deps
ARG ORT_VERSION=1.27.0
ARG TOK_VERSION=1.27.0
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /tmp/ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
    && mkdir -p /ort && tar xzf /tmp/ort.tgz -C /ort --strip-components=1
RUN curl -fsSL -o /tmp/tok.tgz \
      "https://github.com/daulet/tokenizers/releases/download/v${TOK_VERSION}/libtokenizers.linux-amd64.tar.gz" \
    && mkdir -p /tok && tar xzf /tmp/tok.tgz -C /tok

# --- build the agent (cgo ON: both bindings are cgo; build on the target platform) ---
FROM --platform=linux/amd64 golang:1.26-bookworm AS build
WORKDIR /src
COPY go.mod go.sum* ./
RUN go mod download
COPY . .
COPY --from=deps /tok/libtokenizers.a /libtok/libtokenizers.a
RUN CGO_ENABLED=1 CGO_LDFLAGS="-L/libtok" go build -trimpath -ldflags='-s -w' -o /out/yassai ./cmd/agent

# --- runtime: debian-slim WITH a shell + python3 so the action space's
# sh.run / python3 code-execution path works (distroless has no shell). glibc +
# libstdc++ are present for the cgo binary + libonnxruntime; ca-certificates for
# web.fetch TLS. Stays ~150-200MB, far under the 10GB limit. ---
FROM --platform=linux/amd64 python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /
COPY --from=build /out/yassai /yassai
COPY --from=deps /ort/lib/libonnxruntime.so* /opt/ort/
COPY assets/taskclf/ /assets/taskclf/
ENV ONNXRUNTIME_LIB=/opt/ort/libonnxruntime.so \
    TASKCLF_DIR=/assets/taskclf
ENTRYPOINT ["/yassai"]
