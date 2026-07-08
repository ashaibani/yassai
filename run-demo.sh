#!/bin/bash
# yassai demo server launcher
# Downloads native libs if needed and starts the demo web app.

set -e
cd "$(dirname "$0")"

# Load env
source ~/config/.env 2>/dev/null || true

# Download libtokenizers if missing
TOK_DIR=/tmp/libtok
if [ ! -f "$TOK_DIR/libtokenizers.a" ]; then
  echo "Downloading libtokenizers (macOS arm64)..."
  mkdir -p "$TOK_DIR"
  curl -fsSL "https://github.com/daulet/tokenizers/releases/download/v1.27.0/libtokenizers.darwin-arm64.tar.gz" | tar xz -C "$TOK_DIR"
fi

# Download onnxruntime if missing
ORT_DIR=/tmp/ort127/onnxruntime-osx-arm64-1.27.0
if [ ! -f "$ORT_DIR/lib/libonnxruntime.dylib" ]; then
  echo "Downloading onnxruntime 1.27.0 (macOS arm64)..."
  mkdir -p /tmp/ort127
  curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v1.27.0/onnxruntime-osx-arm64-1.27.0.tgz" | tar xz -C /tmp/ort127
fi

export ONNXRUNTIME_LIB="$ORT_DIR/lib/libonnxruntime.dylib"
export TASKCLF_DIR=assets/taskclf
export PORT=7070

echo "Starting yassai demo on http://localhost:$PORT"
CGO_ENABLED=1 CGO_LDFLAGS="-L$TOK_DIR" go run ./cmd/demo/
