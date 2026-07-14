#!/bin/bash
# Automated build script for pegaflow Python package with embedded binary
# Usage: ./scripts/build-wheel.sh [--release]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_DIR="$PROJECT_ROOT/python"

# Parse arguments
RELEASE_FLAG=""
PROFILE="debug"
CARGO_FEATURES=""
if [[ "$1" == "--release" ]]; then
    RELEASE_FLAG="--release"
    PROFILE="release"
    shift
fi

# Remaining args are feature flags (e.g. --no-default-features --features cuda-13)
EXTRA_ARGS=("$@")

echo "==> Building binaries ($PROFILE mode)..."
cd "$PROJECT_ROOT"
cargo build $RELEASE_FLAG "${EXTRA_ARGS[@]}" -p pegaflow-py --bin pegaflow-server-py --bin pegaflow-metaserver-py

echo "==> Copying binaries to Python package..."
for bin in pegaflow-server-py pegaflow-metaserver-py; do
    cp "$PROJECT_ROOT/target/$PROFILE/$bin" "$PYTHON_DIR/pegaflow/$bin"
    chmod +x "$PYTHON_DIR/pegaflow/$bin"
done

echo "==> Building CANN IPC C extension..."
C_EXT_DIR="$PYTHON_DIR/pegaflow/npu_ipc_bindings"
if command -v python3 &> /dev/null; then
    (cd "$C_EXT_DIR" && python3 setup.py build_ext --inplace) || \
        echo "WARNING: C extension build failed; ctypes fallback will be used at runtime"
    # Clean build artifacts but keep the .so
    rm -rf "$C_EXT_DIR/build"
else
    echo "WARNING: python3 not found; skipping C extension build (ctypes fallback)"
fi

echo "==> Building Python wheel with maturin..."
cd "$PYTHON_DIR"
maturin build $RELEASE_FLAG "${EXTRA_ARGS[@]}"

echo ""
echo "==> Done! Wheel built at:"
ls -lh "$PROJECT_ROOT/target/wheels/"pegaflow-*.whl | tail -1
echo ""
echo "To install: pip install $(ls -t $PROJECT_ROOT/target/wheels/pegaflow-*.whl | head -1)"
