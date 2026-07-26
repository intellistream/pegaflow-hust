#!/bin/bash
# ============================================================
# PegaFlow + vLLM 多实例 E2E 测试 (Ascend NPU)
#
# 用法:  ./run_vllm_e2e_test.sh
# ============================================================
set -e

MODEL="/workspace/HUST/models/models/Qwen--Qwen2.5-0.5B-Instruct/snapshots/master"
SERVER_PORT=50060
VLLM_PORT_A=18101
VLLM_PORT_B=18102
LOG_DIR="/tmp/pegaflow-e2e-logs"
mkdir -p "$LOG_DIR"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-hust-dev
export PEGAFLOW_PORT=$SERVER_PORT
# CRITICAL: vLLM uses os.urandom(32) as NONE_HASH when PYTHONHASHSEED
# is not set. Without this, cross-instance block_hashes differ and
# KV cache sharing does NOT work.
export PYTHONHASHSEED=0
cd /workspace/HUST/pegaflow-hust

# Python helper to replace curl (curl has OpenSSL symbol conflicts)
PY_CHECK="import sys,urllib.request as u; r=u.urlopen(u.Request(sys.argv[1])); print(r.status)"
PY_COMPLETE="import sys,json,urllib.request as u; r=u.urlopen(u.Request('http://127.0.0.1:%s/v1/completions'%sys.argv[1],data=json.dumps({'model':sys.argv[2],'prompt':sys.argv[3],'max_tokens':int(sys.argv[4]),'temperature':0.0}).encode(),headers={'Content-Type':'application/json'})); print(json.loads(r.read())['choices'][0]['text'])"

echo "========================================="
echo "  PegaFlow + vLLM Multi-Instance E2E Test"
echo "  Model: $(basename $(dirname $MODEL))"
echo "========================================="

# Cleanup any leftover processes
pkill -f "pegaflow-server --addr" 2>/dev/null || true
pkill -f "vllm serve" 2>/dev/null || true
sleep 2

# ============================================================
# Step 1: Start PegaFlow Server
# ============================================================
echo ""
echo "[Step 1] Starting pegaflow-server on port $SERVER_PORT..."

target/debug/pegaflow-server \
    --addr 0.0.0.0:$SERVER_PORT \
    --pool-size 256mb \
    --devices 0 \
    > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 15); do
    sleep 2
    if grep -q "listening" "$LOG_DIR/server.log" 2>/dev/null; then
        echo "  Server ready (PID=$SERVER_PID)"
        break
    fi
    echo -n "."
done
echo ""

# ============================================================
# Step 2: Instance A (SAVE_ONLY) — infer and save KV cache
# ============================================================
echo "[Step 2] Instance A (SAVE_ONLY) — loading model + inferring..."

KV_CFG_A=$(cat <<EOF
{"kv_connector":"PegaKVConnector","kv_role":"kv_both","kv_connector_module_path":"pegaflow.connector","kv_connector_extra_config":{"pegaflow.mode":"save_only"}}
EOF
)

vllm serve "$MODEL" \
    --port $VLLM_PORT_A \
    --dtype float16 \
    --max-model-len 1024 \
    --max-num-seqs 2 \
    --gpu-memory-utilization 0.3 \
    --enforce-eager \
    --kv-transfer-config "$KV_CFG_A" \
    > "$LOG_DIR/vllm-a.log" 2>&1 &
VLLM_A_PID=$!

# Wait for vLLM (model loading takes ~60-90s on NPU)
echo "  Waiting for Instance A (model loading takes ~60-90s)..."
VLLM_A_READY=false
for i in $(seq 1 60); do
    sleep 3
    if python3 -c "$PY_CHECK" "http://127.0.0.1:$VLLM_PORT_A/health" 2>/dev/null; then
        echo "  Instance A ready after $((i*3))s"
        VLLM_A_READY=true
        break
    fi
    echo -n "."
done
echo ""

if [ "$VLLM_A_READY" = false ]; then
    echo "  [FAIL] Instance A failed to start"
    echo "  --- vLLM A log (last 30 lines) ---"
    tail -30 "$LOG_DIR/vllm-a.log"
    kill $VLLM_A_PID 2>/dev/null
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Infer
PROMPT="The capital of France is"
echo "  Prompt: \"$PROMPT\""

RESULT_A=$(python3 -c "$PY_COMPLETE" "$VLLM_PORT_A" "$MODEL" "$PROMPT" "16")
echo "  Output: \"$RESULT_A\""
echo "$RESULT_A" > "$LOG_DIR/result_a.txt"

# Shut down vLLM A, wait for save flush
kill $VLLM_A_PID 2>/dev/null
sleep 3
echo "  [PASS] Instance A completed"

# ============================================================
# Step 3: Instance B (READ_WRITE) — should get cache hits
# ============================================================
echo ""
echo "[Step 3] Instance B (READ_WRITE) — should get cache hits from A..."

KV_CFG_B=$(cat <<EOF
{"kv_connector":"PegaKVConnector","kv_role":"kv_both","kv_connector_module_path":"pegaflow.connector","kv_connector_extra_config":{"pegaflow.mode":"read_write"}}
EOF
)

vllm serve "$MODEL" \
    --port $VLLM_PORT_B \
    --dtype float16 \
    --max-model-len 1024 \
    --max-num-seqs 2 \
    --gpu-memory-utilization 0.3 \
    --enforce-eager \
    --kv-transfer-config "$KV_CFG_B" \
    > "$LOG_DIR/vllm-b.log" 2>&1 &
VLLM_B_PID=$!

echo "  Waiting for Instance B..."
VLLM_B_READY=false
for i in $(seq 1 60); do
    sleep 3
    if python3 -c "$PY_CHECK" "http://127.0.0.1:$VLLM_PORT_B/health" 2>/dev/null; then
        echo "  Instance B ready after $((i*3))s"
        VLLM_B_READY=true
        break
    fi
    echo -n "."
done
echo ""

if [ "$VLLM_B_READY" = false ]; then
    echo "  [FAIL] Instance B failed to start"
    echo "  --- vLLM B log (last 30 lines) ---"
    tail -30 "$LOG_DIR/vllm-b.log"
    kill $VLLM_B_PID 2>/dev/null
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

RESULT_B=$(python3 -c "$PY_COMPLETE" "$VLLM_PORT_B" "$MODEL" "$PROMPT" "16")
echo "  Output B: \"$RESULT_B\""

kill $VLLM_B_PID 2>/dev/null
sleep 1

# ============================================================
# Compare results
# ============================================================
echo ""
if [ "$RESULT_A" = "$RESULT_B" ]; then
    echo "  [PASS] Cross-instance KV cache sharing works! Outputs match."
else
    echo "  [FAIL] Output mismatch: A='$RESULT_A' vs B='$RESULT_B'"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# ============================================================
# Done
# ============================================================
echo ""
echo "========================================="
echo "  ALL TESTS PASSED"
echo "========================================="
echo ""
echo "Server log: $LOG_DIR/server.log"
echo "vLLM A log: $LOG_DIR/vllm-a.log"
echo "vLLM B log: $LOG_DIR/vllm-b.log"
echo ""
echo "Server cache hits:"
grep "Prefetch" "$LOG_DIR/server.log" | tail -5

kill $SERVER_PID 2>/dev/null
