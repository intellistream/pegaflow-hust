#!/bin/bash
# ============================================================================
# Quick diagnostic: capture system state during vLLM TCP delay window
#
# This script:
#   1. Starts vLLM + pegaflow-server (same as run_vllm_e2e.sh)
#   2. Once vLLM socket is LISTEN, captures:
#      - Open fds of EngineCore process
#      - ss/netstat state of the server socket
#      - iptables/nftables rules
#      - Kernel network parameters
#      - BPF programs attached to both processes
#   3. Polls external connection every second until it succeeds
#   4. Reports the delay and captured diagnostics
# ============================================================================

set -euo pipefail

NPU_DEVICE="${1:-2}"
GRPC_PORT=50055
VLLM_PORT=8101
TAG="diag-$(date +%H%M%S)"
SERVER_LOG="/tmp/pf-diag-${TAG}.log"
VLLM_LOG="/tmp/vllm-diag-${TAG}.log"
DIAG_DIR="/tmp/vllm-diag-${TAG}"
MODEL="/root/.cache/modelscope/models/qwen--Qwen2.5-0.5B-Instruct/snapshots/master"

ASCEND_HOME="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.1}"
ASCEND_DRIVER="${ASCEND_HOME_PATH:+$(dirname "$ASCEND_HOME_PATH")/driver}/lib64/driver"
ASCEND_DRIVER="${ASCEND_DRIVER:-/usr/local/Ascend/driver/lib64/driver}"
ATB_LIB=$(ls -d /usr/local/Ascend/nnal/atb/*/atb/cxx_abi_1/lib 2>/dev/null | head -1)
ATB_LIB="${ATB_LIB:-/usr/local/Ascend/nnal/atb/8.5.1/atb/cxx_abi_1/lib}"
VENV_DIR="${PEGAFLOW_VENV:-/root/miniconda3/envs/vllm-hust-dev}"
PYTHON="${VENV_DIR}/bin/python"

export LD_LIBRARY_PATH="${VENV_DIR}/lib:${ASCEND_HOME}/lib64:${ASCEND_HOME}/aarch64-linux/lib64:${ASCEND_DRIVER}:${ATB_LIB}"
export ASCEND_VISIBLE_DEVICES="${NPU_DEVICE}"
export VLLM_PLUGINS=ascend
CARGO_TARGET="/workspace/pegaflow-hust/target/debug"

mkdir -p "${DIAG_DIR}"

echo "=== vLLM TCP Delay Diagnostic ==="
echo "Tag: ${TAG}"
echo "Diag dir: ${DIAG_DIR}"
echo ""

# ---- Cleanup ----
echo "[0/4] Cleaning up..."
kill $(ps aux | grep "pegaflow-server-py" | grep -v grep | awk '{print $2}') 2>/dev/null || true
kill $(ps aux | grep "vllm.entrypoints" | grep -v grep | awk '{print $2}') 2>/dev/null || true
kill $(ps aux | grep "EngineCore" | grep -v grep | awk '{print $2}') 2>/dev/null || true
sleep 2

# ---- Build ----
echo "[1/4] Building pegaflow..."
cd /workspace/pegaflow-hust
PYO3_PYTHON="${PYTHON}" cargo build --no-default-features --features ascend \
  -p pegaflow-py --bin pegaflow-server-py 2>&1 | tail -1
\cp -f "${CARGO_TARGET}/libpegaflow.so" \
  /workspace/pegaflow-hust/python/pegaflow/pegaflow.cpython-311-aarch64-linux-gnu.so

# ---- Start pegaflow-server ----
echo "[2/4] Starting pegaflow-server..."
nohup "${CARGO_TARGET}/pegaflow-server-py" \
  --addr "127.0.0.1:${GRPC_PORT}" \
  --devices "${NPU_DEVICE}" \
  --pool-size 2gb \
  --disable-numa-affinity \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:9091/health >/dev/null 2>&1; then
    echo "  pegaflow-server ready (${i}s)"
    break
  fi
  sleep 1
done

# ---- Start vLLM ----
echo "[3/4] Starting vLLM..."
nohup "${PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --port "${VLLM_PORT}" \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.1 \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"PegaKVConnector","kv_role":"kv_both","kv_connector_module_path":"pegaflow.connector"}' \
  > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

# ---- Wait for LISTEN state ----
echo "  Waiting for vLLM socket to enter LISTEN state..."
LISTEN_TIME=""
for i in $(seq 1 180); do
  if ss -tlnp 2>/dev/null | grep -q ":${VLLM_PORT}"; then
    LISTEN_TIME=$(date +%H:%M:%S)
    echo "  Socket LISTEN at ${LISTEN_TIME} (iter ${i})"
    break
  fi
  sleep 1
done

if [ -z "${LISTEN_TIME}" ]; then
  echo "FAIL: vLLM socket never entered LISTEN state"
  exit 1
fi

# ---- CAPTURE DIAGNOSTICS (during the stuck window) ----
echo ""
echo "=== CAPTURING DIAGNOSTICS (during stuck window) ==="

# Find EngineCore PID
ENGINE_PID=$(ps aux | grep "EngineCore" | grep -v grep | awk '{print $2}' | head -1)
echo "API Server PID: ${VLLM_PID}"
echo "EngineCore PID:  ${ENGINE_PID:-NONE}"

# 1. Socket state
echo ""
echo "--- ss -tlnp ---"
ss -tlnp 2>/dev/null | grep -E "${VLLM_PORT}|State" || echo "(no matches)"
echo ""
echo "--- ss -tanp (all tcp) ---"
ss -tanp 2>/dev/null | grep -E "${VLLM_PORT}|State|LISTEN" | head -20 || echo "(no matches)"

# 2. Open fds for BOTH processes
for proc_info in "${VLLM_PID}:API_Server" "${ENGINE_PID}:EngineCore"; do
  PID="${proc_info%%:*}"
  LABEL="${proc_info##*:}"
  if [ -n "${PID}" ] && [ -d "/proc/${PID}/fd" ]; then
    echo ""
    echo "--- /proc/${PID}/fd (${LABEL}) ---"
    SOCKET_FDS=$(ls -la /proc/${PID}/fd/ 2>/dev/null | grep socket | head -30)
    echo "${SOCKET_FDS}"

    # Check if server socket fd is present
    PORT_IN_FDS=$(ls -la /proc/${PID}/fd/ 2>/dev/null | grep "socket:.*${VLLM_PORT}" || echo "")
    if [ -n "${PORT_IN_FDS}" ]; then
      echo ">>> FOUND socket referencing port ${VLLM_PORT} in ${LABEL}!"
    fi
  fi
done

# 3. Open fds of ALL child processes
CHILD_PIDS=$(pgrep -P "${VLLM_PID}" 2>/dev/null || echo "")
if [ -n "${CHILD_PIDS}" ]; then
  echo ""
  echo "--- Child processes of API server (${VLLM_PID}) ---"
  for cpid in ${CHILD_PIDS}; do
    PROC_NAME=$(ps -p "${cpid}" -o comm= 2>/dev/null || echo "?")
    SOCKET_COUNT=$(ls /proc/${cpid}/fd/ 2>/dev/null | wc -l)
    echo "  PID ${cpid} (${PROC_NAME}): ${SOCKET_COUNT} fds"
    # Check each child for the server port
    HAS_PORT=$(ls -la /proc/${cpid}/fd/ 2>/dev/null | grep "socket:.*${VLLM_PORT}" || echo "")
    if [ -n "${HAS_PORT}" ]; then
      echo "  >>> PID ${cpid} HAS SERVER SOCKET (port ${VLLM_PORT})!"
    fi
  done
fi

# 4. Kernel network parameters
echo ""
echo "--- Kernel net params ---"
for param in \
  net.core.somaxconn \
  net.core.netdev_max_backlog \
  net.ipv4.tcp_max_syn_backlog \
  net.ipv4.tcp_tw_reuse \
  net.ipv4.tcp_timestamps \
  net.ipv4.tcp_syn_retries \
  net.ipv4.tcp_synack_retries \
  net.ipv4.tcp_abort_on_overflow \
  net.ipv4.ip_local_port_range \
  net.ipv4.tcp_fastopen; do
  val=$(sysctl -n "${param}" 2>/dev/null || echo "N/A")
  echo "  ${param} = ${val}"
done

# 5. iptables/nftables
echo ""
echo "--- iptables INPUT rules ---"
iptables -L INPUT -n -v 2>/dev/null | head -30 || echo "(iptables not available)"
echo ""
echo "--- nftables rules ---"
nft list ruleset 2>/dev/null | head -30 || echo "(nft not available)"

# 6. BPF/cgroup
echo ""
echo "--- BPF programs ---"
bpftool prog list 2>/dev/null | head -20 || echo "(bpftool not available)"
echo ""
echo "--- cgroup for API server ---"
if [ -f "/proc/${VLLM_PID}/cgroup" ]; then
  cat /proc/${VLLM_PID}/cgroup
else
  echo "(not available)"
fi

# 7. conntrack for the port
echo ""
echo "--- conntrack for port ${VLLM_PORT} ---"
conntrack -L -p tcp --dport "${VLLM_PORT}" 2>/dev/null | head -10 || echo "(conntrack not available)"

# 8. SELinux
echo ""
echo "--- SELinux ---"
getenforce 2>/dev/null || echo "(not available)"

# ---- External connect polling ----
echo ""
echo "=== EXTERNAL CONNECT POLLING (from ${LISTEN_TIME}) ==="
echo "Polling every 2s until success or 600s timeout..."

START_TS=$(date +%s)
SUCCESS_TS=""
for i in $(seq 1 300); do
  ELAPSED=$(($(date +%s) - START_TS))

  # Use Python subprocess for the external connect attempt
  if python3 -c "
import socket, sys
s = socket.socket()
s.settimeout(3)
try:
    s.connect(('127.0.0.1', ${VLLM_PORT}))
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
    SUCCESS_TS=$(date +%H:%M:%S)
    echo ""
    echo ">>> EXTERNAL CONNECT SUCCEEDED at ${SUCCESS_TS} (after ${ELAPSED}s)"
    break
  fi

  # Show progress every 30s
  if [ $((i % 15)) -eq 0 ]; then
    echo "  Still failing at ${ELAPSED}s..."

    # Re-check socket state
    echo "  Socket state:"
    ss -tlnp 2>/dev/null | grep ":${VLLM_PORT}" | head -3

    # Re-check if engine core is alive
    if [ -n "${ENGINE_PID}" ] && ! kill -0 "${ENGINE_PID}" 2>/dev/null; then
      echo "  >>> EngineCore DIED! PID ${ENGINE_PID} is gone"
    fi
  fi

  sleep 2
done

if [ -z "${SUCCESS_TS}" ]; then
  echo ""
  echo ">>> EXTERNAL CONNECT NEVER SUCCEEDED (timed out after ${ELAPSED}s)"
fi

# ---- Save all output ----
echo ""
echo "=== LOG TAILS ==="
echo "--- vLLM log (last 50 lines) ---"
tail -50 "${VLLM_LOG}" 2>/dev/null || echo "(empty)"

echo ""
echo "--- pegaflow-server log (last 20 lines) ---"
tail -20 "${SERVER_LOG}" 2>/dev/null || echo "(empty)"

# ---- Cleanup ----
echo ""
echo "=== CLEANUP ==="
kill ${VLLM_PID} 2>/dev/null || true
kill ${SERVER_PID} 2>/dev/null || true
sleep 2

echo ""
echo "Diagnostic complete. Logs at:"
echo "  vLLM:  ${VLLM_LOG}"
echo "  Server: ${SERVER_LOG}"
echo "  Dir:    ${DIAG_DIR}"
