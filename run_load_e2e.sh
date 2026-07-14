#!/bin/bash
# ============================================================================
# PegaFlow + vLLM Load (H2D) 端到端验证脚本
#
# 验证策略:
#   vLLM Session 1: 发送请求 → KV cache save 到 PegaFlow → 关闭 vLLM
#   vLLM Session 2: 全新启动(本地 cache 为空) → 发送相同 prefix 请求
#                   → 必然查询 PegaFlow 外部 cache → 触发 H2D LOAD
#
# 用法:
#   ./run_load_e2e.sh [NPU_DEVICE_ID]
# ============================================================================

set -euo pipefail

NPU_DEVICE="${1:-2}"
GRPC_PORT=50059
VLLM_PORT_1=8101
VLLM_PORT_2=8102
TAG="load-e2e-$(date +%H%M%S)"
SERVER_LOG="/tmp/pf-load-${TAG}.log"
VLLM1_LOG="/tmp/vllm-s1-${TAG}.log"
VLLM2_LOG="/tmp/vllm-s2-${TAG}.log"
MODEL="/root/.cache/modelscope/models/qwen--Qwen2.5-0.5B-Instruct/snapshots/master"

export LD_LIBRARY_PATH="\
/root/miniconda3/envs/vllm-hust-dev/lib:\
/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64:\
/usr/local/Ascend/driver/lib64/driver:\
/usr/local/Ascend/nnal/atb/8.5.1/atb/cxx_abi_1/lib"
export ASCEND_VISIBLE_DEVICES="${NPU_DEVICE}"
export VLLM_PLUGINS=ascend
export PYTHONHASHSEED=0
PYTHON="/root/miniconda3/envs/vllm-hust-dev/bin/python"
CARGO_TARGET="/workspace/pegaflow-hust/target/debug"

# Must be longer than 256 tokens (2 full blocks at block_size=128) so
# at least one full block (tokens 128-255) is pure shared prefix.
SHARED="The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Portugal is Lisbon. The capital of Japan is Tokyo. The capital of China is Beijing. The capital of India is New Delhi. The capital of Brazil is Brasilia. The capital of Canada is Ottawa. The capital of Australia is Canberra. The capital of Russia is Moscow. The capital of Egypt is Cairo. The capital of South Korea is Seoul. The capital of Mexico is Mexico City. The capital of Argentina is Buenos Aires. The capital of Turkey is Ankara. The capital of Indonesia is Jakarta. The capital of Nigeria is Abuja. The capital of Kenya is Nairobi. The capital of South Africa is Pretoria. The capital of Sweden is Stockholm. The capital of Norway is Oslo. The capital of Denmark is Copenhagen. The capital of Finland is Helsinki. The capital of Poland is Warsaw. The capital of Ukraine is Kyiv. The capital of Greece is Athens. The capital of Thailand is Bangkok. The capital of Vietnam is Hanoi. The capital of Malaysia is Kuala Lumpur. The capital of Philippines is Manila. The capital of Iran is Tehran. The capital of Iraq is Baghdad. The capital of Saudi Arabia is Riyadh. The capital of Chile is Santiago. The capital of Peru is Lima."

echo "============================================================"
echo " PegaFlow + vLLM Cross-Session Load (H2D) E2E Test"
echo " NPU: ${NPU_DEVICE}  gRPC: ${GRPC_PORT}"
echo " Model: Qwen2.5-0.5B-Instruct"
echo " Tag: ${TAG}"
echo "============================================================"

# ---- Cleanup ----
echo "[0/7] Cleaning up stale processes..."
kill $(ps aux | grep "pegaflow-server-py"  | grep -v grep | awk '{print $2}') 2>/dev/null || true
kill $(ps aux | grep "vllm.entrypoints"     | grep -v grep | awk '{print $2}') 2>/dev/null || true
kill $(ps aux | grep "EngineCore"           | grep -v grep | awk '{print $2}') 2>/dev/null || true
sleep 2

# ---- Build ----
echo "[1/7] Building pegaflow..."
cd /workspace/pegaflow-hust
PYO3_PYTHON="${PYTHON}" cargo build --no-default-features --features ascend \
  -p pegaflow-py --bin pegaflow-server-py 2>&1 | tail -1
cp -f "${CARGO_TARGET}/libpegaflow.so" \
  /workspace/pegaflow-hust/python/pegaflow/pegaflow.cpython-311-aarch64-linux-gnu.so
echo "  Build OK"

# ---- Start pegaflow-server (shared across both vLLM sessions) ----
echo "[2/7] Starting pegaflow-server..."
nohup "${CARGO_TARGET}/pegaflow-server-py" \
  --addr "127.0.0.1:${GRPC_PORT}" \
  --devices "${NPU_DEVICE}" \
  --pool-size 2gb \
  --disable-numa-affinity \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:9091/health >/dev/null 2>&1; then
    echo "  Server ready (${i}s)"
    break
  fi
  if ! kill -0 ${SERVER_PID} 2>/dev/null; then
    echo "  FAIL: Server crashed."; tail -20 "${SERVER_LOG}"; exit 1
  fi
  sleep 1
done

# ==============================
# vLLM Session 1: Save to PegaFlow
# ==============================
echo "[3/7] vLLM Session 1 — starting (SAVE)..."
nohup "${PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --port "${VLLM_PORT_1}" \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.1 \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"PegaKVConnector","kv_role":"kv_both","kv_connector_module_path":"pegaflow.connector","kv_connector_extra_config":{"pegaflow.port":'"${GRPC_PORT}"'}}' \
  > "${VLLM1_LOG}" 2>&1 &
VLLM1_PID=$!

for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${VLLM_PORT_1}/health" >/dev/null 2>&1; then
    echo "  vLLM-1 ready (${i}s)"
    break
  fi
  if ! kill -0 ${VLLM1_PID} 2>/dev/null; then
    echo "  FAIL: vLLM-1 crashed."
    grep -E "error|Error|panic" "${VLLM1_LOG}" | tail -5
    kill ${SERVER_PID} 2>/dev/null || true
    exit 1
  fi
  sleep 3
done

echo "  S1: Sending request to populate cache..."
"${PYTHON}" -c "
import urllib.request, json
data = json.dumps({
    'model': '${MODEL}',
    'messages': [{'role': 'user', 'content': '${SHARED}. What are the major rivers in each country?'}],
    'max_tokens': 32
}).encode()
req = urllib.request.Request('http://127.0.0.1:${VLLM_PORT_1}/v1/chat/completions',
    data=data, headers={'Content-Type': 'application/json'})
d = json.loads(urllib.request.urlopen(req, timeout=120).read())
if 'error' in d: print(f'ERROR: {d[\"error\"]}')
else: print(f'    OK tokens={d[\"usage\"][\"prompt_tokens\"]}/{d[\"usage\"][\"completion_tokens\"]}')
" 2>&1

# Wait for async save to complete then check cache
sleep 3
echo "  Waiting for async save to land..."
for i in $(seq 1 20); do
  INS=$("${PYTHON}" -c "
import urllib.request
r = urllib.request.urlopen('http://127.0.0.1:9091/metrics')
for l in r.read().decode().split('\n'):
    l = l.strip()
    if 'pegaflow_cache_block_insertions' in l and not l.startswith('#'):
        print(l.split()[-1])
" 2>/dev/null)
  if [ -n "${INS}" ] && [ "${INS}" -ge 1 ]; then
    echo "    ${INS} blocks cached (after ${i}s)"
    break
  fi
  sleep 1
done

echo "  S1 PegaFlow metrics:"
"${PYTHON}" -c "
import urllib.request
r = urllib.request.urlopen('http://127.0.0.1:9091/metrics')
for l in r.read().decode().split('\n'):
    l = l.strip()
    if 'rpc_requests_total{method=\"save\"' in l and 'ok' in l and not l.startswith('#'):
        print(f'    {l}')
    if 'save_bytes_total' in l and not l.startswith('#'):
        print(f'    {l}')
" 2>&1

# ---- Shutdown vLLM-1 ----
echo "  Shutting down vLLM-1..."
kill ${VLLM1_PID} 2>/dev/null || true
sleep 3
# Clean up any orphan EngineCore processes from session 1
kill $(ps aux | grep "EngineCore" | grep -v grep | awk '{print $2}') 2>/dev/null || true
sleep 2
echo "  vLLM-1 down"

# ==============================
# vLLM Session 2: LOAD from PegaFlow
# ==============================
echo "[4/7] vLLM Session 2 — starting (fresh local cache → expect LOAD)..."
nohup "${PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --port "${VLLM_PORT_2}" \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.1 \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"PegaKVConnector","kv_role":"kv_both","kv_connector_module_path":"pegaflow.connector","kv_connector_extra_config":{"pegaflow.port":'"${GRPC_PORT}"'}}' \
  > "${VLLM2_LOG}" 2>&1 &
VLLM2_PID=$!

for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${VLLM_PORT_2}/health" >/dev/null 2>&1; then
    echo "  vLLM-2 ready (${i}s)"
    break
  fi
  if ! kill -0 ${VLLM2_PID} 2>/dev/null; then
    echo "  FAIL: vLLM-2 crashed."
    grep -E "error|Error|panic" "${VLLM2_LOG}" | tail -5
    kill ${SERVER_PID} 2>/dev/null || true
    exit 1
  fi
  sleep 3
done

echo "  S2-1: Same prefix, different suffix → expect cache HIT + LOAD..."
"${PYTHON}" -c "
import urllib.request, json
data = json.dumps({
    'model': '${MODEL}',
    'messages': [{'role': 'user', 'content': '${SHARED}. What is the population of each country?'}],
    'max_tokens': 32
}).encode()
req = urllib.request.Request('http://127.0.0.1:${VLLM_PORT_2}/v1/chat/completions',
    data=data, headers={'Content-Type': 'application/json'})
d = json.loads(urllib.request.urlopen(req, timeout=120).read())
if 'error' in d: print(f'ERROR: {d[\"error\"]}')
else: print(f'    OK tokens={d[\"usage\"][\"prompt_tokens\"]}/{d[\"usage\"][\"completion_tokens\"]}')
" 2>&1

sleep 3

echo "  S2-2: Same prefix again → confirm consistency..."
"${PYTHON}" -c "
import urllib.request, json
data = json.dumps({
    'model': '${MODEL}',
    'messages': [{'role': 'user', 'content': '${SHARED}. List the primary languages spoken in each country.'}],
    'max_tokens': 32
}).encode()
req = urllib.request.Request('http://127.0.0.1:${VLLM_PORT_2}/v1/chat/completions',
    data=data, headers={'Content-Type': 'application/json'})
d = json.loads(urllib.request.urlopen(req, timeout=120).read())
if 'error' in d: print(f'ERROR: {d[\"error\"]}')
else: print(f'    OK tokens={d[\"usage\"][\"prompt_tokens\"]}/{d[\"usage\"][\"completion_tokens\"]}')
" 2>&1

# ---- vLLM-2 KV Transfer metrics ----
echo "[5/7] vLLM-2 KV Transfer metrics..."
"${PYTHON}" -c "
import urllib.request
r = urllib.request.urlopen('http://127.0.0.1:${VLLM_PORT_2}/metrics')
lines = r.read().decode().split('\n')
for l in lines:
    l = l.strip()
    if 'vllm:kv_transfer' in l and not l.startswith('#'):
        print(f'  {l}')
" 2>&1

# ---- S2 cache lookup logs ----
echo "  S2 cache lookup:"
grep "cache_lookup" "${VLLM2_LOG}" 2>/dev/null | while read -r line; do
  echo "    ${line}"
done

# ---- PegaFlow final metrics ----
echo "[6/7] PegaFlow final metrics..."
"${PYTHON}" -c "
import urllib.request
r = urllib.request.urlopen('http://127.0.0.1:9091/metrics')
lines = r.read().decode().split('\n')
for l in lines:
    l = l.strip()
    if 'rpc_requests_total{method=\"register\"' in l and 'ok' in l and not l.startswith('#'):
        print(f'  REG:   {l}')
    if 'rpc_requests_total{method=\"save\"' in l and 'ok' in l and not l.startswith('#'):
        print(f'  SAVE:  {l}')
    if 'rpc_requests_total{method=\"load\"' in l and 'ok' in l and not l.startswith('#'):
        print(f'  LOAD:  {l}')
    if 'save_bytes_total' in l and not l.startswith('#'):
        print(f'  SAVE_BYTES: {l}')
    if 'load_bytes_total' in l and not l.startswith('#'):
        print(f'  LOAD_BYTES: {l}')
    if 'cache_block_insertions' in l and not l.startswith('#'):
        print(f'  CACHE_BLOCKS: {l}')
    if 'pegaflow_cache_resident_bytes' in l and not l.startswith('#'):
        print(f'  RESIDENT: {l}')
" 2>&1

# ---- PegaFlow prefetch logs ----
echo "  PegaFlow prefetch:"
grep "Prefetch local" "${SERVER_LOG}" 2>/dev/null | while read -r line; do
  echo "    ${line}"
done

# ---- Cleanup ----
echo ""
echo "[7/7] Stopping services..."
kill ${VLLM2_PID} 2>/dev/null || true
kill ${SERVER_PID} 2>/dev/null || true
sleep 2

echo ""
echo "============================================================"
echo " Cross-Session Load (H2D) E2E Test Complete"
echo " Tag: ${TAG}"
echo ""
echo " Expected: S1 saves → PegaFlow cache"
echo "           S2 fresh vLLM → external cache HIT → H2D LOAD"
echo " Logs: ${SERVER_LOG} / ${VLLM1_LOG} / ${VLLM2_LOG}"
echo "============================================================"
