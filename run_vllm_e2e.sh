#!/bin/bash
# ============================================================================
# PegaFlow + vLLM 端到端集成验证脚本
#
# 验证 vLLM 与 PegaFlow connector 的完整集成:
#   1. 编译 + 启动 pegaflow-server
#   2. 启动 vLLM API server (加载 PegaKVConnector)
#   3. 发送请求触发 KV cache save
#   4. 检查 pegaflow 指标确认 save 成功
#
# 用法:
#   ./run_vllm_e2e.sh [NPU_DEVICE_ID]
#
# 需要: xxhash (pip install xxhash)
# ============================================================================

set -euo pipefail

NPU_DEVICE="${1:-2}"
GRPC_PORT=50055
VLLM_PORT=8100
TAG="vllm-e2e-$(date +%H%M%S)"
SERVER_LOG="/tmp/pf-server-${TAG}.log"
VLLM_LOG="/tmp/vllm-server-${TAG}.log"
MODEL="/root/.cache/modelscope/models/qwen--Qwen2.5-0.5B-Instruct/snapshots/master"

export LD_LIBRARY_PATH="\
/root/miniconda3/envs/vllm-hust-dev/lib:\
/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64:\
/usr/local/Ascend/driver/lib64/driver:\
/usr/local/Ascend/nnal/atb/8.5.1/atb/cxx_abi_1/lib"
export ASCEND_VISIBLE_DEVICES="${NPU_DEVICE}"
export VLLM_PLUGINS=ascend
PYTHON="/root/miniconda3/envs/vllm-hust-dev/bin/python"
CARGO_TARGET="/workspace/pegaflow-hust/target/debug"

echo "============================================================"
echo " PegaFlow + vLLM E2E Integration Test"
echo " NPU: ${NPU_DEVICE}  gRPC: ${GRPC_PORT}  vLLM: ${VLLM_PORT}"
echo " Model: Qwen2.5-0.5B-Instruct"
echo " Tag: ${TAG}"
echo "============================================================"

# ---- Cleanup ----
echo "[0/5] Cleaning up stale processes..."
kill $(ps aux | grep "pegaflow-server-py" | grep -v grep | awk '{print $2}') 2>/dev/null || true
kill $(ps aux | grep "vllm.entrypoints" | grep -v grep | awk '{print $2}') 2>/dev/null || true
kill $(ps aux | grep "EngineCore" | grep -v grep | awk '{print $2}') 2>/dev/null || true
sleep 2

# ---- Build ----
echo "[1/5] Building pegaflow..."
cd /workspace/pegaflow-hust
PYO3_PYTHON="${PYTHON}" cargo build --no-default-features --features ascend \
  -p pegaflow-py --bin pegaflow-server-py 2>&1 | tail -1
cp -f "${CARGO_TARGET}/libpegaflow.so" \
  /workspace/pegaflow-hust/python/pegaflow/pegaflow.cpython-311-aarch64-linux-gnu.so
echo "  Build OK"

# ---- Start pegaflow-server ----
echo "[2/5] Starting pegaflow-server..."
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

# ---- Start vLLM ----
echo "[3/5] Starting vLLM API server (model loading takes 30-90s)..."
nohup "${PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --port "${VLLM_PORT}" \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.1 \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"PegaKVConnector","kv_role":"kv_both","kv_connector_module_path":"pegaflow.connector"}' \
  > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo "  vLLM ready (${i}s)"
    break
  fi
  if ! kill -0 ${VLLM_PID} 2>/dev/null; then
    echo "  FAIL: vLLM crashed."
    grep -E "error|Error|panic|PegaKV|connect" "${VLLM_LOG}" | tail -10
    kill ${SERVER_PID} 2>/dev/null || true
    exit 1
  fi
  sleep 3
done

# ---- Send requests ----
echo "[4/5] Sending requests..."
LONG_PROMPT="Write a comprehensive essay about artificial intelligence. Cover the history from the 1950s to the present, major machine learning paradigms including supervised unsupervised and reinforcement learning, deep learning architectures like CNNs RNNs and Transformers, large language models, and ethical considerations. Provide detailed technical explanations and specific examples for each topic area discussed."

for i in 1 2 3; do
  echo -n "  Request ${i}: "1
  t0=$(date +%s%N)
  RESP=$("${PYTHON}" -c "
import urllib.request, json, time
data = json.dumps({
    'model': '${MODEL}',
    'messages': [{'role': 'user', 'content': '${LONG_PROMPT}'}],
    'max_tokens': 64
}).encode()
req = urllib.request.Request('http://127.0.0.1:${VLLM_PORT}/v1/chat/completions',
    data=data, headers={'Content-Type': 'application/json'})
resp = urllib.request.urlopen(req, timeout=120)
d = json.loads(resp.read())
if 'error' in d: print(f'ERROR: {d[\"error\"]}')
else: print(f'OK tokens={d[\"usage\"][\"prompt_tokens\"]}/{d[\"usage\"][\"completion_tokens\"]}')
" 2>&1)
  echo "${RESP}"
  sleep 3
done

# ---- Load test: shared prefix → prefix cache hit → H2D ----
echo "[5/6] Load E2E test (shared prefix >128 tokens → cache hit → H2D)..."
# IMPORTANT: the shared prefix must produce >= 256 tokens (2 full blocks at block_size=128)
# so BOTH block 0 and block 1 share identical hashes across requests.
# Each capital line is ~7 tokens — need ~37 unique lines for 256 tokens.
SHARED_PREFIX="The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Portugal is Lisbon. The capital of Japan is Tokyo. The capital of China is Beijing. The capital of India is New Delhi. The capital of Brazil is Brasilia. The capital of Canada is Ottawa. The capital of Australia is Canberra. The capital of Russia is Moscow. The capital of Egypt is Cairo. The capital of South Korea is Seoul. The capital of Mexico is Mexico City. The capital of Argentina is Buenos Aires. The capital of Turkey is Ankara. The capital of Indonesia is Jakarta. The capital of Nigeria is Abuja. The capital of Kenya is Nairobi. The capital of South Africa is Pretoria. The capital of Sweden is Stockholm. The capital of Norway is Oslo. The capital of Denmark is Copenhagen. The capital of Finland is Helsinki. The capital of Poland is Warsaw. The capital of Ukraine is Kyiv. The capital of Greece is Athens. The capital of Thailand is Bangkok. The capital of Vietnam is Hanoi. The capital of Malaysia is Kuala Lumpur. The capital of Philippines is Manila. The capital of Iran is Tehran. The capital of Iraq is Baghdad. The capital of Saudi Arabia is Riyadh. The capital of Chile is Santiago. The capital of Peru is Lima."

echo "  Load-1: writing prefix to cache..."
RESP1=$("${PYTHON}" -c "
import urllib.request, json
data = json.dumps({
    'model': '${MODEL}',
    'messages': [{'role': 'user', 'content': '${SHARED_PREFIX}. What are the major rivers in each country?'}],
    'max_tokens': 32
}).encode()
req = urllib.request.Request('http://127.0.0.1:${VLLM_PORT}/v1/chat/completions',
    data=data, headers={'Content-Type': 'application/json'})
d = json.loads(urllib.request.urlopen(req, timeout=120).read())
if 'error' in d: print(f'ERROR: {d[\"error\"]}')
else: print(f'OK tokens={d[\"usage\"][\"prompt_tokens\"]}/{d[\"usage\"][\"completion_tokens\"]}')
" 2>&1)
echo "    ${RESP1}"
sleep 2

# Wait for async save to land in PegaFlow cache before query.
echo "  Waiting for async save to complete (checking cache insertions)..."
for i in $(seq 1 30); do
  PENDING=$("${PYTHON}" -c "
import urllib.request
resp = urllib.request.urlopen('http://127.0.0.1:9091/metrics')
lines = resp.read().decode().split('\n')
for l in lines:
    l = l.strip()
    if 'pegaflow_cache_block_insertions' in l and not l.startswith('#'):
        print(l.split()[-1])
" 2>/dev/null)
  if [ -n "${PENDING}" ] && [ "${PENDING:-0}" -ge 1 ]; then
    echo "    Save complete (${PENDING} blocks in cache after ${i}s)"
    break
  fi
  sleep 1
done

echo "  Load-2: same prefix, different suffix → expect cache HIT + LOAD..."
RESP2=$("${PYTHON}" -c "
import urllib.request, json
data = json.dumps({
    'model': '${MODEL}',
    'messages': [{'role': 'user', 'content': '${SHARED_PREFIX}. What is the population of each country?'}],
    'max_tokens': 32
}).encode()
req = urllib.request.Request('http://127.0.0.1:${VLLM_PORT}/v1/chat/completions',
    data=data, headers={'Content-Type': 'application/json'})
d = json.loads(urllib.request.urlopen(req, timeout=120).read())
if 'error' in d: print(f'ERROR: {d[\"error\"]}')
else: print(f'OK tokens={d[\"usage\"][\"prompt_tokens\"]}/{d[\"usage\"][\"completion_tokens\"]}')
" 2>&1)
echo "    ${RESP2}"
sleep 2

echo "  Load-3: same prefix again → confirm multiple H2D works..."
RESP3=$("${PYTHON}" -c "
import urllib.request, json
data = json.dumps({
    'model': '${MODEL}',
    'messages': [{'role': 'user', 'content': '${SHARED_PREFIX}. List the primary languages spoken in each country.'}],
    'max_tokens': 32
}).encode()
req = urllib.request.Request('http://127.0.0.1:${VLLM_PORT}/v1/chat/completions',
    data=data, headers={'Content-Type': 'application/json'})
d = json.loads(urllib.request.urlopen(req, timeout=120).read())
if 'error' in d: print(f'ERROR: {d[\"error\"]}')
else: print(f'OK tokens={d[\"usage\"][\"prompt_tokens\"]}/{d[\"usage\"][\"completion_tokens\"]}')
" 2>&1)
echo "    ${RESP3}"

# ---- Check vLLM-side KV transfer metrics ----
echo "  vLLM KV Transfer metrics:"
"${PYTHON}" -c "
import urllib.request, json
resp = urllib.request.urlopen('http://127.0.0.1:${VLLM_PORT}/metrics')
lines = resp.read().decode().split('\n')
for l in lines:
    l = l.strip()
    if 'vllm:kv_transfer' in l and not l.startswith('#'):
        print(f'    {l}')
" 2>&1

# ---- Check PegaFlow metrics ----
echo "[6/6] Checking PegaFlow metrics..."
"${PYTHON}" -c "
import urllib.request
resp = urllib.request.urlopen('http://127.0.0.1:9091/metrics')
lines = resp.read().decode().split('\n')
for l in lines:
    l = l.strip()
    if 'rpc_requests_total{method=\"register\"' in l and 'ok' in l: print(f'  REG:   {l}')
    if 'rpc_requests_total{method=\"save\"' in l and 'ok' in l: print(f'  SAVE:  {l}')
    if 'rpc_requests_total{method=\"load\"' in l and 'ok' in l: print(f'  LOAD:  {l}')
    if 'save_bytes_total' in l and not l.startswith('#'): print(f'  SAVE_BYTES: {l}')
    if 'load_bytes_total' in l and not l.startswith('#'): print(f'  LOAD_BYTES: {l}')
    if 'save_duration_seconds_count' in l and not l.startswith('#'): print(f'  SAVE_OPS: {l}')
    if 'load_duration_seconds_count' in l and not l.startswith('#'): print(f'  LOAD_OPS: {l}')
    if 'cache_block_insertions' in l and not l.startswith('#'): print(f'  CACHE:   {l}')
    if 'pegaflow_cache_resident_bytes' in l and not l.startswith('#'): print(f'  RESIDENT:{l}')
"

# ---- Cleanup ----
echo ""
echo "Stopping services..."
kill ${VLLM_PID} 2>/dev/null || true
kill ${SERVER_PID} 2>/dev/null || true
sleep 2

# ---- Summary ----
echo ""
echo "============================================================"
echo " vLLM + PegaFlow E2E Test Complete"
echo " Tag: ${TAG}"
echo ""
echo " Expected: 3 saves + 3 saves (load test) = 6 saves"
echo "           load test step 2,3 should trigger H2D via cache hit"
echo " Logs: ${SERVER_LOG} / ${VLLM_LOG}"
echo "============================================================"
