#!/bin/bash
# ============================================================================
# PegaFlow Ascend E2E 端到端验证脚本
#
# 验证完整的 KV Cache Save/Load 链路:
#   1. 编译 pegaflow (Ascend feature)
#   2. 启动 pegaflow-server
#   3. Python E2E: gRPC register → save(D2H) → query → load(H2D) → verify
#
# 用法:
#   ./run_e2e_test.sh [NPU_DEVICE_ID]
#
# 默认使用 NPU device 5 (可通过参数指定其他 device)
# ============================================================================

set -euo pipefail

# ---- 配置 ----
NPU_DEVICE="${1:-5}"
GRPC_PORT=50055
HTTP_PORT=9091
TAG="e2e-$(date +%H%M%S)"
SERVER_LOG="/tmp/pegaflow-server-${TAG}.log"

# ---- 环境变量 ----
export LD_LIBRARY_PATH="\
/root/miniconda3/envs/vllm-hust-dev/lib:\
/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64:\
/usr/local/Ascend/driver/lib64/driver:\
/usr/local/Ascend/nnal/atb/8.5.1/atb/cxx_abi_1/lib"
export ASCEND_VISIBLE_DEVICES="${NPU_DEVICE}"
export PYTHONPATH="/root/miniconda3/envs/vllm-hust-dev/lib/python3.11/site-packages"
PYTHON="/root/miniconda3/envs/vllm-hust-dev/bin/python"
CARGO_TARGET="/workspace/pegaflow-hust/target/debug"
PYO3_SRC="${CARGO_TARGET}/libpegaflow.so"
PYO3_DST="/workspace/pegaflow-hust/python/pegaflow/pegaflow.cpython-311-aarch64-linux-gnu.so"

echo "============================================================"
echo " PegaFlow Ascend E2E Test"
echo " NPU Device: ${NPU_DEVICE}   Port: gRPC=${GRPC_PORT} HTTP=${HTTP_PORT}"
echo " Tag: ${TAG}"
echo "============================================================"

# ---- Step 0: Kill stale processes ----
echo "[0/4] Cleaning up..."
kill $(ps aux | grep "pegaflow-server-py" | grep -v grep | awk '{print $2}') 2>/dev/null || true
sleep 2

# ---- Step 1: Build ----
echo "[1/4] Building pegaflow..."
cd /workspace/pegaflow-hust
PYO3_PYTHON="${PYTHON}" cargo build --no-default-features --features ascend \
  -p pegaflow-py --bin pegaflow-server-py 2>&1 | tail -1
cp -f "${PYO3_SRC}" "${PYO3_DST}"
echo "  Build OK"

# ---- Step 2: Start server ----
echo "[2/4] Starting pegaflow-server..."
nohup "${CARGO_TARGET}/pegaflow-server-py" \
  --addr "127.0.0.1:${GRPC_PORT}" \
  --http-addr "127.0.0.1:${HTTP_PORT}" \
  --devices "${NPU_DEVICE}" \
  --pool-size 2gb \
  --disable-numa-affinity \
  > "${SERVER_LOG}" 2>&1 &

SERVER_PID=$!
echo "  PID=${SERVER_PID}"

# Wait for server to be ready (health check)
for i in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:${HTTP_PORT}/health" > /dev/null 2>&1; then
    echo "  Server ready (${i}s)"
    break
  fi
  if ! kill -0 ${SERVER_PID} 2>/dev/null; then
    echo "  FAIL: Server crashed. Log:"
    tail -20 "${SERVER_LOG}"
    exit 1
  fi
  sleep 1
done

# ---- Step 3: Run E2E test ----
echo "[3/4] Running E2E test (register → save → query → load → verify)..."
E2E_OUTPUT=$("${PYTHON}" -c "
import hashlib, pickle, time, torch, urllib.request
from pegaflow.npu_ipc_wrapper import NpuIPCWrapper
from pegaflow.pegaflow import EngineRpcClient, PyLoadState

dev = 'npu:${NPU_DEVICE}'
N, T, H, D = 8, 128, 8, 128
BS = T * H * D * 2  # ~256 KB per block
GRPC = '${GRPC_PORT}'

# Create KV tensors
k = torch.empty(N, T, H, D, dtype=torch.float16, device=dev); k.normal_()
v = torch.empty(N, T, H, D, dtype=torch.float16, device=dev); v.normal_()
ko, vo = k.clone(), v.clone()

# Register via IPC
kw, vw = NpuIPCWrapper(k), NpuIPCWrapper(v)
c = EngineRpcClient(f'http://127.0.0.1:{GRPC}')
ok, _ = c.register_context_batch(
    'e2e-inst', 'e2e-ns', 0, 0, 1, 1, ${NPU_DEVICE},
    ['k', 'v'],
    [pickle.dumps(kw), pickle.dumps(vw)],
    [N, N], [BS, BS], [0, 0], [1, 1],
    'direct', False)
assert ok, f'register failed'

# Hash blocks (K+V combined for multi-slot storage)
torch.npu.synchronize(dev)
S = 4  # save 4 blocks
h = [hashlib.blake2b(
    k[i].contiguous().cpu().view(torch.uint8).numpy().tobytes() +
    v[i].contiguous().cpu().view(torch.uint8).numpy().tobytes(),
    digest_size=32).digest() for i in range(S)]

# Save
ok, _ = c.save('e2e-inst', 0, 0, ${NPU_DEVICE},
              [('k', list(range(S)), h), ('v', list(range(S)), h)])
assert ok, 'save failed'
time.sleep(1)

# Query
r = c.query_prefetch('e2e-inst', h, 'e2e-req')
hit = getattr(r, 'num_hit_blocks', 0)
assert hit == S, f'cache miss: {hit}/{S}'

# Corrupt tensors, then load back
k.zero_(); v.zero_(); torch.npu.synchronize(dev)
ls = PyLoadState()
c.load('e2e-inst', 0, ${NPU_DEVICE}, ls.shm_name(),
       ['k', 'v'], [(r.lease, list(range(S)))])
for _ in range(150):
    if ls.is_ready(): break
    time.sleep(0.2)
assert ls.get_state() >= 0, f'load failed: state={ls.get_state()}'

# Verify data
torch.npu.synchronize(dev)
km = torch.allclose(k[:S], ko[:S])
vm = torch.allclose(v[:S], vo[:S])
assert km and vm, f'data mismatch: K={km} V={vm}'

# Check metrics
resp = urllib.request.urlopen(f'http://127.0.0.1:{GRPC}/metrics').read().decode()
save_bytes = [l for l in resp.split(chr(10)) if 'pegaflow_save_bytes_total' in l and not l.startswith('#')]
load_req  = [l for l in resp.split(chr(10)) if 'rpc_requests_total{method=\"load\"' in l]
query_req = [l for l in resp.split(chr(10)) if 'rpc_requests_total{method=\"query_prefetch\"' in l]

print(f'PASS|save={S*2*BS}|load_state={ls.get_state()}|metrics_save={save_bytes[0].split()[-1] if save_bytes else \"?\"}')
" 2>&1)

if echo "${E2E_OUTPUT}" | grep -q "^PASS"; then
  SAVE_BYTES=$(echo "${E2E_OUTPUT}" | sed 's/.*save=\([0-9]*\).*/\1/')
  echo "  ✅ E2E PASSED: save=${SAVE_BYTES} bytes, query=hit, load=OK, verify=MATCH"
else
  echo "  ❌ E2E FAILED:"
  echo "${E2E_OUTPUT}"
  echo ""
  echo "  Server log (last 10 lines):"
  tail -10 "${SERVER_LOG}"
  kill ${SERVER_PID} 2>/dev/null || true
  exit 1
fi

# ---- Step 4: Cleanup ----
echo "[4/4] Stopping server..."
kill ${SERVER_PID} 2>/dev/null || true
sleep 1
echo "  Done. Server log: ${SERVER_LOG}"

echo ""
echo "============================================================"
echo " E2E Test PASSED"
echo " Tag: ${TAG}"
echo " Server log: ${SERVER_LOG}"
echo "============================================================"
