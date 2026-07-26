#!/bin/bash
# ============================================================================
# PegaFlow Prefix-Caching Performance Benchmark
#
# Strategy (inspired by run_load_e2e.sh):
#   Session 1: vLLM + PegaFlow → send fixed-prefix requests → D2H save
#   Session 2: NEW vLLM + same pegaflow-server → SAME requests → H2D load ★
#
# KEY: Same hardcoded shared prefix across sessions guarantees identical
#      tokenization → identical block_hashes → external cache HIT.
#
# Model: Qwen2.5-7B-Instruct
#
# Usage:
#   ./scripts/bench_pegaflow.sh [NPU_DEVICE_ID]
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ulimit -l unlimited

NPU_DEVICE="${1:-2}"
GRPC_PORT=50059
VLLM_PORT_1=8101
VLLM_PORT_2=8102
TAG="bench-$(date +%Y%m%d-%H%M%S)"
RESULTS_DIR="/tmp/pegaflow-bench-${TAG}"

# Model: use 7B for meaningful KV cache size.
MODEL="/root/.cache/modelscope/models/qwen--Qwen2.5-7B-Instruct/snapshots/master"
MODEL_NAME="Qwen2.5-7B-Instruct"

# Workload.
NUM_REQUESTS="${NUM_REQUESTS:-100}"
CONCURRENCY="${CONCURRENCY:-8}"
MAX_TOKENS=150

# Long shared prefix (>256 tokens = 2+ full blocks at block_size=128).
# run_load_e2e.sh uses a similar pattern — this guarantees at least 2
# full blocks are pure shared content.
SHARED_PREFIX="Here is a comprehensive list of all countries with their capitals, continents, languages, populations, and currencies: Afghanistan Kabul Asia Dari Pashto 41M Afghani; Albania Tirana Europe Albanian 2.8M Lek; Algeria Algiers Africa Arabic Berber 44M Dinar; Andorra Andorra la Vella Europe Catalan 79000 Euro; Angola Luanda Africa Portuguese 35M Kwanza; Argentina Buenos Aires South America Spanish 46M Peso; Armenia Yerevan Asia Armenian 2.8M Dram; Australia Canberra Oceania English 26M Dollar; Austria Vienna Europe German 9M Euro; Azerbaijan Baku Asia Azerbaijani 10M Manat; Bahamas Nassau North America English 400000 Dollar; Bahrain Manama Asia Arabic 1.5M Dinar; Bangladesh Dhaka Asia Bengali 170M Taka; Barbados Bridgetown North America English 280000 Dollar; Belarus Minsk Europe Belarusian Russian 9.2M Ruble; Belgium Brussels Europe Dutch French German 11.7M Euro; Belize Belmopan North America English 410000 Dollar; Benin Porto-Novo Africa French 13M Franc; Bhutan Thimphu Asia Dzongkha 780000 Ngultrum; Bolivia Sucre South America Spanish Quechua Aymara 12M Boliviano; Bosnia Herzegovina Sarajevo Europe Bosnian Croatian Serbian 3.2M Marka; Botswana Gaborone Africa English Tswana 2.6M Pula; Brazil Brasilia South America Portuguese 215M Real; Brunei Bandar Seri Begawan Asia Malay 450000 Dollar; Bulgaria Sofia Europe Bulgarian 6.5M Lev; Burkina Faso Ouagadougou Africa French 23M Franc; Burundi Gitega Africa Kirundi French 13M Franc; Cabo Verde Praia Africa Portuguese 590000 Escudo; Cambodia Phnom Penh Asia Khmer 17M Riel; Cameroon Yaounde Africa French English 28M Franc; Canada Ottawa North America English French 39M Dollar; Chile Santiago South America Spanish 19.5M Peso; China Beijing Asia Mandarin 1412M Yuan Renminbi; Colombia Bogota South America Spanish 52M Peso; Congo Kinshasa Africa French 100M Franc; Costa Rica San Jose North America Spanish 5.2M Colon; Croatia Zagreb Europe Croatian 3.9M Euro; Cuba Havana North America Spanish 11M Peso; Cyprus Nicosia Asia Greek Turkish 1.2M Euro; Czech Republic Prague Europe Czech 10.7M Koruna; Denmark Copenhagen Europe Danish 5.9M Krone; Djibouti Djibouti City Africa French Arabic 1.1M Franc; Ecuador Quito South America Spanish 18M Dollar; Egypt Cairo Africa Arabic 110M Pound; Estonia Tallinn Europe Estonian 1.3M Euro; Ethiopia Addis Ababa Africa Amharic 125M Birr; Fiji Suva Oceania English Fijian Hindi 930000 Dollar; Finland Helsinki Europe Finnish Swedish 5.6M Euro; France Paris Europe French 68M Euro; Gabon Libreville Africa French 2.4M Franc; Gambia Banjul Africa English 2.6M Dalasi; Georgia Tbilisi Asia Georgian 3.7M Lari; Germany Berlin Europe German 84M Euro; Ghana Accra Africa English 33M Cedi; Greece Athens Europe Greek 10.4M Euro; Guatemala Guatemala City North America Spanish 17M Quetzal; Guinea Conakry Africa French 13M Franc; Haiti Port-au-Prince North America Haitian Creole French 11.5M Gourde; Honduras Tegucigalpa North America Spanish 10M Lempira; Hungary Budapest Europe Hungarian 9.6M Forint; Iceland Reykjavik Europe Icelandic 380000 Krona; India New Delhi Asia Hindi English 1400M Rupee; Indonesia Jakarta Asia Indonesian 275M Rupiah; Iran Tehran Asia Persian 88M Rial; Iraq Baghdad Asia Arabic Kurdish 43M Dinar; Ireland Dublin Europe Irish English 5.1M Euro; Israel Jerusalem Asia Hebrew Arabic 9.5M Shekel; Italy Rome Europe Italian 59M Euro; Jamaica Kingston North America English 2.8M Dollar; Japan Tokyo Asia Japanese 125M Yen; Jordan Amman Asia Arabic 11M Dinar; Kazakhstan Astana Asia Kazakh Russian 19M Tenge; Kenya Nairobi Africa Swahili English 55M Shilling; Kuwait Kuwait City Asia Arabic 4.3M Dinar; Kyrgyzstan Bishkek Asia Kyrgyz Russian 6.7M Som; Laos Vientiane Asia Lao 7.5M Kip; Latvia Riga Europe Latvian 1.8M Euro; Lebanon Beirut Asia Arabic French 5.5M Pound; Libya Tripoli Africa Arabic 6.7M Dinar; Lithuania Vilnius Europe Lithuanian 2.8M Euro; Luxembourg Luxembourg City Europe Luxembourgish French German 650000 Euro; Madagascar Antananarivo Africa Malagasy French 29M Ariary; Malaysia Kuala Lumpur Asia Malay 34M Ringgit; Maldives Male Asia Dhivehi 520000 Rufiyaa; Mali Bamako Africa French 22M Franc; Malta Valletta Europe Maltese English 530000 Euro; Mexico Mexico City North America Spanish 128M Peso; Moldova Chisinau Europe Romanian 2.5M Leu; Monaco Monaco Europe French 36000 Euro; Mongolia Ulaanbaatar Asia Mongolian 3.4M Tugrik; Morocco Rabat Africa Arabic Berber 37M Dirham; Mozambique Maputo Africa Portuguese 33M Metical; Myanmar Naypyidaw Asia Burmese 54M Kyat; Namibia Windhoek Africa English 2.5M Dollar; Nepal Kathmandu Asia Nepali 30M Rupee; Netherlands Amsterdam Europe Dutch 17.7M Euro; New Zealand Wellington Oceania English Maori 5.2M Dollar; Nicaragua Managua North America Spanish 6.8M Cordoba; Niger Niamey Africa French 26M Franc; Nigeria Abuja Africa English 220M Naira; North Korea Pyongyang Asia Korean 26M Won; Norway Oslo Europe Norwegian 5.5M Krone; Oman Muscat Asia Arabic 4.6M Rial; Pakistan Islamabad Asia Urdu English 235M Rupee; Panama Panama City North America Spanish 4.4M Balboa; Paraguay Asuncion South America Spanish Guarani 6.7M Guarani; Peru Lima South America Spanish Quechua Aymara 34M Sol; Philippines Manila Asia Filipino English 115M Peso; Poland Warsaw Europe Polish 37M Zloty; Portugal Lisbon Europe Portuguese 10.3M Euro; Qatar Doha Asia Arabic 2.7M Riyal; Romania Bucharest Europe Romanian 19M Leu; Russia Moscow Europe Asia Russian 144M Ruble; Rwanda Kigali Africa Kinyarwanda French English 13.5M Franc; Saudi Arabia Riyadh Asia Arabic 36M Riyal; Senegal Dakar Africa French 17M Franc; Serbia Belgrade Europe Serbian 6.7M Dinar; Singapore Singapore Asia English Malay Mandarin Tamil 5.6M Dollar; Slovakia Bratislava Europe Slovak 5.4M Euro; Slovenia Ljubljana Europe Slovenian 2.1M Euro; Somalia Mogadishu Africa Somali Arabic 17M Shilling; South Africa Pretoria Africa 11 official languages 60M Rand; South Korea Seoul Asia Korean 52M Won; Spain Madrid Europe Spanish 47M Euro; Sri Lanka Colombo Asia Sinhala Tamil 22M Rupee; Sudan Khartoum Africa Arabic English 47M Pound; Sweden Stockholm Europe Swedish 10.5M Krona; Switzerland Bern Europe German French Italian Romansh 8.8M Franc; Syria Damascus Asia Arabic 22M Pound; Taiwan Taipei Asia Mandarin 23.5M Dollar; Tajikistan Dushanbe Asia Tajik 9.7M Somoni; Tanzania Dodoma Africa Swahili English 65M Shilling; Thailand Bangkok Asia Thai 72M Baht; Togo Lome Africa French 8.8M Franc; Tunisia Tunis Africa Arabic 12M Dinar; Turkey Ankara Asia Europe Turkish 85M Lira; Turkmenistan Ashgabat Asia Turkmen 6.3M Manat; Uganda Kampala Africa English Swahili 48M Shilling; Ukraine Kyiv Europe Ukrainian 38M Hryvnia; United Arab Emirates Abu Dhabi Asia Arabic 9.4M Dirham; United Kingdom London Europe English 67M Pound Sterling; United States Washington DC North America English 335M Dollar; Uruguay Montevideo South America Spanish 3.4M Peso; Uzbekistan Tashkent Asia Uzbek 35M Som; Venezuela Caracas South America Spanish 28M Bolivar; Vietnam Hanoi Asia Vietnamese 100M Dong; Yemen Sanaa Asia Arabic 34M Rial; Zambia Lusaka Africa English 20M Kwacha; Zimbabwe Harare Africa 16 official languages 16M Dollar."

# Unique suffixes — same across sessions so block_hashes match.
SUFFIXES=(
  "What are the major rivers in each country?"
  "What is the population of each country?"
  "List the primary languages spoken in each country."
  "Describe the climate of each country."
  "What is the cuisine like in each country?"
  "Name the famous landmarks in each country."
  "What is the history of each country?"
  "Describe the education system in each country."
  "What are the main exports of each country?"
  "How is the transportation system in each country?"
  "What sports are popular in each country?"
  "Describe the architecture styles in each country."
  "What festivals are celebrated in each country?"
  "What is the political system of each country?"
  "How is the healthcare system in each country?"
  "What natural resources does each country have?"
  "Describe the art and music scene in each country."
  "What are the traditional costumes of each country?"
  "How is the economy structured in each country?"
  "What wildlife can be found in each country?"
)

# Paths.
ASCEND_HOME="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.1}"
ASCEND_DRIVER="${ASCEND_HOME_PATH:+$(dirname "$ASCEND_HOME_PATH")/driver}/lib64/driver"
ASCEND_DRIVER="${ASCEND_DRIVER:-/usr/local/Ascend/driver/lib64/driver}"
ATB_LIB=$(ls -d /usr/local/Ascend/nnal/atb/*/atb/cxx_abi_1/lib 2>/dev/null | head -1)
ATB_LIB="${ATB_LIB:-/usr/local/Ascend/nnal/atb/8.5.1/atb/cxx_abi_1/lib}"
VENV_DIR="${PEGAFLOW_VENV:-/root/miniconda3/envs/vllm-hust-dev}"
PYTHON="${VENV_DIR}/bin/python"
PEGAFLOW_DIR="/workspace/pegaflow-hust"
CARGO_TARGET="${PEGAFLOW_DIR}/target/debug"
SERVER_LOG="/tmp/pf-bench-server-${TAG}.log"
VLLM1_LOG="/tmp/pf-bench-vllm1-${TAG}.log"
VLLM2_LOG="/tmp/pf-bench-vllm2-${TAG}.log"
BENCH_SCRIPT="${RESULTS_DIR}/run_bench.py"

export LD_LIBRARY_PATH="\
${VENV_DIR}/lib:\
${ASCEND_HOME}/lib64:\
${ASCEND_HOME}/aarch64-linux/lib64:\
${ASCEND_DRIVER}:\
${ATB_LIB}"
# Device visibility: pegaflow-server uses --devices directly.
# For vLLM processes, ASCEND_RT_VISIBLE_DEVICES is set per-command
# to avoid interference with pegaflow-server's physical device IDs.
# ASCEND_VISIBLE_DEVICES is set for ASCEND ACL compatibility.
export ASCEND_VISIBLE_DEVICES="${NPU_DEVICE}"
export VLLM_PLUGINS=ascend
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=DEBUG
export PYTHONHASHSEED=0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

health_check() {
  local url="$1" timeout="${2:-5}"
  "${PYTHON}" -c "
import urllib.request, sys
try:
    resp = urllib.request.urlopen('${url}', timeout=${timeout})
    resp.read()
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

fetch_metric() {
  local url="$1" name="$2"
  "${PYTHON}" -c "
import urllib.request
r = urllib.request.urlopen('${url}')
for l in r.read().decode().split('\n'):
    l = l.strip()
    if '${name}' in l and not l.startswith('#'):
        print(l.split()[-1])
        break
" 2>/dev/null || echo "0"
}

wait_for_vllm() {
  local pid="$1" log="$2" label="$3" port="$4" timeout="${5:-240}"
  for i in $(seq 1 "${timeout}"); do
    if health_check "http://127.0.0.1:${port}/health"; then
      # If health check passes in the first 3 seconds, the vLLM didn't start
      # from scratch — something else was already listening on this port.
      if [ "${i}" -le 2 ]; then
        local port_pid
        port_pid=$(ss -tlnp "sport = :${port}" 2>/dev/null | grep -oP 'pid=\K\d+' | head -1 || true)
        if [ -n "${port_pid}" ] && [ "${port_pid}" != "${pid}" ]; then
          echo "  WARNING: Port ${port} served by PID ${port_pid}, not our PID ${pid}!"
          echo "  A stale vLLM is occupying this port. Killing it..."
          kill -9 "${port_pid}" 2>/dev/null || true
          sleep 2
          continue
        fi
        # If PID matches but startup was very fast, note it but accept.
        echo "  ${label} ready (${i}s — fast, PID ${pid} confirmed on port)"
      else
        echo "  ${label} ready (${i}s)"
      fi
      return 0
    fi
    if ! kill -0 ${pid} 2>/dev/null; then
      echo "  FAIL: ${label} crashed."
      grep -E "error|Error|panic|FATAL" "${log}" | tail -5
      return 1
    fi
    sleep 3
  done
  echo "  FAIL: ${label} startup timed out"; return 1
}

wait_for_server() {
  for i in $(seq 1 60); do
    if health_check "http://127.0.0.1:9091/health"; then
      echo "  Server ready (${i}s)"; return 0
    fi
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
      echo "  FAIL: Server crashed."; tail -20 "${SERVER_LOG}"; exit 1
    fi
    sleep 1
  done
  echo "  FAIL: Server startup timed out"; exit 1
}

kill_vllm_graceful() {
  local pid="$1" label="$2" port="$3"
  echo "  Shutting down ${label} (PID=${pid}, port=${port})..."

  # Step 1: Find ALL descendants BEFORE killing the parent.
  # Once the parent dies, children are reparented to init and pkill -P won't find them.
  local all_pids="${pid}"
  local child_pids
  child_pids=$(pgrep -P "${pid}" 2>/dev/null || true)
  for cp in ${child_pids}; do
    all_pids="${all_pids} ${cp}"
    # Also find grandchildren (EngineCore may spawn its own children)
    local grandchild
    grandchild=$(pgrep -P "${cp}" 2>/dev/null || true)
    for gc in ${grandchild}; do
      all_pids="${all_pids} ${gc}"
    done
  done

  # Step 2: SIGKILL all descendants first, then the parent.
  # Use SIGKILL directly — SIGTERM on vLLM often leaves EngineCore alive.
  echo "  Killing ${label} process tree: ${all_pids}"
  for p in ${all_pids}; do
    kill -9 ${p} 2>/dev/null || true
  done
  sleep 2

  # Step 3: Verify the port is actually free.
  for i in $(seq 1 30); do
    if ss -tlnp "sport = :${port}" 2>/dev/null | grep -q "pid="; then
      sleep 1
    else
      echo "  Port ${port} free (${i}s)"
      break
    fi
  done
  if ss -tlnp "sport = :${port}" 2>/dev/null | grep -q "pid="; then
    echo "  WARNING: Port ${port} still occupied after killing process tree!"
    local stale_pid
    stale_pid=$(ss -tlnp "sport = :${port}" 2>/dev/null | grep -oP 'pid=\K\d+' | head -1 || true)
    if [ -n "${stale_pid}" ]; then
      echo "  Killing stale PID ${stale_pid} on port ${port}..."
      kill -9 "${stale_pid}" 2>/dev/null || true
      sleep 2
    fi
  fi

  echo "  Waiting for NPU memory recovery..."
  sleep 15
}

# Show memory status of all NPUs for diagnostics (uses Python for reliable parsing).
show_npu_memory() {
  echo "  NPU memory status:"
  "${PYTHON}" -c "
import subprocess, sys, re
try:
    out = subprocess.check_output(['npu-smi', 'info'], text=True, timeout=10)
except Exception as e:
    print(f'    npu-smi failed: {e}')
    sys.exit(0)
current_npu = None
for line in out.split(chr(10)):
    line = line.strip()
    if not line.startswith('|'):
        continue
    parts = [p.strip() for p in line.split('|')]
    if len(parts) < 3:
        continue
    first = parts[1]
    # NPU row: starts with digit, contains chip model name
    if first and first[0].isdigit() and len(first.split()) > 1:
        current_npu = first.split()[0]
    # Chip row: starts with single digit (chip id within NPU)
    elif first and first.isdigit() and len(first) <= 2 and current_npu is not None:
        # Last pipe-field contains AICore MemUsed/MemTotal HbmUsed/HbmTotal
        last = parts[-2]  # penultimate field: trailing | creates empty last element
        nums = re.findall(r'(\\d+)\s*/\s*(\\d+)', last)
        if len(nums) >= 2:
            hbm_used, hbm_total = int(nums[-1][0]), int(nums[-1][1])
            hbm_free = hbm_total - hbm_used
            print(f'    NPU {current_npu}: {hbm_free} MiB free / {hbm_total} MiB total ({hbm_used} MiB used)')
" 2>/dev/null
}

# Get free HBM memory (MiB) for the target NPU.
get_npu_free_mb() {
  "${PYTHON}" -c "
import subprocess, sys, re
target = '${1}'
try:
    out = subprocess.check_output(['npu-smi', 'info'], text=True, timeout=10)
except Exception:
    print(0); sys.exit(0)
current_npu = None
for line in out.split(chr(10)):
    line = line.strip()
    if not line.startswith('|'):
        continue
    parts = [p.strip() for p in line.split('|')]
    if len(parts) < 3:
        continue
    first = parts[1]
    if first and first[0].isdigit() and len(first.split()) > 1:
        current_npu = first.split()[0]
    elif first and first.isdigit() and len(first) <= 2 and current_npu == target:
        last = parts[-2]  # penultimate field: trailing | creates empty last element
        nums = re.findall(r'(\\d+)\s*/\s*(\\d+)', last)
        if len(nums) >= 2:
            hbm_used, hbm_total = int(nums[-1][0]), int(nums[-1][1])
            print(hbm_total - hbm_used)
            sys.exit(0)
print(0)
" 2>/dev/null
}


# Poll until NPU has enough free memory. FATAL on timeout.
wait_npu_memory() {
  local min_free_mb="${1:-10000}" timeout="${2:-120}"
  echo "  Checking NPU ${NPU_DEVICE} memory (need >= ${min_free_mb} MiB)..."
  show_npu_memory
  for i in $(seq 1 "${timeout}"); do
    local free_mb
    free_mb=$(get_npu_free_mb "${NPU_DEVICE}")
    if [ -z "${free_mb}" ]; then
      echo "  ERROR: Cannot read NPU memory via npu-smi. Is CANN installed?"
      exit 1
    fi
    if [ "${free_mb}" -ge "${min_free_mb}" ]; then
      echo "  NPU ${NPU_DEVICE}: ${free_mb} MiB free (need ${min_free_mb}) ✓"
      return 0
    fi
    if [ $((i % 15)) -eq 0 ]; then
      echo "  Waiting for NPU ${NPU_DEVICE} (${i}s, ${free_mb} < ${min_free_mb} MiB)..."
    fi
    sleep 1
  done
  local final_free
  final_free=$(get_npu_free_mb "${NPU_DEVICE}")
  echo ""
  echo "  ERROR: NPU ${NPU_DEVICE} only ${final_free:-?} MiB free, need ${min_free_mb} MiB."
  echo "  Leaked memory from force-killed processes. Reboot container or use clean NPU."
  exit 1
}

# Full cleanup.
cleanup_all() {
  echo "  [cleanup] stopping everything..."
  kill ${VLLM1_PID:-} 2>/dev/null || true
  wait ${VLLM1_PID:-} 2>/dev/null || true
  kill ${VLLM2_PID:-} 2>/dev/null || true
  wait ${VLLM2_PID:-} 2>/dev/null || true
  kill ${SERVER_PID:-} 2>/dev/null || true
  wait ${SERVER_PID:-} 2>/dev/null || true
  ps aux | grep -E "pegaflow-server-py|vllm.entrypoints|EngineCore" \
    | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
  sleep 3
}

trap cleanup_all EXIT

mkdir -p "${RESULTS_DIR}"

echo "============================================================"
echo " PegaFlow Prefix-Caching Benchmark"
echo " NPU: ${NPU_DEVICE}  Model: ${MODEL_NAME}"
echo " Requests: ${NUM_REQUESTS}  Concurrency: ${CONCURRENCY}"
echo " Tag: ${TAG}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
echo ""
echo "[Build] Compiling pegaflow..."
cd "${PEGAFLOW_DIR}"
PYO3_PYTHON="${PYTHON}" cargo build --no-default-features --features ascend \
  -p pegaflow-py --lib --bin pegaflow-server-py 2>&1 | tail -3
\cp -f "${CARGO_TARGET}/libpegaflow.so" \
  "${PEGAFLOW_DIR}/python/pegaflow/pegaflow.cpython-311-aarch64-linux-gnu.so"
echo "  Build OK"

# ---------------------------------------------------------------------------
# Cleanup before starting
# ---------------------------------------------------------------------------
echo ""
echo "[Init] Checking NPU ${NPU_DEVICE} readiness..."
show_npu_memory
# Check if our target ports are free — fail early if something else is using them.
for port in ${VLLM_PORT_1} ${VLLM_PORT_2} ${GRPC_PORT}; do
  stale_pid=$(ss -tlnp "sport = :${port}" 2>/dev/null | grep -oP 'pid=\K\d+' | head -1 || true)
  if [ -n "${stale_pid}" ]; then
    cmd=$(ps -p "${stale_pid}" -o comm= 2>/dev/null || echo "unknown")
    echo "  WARNING: Port ${port} held by PID ${stale_pid} (${cmd})"
    echo "  This may be a leftover from a previous run."
    echo "  Kill it manually: kill -9 ${stale_pid}"
    echo "  Or wait and re-run the script."
    exit 1
  fi
done
echo "  All target ports free ✓"

# ---------------------------------------------------------------------------
# Generate benchmark Python script
# ---------------------------------------------------------------------------
cat > "${BENCH_SCRIPT}" << 'PYEOF'
import sys, os, json, time, threading, queue
import urllib.request

NUM_REQUESTS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
CONCURRENCY = int(sys.argv[2]) if len(sys.argv) > 2 else 8
BASE_URL    = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8101"
MODEL       = sys.argv[4] if len(sys.argv) > 4 else ""
OUTPUT_FILE = sys.argv[5] if len(sys.argv) > 5 else None

SHARED_PREFIX = __import__('os').environ.get('PEGAFLOW_SHARED_PREFIX', 'The capital of France is Paris.')

SUFFIXES = [
    "What are the major rivers in each country?",
    "What is the population of each country?",
    "List the primary languages spoken in each country.",
    "Describe the climate of each country.",
    "What is the cuisine like in each country?",
    "Name the famous landmarks in each country.",
    "What is the history of each country?",
    "Describe the education system in each country.",
    "What are the main exports of each country?",
    "How is the transportation system in each country?",
    "What sports are popular in each country?",
    "Describe the architecture styles in each country.",
    "What festivals are celebrated in each country?",
    "What is the political system of each country?",
    "How is the healthcare system in each country?",
    "What natural resources does each country have?",
    "Describe the art and music scene in each country.",
    "What are the traditional costumes of each country?",
    "How is the economy structured in each country?",
    "What wildlife can be found in each country?",
]

# Generate prompts deterministically: cycle through suffixes.
def make_prompt(idx):
    suffix = SUFFIXES[idx % len(SUFFIXES)]
    return f"{SHARED_PREFIX}. {suffix}"

results_lock = threading.Lock()
results = []
start_event = threading.Event()

def send_request(idx):
    prompt = make_prompt(idx)
    data = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        body = json.loads(resp.read())
        ttft = (time.perf_counter() - t0) * 1000
        usage = body.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        with results_lock:
            results.append({
                "idx": idx,
                "ttft_ms": ttft,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "success": True,
            })
    except Exception as e:
        ttft = (time.perf_counter() - t0) * 1000
        with results_lock:
            results.append({
                "idx": idx,
                "ttft_ms": ttft,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "success": False,
                "error": str(e)[:200],
            })

def worker(idx_q):
    while True:
        try:
            idx = idx_q.get_nowait()
        except queue.Empty:
            return
        start_event.wait()
        send_request(idx)

print(f"Benchmark: {NUM_REQUESTS} requests, concurrency={CONCURRENCY}")
print(f"Model: {MODEL}")
print(f"Prefix: {SHARED_PREFIX[:80]}...")

# Build work queue
idx_q = queue.Queue()
for i in range(NUM_REQUESTS):
    idx_q.put(i)

threads = []
for _ in range(CONCURRENCY):
    t = threading.Thread(target=worker, args=(idx_q,))
    t.start()
    threads.append(t)

t_start = time.perf_counter()
start_event.set()

for t in threads:
    t.join()

duration = time.perf_counter() - t_start

# Summarize
successes = [r for r in results if r["success"]]
failures  = [r for r in results if not r["success"]]

ttfts = sorted([r["ttft_ms"] for r in successes])
prompt_tokens  = sum(r["prompt_tokens"] for r in successes)
completion_tokens = sum(r["completion_tokens"] for r in successes)

def pct(lst, p):
    if not lst: return float('nan')
    k = (len(lst) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(lst) - 1)
    return lst[f] + (lst[c] - lst[f]) * (k - f) if f < c else lst[f]

output = {
    "num_requests": NUM_REQUESTS,
    "completed": len(successes),
    "failed": len(failures),
    "duration_s": duration,
    "total_prompt_tokens": prompt_tokens,
    "total_completion_tokens": completion_tokens,
    "request_throughput": len(successes) / duration if duration > 0 else 0,
    "output_throughput": completion_tokens / duration if duration > 0 else 0,
    "mean_ttft_ms": sum(ttfts) / len(ttfts) if ttfts else float('nan'),
    "median_ttft_ms": pct(ttfts, 50),
    "p99_ttft_ms": pct(ttfts, 99),
    "min_ttft_ms": ttfts[0] if ttfts else float('nan'),
    "max_ttft_ms": ttfts[-1] if ttfts else float('nan'),
}

print(f"\nResults: {len(successes)}/{NUM_REQUESTS} ok, {len(failures)} failed")
print(f"Duration:  {duration:.2f}s")
print(f"Throughput: {output['request_throughput']:.2f} req/s")
print(f"TTFT p50:   {output['median_ttft_ms']:.2f} ms")
print(f"TTFT p99:   {output['p99_ttft_ms']:.2f} ms")
print(f"TTFT mean:  {output['mean_ttft_ms']:.2f} ms")

if OUTPUT_FILE:
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f)
    print(f"Saved: {OUTPUT_FILE}")
PYEOF

export PEGAFLOW_SHARED_PREFIX="${SHARED_PREFIX}"

# ============================================================================
# Phase A: BASELINE — vLLM standalone (no PegaFlow)
# ============================================================================
echo ""
echo "============================================================"
echo " Phase A: BASELINE (vLLM standalone)"
echo "============================================================"

echo "[A1] Starting vLLM..."
wait_npu_memory 20000 120
ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICE}" nohup "${PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --port "${VLLM_PORT_1}" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.5 \
    --max-num-seqs 16 \
  --enforce-eager \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"./vllm_profile","torch_profiler_with_stack":false}' \
  > "${VLLM1_LOG}" 2>&1 &
VLLM1_PID=$!
disown ${VLLM1_PID} 2>/dev/null || true
wait_for_vllm "${VLLM1_PID}" "${VLLM1_LOG}" "vLLM-baseline" "${VLLM_PORT_1}" || exit 1

echo "[A2] Benchmark (${NUM_REQUESTS} req)..."
BASELINE_OUT="${RESULTS_DIR}/baseline_serve.json"
"${PYTHON}" "${BENCH_SCRIPT}" "${NUM_REQUESTS}" "${CONCURRENCY}" \
  "http://127.0.0.1:${VLLM_PORT_1}" "${MODEL}" "${BASELINE_OUT}"

kill_vllm_graceful "${VLLM1_PID}" "vLLM-baseline" "${VLLM_PORT_1}"
wait_npu_memory 20000 120

# ============================================================================
# Start pegaflow-server (shared across both sessions)
# ============================================================================
echo ""
echo "============================================================"
echo " Starting pegaflow-server (shared across Sessions 1 & 2)"
echo "============================================================"

echo "[Server] Starting..."
nohup "${CARGO_TARGET}/pegaflow-server-py" \
  --addr "127.0.0.1:${GRPC_PORT}" \
  --devices "${NPU_DEVICE}" \
  --pool-size 2gb \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
disown ${SERVER_PID} 2>/dev/null || true
wait_for_server

# ============================================================================
# Session 1: vLLM + PegaFlow → FILL external cache
# ============================================================================
echo ""
echo "============================================================"
echo " Session 1: R1 FILL — vLLM + PegaFlow → D2H save"
echo "============================================================"

echo "[S1] Starting vLLM + PegaFlow..."
wait_npu_memory 30000 120
ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICE}" nohup "${PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --port "${VLLM_PORT_1}" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.5 \
    --max-num-seqs 16 \
  --enforce-eager \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"./vllm_profile","torch_profiler_with_stack":false}' \
  --kv-transfer-config '{"kv_connector":"PegaKVConnector","kv_role":"kv_both","kv_connector_module_path":"pegaflow.connector","kv_connector_extra_config":{"pegaflow.port":'"${GRPC_PORT}"'}}' \
  > "${VLLM1_LOG}" 2>&1 &
VLLM1_PID=$!
disown ${VLLM1_PID} 2>/dev/null || true
wait_for_vllm "${VLLM1_PID}" "${VLLM1_LOG}" "vLLM-S1" "${VLLM_PORT_1}" || exit 1

R1_SAVE_BEFORE=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_save_bytes_total")
R1_LOAD_BEFORE=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_load_bytes_total")
R1_BLOCKS_BEFORE=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_cache_block_insertions")
R1_HITS_BEFORE=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_cache_block_hits_total")

echo "[S1] Sending ${NUM_REQUESTS} requests → D2H save..."
R1_OUT="${RESULTS_DIR}/r1_fill_serve.json"
"${PYTHON}" "${BENCH_SCRIPT}" "${NUM_REQUESTS}" "${CONCURRENCY}" \
  "http://127.0.0.1:${VLLM_PORT_1}" "${MODEL}" "${R1_OUT}"

# Wait for async saves
echo "  Waiting for async saves..."
for i in $(seq 1 30); do
  INS=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_cache_block_insertions")
  if [ -n "${INS}" ] && [ "${INS}" -ge 1 ]; then
    echo "    ${INS} blocks in cache (after ${i}s)"
    break
  fi
  sleep 1
done

R1_SAVE_AFTER=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_save_bytes_total")
R1_LOAD_AFTER=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_load_bytes_total")
R1_BLOCKS_AFTER=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_cache_block_insertions")
R1_HITS_AFTER=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_cache_block_hits_total")
R1_RESIDENT_AFTER=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_cache_resident_bytes")

echo "  R1 cache deltas:"
echo "    save_bytes:      ${R1_SAVE_BEFORE} → ${R1_SAVE_AFTER}"
echo "    load_bytes:      ${R1_LOAD_BEFORE} → ${R1_LOAD_AFTER}"
echo "    blocks:           ${R1_BLOCKS_BEFORE} → ${R1_BLOCKS_AFTER}"
echo "    resident_bytes:   ${R1_RESIDENT_AFTER}"

# ---- Shutdown vLLM-1 (keep server alive) ----
echo ""
echo "  Shutting down vLLM-S1..."
kill_vllm_graceful "${VLLM1_PID}" "vLLM-S1" "${VLLM_PORT_1}"

# Verify cache survived
echo "  Checking cache survived vLLM restart..."
sleep 3
RESTART_RESIDENT=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_cache_resident_bytes")
RESTART_BLOCKS=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_cache_block_insertions")
echo "    resident_bytes=${RESTART_RESIDENT} blocks=${RESTART_BLOCKS}"

# Wait for NPU memory to stabilize
echo "  Waiting for NPU memory to stabilize..."
wait_npu_memory 20000 120

# Verify server is still alive before S2
echo "  Checking pegaflow-server is alive..."
if ! health_check "http://127.0.0.1:9091/health" 5; then
  echo "  FAIL: pegaflow-server is not reachable (likely crashed during S1)."
  echo "  Check server log: ${SERVER_LOG}"
  echo "  S2 cannot proceed — server died with aclrtSynchronizeStream error."
  cleanup_all
  exit 1
fi
echo "  Server OK"

# ============================================================================
# Session 2: NEW vLLM + SAME pegaflow-server → H2D LOAD ★
# ============================================================================
echo ""
echo "============================================================"
echo " Session 2: R2 HIT — New vLLM (cold APC) → H2D load"
echo "============================================================"

echo "[S2] Starting new vLLM + PegaFlow..."
wait_npu_memory 30000 120
ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICE}" nohup "${PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --port "${VLLM_PORT_2}" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.5 \
    --max-num-seqs 16 \
  --enforce-eager \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"./vllm_profile","torch_profiler_with_stack":false}' \
  --kv-transfer-config '{"kv_connector":"PegaKVConnector","kv_role":"kv_both","kv_connector_module_path":"pegaflow.connector","kv_connector_extra_config":{"pegaflow.port":'"${GRPC_PORT}"'}}' \
  > "${VLLM2_LOG}" 2>&1 &
VLLM2_PID=$!
disown ${VLLM2_PID} 2>/dev/null || true
wait_for_vllm "${VLLM2_PID}" "${VLLM2_LOG}" "vLLM-S2" "${VLLM_PORT_2}" || {
  # S2 may fail due to insufficient NPU memory if orphan cleanup didn't work.
  # Check if it's a memory error and provide actionable info.
  if grep -q "Free memory.*less than desired" "${VLLM2_LOG}" 2>/dev/null; then
    echo "  S2 failed: insufficient NPU memory. Orphan EngineCore may still hold memory."
    echo "  Try: ps aux | grep EngineCore && kill -9 those PIDs"
  fi
  exit 1
}

R2_SAVE_BEFORE=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_save_bytes_total")
R2_LOAD_BEFORE=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_load_bytes_total")
R2_BLOCKS_BEFORE=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_cache_block_insertions")
R2_HITS_BEFORE=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_cache_block_hits_total")
R2_MISSES_BEFORE=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_cache_block_misses_total")

echo "[S2] Sending ${NUM_REQUESTS} requests (SAME prompts → H2D load)..."
R2_OUT="${RESULTS_DIR}/r2_hit_serve.json"
"${PYTHON}" "${BENCH_SCRIPT}" "${NUM_REQUESTS}" "${CONCURRENCY}" \
  "http://127.0.0.1:${VLLM_PORT_2}" "${MODEL}" "${R2_OUT}"

sleep 3

R2_SAVE_AFTER=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_save_bytes_total")
R2_LOAD_AFTER=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_load_bytes_total")
R2_BLOCKS_AFTER=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_cache_block_insertions")
R2_HITS_AFTER=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_cache_block_hits_total")
R2_MISSES_AFTER=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_cache_block_misses_total")

# ---- S2 KV Transfer metrics ----
echo ""
echo "  S2 vLLM KV Transfer metrics:"
"${PYTHON}" -c "
import urllib.request
r = urllib.request.urlopen('http://127.0.0.1:${VLLM_PORT_2}/metrics')
for l in r.read().decode().split('\n'):
    l = l.strip()
    if 'vllm:kv_transfer' in l and not l.startswith('#'):
        print(f'    {l}')
    if 'external_prefix_cache' in l and not l.startswith('#'):
        print(f'    {l}')
" 2>&1

# ---- S2 cache lookup logs ----
echo ""
echo "  S2 cache lookup log (first 10):"
grep "cache_lookup" "${VLLM2_LOG}" 2>/dev/null | head -10 | while read -r line; do
  echo "    ${line}"
done

echo ""
echo "  R2 cache deltas (vs R2 start):"
echo "    save_bytes:    ${R2_SAVE_BEFORE} → ${R2_SAVE_AFTER}"
echo "    load_bytes:    ${R2_LOAD_BEFORE} → ${R2_LOAD_AFTER}"
echo "    blocks:        ${R2_BLOCKS_BEFORE} → ${R2_BLOCKS_AFTER}"
echo ""
echo "  ★ R2 external hits:   $(( $(echo "${R2_HITS_AFTER:-0}" | tr -d '\n') - $(echo "${R2_HITS_BEFORE:-0}" | tr -d '\n') ))"
echo "  ★ R2 external misses: $(( $(echo "${R2_MISSES_AFTER:-0}" | tr -d '\n') - $(echo "${R2_MISSES_BEFORE:-0}" | tr -d '\n') ))"
echo "  ★ R2 H2D load_bytes:  $(( $(echo "${R2_LOAD_AFTER:-0}" | tr -d '\n') - $(echo "${R2_LOAD_BEFORE:-0}" | tr -d '\n') ))"

# ---- PegaFlow final metrics ----
echo ""
echo "  PegaFlow final metrics:"
"${PYTHON}" -c "
import urllib.request
r = urllib.request.urlopen('http://127.0.0.1:9091/metrics')
for l in r.read().decode().split('\n'):
    l = l.strip()
    if 'rpc_requests_total' in l and not l.startswith('#'):
        if 'method=\"save\"' in l or 'method=\"load\"' in l or 'method=\"register\"' in l:
            print(f'    {l}')
    if 'save_bytes_total' in l or 'load_bytes_total' in l and not l.startswith('#'):
        print(f'    {l}')
    if 'cache_block_insertions' in l or 'cache_block_hits' in l or 'cache_block_misses' in l:
        if not l.startswith('#'):
            print(f'    {l}')
    if 'cache_resident_bytes' in l and not l.startswith('#'):
        print(f'    {l}')
" 2>&1

# ---- Server prefetch logs ----
echo ""
echo "  PegaFlow prefetch (first 5 of each session):"
grep "Prefetch local" "${SERVER_LOG}" 2>/dev/null | head -5 | while read -r line; do
  echo "    ${line}"
done
echo "    ..."
grep "Prefetch local" "${SERVER_LOG}" 2>/dev/null | tail -5 | while read -r line; do
  echo "    ${line}"
done

# ---- Check for hits ----
S2_HIT_COUNT=$(grep -c "hit_blocks=[1-9]" "${VLLM2_LOG}" 2>/dev/null || echo "0")
echo ""
echo "  ★ S2 requests with external cache hits: ${S2_HIT_COUNT}"

cleanup_all
sleep 2

# ============================================================================
# Report
# ============================================================================
echo ""
echo "============================================================"
echo " Results"
echo "============================================================"

REPORT="${RESULTS_DIR}/report.md"

"${PYTHON}" -c "
import json, os

results_dir = '${RESULTS_DIR}'

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return {'error': str(e)}

baseline = load_json(os.path.join(results_dir, 'baseline_serve.json'))
r1       = load_json(os.path.join(results_dir, 'r1_fill_serve.json'))
r2       = load_json(os.path.join(results_dir, 'r2_hit_serve.json'))

def safe(v, default=0.0):
    try: return float(v)
    except: return default

def delta(a, b):
    return (a - b) / b * 100 if b > 0 else float('nan')

# Metrics
m = {}
for k, val in [
    ('r1_save_b', '${R1_SAVE_BEFORE:-0}'), ('r1_save_a', '${R1_SAVE_AFTER:-0}'),
    ('r1_load_b', '${R1_LOAD_BEFORE:-0}'), ('r1_load_a', '${R1_LOAD_AFTER:-0}'),
    ('r1_blk_b', '${R1_BLOCKS_BEFORE:-0}'), ('r1_blk_a', '${R1_BLOCKS_AFTER:-0}'),
    ('r1_res_a', '${R1_RESIDENT_AFTER:-0}'),
    ('r2_save_b', '${R2_SAVE_BEFORE:-0}'), ('r2_save_a', '${R2_SAVE_AFTER:-0}'),
    ('r2_load_b', '${R2_LOAD_BEFORE:-0}'), ('r2_load_a', '${R2_LOAD_AFTER:-0}'),
    ('r2_blk_b', '${R2_BLOCKS_BEFORE:-0}'), ('r2_blk_a', '${R2_BLOCKS_AFTER:-0}'),
    ('r2_hits_b', '${R2_HITS_BEFORE:-0}'), ('r2_hits_a', '${R2_HITS_AFTER:-0}'),
    ('r2_misses_b', '${R2_MISSES_BEFORE:-0}'), ('r2_misses_a', '${R2_MISSES_AFTER:-0}'),
]:
    m[k] = safe(val)

r1_save     = m['r1_save_a'] - m['r1_save_b']
r1_blocks   = m['r1_blk_a'] - m['r1_blk_b']
r2_save     = m['r2_save_a'] - m['r2_save_b']
r2_load     = m['r2_load_a'] - m['r2_load_b']
r2_blocks   = m['r2_blk_a'] - m['r2_blk_b']
r2_hits     = m['r2_hits_a'] - m['r2_hits_b']
r2_misses   = m['r2_misses_a'] - m['r2_misses_b']

external_hit = r2_hits > 0
h2d_load     = r2_load > 0

lines = []
lines.append('# PegaFlow Benchmark Report')
lines.append('')
lines.append(f'**Tag:** ${TAG}')
lines.append(f'**Model:** ${MODEL_NAME}')
lines.append(f'**Requests:** ${NUM_REQUESTS} per session | **Concurrency:** ${CONCURRENCY}')
lines.append('')
lines.append('**Strategy:** Session 1 saves KV blocks to pegaflow-server.')
lines.append('Session 2 starts a fresh vLLM (cold internal APC) and sends the SAME')
lines.append('prompts. Since token sequences are identical, block hashes match and')
lines.append('the external cache should deliver HITS.')
lines.append('')

# Performance
lines.append('## Serving Performance')
lines.append('')
lines.append('| Metric | Baseline | R1 (fill) | R2 (hit) | R1 vs Base | R2 vs Base | R2 vs R1 |')
lines.append('|--------|----------|-----------|----------|------------|------------|----------|')

for key, label in [
    ('mean_ttft_ms', 'TTFT mean (ms)'),
    ('median_ttft_ms', 'TTFT p50 (ms)'),
    ('p99_ttft_ms', 'TTFT p99 (ms)'),
    ('request_throughput', 'Throughput (req/s)'),
    ('output_throughput', 'Output TPS'),
]:
    b  = safe(baseline.get(key)) if isinstance(baseline, dict) else 0.0
    r1v = safe(r1.get(key))       if isinstance(r1, dict)       else 0.0
    r2v = safe(r2.get(key))       if isinstance(r2, dict)       else 0.0
    lines.append(f'| {label} | {b:.2f} | {r1v:.2f} | {r2v:.2f} | {delta(r1v,b):+.1f}% | {delta(r2v,b):+.1f}% | {delta(r2v,r1v):+.1f}% |')

lines.append('')

# PegaFlow metrics
lines.append('## PegaFlow KV Cache')
lines.append('')
lines.append('| Metric | R1 (fill) Δ | R2 (hit) Δ | Meaning |')
lines.append('|--------|-------------|------------|---------|')
lines.append(f'| save_bytes | {r1_save:.0f} | {r2_save:.0f} | D2H written |')
lines.append(f'| load_bytes | — | {r2_load:.0f} | H2D read ★ |')
lines.append(f'| blocks inserted | {r1_blocks:.0f} | {r2_blocks:.0f} | Blocks cached |')
lines.append(f'| cache_hits | — | {r2_hits:.0f} | External hits ★ |')
lines.append(f'| cache_misses | — | {r2_misses:.0f} | External misses |')
lines.append('')

# Interpretation
lines.append('## Interpretation')
lines.append('')
lines.append('| Question | Answer |')
lines.append('|----------|--------|')
lines.append(f'| R2 external cache_hits > 0? | {\"✅ YES — HIT path works!\" if external_hit else \"❌ NO — blocks not found\"} |')
lines.append(f'| R2 load_bytes > 0? | {\"✅ YES — H2D DMA occurred\" if h2d_load else \"❌ NO — no H2D load\"} |')
if external_hit:
    lines.append(f'| R2 p50 vs Baseline? | {\"✅ YES — PegaFlow beats native\" if safe(r2.get(\"median_ttft_ms\")) < safe(baseline.get(\"median_ttft_ms\")) else \"No — external hit but still slower\"} |')
else:
    lines.append('| R2 p50 < R1 p50? | {} |'.format(
        \"✅ YES\" if safe(r2.get(\"median_ttft_ms\")) < safe(r1.get(\"median_ttft_ms\")) else \"No\"
    ))

if not external_hit:
    lines.append('')
    lines.append('**Possible root causes:**')
    lines.append('1. Block hashes differ between Session 1 and 2 (extra_keys include cache_salt?)')
    lines.append('2. Saved blocks were evicted from the 2 GiB pinned memory pool')
    lines.append('3. Session cleanup evicted blocks from ReadCache')
    lines.append('4. The Python benchmark script generates different prompts across runs')

report = '\n'.join(lines)
print(report)
with open('${REPORT}', 'w') as f:
    f.write(report)
print(f'\nReport: ${REPORT}')
"

echo ""
echo "============================================================"
echo " Complete: ${TAG}"
echo " Results:  ${RESULTS_DIR}"
echo " Report:   ${REPORT}"
echo " Logs:     ${SERVER_LOG} / ${VLLM1_LOG} / ${VLLM2_LOG}"
echo "============================================================"
