#!/bin/bash
# ============================================================================
# Multi-Instance PEGAFLOW Throughput Benchmark
#
# Starts N vLLM instances sharing a single pegaflow-server. First fills the
# external KV cache via instance-0, then benchmarks all instances concurrently
# to measure aggregate throughput with shared cache.
#
# PREREQUISITE: Run bench_multi_baseline.sh first for comparison.
#
# Usage:
#   ./scripts/bench_multi_pegaflow.sh [NUM_INSTANCES] [BASE_NPU_DEVICE]
#
#   NUM_INSTANCES   — number of vLLM instances (default: 2)
#   BASE_NPU_DEVICE — first NPU device to use (default: 6)
#
# Examples:
#   ./scripts/bench_multi_pegaflow.sh          # 2 instances on NPU 6,7
#   ./scripts/bench_multi_pegaflow.sh 4 0      # 4 instances on NPU 0-3
#   NUM_REQUESTS=50 CONCURRENCY=4 ./scripts/bench_multi_pegaflow.sh
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ulimit -l unlimited

NUM_INSTANCES="${1:-2}"
BASE_NPU="${2:-6}"

if [ "${NUM_INSTANCES}" -lt 1 ] || [ "${NUM_INSTANCES}" -gt 8 ]; then
  echo "ERROR: NUM_INSTANCES must be 1-8 (got ${NUM_INSTANCES})"
  exit 1
fi

GRPC_PORT=50059
BASE_VLLM_PORT=8201
TAG="pf-$(date +%Y%m%d-%H%M%S)"
RESULTS_DIR="/tmp/pegaflow-multi-${TAG}"

MODEL="/root/.cache/modelscope/models/qwen--Qwen2.5-7B-Instruct/snapshots/master"
MODEL_NAME="Qwen2.5-7B-Instruct"

NUM_REQUESTS="${NUM_REQUESTS:-80}"
CONCURRENCY="${CONCURRENCY:-8}"
MAX_TOKENS=150

SHARED_PREFIX="Here is a comprehensive list of all countries with their capitals, continents, languages, populations, and currencies: Afghanistan Kabul Asia Dari Pashto 41M Afghani; Albania Tirana Europe Albanian 2.8M Lek; Algeria Algiers Africa Arabic Berber 44M Dinar; Andorra Andorra la Vella Europe Catalan 79000 Euro; Angola Luanda Africa Portuguese 35M Kwanza; Argentina Buenos Aires South America Spanish 46M Peso; Armenia Yerevan Asia Armenian 2.8M Dram; Australia Canberra Oceania English 26M Dollar; Austria Vienna Europe German 9M Euro; Azerbaijan Baku Asia Azerbaijani 10M Manat; Bahamas Nassau North America English 400000 Dollar; Bahrain Manama Asia Arabic 1.5M Dinar; Bangladesh Dhaka Asia Bengali 170M Taka; Barbados Bridgetown North America English 280000 Dollar; Belarus Minsk Europe Belarusian Russian 9.2M Ruble; Belgium Brussels Europe Dutch French German 11.7M Euro; Belize Belmopan North America English 410000 Dollar; Benin Porto-Novo Africa French 13M Franc; Bhutan Thimphu Asia Dzongkha 780000 Ngultrum; Bolivia Sucre South America Spanish Quechua Aymara 12M Boliviano; Bosnia Herzegovina Sarajevo Europe Bosnian Croatian Serbian 3.2M Marka; Botswana Gaborone Africa English Tswana 2.6M Pula; Brazil Brasilia South America Portuguese 215M Real; Brunei Bandar Seri Begawan Asia Malay 450000 Dollar; Bulgaria Sofia Europe Bulgarian 6.5M Lev; Burkina Faso Ouagadougou Africa French 23M Franc; Burundi Gitega Africa Kirundi French 13M Franc; Cabo Verde Praia Africa Portuguese 590000 Escudo; Cambodia Phnom Penh Asia Khmer 17M Riel; Cameroon Yaounde Africa French English 28M Franc; Canada Ottawa North America English French 39M Dollar; Chile Santiago South America Spanish 19.5M Peso; China Beijing Asia Mandarin 1412M Yuan Renminbi; Colombia Bogota South America Spanish 52M Peso; Congo Kinshasa Africa French 100M Franc; Costa Rica San Jose North America Spanish 5.2M Colon; Croatia Zagreb Europe Croatian 3.9M Euro; Cuba Havana North America Spanish 11M Peso; Cyprus Nicosia Asia Greek Turkish 1.2M Euro; Czech Republic Prague Europe Czech 10.7M Koruna; Denmark Copenhagen Europe Danish 5.9M Krone; Djibouti Djibouti City Africa French Arabic 1.1M Franc; Ecuador Quito South America Spanish 18M Dollar; Egypt Cairo Africa Arabic 110M Pound; Estonia Tallinn Europe Estonian 1.3M Euro; Ethiopia Addis Ababa Africa Amharic 125M Birr; Fiji Suva Oceania English Fijian Hindi 930000 Dollar; Finland Helsinki Europe Finnish Swedish 5.6M Euro; France Paris Europe French 68M Euro; Gabon Libreville Africa French 2.4M Franc; Gambia Banjul Africa English 2.6M Dalasi; Georgia Tbilisi Asia Georgian 3.7M Lari; Germany Berlin Europe German 84M Euro; Ghana Accra Africa English 33M Cedi; Greece Athens Europe Greek 10.4M Euro; Guatemala Guatemala City North America Spanish 17M Quetzal; Guinea Conakry Africa French 13M Franc; Haiti Port-au-Prince North America Haitian Creole French 11.5M Gourde; Honduras Tegucigalpa North America Spanish 10M Lempira; Hungary Budapest Europe Hungarian 9.6M Forint; Iceland Reykjavik Europe Icelandic 380000 Krona; India New Delhi Asia Hindi English 1400M Rupee; Indonesia Jakarta Asia Indonesian 275M Rupiah; Iran Tehran Asia Persian 88M Rial; Iraq Baghdad Asia Arabic Kurdish 43M Dinar; Ireland Dublin Europe Irish English 5.1M Euro; Israel Jerusalem Asia Hebrew Arabic 9.5M Shekel; Italy Rome Europe Italian 59M Euro; Jamaica Kingston North America English 2.8M Dollar; Japan Tokyo Asia Japanese 125M Yen; Jordan Amman Asia Arabic 11M Dinar; Kazakhstan Astana Asia Kazakh Russian 19M Tenge; Kenya Nairobi Africa Swahili English 55M Shilling; Kuwait Kuwait City Asia Arabic 4.3M Dinar; Kyrgyzstan Bishkek Asia Kyrgyz Russian 6.7M Som; Laos Vientiane Asia Lao 7.5M Kip; Latvia Riga Europe Latvian 1.8M Euro; Lebanon Beirut Asia Arabic French 5.5M Pound; Libya Tripoli Africa Arabic 6.7M Dinar; Lithuania Vilnius Europe Lithuanian 2.8M Euro; Luxembourg Luxembourg City Europe Luxembourgish French German 650000 Euro; Madagascar Antananarivo Africa Malagasy French 29M Ariary; Malaysia Kuala Lumpur Asia Malay 34M Ringgit; Maldives Male Asia Dhivehi 520000 Rufiyaa; Mali Bamako Africa French 22M Franc; Malta Valletta Europe Maltese English 530000 Euro; Mexico Mexico City North America Spanish 128M Peso; Moldova Chisinau Europe Romanian 2.5M Leu; Monaco Monaco Europe French 36000 Euro; Mongolia Ulaanbaatar Asia Mongolian 3.4M Tugrik; Morocco Rabat Africa Arabic Berber 37M Dirham; Mozambique Maputo Africa Portuguese 33M Metical; Myanmar Naypyidaw Asia Burmese 54M Kyat; Namibia Windhoek Africa English 2.5M Dollar; Nepal Kathmandu Asia Nepali 30M Rupee; Netherlands Amsterdam Europe Dutch 17.7M Euro; New Zealand Wellington Oceania English Maori 5.2M Dollar; Nicaragua Managua North America Spanish 6.8M Cordoba; Niger Niamey Africa French 26M Franc; Nigeria Abuja Africa English 220M Naira; North Korea Pyongyang Asia Korean 26M Won; Norway Oslo Europe Norwegian 5.5M Krone; Oman Muscat Asia Arabic 4.6M Rial; Pakistan Islamabad Asia Urdu English 235M Rupee; Panama Panama City North America Spanish 4.4M Balboa; Paraguay Asuncion South America Spanish Guarani 6.7M Guarani; Peru Lima South America Spanish Quechua Aymara 34M Sol; Philippines Manila Asia Filipino English 115M Peso; Poland Warsaw Europe Polish 37M Zloty; Portugal Lisbon Europe Portuguese 10.3M Euro; Qatar Doha Asia Arabic 2.7M Riyal; Romania Bucharest Europe Romanian 19M Leu; Russia Moscow Europe Asia Russian 144M Ruble; Rwanda Kigali Africa Kinyarwanda French English 13.5M Franc; Saudi Arabia Riyadh Asia Arabic 36M Riyal; Senegal Dakar Africa French 17M Franc; Serbia Belgrade Europe Serbian 6.7M Dinar; Singapore Singapore Asia English Malay Mandarin Tamil 5.6M Dollar; Slovakia Bratislava Europe Slovak 5.4M Euro; Slovenia Ljubljana Europe Slovenian 2.1M Euro; Somalia Mogadishu Africa Somali Arabic 17M Shilling; South Africa Pretoria Africa 11 official languages 60M Rand; South Korea Seoul Asia Korean 52M Won; Spain Madrid Europe Spanish 47M Euro; Sri Lanka Colombo Asia Sinhala Tamil 22M Rupee; Sudan Khartoum Africa Arabic English 47M Pound; Sweden Stockholm Europe Swedish 10.5M Krona; Switzerland Bern Europe German French Italian Romansh 8.8M Franc; Syria Damascus Asia Arabic 22M Pound; Taiwan Taipei Asia Mandarin 23.5M Dollar; Tajikistan Dushanbe Asia Tajik 9.7M Somoni; Tanzania Dodoma Africa Swahili English 65M Shilling; Thailand Bangkok Asia Thai 72M Baht; Togo Lome Africa French 8.8M Franc; Tunisia Tunis Africa Arabic 12M Dinar; Turkey Ankara Asia Europe Turkish 85M Lira; Turkmenistan Ashgabat Asia Turkmen 6.3M Manat; Uganda Kampala Africa English Swahili 48M Shilling; Ukraine Kyiv Europe Ukrainian 38M Hryvnia; United Arab Emirates Abu Dhabi Asia Arabic 9.4M Dirham; United Kingdom London Europe English 67M Pound Sterling; United States Washington DC North America English 335M Dollar; Uruguay Montevideo South America Spanish 3.4M Peso; Uzbekistan Tashkent Asia Uzbek 35M Som; Venezuela Caracas South America Spanish 28M Bolivar; Vietnam Hanoi Asia Vietnamese 100M Dong; Yemen Sanaa Asia Arabic 34M Rial; Zambia Lusaka Africa English 20M Kwacha; Zimbabwe Harare Africa 16 official languages 16M Dollar."

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
SERVER_LOG="/tmp/pf-multi-server-${TAG}.log"
BENCH_SCRIPT="${RESULTS_DIR}/run_bench.py"

export LD_LIBRARY_PATH="\
${VENV_DIR}/lib:\
${ASCEND_HOME}/lib64:\
${ASCEND_HOME}/aarch64-linux/lib64:\
${ASCEND_DRIVER}:\
${ATB_LIB}"
export VLLM_PLUGINS=ascend
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=DEBUG
# CRITICAL for cross-instance cache sharing:
# Without this, vLLM's NONE_HASH = os.urandom(32) per process → all block hashes
# differ between instances → external cache LOOKUP FAILS for non-fill instances.
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
      if [ "${i}" -le 2 ]; then
        local port_pid
        port_pid=$(ss -tlnp "sport = :${port}" 2>/dev/null | grep -oP 'pid=\K\d+' | head -1 || true)
        if [ -n "${port_pid}" ] && [ "${port_pid}" != "${pid}" ]; then
          echo "  WARNING: Port ${port} served by PID ${port_pid}, not our PID ${pid}!"
          kill -9 "${port_pid}" 2>/dev/null || true
          sleep 2
          continue
        fi
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
  local all_pids="${pid}"
  local child_pids
  child_pids=$(pgrep -P "${pid}" 2>/dev/null || true)
  for cp in ${child_pids}; do
    all_pids="${all_pids} ${cp}"
    local grandchild
    grandchild=$(pgrep -P "${cp}" 2>/dev/null || true)
    for gc in ${grandchild}; do
      all_pids="${all_pids} ${gc}"
    done
  done
  echo "  Killing ${label} process tree: ${all_pids}"
  for p in ${all_pids}; do
    kill -9 ${p} 2>/dev/null || true
  done
  sleep 2
  for i in $(seq 1 30); do
    if ss -tlnp "sport = :${port}" 2>/dev/null | grep -q "pid="; then
      sleep 1
    else
      echo "  Port ${port} free (${i}s)"
      break
    fi
  done
  echo "  Waiting for NPU memory recovery..."
  sleep 15
}

show_npu_memory() {
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
    if first and first[0].isdigit() and len(first.split()) > 1:
        current_npu = first.split()[0]
    elif first and first.isdigit() and len(first) <= 2 and current_npu is not None:
        last = parts[-2]
        nums = re.findall(r'(\\d+)\s*/\s*(\\d+)', last)
        if len(nums) >= 2:
            hbm_used, hbm_total = int(nums[-1][0]), int(nums[-1][1])
            hbm_free = hbm_total - hbm_used
            print(f'    NPU {current_npu}: {hbm_free} MiB free / {hbm_total} MiB total ({hbm_used} MiB used)')
" 2>/dev/null
}

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
        last = parts[-2]
        nums = re.findall(r'(\\d+)\s*/\s*(\\d+)', last)
        if len(nums) >= 2:
            hbm_used, hbm_total = int(nums[-1][0]), int(nums[-1][1])
            print(hbm_total - hbm_used)
            sys.exit(0)
print(0)
" 2>/dev/null
}

wait_npu_memory() {
  local npu="${1}" min_free_mb="${2:-10000}" timeout="${3:-120}"
  for i in $(seq 1 "${timeout}"); do
    local free_mb
    free_mb=$(get_npu_free_mb "${npu}")
    if [ -z "${free_mb}" ]; then
      echo "  ERROR: Cannot read NPU memory via npu-smi."
      exit 1
    fi
    if [ "${free_mb}" -ge "${min_free_mb}" ]; then
      echo "  NPU ${npu}: ${free_mb} MiB free (need ${min_free_mb}) ✓"
      return 0
    fi
    if [ $((i % 15)) -eq 0 ]; then
      echo "  Waiting for NPU ${npu} (${i}s, ${free_mb} < ${min_free_mb} MiB)..."
    fi
    sleep 1
  done
  echo "  ERROR: NPU ${npu} timed out waiting for memory."
  exit 1
}

wait_all_npu_memory() {
  local min_free_mb="${1:-10000}" timeout="${2:-120}"
  local end_npu=$((BASE_NPU + NUM_INSTANCES - 1))
  echo "  Checking NPUs ${BASE_NPU}-${end_npu} (need >= ${min_free_mb} MiB each)..."
  show_npu_memory
  for ((n = BASE_NPU; n < BASE_NPU + NUM_INSTANCES; n++)); do
    wait_npu_memory "${n}" "${min_free_mb}" "${timeout}" || exit 1
  done
  echo "  All NPUs ready ✓"
}

cleanup_all() {
  echo "  [cleanup] stopping everything..."
  for ((i = 0; i < NUM_INSTANCES; i++)); do
    local pid_var="VLLM_PID_${i}"
    kill ${!pid_var:-} 2>/dev/null || true
    wait ${!pid_var:-} 2>/dev/null || true
  done
  kill ${SERVER_PID:-} 2>/dev/null || true
  wait ${SERVER_PID:-} 2>/dev/null || true
  ps aux | grep -E "pegaflow-server-py|vllm.entrypoints|EngineCore" \
    | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
  sleep 3
}

trap cleanup_all EXIT

mkdir -p "${RESULTS_DIR}"

# ---------------------------------------------------------------------------
# Build PegaFlow
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
# Pre-flight: check ports
# ---------------------------------------------------------------------------
echo ""
echo "[Init] Checking port availability..."
ALL_PORTS="${GRPC_PORT}"
for ((i = 0; i < NUM_INSTANCES; i++)); do
  ALL_PORTS="${ALL_PORTS} $((BASE_VLLM_PORT + i))"
done
for port in ${ALL_PORTS}; do
  stale_pid=$(ss -tlnp "sport = :${port}" 2>/dev/null | grep -oP 'pid=\K\d+' | head -1 || true)
  if [ -n "${stale_pid}" ]; then
    cmd=$(ps -p "${stale_pid}" -o comm= 2>/dev/null || echo "unknown")
    echo "  ERROR: Port ${port} held by PID ${stale_pid} (${cmd})"
    echo "  Kill it: kill -9 ${stale_pid}"
    exit 1
  fi
done
echo "  All ports free ✓"

# ---------------------------------------------------------------------------
# Generate benchmark Python script
# ---------------------------------------------------------------------------
cat > "${BENCH_SCRIPT}" << 'PYEOF'
import sys, os, json, time, threading, queue
import urllib.request

NUM_REQUESTS = int(sys.argv[1]) if len(sys.argv) > 1 else 80
CONCURRENCY = int(sys.argv[2]) if len(sys.argv) > 2 else 8
BASE_URL    = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8201"
MODEL       = sys.argv[4] if len(sys.argv) > 4 else ""
OUTPUT_FILE = sys.argv[5] if len(sys.argv) > 5 else None

SHARED_PREFIX = os.environ.get('PEGAFLOW_SHARED_PREFIX', 'The capital of France is Paris.')

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

print(f"[{BASE_URL}] Benchmark: {NUM_REQUESTS} requests, concurrency={CONCURRENCY}")

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
    "instance_url": BASE_URL,
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

print(f"[{BASE_URL}] Results: {len(successes)}/{NUM_REQUESTS} ok, {len(failures)} failed")
print(f"[{BASE_URL}] Duration:  {duration:.2f}s")
print(f"[{BASE_URL}] Throughput: {output['request_throughput']:.2f} req/s")
print(f"[{BASE_URL}] TTFT p50:   {output['median_ttft_ms']:.2f} ms")
print(f"[{BASE_URL}] TTFT p99:   {output['p99_ttft_ms']:.2f} ms")
print(f"[{BASE_URL}] TTFT mean:  {output['mean_ttft_ms']:.2f} ms")

if OUTPUT_FILE:
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f)
    print(f"[{BASE_URL}] Saved: {OUTPUT_FILE}")
PYEOF

export PEGAFLOW_SHARED_PREFIX="${SHARED_PREFIX}"

# ============================================================================
# Print banner
# ============================================================================
echo ""
echo "============================================================"
echo " Multi-Instance PEGAFLOW Throughput Benchmark"
echo " Instances: ${NUM_INSTANCES}  NPUs: ${BASE_NPU}-$((BASE_NPU + NUM_INSTANCES - 1))"
echo " Model: ${MODEL_NAME}"
echo " Requests/instance: ${NUM_REQUESTS}  Concurrency/instance: ${CONCURRENCY}"
echo " Tag: ${TAG}"
echo "============================================================"

# ============================================================================
# Start pegaflow-server
# ============================================================================
echo ""
echo "============================================================"
echo " Starting pegaflow-server"
echo "============================================================"

echo "[Server] Launching on devices ${BASE_NPU}-$((BASE_NPU + NUM_INSTANCES - 1))..."
nohup "${CARGO_TARGET}/pegaflow-server-py" \
  --addr "127.0.0.1:${GRPC_PORT}" \
  --devices $(for ((n = BASE_NPU; n < BASE_NPU + NUM_INSTANCES; n++)); do echo -n "${n},"; done | sed 's/,$//') \
  --pool-size $((NUM_INSTANCES * 2))gb \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
disown ${SERVER_PID} 2>/dev/null || true
wait_for_server

# ============================================================================
# Warmup: start ONLY instance-0, fill cache, then kill it so NPU is idle
#         and the pegaflow-server's DMA saves can drain.
#         This mirrors the original bench_pegaflow.sh Session-1 → kill →
#         Session-2 cadence, scaled to multi-instance.
# ============================================================================
echo ""
echo "============================================================"
echo " Warmup: start instance-0 → fill cache → kill (let DMA drain)"
echo "============================================================"

# Start ONLY instance-0 for warmup.
WARMUP_NPU=$((BASE_NPU + 0))
WARMUP_PORT=$((BASE_VLLM_PORT + 0))
WARMUP_LOG="/tmp/pf-multi-warmup-${TAG}.log"

echo "  Starting warmup instance: NPU=${WARMUP_NPU} port=${WARMUP_PORT}"
ASCEND_RT_VISIBLE_DEVICES="${WARMUP_NPU}" \
ASCEND_VISIBLE_DEVICES="${WARMUP_NPU}" \
nohup "${PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --port "${WARMUP_PORT}" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.35 \
  --max-num-seqs 16 \
  --enforce-eager \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"./vllm_profile","torch_profiler_with_stack":false}' \
  --kv-transfer-config '{"kv_connector":"PegaKVConnector","kv_role":"kv_both","kv_connector_module_path":"pegaflow.connector","kv_connector_extra_config":{"pegaflow.port":'"${GRPC_PORT}"'}}' \
  > "${WARMUP_LOG}" 2>&1 &
WARMUP_PID=$!
disown ${WARMUP_PID} 2>/dev/null || true
wait_for_vllm "${WARMUP_PID}" "${WARMUP_LOG}" "vLLM-warmup" "${WARMUP_PORT}" || exit 1

SAVE_BEFORE=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_save_bytes_total")
BLOCKS_BEFORE=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_cache_block_insertions")

echo "  Sending ${NUM_REQUESTS} requests to fill cache..."
WARMUP_OUT="${RESULTS_DIR}/warmup.json"
"${PYTHON}" "${BENCH_SCRIPT}" "${NUM_REQUESTS}" "${CONCURRENCY}" \
  "http://127.0.0.1:${WARMUP_PORT}" "${MODEL}" "${WARMUP_OUT}"

# Kill warmup vLLM so NPU is idle → DMA saves can drain without competing
# with compute. This is the same pattern as bench_pegaflow.sh Session-1 kill.
echo "  Killing warmup instance to let DMA saves drain..."
# Give connector time to submit pending save RPCs before kill
sleep 30
kill_vllm_graceful "${WARMUP_PID}" "vLLM-warmup" "${WARMUP_PORT}"

# Wait for DMA saves to drain.  The save worker may have been blocked on
# stream sync during warmup; after NPU is freed (vLLM killed), pending
# saves complete.  Poll for up to 3 minutes with 5s intervals.
echo "  Waiting for async DMA saves to drain..."
SAVE_TIMEOUT=36  # 36 × 5s = 180s
for i in $(seq 1 ${SAVE_TIMEOUT}); do
  sleep 5
  INS=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_cache_block_insertions")
  SAV=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_save_bytes_total")
  if [ -n "${SAV}" ] && [ "${SAV}" -gt 0 ] && [ -n "${INS}" ] && [ "${INS}" -gt 0 ]; then
    echo "    ${INS} blocks, ${SAV} bytes saved (after $((i*5))s)"
    # Continue polling in case more saves arrive
  fi
done

SAVE_AFTER=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_save_bytes_total")
BLOCKS_AFTER=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_cache_block_insertions")
RESIDENT_AFTER=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_cache_resident_bytes")

echo "  Warmup result:"
echo "    save_bytes:     ${SAVE_BEFORE} → ${SAVE_AFTER}"
echo "    blocks:         ${BLOCKS_BEFORE} → ${BLOCKS_AFTER}"
echo "    resident_bytes: ${RESIDENT_AFTER}"

if [ -z "${RESIDENT_AFTER}" ] || [ "${RESIDENT_AFTER:-0}" -eq 0 ]; then
  echo "  WARNING: Cache appears empty after warmup — saves may have all failed."
fi

# Ensure server survived the warmup.
echo "  Checking pegaflow-server is alive..."
if ! health_check "http://127.0.0.1:9091/health" 5; then
  echo "  FAIL: pegaflow-server is not reachable after warmup."
  exit 1
fi
echo "  Server OK"

# Wait for NPU memory on the warmup NPU to recover.
wait_npu_memory "${WARMUP_NPU}" 20000 60

# ============================================================================
# Start ALL vLLM instances with PegaFlow (fresh, for benchmark)
# ============================================================================
echo ""
echo "============================================================"
echo " Starting ${NUM_INSTANCES} vLLM instances for benchmark"
echo "============================================================"

for ((i = 0; i < NUM_INSTANCES; i++)); do
  npu=$((BASE_NPU + i))
  port=$((BASE_VLLM_PORT + i))
  log="/tmp/pf-multi-${i}-${TAG}.log"
  echo "  Instance ${i}: NPU=${npu} port=${port}"

  ASCEND_RT_VISIBLE_DEVICES="${npu}" \
  ASCEND_VISIBLE_DEVICES="${npu}" \
  nohup "${PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${port}" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.35 \
    --max-num-seqs 16 \
    --enforce-eager \
    --profiler-config '{"profiler":"torch","torch_profiler_dir":"./vllm_profile","torch_profiler_with_stack":false}' \
    --kv-transfer-config '{"kv_connector":"PegaKVConnector","kv_role":"kv_both","kv_connector_module_path":"pegaflow.connector","kv_connector_extra_config":{"pegaflow.port":'"${GRPC_PORT}"'}}' \
    > "${log}" 2>&1 &
  eval "VLLM_PID_${i}=$!"
  disown $! 2>/dev/null || true
done

echo ""
echo "Waiting for all instances..."
for ((i = 0; i < NUM_INSTANCES; i++)); do
  port=$((BASE_VLLM_PORT + i))
  pid_var="VLLM_PID_${i}"
  log="/tmp/pf-multi-${i}-${TAG}.log"
  wait_for_vllm "${!pid_var}" "${log}" "vLLM-pf-${i}" "${port}" || exit 1
done
echo "  All ${NUM_INSTANCES} instances ready ✓"

# ============================================================================
# Pre-benchmark metric snapshot
# ============================================================================
HITS_BEFORE=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_cache_block_hits_total")
MISSES_BEFORE=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_cache_block_misses_total")
LOAD_BEFORE=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_load_bytes_total")
SAVE_PRE=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_save_bytes_total")

# ============================================================================
# Benchmark: ALL instances concurrently
# ============================================================================
echo ""
echo "============================================================"
echo " Benchmark: ${NUM_INSTANCES} instances × ${NUM_REQUESTS} req each (concurrent)"
echo "============================================================"

BENCH_PIDS=()
BENCH_START=$(date +%s.%N)
for ((i = 0; i < NUM_INSTANCES; i++)); do
  port=$((BASE_VLLM_PORT + i))
  out="${RESULTS_DIR}/instance_${i}.json"
  "${PYTHON}" "${BENCH_SCRIPT}" "${NUM_REQUESTS}" "${CONCURRENCY}" \
    "http://127.0.0.1:${port}" "${MODEL}" "${out}" &
  BENCH_PIDS+=($!)
done

echo "  Waiting for ${#BENCH_PIDS[@]} benchmark clients..."
for pid in "${BENCH_PIDS[@]}"; do
  wait ${pid}
done
BENCH_END=$(date +%s.%N)
BENCH_WALL=$(echo "${BENCH_END} - ${BENCH_START}" | bc)
echo "  Wall-clock: ${BENCH_WALL}s"

sleep 3

# ============================================================================
# Post-benchmark metrics
# ============================================================================
HITS_AFTER=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_cache_block_hits_total")
MISSES_AFTER=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_cache_block_misses_total")
LOAD_AFTER=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_load_bytes_total")
SAVE_POST=$(fetch_metric "http://127.0.0.1:9091/metrics" "pegaflow_save_bytes_total")

HITS_DELTA=$(( $(echo "${HITS_AFTER:-0}" | tr -d '\n') - $(echo "${HITS_BEFORE:-0}" | tr -d '\n') ))
MISSES_DELTA=$(( $(echo "${MISSES_AFTER:-0}" | tr -d '\n') - $(echo "${MISSES_BEFORE:-0}" | tr -d '\n') ))
LOAD_DELTA=$(( $(echo "${LOAD_AFTER:-0}" | tr -d '\n') - $(echo "${LOAD_BEFORE:-0}" | tr -d '\n') ))
SAVE_DELTA=$(( $(echo "${SAVE_POST:-0}" | tr -d '\n') - $(echo "${SAVE_PRE:-0}" | tr -d '\n') ))

echo ""
echo "  Cache metrics (benchmark phase):"
echo "    hits:    ${HITS_DELTA}"
echo "    misses:  ${MISSES_DELTA}"
echo "    load (H2D):  ${LOAD_DELTA} B"
echo "    save (D2H):  ${SAVE_DELTA} B"
if [ $((HITS_DELTA + MISSES_DELTA)) -gt 0 ]; then
  echo "    hit rate: $(( HITS_DELTA * 100 / (HITS_DELTA + MISSES_DELTA) ))%"
fi

# ============================================================================
# Per-instance KV transfer metrics
# ============================================================================
echo ""
echo "  Per-instance KV Transfer:"
for ((i = 0; i < NUM_INSTANCES; i++)); do
  port=$((BASE_VLLM_PORT + i))
  echo "  --- Instance ${i} (port ${port}) ---"
  "${PYTHON}" -c "
import urllib.request
r = urllib.request.urlopen('http://127.0.0.1:${port}/metrics')
for l in r.read().decode().split('\n'):
    l = l.strip()
    if ('kv_transfer' in l or 'external_prefix_cache' in l) and not l.startswith('#'):
        print(f'    {l}')
" 2>&1
done

# ============================================================================
# PegaFlow server final metrics
# ============================================================================
echo ""
echo "  PegaFlow server final metrics:"
"${PYTHON}" -c "
import urllib.request
r = urllib.request.urlopen('http://127.0.0.1:9091/metrics')
for l in r.read().decode().split('\n'):
    l = l.strip()
    if 'rpc_requests_total' in l and not l.startswith('#'):
        if 'method=\"save\"' in l or 'method=\"load\"' in l:
            print(f'    {l}')
    if ('save_bytes_total' in l or 'load_bytes_total' in l) and not l.startswith('#'):
        print(f'    {l}')
    if 'cache_block_insertions' in l or 'cache_block_hits' in l or 'cache_block_misses' in l:
        if not l.startswith('#'):
            print(f'    {l}')
    if 'cache_resident_bytes' in l and not l.startswith('#'):
        print(f'    {l}')
" 2>&1

# ============================================================================
# Cache hit log counts per instance
# ============================================================================
echo ""
echo "  Cache hit log counts per instance:"
for ((i = 0; i < NUM_INSTANCES; i++)); do
  log="/tmp/pf-multi-${i}-${TAG}.log"
  hit_count=$(grep -c "hit_blocks=[1-9]" "${log}" 2>/dev/null || echo "0")
  echo "    Instance ${i}: ${hit_count} requests with external cache hits"
done

cleanup_all
sleep 2

# ============================================================================
# Aggregate results
# ============================================================================
echo ""
echo "============================================================"
echo " Results"
echo "============================================================"

"${PYTHON}" -c "
import json, os, statistics

results_dir = '${RESULTS_DIR}'
num_instances = ${NUM_INSTANCES}

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {'error': str(e)}

def safe(v, default=0.0):
    try: return float(v)
    except: return default

instances = []
for i in range(num_instances):
    r = load_json(os.path.join(results_dir, f'instance_{i}.json'))
    instances.append(r)

total_completed = 0
total_failed = 0
total_completion_tokens = 0
max_duration = 0.0
per_instance_tps = []

for r in instances:
    if isinstance(r, dict) and 'completed' in r:
        total_completed += r.get('completed', 0)
        total_failed += r.get('failed', 0)
        total_completion_tokens += r.get('total_completion_tokens', 0)
        dur = safe(r.get('duration_s'))
        if dur > max_duration:
            max_duration = dur
        per_instance_tps.append(safe(r.get('request_throughput')))

agg_req_per_sec = total_completed / max_duration if max_duration > 0 else 0
agg_output_tps = total_completion_tokens / max_duration if max_duration > 0 else 0

print()
print(f'Mode:                PegaFlow')
print(f'Instances:           {num_instances}')
print(f'Requests/instance:   ${NUM_REQUESTS}')
print(f'Total requests:      {num_instances * ${NUM_REQUESTS}}')
print(f'Completed:           {total_completed}')
print(f'Failed:              {total_failed}')
print(f'Wall-clock (s):      ${BENCH_WALL}')
print(f'Max duration (s):    {max_duration:.2f}')
print(f'Aggregate req/s:     {agg_req_per_sec:.2f}')
print(f'Aggregate output TPS:{agg_output_tps:.2f}')
print(f'Sum per-inst req/s:  {sum(per_instance_tps):.2f}')
print(f'Mean per-inst req/s: {statistics.mean(per_instance_tps) if per_instance_tps else 0:.2f}')
print()
print('--- Cache ---')
print(f' Block hits:     ${HITS_DELTA}')
print(f' Block misses:   ${MISSES_DELTA}')
print(f' H2D load (B):   ${LOAD_DELTA}')
print(f' D2H save (B):   ${SAVE_DELTA}')
hit_rate = 0
if ${HITS_DELTA} + ${MISSES_DELTA} > 0:
    hit_rate = ${HITS_DELTA} / (${HITS_DELTA} + ${MISSES_DELTA}) * 100
    print(f' Hit rate:       {hit_rate:.1f}%')
else:
    print(f' Hit rate:       N/A')
print()
print('--- Per-Instance ---')
print(f' {\"Instance\":<10} {\"Completed\":>10} {\"Duration(s)\":>12} {\"Req/s\":>10} {\"TTFT p50(ms)\":>14}')
for i, r in enumerate(instances):
    if isinstance(r, dict) and 'completed' in r:
        print(f' {i:<10} {r.get(\"completed\",0):>10} {safe(r.get(\"duration_s\")):>12.2f} {safe(r.get(\"request_throughput\")):>10.2f} {safe(r.get(\"median_ttft_ms\")):>14.2f}')
    else:
        print(f' {i:<10} {\"ERROR\":>10} {r.get(\"error\",\"?\")}')

if ${HITS_DELTA} > 0:
    print()
    print(' ✅ External cache HITs observed — KV cache sharing is working')
else:
    print()
    print(' ❌ No external cache HITs — KV cache sharing NOT effective')
    print('    Check: PYTHONHASHSEED=0 is set in all vLLM processes')

# Save aggregate
agg_out = {
    'tag': '${TAG}',
    'mode': 'pegaflow',
    'num_instances': num_instances,
    'requests_per_instance': ${NUM_REQUESTS},
    'concurrency_per_instance': ${CONCURRENCY},
    'total_completed': total_completed,
    'total_failed': total_failed,
    'wall_clock_s': ${BENCH_WALL},
    'max_duration_s': max_duration,
    'aggregate_req_per_sec': agg_req_per_sec,
    'aggregate_output_tps': agg_output_tps,
    'per_instance_tps': per_instance_tps,
    'cache_metrics': {
        'hits': ${HITS_DELTA},
        'misses': ${MISSES_DELTA},
        'load_bytes': ${LOAD_DELTA},
        'save_bytes': ${SAVE_DELTA},
        'hit_rate_pct': hit_rate,
    },
}
with open(os.path.join(results_dir, 'results.json'), 'w') as f:
    json.dump(agg_out, f, indent=2)
print(f'\nSaved: {results_dir}/results.json')
"

echo ""
echo "============================================================"
echo " Complete: ${TAG}"
echo " Results:  ${RESULTS_DIR}"
echo "============================================================"
ls -la "${RESULTS_DIR}/"
