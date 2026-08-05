#!/usr/bin/env python3
"""
Trace Audit: PegaFlow Ascend KV Transfer Break-Even (Host-Only Preregistration)

STATUS: Host-only — NOT yet executed on NPU.
        See docs/trace_preregistration.md for full experimental design.

Prior artifact: results/trace-audit-INVALID/ — methodological asymmetry discovered
and preserved as negative example (see preregistration Section 8).

Usage:
  python run_trace_audit.py --cycles 3 --requests-per-phase 3
"""

from __future__ import annotations

import argparse, hashlib, json, os, random, re, signal, subprocess, sys
import threading, time, urllib.request, urllib.error, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/workspace/HUST/pegaflow-hust")
_RUN_ID = time.strftime("%Y%m%d-%H%M%S")
LOG_DIR = Path(f"/workspace/HUST/pegaflow-hust/results/trace-audit/{_RUN_ID}/logs")
OUT_DIR = Path(f"/workspace/HUST/pegaflow-hust/results/trace-audit/{_RUN_ID}")

MODEL_PATH = "/workspace/HUST/models/Qwen3-8B"
VLLM_HUST_ROOT = Path("/workspace/HUST/vllm-hust")  # runtime vLLM checkout (A5)
SERVER_PORT = 50080
VLLM_BASE_PORT = 19000
SHARED_NS = "audit-shared"
ISOLATED_NS_PREFIX = "audit-iso"
NUM_INSTANCES = 8
HBM_TOTAL_MB = 65536
MIN_FREE_HBM_MB = 28 * 1024
DMA_TIME_WINDOW_S = 30.0           # max seconds between prefetch and DMA
ADMISSION_POLL_INTERVAL_S = 10.0   # mid-arm admission drift poll cadence

_SYS_BLOCK = (
    "You are an expert AI assistant with deep knowledge across many domains "
    "including computer science, mathematics, physics, biology, history, "
    "philosophy, literature, economics, law, medicine, engineering, and the "
    "arts. You provide accurate, detailed, and well-structured responses. "
    "Always follow instructions precisely and think step by step before "
    "answering. Your responses should be helpful, harmless, and honest. "
    "When asked factual questions, provide evidence-based answers with "
    "citations where possible. When asked for opinions, provide balanced "
    "perspectives that acknowledge multiple viewpoints. When the user asks "
    "you to perform a task, break it down into clear, actionable steps and "
    "explain your reasoning at each stage. When you encounter ambiguity, "
    "ask clarifying questions rather than making assumptions. Be mindful "
    "of the user's time and keep responses focused and relevant. If you are "
    "unsure about something, acknowledge it honestly rather than speculating. "
    "Your goal is to be as helpful as possible while maintaining high "
    "standards of accuracy and clarity. Remember to adapt your communication "
    "style to the user's level of expertise — use technical language when "
    "appropriate but be ready to explain concepts in simpler terms when "
    "needed. Always prioritize the user's safety and well-being, and avoid "
    "generating harmful, unethical, or dangerous content under any "
    "circumstances. You should respect user privacy and not ask for or "
    "store personal information unnecessarily."
)
SYSTEM_PROMPT = (_SYS_BLOCK + " ") * 38  # ~10k tokens

USER_QUERIES = [
    "What is the capital of France?",
    "Explain how photosynthesis works in plants.",
    "Who wrote the play Hamlet and when?",
    "What is Einstein's theory of relativity?",
    "Describe the water cycle in detail.",
    "How does an electric motor work?",
    "Name the planets in our solar system in order.",
    "What is machine learning and how does it differ from traditional programming?",
]


# =========================================================================
# Artifact binding
# =========================================================================

def capture_environment() -> dict:
    """Record everything needed to reproduce this run."""
    info: dict = {}

    # Git — HEAD, parent, and runtime vLLM checkout (A5: audit artifact must
    # bind more than a single HEAD so the exact analyzed code is recoverable).
    for _cmd, _key in [
        (["rev-parse", "HEAD"], "git_commit"),
        (["rev-parse", "HEAD^"], "git_parent"),
        (["rev-parse", "--abbrev-ref", "HEAD"], "git_branch"),
    ]:
        try:
            info[_key] = subprocess.check_output(
                ["git", "-C", str(PROJECT_ROOT)] + _cmd, timeout=10,
            ).decode().strip()
        except Exception:
            info[_key] = "unknown"
    try:
        info["runtime_commit_vllm"] = subprocess.check_output(
            ["git", "-C", str(VLLM_HUST_ROOT), "rev-parse", "HEAD"],
            timeout=10,
        ).decode().strip()
    except Exception:
        info["runtime_commit_vllm"] = "unknown"

    # Model
    cfg_path = Path(MODEL_PATH) / "config.json"
    if cfg_path.exists():
        cfg = json.load(open(cfg_path))
        info["model"] = str(Path(MODEL_PATH).name)
        info["model_arch"] = cfg.get("architectures", [])
        info["model_config_md5"] = hashlib.md5(
            open(cfg_path, "rb").read()
        ).hexdigest()

    # NPU
    try:
        info["npu_smi"] = subprocess.check_output(
            ["npu-smi", "info"], timeout=30,
        ).decode()
    except Exception:
        info["npu_smi"] = "unavailable"

    # Env
    info["env_vars"] = {
        k: os.environ.get(k, "")
        for k in ["PYTHONHASHSEED", "ASCEND_RT_VISIBLE_DEVICES",
                   "PEGAFLOW_HOST", "PEGAFLOW_PORT",
                   "LD_LIBRARY_PATH", "PATH"]
    }

    info["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return info


# =========================================================================
# NPU helpers
# =========================================================================

def get_npu_free_memory() -> dict[int, int]:
    free: dict[int, int] = {i: -1 for i in range(8)}
    try:
        out = subprocess.check_output(
            ["npu-smi", "info"], stderr=subprocess.STDOUT, timeout=30,
        ).decode()
    except Exception:
        return free
    lines = out.split("\n")
    current_npu: int | None = None
    for line in lines:
        if "Process id" in line and "Process name" in line:
            break
        m1 = re.match(r"\|\s*(\d+)\s+\d+\w+\s+\|", line)
        if m1:
            current_npu = int(m1.group(1))
            continue
        if current_npu is not None and current_npu < 8:
            m2 = re.search(r"(\d+)\s*/\s*(\d+)\s*\|?\s*$", line)
            if m2:
                used = int(m2.group(1))
                total = int(m2.group(2))
                free[current_npu] = total - used
                current_npu = None
    return free


def get_npu_processes() -> dict[int, list[int]]:
    """Parse npu-smi process table: NPU id -> attached pids."""
    procs: dict[int, list[int]] = {}
    try:
        out = subprocess.check_output(
            ["npu-smi", "info"], stderr=subprocess.STDOUT, timeout=30,
        ).decode()
    except Exception:
        return procs
    in_proc_table = False
    for line in out.split("\n"):
        if "Process id" in line and "Process name" in line:
            in_proc_table = True
            continue
        if not in_proc_table:
            continue
        m = re.match(r"\|\s*(\d+)\s+\d+\s+\|\s+(\d+)\s+(\S+)\s+\d+\s+\|", line)
        if m:
            procs.setdefault(int(m.group(1)), []).append(int(m.group(2)))
    return procs


def tracked_pgids() -> set[int]:
    """Process-group ids of everything this runner spawned."""
    pgids: set[int] = set()
    for pid in list(_tracked_pids):
        try:
            pgids.add(os.getpgid(pid))
        except (ProcessLookupError, PermissionError):
            pgids.add(pid)
    return pgids


def expected_used_mb(npu: int, free_mb: dict[int, int]) -> int:
    """Mirror of the gmu sizing used at launch — expected HBM footprint."""
    fm = free_mb.get(npu, -1)
    gmu = (max(0.15, min(0.85, (fm - 4096) / HBM_TOTAL_MB))
           if fm > 4096 else 0.15)
    return round(gmu * HBM_TOTAL_MB)


def check_admission_drift(
    admitted: list[int],
    free_mb_pre: dict[int, int],
    free_mb_now: dict[int, int],
    expected_used_mb_by_npu: dict[int, int],
    npu_procs: dict[int, list[int]],
    tracked_pgids_set: set[int],
    pgid_of=os.getpgid,
    slack_mb: int = 8 * 1024,
    min_free_mb: int = MIN_FREE_HBM_MB,
) -> list[str]:
    """Re-verify admission during arm execution (P2-6): owner PID + HBM.

    free_mb_pre / free_mb_now: free HBM sampled at admission vs mid-arm.
    expected_used_mb_by_npu: MB our own instances were expected to consume.
    npu_procs: NPU id -> pids attached mid-arm (from get_npu_processes).
    tracked_pgids_set: process groups we spawned (ours).

    Returns violation strings; empty list == no drift, admission holds.
    """
    violations: list[str] = []
    for npu in admitted:
        pre = free_mb_pre.get(npu, -1)
        now = free_mb_now.get(npu, -1)
        exp = expected_used_mb_by_npu.get(npu, 0)
        floor = min(min_free_mb, pre - exp - slack_mb) if pre >= 0 else min_free_mb
        if now < floor:
            violations.append(
                f"NPU{npu} HBM drift: free={now}MB < floor={floor}MB "
                f"(pre={pre}MB, expected_use={exp}MB, slack={slack_mb}MB)")
        attached = npu_procs.get(npu, [])
        if not attached:
            violations.append(
                f"NPU{npu} owner drift: no process attached (expected our instance)")
            continue
        for pid in attached:
            try:
                owned = pid in tracked_pgids_set or pgid_of(pid) in tracked_pgids_set
            except (ProcessLookupError, PermissionError):
                owned = False  # cannot verify -> fail-close
            if not owned:
                violations.append(
                    f"NPU{npu} owner drift: foreign pid={pid} attached to admitted device")
    return violations


# =========================================================================
# Process management
# =========================================================================

# Tracked child processes — cleanup only kills what we spawned.
_tracked_pids: list[int] = []


def _track_proc(proc: subprocess.Popen) -> None:
    _tracked_pids.append(proc.pid)


def kill_tracked() -> None:
    """Kill only processes spawned by this runner, never touch external tasks."""
    for pid in list(_tracked_pids):
        try:
            os.kill(-pid, signal.SIGTERM)  # negative = process group
        except (ProcessLookupError, PermissionError):
            pass
    for pid in list(_tracked_pids):
        try:
            os.kill(-pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    _tracked_pids.clear()


def stop_proc(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def start_server(pool_size="4096mb", log_dir=None, devices=None):
    if devices is None:
        devices = list(range(NUM_INSTANCES))
    if log_dir is None:
        log_dir = LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / "server.log"
    proc = subprocess.Popen(
        [
            str(PROJECT_ROOT / "target" / "debug" / "pegaflow-server"),
            "--addr", f"0.0.0.0:{SERVER_PORT}",
            "--pool-size", pool_size,
            "--devices", ",".join(str(d) for d in devices),
        ],
        stdout=open(log, "w"), stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    _track_proc(proc)
    deadline = time.time() + 30
    while time.time() < deadline:
        time.sleep(1)
        try:
            if log.exists() and "listening" in log.read_text():
                return proc
        except Exception:
            pass
    raise RuntimeError("Server failed to start")


def start_vllm(port, mode, namespace, physical_npu, label, *,
               model_path, gpu_memory_utilization=0.85,
               use_pegaflow=True):
    log = LOG_DIR / f"vllm_{label}.log"
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(physical_npu)
    if use_pegaflow:
        if namespace:
            env["PEGAFLOW_NAMESPACE"] = namespace
        env["PEGAFLOW_HOST"] = "http://127.0.0.1"
        env["PEGAFLOW_PORT"] = str(SERVER_PORT)
    gmu = gpu_memory_utilization
    cmd_parts = [
        f"source /root/miniconda3/etc/profile.d/conda.sh && conda activate vllm-hust-dev",
        f"vllm serve {model_path} --port {port} --dtype float16",
        f"--max-model-len 16384 --max-num-seqs 4",
        f"--gpu-memory-utilization {gmu:.2f}",
    ]
    if use_pegaflow:
        kv_cfg = json.dumps({
            "kv_connector": "PegaKVConnector", "kv_role": "kv_both",
            "kv_connector_module_path": "pegaflow.connector",
            "kv_connector_extra_config": {
                "pegaflow.mode": mode,
                "pegaflow.transfer_backend": "ascend_direct",
            },
        })
        cmd_parts.append(f"--kv-transfer-config '{kv_cfg}'")
    cmd = " && ".join([cmd_parts[0], " ".join(cmd_parts[1:])])
    proc = subprocess.Popen(
        ["bash", "-c", cmd], env=env,
        stdout=open(log, "w"), stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    _track_proc(proc)
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            return proc
        except Exception:
            time.sleep(3)
    raise RuntimeError(f"vLLM {label} failed")


def launch_all_instances(specs, model_path):
    """Start all instances in parallel, return [(spec, proc), ...]."""
    running: list[tuple[dict, subprocess.Popen]] = []
    rlock = threading.Lock()

    def _start_one(spec):
        npu = spec["physical_npu"]
        fm = get_npu_free_memory().get(npu, -1)
        gmu = max(0.15, min(0.85, (fm - 4096) / HBM_TOTAL_MB))
        label = spec["label"]
        print(f"    → [{label}] NPU{npu} gmu={gmu:.2f} ...")
        try:
            proc = start_vllm(
                spec["port"], spec["mode"], spec["namespace"],
                npu, label, model_path=model_path,
                gpu_memory_utilization=gmu,
                use_pegaflow=spec.get("use_pegaflow", True),
            )
            with rlock:
                running.append((spec, proc))
            print(f"    → [{label}] ready ({len(running)}/{len(specs)})")
        except Exception as e:
            print(f"    → [{label}] FAILED: {e}")

    with ThreadPoolExecutor(max_workers=len(specs)) as ex:
        futures = [ex.submit(_start_one, s) for s in specs]
        for _ in as_completed(futures, timeout=300):
            pass
    return running


# =========================================================================
# Streaming request + log parsing
# =========================================================================

def send_one_streaming(port, prompt, model_path, max_tokens=64, timeout=600):
    """Send streaming request, return {ttft_s, total_s, text, ok, req_id}.

    The client-generated request_id (UUID) is passed to vLLM via the
    request body. vLLM uses it as EngineCoreRequest.request_id, which
    flows through connector cache_lookup logs and server prefetch logs.
    """
    client_req_id = f"trace-{uuid.uuid4().hex[:12]}"
    data = json.dumps({
        "model": model_path, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0.0,
        "stream": True,
        "request_id": client_req_id,
    }).encode()
    t0 = time.perf_counter()
    ttft_s = -1.0
    total_s = -1.0
    text = ""
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            data=data, headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=timeout)
        for line_bytes in resp:
            line = line_bytes.decode().strip()
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                if ttft_s < 0:
                    ttft_s = time.perf_counter() - t0
                choices = chunk.get("choices", [])
                if choices:
                    text += choices[0].get("text", "")
            except json.JSONDecodeError:
                continue
        total_s = time.perf_counter() - t0
        return {"ttft_s": round(ttft_s, 4), "total_s": round(total_s, 4),
                "text": text[:80], "ok": True, "req_id": client_req_id}
    except Exception as e:
        return {"ttft_s": -1, "total_s": -1, "text": "", "ok": False,
                "error": str(e)[:200], "req_id": client_req_id}


def extract_dma_from_server_log(req_id: str) -> dict:
    """Parse server log for DMA details matching req_id."""
    server_log = LOG_DIR / "server.log"
    if not server_log.exists():
        return {}
    text = server_log.read_text()
    dma_info: dict = {"hit_blocks": 0, "missing_blocks": 0,
                       "dma_bytes": 0, "dma_ms": 0.0, "dma_gbps": 0.0}

    # Find prefetch line for this req_id
    pattern = rf"req_id={re.escape(req_id)}.*?total_keys=(\d+)\s+hit=(\d+)\s+missing=(\d+)"
    m = re.search(pattern, text)
    if m:
        dma_info["total_keys"] = int(m.group(1))
        dma_info["hit_blocks"] = int(m.group(2))
        dma_info["missing_blocks"] = int(m.group(3))

    # Find DMA completion line following the prefetch for this device
    # We can't tie DMA to specific req_id from the log, so grab the
    # nearest DMA line after the prefetch timestamp
    return dma_info


def extract_vllm_timing(log_text: str) -> dict[str, dict]:
    """Parse per-request prefill/queue timing from a vLLM log (A2).

    Matches vLLM "Finished request <req_id>: <key=value ...>" lines and
    keeps prompt_processing_ms -> prefill_time_ms and queue_time ->
    queue_time_ms (both ms). Unknown keys are ignored; requests without a
    matching line are simply absent (merge fills -1 defaults — these fields
    are informational per-request timing, not fail-close evidence).
    """
    timing: dict[str, dict] = {}
    for m in re.finditer(r"Finished request (?P<req>\S+):\s*(?P<kv>.*)",
                         log_text):
        entry = timing.setdefault(m.group("req"), {})
        for kv in re.finditer(r"(\w+)=([\d.+-]+)", m.group("kv")):
            key, val = kv.group(1), float(kv.group(2))
            if key == "prompt_processing_ms":
                entry["prefill_time_ms"] = val
            elif key == "queue_time":
                entry["queue_time_ms"] = val
    return timing


def extract_connector_log(vllm_log_path: Path, label: str) -> list[dict]:
    """Parse vLLM log for PegaKVConnector cache_lookup lines."""
    if not vllm_log_path.exists():
        return []
    entries = []
    text = vllm_log_path.read_text()
    for m in re.finditer(
        r"\[PegaKVConnector\] req=(?P<req_id>\S+)\s+"
        r"cache_lookup: hit_blocks=(?P<hit>\d+) "
        r"computed_blocks=(?P<computed>\d+) "
        r"hit_tokens=(?P<hit_tokens>\d+) num_tokens=(?P<num_tokens>\d+).*?",
        text,
    ):
        entries.append({
            "req_id": m.group("req_id"),
            "label": label,
            "hit_blocks": int(m.group("hit")),
            "computed_blocks": int(m.group("computed")),
            "hit_tokens": int(m.group("hit_tokens")),
            "num_tokens": int(m.group("num_tokens")),
        })
    return entries


# =========================================================================
# Phase runner
# =========================================================================

def run_phase_sequential(phase_name, instances, queries, model_path,
                         warmup_first, cycle):
    """Send requests one at a time (round-robin), return per-request records."""
    records: list[dict] = []
    t0 = time.perf_counter()

    # Warmup (if applicable) — produces a record marked producer=True.
    # Producer = the single warmup REQUEST only (prereg §2.3 step 4: timed
    # phase is 3 queries x 8 instances = 24 consumers per arm). Marking the
    # whole warmup instance as producer would silently drop its Q1/Q2 (which
    # are ordinary consumers) and shrink analyzed N to 21.
    if warmup_first and len(instances) >= 1:
        warmup_spec, warmup_proc = instances[0]
        prompt = f"{SYSTEM_PROMPT}\n\nUser: {queries[0]}\n\nAssistant:"
        t_req = time.perf_counter()
        r = send_one_streaming(warmup_spec["port"], prompt, model_path)
        records.append({
            "cycle": cycle, "phase": phase_name, "req_idx": -1,
            "query_idx": -1, "instance": warmup_spec["label"],
            "npu": warmup_spec["physical_npu"],
            "port": warmup_spec["port"],
            "query": queries[0][:50],
            **r,
            "wall_clock_at_send": round(t_req - t0, 4),
            "producer": True,  # seeds cache, not a consumer
        })
        print(f"    [WARMUP] {warmup_spec['label']} "
              f"TTFT={r['ttft_s']:.4f}s ok={r['ok']}")
        time.sleep(30)

    # Timed phase
    idx = 0
    for qi in range(len(queries)):
        for spec, proc in instances:
            if proc.poll() is not None:
                continue
            q = queries[qi]
            prompt = f"{SYSTEM_PROMPT}\n\nUser: {q}\n\nAssistant:"
            t_req = time.perf_counter()
            r = send_one_streaming(spec["port"], prompt, model_path)
            record = {
                "cycle": cycle,
                "phase": phase_name,
                "req_idx": idx,
                "query_idx": qi,
                "instance": spec["label"],
                "npu": spec["physical_npu"],
                "port": spec["port"],
                "query": q[:50],
                **r,
                "wall_clock_at_send": round(t_req - t0, 4),
                # Timed-phase requests are consumers — producer flag is
                # reserved for the warmup seed request itself (prereg §2.3).
                "producer": False,
            }
            records.append(record)
            status = (f"TTFT={r['ttft_s']:.4f}s" if r["ok"]
                      else f"ERR={r.get('error','?')[:30]}")
            print(f"    [{idx:>3d}] {spec['label']} Q{qi} {status} | {q[:30]}")
            idx += 1
            time.sleep(0.5)
    return records


# =========================================================================
# Pure matching function — testable without hardware
# =========================================================================

def merge_by_req_id(
    all_records: list[dict],
    connector_by_req: dict[str, dict],
    prefetch_by_req: dict[str, dict],
    prefetch_dma_map: dict[str, dict],
    dma_leftover_count: int = 0,
    dma_fallback_only_count: int = 0,
    timing_by_req: dict | None = None,
) -> dict:
    """Merge per-request hit/DMA data by client-generated request_id.

    Client sends UUID in JSON body. vLLM prepends "cmpl-" and appends
    "-{idx}-{hash}". We match by substring containment.

    Fail-close conservation (P2-6): connector / prefetch / DMA events are
    formal evidence. Every consumer request must have exactly one connector
    event and one prefetch event; a request with hit_blocks > 0 additionally
    requires exactly one DMA event. Duplicate events for one request, orphan
    events matching no request, and leftover DMA completions all break
    conservation and invalidate the run (never "first match wins").

    Records are flagged with _audit_invalid / _audit_reason when their
    evidence chain is broken; producer (warmup seed) events are claimed so
    they do not count as orphans.

    Returns dict with matched/unmatched/coverage plus per-type uniqueness
    and conservation accounting; every violation is listed in `violations`.
    """
    matched = 0
    unmatched = 0
    producer_skipped = 0
    total_consumers = 0
    violations: list[str] = []
    invalid_records: list[dict] = []
    connector_duplicates = 0  # cross-key + occurrence duplicates

    def occ(ev: dict | None) -> int:
        return int(ev.get("occurrences", 1)) if ev else 0

    claimed: set[str] = set()           # connector keys claimed
    prefetch_claimed: set[str] = set()  # prefetch keys claimed
    dma_claimed: set[str] = set()       # DMA keys claimed

    # Producer claims: warmup-seed connector/prefetch/DMA events are
    # legitimate (they seeded the cache) — claim so they are not orphans.
    for r in all_records:
        if not r.get("producer"):
            continue
        client_req_id = r.get("req_id", "")
        keys = [k for k in connector_by_req if client_req_id in k]
        if len(keys) > 1:
            violations.append(
                f"duplicate connector event (producer): "
                f"req={client_req_id} keys={keys}")
        if keys:
            claimed.add(keys[0])
            if keys[0] in prefetch_by_req:
                prefetch_claimed.add(keys[0])
            if keys[0] in prefetch_dma_map:
                dma_claimed.add(keys[0])

    for r in all_records:
        r.setdefault("hit_blocks", 0)
        r.setdefault("hit_tokens", 0)
        r.setdefault("missing_blocks", 0)
        r.setdefault("dma_bytes", 0)
        r.setdefault("dma_ms", 0.0)
        r.setdefault("dma_gbps", 0.0)
        # A2: per-request prefill/queue timing (informational, default -1)
        r.setdefault("prefill_time_ms", -1.0)
        r.setdefault("queue_time_ms", -1.0)

        if r.get("producer"):
            producer_skipped += 1
            continue

        total_consumers += 1
        client_req_id = r.get("req_id", "")
        keys = [k for k in connector_by_req if client_req_id in k]

        if not keys:
            unmatched += 1
            reason = f"missing connector event: req={client_req_id}"
            r["_audit_invalid"] = True
            r["_audit_reason"] = reason
            violations.append(reason)
            invalid_records.append({"req_id": client_req_id,
                                    "reason": "missing connector event"})
            continue

        if len(keys) > 1:
            connector_duplicates += len(keys) - 1
            reason = (f"duplicate connector event: req={client_req_id} "
                      f"keys={keys}")
            r["_audit_invalid"] = True
            r["_audit_reason"] = reason
            violations.append(reason)
            invalid_records.append({"req_id": client_req_id,
                                    "reason": f"duplicate connector event ({len(keys)})"})

        conn_key = keys[0]
        cinfo = connector_by_req[conn_key]
        claimed.add(conn_key)
        matched += 1
        r["hit_blocks"] = cinfo["hit_blocks"]
        r["hit_tokens"] = cinfo["hit_tokens"]
        r["num_tokens_total"] = cinfo.get("num_tokens", 0)
        if timing_by_req:
            t = timing_by_req.get(conn_key)
            if t:
                if "prefill_time_ms" in t:
                    r["prefill_time_ms"] = t["prefill_time_ms"]
                if "queue_time_ms" in t:
                    r["queue_time_ms"] = t["queue_time_ms"]

        # Prefetch: exactly one server-side prefetch for this connector
        pref = prefetch_by_req.get(conn_key)
        if pref is None:
            reason = (f"missing prefetch event: req={client_req_id} "
                      f"server={conn_key}")
            r["_audit_invalid"] = True
            r["_audit_reason"] = reason
            violations.append(reason)
            invalid_records.append({"req_id": client_req_id,
                                    "reason": "missing prefetch event"})
        else:
            prefetch_claimed.add(conn_key)
            if occ(pref) > 1:
                reason = (f"duplicate prefetch event: server={conn_key} "
                          f"occurrences={occ(pref)}")
                r["_audit_invalid"] = True
                r["_audit_reason"] = reason
                violations.append(reason)
                invalid_records.append({"req_id": client_req_id,
                                        "reason": "duplicate prefetch event"})
            r["missing_blocks"] = pref.get("missing_blocks", 0)

        # DMA: required iff the request actually hit and transfers KV.
        # A claimed hit with no DMA evidence is INVALID, not a silent 0.
        dma = prefetch_dma_map.get(conn_key)
        dma_required = r["hit_blocks"] > 0
        if dma_required:
            if dma is None:
                reason = (f"missing DMA event: req={client_req_id} "
                          f"hit_blocks={r['hit_blocks']} "
                          f"(transfer claimed, no DMA evidence)")
                r["_audit_invalid"] = True
                r["_audit_reason"] = reason
                violations.append(reason)
                invalid_records.append({"req_id": client_req_id,
                                        "reason": "missing DMA event"})
            else:
                dma_claimed.add(conn_key)
                if occ(dma) > 1:
                    reason = (f"duplicate DMA event: server={conn_key} "
                              f"occurrences={occ(dma)}")
                    r["_audit_invalid"] = True
                    r["_audit_reason"] = reason
                    violations.append(reason)
                    invalid_records.append({"req_id": client_req_id,
                                            "reason": "duplicate DMA event"})
                r["dma_bytes"] = dma["dma_bytes"]
                r["dma_ms"] = dma["dma_ms"]
                r["dma_gbps"] = dma["dma_gbps"]

    # Orphan accounting: produced-but-unclaimed formal events
    connector_orphans = [k for k in connector_by_req if k not in claimed]
    for k in connector_orphans:
        violations.append(f"orphan connector event: {k}")
    prefetch_orphans = [k for k in prefetch_by_req if k not in prefetch_claimed]
    for k in prefetch_orphans:
        violations.append(f"orphan prefetch event: {k}")
    dma_orphans = [k for k in prefetch_dma_map if k not in dma_claimed]
    for k in dma_orphans:
        violations.append(f"orphan DMA event: {k}")
    if dma_leftover_count > 0:
        violations.append(
            f"leftover DMA completions: {dma_leftover_count} unbound to any prefetch")
    if dma_fallback_only_count > 0:
        # R6: per-copy fallback is NOT formal batch-DMA evidence — every
        # such request is already flagged "missing DMA event" above.
        violations.append(
            f"fallback-only DMA (no batch DMA evidence): "
            f"{dma_fallback_only_count} request(s)")

    # Duplicates beyond the first occurrence, per event type.
    # connector_duplicates also accumulates cross-key duplicates (two keys
    # matching the same client req) from the consumer loop above.
    connector_duplicates += sum(
        occ(ev) - 1 for ev in connector_by_req.values() if occ(ev) > 1)
    prefetch_duplicates = sum(
        occ(ev) - 1 for ev in prefetch_by_req.values() if occ(ev) > 1)
    dma_duplicates = sum(
        occ(ev) - 1 for ev in prefetch_dma_map.values() if occ(ev) > 1)

    coverage_pct = (matched / total_consumers * 100.0
                    if total_consumers > 0 else 0.0)
    # Fail-close: conservation breaks on ANY missing/duplicate/orphan/
    # leftover formal event, on unmatched consumers, and on records whose
    # evidence chain is broken — coverage alone is not enough.
    conservation_ok = (
        unmatched == 0 and not invalid_records
        and not connector_orphans and not prefetch_orphans and not dma_orphans
        and connector_duplicates == 0 and prefetch_duplicates == 0
        and dma_duplicates == 0 and dma_leftover_count == 0
        and dma_fallback_only_count == 0
    )
    return {
        "matched": matched,
        "unmatched": unmatched,
        "producer_skipped": producer_skipped,
        "total_consumers": total_consumers,
        "coverage_pct": round(coverage_pct, 1),
        "connector_total": len(connector_by_req),
        "connector_consumed": len(claimed),
        "connector_duplicates": connector_duplicates,
        "connector_orphans": len(connector_orphans),
        "prefetch_total": len(prefetch_by_req),
        "prefetch_consumed": len(prefetch_claimed),
        "prefetch_duplicates": prefetch_duplicates,
        "prefetch_orphans": len(prefetch_orphans),
        "dma_total": len(prefetch_dma_map),
        "dma_consumed": len(dma_claimed),
        "dma_duplicates": dma_duplicates,
        "dma_orphans": len(dma_orphans),
        "dma_leftover": dma_leftover_count,
        "dma_fallback_only": dma_fallback_only_count,
        "conservation_ok": conservation_ok,
        "violations": violations,
        "invalid_records": invalid_records,
    }


def bind_dma_to_prefetch(
    ts_prefetches: list[dict],
    ts_dmas: list[dict],
    connector_by_req: dict[str, dict],
    label_to_npu: dict[str, int],
    dma_time_window_s: float = DMA_TIME_WINDOW_S,
) -> tuple[dict, int, int, list[str]]:
    """Bind DMA completions to prefetch events (pure — R9).

    Nearest subsequent DMA on the same device + same arm + within the time
    window binds to each hit>0 prefetch, 1:1 (bound DMA is consumed).

    R6: a per-copy fallback line is NOT formal batch-DMA evidence — a hit
    prefetch with only a fallback candidate binds nothing and is counted in
    fallback_only_count; merge_by_req_id then flags it "missing DMA event"
    and marks the record _audit_invalid.

    Returns (prefetch_dma_map, fallback_only_count, leftover_count,
             violations). Mutates ts_dmas (consumes bound entries).
    """
    prefetch_dma_map: dict[str, dict] = {}
    fallback_only_count = 0
    violations: list[str] = []
    ts_fmt = "%Y-%m-%dT%H:%M:%S.%f"

    for pf in ts_prefetches:
        if pf["hit"] == 0:
            continue  # no DMA for miss
        req_id = pf["req_id"]
        # Target device from connector label (label format: C1_shared_3 → NPU 3)
        target_device = None
        if req_id in connector_by_req:
            target_device = label_to_npu.get(
                connector_by_req[req_id].get("label", ""), None
            )
        pf_arm = pf.get("arm_label", "")
        best, best_fallback = None, None
        for dma in ts_dmas:
            if dma["ts"] <= pf["ts"]:
                continue
            if dma.get("arm_label", "") != pf_arm:
                continue  # P1-4: arm scope
            if target_device is not None and dma["device_id"] != target_device:
                continue  # P1-4: device scope
            # P1-4 / P2-3: full ISO timestamp time window (cross-minute safe).
            try:
                pf_dt = datetime.strptime(pf["ts"][:26], ts_fmt)
                dma_dt = datetime.strptime(dma["ts"][:26], ts_fmt)
                if abs((dma_dt - pf_dt).total_seconds()) > dma_time_window_s:
                    continue
            except (ValueError, IndexError):
                pass
            if dma.get("fallback"):
                if best_fallback is None or dma["ts"] < best_fallback["ts"]:
                    best_fallback = dma
            else:
                if best is None or dma["ts"] < best["ts"]:
                    best = dma
        if best is None:
            # R6: fallback-only is not formal evidence — no binding.
            if best_fallback is not None:
                fallback_only_count += 1
                violations.append(
                    f"fallback-only DMA (no batch DMA evidence): req={req_id}")
                try:
                    ts_dmas.remove(best_fallback)
                except ValueError:
                    pass
            continue
        # P2-6: two DMA completions for the same prefetch → count, not overwrite.
        entry = prefetch_dma_map.get(req_id)
        if entry is not None:
            entry["occurrences"] = entry.get("occurrences", 1) + 1
            continue
        prefetch_dma_map[req_id] = {
            "dma_ms": best["dma_ms"],
            "dma_bytes": best["dma_bytes"],
            "dma_gbps": best["dma_gbps"],
            "occurrences": 1,
        }
        ts_dmas.remove(best)  # consume — one DMA per prefetch

    return prefetch_dma_map, fallback_only_count, len(ts_dmas), violations


def monitor_admission_drift(
    admitted: list[int],
    free_mb_pre: dict[int, int],
    expected_used_mb_by_npu: dict[int, int],
    tracked_pgids_set: set[int],
    stop_event: threading.Event,
    violations_out: list[str],
    interval_s: float = ADMISSION_POLL_INTERVAL_S,
    sampler=None,
    pgid_of=os.getpgid,
) -> None:
    """Periodically re-verify admission during a phase (P2-6/R7).

    Boundary sampling misses transient drift mid-phase; this thread polls
    check_admission_drift every interval_s until stop_event is set. Every
    drift found is printed as [INVALID] and appended to violations_out
    (the run then exits non-zero at the final gate).

    sampler() -> (free_mb_now, npu_procs) — injectable for host-only tests.
    """
    def _sample() -> tuple[dict[int, int], dict[int, list[int]]]:
        return get_npu_free_memory(), get_npu_processes()

    sampler = sampler or _sample
    while not stop_event.wait(interval_s):
        try:
            free_now, npu_procs = sampler()
        except Exception as e:
            violations_out.append(f"admission monitor sample failed: {e}")
            continue
        drift = check_admission_drift(
            admitted, free_mb_pre, free_now, expected_used_mb_by_npu,
            npu_procs, tracked_pgids_set, pgid_of=pgid_of)
        for v in drift:
            print(f"  [INVALID] {v}")
            violations_out.append(v)


# =========================================================================
# Statistics
# =========================================================================

def compute_paired_analysis(shared: list[dict], isolated: list[dict]) -> dict:
    """Per-query-class paired analysis (prereg §4.1/§4.4, C5, D5).

    Inputs: consumer records (ok, ttft_s > 0) of ONE query class, per arm.
    Pairing key: (cycle, physical_npu) — each NPU is a consumer instance in
    both arms (D5). Everything else is derived from the per-instance paired
    deltas: median prefill saving, per-instance median deltas, and a
    lifecycle cluster bootstrap CI (one mean-delta per cycle, resampled 1000
    times, seed 42 — C5: stratified per query class, producers excluded).

    Returns dict with n, medians, prefill_saved_ms, dma_cost_ms,
    per_instance_deltas_ms, cluster_ci (low, high), significant, verdict
    ('GO' when prefill_saved > dma_cost AND CI excludes 0, else 'BREAK-EVEN').
    """
    def _by(recs):
        return {(r.get("cycle"), r.get("npu")): r for r in recs}

    s_map, i_map = _by(shared), _by(isolated)
    pairs = []
    for key in sorted(set(s_map) & set(i_map)):
        s, i = s_map[key], i_map[key]
        pairs.append((key, i["ttft_s"] - s["ttft_s"]))
    deltas_ms = [d * 1000.0 for _, d in pairs]

    # Per-instance median paired delta (D5)
    per_inst: dict[int, list[float]] = {}
    for (_, npu), d in pairs:
        per_inst.setdefault(npu, []).append(d * 1000.0)
    per_instance_deltas_ms = {
        npu: sorted(v)[len(v) // 2] for npu, v in sorted(per_inst.items())}

    # Lifecycle cluster bootstrap CI, stratified per class (C5)
    cluster_deltas: dict[int, list[float]] = {}
    for (cycle, _), d in pairs:
        cluster_deltas.setdefault(cycle, []).append(d * 1000.0)
    cluster_means = [sum(v) / len(v) for v in cluster_deltas.values()]
    if cluster_means:
        random.seed(42)
        boot = []
        for _ in range(1000):
            sample = [random.choice(cluster_means) for __ in range(len(cluster_means))]
            boot.append(sum(sample) / len(sample))
        boot.sort()
        cluster_ci = (round(boot[25], 1), round(boot[974], 1))
        significant = cluster_ci[0] > 0 or cluster_ci[1] < 0
    else:
        cluster_ci = (0.0, 0.0)
        significant = False

    # Break-even: prefill_saved vs DMA cost (prereg §4.4)
    s_vals = sorted(r["ttft_s"] for r in shared)
    i_vals = sorted(r["ttft_s"] for r in isolated)
    s_med = s_vals[len(s_vals) // 2] if s_vals else 0.0
    i_med = i_vals[len(i_vals) // 2] if i_vals else 0.0
    prefill_saved_ms = (i_med - s_med) * 1000.0
    dma_vals = [r.get("dma_ms", 0) for r in shared if r.get("dma_ms", 0) > 0]
    dma_cost_ms = (sum(dma_vals) / len(dma_vals)) if dma_vals else None

    if dma_cost_ms is not None and prefill_saved_ms > dma_cost_ms and significant:
        verdict = "GO"
    else:
        verdict = "BREAK-EVEN"

    return {
        "n": len(pairs),
        "shared_median": round(s_med, 4),
        "isolated_median": round(i_med, 4),
        "prefill_saved_ms": round(prefill_saved_ms, 1),
        "dma_cost_ms": (round(dma_cost_ms, 1) if dma_cost_ms is not None else None),
        "per_instance_deltas_ms": per_instance_deltas_ms,
        "cluster_ci": cluster_ci,
        "significant": significant,
        "verdict": verdict,
    }


def compute_stats(records: list[dict], key="ttft_s") -> dict:
    vals = sorted(r[key] for r in records if r.get(key, -1) > 0)
    if not vals:
        return {"n": 0, "median": 0, "mean": 0, "std": 0}
    n = len(vals)
    mean = sum(vals) / n
    median = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    # 95% CI via bootstrap percentile
    random.seed(42)
    boot_means = []
    for _ in range(1000):
        sample = [random.choice(vals) for __ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    ci_low = boot_means[25]
    ci_high = boot_means[974]
    q1 = vals[n // 4] if n >= 4 else vals[0]
    q3 = vals[3 * n // 4] if n >= 4 else vals[-1]
    return {
        "n": n, "mean": round(mean, 4), "median": round(median, 4),
        "iqr": round(q3 - q1, 4),
        "ci_95_low": round(ci_low, 4), "ci_95_high": round(ci_high, 4),
        "min": round(vals[0], 4), "max": round(vals[-1], 4),
    }


# =========================================================================
# Summary + break-even
# =========================================================================

def write_summary(env_info, all_records, negative_examples, merge_result,
                  out_dir, drift_violations=None):
    """Write trace_summary.md with median/CI and break-even analysis.

    Returns True if every validity gate passes (fail-close), else False —
    the caller must then exit non-zero.
    """
    drift_violations = drift_violations or []
    # P2-5: separate producer (warmup seed) from consumer (cross-instance).
    # Producer records seed the shared cache but are not themselves consumers.
    shared = [r for r in all_records
              if r.get("phase") == "shared" and r.get("ok") and not r.get("producer")]
    isolated = [r for r in all_records
                if r.get("phase") == "isolated" and r.get("ok") and not r.get("producer")]
    producers = [r for r in all_records if r.get("ok") and r.get("producer")]

    ttft_s = compute_stats(shared, "ttft_s")
    ttft_i = compute_stats(isolated, "ttft_s")
    total_s = compute_stats(shared, "total_s")
    total_i = compute_stats(isolated, "total_s")

    lines = [
        "# Trace Audit Summary",
        "",
        "## Environment",
        f"- Commit: `{env_info.get('git_commit','?')[:12]}` "
        f"(parent: `{env_info.get('git_parent','?')[:12]}`)",
        f"- Branch: `{env_info.get('git_branch','?')}`",
        f"- Runtime vLLM commit: `{env_info.get('runtime_commit_vllm','?')[:12]}`",
        f"- Model: `{env_info.get('model','?')}` "
        f"(md5: `{env_info.get('model_config_md5','?')[:12]}`)",
        f"- Timestamp: {env_info.get('timestamp','?')}",
        f"- NPUs: 8× Ascend 910B2 (see artifact for full npu-smi)",
        "",
        "## TTFT (Time-To-First-Token)",
        "",
        "| Phase | N | Median | Mean | IQR | 95% CI | Min | Max |",
        "|---|---|---|---|---|---|---|---|",
        f"| Shared | {ttft_s['n']} | **{ttft_s['median']:.4f}s** | "
        f"{ttft_s['mean']:.4f}s | {ttft_s['iqr']:.4f}s | "
        f"[{ttft_s['ci_95_low']:.4f}, "
        f"{ttft_s['ci_95_high']:.4f}] | {ttft_s['min']:.4f}s | "
        f"{ttft_s['max']:.4f}s |",
        f"| Isolated | {ttft_i['n']} | **{ttft_i['median']:.4f}s** | "
        f"{ttft_i['mean']:.4f}s | {ttft_i['iqr']:.4f}s | "
        f"[{ttft_i['ci_95_low']:.4f}, "
        f"{ttft_i['ci_95_high']:.4f}] | {ttft_i['min']:.4f}s | "
        f"{ttft_i['max']:.4f}s |",
        "",
        "## Total Latency",
        "",
        "| Phase | N | Median | Mean | IQR | 95% CI |",
        "|---|---|---|---|---|---|",
        f"| Shared | {total_s['n']} | {total_s['median']:.3f}s | "
        f"{total_s['mean']:.3f}s | {total_s['iqr']:.3f}s | "
        f"[{total_s['ci_95_low']:.3f}, "
        f"{total_s['ci_95_high']:.3f}] |",
        f"| Isolated | {total_i['n']} | {total_i['median']:.3f}s | "
        f"{total_i['mean']:.3f}s | {total_i['iqr']:.3f}s | "
        f"[{total_i['ci_95_low']:.3f}, "
        f"{total_i['ci_95_high']:.3f}] |",
        "",
    ]

    # Per-query paired break-even (NOT mixed-class mean)
    lines += [
        "",
        "## Per-Query Paired Analysis",
        "",
        "Q0 measures cross-instance cold prefill (cache hit vs full compute).",
        "Q1/Q2 measure same-instance prefix cache (vLLM internal).",
        "Mixed-class mean conflates these — per-query pairing is the correct metric.",
        "",
    ]
    for qidx in sorted(set(r["query_idx"] for r in all_records)):
        s_subset = [r for r in shared if r.get("query_idx") == qidx
                    and r.get("ok") and r.get("ttft_s", -1) > 0]
        i_subset = [r for r in isolated if r.get("query_idx") == qidx
                    and r.get("ok") and r.get("ttft_s", -1) > 0]
        if not s_subset or not i_subset:
            lines += [
                f"### Q{qidx}",
                "- No paired observations (arm aborted by fail-close).",
                "- Verdict: BREAK-EVEN (no data)",
            ]
            continue
        pa = compute_paired_analysis(s_subset, i_subset)
        per_inst_str = ", ".join(
            f"NPU{npu}:{d:+.1f}ms" for npu, d in pa["per_instance_deltas_ms"].items())
        dma_note = (f"{pa['dma_cost_ms']:.1f}ms" if pa["dma_cost_ms"] is not None
                    else "n/a (no DMA evidence)")
        ci_note = (f"CI [{pa['cluster_ci'][0]:.1f}, {pa['cluster_ci'][1]:.1f}]ms "
                   f"{'excludes 0' if pa['significant'] else 'includes 0'}")
        lines += [
            f"### Q{qidx}",
            f"- Shared: n={len(s_subset)}, median={pa['shared_median']:.4f}s",
            f"- Isolated: n={len(i_subset)}, median={pa['isolated_median']:.4f}s",
            f"- Prefill saved (median): {pa['prefill_saved_ms']:+.1f}ms",
            f"- DMA cost (per-request, bound): {dma_note}",
            f"- Per-instance median paired delta (D5): {per_inst_str}",
            f"- Lifecycle cluster CI, per query class (C5): {ci_note}",
            f"- Break-even (prereg §4.4): "
            f"{'prefill_saved > dma_cost AND significant' if pa['verdict'] == 'GO' else 'prefill_saved <= dma_cost OR not significant'} "
            f"-> **{pa['verdict']}**",
        ]

    lines += [
        "",
        "## Methodological Notes",
        "",
        "- AB/BA arm order alternation per cycle (shared→isolated, isolated→shared, ...)",
        "- Both arms use PegaFlow with symmetric warmup count",
        "- Independent server lifecycle per arm (not shared across cycles)",
        "- Per-query-class paired reporting (Q0 vs Q0, Q1 vs Q1, Q2 vs Q2)",
        "- DMA cost per-request bound to corresponding hit (not shared mean)",
        "- See prior artifact (results/trace-audit/) for methodological asymmetry discovered",
    ]

    lines += [
        "",
        "## Per-Cycle TTFT (Lifecycle-Level Paired Delta)",
        "",
        "Each lifecycle = 1 independent paired observation (shared - isolated).",
        "Cluster bootstrap over n=3 lifecycles, per query class.",
        "",
        "| Cycle | Shared Mean | Isolated Mean | Paired Delta |",
        "|---|---|---|---|",
    ]
    per_cycle_deltas: dict[int, float] = {}
    for c in sorted(set(r["cycle"] for r in all_records)):
        sc = [r for r in shared if r["cycle"] == c]
        ic = [r for r in isolated if r["cycle"] == c]
        sm = sum(r["ttft_s"] for r in sc) / len(sc) if sc else 0
        im = sum(r["ttft_s"] for r in ic) / len(ic) if ic else 0
        delta = im - sm
        per_cycle_deltas[c] = delta
        gain = delta / im * 100 if im > 0 else 0
        lines.append(f"| {c} | {sm:.4f}s | {im:.4f}s | {delta:+.4f}s ({gain:+.1f}%) |")

    # Lifecycle-level cluster bootstrap for paired delta
    if len(per_cycle_deltas) >= 3:
        deltas = list(per_cycle_deltas.values())
        boot_deltas = []
        for _ in range(1000):
            sample = [random.choice(deltas) for __ in range(len(deltas))]
            boot_deltas.append(sum(sample) / len(sample))
        boot_deltas.sort()
        ci_low = boot_deltas[25]
        ci_high = boot_deltas[974]
        mean_delta = sum(deltas) / len(deltas)
        lines += [
            "",
            "## Lifecycle-Level Paired Delta (n=3 clusters)",
            f"- Mean paired delta: {mean_delta*1000:.0f}ms "
            f"(shared saves {(mean_delta/ttft_i['mean']*100):.1f}% vs isolated mean)",
            f"- 95% CI (cluster bootstrap): [{ci_low*1000:.0f}ms, {ci_high*1000:.0f}ms]",
            f"- Per-lifecycle deltas: "
            + ", ".join(f"C{c}:{d*1000:.0f}ms" for c, d in per_cycle_deltas.items()),
        ]

    # Per-query breakdown (Q0 = cold prefill, Q1/Q2 = prefix cache)
    for qidx in sorted(set(r["query_idx"] for r in all_records)):
        lines += [
            "",
            f"## Per-Query TTFT: Q{qidx}",
            "",
            "| Phase | N | Median | Mean | IQR | 95% CI |",
            "|---|---|---|---|---|---|",
        ]
        for phase_name in ["shared", "isolated"]:
            subset = [r for r in all_records
                      if r.get("phase") == phase_name
                      and r.get("query_idx") == qidx
                      and r.get("ok")]
            if not subset:
                # Empty subset (e.g. an arm aborted by fail-close) must not
                # crash before the validity manifest is written.
                lines.append(f"| {phase_name} | 0 | — | — | — | — |")
                continue
            stats = compute_stats(subset, "ttft_s")
            lines.append(
                f"| {phase_name} | {stats['n']} | "
                f"**{stats['median']:.4f}s** | {stats['mean']:.4f}s | "
                f"{stats['iqr']:.4f}s | "
                f"[{stats['ci_95_low']:.4f}, {stats['ci_95_high']:.4f}] |"
            )

    lines += [
        "",
        "## Negative Examples (Preserved)",
        "",
        "### Burst Concurrent (PCIe DMA Contention)",
        "",
        f"- Shared avg TTFT: {negative_examples['burst_concurrent_8inst']['shared_avg_ttft_s']}s",
        f"- Isolated avg TTFT: {negative_examples['burst_concurrent_8inst']['isolated_avg_ttft_s']}s",
        f"- Result: {negative_examples['burst_concurrent_8inst']['shared_vs_isolated']}",
        f"- Root cause: {negative_examples['burst_concurrent_8inst']['root_cause']}",
        f"- Verdict: {negative_examples['burst_concurrent_8inst']['verdict']}",
        "",
        "### MLA+TP8 (Prefill Too Cheap)",
        "",
        f"- Shared avg TTFT: {negative_examples['mla_tp8_deepseek_v2_lite']['shared_avg_ttft_s']}s",
        f"- Isolated avg TTFT: {negative_examples['mla_tp8_deepseek_v2_lite']['isolated_avg_ttft_s']}s",
        f"- Result: {negative_examples['mla_tp8_deepseek_v2_lite']['shared_vs_isolated']}",
        f"- Root cause: {negative_examples['mla_tp8_deepseek_v2_lite']['root_cause']}",
        f"- Verdict: {negative_examples['mla_tp8_deepseek_v2_lite']['verdict']}",
        "",
    ]

    # Producer (warmup seed) stats — default so the manifest renders even
    # when fail-close aborted before any producer record was produced.
    p_ttft = {"n": 0, "median": 0, "mean": 0}
    if producers:
        p_ttft = compute_stats(producers, "ttft_s")
        lines += [
            "",
            "## Producer (Warmup Seed) Records",
            f"- Count: {p_ttft['n']}",
            f"- Median TTFT: {p_ttft['median']:.4f}s",
            f"- Mean TTFT: {p_ttft['mean']:.4f}s",
            "- These records are excluded from consumer paired-delta analysis.",
            "- They seed the shared cache but are not themselves cross-instance consumers.",
        ]

    # Validity manifest (coverage + conservation gates — P2-3/P2-6)
    base_ok = (
        ttft_s["n"] >= 12 and ttft_i["n"] >= 12
        and len(all_records) > 0
    )
    evidence_ok = (
        merge_result.get("conservation_ok", False)
        and merge_result.get("coverage_pct", 0) >= 100.0
        and merge_result.get("unmatched", 0) == 0
        and not drift_violations
    )
    validity_ok = base_ok and evidence_ok
    if merge_result.get("coverage_pct", 0) < 100.0:
        lines.append(
            f"- Coverage gate FAILED: {merge_result['coverage_pct']:.1f}% "
            f"(requires 100%)")
    lines += [
        "",
        "## Evidence Violations (Fail-Close)",
    ]
    all_violations = merge_result.get("violations", []) + list(drift_violations)
    if not all_violations:
        lines.append("- None — all connector/prefetch/DMA events unique and conserved.")
    else:
        for v in all_violations:
            lines.append(f"- [INVALID] {v}")
    lines += [
        "",
        "## Validity Manifest",
        f"- Run ID: {_RUN_ID}",
        f"- Total records: {len(all_records)}",
        f"- Consumer shared records: {ttft_s['n']}",
        f"- Consumer isolated records: {ttft_i['n']}",
        f"- Producer records: {p_ttft['n']}",
        f"- INVALID records: {sum(1 for r in all_records if not r.get('ok'))}",
        f"- Audit-invalid records (evidence): "
        f"{sum(1 for r in all_records if r.get('_audit_invalid'))}",
        f"- Conservation: "
        f"{'OK' if merge_result.get('conservation_ok') else 'BROKEN'} "
        f"(connector dup={merge_result.get('connector_duplicates', 0)}, "
        f"orphans={merge_result.get('connector_orphans', 0)}/"
        f"{merge_result.get('prefetch_orphans', 0)}/"
        f"{merge_result.get('dma_orphans', 0)}, "
        f"leftover DMA={merge_result.get('dma_leftover', 0)}, "
        f"fallback-only DMA={merge_result.get('dma_fallback_only', 0)})",
        f"- Validity gate: {'PASS' if validity_ok else 'FAIL'}",
        f"- Audit verdict: {'VALID' if validity_ok else 'INVALID'}",
    ]

    lines += [
        "",
        "## Artifacts",
        f"- Raw records: `{out_dir}/trace_audit.json`",
        f"- Per-arm server logs: `{LOG_DIR}/arm_*/server.log`",
        f"- vLLM logs: `{LOG_DIR}/vllm_*.log`",
        f"- Environment snapshot: `trace_audit.json` → `_env` key",
    ]

    summary_path = out_dir / "trace_summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"\nSummary: {summary_path}")
    return validity_ok


def fail_close(reasons: list[str]) -> None:
    """Print every gate failure as [INVALID] and exit non-zero (fail-close).

    P2-6: a run with broken evidence is never released as PASS — summary
    must already be written so the artifact documents the INVALID verdict.
    """
    for reason in reasons:
        print(f"  [INVALID] {reason}")
    sys.exit(1)


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Trace Audit: PegaFlow Ascend KV Transfer Benchmark"
    )
    parser.add_argument("--cycles", type=int, default=3,
                        help="Independent lifecycles (default: 3)")
    parser.add_argument("--requests-per-phase", type=int, default=3,
                        help="Requests per phase (default: 3)")
    parser.add_argument("--pool-size", type=str, default="4096mb")
    parser.add_argument("--min-free-gb", type=int, default=28)
    args = parser.parse_args()

    min_free_mb = args.min_free_gb * 1024
    model_path = MODEL_PATH
    queries = USER_QUERIES[:args.requests_per_phase]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Artifact binding
    # ------------------------------------------------------------------
    env_info = capture_environment()

    print("=" * 70)
    print("  Trace Audit: PegaFlow Ascend KV Transfer")
    print(f"  Model:    {env_info.get('model', '?')}")
    print(f"  Commit:   {env_info.get('git_commit','?')[:12]}")
    print(f"  Cycles:   {args.cycles}")
    print(f"  Requests: {args.requests_per_phase}/phase")
    print(f"  Pool:     {args.pool_size}")
    print("=" * 70)

    kill_tracked()
    time.sleep(2)

    # ------------------------------------------------------------------
    # Matched-lifecycle runner: each arm gets independent server lifecycle,
    # AB/reverse-AB alternation across cycles, symmetric warmup, both arms use
    # PegaFlow with symmetric namespace configuration.
    # ------------------------------------------------------------------
    all_records: list[dict] = []
    drift_violations: list[str] = []  # P2-6: mid-arm admission drift
    ARM_ORDER = [
        # cycle 1: AB, cycle 2: reverse-AB, cycle 3: AB, ...
        [("shared", True), ("isolated", False)],
        [("isolated", False), ("shared", True)],
        [("shared", True), ("isolated", False)],
    ]

    try:
        for cycle in range(1, args.cycles + 1):
            print(f"\n{'#'*70}")
            print(f"# CYCLE {cycle}/{args.cycles}")
            arms = ARM_ORDER[(cycle - 1) % len(ARM_ORDER)]
            print(f"# Arm order: {arms[0][0]} → {arms[1][0]}")
            print(f"{'#'*70}")

            for arm_name, is_shared in arms:
                # Both arms get fresh per-cycle namespace strings
                arm_label = f"C{cycle}_{arm_name}"
                if is_shared:
                    ns_base = f"{SHARED_NS}-c{cycle}"
                else:
                    ns_base = f"{ISOLATED_NS_PREFIX}-c{cycle}"

                # P1-1: Admitted-device check BEFORE starting server.
                # Admission gate — only admit NPUs with enough free HBM.
                # If < NUM_INSTANCES available, mark arm INVALID and skip.
                free_mem = get_npu_free_memory()
                admitted = [i for i in range(NUM_INSTANCES)
                            if free_mem.get(i, -1) >= min_free_mb]
                if len(admitted) < NUM_INSTANCES:
                    print(f"  [INVALID] Only {len(admitted)}/{NUM_INSTANCES} "
                          f"NPUs have >= {min_free_mb}MB free: {admitted}. "
                          f"Arm {arm_label} ABORTED.")
                    all_records.append({
                        "cycle": cycle, "phase": arm_name, "req_idx": -1,
                        "query_idx": -1, "instance": "INVALID", "npu": -1,
                        "port": -1, "query": "", "ttft_s": -1, "total_s": -1,
                        "ok": False, "text": "",
                        "error": f"insufficient NPU free: {len(admitted)}/{NUM_INSTANCES}",
                        "producer": False,
                    })
                    continue

                # Independent server lifecycle per arm
                print(f"\n--- {arm_label} (namespace_base={ns_base}) ---")
                print(f"  Admitting NPUs: {admitted}")
                print(f"  Starting independent server lifecycle...")
                arm_log_dir = LOG_DIR / f"arm_{arm_label}"
                arm_log_dir.mkdir(parents=True, exist_ok=True)
                server = start_server(args.pool_size, log_dir=arm_log_dir,
                                      devices=admitted)
                print(f"  Server ready on :{SERVER_PORT}")
                time.sleep(2)

                specs = [
                    {"label": f"{arm_label}_{i}", "port": VLLM_BASE_PORT + i,
                     "mode": "read_write",
                     "namespace": ns_base if is_shared else f"{ns_base}-{i}",
                     "physical_npu": i, "use_pegaflow": True}
                    for i in admitted
                ]
                running = launch_all_instances(specs, model_path)
                if len(running) < len(specs):
                    print(f"  [INVALID] Only {len(running)}/{len(specs)} "
                          f"instances started. Arm {arm_label} ABORTED.")
                    for _, proc in running:
                        stop_proc(proc)
                    records = [{
                        "cycle": cycle, "phase": arm_name, "req_idx": -1,
                        "query_idx": -1, "instance": "INVALID",
                        "npu": -1, "port": -1, "query": "",
                        "ttft_s": -1, "total_s": -1, "ok": False, "text": "",
                        "error": f"instance launch failed: {len(running)}/{len(specs)}",
                    }]
                    all_records.extend(records)
                    stop_proc(server)
                    kill_tracked()
                    time.sleep(5)
                    continue

                # P2-6: re-verify admission right after launch — owner PID
                # and HBM must still hold for every admitted device.
                drift = check_admission_drift(
                    admitted, free_mem, get_npu_free_memory(),
                    {i: expected_used_mb(i, free_mem) for i in admitted},
                    get_npu_processes(), tracked_pgids())
                if drift:
                    print(f"  [INVALID] Admission drift after launch — "
                          f"arm {arm_label} ABORTED.")
                    for v in drift:
                        print(f"  [INVALID] {v}")
                    drift_violations.extend(drift)
                    for _, proc in running:
                        stop_proc(proc)
                    all_records.append({
                        "cycle": cycle, "phase": arm_name, "req_idx": -1,
                        "query_idx": -1, "instance": "INVALID", "npu": -1,
                        "port": -1, "query": "", "ttft_s": -1, "total_s": -1,
                        "ok": False, "text": "",
                        "error": "admission drift: " + "; ".join(drift)[:200],
                        "producer": False,
                    })
                    stop_proc(server)
                    kill_tracked()
                    time.sleep(5)
                    continue
                # R7: periodic drift polling DURING the phase — boundary
                # samples miss transient mid-phase drift. The monitor thread
                # polls check_admission_drift until the phase ends.
                stop_event = threading.Event()
                monitor_out: list[str] = []
                monitor = threading.Thread(
                    target=monitor_admission_drift,
                    args=(admitted, free_mem,
                          {i: expected_used_mb(i, free_mem) for i in admitted},
                          tracked_pgids(), stop_event, monitor_out),
                    daemon=True,
                )
                monitor.start()
                try:
                    records = run_phase_sequential(
                        arm_name, running, queries, model_path,
                        warmup_first=True, cycle=cycle,  # symmetric warmup
                    )
                finally:
                    stop_event.set()
                    monitor.join(timeout=5)
                all_records.extend(records)
                drift_violations.extend(monitor_out)

                # P2-6: post-phase admission re-check (owner PID + HBM).
                # Drift here invalidates the arm's formal evidence.
                drift = check_admission_drift(
                    admitted, free_mem, get_npu_free_memory(),
                    {i: expected_used_mb(i, free_mem) for i in admitted},
                    get_npu_processes(), tracked_pgids())
                drift_violations.extend(drift)
                for v in drift:
                    print(f"  [INVALID] {v}")
                for _, proc in running:
                    stop_proc(proc)
                print(f"  {arm_label}: {len(records)} records")

                # Kill server to ensure independent lifecycle
                stop_proc(server)
                kill_tracked()
                time.sleep(5)

    finally:
        print("\nShutting down...")
        kill_tracked()

    # ------------------------------------------------------------------
    # Merge per-request hit/miss/DMA from server + vLLM logs
    # ------------------------------------------------------------------
    print("\nMerging per-request cache hit + DMA data from logs...")

    # Step 1: Extract connector cache_lookup from each vLLM log
    # Format: [PegaKVConnector] req=<req_id> cache_lookup: hit_blocks=N ...
    connector_by_req: dict[str, dict] = {}  # req_id -> {label, hit_blocks, hit_tokens, num_tokens}
    timing_by_req: dict[str, dict] = {}     # req_id -> {prefill_time_ms, queue_time_ms} (A2)
    for vllm_log in sorted((LOG_DIR).glob("vllm_*.log")):
        label = vllm_log.name.replace("vllm_", "").replace(".log", "")
        text = vllm_log.read_text()
        timing_by_req.update(extract_vllm_timing(text))
        for m in re.finditer(
            r"\[PegaKVConnector\] req=(?P<req_id>\S+)\s+"
            r"cache_lookup: hit_blocks=(?P<hit>\d+) "
            r"computed_blocks=(?P<computed>\d+) "
            r"hit_tokens=(?P<hit_tokens>\d+) num_tokens=(?P<num_tokens>\d+)",
            text,
        ):
            req_id = m.group("req_id")
            # P2-6: count occurrences instead of silently overwriting —
            # duplicate connector events must be reported, not last-wins.
            entry = connector_by_req.get(req_id)
            if entry is not None:
                entry["occurrences"] = entry.get("occurrences", 1) + 1
                continue
            connector_by_req[req_id] = {
                "req_id": req_id,
                "label": label,
                "hit_blocks": int(m.group("hit")),
                "computed_blocks": int(m.group("computed")),
                "hit_tokens": int(m.group("hit_tokens")),
                "num_tokens": int(m.group("num_tokens")),
                "occurrences": 1,
            }

    # Step 2: Extract prefetch + DMA from per-arm server logs, keeping
    # arm_label for scoped matching (P1-4). Each arm's prefetches only
    # match DMA within the same arm and within a time window.
    prefetch_by_req: dict[str, dict] = {}
    ts_prefetches: list[dict] = []   # {ts, req_id, arm_label, hit, missing}
    ts_dmas: list[dict] = []         # {ts, device_id, arm_label, dma_ms, ...}

    # Read from all per-arm log directories
    for arm_log_dir in sorted(LOG_DIR.glob("arm_*")):
        arm_label = arm_log_dir.name.replace("arm_", "")  # e.g. "C1_shared"
        server_log = arm_log_dir / "server.log"
        if not server_log.exists():
            continue
        text = server_log.read_text()

        # Extract prefetch entries
        for m in re.finditer(
            r"Prefetch local-hit timing: "
            r"req_id=(?P<req_id>\S+)\s+"
            r"total_keys=(?P<total>\d+)\s+"
            r"hit=(?P<hit>\d+)\s+"
            r"missing=(?P<missing>\d+)",
            text,
        ):
            req_id = m.group("req_id")
            # P2-6: count occurrences — duplicate prefetch events must be
            # reported, not silently overwritten.
            pf_entry = prefetch_by_req.get(req_id)
            if pf_entry is not None:
                pf_entry["occurrences"] = pf_entry.get("occurrences", 1) + 1
                continue
            prefetch_by_req[req_id] = {
                "total_keys": int(m.group("total")),
                "hit_blocks": int(m.group("hit")),
                "missing_blocks": int(m.group("missing")),
                "occurrences": 1,
            }

        # Parse timestamped events for DMA binding (per-arm scope)
        for line in text.split("\n"):
            ts_match = re.match(
                r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)", line,
            )
            if not ts_match:
                continue
            ts = ts_match.group(1)

            # Prefetch hit
            m = re.search(
                r"Prefetch local-hit timing: "
                r"req_id=(?P<req_id>\S+)\s+.*?"
                r"hit=(?P<hit>\d+)\s+missing=(?P<missing>\d+)",
                line,
            )
            if m:
                ts_prefetches.append({
                    "ts": ts, "req_id": m.group("req_id"),
                    "arm_label": arm_label,
                    "hit": int(m.group("hit")), "missing": int(m.group("missing")),
                })

            # DMA completion (batch path)
            m = re.search(
                r"Load task completed:.*?"
                r"bytes=(?P<bytes>\d+)\s+"
                r"elapsed_ms=(?P<ms>[\d.]+)\s+"
                r"bandwidth_gbps=(?P<gbps>[\d.]+)\s+"
                r".*?device_id=(?P<dev>\d+)",
                line,
            )
            if m:
                ts_dmas.append({
                    "ts": ts,
                    "arm_label": arm_label,
                    "device_id": int(m.group("dev")),
                    "dma_bytes": int(m.group("bytes")),
                    "dma_ms": float(m.group("ms")),
                    "dma_gbps": float(m.group("gbps")),
                })

            # Per-copy H2D fallback
            if "falling back to per-copy aclrtMemcpyAsync" in line:
                m_dev = re.search(r"device_id=(\d+)", line)
                if m_dev:
                    ts_dmas.append({
                        "ts": ts,
                        "arm_label": arm_label,
                        "device_id": int(m_dev.group(1)),
                        "dma_bytes": 0,
                        "dma_ms": 0.0,
                        "dma_gbps": 0.0,
                        "fallback": True,
                    })

    # Step 4: Bind DMA to prefetch (pure function — R9). connector_by_req
    # maps req_id → {label: "C1_shared_3", ...}; DMA has device_id (NPU ID),
    # mapped via label→npu built from instance labels.
    label_to_npu: dict[str, int] = {}
    for cinfo in connector_by_req.values():
        label = cinfo.get("label", "")
        # label format: C1_shared_3 → NPU 3
        parts = label.rsplit("_", 1)
        if len(parts) == 2:
            try:
                label_to_npu[label] = int(parts[1])
            except ValueError:
                pass

    prefetch_dma_map, fallback_only_count, dma_leftover, bind_violations = \
        bind_dma_to_prefetch(ts_prefetches, ts_dmas, connector_by_req,
                             label_to_npu)
    for v in bind_violations:
        print(f"  [INVALID] {v}")

    # Step 5: Merge per-request by client-generated request_id (pure function).
    # Fail-close (P2-6): any missing/duplicate/orphan formal event or
    # coverage < 100% invalidates the run — reported here, and the process
    # exits non-zero after the summary is written.
    merge_result = merge_by_req_id(all_records, connector_by_req,
                                   prefetch_by_req, prefetch_dma_map,
                                   dma_leftover_count=dma_leftover,
                                   dma_fallback_only_count=fallback_only_count,
                                   timing_by_req=timing_by_req)
    print(f"  Merged hit/DMA: {merge_result['matched']} records via req_id lookup "
          f"(unmatched={merge_result['unmatched']}, "
          f"producer_skipped={merge_result['producer_skipped']}, "
          f"coverage={merge_result['coverage_pct']:.1f}%, "
          f"conservation={'OK' if merge_result['conservation_ok'] else 'BROKEN'}, "
          f"conn_dup={merge_result['connector_duplicates']}, "
          f"orphans={merge_result['connector_orphans']}/"
          f"{merge_result['prefetch_orphans']}/"
          f"{merge_result['dma_orphans']}, "
          f"leftover_dma={merge_result['dma_leftover']}, "
          f"fallback_only={merge_result['dma_fallback_only']})")

    # Fail-close: every evidence violation is printed as [INVALID] now.
    for v in merge_result["violations"]:
        print(f"  [INVALID] {v}")

    # ------------------------------------------------------------------
    # Negative examples: burst + MLA from previous benchmarks
    # ------------------------------------------------------------------
    negative_examples = {
        "burst_concurrent_8inst": {
            "description": "Burst 8-instance concurrent — PCIe DMA contention destroys PegaFlow benefit",
            "source": "run_bench_8inst_concurrent.py (old version, semaphore=unlimited)",
            "shared_avg_ttft_s": 2.70,
            "isolated_avg_ttft_s": 1.73,
            "shared_vs_isolated": "+56% (shared WORSE)",
            "root_cause": "8 concurrent DMA streams saturate PCIe 4.0 uplink: 15 GB/s / 8 = 1.9 GB/s per stream, single DMA inflates from 85ms to ~750ms",
            "verdict": "Burst is unrealistic workload; staggered/normal serving load unaffected"
        },
        "mla_tp8_deepseek_v2_lite": {
            "description": "DeepSeek-V2-Lite MLA+TP8 — prefill too cheap for PegaFlow to matter",
            "source": "run_bench_mla_tp8_concurrent.py",
            "shared_avg_ttft_s": 0.184,
            "isolated_avg_ttft_s": 0.187,
            "shared_vs_isolated": "+1.6% (no meaningful gain)",
            "root_cause": "MLA kv_lora_rank=512 compresses KV compute to ~100ms; DMA of compressed KV (~40 MB) takes ~3ms; prefill cost too small to save",
            "verdict": "PegaFlow requires large enough prefill gap to overcome DMA cost. 16B MLA model does not meet threshold; 236B+ may."
        }
    }

    # ------------------------------------------------------------------
    # Save and summarize
    # ------------------------------------------------------------------
    output = {
        "_env": env_info,
        "_negative_examples": negative_examples,
        "records": all_records,
    }
    out_json = OUT_DIR / "trace_audit.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Raw data: {out_json} ({len(all_records)} records)")

    summary_ok = write_summary(env_info, all_records, negative_examples,
                               merge_result, OUT_DIR, drift_violations)

    if not summary_ok:
        fail_close([f"validity gate FAILED — see {OUT_DIR / 'trace_summary.md'}"])

    print("\n" + "=" * 70)
    print("  Trace audit complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
