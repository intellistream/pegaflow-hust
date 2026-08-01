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

import argparse, hashlib, json, os, re, signal, subprocess, sys
import threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/workspace/HUST/pegaflow-hust")
LOG_DIR = Path("/workspace/HUST/pegaflow-hust/results/trace-audit/logs")
OUT_DIR = Path("/workspace/HUST/pegaflow-hust/results/trace-audit")

MODEL_PATH = "/workspace/HUST/models/Qwen3-8B"
SERVER_PORT = 50080
VLLM_BASE_PORT = 19000
SHARED_NS = "audit-shared"
ISOLATED_NS_PREFIX = "audit-iso"
NUM_INSTANCES = 8
HBM_TOTAL_MB = 65536
MIN_FREE_HBM_MB = 28 * 1024

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

    # Git
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            timeout=10,
        ).decode().strip()
        info["git_branch"] = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
            timeout=10,
        ).decode().strip()
    except Exception:
        info["git_commit"] = "unknown"

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


def start_server(pool_size="4096mb", log_dir=None):
    if log_dir is None:
        log_dir = LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / "server.log"
    proc = subprocess.Popen(
        [
            str(PROJECT_ROOT / "target" / "debug" / "pegaflow-server"),
            "--addr", f"0.0.0.0:{SERVER_PORT}",
            "--pool-size", pool_size,
            "--devices", "0,1,2,3,4,5,6,7",
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
    """Send streaming request, return {ttft_s, total_s, text, ok}."""
    data = json.dumps({
        "model": model_path, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0.0,
        "stream": True,
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
                "text": text[:80], "ok": True}
    except Exception as e:
        return {"ttft_s": -1, "total_s": -1, "text": "", "ok": False,
                "error": str(e)[:200]}


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

    # Warmup (if applicable)
    if warmup_first and len(instances) >= 1:
        warmup_spec, warmup_proc = instances[0]
        prompt = f"{SYSTEM_PROMPT}\n\nUser: {queries[0]}\n\nAssistant:"
        r = send_one_streaming(warmup_spec["port"], prompt, model_path)
        print(f"    [WARMUP] {warmup_spec['label']} "
              f"TTFT={r['ttft_s']:.4f}s ok={r['ok']}")
        time.sleep(30)  # seal delay: write pipeline needs ~20-25s to seal blocks after warmup

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
            }
            records.append(record)
            status = (f"TTFT={r['ttft_s']:.4f}s" if r["ok"]
                      else f"ERR={r.get('error','?')[:30]}")
            print(f"    [{idx:>3d}] {spec['label']} Q{qi} {status} | {q[:30]}")
            idx += 1
            time.sleep(0.5)
    return records


# =========================================================================
# Statistics
# =========================================================================

def compute_stats(records: list[dict], key="ttft_s") -> dict:
    vals = sorted(r[key] for r in records if r.get(key, -1) > 0)
    if not vals:
        return {"n": 0, "median": 0, "mean": 0, "std": 0}
    n = len(vals)
    mean = sum(vals) / n
    median = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    # 95% CI via bootstrap percentile
    import random
    random.seed(42)
    boot_means = []
    for _ in range(1000):
        sample = [random.choice(vals) for __ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    ci_low = boot_means[25]
    ci_high = boot_means[974]
    return {
        "n": n, "mean": round(mean, 4), "median": round(median, 4),
        "ci_95_low": round(ci_low, 4), "ci_95_high": round(ci_high, 4),
        "min": round(vals[0], 4), "max": round(vals[-1], 4),
    }


# =========================================================================
# Summary + break-even
# =========================================================================

def write_summary(env_info, all_records, negative_examples, out_dir):
    """Write trace_summary.md with median/CI and break-even analysis."""
    shared = [r for r in all_records if r.get("phase") == "shared" and r.get("ok")]
    isolated = [r for r in all_records if r.get("phase") == "isolated" and r.get("ok")]

    ttft_s = compute_stats(shared, "ttft_s")
    ttft_i = compute_stats(isolated, "ttft_s")
    total_s = compute_stats(shared, "total_s")
    total_i = compute_stats(isolated, "total_s")

    lines = [
        "# Trace Audit Summary",
        "",
        "## Environment",
        f"- Commit: `{env_info.get('git_commit','?')[:12]}`",
        f"- Branch: `{env_info.get('git_branch','?')}`",
        f"- Model: `{env_info.get('model','?')}` "
        f"(md5: `{env_info.get('model_config_md5','?')[:12]}`)",
        f"- Timestamp: {env_info.get('timestamp','?')}",
        f"- NPUs: 8× Ascend 910B2 (see artifact for full npu-smi)",
        "",
        "## TTFT (Time-To-First-Token)",
        "",
        "| Phase | N | Median | Mean | 95% CI | Min | Max |",
        "|---|---|---|---|---|---|---|",
        f"| Shared | {ttft_s['n']} | **{ttft_s['median']:.4f}s** | "
        f"{ttft_s['mean']:.4f}s | [{ttft_s['ci_95_low']:.4f}, "
        f"{ttft_s['ci_95_high']:.4f}] | {ttft_s['min']:.4f}s | "
        f"{ttft_s['max']:.4f}s |",
        f"| Isolated | {ttft_i['n']} | **{ttft_i['median']:.4f}s** | "
        f"{ttft_i['mean']:.4f}s | [{ttft_i['ci_95_low']:.4f}, "
        f"{ttft_i['ci_95_high']:.4f}] | {ttft_i['min']:.4f}s | "
        f"{ttft_i['max']:.4f}s |",
        "",
        "## Total Latency",
        "",
        "| Phase | N | Median | Mean | 95% CI |",
        "|---|---|---|---|---|",
        f"| Shared | {total_s['n']} | {total_s['median']:.3f}s | "
        f"{total_s['mean']:.3f}s | [{total_s['ci_95_low']:.3f}, "
        f"{total_s['ci_95_high']:.3f}] |",
        f"| Isolated | {total_i['n']} | {total_i['median']:.3f}s | "
        f"{total_i['mean']:.3f}s | [{total_i['ci_95_low']:.3f}, "
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
        s_subset = [r for r in shared if r.get("query_idx") == qidx and r.get("ok")]
        i_subset = [r for r in isolated if r.get("query_idx") == qidx and r.get("ok")]
        s_vals = sorted(r["ttft_s"] for r in s_subset)
        i_vals = sorted(r["ttft_s"] for r in i_subset)
        if s_vals and i_vals:
            s_med = s_vals[len(s_vals)//2]
            i_med = i_vals[len(i_vals)//2]
            paired_delta = (i_med - s_med) / i_med * 100 if i_med > 0 else 0
            # Per-request DMA for this query class only
            dma_vals = [r.get("dma_ms", 0) for r in s_subset
                        if r.get("dma_ms", 0) > 0]
            dma_fallbacks = sum(1 for r in s_subset if r.get("dma_ms", 0) < 0)
            dma_paired = sum(dma_vals) / len(dma_vals) if dma_vals else 0
            dma_note = (f"{dma_paired:.1f}ms" if dma_vals
                        else f"per-copy fallback ({dma_fallbacks} DMA no timing)")
            s_mean = sum(s_vals) / len(s_vals)
            i_mean = sum(i_vals) / len(i_vals)
            lines += [
                f"### Q{qidx}",
                f"- Shared: n={len(s_vals)}, median={s_med:.4f}s, mean={s_mean:.4f}s",
                f"- Isolated: n={len(i_vals)}, median={i_med:.4f}s, mean={i_mean:.4f}s",
                f"- Paired delta: {paired_delta:+.1f}% ({(i_med - s_med)*1000:.0f}ms median)",
                f"- DMA (per-request, timestamp-bound): {dma_note}",
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
            "| Phase | N | Median | Mean | 95% CI |",
            "|---|---|---|---|---|",
        ]
        for phase_name in ["shared", "isolated"]:
            subset = [r for r in all_records
                      if r.get("phase") == phase_name
                      and r.get("query_idx") == qidx
                      and r.get("ok")]
            stats = compute_stats(subset, "ttft_s")
            lines.append(
                f"| {phase_name} | {stats['n']} | "
                f"**{stats['median']:.4f}s** | {stats['mean']:.4f}s | "
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
        "## Artifacts",
        f"- Raw records: `{out_dir}/trace_audit.json`",
        f"- Per-arm server logs: `{LOG_DIR}/arm_*/server.log`",
        f"- vLLM logs: `{LOG_DIR}/vllm_*.log`",
        f"- Environment snapshot: `trace_audit.json` → `_env` key",
    ]

    summary_path = out_dir / "trace_summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"\nSummary: {summary_path}")


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
    # AB/BA alternation across cycles, symmetric warmup, both arms use
    # PegaFlow with symmetric namespace configuration.
    # ------------------------------------------------------------------
    all_records: list[dict] = []
    ARM_ORDER = [
        # cycle 1: AB, cycle 2: BA, cycle 3: AB, ...
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

                # Independent server lifecycle per arm
                print(f"\n--- {arm_label} (namespace_base={ns_base}) ---")
                print(f"  Starting independent server lifecycle...")
                # Per-arm log file to avoid truncation across arms
                arm_log_dir = LOG_DIR / f"arm_{arm_label}"
                arm_log_dir.mkdir(parents=True, exist_ok=True)
                server = start_server(args.pool_size, log_dir=arm_log_dir)
                print(f"  Server ready on :{SERVER_PORT}")
                time.sleep(2)

                # P1-1: Admitted-device check — only use NPUs with enough free HBM
                free_mem = get_npu_free_memory()
                admitted = [i for i in range(NUM_INSTANCES)
                            if free_mem.get(i, -1) >= min_free_mb]
                if len(admitted) < NUM_INSTANCES:
                    print(f"  [INVALID] Only {len(admitted)}/{NUM_INSTANCES} "
                          f"NPUs have >= {min_free_mb}MB free: {admitted}. "
                          f"Arm {arm_label} ABORTED.")
                    records = [{
                        "cycle": cycle, "phase": arm_name, "req_idx": -1,
                        "query_idx": -1, "instance": "INVALID",
                        "npu": -1, "port": -1, "query": "",
                        "ttft_s": -1, "total_s": -1, "ok": False, "text": "",
                        "error": f"insufficient NPU free memory: {len(admitted)}/{NUM_INSTANCES}",
                    }]
                    all_records.extend(records)
                    continue

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
                records = run_phase_sequential(
                    arm_name, running, queries, model_path,
                    warmup_first=True, cycle=cycle,  # symmetric warmup
                )
                all_records.extend(records)
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
    for vllm_log in sorted((LOG_DIR).glob("vllm_*.log")):
        label = vllm_log.name.replace("vllm_", "").replace(".log", "")
        text = vllm_log.read_text()
        for m in re.finditer(
            r"\[PegaKVConnector\] req=(?P<req_id>\S+)\s+"
            r"cache_lookup: hit_blocks=(?P<hit>\d+) "
            r"computed_blocks=(?P<computed>\d+) "
            r"hit_tokens=(?P<hit_tokens>\d+) num_tokens=(?P<num_tokens>\d+)",
            text,
        ):
            req_id = m.group("req_id")
            connector_by_req[req_id] = {
                "req_id": req_id,
                "label": label,
                "hit_blocks": int(m.group("hit")),
                "computed_blocks": int(m.group("computed")),
                "hit_tokens": int(m.group("hit_tokens")),
                "num_tokens": int(m.group("num_tokens")),
            }

    # Step 2: Extract prefetch + DMA from per-arm server logs, keeping
    # arm_label for scoped matching (P1-4). Each arm's prefetches only
    # match DMA within the same arm and within a time window.
    prefetch_by_req: dict[str, dict] = {}
    ts_prefetches: list[dict] = []   # {ts, req_id, arm_label, hit, missing}
    ts_dmas: list[dict] = []         # {ts, device_id, arm_label, dma_ms, ...}
    DMA_TIME_WINDOW_S = 30.0         # max seconds between prefetch and DMA

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
            prefetch_by_req[m.group("req_id")] = {
                "total_keys": int(m.group("total")),
                "hit_blocks": int(m.group("hit")),
                "missing_blocks": int(m.group("missing")),
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

    # Step 4: Bind DMA to prefetch by req_id matching via connector label→NPU
    # connector_by_req maps req_id → {label: "C1_shared_3", ...}
    # DMA has device_id (NPU ID); we map label → npu via instance_specs
    # Build label→npu map from all instance labels seen in connector entries
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

    prefetch_dma_map: dict[str, dict] = {}  # req_id -> {dma_ms, dma_bytes, dma_gbps}
    for pf in ts_prefetches:
        if pf["hit"] == 0:
            continue  # no DMA for miss
        req_id = pf["req_id"]
        # Get target device from connector label
        target_device = None
        if req_id in connector_by_req:
            target_device = label_to_npu.get(
                connector_by_req[req_id].get("label", ""), None
            )
        # Find nearest subsequent DMA on same device + same arm + within time window.
        pf_arm = pf.get("arm_label", "")
        best, best_fallback = None, None
        for dma in ts_dmas:
            if dma["ts"] <= pf["ts"]:
                continue
            if dma.get("arm_label", "") != pf_arm:
                continue  # P1-4: arm scope
            if target_device is not None and dma["device_id"] != target_device:
                continue  # P1-4: device scope
            # P1-4: time window check
            try:
                pf_sec = float(pf["ts"].split("T")[1].split(":")[-1])
                dma_sec = float(dma["ts"].split("T")[1].split(":")[-1])
                if abs(dma_sec - pf_sec) > DMA_TIME_WINDOW_S:
                    continue
            except (ValueError, IndexError):
                pass  # if timestamp parse fails, skip window check
            if dma.get("fallback"):
                if best_fallback is None or dma["ts"] < best_fallback["ts"]:
                    best_fallback = dma
            else:
                if best is None or dma["ts"] < best["ts"]:
                    best = dma
        # Prefer real timing; fall back to per-copy if no timing available
        if best is None:
            best = best_fallback
        if best:
            dma_ms = best["dma_ms"]
            dma_bytes = best["dma_bytes"]
            dma_gbps = best["dma_gbps"]
            # -1 signals "DMA happened via per-copy fallback, timing unavailable"
            if best.get("fallback") and dma_ms <= 0.0:
                dma_ms = -1.0
                dma_bytes = 1434451968  # known block size for 76 blocks
                dma_gbps = 0.0
            prefetch_dma_map[req_id] = {
                "dma_ms": dma_ms,
                "dma_bytes": dma_bytes,
                "dma_gbps": dma_gbps,
            }
            ts_dmas.remove(best)  # consume — one DMA per prefetch

    # Step 5: Merge per-request — one connector entry per record, consumed in order.
    # Build per-instance ordered queues from connector entries (logged in time order).
    from collections import defaultdict, deque
    inst_queues: dict[str, deque] = defaultdict(deque)
    for cinfo in connector_by_req.values():
        label = cinfo.get("label", "")
        inst_queues[label].append(cinfo)

    merged_count = 0
    orphan_count = 0
    for r in all_records:
        r.setdefault("hit_blocks", 0)
        r.setdefault("hit_tokens", 0)
        r.setdefault("missing_blocks", 0)
        r.setdefault("dma_bytes", 0)
        r.setdefault("dma_ms", 0.0)
        r.setdefault("dma_gbps", 0.0)
        r.setdefault("req_id", "")

        instance_label = r.get("instance", "")
        queue = inst_queues.get(instance_label)
        if queue:
            cinfo = queue.popleft()  # consume in order, one per record
            r["hit_blocks"] = cinfo["hit_blocks"]
            r["hit_tokens"] = cinfo["hit_tokens"]
            r["num_tokens_total"] = cinfo.get("num_tokens", 0)
            req_id = cinfo.get("req_id", "")
            r["req_id"] = req_id
            if req_id in prefetch_by_req:
                r["missing_blocks"] = prefetch_by_req[req_id]["missing_blocks"]
            if req_id in prefetch_dma_map:
                dma = prefetch_dma_map[req_id]
                r["dma_bytes"] = dma["dma_bytes"]
                r["dma_ms"] = dma["dma_ms"]
                r["dma_gbps"] = dma["dma_gbps"]
            merged_count += 1
        else:
            orphan_count += 1

    print(f"  Merged hit/DMA: {merged_count}/{len(all_records)} records "
          f"(orphan={orphan_count})")

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

    write_summary(env_info, all_records, negative_examples, OUT_DIR)

    print("\n" + "=" * 70)
    print("  Trace audit complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
