#!/usr/bin/env python3
"""
PegaFlow Benchmark — 8-Instance Shared Cache vs Isolated Pools

Reproduces: 单节点 8 个 Qwen3-8B 实例, 相同 500 GiB Cache 预算

  | 设置       | Cache 布局          | 吞吐量    | 平均 TTFT | 请求命中率 |
  |------------|--------------------|-----------|-----------|-----------|
  | PegaFlow   | 500 GiB 共享池      | 11.97 r/s | 5.26 s    | 52.35%    |
  | 进程内     | 8 × 62.5 GiB 隔离池 | 7.68 r/s  | 8.22 s    | 11.77%    |

Startup strategy
---------------
  1. Start pegaflow-server
  2. Poll npu-smi every 10 s.  Whenever an NPU is free AND not yet
     assigned, start a vLLM instance on it.  Increment counter.
  3. Once all 8 instances are running → begin test.
  4. Shut down instances → next phase.

  This is "grab-as-soon-as-free" — no need for all 8 to be free
  simultaneously.  As soon as one NPU frees up, we claim it and wait
  for the rest.

Usage
-----
  python run_bench_8inst.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/workspace/HUST/pegaflow-hust")
LOG_DIR = Path("/tmp/pegaflow-bench-8inst")

MODEL_PATH = "/workspace/HUST/models/Qwen3-8B"
MODEL_FALLBACK = "/workspace/HUST/models/Qwen2.5-7B-Instruct"

SERVER_PORT = 50070
VLLM_BASE_PORT = 18700
SHARED_NS = "bench-8inst-shared"
ISOLATED_NS_PREFIX = "bench-8inst-iso"
NUM_INSTANCES = 8

_SYS_BLOCK = (
    "You are an expert AI assistant with deep knowledge across many domains including "
    "computer science, mathematics, physics, biology, history, philosophy, literature, "
    "economics, law, medicine, engineering, and the arts. You provide accurate, detailed, "
    "and well-structured responses. Always follow instructions precisely and think step "
    "by step before answering. Your responses should be helpful, harmless, and honest. "
    "When asked factual questions, provide evidence-based answers with citations where "
    "possible. When asked for opinions, provide balanced perspectives that acknowledge "
    "multiple viewpoints. When the user asks you to perform a task, break it down into "
    "clear, actionable steps and explain your reasoning at each stage. When you encounter "
    "ambiguity, ask clarifying questions rather than making assumptions. Be mindful of "
    "the user's time and keep responses focused and relevant. If you are unsure about "
    "something, acknowledge it honestly rather than speculating. Your goal is to be as "
    "helpful as possible while maintaining high standards of accuracy and clarity. "
    "Remember to adapt your communication style to the user's level of expertise — "
    "use technical language when appropriate but be ready to explain concepts in "
    "simpler terms when needed. Always prioritize the user's safety and well-being, "
    "and avoid generating harmful, unethical, or dangerous content under any "
    "circumstances. You should respect user privacy and not ask for or store personal "
    "information unnecessarily."
)
SYSTEM_PROMPT = (_SYS_BLOCK + " ") * 38  # ~10000 tokens (Qwen3 tokenizer: ~260 tok/block × 38)

USER_QUERIES = [
    "What is the capital of France?",
    "Explain how photosynthesis works in plants.",
    "Who wrote the play Hamlet and when?",
    "What is Einstein's theory of relativity?",
    "Describe the water cycle in detail.",
    "How does an electric motor work?",
    "Name the planets in our solar system in order.",
    "What is machine learning and how does it differ from traditional programming?",
    "Explain the concept of quantum entanglement in simple terms.",
    "What are the three laws of thermodynamics?",
    "How does a nuclear reactor generate electricity?",
    "What is CRISPR and how is it used in genetic engineering?",
    "Describe the Big Bang theory of the universe.",
    "How does blockchain technology work?",
    "What causes earthquakes and how are they measured?",
    "Explain the difference between TCP and UDP protocols.",
    "What is dark matter and why is it important in cosmology?",
    "How do vaccines work in the human immune system?",
    "What is the difference between mitosis and meiosis?",
    "Explain the concept of supply and demand in economics.",
    "How does a jet engine work?",
    "What is the greenhouse effect and how does it impact climate?",
    "Describe how the human brain processes visual information.",
    "What is quantum computing and how does it differ from classical computing?",
    "How does the international space station maintain orbit?",
    "What is DNA replication and why is it important?",
    "Explain the difference between renewable and non-renewable energy.",
    "How do antibiotics work against bacterial infections?",
    "What is the placebo effect in medical research?",
    "Describe the process of plate tectonics.",
    "How does public-key cryptography work?",
    "What is machine translation and how has it evolved?",
]


# =========================================================================
# NPU polling — memory-based
# =========================================================================

# Qwen3-8B in FP16: ~16 GB weights + KV cache + runtime overhead
# vLLM with --gpu-memory-utilization 0.85 needs ~56 GB max, but for a
# single 8B model we set a safe floor.  If vLLM OOMs on startup, the
# launcher catches the error and retries.
MIN_FREE_HBM_MB = 28 * 1024  # 28 GB — Qwen3-8B with 10k-token prompt
HBM_TOTAL_MB = 64 * 1024     # Ascend 910B2 has 64 GB HBM


def get_npu_free_memory() -> dict[int, int]:
    """Return {npu_id: free_hbm_mb} for NPUs 0-7 by parsing npu-smi info.

    Parses the HBM-Usage column (used/total in MB) from the body section
    (before the process table).  Returns -1 for unparsable NPUs.
    """
    free: dict[int, int] = {i: -1 for i in range(8)}
    try:
        out = subprocess.check_output(
            ["npu-smi", "info"], stderr=subprocess.STDOUT, timeout=30,
        ).decode()
    except Exception as e:
        print(f"  [WARN] npu-smi failed: {e}")
        return free

    # Parse body section (before "Process id" table).
    # Each NPU = 2 lines:
    #   | NPU_ID  Name  | Health | Power | Temp | Hugepages |
    #   | Chip    | Bus-Id   | AICore | Mem-Usage | HBM-Used/Total |
    lines = out.split("\n")
    current_npu: int | None = None
    for line in lines:
        if "Process id" in line and "Process name" in line:
            break  # stop at process table
        # Line 1: detect NPU ID (first number after '|')
        m1 = re.match(r"\|\s*(\d+)\s+\d+\w+\s+\|", line)
        if m1:
            current_npu = int(m1.group(1))
            continue
        # Line 2: extract HBM used/total from last column
        if current_npu is not None and current_npu < 8:
            m2 = re.search(r"(\d+)\s*/\s*(\d+)\s*\|?\s*$", line)
            if m2:
                used = int(m2.group(1))
                total = int(m2.group(2))
                free[current_npu] = total - used
                current_npu = None  # consumed this NPU pair
    return free


# =========================================================================
# Process helpers
# =========================================================================

def kill_all() -> None:
    for p in ["pegaflow-server", "vllm serve"]:
        os.system(f"pkill -f '{p}' 2>/dev/null || true")


def stop_proc(proc: subprocess.Popen | None) -> None:
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


def start_server(free_npus: list[int], pool_size: str) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / "server.log"
    devices = ",".join(str(i) for i in free_npus)
    proc = subprocess.Popen(
        [
            str(PROJECT_ROOT / "target" / "debug" / "pegaflow-server"),
            "--addr", f"0.0.0.0:{SERVER_PORT}",
            "--pool-size", pool_size,
            "--devices", devices,
        ],
        stdout=open(log, "w"), stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        time.sleep(1)
        try:
            if log.exists() and "listening" in log.read_text():
                return proc
        except Exception:
            pass
    raise RuntimeError(f"Server failed. Log: {log.read_text()[-300:] if log.exists() else 'N/A'}")


def start_vllm(
    port: int, mode: str, namespace: str | None,
    physical_npu: int, label: str, *, model_path: str,
    gpu_memory_utilization: float = 0.85,
    use_pegaflow: bool = True,
) -> subprocess.Popen:
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
        "source /root/miniconda3/etc/profile.d/conda.sh",
        "conda activate vllm-hust-dev",
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
    cmd = " && ".join([cmd_parts[0] + " && " + cmd_parts[1], " ".join(cmd_parts[2:])])
    proc = subprocess.Popen(
        ["bash", "-c", cmd], env=env,
        stdout=open(log, "w"), stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            return proc
        except Exception:
            time.sleep(3)
    raise RuntimeError(
        f"vLLM {label} not healthy after 180s. "
        f"Log: {log.read_text()[-400:] if log.exists() else 'N/A'}"
    )


# =========================================================================
# Opportunistic instance launcher
# =========================================================================

def launch_instances_opportunistically(
    instance_specs: list[dict],
    already_assigned: set[int],
    model_path: str,
    min_free_mb: int = MIN_FREE_HBM_MB,
    poll_interval: int = 10,
    startup_max_wait: int = 3600,
    use_pegaflow: bool = True,
) -> list[tuple[dict, subprocess.Popen]]:
    """Start vLLM instances one-by-one as NPUs have enough free HBM.

    instance_specs: list of {port, mode, namespace, physical_npu, label}
    If use_pegaflow is True, checks that pegaflow-server is reachable
    before launching.
    already_assigned: set of NPU IDs already claimed by us

    Returns list of (spec, proc) for successfully started instances.
    """
    target = len(instance_specs)
    running: list[tuple[dict, subprocess.Popen]] = []
    assigned_npus: set[int] = set()
    failed_npus: dict[int, float] = {}  # npu_id → cooldown_until timestamp
    needed_npus: set[int] = {s["physical_npu"] for s in instance_specs}

    print(f"  Need {target} instances on NPUs {sorted(needed_npus)}")
    print(f"  Min free HBM: {min_free_mb} MB (~{min_free_mb//1024} GB)")
    print(f"  Polling every {poll_interval}s — grabbing NPUs with enough space...")

    start_time = time.monotonic()
    last_status = ""

    while len(running) < target:
        elapsed = time.monotonic() - start_time
        if elapsed > startup_max_wait:
            break

        free_mem = get_npu_free_memory()
        now = time.monotonic()

        # Build status
        candidates = []
        for i in needed_npus:
            fm = free_mem.get(i, -1)
            tag = ""
            if i in assigned_npus:
                tag = "[MINE]"
            elif i in failed_npus and now < failed_npus[i]:
                tag = f"[cooldown {failed_npus[i] - now:.0f}s]"
            elif fm >= min_free_mb:
                tag = "[READY]"
            elif fm >= 0:
                tag = f"[{fm}MB < {min_free_mb}MB]"
            else:
                tag = "[?]"
            candidates.append(f"NPU{i}:{tag}")

        status = (
            f"  [{time.strftime('%H:%M:%S')}] "
            f"running={len(running)}/{target}  " + "  ".join(candidates)
        )
        if status != last_status:
            print(status)
            last_status = status

        # If using PegaFlow, verify server is alive before launching
        if use_pegaflow:
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:9091/health", timeout=5)
            except Exception:
                print(f"  [{time.strftime('%H:%M:%S')}] pegaflow-server not "
                      f"reachable — waiting...")
                time.sleep(poll_interval)
                continue

        # Collect all candidates with enough free space
        ready_specs: list[tuple[dict, int, float]] = []  # (spec, npu, gmu)
        for spec in instance_specs:
            npu = spec["physical_npu"]
            if npu in assigned_npus:
                continue
            if npu in failed_npus and now < failed_npus[npu]:
                continue
            fm = free_mem.get(npu, -1)
            if fm < min_free_mb:
                continue
            gmu = max(0.15, min(0.85, (fm - 4096) / HBM_TOTAL_MB))
            ready_specs.append((spec, npu, gmu))

        if not ready_specs:
            time.sleep(poll_interval)
            continue

        # Start all ready instances in parallel
        rlock = threading.Lock()

        def _start_one(spec, npu, gmu):
            label = spec["label"]
            print(f"    → NPU{npu} starting [{label}] :{spec['port']} "
                  f"(gmu={gmu:.2f})...")
            try:
                proc = start_vllm(
                    spec["port"], spec["mode"], spec["namespace"],
                    npu, label, model_path=model_path,
                    gpu_memory_utilization=gmu,
                    use_pegaflow=use_pegaflow,
                )
                with rlock:
                    running.append((spec, proc))
                    assigned_npus.add(npu)
                    failed_npus.pop(npu, None)
                print(f"    → [{label}] ready on NPU{npu} "
                      f"({len(running)}/{target} instances running)")
            except Exception as e:
                print(f"    → [{label}] FAILED on NPU{npu}: {e}")
                with rlock:
                    failed_npus[npu] = now + 30

        with ThreadPoolExecutor(max_workers=len(ready_specs)) as ex:
            futures = [ex.submit(_start_one, s, n, g)
                       for s, n, g in ready_specs]
            for _ in as_completed(futures, timeout=300):
                pass

        if len(running) >= target:
            break
        time.sleep(poll_interval)

    if len(running) < target:
        print(f"  [WARN] Only {len(running)}/{target} instances started "
              f"after {time.monotonic() - start_time:.0f}s")
    else:
        print(f"  All {target} instances running after "
              f"{time.monotonic() - start_time:.0f}s")

    return running


# =========================================================================
# Request helpers
# =========================================================================

def send_one(port: int, prompt: str, model_path: str,
             max_tokens: int = 64, timeout: int = 300) -> dict:
    data = json.dumps({
        "model": model_path, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0.0,
        "stream": True,
    }).encode()
    t0 = time.perf_counter()
    ttft_s = -1.0
    total_s = -1.0
    completion_tokens = 0
    text = ""
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            data=data, headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=timeout)
        # Read SSE stream: measure TTFT from first chunk, total from last
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
                if chunk.get("usage"):
                    completion_tokens = chunk["usage"].get(
                        "completion_tokens", completion_tokens,
                    )
            except json.JSONDecodeError:
                continue
        total_s = time.perf_counter() - t0
        return {
            "ok": True, "port": port,
            "ttft_s": round(ttft_s, 3),
            "total_s": round(total_s, 3),
            "prompt_tokens": 0,  # not in streaming by default
            "completion_tokens": completion_tokens,
            "text": text[:80],
        }
    except Exception as e:
        return {
            "ok": False, "port": port, "ttft_s": -1, "total_s": -1,
            "prompt_tokens": 0, "completion_tokens": 0,
            "error": str(e)[:200],
        }


def send_requests_sequential(
    instances: list[tuple[dict, subprocess.Popen]],
    queries: list[str],
    model_path: str,
    extra_label: str = "",
    identical_prompts: bool = False,
) -> list[dict]:
    """Round-robin requests across instances, ONE at a time.

    Req 0 → I0, Req 1 → I1, ..., Req 7 → I7, Req 8 → I0, ...

    If *identical_prompts* is True, all requests use the SAME prompt
    (system prompt only, no query suffix).  This gives 100% cache hits
    after the first request — max PegaFlow benefit.
    """
    all_results: list[dict] = []
    idx = 0
    for qi in range(len(queries)):
        for ii, (spec, proc) in enumerate(instances):
            if proc.poll() is not None:
                continue
            if identical_prompts:
                # All requests use the exact same prompt → all blocks cacheable
                prompt = f"{SYSTEM_PROMPT}\n\nAssistant:"
                q_label = "IDENTICAL"
            else:
                q = queries[qi]
                prompt = f"{SYSTEM_PROMPT}\n\nUser: {q}\n\nAssistant:"
                q_label = q[:30]
            r = send_one(spec["port"], prompt, model_path)
            r["label"] = spec["label"]
            r["query_idx"] = qi
            r["round"] = idx
            r["extra_label"] = extra_label
            all_results.append(r)
            status = (f"TTFT={r['ttft_s']:.2f}s" if r["ok"]
                      else f"ERR={r.get('error','?')[:40]}")
            print(f"    [{idx:>3d}] {spec['label']} Q{qi} {status} | {q_label}...")
            idx += 1
            time.sleep(0.5)
    return all_results


# =========================================================================
# Phase runner
# =========================================================================

def run_phase(
    name: str,
    instance_specs: list[dict],
    queries: list[str],
    model_path: str,
    already_assigned: set[int],
    warmup_first: bool,
    min_free_mb: int = MIN_FREE_HBM_MB,
    use_pegaflow: bool = True,
    identical_prompts: bool = False,
) -> tuple[dict, set[int]]:
    """Run a full benchmark phase with opportunistic startup.

    Returns (metrics_dict, assigned_npu_ids).
    """
    print(f"\n{'─' * 70}")
    print(f"Phase: {name}")
    print(f"{'─' * 70}")

    t0 = time.perf_counter()

    # 1. Opportunistic startup
    running = launch_instances_opportunistically(
        instance_specs, already_assigned, model_path,
        min_free_mb=min_free_mb,
        use_pegaflow=use_pegaflow,
    )

    if len(running) < len(instance_specs):
        print(f"  [WARN] proceeding with {len(running)}/{len(instance_specs)} instances")

    t_startup = time.perf_counter()
    print(f"  Startup took {t_startup - t0:.0f}s")

    # 2. Warmup (if applicable) — runs BEFORE the timed phase
    if warmup_first and len(running) >= 1:
        warmup_spec, warmup_proc = running[0]
        print(f"  Warmup: [{warmup_spec['label']}] seeds cache "
              f"({len(queries)} queries, not timed)...")
        _warmup_results = send_requests_sequential(
            [(warmup_spec, warmup_proc)], queries, model_path,
            extra_label="WARMUP",
            identical_prompts=identical_prompts,
        )
        ok_w = sum(1 for r in _warmup_results if r.get("ok"))
        print(f"  Warmup done — {ok_w}/{len(queries)} ok, cache seeded")
        print(f"  Waiting 15s for write pipeline to seal blocks...")
        time.sleep(15)

    # 3. Timed phase — round-robin across all instances, one at a time
    print(f"  Timed: round-robin {len(queries)} queries × "
          f"{len(running)} instances = {len(queries) * len(running)} requests...")
    t_send_start = time.perf_counter()
    results = send_requests_sequential(
        running, queries, model_path, extra_label="TIMED",
        identical_prompts=identical_prompts,
    )
    t_send_end = time.perf_counter()

    # 4. Shutdown instances
    assigned = {s["physical_npu"] for s, p in running}
    for _, proc in running:
        stop_proc(proc)

    # 5. Aggregate — only timed results count
    ok = [r for r in results if r.get("ok")]
    failed = len(results) - len(ok)
    ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s", -1) > 0]
    totals = [r["total_s"] for r in ok if r.get("total_s", -1) > 0]
    wall_s = t_send_end - t_send_start

    per_inst: dict[str, dict] = {}
    for spec, _ in running:
        label = spec["label"]
        ir = [r for r in results if r.get("label") == label]
        irok = [r for r in ir if r.get("ok")]
        per_inst[label] = {
            "ok": len(irok), "total": len(ir),
            "avg_ttft_s": round(sum(r["ttft_s"] for r in irok) / len(irok), 2)
            if irok else 0,
        }

    metrics = {
        "phase": name,
        "instances_target": len(instance_specs),
        "instances_running": len(running),
        "requests_total": len(results),
        "requests_ok": len(ok),
        "requests_failed": failed,
        "wall_clock_s": round(wall_s, 1),
        "throughput_req_s": round(len(ok) / wall_s, 2) if wall_s > 0 else 0,
        "avg_ttft_s": round(sum(ttfts) / len(ttfts), 2) if ttfts else 0,
        "min_ttft_s": round(min(ttfts), 2) if ttfts else 0,
        "max_ttft_s": round(max(ttfts), 2) if ttfts else 0,
        "avg_total_s": round(sum(totals) / len(totals), 2) if totals else 0,
        "per_instance": per_inst,
    }

    return metrics, assigned


# =========================================================================
# Hit-rate extraction
# =========================================================================

def extract_hit_rate() -> float:
    server_log = LOG_DIR / "server.log"
    if not server_log.exists():
        return -1.0
    text = server_log.read_text()
    hits = len(re.findall(r"Prefetch local-hit", text))
    total = len(re.findall(r"Prefetch", text))
    return round(hits / total * 100, 1) if total > 0 else -1.0


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PegaFlow 8-Instance Shared Cache Benchmark"
    )
    parser.add_argument("--requests", type=int, default=3,
                        help="Requests per instance (default: 3)")
    parser.add_argument("--pool-size", type=str, default="4096mb",
                        help="PegaFlow pool size (default: 4096mb)")
    parser.add_argument("--startup-max-wait", type=int, default=3600,
                        help="Max seconds to wait for all 8 instances to start (default: 3600)")
    parser.add_argument("--shared-only", action="store_true")
    parser.add_argument("--isolated-only", action="store_true")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip warmup (instance 0 seeding cache first)")
    parser.add_argument("--identical", action="store_true",
                        help="Use IDENTICAL prompts for all requests "
                        "(max cache hit rate, best PegaFlow demo)")
    parser.add_argument("--min-free-gb", type=int, default=28,
                        help="Min free HBM per NPU before attempting startup (default: 20 GB)")
    args = parser.parse_args()

    min_free_mb = args.min_free_gb * 1024

    # ---- Validate model ----
    model_path = MODEL_PATH
    if not Path(model_path).is_dir():
        print(f"[WARN] {model_path} not found, trying fallback...")
        if Path(MODEL_FALLBACK).is_dir():
            model_path = MODEL_FALLBACK
            print(f"  Using {model_path}")
        else:
            print(f"  Download: python -c \"from modelscope import snapshot_download; "
                  f"snapshot_download('Qwen/Qwen3-8B')\"")
            sys.exit(1)

    num_requests = args.requests
    queries = USER_QUERIES[:num_requests]

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Build instance specs ----
    # Each instance gets NPU 0-7 (one per physical NPU)
    def make_specs(namespace_fn) -> list[dict]:
        return [
            {"label": f"I{i}", "port": VLLM_BASE_PORT + i,
             "mode": "read_write", "namespace": namespace_fn(i),
             "physical_npu": i}
            for i in range(NUM_INSTANCES)
        ]

    shared_specs = make_specs(lambda _: SHARED_NS)
    isolated_specs = make_specs(lambda i: f"{ISOLATED_NS_PREFIX}_{i}")

    # ================================================================
    # Start pegaflow-server FIRST, then opportunistically launch vLLM
    # ================================================================
    print("=" * 70)
    print("  PegaFlow 8-Instance Shared Cache Benchmark")
    print(f"  Model: {Path(model_path).name}")
    print(f"  Pool:  {args.pool_size}")
    print(f"  Reqs:  {num_requests}/instance ({num_requests * NUM_INSTANCES} total)")
    print(f"  Prompt: ~{len(SYSTEM_PROMPT.split())} words")
    print("=" * 70)

    kill_all()
    time.sleep(2)

    print(f"\n[1/4] Starting pegaflow-server...")
    server = start_server(list(range(8)), args.pool_size)
    print(f"  Server ready on :{SERVER_PORT}")
    time.sleep(2)

    results_shared = None
    results_isolated = None

    try:
        assigned_npus: set[int] = set()

        # ============================================================
        # Phase 1 — Shared Pool
        # ============================================================
        if not args.isolated_only:
            print(f"\n[2/4] Starting SHARED phase "
                  f"(namespace='{SHARED_NS}', warmup={'ON' if not args.no_warmup else 'OFF'})")
            metrics_s, assigned_s = run_phase(
                "SHARED", shared_specs, queries, model_path,
                already_assigned=assigned_npus,
                warmup_first=not args.no_warmup,
                min_free_mb=min_free_mb,
                use_pegaflow=True,
                identical_prompts=args.identical,
            )
            results_shared = metrics_s
            assigned_npus |= assigned_s
            time.sleep(3)

        # ============================================================
        # Phase 2 — Isolated
        # ============================================================
        if not args.shared_only:
            # After shared phase, instances are shut down. assigned_npus
            # tracks which NPUs we last used — they should free up soon.
            print(f"\n[3/4] Starting ISOLATED phase "
                  f"(8 × unique namespaces, warmup=OFF)")
            metrics_i, assigned_i = run_phase(
                "ISOLATED", isolated_specs, queries, model_path,
                already_assigned=assigned_npus,
                warmup_first=False,
                min_free_mb=min_free_mb,
                use_pegaflow=False,
                identical_prompts=args.identical,
            )
            results_isolated = metrics_i
            assigned_npus |= assigned_i

        # ============================================================
        # Results
        # ============================================================
        hit_rate = extract_hit_rate()

        print("\n" + "=" * 70)
        print("  RESULTS")
        print("=" * 70)

        if results_shared:
            s = results_shared
            print(f"\n  Shared Pool ({s['instances_running']} instances, "
                  f"namespace='{SHARED_NS}'):")
            print(f"    Wall clock:     {s['wall_clock_s']:.1f}s")
            print(f"    Requests:       {s['requests_ok']}/{s['requests_total']} ok")
            print(f"    Throughput:     {s['throughput_req_s']:.2f} req/s")
            print(f"    Avg TTFT:       {s['avg_ttft_s']:.3f}s (first token)")
            print(f"    Avg Total:      {s['avg_total_s']:.2f}s (all tokens)")
            print(f"    Min/Max TTFT:   {s['min_ttft_s']:.3f}s / {s['max_ttft_s']:.3f}s")

        if results_isolated:
            s = results_isolated
            print(f"\n  Isolated ({s['instances_running']} instances, "
                  f"unique namespaces):")
            print(f"    Wall clock:     {s['wall_clock_s']:.1f}s")
            print(f"    Requests:       {s['requests_ok']}/{s['requests_total']} ok")
            print(f"    Throughput:     {s['throughput_req_s']:.2f} req/s")
            print(f"    Avg TTFT:       {s['avg_ttft_s']:.3f}s (first token)")
            print(f"    Avg Total:      {s['avg_total_s']:.2f}s (all tokens)")
            print(f"    Min/Max TTFT:   {s['min_ttft_s']:.3f}s / {s['max_ttft_s']:.3f}s")

        # Per-instance breakdown
        if results_shared and results_isolated:
            print(f"\n  Per-instance TTFT:")
            print(f"  {'Inst':<6} {'Shared':>10} {'Isolated':>10} {'Delta':>10}")
            print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
            for i in range(NUM_INSTANCES):
                label = f"I{i}"
                s_ttft = results_shared.get("per_instance", {}).get(label, {}).get("avg_ttft_s", 0)
                i_ttft = results_isolated.get("per_instance", {}).get(label, {}).get("avg_ttft_s", 0)
                if s_ttft > 0 and i_ttft > 0:
                    delta = i_ttft - s_ttft
                    print(f"  {label:<6} {s_ttft:>7.2f}s   {i_ttft:>7.2f}s   {delta:>+7.2f}s")
                else:
                    print(f"  {label:<6} {s_ttft:>7.2f}s   {i_ttft:>7.2f}s")

        # Comparison table
        if results_shared and results_isolated:
            tp_s = results_shared["throughput_req_s"]
            tp_i = results_isolated["throughput_req_s"]
            ttft_s = results_shared["avg_ttft_s"]
            ttft_i = results_isolated["avg_ttft_s"]

            print(f"\n  {'─' * 60}")
            print(f"  {'Metric':<28} {'Ours':>14} {'Upstream':>14}")
            print(f"  {'-'*28} {'-'*14} {'-'*14}")

            tp_gain = (tp_s - tp_i) / tp_i * 100 if tp_i > 0 else 0
            ttft_red = (ttft_i - ttft_s) / ttft_i * 100 if ttft_i > 0 else 0

            print(f"  {'Throughput gain':<28} {tp_gain:>+13.1f}% {'+56%':>14}")
            print(f"  {'TTFT reduction':<28} {ttft_red:>+13.1f}% {'-36%':>14}")
            print(f"  {'Throughput (shared)':<28} {tp_s:>11.2f} r/s {'11.97':>14}")
            print(f"  {'Throughput (isolated)':<28} {tp_i:>11.2f} r/s {'7.68':>14}")
            print(f"  {'TTFT (shared)':<28} {ttft_s:>11.2f} s {'5.26':>14}")
            print(f"  {'TTFT (isolated)':<28} {ttft_i:>11.2f} s {'8.22':>14}")
            if hit_rate > 0:
                print(f"  {'Hit rate':<28} {hit_rate:>11.1f}% {'52.35%':>14}")

        # Server log diagnostics
        print(f"\n  Server diagnostics:")
        server_log = LOG_DIR / "server.log"
        if server_log.exists():
            text = server_log.read_text()
            for kw in ["Prefetch local-hit", "ERROR", "sealed"]:
                cnt = text.count(kw)
                if cnt:
                    print(f"    '{kw}': {cnt}")

        # Save
        output = LOG_DIR / "results.json"
        with open(output, "w") as f:
            json.dump({
                "benchmark": "8inst_shared_vs_isolated",
                "model": str(Path(model_path).name),
                "instances": NUM_INSTANCES,
                "pool_size": args.pool_size,
                "requests_per_instance": num_requests,
                "shared": results_shared,
                "isolated": results_isolated,
                "hit_rate_from_log": hit_rate,
            }, f, indent=2, default=str)
        print(f"\n  Results: {output}")
        print(f"  Logs:    {LOG_DIR}/")

    finally:
        print(f"\n[4/4] Shutting down...")
        stop_proc(server)
        kill_all()

    print("=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
