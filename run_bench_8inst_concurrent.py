#!/usr/bin/env python3
"""
PegaFlow Benchmark — Concurrent Throughput Test

Sends requests to ALL 8 instances concurrently to measure aggregate
throughput under burst load.  Compares shared pool vs isolated cache.

Key difference from run_bench_8inst.py:
  - Requests fire simultaneously (not round-robin)
  - Measures aggregate req/s across all 8 instances
  - Designed to show throughput gains from cached prefill

Usage:
  python run_bench_8inst_concurrent.py --identical --requests 5
"""

from __future__ import annotations

import argparse, json, os, re, signal, subprocess, sys, threading, time
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/workspace/HUST/pegaflow-hust")
LOG_DIR = Path("/tmp/pegaflow-bench-concurrent")
MODEL_PATH = "/workspace/HUST/models/Qwen3-8B"
MODEL_FALLBACK = "/workspace/HUST/models/Qwen2.5-7B-Instruct"
SERVER_PORT = 50071
VLLM_BASE_PORT = 18800
SHARED_NS = "bench-conc-shared"
ISOLATED_NS_PREFIX = "bench-conc-iso"
NUM_INSTANCES = 8
HBM_TOTAL_MB = 65536
MIN_FREE_HBM_MB = 28 * 1024

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
    "Explain the concept of quantum entanglement in simple terms.",
    "What are the three laws of thermodynamics?",
    "How does a nuclear reactor generate electricity?",
    "What is CRISPR and how is it used in genetic engineering?",
    "Describe the Big Bang theory of the universe.",
    "How does blockchain technology work?",
    "What causes earthquakes and how are they measured?",
    "Explain the difference between TCP and UDP protocols.",
]


# =========================================================================
# NPU polling (shared with run_bench_8inst)
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
# Process helpers
# =========================================================================

def kill_all():
    for p in ["pegaflow-server", "vllm serve"]:
        os.system(f"pkill -f '{p}' 2>/dev/null || true")


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


def start_server(free_npus, pool_size):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / "server.log"
    devices = ",".join(str(i) for i in free_npus)
    proc = subprocess.Popen(
        [
            str(PROJECT_ROOT / "target" / "debug" / "pegaflow-server"),
            "--addr", f"0.0.0.0:{SERVER_PORT}",
            "--pool-size", pool_size, "--devices", devices,
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
    raise RuntimeError(f"Server failed. {log.read_text()[-300:] if log.exists() else ''}")


def start_vllm(port, mode, namespace, physical_npu, label, *,
               model_path, gpu_memory_utilization=0.85, use_pegaflow=True):
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
        f"--max-model-len 16384 --max-num-seqs 8",
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
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            return proc
        except Exception:
            time.sleep(3)
    raise RuntimeError(f"vLLM {label} failed. {log.read_text()[-400:] if log.exists() else ''}")


# =========================================================================
# Opportunistic launcher (same logic as run_bench_8inst)
# =========================================================================

def launch_instances(specs, model_path, min_free_mb=MIN_FREE_HBM_MB,
                     use_pegaflow=True, max_wait=3600):
    target = len(specs)
    running: list[tuple[dict, subprocess.Popen]] = []
    assigned: set[int] = set()
    failed_cooldown: dict[int, float] = {}
    needed = {s["physical_npu"] for s in specs}

    print(f"  Need {target} instances, min free HBM: {min_free_mb}MB")
    start_t = time.monotonic()
    last_status = ""

    while len(running) < target:
        if time.monotonic() - start_t > max_wait:
            break
        free_mem = get_npu_free_memory()
        now = time.monotonic()

        if use_pegaflow:
            try:
                urllib.request.urlopen("http://127.0.0.1:9091/health", timeout=5)
            except Exception:
                print(f"  [{time.strftime('%H:%M:%S')}] server not ready...")
                time.sleep(10)
                continue

        status_parts = [f"running={len(running)}/{target}"]
        for i in sorted(needed):
            fm = free_mem.get(i, -1)
            if i in assigned:
                tag = "[MINE]"
            elif i in failed_cooldown and now < failed_cooldown[i]:
                tag = f"[CD {failed_cooldown[i]-now:.0f}s]"
            elif fm >= min_free_mb:
                tag = "[OK]"
            else:
                tag = f"[{fm}MB]"
            status_parts.append(f"NPU{i}:{tag}")
        status = f"  [{time.strftime('%H:%M:%S')}] " + "  ".join(status_parts)
        if status != last_status:
            print(status)
            last_status = status

        # Collect all candidates with enough free space
        ready_specs: list[tuple[dict, int, float]] = []
        for spec in specs:
            npu = spec["physical_npu"]
            if npu in assigned or (npu in failed_cooldown and now < failed_cooldown[npu]):
                continue
            fm = free_mem.get(npu, -1)
            if fm < min_free_mb:
                continue
            gmu = max(0.15, min(0.85, (fm - 4096) / HBM_TOTAL_MB))
            ready_specs.append((spec, npu, gmu))

        if not ready_specs:
            time.sleep(10)
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
                    assigned.add(npu)
                    failed_cooldown.pop(npu, None)
                print(f"    → [{label}] ready on NPU{npu} "
                      f"({len(running)}/{target})")
            except Exception as e:
                print(f"    → [{label}] FAILED on NPU{npu}: {e}")
                with rlock:
                    failed_cooldown[npu] = now + 30

        with ThreadPoolExecutor(max_workers=len(ready_specs)) as ex:
            futures = [ex.submit(_start_one, s, n, g)
                       for s, n, g in ready_specs]
            for _ in as_completed(futures, timeout=300):
                pass

        if len(running) >= target:
            break
        time.sleep(10)

    elapsed = time.monotonic() - start_t
    print(f"  {len(running)}/{target} instances running after {elapsed:.0f}s")
    return running


# =========================================================================
# Streaming send_one — measures TTFT (first token) + total time
# =========================================================================

def send_one_streaming(port, prompt, model_path, max_tokens=64, timeout=300):
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
            "completion_tokens": completion_tokens,
            "text": text[:80],
        }
    except Exception as e:
        return {
            "ok": False, "port": port,
            "ttft_s": -1, "total_s": -1,
            "completion_tokens": 0, "error": str(e)[:200],
        }


# =========================================================================
# Phase runner — concurrent burst
# =========================================================================

def run_phase_concurrent(
    name, instance_specs, queries, model_path,
    already_assigned, use_pegaflow, identical_prompts, min_free_mb,
    warmup_first, max_output_tokens=64,
):
    print(f"\n{'─'*70}")
    print(f"Phase: {name} ({len(instance_specs)} instances, "
          f"pegaflow={'ON' if use_pegaflow else 'OFF'}, "
          f"identical={'ON' if identical_prompts else 'OFF'})")
    print(f"{'─'*70}")

    # 1. Opportunistic startup
    running = launch_instances(instance_specs, model_path,
                               min_free_mb=min_free_mb,
                               use_pegaflow=use_pegaflow)

    # 2. Warmup — seed cache with one instance (not timed)
    if warmup_first and len(running) >= 1 and use_pegaflow:
        warmup_spec, warmup_proc = running[0]
        print(f"  Warmup: [{warmup_spec['label']}] seeds cache...")
        prompt = f"{SYSTEM_PROMPT}\n\nAssistant:"
        r = send_one_streaming(
            warmup_spec["port"], prompt, model_path,
            max_tokens=max_output_tokens,
        )
        print(f"  Warmup done — TTFT={r['ttft_s']:.3f}s, "
              f"total={r['total_s']:.2f}s, ok={r['ok']}")
        print(f"  Waiting 15s for write pipeline to seal blocks...")
        time.sleep(15)

    # 3. Concurrent burst
    # Build all (instance, query) tasks, shuffled to interleave instances.
    import random
    tasks = []
    for qi, q in enumerate(queries):
        for spec, proc in running:
            if proc.poll() is not None:
                continue
            tasks.append((spec, q, qi))
    random.shuffle(tasks)  # interleave instances to avoid burst

    max_concurrent = 4  # limit concurrent DMA to avoid PCIe saturation
    total = len(tasks)
    print(f"  Sending {total} requests "
          f"({len(queries)} queries × {len(running)} instances) "
          f"with max {max_concurrent} concurrent...")

    t0 = time.perf_counter()
    results: list[dict] = []
    rlock = threading.Lock()
    sem = threading.Semaphore(max_concurrent)
    next_delay = 0.1  # 100ms stagger between batch submissions

    def fire_one(inst_spec, query, qi, stagger_delay=0):
        if stagger_delay > 0:
            time.sleep(stagger_delay)
        with sem:  # limit concurrent in-flight requests
            npu = inst_spec["physical_npu"]
            if identical_prompts:
                prompt = f"{SYSTEM_PROMPT}\n\nAssistant:"
            else:
                prompt = f"{SYSTEM_PROMPT}\n\nUser: {query}\n\nAssistant:"
            r = send_one_streaming(
                inst_spec["port"], prompt, model_path,
                max_tokens=max_output_tokens,
            )
            r["label"] = inst_spec["label"]
            r["npu"] = npu
            r["query_idx"] = qi
            with rlock:
                results.append(r)
            status = (f"TTFT={r['ttft_s']:.3f}s" if r["ok"]
                      else f"ERR={r.get('error','?')[:30]}")
            print(f"    [{inst_spec['label']} Q{qi} NPU{npu}] {status}")

    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        futures = []
        for idx, (spec, q, qi) in enumerate(tasks):
            delay = (idx // max_concurrent) * next_delay
            futures.append(ex.submit(fire_one, spec, q, qi, delay))
        for _ in as_completed(futures, timeout=600):
            pass

    t1 = time.perf_counter()
    wall_s = t1 - t0

    # 4. Shutdown
    assigned = {s["physical_npu"] for s, p in running}
    for _, proc in running:
        stop_proc(proc)

    # 5. Aggregate
    ok = [r for r in results if r.get("ok")]
    failed = len(results) - len(ok)
    ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s", -1) > 0]
    totals = [r["total_s"] for r in ok if r.get("total_s", -1) > 0]

    per_inst = {}
    for spec, _ in running:
        label = spec["label"]
        ir = [r for r in results if r.get("label") == label]
        irok = [r for r in ir if r.get("ok")]
        per_inst[label] = {
            "ok": len(irok), "total": len(ir),
            "avg_ttft_s": round(sum(r["ttft_s"] for r in irok) / len(irok), 3)
            if irok else 0,
            "avg_total_s": round(sum(r["total_s"] for r in irok) / len(irok), 2)
            if irok else 0,
            "npu": spec["physical_npu"],
        }

    return {
        "phase": name,
        "instances_target": len(instance_specs),
        "instances_running": len(running),
        "requests_total": len(results),
        "requests_ok": len(ok),
        "requests_failed": failed,
        "wall_clock_s": round(wall_s, 1),
        "throughput_req_s": round(len(ok) / wall_s, 2) if wall_s > 0 else 0,
        "avg_ttft_s": round(sum(ttfts) / len(ttfts), 3) if ttfts else 0,
        "min_ttft_s": round(min(ttfts), 3) if ttfts else 0,
        "max_ttft_s": round(max(ttfts), 3) if ttfts else 0,
        "avg_total_s": round(sum(totals) / len(totals), 2) if totals else 0,
        "per_instance": per_inst,
        "raw": results,
    }, assigned


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PegaFlow Concurrent Throughput Benchmark"
    )
    parser.add_argument("--requests", type=int, default=3,
                        help="Queries per instance (default: 3)")
    parser.add_argument("--pool-size", type=str, default="4096mb")
    parser.add_argument("--min-free-gb", type=int, default=28)
    parser.add_argument("--max-wait", type=int, default=3600)
    parser.add_argument("--identical", action="store_true",
                        help="Use identical prompts for all requests")
    parser.add_argument("--shared-only", action="store_true")
    parser.add_argument("--isolated-only", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--max-output-tokens", type=int, default=64,
                        help="Max output tokens per request (default: 64)")
    args = parser.parse_args()

    min_free_mb = args.min_free_gb * 1024
    model_path = MODEL_PATH
    if not Path(model_path).is_dir():
        if Path(MODEL_FALLBACK).is_dir():
            model_path = MODEL_FALLBACK
        else:
            sys.exit("Model not found")

    queries = USER_QUERIES[:args.requests]
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Instance specs
    def make_specs(ns_fn, mode="read_write"):
        return [
            {"label": f"I{i}", "port": VLLM_BASE_PORT + i,
             "mode": mode, "namespace": ns_fn(i), "physical_npu": i}
            for i in range(NUM_INSTANCES)
        ]

    shared_specs = make_specs(lambda _: SHARED_NS)
    isolated_specs = make_specs(lambda i: f"{ISOLATED_NS_PREFIX}_{i}")

    print("=" * 70)
    print("  PegaFlow Concurrent Throughput Benchmark")
    print(f"  Model: {Path(model_path).name}")
    print(f"  Queries/instance: {args.requests}")
    print(f"  Total requests: {args.requests * NUM_INSTANCES}")
    print(f"  Mode: {'identical' if args.identical else 'mixed'} prompts")
    print(f"  Max output tokens: {args.max_output_tokens}")
    print("=" * 70)

    kill_all()
    time.sleep(2)

    print("\n[1/3] Starting pegaflow-server...")
    server = start_server(list(range(8)), args.pool_size)
    print(f"  Server ready on :{SERVER_PORT}")
    time.sleep(2)

    assigned = set()
    results_s, results_i = None, None

    try:
        if not args.isolated_only:
            print(f"\n[2/3] SHARED phase")
            results_s, assigned_s = run_phase_concurrent(
                "SHARED", shared_specs, queries, model_path,
                already_assigned=assigned,
                use_pegaflow=True,
                identical_prompts=args.identical,
                min_free_mb=min_free_mb,
                warmup_first=not args.no_warmup,
                max_output_tokens=args.max_output_tokens,
            )
            assigned |= assigned_s
            time.sleep(3)

        if not args.shared_only:
            print(f"\n[3/3] ISOLATED phase")
            results_i, assigned_i = run_phase_concurrent(
                "ISOLATED", isolated_specs, queries, model_path,
                already_assigned=assigned,
                use_pegaflow=False,
                identical_prompts=args.identical,
                min_free_mb=min_free_mb,
                warmup_first=False,
                max_output_tokens=args.max_output_tokens,
            )
            assigned |= assigned_i

        # === Results ===
        print("\n" + "=" * 70)
        print("  RESULTS — Concurrent Throughput")
        print("=" * 70)

        for tag, r in [("Shared", results_s), ("Isolated", results_i)]:
            if r is None:
                continue
            print(f"\n  {tag} ({r['instances_running']} instances):")
            print(f"    Wall clock:     {r['wall_clock_s']:.1f}s")
            print(f"    Requests:       {r['requests_ok']}/{r['requests_total']} ok")
            print(f"    Throughput:     {r['throughput_req_s']:.2f} req/s")
            print(f"    Avg TTFT:       {r['avg_ttft_s']:.3f}s (first token)")
            print(f"    Avg Total:      {r['avg_total_s']:.2f}s (all tokens)")
            print(f"    TTFT range:     {r['min_ttft_s']:.3f}s — {r['max_ttft_s']:.3f}s")

        if results_s and results_i:
            tp_s = results_s["throughput_req_s"]
            tp_i = results_i["throughput_req_s"]
            ttft_s = results_s["avg_ttft_s"]
            ttft_i = results_i["avg_ttft_s"]

            print(f"\n  {'─'*60}")
            print(f"  {'Metric':<30} {'Shared':>12} {'Isolated':>12} {'Delta':>12}")
            print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*12}")
            tp_gain = (tp_s - tp_i) / tp_i * 100 if tp_i > 0 else 0
            ttft_red = (ttft_i - ttft_s) / ttft_i * 100 if ttft_i > 0 else 0
            print(f"  {'Throughput (req/s)':<30} {tp_s:>10.2f}   {tp_i:>10.2f}   {tp_gain:>+10.1f}%")
            print(f"  {'Avg TTFT (s)':<30} {ttft_s:>10.3f}   {ttft_i:>10.3f}   {ttft_red:>+10.1f}%")
            print(f"  {'TTFT reduction':<30} {'':>12} {'':>12} {ttft_red:>+10.1f}%" if ttft_red < 0 else
                  f"  {'TTFT reduction':<30} {'':>12} {'':>12} {ttft_red:>+10.1f}%")

            print(f"\n  Per-instance TTFT:")
            print(f"  {'Inst':<6} {'NPU':>4} {'Shared':>10} {'Isolated':>10} {'Speedup':>10}")
            print(f"  {'─'*6} {'─'*4} {'─'*10} {'─'*10} {'─'*10}")
            for i in range(NUM_INSTANCES):
                label = f"I{i}"
                s_ttft = results_s["per_instance"].get(label, {}).get("avg_ttft_s", 0)
                i_ttft = results_i["per_instance"].get(label, {}).get("avg_ttft_s", 0)
                npu = results_s["per_instance"].get(label, {}).get("npu", i)
                if s_ttft > 0 and i_ttft > 0:
                    sp = i_ttft / s_ttft if s_ttft > 0 else 0
                    print(f"  {label:<6} NPU{npu:<1}  {s_ttft:>7.3f}s   {i_ttft:>7.3f}s   {sp:>7.1f}×")
                else:
                    print(f"  {label:<6} NPU{npu:<1}  {s_ttft:>7.3f}s   {i_ttft:>7.3f}s")

        # Save
        output = LOG_DIR / "results.json"
        with open(output, "w") as f:
            json.dump({
                "benchmark": "concurrent_throughput",
                "model": str(Path(model_path).name),
                "queries_per_instance": args.requests,
                "identical_prompts": args.identical,
                "max_output_tokens": args.max_output_tokens,
                "shared": results_s,
                "isolated": results_i,
            }, f, indent=2, default=str)
        print(f"\n  Results: {output}")
        print(f"  Logs:    {LOG_DIR}/")

    finally:
        print("\nShutting down...")
        stop_proc(server)
        kill_all()

    print("=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
