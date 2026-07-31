#!/usr/bin/env python3
"""
PegaFlow Benchmark — MLA + TP8 Concurrent Throughput Test

DeepSeek-V2-Lite (MLA) with tensor parallelism across 8 NPUs.
Measures throughput improvement from PegaFlow shared cache.

MLA (Multi-head Latent Attention): compressed KV cache (kv_lora_rank=512).
DMA transfers are proportionally cheaper vs recomputation, amplifying
PegaFlow's benefit.

Test plan:
  Phase 1 — Instance A (SAVE_ONLY, TP8, SHARED namespace):
      Process long system prompt, save KV blocks to PegaFlow.
  Phase 2 — Instance B (READ_WRITE, TP8, SHARED namespace):
      Same prompt → cache hits → DMA from host instead of recompute.
  Phase 3 — Instance C (READ_WRITE, TP8, ISOLATED namespace):
      Same prompt, different namespace → full recompute (baseline).

Each phase fires N requests concurrently to measure throughput.

Usage:
  python run_bench_mla_tp8_concurrent.py --requests 5 --identical
"""

from __future__ import annotations

import argparse, json, os, re, signal, subprocess, sys, threading, time
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/workspace/HUST/pegaflow-hust")
LOG_DIR = Path("/tmp/pegaflow-bench-mla-tp8-conc")

MODEL_PATH = "/workspace/HUST/models/DeepSeek-V2-Lite"
MODEL_FALLBACK = "/workspace/HUST/models/Qwen2.5-7B-Instruct"

SERVER_PORT = 50072
VLLM_PORT_A = 18901  # Phase 1: SAVE_ONLY
VLLM_PORT_B = 18902  # Phase 2: SHARED
VLLM_PORT_C = 18903  # Phase 3: ISOLATED

SHARED_NS = "bench-mla-tp8-shared"
ISOLATED_NS = "bench-mla-tp8-isolated"
HBM_TOTAL_MB = 65536
MIN_FREE_HBM_MB = 20 * 1024  # TP8 means each NPU holds 1/8 of model

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
]


# =========================================================================
# NPU polling — all 8 needed for TP8
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


def wait_for_all_npus(min_free_mb=MIN_FREE_HBM_MB, max_wait_s=3600, poll_s=10):
    """Wait until ALL 8 NPUs have >= min_free_mb of free HBM.

    TP8 splits model across 8 NPUs (~4 GB weights/NPU for 16B model),
    so individual NPU requirements are lower than single-instance tests.
    """
    print(f"  Waiting for all 8 NPUs to have >= {min_free_mb}MB "
          f"(~{min_free_mb//1024}GB) free...")
    start = time.monotonic()
    last_status = ""
    while True:
        free_mem = get_npu_free_memory()
        ready = sum(1 for i in range(8) if free_mem.get(i, -1) >= min_free_mb)
        min_free = min(free_mem.get(i, -1) for i in range(8))
        status = "  ".join(f"NPU{i}:{free_mem.get(i,-1)}MB" for i in range(8))
        if status != last_status:
            print(f"  [{time.strftime('%H:%M:%S')}] ready={ready}/8 "
                  f"min_free={min_free}MB  {status}")
            last_status = status
        if ready >= 8:
            print(f"  All 8 NPUs ready after {time.monotonic()-start:.0f}s")
            # Return free memory for dynamic gpu_memory_utilization
            return {i: free_mem.get(i, -1) for i in range(8)}
        if time.monotonic() - start > max_wait_s:
            raise RuntimeError(
                f"Timeout waiting for NPUs ({ready}/8 ready). "
                f"Min free: {min_free}MB. Lower --min-free-gb to relax."
            )
        time.sleep(poll_s)


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


def start_server(pool_size="4096mb"):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / "server.log"
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
    deadline = time.time() + 30
    while time.time() < deadline:
        time.sleep(1)
        try:
            if log.exists() and "listening" in log.read_text():
                return proc
        except Exception:
            pass
    raise RuntimeError(f"Server failed. {log.read_text()[-300:] if log.exists() else ''}")


def start_vllm_tp8(port, mode, namespace, label, *, model_path,
                   tp_size=8, use_pegaflow=True,
                   gpu_memory_utilization=0.85):
    """Start a vLLM instance with TP8 across all 8 NPUs.

    gpu_memory_utilization should be computed from the least-free NPU
    to avoid OOM when NPUs have uneven free space.
    """
    log = LOG_DIR / f"vllm_{label}.log"
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
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
        f"--tensor-parallel-size {tp_size}",
        f"--gpu-memory-utilization {gmu:.2f}",
        f"--enforce-eager",  # DeepSeek MoE incompatible with ACLGraph compile
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
    deadline = time.time() + 300  # TP8 model takes longer to load
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            return proc
        except Exception:
            time.sleep(3)
    raise RuntimeError(f"vLLM {label} failed. {log.read_text()[-400:] if log.exists() else ''}")


# =========================================================================
# Streaming request
# =========================================================================

def send_one_streaming(port, prompt, model_path, max_tokens=64, timeout=600):
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
# Phase runner
# =========================================================================

def run_concurrent_phase(phase_name, port, queries, model_path,
                         identical_prompts, max_output_tokens):
    """Fire N requests concurrently to a single vLLM instance."""
    total = len(queries)
    print(f"  Firing {total} concurrent requests...")
    t0 = time.perf_counter()
    results: list[dict] = []
    rlock = threading.Lock()

    def fire_one(q, qi):
        if identical_prompts:
            prompt = f"{SYSTEM_PROMPT}\n\nAssistant:"
        else:
            prompt = f"{SYSTEM_PROMPT}\n\nUser: {q}\n\nAssistant:"
        r = send_one_streaming(port, prompt, model_path,
                               max_tokens=max_output_tokens)
        r["query_idx"] = qi
        with rlock:
            results.append(r)
        status = (f"TTFT={r['ttft_s']:.3f}s tot={r['total_s']:.2f}s"
                  if r["ok"] else f"ERR={r.get('error','?')[:30]}")
        print(f"    [{phase_name} Q{qi}] {status}")

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(fire_one, q, i) for i, q in enumerate(queries)]
        for _ in as_completed(futures, timeout=600):
            pass

    t1 = time.perf_counter()
    wall_s = t1 - t0
    ok = [r for r in results if r.get("ok")]
    ok_count = len(ok)
    ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s", -1) > 0]
    totals = [r["total_s"] for r in ok if r.get("total_s", -1) > 0]

    return {
        "phase": phase_name,
        "queries": total,
        "ok": ok_count,
        "failed": total - ok_count,
        "wall_clock_s": round(wall_s, 1),
        "throughput_req_s": round(ok_count / wall_s, 2) if wall_s > 0 else 0,
        "avg_ttft_s": round(sum(ttfts) / len(ttfts), 3) if ttfts else 0,
        "min_ttft_s": round(min(ttfts), 3) if ttfts else 0,
        "max_ttft_s": round(max(ttfts), 3) if ttfts else 0,
        "avg_total_s": round(sum(totals) / len(totals), 2) if totals else 0,
        "raw": results,
    }


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PegaFlow MLA+TP8 Concurrent Throughput Benchmark"
    )
    parser.add_argument("--requests", type=int, default=5,
                        help="Concurrent requests per phase (default: 5)")
    parser.add_argument("--pool-size", type=str, default="4096mb")
    parser.add_argument("--min-free-gb", type=int, default=20,
                        help="Min free HBM per NPU (default: 20)")
    parser.add_argument("--max-wait", type=int, default=3600)
    parser.add_argument("--identical", action="store_true",
                        help="Use identical prompts for all requests")
    parser.add_argument("--max-output-tokens", type=int, default=64,
                        help="Max output tokens per request (default: 64)")
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    args = parser.parse_args()

    min_free_mb = args.min_free_gb * 1024
    model_path = args.model
    num_requests = args.requests
    queries = USER_QUERIES[:num_requests]
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Check model
    if not Path(model_path).is_dir():
        if Path(MODEL_FALLBACK).is_dir():
            model_path = MODEL_FALLBACK
        else:
            sys.exit(f"Model not found: {model_path}")

    # Check if MLA
    is_mla = False
    cfg_path = Path(model_path) / "config.json"
    if cfg_path.exists():
        cfg = json.load(open(cfg_path))
        is_mla = "kv_lora_rank" in cfg

    print("=" * 70)
    print("  PegaFlow MLA+TP8 Concurrent Throughput Benchmark")
    print(f"  Model:     {Path(model_path).name}")
    print(f"  MLA:       {'YES (kv_lora_rank)' if is_mla else 'NO'}")
    print(f"  TP size:   8")
    print(f"  Requests:  {num_requests} concurrent/phase")
    print(f"  Prompt:    {'identical' if args.identical else 'mixed'}")
    print(f"  Output:    max {args.max_output_tokens} tokens")
    print(f"  Pool:      {args.pool_size}")
    print("=" * 70)

    kill_all()
    time.sleep(2)

    def _compute_gmu():
        """Re-poll NPUs and compute safe gpu_memory_utilization."""
        free_mem = wait_for_all_npus(min_free_mb, args.max_wait)
        min_free = min(free_mem.values())
        gmu = max(0.10, min(0.85, (min_free - 2048) / HBM_TOTAL_MB))
        print(f"  GPU mem utilization: {gmu:.2f} "
              f"(least-free NPU: {min_free}MB)")
        return gmu

    # ---- Start server ----
    print(f"\n[Server] Starting...")
    server = start_server(args.pool_size)
    print(f"  Server ready on :{SERVER_PORT}")
    time.sleep(2)

    results_a = results_b = None

    try:
        # ============================================================
        # Instance A: SHARED namespace — warmup then test (same instance)
        # Using read_write so it both saves and reads from cache.
        # ============================================================
        gmu = _compute_gmu()
        print(f"\n{'─'*70}")
        print("Phase 1+2: Instance A (read_write, SHARED) — seed + test")
        print(f"{'─'*70}")
        a = start_vllm_tp8(VLLM_PORT_A, "read_write", SHARED_NS, "A_shared",
                           model_path=model_path, use_pegaflow=True,
                           gpu_memory_utilization=gmu)

        # Phase 1: Warmup — first request seeds the shared cache
        print(f"\n  Phase 1 (WARMUP): seeding cache with first query...")
        warmup_q = queries[:1]
        results_warmup = run_concurrent_phase(
            "WARMUP", VLLM_PORT_A, warmup_q, model_path,
            args.identical, args.max_output_tokens,
        )
        print(f"  Warmup done — TTFT={results_warmup['avg_ttft_s']:.3f}s")
        print(f"  Waiting 15s for write pipeline to seal blocks...")
        time.sleep(15)

        # Phase 2: Concurrent burst — same instance, should get cache hits
        print(f"\n  Phase 2 (SHARED): {len(queries)} concurrent requests "
              f"(same instance)...")
        results_a = run_concurrent_phase(
            "A_SHARED", VLLM_PORT_A, queries, model_path,
            args.identical, args.max_output_tokens,
        )
        print(f"  Phase 2 done — TTFT={results_a['avg_ttft_s']:.3f}s, "
              f"throughput={results_a['throughput_req_s']:.2f} req/s")
        stop_proc(a)
        time.sleep(5)

        # ============================================================
        # Instance B: ISOLATED namespace — baseline (no shared cache)
        # ============================================================
        gmu = _compute_gmu()
        print(f"\n{'─'*70}")
        print("Phase 3: Instance B (read_write, ISOLATED) — baseline")
        print(f"{'─'*70}")
        b = start_vllm_tp8(VLLM_PORT_B, "read_write", ISOLATED_NS, "B_iso",
                           model_path=model_path, use_pegaflow=True,
                           gpu_memory_utilization=gmu)
        results_b = run_concurrent_phase(
            "B_ISOLATED", VLLM_PORT_B, queries, model_path,
            args.identical, args.max_output_tokens,
        )
        print(f"  Phase 3 done — TTFT={results_b['avg_ttft_s']:.3f}s, "
              f"throughput={results_b['throughput_req_s']:.2f} req/s")
        stop_proc(b)

        # ============================================================
        # Results
        # ============================================================
        print("\n" + "=" * 70)
        print("  RESULTS — MLA+TP8 Throughput")
        print("=" * 70)

        for tag, r in [("A (Shared)", results_a), ("B (Isolated)", results_b)]:
            if r is None:
                continue
            print(f"\n  {tag}:")
            print(f"    Wall clock:     {r['wall_clock_s']:.1f}s")
            print(f"    Requests:       {r['ok']}/{r['queries']} ok")
            print(f"    Throughput:     {r['throughput_req_s']:.2f} req/s")
            print(f"    Avg TTFT:       {r['avg_ttft_s']:.3f}s (first token)")
            print(f"    Avg Total:      {r['avg_total_s']:.2f}s (all tokens)")
            print(f"    TTFT range:     {r['min_ttft_s']:.3f}s — {r['max_ttft_s']:.3f}s")

        if results_a and results_b:
            tp_shared = results_a["throughput_req_s"]
            tp_iso = results_b["throughput_req_s"]
            ttft_shared = results_a["avg_ttft_s"]
            ttft_iso = results_b["avg_ttft_s"]

            print(f"\n  {'─'*60}")
            print(f"  {'Metric':<30} {'Shared':>12} {'Isolated':>12} {'Delta':>12}")
            print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*12}")

            tp_gain = (tp_shared - tp_iso) / tp_iso * 100 if tp_iso > 0 else 0
            ttft_red = (ttft_iso - ttft_shared) / ttft_iso * 100 if ttft_iso > 0 else 0

            print(f"  {'Throughput (req/s)':<30} {tp_shared:>10.2f}   {tp_iso:>10.2f}   {tp_gain:>+10.1f}%")
            print(f"  {'Avg TTFT (s)':<30} {ttft_shared:>10.3f}   {ttft_iso:>10.3f}   {ttft_red:>+10.1f}%")

            print(f"\n  Upstream reference (H800, DeepSeek-V3.2):")
            print(f"    Throughput gain: +72%")
            print(f"    (MLA compressed KV → faster DMA → bigger PegaFlow benefit)")

        # Server diagnostics
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
                "benchmark": "mla_tp8_concurrent",
                "model": str(Path(model_path).name),
                "is_mla": is_mla,
                "tp_size": 8,
                "requests": num_requests,
                "identical_prompts": args.identical,
                "max_output_tokens": args.max_output_tokens,
                "phase_shared": results_a,
                "phase_isolated": results_b,
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
