#!/usr/bin/env python3
"""
PegaFlow NPU Benchmark: Shared Pool vs Isolated Cache — 7B Model + Long System Prompt

Compares TTFT for multi-instance KV cache sharing with Qwen2.5-7B-Instruct:
  - Shared Pool:  Instance A saves → Instance B loads (same namespace)
  - Isolated:     Instance A saves → Instance C loads (different namespace)

Key fix: PYTHONHASHSEED=0 ensures deterministic block hashes across instances.
Long prompt (~2000 tokens) ensures prefill is expensive enough to show TTFT benefit.

Usage:
    python run_npu_benchmark.py [num_requests]
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — 7B model
# ---------------------------------------------------------------------------
MODEL_PATH = "/workspace/HUST/models/Qwen2.5-7B-Instruct"
PROJECT_ROOT = Path("/workspace/HUST/pegaflow-hust")
LOG_DIR = Path("/tmp/pegaflow-bench-7b")
SERVER_PORT = 50063
VLLM_PORT_A = 18401
VLLM_PORT_B = 18402
VLLM_PORT_C = 18403

# ~2000 token system prompt — makes prefill expensive enough to measure
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
SYSTEM_PROMPT = (_SYS_BLOCK + " ") * 30  # ~3500 tokens, makes prefill expensive

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

SHARED_NS = "bench-7b-shared"
ISOLATED_NS = "bench-7b-isolated"


def make_prompt(suffix: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nUser: {suffix}\n\nAssistant:"


def kill_all():
    for p in ["pegaflow-server", "vllm serve"]:
        os.system(f"pkill -f '{p}' 2>/dev/null || true")


def start_server() -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / "server.log"
    proc = subprocess.Popen(
        [str(PROJECT_ROOT / "target" / "debug" / "pegaflow-server"),
         "--addr", f"0.0.0.0:{SERVER_PORT}", "--pool-size", "1024mb", "--devices", "4"],
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


def start_vllm(port, mode, namespace, device_id, label):
    log = LOG_DIR / f"vllm_{label}.log"
    env = os.environ.copy()
    if namespace:
        env["PEGAFLOW_NAMESPACE"] = namespace
    env["PEGAFLOW_HOST"] = "http://127.0.0.1"
    env["PEGAFLOW_PORT"] = str(SERVER_PORT)
    env["PYTHONHASHSEED"] = "0"
    # Restrict vLLM to free NPUs 4-7 (maps to local 0-3).
    # NpuIPCWrapper and _resolve_device_id both map local→global (→4).
    env["ASCEND_RT_VISIBLE_DEVICES"] = "4,5,6,7"

    kv_cfg = json.dumps({
        "kv_connector": "PegaKVConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "pegaflow.connector",
        "kv_connector_extra_config": {
            "pegaflow.mode": mode,
            "pegaflow.transfer_backend": "ascend_direct",
        },
    })

    cmd = (
        f"source /root/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate vllm-hust-dev && "
        f"vllm serve {MODEL_PATH} --port {port} --dtype float16 "
        f"--max-model-len 8192 --max-num-seqs 2 --gpu-memory-utilization 0.85 "
        f"--enforce-eager --kv-transfer-config '{kv_cfg}'"
    )
    proc = subprocess.Popen(
        ["bash", "-c", cmd], env=env,
        stdout=open(log, "w"), stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    # 7B model loading + compile takes ~120-180s on NPU
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            return proc
        except Exception:
            time.sleep(3)
    raise RuntimeError(f"vLLM {label} failed")


def stop_proc(proc):
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=15)
        except Exception:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def send_requests(port, queries, label):
    results = []
    for i, query in enumerate(queries):
        prompt = make_prompt(query)
        data = json.dumps({
            "model": MODEL_PATH, "prompt": prompt,
            "max_tokens": 32, "temperature": 0.0,
        }).encode()
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/completions",
                data=data, headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=300)
            body = json.loads(resp.read())
            t1 = time.perf_counter()
            ttft_ms = round((t1 - t0) * 1000, 1)
            text = body["choices"][0]["text"]
            results.append({"idx": i, "query": query[:40], "ttft_ms": ttft_ms, "text": text[:60]})
            print(f"    {label}[{i}] TTFT={ttft_ms:>8.0f}ms | {query[:35]}... → {text[:30].strip()}")
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:300]
            results.append({"idx": i, "query": query[:40], "ttft_ms": -1, "text": "", "error": body})
            print(f"    {label}[{i}] HTTP {e.code}: {body}")
        except Exception as e:
            results.append({"idx": i, "query": query[:40], "ttft_ms": -1, "text": "", "error": str(e)[:200]})
            print(f"    {label}[{i}] ERROR: {str(e)[:200]}")
        time.sleep(1)
    return results


def stats(results):
    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] > 0]
    if not ttfts:
        return {"count": 0, "avg": 0, "min": 0, "max": 0}
    return {"count": len(ttfts), "avg": round(sum(ttfts) / len(ttfts), 1),
            "min": round(min(ttfts), 1), "max": round(max(ttfts), 1)}


# ---------------------------------------------------------------------------
def main():
    num_requests = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    queries = USER_QUERIES[:num_requests]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    kill_all()
    time.sleep(2)

    print("=" * 65)
    print("  PegaFlow NPU Benchmark — 7B Model + Long System Prompt")
    print(f"  Model: Qwen2.5-7B-Instruct | Requests: {num_requests}")
    print(f"  System prompt: ~{len(SYSTEM_PROMPT.split())} words (~3500 tokens)")
    print("  PYTHONHASHSEED=0 (deterministic cross-instance block hashes)")
    print("=" * 65)

    server = start_server()
    print(f"\n[Server] Ready on :{SERVER_PORT}\n")
    time.sleep(2)

    try:
        # ============================================================
        # Phase 1: Instance A (SAVE_ONLY) — seed cache
        # ============================================================
        print("─" * 65)
        print("Phase 1: Instance A (SAVE_ONLY) — seeding cache")
        print("─" * 65)
        print("  Starting 7B model (loading + compile ~120-180s)...")
        a = start_vllm(VLLM_PORT_A, "save_only", SHARED_NS, 0, "A_save")
        results_a = send_requests(VLLM_PORT_A, queries, "A")
        stats_a = stats(results_a)
        print(f"  Instance A: avg TTFT = {stats_a['avg']}ms (baseline, {stats_a['count']} ok)")
        stop_proc(a)
        time.sleep(5)  # Allow write pipeline to seal all blocks

        # ============================================================
        # Phase 2: Instance B (READ_WRITE, SHARED namespace)
        # ============================================================
        print(f"\n{'─' * 65}")
        print("Phase 2: Instance B (READ_WRITE, SHARED namespace)")
        print("─" * 65)
        print("  Starting 7B model...")
        b = start_vllm(VLLM_PORT_B, "read_write", SHARED_NS, 0, "B_shared")
        results_b = send_requests(VLLM_PORT_B, queries, "B")
        stats_b = stats(results_b)
        print(f"  Instance B (shared): avg TTFT = {stats_b['avg']}ms ({stats_b['count']} ok)")
        stop_proc(b)
        time.sleep(2)

        # ============================================================
        # Phase 3: Instance C (READ_WRITE, ISOLATED namespace)
        # ============================================================
        print(f"\n{'─' * 65}")
        print("Phase 3: Instance C (READ_WRITE, ISOLATED namespace)")
        print("─" * 65)
        print("  Starting 7B model...")
        c = start_vllm(VLLM_PORT_C, "read_write", ISOLATED_NS, 0, "C_isolated")
        results_c = send_requests(VLLM_PORT_C, queries, "C")
        stats_c = stats(results_c)
        print(f"  Instance C (isolated): avg TTFT = {stats_c['avg']}ms ({stats_c['count']} ok)")
        stop_proc(c)

        # ============================================================
        # Results
        # ============================================================
        print("\n" + "=" * 65)
        print("  RESULTS: 7B Model — Shared Pool vs Isolated Cache")
        print("=" * 65)
        print(f"  {'Metric':<30} {'Shared (B)':>12} {'Isolated (C)':>12} {'Delta':>12}")
        print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")

        for metric in ["avg", "min", "max"]:
            sb = stats_b[metric]
            sc = stats_c[metric]
            if sb > 0 and sc > 0:
                diff = sc - sb
                pct_str = f" ({diff / sc * 100:+.1f}%)" if sc > 0 else ""
                print(f"  {metric.capitalize() + ' TTFT':<30} {sb:>8.0f}ms   {sc:>8.0f}ms   {diff:>+8.0f}ms{pct_str}")
            else:
                print(f"  {metric.capitalize() + ' TTFT':<30} {sb:>8.0f}ms   {sc:>8.0f}ms")

        # First-request comparison
        if len(results_b) > 0 and len(results_c) > 0:
            b0 = results_b[0]["ttft_ms"]
            c0 = results_c[0]["ttft_ms"]
            print("\n  First request TTFT (cold for both):")
            print(f"    Shared B[0]:   {b0:.0f}ms")
            print(f"    Isolated C[0]: {c0:.0f}ms")
            if b0 > 0 and c0 > 0:
                r = c0 - b0
                print(f"    Reduction:     {r:+.0f}ms ({r / c0 * 100:+.1f}%)")

        # Server diagnostics
        print("\n  --- Server diagnostics ---")
        server_log = LOG_DIR / "server.log"
        if server_log.exists():
            content = server_log.read_text()
            for kw in ["listening", "Prefetch local-hit", "ERROR", "aclrtMemcpyBatchAsync"]:
                count = content.count(kw)
                if count:
                    print(f"    '{kw}': {count} occurrences")

        # Show prefetch hit lines
        for line in server_log.read_text().split("\n"):
            if "Prefetch local-hit" in line:
                print(f"    PREFETCH: {line.strip()[:200]}")

        # Save results
        output = LOG_DIR / "results.json"
        with open(output, "w") as f:
            json.dump({
                "model": "Qwen2.5-7B-Instruct",
                "prompt_words": len(SYSTEM_PROMPT.split()),
                "requests": num_requests,
                "stats_a_save": stats_a,
                "stats_b_shared": stats_b,
                "stats_c_isolated": stats_c,
                "raw_b": results_b,
                "raw_c": results_c,
            }, f, indent=2)
        print(f"\n  Results: {output}")
        print(f"  Logs:    {LOG_DIR}/")

    finally:
        stop_proc(server)
        kill_all()

    # Verdict
    print(f"\n{'=' * 65}")
    if stats_b["avg"] > 0 and stats_c["avg"] > 0:
        reduction = (stats_c["avg"] - stats_b["avg"]) / stats_c["avg"] * 100
        if reduction > 10:
            print(f"  ✓ Shared pool reduces TTFT by {reduction:.1f}% — PegaFlow works on NPU!")
        elif reduction > 3:
            print(f"  ✓ Modest {reduction:.1f}% TTFT reduction — cache sharing active,"
                  f" benefit scales with model/prompt size")
        else:
            print(f"  ⚠ Only {reduction:.1f}% reduction — check server log for hit/miss details")
    else:
        print("  ⚠ Insufficient data")
    print("=" * 65)


if __name__ == "__main__":
    main()
