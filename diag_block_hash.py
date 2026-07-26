#!/usr/bin/env python3
"""
Diagnostic: verify that blocks saved via vLLM's PegaKVConnector are findable.

Two scenarios:
  A) Save via vLLM connector → query via EngineRpcClient directly
  B) Save via EngineRpcClient directly → query via EngineRpcClient

This isolates whether the issue is in the vLLM connector's save path
or in the PegaFlow server's read_cache.
"""
import json
import os
import signal
import subprocess
import sys
import time
import uuid
import urllib.request
from pathlib import Path

MODEL_PATH = "/workspace/HUST/models/models/Qwen--Qwen2.5-0.5B-Instruct/snapshots/master"
PROJECT_ROOT = Path("/workspace/HUST/pegaflow-hust")
LOG_DIR = Path("/tmp/pegaflow-diag")
SERVER_PORT = 50062
VLLM_PORT = 18400

SHARED_NS = "diag-shared-ns"

SYSTEM_PROMPT = (
    "You are an expert AI assistant. Answer questions accurately and concisely. "
    "Always follow instructions precisely and think step by step. "
    "Provide evidence-based answers when asked factual questions. "
    "You are an expert AI assistant. Answer questions accurately and concisely. "
    "Always follow instructions precisely and think step by step. "
    "Provide evidence-based answers when asked factual questions. "
)

PROMPT = f"{SYSTEM_PROMPT}\n\nUser: What is the capital of France?\n\nAssistant:"


def kill_all():
    for p in ["pegaflow-server", "vllm serve"]:
        os.system(f"pkill -f '{p}' 2>/dev/null || true")


def start_server() -> subprocess.Popen:
    log = LOG_DIR / "server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [str(PROJECT_ROOT / "target" / "debug" / "pegaflow-server"),
         "--addr", f"0.0.0.0:{SERVER_PORT}", "--pool-size", "256mb", "--devices", "0"],
        stdout=open(log, "w"), stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    for _ in range(15):
        time.sleep(1)
        if log.exists() and "listening" in log.read_text():
            return proc
    raise RuntimeError("Server failed")


def start_vllm(mode="save_only", namespace=None) -> subprocess.Popen:
    """Start vLLM with PegaKVConnector."""
    log = LOG_DIR / f"vllm_{mode}.log"
    env = os.environ.copy()
    env["ASCEND_VISIBLE_DEVICES"] = "0"
    env["PEGAFLOW_HOST"] = "http://127.0.0.1"
    env["PEGAFLOW_PORT"] = str(SERVER_PORT)
    if namespace:
        env["PEGAFLOW_NAMESPACE"] = namespace

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
        f"vllm serve {MODEL_PATH} --port {VLLM_PORT} --dtype float16 "
        f"--max-model-len 1024 --max-num-seqs 2 --gpu-memory-utilization 0.4 "
        f"--enforce-eager --kv-transfer-config '{kv_cfg}'"
    )
    proc = subprocess.Popen(
        ["bash", "-c", cmd], env=env,
        stdout=open(log, "w"), stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{VLLM_PORT}/health", timeout=5)
            return proc
        except Exception:
            time.sleep(2)
    raise RuntimeError("vLLM failed")


def stop_proc(proc):
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def send_completion(prompt, max_tokens=16):
    data = json.dumps({
        "model": MODEL_PATH, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{VLLM_PORT}/v1/completions",
        data=data, headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=120)
    return json.loads(resp.read())["choices"][0]["text"]


def check_server_cache():
    """Check server /metrics for cache stats."""
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:9091/metrics", timeout=5)
        # Look for any cache-related metrics
        body = resp.read().decode()
        for line in body.split("\n"):
            if "pega" in line.lower() or "cache" in line.lower() or "block" in line.lower():
                print(f"  METRIC: {line}")
    except Exception as e:
        print(f"  Cannot reach /metrics: {e}")


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    kill_all()
    time.sleep(2)

    print("=" * 60)
    print("  PegaFlow Diagnostic: Block Save/Query Test")
    print("=" * 60)

    server = start_server()
    print(f"\nServer ready on :{SERVER_PORT}\n")
    time.sleep(2)

    try:
        # =====================================================
        # Phase 1: Save via vLLM connector
        # =====================================================
        print("Phase 1: Save via vLLM PegaKVConnector (SAVE_ONLY)")
        print("-" * 40)

        va = start_vllm("save_only", SHARED_NS)
        output = send_completion(PROMPT)
        print(f"  vLLM output: '{output[:60]}...'")
        time.sleep(2)  # Let save flush

        # Check vLLM metrics for save stats
        vllm_log = LOG_DIR / "vllm_save_only.log"
        if vllm_log.exists():
            for line in vllm_log.read_text().split("\n"):
                if "KV Transfer metrics" in line or "cache_lookup" in line or "save_intent" in line:
                    print(f"  vLLM: {line.strip()[:150]}")

        stop_proc(va)
        time.sleep(3)  # Wait for write pipeline to seal

        # =====================================================
        # Phase 2: Query via EngineRpcClient directly
        # =====================================================
        print("\nPhase 2: Query directly via EngineRpcClient")
        print("-" * 40)

        from pegaflow.pegaflow import EngineRpcClient, QueryReady, QueryLoading
        import torch
        from pegaflow.npu_ipc_wrapper import NpuIPCWrapper
        import pickle
        import hashlib

        client = EngineRpcClient(f"http://127.0.0.1:{SERVER_PORT}")
        ok, msg = client.health()
        print(f"  Engine health: {ok} — {msg}")

        # Allocate a tensor for registration
        num_blocks = 32
        block_tokens = 64
        k = torch.zeros(num_blocks, block_tokens, dtype=torch.float16, device="npu:0")
        wrapper = NpuIPCWrapper(k)
        bytes_per_block = k.stride(0) * k.element_size()

        iid = f"diag-{uuid.uuid4().hex[:8]}"

        try:
            ok, msg = client.register_context_batch(
                iid, SHARED_NS,
                tp_rank=0, pp_rank=0, tp_size=1, world_size=1, device_id=0,
                layer_names=["layer_0"],
                wrapper_bytes_list=[pickle.dumps(wrapper)],
                num_blocks_list=[num_blocks],
                bytes_per_block_list=[bytes_per_block],
                kv_stride_bytes_list=[0],
                segments_list=[1],
                transfer_backend="ascend_direct",
                page_first=False,
            )
            print(f"  Register: {ok} — {msg}")

            # Now compute the SAME block hashes that vLLM would compute.
            # vLLM computes hashes as: hash((parent_hash, token_ids_tuple, extra_keys))
            # We need the actual token IDs from the prompt.
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            token_ids = tokenizer.encode(PROMPT)
            print(f"  Prompt tokens: {len(token_ids)}")

            # Compute vLLM-style block hashes (hash chain)
            block_size = 16
            block_hashes = []
            parent_hash = None
            import hashlib
            for i in range(0, len(token_ids), block_size):
                chunk = tuple(token_ids[i:i + block_size])
                if not chunk:
                    break
                if parent_hash is None:
                    parent_hash = b'\x00' * 32  # vLLM's NONE_HASH is 32 zero bytes
                data = (parent_hash, chunk, None)  # extra_keys=None for text-only
                h = hashlib.sha256(str(data).encode()).digest()
                parent_hash = h
                block_hashes.append(h)
            print(f"  Computed {len(block_hashes)} block hashes for prompt")

            # Query for these block hashes
            for attempt in range(15):
                result = client.query_prefetch(iid, block_hashes, "diag-req-0")
                if isinstance(result, QueryReady):
                    print(f"  Query result (attempt {attempt}): hit_blocks={result.num_hit_blocks}/{len(block_hashes)}")
                    if result.lease:
                        client.release(result.lease)
                    break
                elif isinstance(result, QueryLoading):
                    time.sleep(0.2)
            else:
                print(f"  Query still loading after 15 attempts")

            # Also try querying with just the first hash
            first_hash = block_hashes[:1]
            result = client.query_prefetch(iid, first_hash, "diag-req-1")
            if isinstance(result, QueryReady):
                print(f"  Query first-hash-only: hit_blocks={result.num_hit_blocks}/1")
                if result.lease:
                    client.release(result.lease)
            else:
                print(f"  First-hash query: {type(result).__name__}")

        finally:
            client.unregister_context(iid)

        # Check server log for write pipeline activity
        print("\nPhase 3: Server write pipeline diagnostics")
        print("-" * 40)
        server_log = LOG_DIR / "server.log"
        if server_log.exists():
            for line in server_log.read_text().split("\n"):
                if any(kw in line.lower() for kw in ["seal", "insert_worker", "inflight", "save_batch", "prefetch", "hit"]):
                    print(f"  {line.strip()[:200]}")
            print(f"\n  Full log: {LOG_DIR / 'server.log'}")

        check_server_cache()

    finally:
        stop_proc(server)
        kill_all()


if __name__ == "__main__":
    main()
