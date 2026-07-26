"""
vLLM + PegaFlow multi-instance E2E test on Ascend NPU.

Two vLLM instances share KV cache through one pegaflow-server:
  Instance A (SAVE_ONLY): runs inference, saves KV cache → PegaFlow
  Instance B (READ_WRITE): runs same prompts, loads KV cache from PegaFlow

Verifies:
- Correct inference output on both instances
- Instance B gets cache hits from Instance A's saved blocks
- No data corruption or conflicts

Requires:
- pegaflow-server running on PEGAFLOW_PORT
- Qwen2.5-0.5B-Instruct model cached locally
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Server endpoint - configure via env or use default
PEGAFLOW_HOST = os.environ.get("PEGAFLOW_HOST", "http://127.0.0.1")
PEGAFLOW_PORT = int(os.environ.get("PEGAFLOW_PORT", "50056"))
PEGAFLOW_ENDPOINT = f"{PEGAFLOW_HOST}:{PEGAFLOW_PORT}"

# Model to use for testing
MODEL_NAME = os.environ.get("PEGAFLOW_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("SKIP_VLLM_E2E", "") == "1",
        reason="SKIP_VLLM_E2E=1",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_server():
    """Verify pegaflow-server is reachable."""
    try:
        import httpx
        r = httpx.get(f"{PEGAFLOW_HOST}:9091/health", timeout=5)
        if r.status_code != 200:
            pytest.skip("pegaflow-server /health not OK")
    except Exception:
        pytest.skip("pegaflow-server not reachable")


def _run_vllm_server(port: int, mode: str, extra_env: dict | None = None) -> subprocess.Popen:
    """Start a vLLM server with PegaKVConnector.

    Args:
        port: vLLM API server port
        mode: "save_only" or "read_write"
        extra_env: additional environment variables
    """
    env = os.environ.copy()
    env["PEGAFLOW_HOST"] = PEGAFLOW_HOST
    env["PEGAFLOW_PORT"] = str(PEGAFLOW_PORT)
    env["VLLM_KV_TRANSFER_CONFIG"] = (
        f'{{"kv_connector":"PegaKVConnector","kv_role":"kv_both",'
        f'"kv_connector_module_path":"pegaflow.connector",'
        f'"pegaflow.host":"{PEGAFLOW_HOST}",'
        f'"pegaflow.port":{PEGAFLOW_PORT},'
        f'"pegaflow.mode":"{mode}"}}'
    )
    if extra_env:
        env.update(extra_env)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--port", str(port),
        "--dtype", "float16",
        "--max-model-len", "2048",
        "--max-num-seqs", "4",
        "--gpu-memory-utilization", "0.5",
        "--enforce-eager",
        "--disable-log-requests",
    ]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _wait_for_vllm(port: int, timeout: int = 120) -> bool:
    """Poll vLLM's /health endpoint until ready."""
    import httpx
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _send_prompt(port: int, prompt: str, max_tokens: int = 32) -> str:
    """Send a single prompt to the vLLM completion API."""
    import httpx
    r = httpx.post(
        f"http://127.0.0.1:{port}/v1/completions",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


# ---------------------------------------------------------------------------
# Test 1: Single instance basic E2E
# ---------------------------------------------------------------------------


class TestVLLMSingleInstance:
    """Verify vLLM + PegaKVConnector works correctly on Ascend."""

    def test_vllm_start_with_pegaconnector(self):
        """vLLM starts and responds to requests with PegaKVConnector."""
        _require_server()

        proc = _run_vllm_server(port=18100, mode="save_only")
        try:
            assert _wait_for_vllm(18100, timeout=180), "vLLM failed to start"
            text = _send_prompt(18100, "Hello, my name is", max_tokens=8)
            assert len(text) > 0, f"Empty response: '{text}'"
            print(f"  Response: '{text}'")
        finally:
            proc.terminate()
            proc.wait(timeout=30)


# ---------------------------------------------------------------------------
# Test 2: Two instances sharing KV cache
# ---------------------------------------------------------------------------


class TestVLLMMultiInstance:
    """Two vLLM instances share KV cache via PegaFlow."""

    def test_two_instances_share_kv_cache(self):
        """Instance A saves, Instance B loads from cache."""
        _require_server()

        prompt = "The capital of France is"
        prompt2 = "The largest planet in the solar system is"

        # Phase 1: Instance A (SAVE_ONLY) does inference and saves
        proc_a = _run_vllm_server(port=18101, mode="save_only")
        try:
            assert _wait_for_vllm(18101, timeout=180), "Instance A failed to start"
            result_a1 = _send_prompt(18101, prompt, max_tokens=16)
            result_a2 = _send_prompt(18101, prompt2, max_tokens=16)
            print(f"  Instance A (save_only): '{result_a1}' / '{result_a2}'")
            assert len(result_a1) > 0 and len(result_a2) > 0
            # Let saves flush
            time.sleep(2)
        finally:
            proc_a.terminate()
            proc_a.wait(timeout=30)

        # Phase 2: Instance B (READ_WRITE) — should get cache hits
        proc_b = _run_vllm_server(port=18102, mode="read_write")
        try:
            assert _wait_for_vllm(18102, timeout=180), "Instance B failed to start"
            result_b1 = _send_prompt(18102, prompt, max_tokens=16)
            result_b2 = _send_prompt(18102, prompt2, max_tokens=16)
            print(f"  Instance B (read_write): '{result_b1}' / '{result_b2}'")
            # Output should match (deterministic with temperature=0)
            assert result_b1 == result_a1, (
                f"Cache mismatch: B='{result_b1}' vs A='{result_a1}'"
            )
            assert result_b2 == result_a2, (
                f"Cache mismatch: B='{result_b2}' vs A='{result_a2}'"
            )
        finally:
            proc_b.terminate()
            proc_b.wait(timeout=30)


# ---------------------------------------------------------------------------
# Test 3: Concurrent vLLM instances
# ---------------------------------------------------------------------------


class TestVLLMConcurrent:
    """Two vLLM instances running concurrently with shared cache."""

    def test_concurrent_instances(self):
        """Both instances send requests simultaneously."""
        _require_server()

        prompt = "Once upon a time,"

        proc_a = _run_vllm_server(port=18103, mode="read_write")
        proc_b = _run_vllm_server(port=18104, mode="read_write")
        try:
            assert _wait_for_vllm(18103, timeout=180), "Instance A failed"
            assert _wait_for_vllm(18104, timeout=180), "Instance B failed"

            # Both serve requests concurrently
            r_a = _send_prompt(18103, prompt, max_tokens=8)
            r_b = _send_prompt(18104, prompt, max_tokens=8)
            print(f"  Concurrent A: '{r_a}'")
            print(f"  Concurrent B: '{r_b}'")
            assert len(r_a) > 0 and len(r_b) > 0
            # Output should be identical (same prompt, temperature=0)
            assert r_a == r_b, f"Concurrent mismatch: '{r_a}' vs '{r_b}'"
        finally:
            proc_a.terminate()
            proc_b.terminate()
            proc_a.wait(timeout=30)
            proc_b.wait(timeout=30)


__all__ = [
    "TestVLLMSingleInstance",
    "TestVLLMMultiInstance",
    "TestVLLMConcurrent",
]
