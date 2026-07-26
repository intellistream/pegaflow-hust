"""
End-to-end NPU multi-instance tests for PegaFlow on Ascend hardware.

Prerequisites:
- Ascend NPU device(s) accessible
- pegaflow-server running on 127.0.0.1:50055
"""

import os
import pickle
import uuid

import pytest

pytestmark = [
    pytest.mark.npu_multi,
    pytest.mark.skipif(
        os.environ.get("SKIP_NPU_E2E_TESTS", "") == "1",
        reason="SKIP_NPU_E2E_TESTS=1",
    ),
]


def _require_npu_e2e():
    """Check server is reachable; skip otherwise."""
    try:
        from pegaflow.pegaflow import EngineRpcClient
        c = EngineRpcClient()
        ok, _ = c.health()
        if not ok:
            pytest.skip("pegaflow-server not healthy")
    except Exception:
        pytest.skip("pegaflow-server not reachable")


def _make_registration_params(device: str = "npu:0"):
    """Create a real NpuIPCWrapper and compute registration parameters.

    Returns (wrapper_bytes, num_blocks, bytes_per_block) derived from
    a real NPU tensor. Must use actual tensor metadata — hardcoded
    values cause 'registered memory too small' errors server-side.
    """
    import torch
    from pegaflow.npu_ipc_wrapper import NpuIPCWrapper

    num_blocks = 16
    block_tokens = 128
    t = torch.zeros(num_blocks, block_tokens, dtype=torch.float16, device=device)
    wrapper = NpuIPCWrapper(t)
    # bytes_per_block = stride(0) * element_size (vllm-ascend pattern)
    bytes_per_block = t.stride(0) * t.element_size()
    return pickle.dumps(wrapper), num_blocks, bytes_per_block


def _do_register(client, instance_id, ns, wrapper_bytes, num_blocks, bytes_per_block):
    return client.register_context_batch(
        instance_id, ns,
        tp_rank=0, pp_rank=0, tp_size=1, world_size=1, device_id=0,
        layer_names=["layer_0"],
        wrapper_bytes_list=[wrapper_bytes],
        num_blocks_list=[num_blocks],
        bytes_per_block_list=[bytes_per_block],
        kv_stride_bytes_list=[0],
        segments_list=[1],
        transfer_backend="ascend_direct",
        page_first=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNPUE2EBasic:
    """Basic connectivity and lifecycle tests with real NPU tensors."""

    def test_server_health(self):
        _require_npu_e2e()
        from pegaflow.pegaflow import EngineRpcClient
        c = EngineRpcClient()
        ok, msg = c.health()
        assert ok, f"Health check: {msg}"

    def test_register_with_real_tensor(self):
        """Register context with properly computed tensor parameters."""
        _require_npu_e2e()
        from pegaflow.pegaflow import EngineRpcClient

        client = EngineRpcClient()
        iid = f"test-{uuid.uuid4().hex[:8]}"
        ns = "test-ns"
        wb, nb, bpb = _make_registration_params("npu:0")

        try:
            ok, msg = _do_register(client, iid, ns, wb, nb, bpb)
            assert ok, f"Register: {msg}"
        finally:
            client.unregister_context(iid)

    def test_register_multi_layer(self):
        """Register context with two layers."""
        _require_npu_e2e()
        from pegaflow.pegaflow import EngineRpcClient

        client = EngineRpcClient()
        iid = f"test-{uuid.uuid4().hex[:8]}"
        ns = "test-ns"
        wb0, nb0, bpb0 = _make_registration_params("npu:0")
        wb1, nb1, bpb1 = _make_registration_params("npu:0")

        try:
            ok, msg = client.register_context_batch(
                iid, ns,
                tp_rank=0, pp_rank=0, tp_size=1, world_size=1, device_id=0,
                layer_names=["layer_0", "layer_1"],
                wrapper_bytes_list=[wb0, wb1],
                num_blocks_list=[nb0, nb1],
                bytes_per_block_list=[bpb0, bpb1],
                kv_stride_bytes_list=[0, 0],
                segments_list=[1, 1],
                transfer_backend="ascend_direct",
                page_first=False,
            )
            assert ok, f"Register multi: {msg}"
        finally:
            client.unregister_context(iid)

    def test_instance_re_register(self):
        """Unregister then re-register with same topology works."""
        _require_npu_e2e()
        from pegaflow.pegaflow import EngineRpcClient

        client = EngineRpcClient()
        iid = f"test-{uuid.uuid4().hex[:8]}"
        ns = "test-ns"
        wb, nb, bpb = _make_registration_params("npu:0")

        try:
            ok, msg = _do_register(client, iid, ns, wb, nb, bpb)
            assert ok, f"First register: {msg}"
            client.unregister_context(iid)

            ok, msg = _do_register(client, iid, ns, wb, nb, bpb)
            assert ok, f"Second register: {msg}"
        finally:
            client.unregister_context(iid)


class TestNPUErrorHandling:
    """Verify error handling paths on the server."""

    def test_invalid_device_rejected(self):
        _require_npu_e2e()
        from pegaflow.pegaflow import EngineRpcClient, PegaFlowError
        client = EngineRpcClient()
        wb, nb, bpb = _make_registration_params("npu:0")
        # Direct call with invalid device_id must raise
        with pytest.raises((ValueError, PegaFlowError)):
            client.register_context_batch(
                f"t-{uuid.uuid4().hex[:8]}", "ns",
                tp_rank=0, pp_rank=0, tp_size=1, world_size=1, device_id=-1,
                layer_names=["layer_0"],
                wrapper_bytes_list=[wb],
                num_blocks_list=[nb],
                bytes_per_block_list=[bpb],
                kv_stride_bytes_list=[0],
                segments_list=[1],
                transfer_backend="ascend_direct",
                page_first=False,
            )

    def test_unregister_nonexistent(self):
        _require_npu_e2e()
        from pegaflow.pegaflow import EngineRpcClient, PegaFlowError
        client = EngineRpcClient()
        with pytest.raises((ValueError, PegaFlowError)):
            client.unregister_context("no-such-instance-xyz")

    def test_query_nonexistent(self):
        _require_npu_e2e()
        from pegaflow.pegaflow import EngineRpcClient, PegaFlowError
        client = EngineRpcClient()
        with pytest.raises((ValueError, PegaFlowError)):
            client.query_prefetch("no-such-instance", [b"hash"], "req-0")


__all__ = ["TestNPUE2EBasic", "TestNPUErrorHandling"]
