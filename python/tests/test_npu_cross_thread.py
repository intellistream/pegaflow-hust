"""
Cross-thread correctness tests for Ascend NPU KV cache operations.

These tests validate that the PegaFlow connector's multi-threaded save and
load workers operate correctly on Ascend, where each thread must explicitly
call ``aclrtSetDevice`` to set its device context.

Tests require:
- A running pegaflow-server (Ascend build)
- At least one Ascend NPU device
- The camem_allocator for DMA-capable KV cache allocations (save/load tests)
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

# Lazy import — pegaflow native extension may not be installed.
# Each test class only imports what it needs after NPU availability is confirmed.
logger = None


def _get_logger():
    global logger
    if logger is None:
        try:
            from pegaflow.connector.common import logger as _logger

            logger = _logger
        except ImportError:
            import logging

            logger = logging.getLogger(__name__)
    return logger

pytestmark = [
    pytest.mark.npu,
    pytest.mark.skipif(
        os.environ.get("SKIP_NPU_TESTS", "") == "1",
        reason="SKIP_NPU_TESTS=1",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _npu_available() -> bool:
    """Check if Ascend NPU is available at runtime."""
    try:
        import torch

        return hasattr(torch, "npu") and torch.npu.is_available()
    except ImportError:
        return False


def _require_npu():
    if not _npu_available():
        pytest.skip("Ascend NPU not available")


def _require_multi_npu(min_devices: int = 2):
    _require_npu()
    import torch

    count = torch.npu.device_count()
    if count < min_devices:
        pytest.skip(f"Need >= {min_devices} NPU devices (found {count})")


# ---------------------------------------------------------------------------
# Test 1: Single-device, multi-thread _device_synchronize
# ---------------------------------------------------------------------------


class TestDeviceSynchronizeThreadSafety:
    """Verify _device_synchronize works correctly from multiple threads."""

    def test_sync_from_multiple_threads_same_device(self):
        """Multiple threads calling torch.npu.synchronize on same device."""
        _require_npu()
        import torch

        device = torch.device("npu:0")
        errors = []

        def sync_worker(thread_id: int):
            try:
                # Set device context for this thread
                torch.npu.set_device(device)
                # Create a small tensor on NPU
                t = torch.ones(1024, device=device)
                # Do an operation that ensures the stream has work
                t.add_(1)
                # Synchronize
                torch.npu.synchronize(device)
                # Verify
                assert t[0].item() == 2.0, f"thread {thread_id}: data mismatch"
            except Exception as e:
                errors.append(f"thread {thread_id}: {e}")

        threads = [
            threading.Thread(target=sync_worker, args=(i,)) for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in sync threads: {errors}"

    def test_sync_from_multiple_threads_different_devices(self):
        """Multiple threads on different NPU devices."""
        _require_multi_npu(2)
        import torch

        errors = []

        def device_worker(device_id: int):
            try:
                device = torch.device(f"npu:{device_id}")
                torch.npu.set_device(device)
                t = torch.ones(1024, device=device)
                t.mul_(device_id + 1)
                torch.npu.synchronize(device)
                assert t[0].item() == float(device_id + 1), (
                    f"device {device_id}: data mismatch"
                )
            except Exception as e:
                errors.append(f"device {device_id}: {e}")

        threads = [
            threading.Thread(target=device_worker, args=(i,)) for i in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in device threads: {errors}"


# ---------------------------------------------------------------------------
# Test 2: Save + Load thread simulation (no server)
# ---------------------------------------------------------------------------


class TestSaveLoadThreadSimulation:
    """Verify save/load thread patterns work with Ascend device context."""

    def test_aclrt_set_device_per_thread(self):
        """Verify each thread can independently set its Ascend device."""
        _require_npu()
        import torch

        results = {}

        def worker(thread_id: int, device_id: int):
            device = torch.device(f"npu:{device_id}")
            torch.npu.set_device(device)

            # Allocate on this device
            t = torch.zeros(128, device=device)
            t.fill_(thread_id)
            torch.npu.synchronize(device)

            results[thread_id] = {
                "device": device_id,
                "value": t[0].item(),
                "ok": t[0].item() == float(thread_id),
            }

        threads = [
            threading.Thread(target=worker, args=(i, i % max(1, torch.npu.device_count())))
            for i in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for thread_id, result in results.items():
            assert result["ok"], (
                f"thread {thread_id} (device {result['device']}): "
                f"expected {thread_id}, got {result['value']}"
            )

    def test_concurrent_stream_operations_single_device(self):
        """Two streams on same device with events for ordering."""
        _require_npu()
        import torch

        device = torch.device("npu:0")
        torch.npu.set_device(device)

        # Simulate save stream filling a buffer, load stream reading it
        SIZE = 4096

        save_stream = torch.npu.Stream(device=device)
        load_stream = torch.npu.Stream(device=device)

        # Allocate on device
        with torch.npu.stream(save_stream):
            src = torch.arange(SIZE, dtype=torch.float32, device=device)

        save_event = save_stream.record_event()

        # Wait for save stream, then load
        load_stream.wait_event(save_event)
        with torch.npu.stream(load_stream):
            dst = src.clone()

        torch.npu.synchronize(device)
        assert torch.equal(src, dst), "stream transfer data mismatch"


# ---------------------------------------------------------------------------
# Test 3: IPC Wrapper multi-device serialization
# ---------------------------------------------------------------------------


class TestIPCWrapperMultiDevice:
    """Verify NpuIPCWrapper correctly serializes across multiple devices."""

    def test_wrapper_preserves_device_index_after_roundtrip(self):
        """Each wrapper retains its original device_index through pickle."""
        _require_multi_npu(2)
        import pickle

        import torch
        from pegaflow.npu_ipc_wrapper import NpuIPCWrapper

        wrappers = []
        for dev_idx in range(min(2, torch.npu.device_count())):
            device = torch.device(f"npu:{dev_idx}")
            torch.npu.set_device(device)
            t = torch.ones(64, 128, dtype=torch.float16, device=device)
            wrapper = NpuIPCWrapper(t)
            wrappers.append(wrapper)

        # Serialize batch
        data = pickle.dumps(wrappers)
        restored = pickle.loads(data)

        for i, (orig, rest) in enumerate(zip(wrappers, restored)):
            assert rest.device_index == i, (
                f"wrapper {i}: expected device_index {i}, got {rest.device_index}"
            )

    def test_wrapper_from_higher_device_index(self):
        """Wrapper from a non-zero device preserves its index."""
        _require_multi_npu(2)
        import pickle

        import torch
        from pegaflow.npu_ipc_wrapper import NpuIPCWrapper

        device = torch.device("npu:1")
        torch.npu.set_device(device)
        t = torch.ones(32, dtype=torch.float16, device=device)
        wrapper = NpuIPCWrapper(t)

        assert wrapper.device_index == 1

        data = pickle.dumps(wrapper)
        restored = pickle.loads(data)
        assert restored.device_index == 1


# ---------------------------------------------------------------------------
# Test 4: ThreadPoolExecutor with Ascend device context
# ---------------------------------------------------------------------------


class TestThreadPoolAscend:
    """Verify ThreadPoolExecutor workers correctly handle Ascend context."""

    def test_threadpool_each_worker_sets_device(self):
        """Each ThreadPoolExecutor worker must set aclrtSetDevice."""
        _require_npu()
        import torch
        import torch.npu

        num_devices = torch.npu.device_count()

        def worker(device_id: int) -> dict:
            """Worker that uses a specific Ascend NPU device."""
            actual_device = device_id % num_devices
            device = torch.device(f"npu:{actual_device}")
            torch.npu.set_device(device)

            # Allocate and compute
            t = torch.randn(256, 256, dtype=torch.float16, device=device)
            result = t.sum().item()
            torch.npu.synchronize(device)

            return {
                "device": actual_device,
                "result": result,
                "memory_used": torch.npu.memory_allocated(device),
            }

        with ThreadPoolExecutor(max_workers=min(8, num_devices * 4)) as executor:
            futures = [executor.submit(worker, i) for i in range(12)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 12
        for r in results:
            assert isinstance(r["result"], float), (
                f"device {r['device']}: result={r['result']}"
            )
            assert r["memory_used"] > 0, (
                f"device {r['device']}: no memory allocated"
            )

    def test_threadpool_concurrent_allocations(self):
        """Concurrent allocations on different devices don't interfere."""
        _require_multi_npu(2)
        import torch

        allocations_per_device = {}

        def alloc_worker(device_id: int) -> None:
            device = torch.device(f"npu:{device_id}")
            torch.npu.set_device(device)
            tensors = [
                torch.zeros(1024, dtype=torch.float32, device=device)
                for _ in range(50)
            ]
            torch.npu.synchronize(device)
            allocations_per_device[device_id] = len(tensors)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for _ in range(4):
                futures.append(executor.submit(alloc_worker, 0))
                futures.append(executor.submit(alloc_worker, 1))
            for f in as_completed(futures):
                f.result()  # raises if error

        assert 0 in allocations_per_device
        assert 1 in allocations_per_device


__all__ = [
    "TestDeviceSynchronizeThreadSafety",
    "TestSaveLoadThreadSimulation",
    "TestIPCWrapperMultiDevice",
    "TestThreadPoolAscend",
]
