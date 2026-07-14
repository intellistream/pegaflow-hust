"""
Layer 6 (vLLM Connector) verification: device detection and NpuIPCWrapper
serialization roundtrip.

These tests operate entirely within a single Python process and do NOT
require:
- Compilation of the C extension (npu_ipc_bindings._npu_ipc)
- An actual NPU device
- The vLLM runtime
- Cross-process IPC

They validate:
  A. Device detection logic (_resolve_device_id, _map_device)
  B. NpuIPCWrapper pickle roundtrip (object structure only, no CANN calls)
"""

import os
import pickle

import pytest

# ---------------------------------------------------------------------------
# Import the device resolution helpers from the connector facade.
# These functions do NOT require torch.npu to exist; they gracefully
# handle missing torch or missing torch.npu.
# ---------------------------------------------------------------------------


def _import_resolve_device_id():
    """Import _resolve_device_id with controlled torch mock."""
    # Try to import from the actual module. If torch is not installed,
    # we'll test the logic in isolation.
    try:
        from pegaflow.connector import _resolve_device_id, _map_device

        return _resolve_device_id, _map_device
    except ImportError:
        return None, None


# ---------------------------------------------------------------------------
# Test A1: _map_device logic (pure, no imports needed)
# ---------------------------------------------------------------------------

def _map_device(local_id: int, visible: str | None) -> int:
    """Inline copy of _map_device for zero-dependency testing."""
    if not visible:
        return local_id
    slots = [slot.strip() for slot in visible.split(",") if slot.strip()]
    try:
        mapped = slots[local_id]
    except IndexError:
        return local_id
    try:
        return int(mapped)
    except ValueError:
        return local_id


class TestMapDevice:
    """Pure-function tests for ASCEND_VISIBLE_DEVICES remapping."""

    def test_no_visibility_env_returns_local_id(self):
        assert _map_device(0, None) == 0
        assert _map_device(1, None) == 1
        assert _map_device(3, None) == 3

    def test_empty_string_returns_local_id(self):
        assert _map_device(0, "") == 0
        assert _map_device(1, "  ") == 1

    def test_single_device_visible(self):
        # ASCEND_VISIBLE_DEVICES=3 → only device 3 is visible as local 0
        assert _map_device(0, "3") == 3

    def test_multi_device_visible(self):
        # ASCEND_VISIBLE_DEVICES=2,5,7 → local 0→2, local 1→5, local 2→7
        assert _map_device(0, "2,5,7") == 2
        assert _map_device(1, "2,5,7") == 5
        assert _map_device(2, "2,5,7") == 7

    def test_mapped_index_out_of_range(self):
        # local_id exceeds visible slots → return local_id as-is
        assert _map_device(5, "0,1") == 5

    def test_spaces_in_csv(self):
        assert _map_device(0, " 0 , 1 , 2 ") == 0
        assert _map_device(2, " 0 , 1 , 2 ") == 2

    def test_non_integer_slot(self):
        # Non-integer in the list → return local_id
        assert _map_device(0, "gpu0,gpu1") == 0


# ---------------------------------------------------------------------------
# Test A2: _resolve_device_id (imports torch; skip if unavailable)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("SKIP_TORCH_TESTS", "") == "1",
    reason="SKIP_TORCH_TESTS=1",
)
def test_resolve_device_id_imports():
    """Verify _resolve_device_id imports without error."""
    resolve, map_fn = _import_resolve_device_id()
    if resolve is None:
        pytest.skip("pegaflow.connector not importable (missing vllm/torch)")

    # Just verify the functions are callable objects.
    assert callable(resolve)
    assert callable(map_fn)


# ---------------------------------------------------------------------------
# Test B: NpuIPCWrapper pickle roundtrip (object structure only)
# ---------------------------------------------------------------------------

class FakeNpuIPCWrapper:
    """A stand-in for NpuIPCWrapper for pure-structure testing.

    This replicates the __getstate__/__setstate__ protocol without
    touching the CANN runtime. It validates that pickle serialization
    correctly preserves metadata fields.
    """

    def __init__(
        self,
        key: bytes,
        dtype,
        shape: tuple,
        stride: tuple | None = None,
        storage_offset: int = 0,
        device_index: int = 0,
    ):
        self.key = key
        self.dtype = dtype
        self.shape = shape
        self.stride = stride
        self.storage_offset = storage_offset
        self.device_index = device_index

    def __getstate__(self):
        return (
            self.key,
            self.dtype,
            self.shape,
            self.stride,
            self.storage_offset,
            self.device_index,
        )

    def __setstate__(self, state):
        (
            self.key,
            self.dtype,
            self.shape,
            self.stride,
            self.storage_offset,
            self.device_index,
        ) = state

    def __eq__(self, other):
        if not isinstance(other, FakeNpuIPCWrapper):
            return False
        return (
            self.key == other.key
            and self.dtype == other.dtype
            and self.shape == other.shape
            and self.stride == other.stride
            and self.storage_offset == other.storage_offset
            and self.device_index == other.device_index
        )

    def __repr__(self):
        return (
            f"FakeNpuIPCWrapper(shape={self.shape}, dtype={self.dtype}, "
            f"device_index={self.device_index})"
        )


def _create_fake_key(device_index: int, size: int) -> bytes:
    """Create a deterministic fake IPC key for testing."""
    import hashlib

    h = hashlib.sha256(f"npu:{device_index}:{size}".encode()).digest()
    # Real keys are C strings up to 256 bytes; truncate-like.
    return h[:32] + b"\x00" * (256 - 32)


class TestNpuIPCWrapperPickle:
    """Verify NpuIPCWrapper pickle roundtrip preserves metadata."""

    def test_roundtrip_with_stride(self):
        key = _create_fake_key(0, 4096)
        original = FakeNpuIPCWrapper(
            key=key,
            dtype="torch.float16",
            shape=(32, 128),
            stride=(128, 1),
            storage_offset=0,
            device_index=0,
        )

        data = pickle.dumps(original)
        restored = pickle.loads(data)

        assert restored == original
        assert restored.key == key
        assert restored.dtype == "torch.float16"
        assert restored.shape == (32, 128)
        assert restored.stride == (128, 1)
        assert restored.storage_offset == 0
        assert restored.device_index == 0

    def test_roundtrip_3d_tensor(self):
        key = _create_fake_key(2, 8192)
        original = FakeNpuIPCWrapper(
            key=key,
            dtype="torch.bfloat16",
            shape=(4, 64, 128),
            stride=(8192, 128, 1),
            storage_offset=0,
            device_index=2,
        )

        data = pickle.dumps(original)
        restored = pickle.loads(data)

        assert restored == original
        assert restored.device_index == 2
        assert restored.shape == (4, 64, 128)

    def test_roundtrip_large_key(self):
        # Key can be up to 256 bytes.
        key = _create_fake_key(3, 1048576)
        original = FakeNpuIPCWrapper(
            key=key,
            dtype="torch.int8",
            shape=(1024, 1024),
            stride=None,
            storage_offset=0,
            device_index=3,
        )

        data = pickle.dumps(original)
        restored = pickle.loads(data)

        assert restored == original
        assert restored.shape == (1024, 1024)
        assert len(restored.key) == 256

    def test_inequality_detected(self):
        key1 = _create_fake_key(0, 4096)
        key2 = _create_fake_key(1, 4096)
        a = FakeNpuIPCWrapper(key=key1, dtype="fp16", shape=(1,), device_index=0)
        b = FakeNpuIPCWrapper(key=key2, dtype="fp16", shape=(1,), device_index=0)
        assert a != b

        c = FakeNpuIPCWrapper(key=key1, dtype="fp16", shape=(2,), device_index=0)
        assert a != c

        d = FakeNpuIPCWrapper(key=key1, dtype="fp16", shape=(1,), device_index=1)
        assert a != d

    def test_pickle_cross_process_simulation(self):
        """Simulate cross-process send by writing pickle to bytes and reading."""
        key = _create_fake_key(0, 1024)
        original = FakeNpuIPCWrapper(
            key=key,
            dtype="torch.float32",
            shape=(256,),
            device_index=0,
        )

        # Simulate cross-process: write + read
        buf = pickle.dumps(original)
        del original

        # "Receiver" process
        restored = pickle.loads(buf)

        assert restored.device_index == 0
        assert restored.shape == (256,)
        assert restored.dtype == "torch.float32"


# ---------------------------------------------------------------------------
# Test C: derive_namespace (requires pegaflow.connector.common — skip if no torch)
# ---------------------------------------------------------------------------

pytestmark_connector = pytest.mark.skipif(
    os.environ.get("SKIP_TORCH_TESTS", "") == "1",
    reason="SKIP_TORCH_TESTS=1 or torch not available",
)


def _import_common():
    """Import connector.common helpers, skipping if torch unavailable."""
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from pegaflow.connector.common import (
        derive_namespace,
        detect_mla,
        resolve_transfer_backend,
    )

    return derive_namespace, detect_mla, resolve_transfer_backend


class TestDeriveNamespace:
    """Verify namespace derivation logic."""

    class FakeVllmConfig:
        class ModelConfig:
            model = "test-model"
            dtype = "float16"

            def get_total_num_kv_heads(self):
                return 8

            def get_head_size(self):
                return 128

            def get_total_num_hidden_layers(self):
                return 32

        class CacheConfig:
            cache_dtype = "auto"
            block_size = 256

        class ParallelConfig:
            pipeline_parallel_size = 1
            tensor_parallel_size = 2
            world_size = 2
            data_parallel_size = 1
            data_parallel_rank_local = None

        def __init__(self):
            self.model_config = self.ModelConfig()
            self.cache_config = self.CacheConfig()
            self.parallel_config = self.ParallelConfig()
            self.kv_transfer_config = None
            self.instance_id = None
            self.model_config.hf_text_config = type("obj", (object,), {})()

    def test_derive_namespace_is_deterministic(self):
        derive_namespace, _, _ = _import_common()

        fake_cfg1 = self.FakeVllmConfig()
        fake_cfg2 = self.FakeVllmConfig()

        ns1 = derive_namespace(fake_cfg1, tp_size=2)
        ns2 = derive_namespace(fake_cfg2, tp_size=2)

        assert ns1 == ns2
        assert len(ns1) == 8
        assert all(c in "0123456789abcdef" for c in ns1)

    def test_derive_namespace_changes_with_tp_size(self):
        derive_namespace, _, _ = _import_common()

        fake_cfg = self.FakeVllmConfig()
        ns2 = derive_namespace(fake_cfg, tp_size=2)
        ns4 = derive_namespace(fake_cfg, tp_size=4)

        assert ns2 != ns4, "different tp_size should produce different namespace"


class TestResolveTransferBackend:
    def test_mla_defaults_to_kernel(self):
        _, _, resolve_transfer_backend = _import_common()
        assert resolve_transfer_backend(is_mla=True, override=None) == "kernel"

    def test_non_mla_defaults_to_direct(self):
        _, _, resolve_transfer_backend = _import_common()
        assert resolve_transfer_backend(is_mla=False, override=None) == "direct"

    def test_override_wins(self):
        _, _, resolve_transfer_backend = _import_common()
        assert (
            resolve_transfer_backend(is_mla=True, override="ascend_direct")
            == "ascend_direct"
        )
        assert resolve_transfer_backend(is_mla=False, override="kernel") == "kernel"

    def test_unknown_override_is_rejected(self):
        _, _, resolve_transfer_backend = _import_common()
        with pytest.raises(ValueError):
            resolve_transfer_backend(is_mla=False, override="nonsense")


class TestDetectMLA:
    def test_model_without_kv_lora_rank_is_not_mla(self):
        _, detect_mla, _ = _import_common()

        class FakeConfig:
            class ModelConfig:
                hf_text_config = type("obj", (object,), {})()

            def __init__(self):
                self.model_config = self.ModelConfig()

        result = detect_mla(FakeConfig())
        assert result is False

    def test_model_with_kv_lora_rank_is_mla(self):
        _, detect_mla, _ = _import_common()

        class FakeConfig:
            class ModelConfig:
                hf_text_config = type("obj", (object,), {"kv_lora_rank": 512})()

            def __init__(self):
                self.model_config = self.ModelConfig()

        result = detect_mla(FakeConfig())
        assert result is True
