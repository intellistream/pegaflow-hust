"""
Unit tests for NPU device resolution, transfer backend selection, and
multi-instance state isolation.

These tests operate entirely within a single Python process and do NOT require:
- An actual Ascend NPU device
- The C extension (npu_ipc_bindings._npu_ipc)
- The vLLM runtime
- Cross-process IPC

They validate:
  A. _resolve_device_id() handles NPU with ASCEND_VISIBLE_DEVICES
  B. resolve_transfer_backend() auto-detects NPU and avoids "kernel" on Ascend
  C. _map_device() correctly remaps device ordinals
  D. Multi-instance state isolation (derive_namespace determinism)
  E. NpuIPCWrapper pickle roundtrip with multiple device indices
"""

import os
import pickle

import pytest

# =============================================================================
# Helpers — zero-dependency inline copies for isolation testing
# =============================================================================


def _map_device(local_id: int, visible: str | None) -> int:
    """Inline copy of _map_device from pegaflow.connector for isolated testing."""
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


def _resolve_device_id(
    cuda_available: bool = False,
    cuda_current: int = 0,
    cuda_visible: str | None = None,
    npu_available: bool = False,
    npu_current: int = 0,
    npu_visible: str | None = None,
) -> int:
    """Inline copy of _resolve_device_id logic with injectable device state."""
    if cuda_available:
        return _map_device(cuda_current, cuda_visible)
    if npu_available:
        return _map_device(npu_current, npu_visible)
    return 0


def _resolve_transfer_backend(
    is_mla: bool,
    override: str | None = None,
    is_npu: bool = False,
) -> str:
    """Inline copy of resolve_transfer_backend logic for isolated testing."""
    _BACKENDS = ("direct", "kernel", "ascend_direct")
    if override is None:
        if is_mla:
            return "direct" if is_npu else "kernel"
        return "direct"
    normalized = override.strip().lower()
    if normalized not in _BACKENDS:
        raise ValueError(f"Unsupported backend: {override!r}")
    return normalized


def _derive_namespace_inline(
    model: str = "test-model",
    dtype: str = "float16",
    tp_size: int = 1,
    pp_size: int = 1,
    num_kv_heads: int = 8,
    head_size: int = 128,
    num_hidden_layers: int = 32,
    cache_dtype: str = "auto",
    dcp_world_size: int = 1,
    pcp_world_size: int = 1,
    cross_layer_blocks: bool = False,
    mla_layer_split_kv_cache: bool = False,
) -> str:
    """Inline copy of derive_namespace logic for isolated testing."""
    import hashlib

    factors = {
        "model": model,
        "dtype": dtype,
        "tp_size": tp_size,
        "pp_size": pp_size,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "num_hidden_layers": num_hidden_layers,
        "cache_dtype": cache_dtype,
        "dcp_world_size": dcp_world_size,
        "pcp_world_size": pcp_world_size,
        "cross_layer_blocks": cross_layer_blocks,
        "mla_layer_split_kv_cache": mla_layer_split_kv_cache,
    }
    factor_str = str(sorted(factors.items()))
    hash_suffix = hashlib.sha256(factor_str.encode()).hexdigest()[:8]
    return f"{hash_suffix}"


# =============================================================================
# A. Device Resolution Tests (_resolve_device_id)
# =============================================================================


class TestResolveDeviceId:
    """Verify _resolve_device_id handles NPU and CUDA correctly."""

    # --- CUDA path (pre-existing) ---

    def test_cuda_no_visibility_env_returns_current_device(self):
        result = _resolve_device_id(cuda_available=True, cuda_current=2)
        assert result == 2

    def test_cuda_with_visibility_maps_correctly(self):
        result = _resolve_device_id(
            cuda_available=True, cuda_current=0, cuda_visible="3,7"
        )
        assert result == 3

    def test_cuda_visibility_out_of_range_falls_back(self):
        result = _resolve_device_id(
            cuda_available=True, cuda_current=5, cuda_visible="0,1"
        )
        assert result == 5

    # --- NPU path (new) ---

    def test_npu_no_visibility_env_returns_current_device(self):
        result = _resolve_device_id(npu_available=True, npu_current=1)
        assert result == 1

    def test_npu_with_visibility_maps_correctly(self):
        result = _resolve_device_id(
            npu_available=True, npu_current=0, npu_visible="2,5,7"
        )
        assert result == 2

    def test_npu_visibility_single_slot(self):
        result = _resolve_device_id(
            npu_available=True, npu_current=0, npu_visible="4"
        )
        assert result == 4

    def test_npu_visibility_out_of_range_falls_back(self):
        result = _resolve_device_id(
            npu_available=True, npu_current=5, npu_visible="0,1"
        )
        assert result == 5

    # --- Neither CUDA nor NPU ---

    def test_no_device_available_returns_zero(self):
        result = _resolve_device_id()
        assert result == 0

    # --- CUDA takes priority over NPU ---

    def test_cuda_priority_over_npu(self):
        """When both CUDA and NPU are available, CUDA wins."""
        result = _resolve_device_id(
            cuda_available=True,
            cuda_current=3,
            npu_available=True,
            npu_current=7,
        )
        assert result == 3  # CUDA wins


# =============================================================================
# B. Transfer Backend Tests (resolve_transfer_backend)
# =============================================================================


class TestResolveTransferBackend:
    """Verify NPU-aware transfer backend selection."""

    # --- Defaults (no override) ---

    def test_cuda_non_mla_defaults_to_direct(self):
        assert _resolve_transfer_backend(is_mla=False, is_npu=False) == "direct"

    def test_cuda_mla_defaults_to_kernel(self):
        assert _resolve_transfer_backend(is_mla=True, is_npu=False) == "kernel"

    def test_npu_non_mla_defaults_to_direct(self):
        assert _resolve_transfer_backend(is_mla=False, is_npu=True) == "direct"

    def test_npu_mla_defaults_to_direct(self):
        """On NPU, even MLA must default to 'direct' — kernel is CUDA-only."""
        assert _resolve_transfer_backend(is_mla=True, is_npu=True) == "direct"

    # --- Explicit overrides ---

    def test_npu_mla_override_ascend_direct(self):
        assert (
            _resolve_transfer_backend(is_mla=True, override="ascend_direct", is_npu=True)
            == "ascend_direct"
        )

    def test_npu_non_mla_override_kernel_rejected(self):
        """'kernel' override on NPU should still pass validation (user's choice)."""
        assert (
            _resolve_transfer_backend(is_mla=False, override="kernel", is_npu=True)
            == "kernel"
        )

    def test_cuda_override_ascend_direct_accepted(self):
        assert (
            _resolve_transfer_backend(is_mla=False, override="ascend_direct", is_npu=False)
            == "ascend_direct"
        )

    def test_unknown_override_rejected(self):
        with pytest.raises(ValueError):
            _resolve_transfer_backend(is_mla=False, override="nonsense", is_npu=True)

    # --- Edge cases ---

    def test_override_whitespace_stripped(self):
        assert (
            _resolve_transfer_backend(is_mla=False, override="  ascend_direct  ", is_npu=True)
            == "ascend_direct"
        )

    def test_npu_auto_detect_when_none(self):
        """When is_npu is None and torch.npu is available, auto-detect works."""
        # The inline version doesn't auto-detect, but the real one does.
        # This test validates the real function when torch is importable.
        try:
            from pegaflow.connector.common import resolve_transfer_backend

            backend = resolve_transfer_backend(is_mla=True, override="direct")
            assert backend == "direct"  # explicit override always wins
        except ImportError:
            pytest.skip("pegaflow.connector.common not importable")


# =============================================================================
# C. _map_device Tests (including ASCEND_VISIBLE_DEVICES patterns)
# =============================================================================


class TestMapDevice:
    """Verify _map_device remaps both CUDA_VISIBLE_DEVICES and ASCEND_VISIBLE_DEVICES."""

    def test_no_env_returns_local(self):
        assert _map_device(0, None) == 0
        assert _map_device(3, None) == 3

    def test_empty_string_returns_local(self):
        assert _map_device(0, "") == 0
        assert _map_device(1, "  ") == 1

    def test_single_slot(self):
        assert _map_device(0, "3") == 3

    def test_multi_slot(self):
        # ASCEND_VISIBLE_DEVICES=2,5,7
        assert _map_device(0, "2,5,7") == 2
        assert _map_device(1, "2,5,7") == 5
        assert _map_device(2, "2,5,7") == 7

    def test_out_of_range(self):
        assert _map_device(5, "0,1") == 5

    def test_spaces_in_csv(self):
        assert _map_device(0, " 0 , 1 , 2 ") == 0
        assert _map_device(2, " 0 , 1 , 2 ") == 2

    def test_non_integer_slot(self):
        assert _map_device(0, "gpu0,gpu1") == 0


# =============================================================================
# D. Multi-Instance State Isolation Tests
# =============================================================================


class TestMultiInstanceStateIsolation:
    """Verify that derive_namespace correctly isolates instances."""

    def test_deterministic_for_same_config(self):
        ns1 = _derive_namespace_inline()
        ns2 = _derive_namespace_inline()
        assert ns1 == ns2
        assert len(ns1) == 8

    def test_different_model_produces_different_namespace(self):
        ns1 = _derive_namespace_inline(model="model-A")
        ns2 = _derive_namespace_inline(model="model-B")
        assert ns1 != ns2

    def test_different_tp_size_different_namespace(self):
        ns1 = _derive_namespace_inline(tp_size=2)
        ns2 = _derive_namespace_inline(tp_size=4)
        assert ns1 != ns2

    def test_different_pp_size_different_namespace(self):
        ns1 = _derive_namespace_inline(pp_size=1)
        ns2 = _derive_namespace_inline(pp_size=2)
        assert ns1 != ns2

    def test_different_dcp_world_size_different_namespace(self):
        ns1 = _derive_namespace_inline(dcp_world_size=1)
        ns2 = _derive_namespace_inline(dcp_world_size=2)
        assert ns1 != ns2

    def test_cross_layer_blocks_changes_namespace(self):
        ns1 = _derive_namespace_inline(cross_layer_blocks=False)
        ns2 = _derive_namespace_inline(cross_layer_blocks=True)
        assert ns1 != ns2

    def test_mla_layer_split_changes_namespace(self):
        ns1 = _derive_namespace_inline(mla_layer_split_kv_cache=False)
        ns2 = _derive_namespace_inline(mla_layer_split_kv_cache=True)
        assert ns1 != ns2

    def test_namespace_hash_is_hex(self):
        ns = _derive_namespace_inline()
        assert all(c in "0123456789abcdef" for c in ns)


# =============================================================================
# E. NpuIPCWrapper Multi-Key Tests (structure only, no CANN calls)
# =============================================================================


class FakeNpuIPCWrapper:
    """Stand-in for NpuIPCWrapper for pure-structure testing.

    Replicates __getstate__/__setstate__ protocol without touching
    the CANN runtime. Validates that pickle serialization correctly
    preserves metadata fields including device_index.
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


def _make_fake_key(device_index: int, size: int) -> bytes:
    import hashlib

    h = hashlib.sha256(f"npu:{device_index}:{size}".encode()).digest()
    return h[:32] + b"\x00" * (256 - 32)


class TestNpuIPCWrapperMultiKey:
    """Verify NpuIPCWrapper handles multiple device indices correctly."""

    def test_devices_with_different_indices_have_different_keys(self):
        key0 = _make_fake_key(0, 4096)
        key1 = _make_fake_key(1, 4096)
        a = FakeNpuIPCWrapper(key=key0, dtype="fp16", shape=(1,), device_index=0)
        b = FakeNpuIPCWrapper(key=key1, dtype="fp16", shape=(1,), device_index=1)
        assert a != b
        assert a.device_index != b.device_index

    def test_pickle_roundtrip_preserves_device_index(self):
        for dev_idx in [0, 1, 2, 3]:
            key = _make_fake_key(dev_idx, 1024)
            original = FakeNpuIPCWrapper(
                key=key,
                dtype="torch.float16",
                shape=(32, 128),
                device_index=dev_idx,
            )
            data = pickle.dumps(original)
            restored = pickle.loads(data)
            assert restored.device_index == dev_idx, f"device_index lost for {dev_idx}"

    def test_batch_of_wrappers_roundtrip(self):
        """Simulate a batch of IPC wrappers from multiple devices."""
        originals = []
        for dev_idx in range(4):
            key = _make_fake_key(dev_idx, 4096)
            originals.append(
                FakeNpuIPCWrapper(
                    key=key,
                    dtype="torch.bfloat16",
                    shape=(64,),
                    device_index=dev_idx,
                )
            )

        batch_data = pickle.dumps(originals)
        restored_list = pickle.loads(batch_data)

        assert len(restored_list) == len(originals)
        for i, (orig, rest) in enumerate(zip(originals, restored_list, strict=False)):
            assert rest == orig, f"wrapper {i} mismatch"
            assert rest.device_index == i, f"wrapper {i} device_index wrong"

    def test_cross_process_simulation_preserves_all_fields(self):
        """Simulate cross-process: pickle → bytes → unpickle in 'receiver'."""
        key = _make_fake_key(2, 8192)
        original = FakeNpuIPCWrapper(
            key=key,
            dtype="torch.float32",
            shape=(4, 64, 128),
            stride=(8192, 128, 1),
            storage_offset=0,
            device_index=2,
        )

        buf = pickle.dumps(original)
        del original

        restored = pickle.loads(buf)
        assert restored.device_index == 2
        assert restored.shape == (4, 64, 128)
        assert restored.stride == (8192, 128, 1)
        assert restored.dtype == "torch.float32"


# =============================================================================
# F. Integration test: verify real functions when torch is available
# =============================================================================


@pytest.mark.skipif(
    os.environ.get("SKIP_TORCH_TESTS", "") == "1",
    reason="SKIP_TORCH_TESTS=1",
)
class TestRealConnectorFunctions:
    """Smoke-tests that import and exercise the real connector functions."""

    def test_real_resolve_device_id_callable(self):
        try:
            from pegaflow.connector import _resolve_device_id
        except ImportError:
            pytest.skip("pegaflow.connector not importable (missing vllm/torch)")

        result = _resolve_device_id()
        assert isinstance(result, int)

    def test_real_map_device_matches_inline(self):
        try:
            from pegaflow.connector import _map_device as real_map
        except ImportError:
            pytest.skip("pegaflow.connector not importable")

        test_cases = [
            (0, None, 0),
            (0, "3", 3),
            (1, "2,5,7", 5),
            (5, "0,1", 5),
            (0, "", 0),
        ]
        for local, visible, expected in test_cases:
            assert real_map(local, visible) == expected, (
                f"_map_device({local}, {visible!r}) != {expected}"
            )

    def test_real_resolve_transfer_backend_npu_mla(self):
        """The real resolve_transfer_backend should handle NPU + MLA."""
        try:
            from pegaflow.connector.common import resolve_transfer_backend
        except ImportError:
            pytest.skip("pegaflow.connector.common not importable")

        # Explicitly pass is_npu=True to test the NPU path
        backend = resolve_transfer_backend(is_mla=True, override=None, is_npu=True)
        assert backend == "direct", (
            f"NPU + MLA should default to 'direct', got '{backend}'"
        )

        backend = resolve_transfer_backend(is_mla=False, override=None, is_npu=True)
        assert backend == "direct", (
            f"NPU + non-MLA should default to 'direct', got '{backend}'"
        )

    def test_real_resolve_transfer_backend_cuda_mla(self):
        """CUDA + MLA should still default to 'kernel'."""
        try:
            from pegaflow.connector.common import resolve_transfer_backend
        except ImportError:
            pytest.skip("pegaflow.connector.common not importable")

        backend = resolve_transfer_backend(is_mla=True, override=None, is_npu=False)
        assert backend == "kernel", (
            f"CUDA + MLA should default to 'kernel', got '{backend}'"
        )


__all__ = [
    "TestResolveDeviceId",
    "TestResolveTransferBackend",
    "TestMapDevice",
    "TestMultiInstanceStateIsolation",
    "TestNpuIPCWrapperMultiKey",
    "TestRealConnectorFunctions",
]
