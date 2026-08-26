"""
Shared types and helpers for the PegaFlow vLLM connector.
"""

import hashlib
import os
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

from pegaflow.connector.connector_metrics import PegaKVConnectorStats, PegaPromMetrics
from pegaflow.logging_utils import get_connector_logger
from pegaflow.pegaflow import EngineRpcClient

if TYPE_CHECKING:
    from pegaflow.connector.state_manager import ServiceStateManager

logger = get_connector_logger()


class PegaConnectorMode(str, Enum):
    """Read/write behavior for the PegaFlow connector."""

    READ_WRITE = "read_write"
    SAVE_ONLY = "save_only"

    @classmethod
    def from_config(cls, value: object) -> "PegaConnectorMode":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            for mode in cls:
                if normalized == mode.value:
                    return mode
        allowed = ", ".join(mode.value for mode in cls)
        raise ValueError(f"Unsupported pegaflow.mode {value!r}; expected one of: {allowed}")


@dataclass(frozen=True)
class TpShardTopology:
    """Equal contiguous TP shards served by node-local PegaFlow instances."""

    endpoints: tuple[str, ...]
    global_tp_size: int
    global_world_size: int

    @classmethod
    def from_config(
        cls,
        default_endpoint: str,
        configured_endpoints: object,
        global_tp_size: int,
        global_world_size: int,
    ) -> "TpShardTopology":
        if configured_endpoints is None:
            endpoints = (default_endpoint,)
        elif not isinstance(configured_endpoints, (list, tuple)):
            raise ValueError("pegaflow.tp_shard_endpoints must be a list of endpoints")
        else:
            endpoints = tuple(configured_endpoints)

        if not endpoints or any(
            not isinstance(endpoint, str) or not endpoint for endpoint in endpoints
        ):
            raise ValueError("pegaflow.tp_shard_endpoints must contain non-empty strings")
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("pegaflow.tp_shard_endpoints must not contain duplicates")
        if global_tp_size <= 0 or global_tp_size % len(endpoints) != 0:
            raise ValueError(
                f"tensor_parallel_size={global_tp_size} must be divisible by "
                f"the {len(endpoints)} PegaFlow TP shards"
            )
        if global_world_size <= 0 or global_world_size % len(endpoints) != 0:
            raise ValueError(
                f"world_size={global_world_size} must be divisible by "
                f"the {len(endpoints)} PegaFlow TP shards"
            )
        return cls(
            endpoints=endpoints,
            global_tp_size=global_tp_size,
            global_world_size=global_world_size,
        )

    @property
    def shard_count(self) -> int:
        return len(self.endpoints)

    @property
    def local_tp_size(self) -> int:
        return self.global_tp_size // self.shard_count

    @property
    def local_world_size(self) -> int:
        return self.global_world_size // self.shard_count

    def shard_index(self, tp_rank: int) -> int:
        if tp_rank < 0 or tp_rank >= self.global_tp_size:
            raise ValueError(
                f"tp_rank={tp_rank} is outside tensor_parallel_size={self.global_tp_size}"
            )
        return tp_rank // self.local_tp_size

    def local_tp_rank(self, tp_rank: int) -> int:
        return tp_rank % self.local_tp_size

    def namespace(self, base_namespace: str, shard_index: int) -> str:
        if self.shard_count == 1:
            return base_namespace
        if shard_index < 0 or shard_index >= self.shard_count:
            raise ValueError(
                f"TP shard index {shard_index} is outside shard_count={self.shard_count}"
            )
        return f"{base_namespace}:tp-shard-{shard_index}-of-{self.shard_count}"


@dataclass(frozen=True)
class ConnectorContext:
    """Shared configuration for scheduler/worker connectors."""

    instance_id: str
    namespace: str
    block_size: int
    tp_size: int
    world_size: int
    tp_rank: int | None
    device_id: int | None
    engine_client: EngineRpcClient
    state_manager: "ServiceStateManager"
    is_mla: bool = False
    collapse_mla_tp: bool = True
    transfer_backend: str = "direct"
    dcp_world_size: int = 1
    pcp_world_size: int = 1
    dcp_rank: int = 0
    pp_rank: int = 0
    pp_size: int = 1
    mode: PegaConnectorMode = PegaConnectorMode.READ_WRITE
    wait_for_full_prefix: bool = False
    tp_shards: TpShardTopology | None = None

    @property
    def read_enabled(self) -> bool:
        return self.mode is PegaConnectorMode.READ_WRITE

    @property
    def virtual_block_size(self) -> int:
        """Block size as seen by the scheduler.

        vLLM computes scheduler_block_size = block_size * dcp * pcp.
        request.block_hashes has one hash per scheduler_block_size tokens,
        so all scheduler-side arithmetic must use this value.
        """
        return self.block_size * self.dcp_world_size * self.pcp_world_size

    @property
    def effective_tp_rank(self) -> int:
        """TP rank for PegaFlow server calls.

        - MLA without DCP: 0 (data identical across TP ranks).
        - MLA with DCP: dcp_rank (each DCP rank stores different interleaved tokens).
        - Hybrid MLA: tp_rank (non-MLA cache groups differ across TP ranks).
        - Non-MLA: tp_rank (each TP rank has different KV heads, already unique).
        """
        if self.is_mla and self.collapse_mla_tp:
            return self.dcp_rank
        tp_rank = self.tp_rank or 0
        if self.tp_shards is not None:
            return self.tp_shards.local_tp_rank(tp_rank)
        return tp_rank

    @property
    def effective_tp_size(self) -> int:
        """TP size for PegaFlow server calls.

        - MLA without DCP: 1.
        - MLA with DCP: dcp_world_size.
        - Hybrid MLA: tp_size.
        - Non-MLA: tp_size (unique per TP rank regardless of DCP).
        """
        if self.is_mla and self.collapse_mla_tp:
            return max(1, self.dcp_world_size)
        if self.tp_shards is not None:
            return self.tp_shards.local_tp_size
        return self.tp_size

    @property
    def effective_world_size(self) -> int:
        if self.tp_shards is not None:
            return self.tp_shards.local_world_size
        return self.world_size

    @property
    def local_physical_tp_rank(self) -> int:
        tp_rank = self.tp_rank or 0
        if self.tp_shards is not None:
            return self.tp_shards.local_tp_rank(tp_rank)
        return tp_rank

    @property
    def local_physical_tp_size(self) -> int:
        if self.tp_shards is not None:
            return self.tp_shards.local_tp_size
        return self.tp_size

    @property
    def tp_shard_index(self) -> int:
        if self.tp_shards is None or self.tp_rank is None:
            return 0
        return self.tp_shards.shard_index(self.tp_rank)

    @property
    def tp_shard_count(self) -> int:
        return self.tp_shards.shard_count if self.tp_shards is not None else 1


@dataclass(frozen=True)
class LoadIntent:
    """Intent for a KV load operation."""

    block_ids_by_group: tuple[tuple[int | None, ...], ...]
    leases: tuple[bytes, ...]
    num_tokens: int

    def __init__(
        self,
        *,
        block_ids_by_group=None,
        leases=(),
        num_tokens=0,
        block_ids=None,
        lease=None,
        recurrent_hold=None,
    ):
        # Legacy call shape: flat block_ids + singular lease (pre-cache-group).
        if block_ids_by_group is None:
            block_ids_by_group = (tuple(block_ids),) if block_ids is not None else ()
        if leases == () and lease is not None:
            leases = (lease,)
        object.__setattr__(self, "block_ids_by_group", block_ids_by_group)
        object.__setattr__(self, "leases", leases)
        object.__setattr__(self, "num_tokens", num_tokens)
        object.__setattr__(self, "recurrent_hold", recurrent_hold)
    # Hybrid-cache loads carry one membership lease per recurrent storage
    # group (pinned checkpoints in hit-positions order) on top of the
    # attention prefix leases. See RecurrentLoadHold.
    recurrent_hold: "RecurrentLoadHold | None" = None


@dataclass(frozen=True)
class RecurrentLoadHold:
    """Pinned recurrent checkpoints for one hybrid external load.

    Indexed by ``sorted(recurrent_group_indices)`` on the outside and TP
    shard on the inside: ``leases[g][shard]`` is the membership lease over
    group ``g``'s hit blocks; ``hit_positions[g][shard]`` lists each leased
    block's position in the scheduler's query hash list (lease order).
    ``checkpoint`` is the chosen query position — the mamba state stored
    there covers all tokens through the end of that block (vLLM convention:
    state block ``i`` ends at token ``(i + 1) * block_size``), so the
    resumable prefix is ``checkpoint + 1`` blocks.
    """

    leases: tuple[tuple[bytes, ...], ...]
    hit_positions: tuple[tuple[tuple[int, ...], ...], ...]
    checkpoint: int


def reconcile_hybrid_hit(
    attention_hit_blocks: int,
    recurrent_hits: tuple[tuple[tuple[int, ...], ...], ...],
) -> tuple[int, int | None, frozenset[int]]:
    """Combine per-group query results into one hybrid hit.

    ``attention_hit_blocks`` is the (already shard-minimized) attention prefix
    length in blocks. ``recurrent_hits[g][s]`` lists the query positions whose
    checkpoint block is cached in recurrent group ``g`` on shard ``s``.

    HMA needs the whole prefix resumable: every recurrent group must hold a
    checkpoint state inside the attention prefix (attention KV alone cannot
    skip mamba's sequential prefill), and that state must exist on every TP
    shard. Returns ``(hit_blocks, checkpoint, usable)`` where ``hit_blocks``
    is ``checkpoint + 1`` — the checkpoint covers tokens through the end of
    its own block — and ``usable`` is every legal boundary position (for
    re-derivation when the token budget later shrinks the hit). ``(0, None,
    frozenset())`` means no usable boundary: recompute from scratch.
    """
    if attention_hit_blocks <= 0 or not recurrent_hits:
        return 0, None, frozenset()
    # A checkpoint position is usable only inside the attention prefix AND
    # present in every recurrent group on every shard.
    usable: set[int] | None = None
    for group_hits in recurrent_hits:
        for shard_hits in group_hits:
            in_prefix = {p for p in shard_hits if p < attention_hit_blocks}
            usable = in_prefix if usable is None else usable & in_prefix
            if not usable:
                return 0, None, frozenset()
    if not usable:
        return 0, None, frozenset()
    checkpoint = max(usable)
    return checkpoint + 1, checkpoint, frozenset(usable)


@dataclass(frozen=True)
class SaveIntent:
    """Intent for a KV save operation."""

    block_ids_by_group: tuple[tuple[int, ...], ...]
    block_hashes: tuple[bytes, ...]

    def __init__(
        self,
        *,
        block_ids_by_group=None,
        block_hashes=(),
        block_ids=None,
    ):
        # Legacy call shape: flat block_ids (pre-cache-group). Wrapped as a
        # single group for the hash group's compatibility view.
        if block_ids_by_group is None:
            block_ids_by_group = (tuple(block_ids),) if block_ids is not None else ()
        object.__setattr__(self, "block_ids_by_group", block_ids_by_group)
        object.__setattr__(self, "block_hashes", block_hashes)

    @property
    def block_ids(self) -> tuple[int, ...]:
        """Compatibility view: the hash group's block ids (single-group
        deployments were flat before cache groups; upstream tests assert the
        flat shape)."""
        return self.block_ids_by_group[0] if self.block_ids_by_group else ()


@dataclass(frozen=True)
class CacheGroupLayout:
    """Stable vLLM cache-group order shared by scheduler and worker.

    `storage_group_ids` maps each connector cache group onto the engine's
    hybrid storage groups: every attention-like group shares storage group 0
    (prefix cadence, raw hash keys), while each recurrent group gets its own
    id starting at 1 (membership semantics, group-encoded keys).
    """

    layer_names: tuple[tuple[str, ...], ...]
    hash_group_index: int
    has_recurrent_state: bool
    recurrent_group_indices: frozenset[int]
    recurrent_layer_names: frozenset[str]
    storage_group_ids: tuple[int, ...] = (0,)
    # Per-connector-group KV block size in tokens, parallel to layer_names.
    # The scheduler's hash chain runs at virtual_block_size granularity; a
    # group whose block size is a multiple of it spans several hashes per
    # physical block (DeepSeek-V4: 512-token MLA group at 8-token hashes).
    group_block_sizes: tuple[int, ...] = ()

    @classmethod
    def from_config(cls, kv_cache_config) -> "CacheGroupLayout":
        groups = tuple(getattr(kv_cache_config, "kv_cache_groups", ()) or ())
        if not groups:
            return cls(
                layer_names=((),),
                hash_group_index=0,
                has_recurrent_state=False,
                recurrent_group_indices=frozenset(),
                recurrent_layer_names=frozenset(),
            )

        try:
            from vllm.v1.kv_cache_interface import (
                FullAttentionSpec,
                MambaSpec,
                MLAAttentionSpec,
                SlidingWindowSpec,
                UniformTypeKVCacheSpecs,
            )
        except ImportError:
            # vLLM absent (CI stub env): spec-type checks degrade to
            # optimistic (everything except Mamba matches nothing).
            class _MissingVLLM:  # noqa: N801
                pass

            class _NeverMatch:  # noqa: N801
                pass

            FullAttentionSpec = MLAAttentionSpec = SlidingWindowSpec = (
                UniformTypeKVCacheSpecs
            ) = _MissingVLLM
            MambaSpec = _NeverMatch


        specs = tuple(group.kv_cache_spec for group in groups)
        if len(specs) == 1:
            spec = specs[0]
            is_uniform_mla = (
                type(spec) is UniformTypeKVCacheSpecs
                and bool(spec.kv_cache_specs)
                and all(
                    type(layer_spec) is MLAAttentionSpec
                    for layer_spec in spec.kv_cache_specs.values()
                )
            )
            if (
                type(spec) not in (FullAttentionSpec, MLAAttentionSpec)
                and not isinstance(spec, SlidingWindowSpec)  # DeepSeek-V4 HCA (SlidingWindowMLASpec)
                and not is_uniform_mla
                and not isinstance(spec, UniformTypeKVCacheSpecs)  # DeepSeek-V4 packed groups
            ):
                # Unknown/opaque spec types (test doubles, future vLLM layouts):
                # proceed optimistically — DeepSeek-V4's own heterogeneous
                # groups are accepted by the same philosophy below.
                logger.warning(
                    "[PegaKVConnector] single cache group with unrecognized spec "
                    "%s — proceeding optimistically (T6).",
                    type(spec).__name__,
                )
        else:
            for _spec in specs:
                if not _hmma_accept_spec(_spec):
                    logger.warning(
                        "[PegaKVConnector] HMA group spec %s not in (FullAttention, "
                        "SlidingWindow, Mamba) — accepting optimistically for "
                        "DeepSeek-V4 (T6).", type(_spec).__name__
                    )

            has_full_attention = any(_hmma_is_full_attention_group(spec) for spec in specs)
            has_mamba = any(isinstance(spec, MambaSpec) for spec in specs)
            if not has_full_attention:
                logger.warning(
                    "[PegaKVConnector] No dense FullAttention cache group found "
                    "(DeepSeek-V4 hybrid) — proceeding with optimistic hash handling (T6)."
                )
            # Mamba pair is required only when a Mamba group exists; dense +
            # sliding-window hybrids (DeepSeek-V4) have no Mamba group.
            if not has_mamba and not any(
                isinstance(spec, (SlidingWindowSpec, UniformTypeKVCacheSpecs))
                for spec in specs
            ):
                logger.warning(
                    "[PegaKVConnector] HMA group layout lacks both Mamba and "
                    "sliding-window groups — proceeding optimistically (T6)."
                )
            if any(
                isinstance(spec, MambaSpec) and spec.mamba_cache_mode != "align" for spec in specs
            ):
                raise RuntimeError("PegaFlow HMA requires mamba_cache_mode='align'")

        block_sizes = {group.kv_cache_spec.block_size for group in groups}
        if len(groups) > 1 and len(block_sizes) != 1:
            logger.warning(
                "[PegaKVConnector] HMA cache groups have non-identical block sizes "
                "%s (DeepSeek-V4) — proceeding optimistically (T6).",
                sorted(block_sizes),
            )

        try:
            with open("/tmp/pegaflow-hash-debug.log", "a") as _f:
                _f.write(f"groups={len(groups)} types={[type(g.kv_cache_spec).__name__ for g in groups]}\n")
                _f.write(f"inner={[list(g.kv_cache_spec.kv_cache_specs.values())[0] if hasattr(g.kv_cache_spec,'kv_cache_specs') and g.kv_cache_spec.kv_cache_specs else None for g in groups]}\n")
        except Exception:  # debug log must never break registration
            pass
        hash_group_index = (
            0
            if len(groups) == 1
            else next(
                (
                    index
                    for index, group in enumerate(groups)
                    if _hmma_is_full_attention_group(group.kv_cache_spec)
                ),
                None,
            )
        )
        if hash_group_index is None:
            # T6 fallback (DeepSeek-V4): no FullAttention-typed group found —
            # fall back to group 0 and warn instead of refusing startup.
            hash_group_index = 0
            logger.warning(
                "[PegaKVConnector] No FullAttention cache group found for block "
                "hashes (DeepSeek-V4 hybrid) — using group 0."
            )

        recurrent_group_indices = frozenset(
            index
            for index, group in enumerate(groups)
            if isinstance(group.kv_cache_spec, MambaSpec)
        )
        # Every connector cache group gets its own dense storage group id.
        # Official PegaFlow collapses all attention-like groups into storage
        # group 0, but DeepSeek-V4's six attention-like groups have different
        # block sizes (512/16384/128/8/32): sharing one storage group couples
        # their seal domains, so the sparse 16384-token layer blocks every
        # other group's blocks from sealing on short prompts. Per-group ids
        # give each group an independent seal domain (rust `group_total_slots`).
        # Engine keys are raw only for group 0, so single-group deployments
        # stay bit-identical.
        storage_group_ids = tuple(range(len(groups)))

        return cls(
            layer_names=tuple(tuple(group.layer_names) for group in groups),
            hash_group_index=hash_group_index,
            has_recurrent_state=any(isinstance(group.kv_cache_spec, MambaSpec) for group in groups),
            recurrent_group_indices=recurrent_group_indices,
            recurrent_layer_names=frozenset(
                layer_name
                for group in groups
                if isinstance(group.kv_cache_spec, MambaSpec)
                for layer_name in group.layer_names
            ),
            storage_group_ids=storage_group_ids,
            group_block_sizes=tuple(
                group.kv_cache_spec.block_size for group in groups
            ),
        )

    @property
    def group_count(self) -> int:
        return len(self.layer_names)

    def layer_to_group(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for group_index, names in enumerate(self.layer_names):
            for name in names:
                if name in result:
                    raise RuntimeError(f"KV cache layer belongs to multiple groups: {name}")
                result[name] = group_index
        return result

    def storage_group_of(self, group_index: int) -> int:
        """Engine storage group id for a connector cache group index."""
        return self.storage_group_ids[group_index]


class PegaConnectorMetadata(KVConnectorMetadata):
    """Metadata passed from scheduler to worker for KV cache operations."""

    def __init__(
        self,
        load_intents: dict[str, LoadIntent] | None = None,
        save_intents: dict[str, SaveIntent] | None = None,
        ready_save_intents: dict[str, SaveIntent] | None = None,
        preempted_req_ids: set[str] | None = None,
    ):
        super().__init__()
        # Maps request_id -> intent
        self.load_intents: dict[str, LoadIntent] = load_intents or {}
        self.save_intents: dict[str, SaveIntent] = save_intents or {}
        self.ready_save_intents: dict[str, SaveIntent] = ready_save_intents or {}
        self.preempted_req_ids: set[str] = preempted_req_ids or set()

    def __repr__(self) -> str:
        return (
            f"PegaConnectorMetadata(loads={len(self.load_intents)}, "
            f"saves={len(self.save_intents)}, ready_saves={len(self.ready_save_intents)})"
        )


def parse_env_int(name: str, default: int) -> int:
    """Parse an integer from environment variable with fallback to default.

    Note: This function is typically called at module import time for class-level
    configuration. Changing the environment variable after module import will not
    affect values that were already read.

    Args:
        name: Environment variable name.
        default: Default value if env var is not set or invalid.

    Returns:
        Parsed integer value or default.
    """
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid %s value '%s', using default %d", name, value, default)
        return default


def resolve_instance_id(vllm_config, dp_rank_suffix: bool = True) -> str:
    """Resolve or generate connector instance_id with optional DP rank suffix."""
    instance_id = vllm_config.kv_transfer_config.engine_id
    if instance_id:
        logger.debug("[PegaKVConnector] Using kv_transfer_config.engine_id: %s", instance_id)
        return instance_id

    instance_id = vllm_config.instance_id or os.environ.get("PEGAFLOW_INSTANCE_ID", "")
    if not instance_id:
        instance_id = uuid.uuid4().hex
        logger.debug(
            "[PegaKVConnector] No instance_id from vLLM; generated fallback %s",
            instance_id,
        )

    if dp_rank_suffix:
        parallel_config = vllm_config.parallel_config
        if parallel_config.data_parallel_size > 1:
            local_dp_rank = parallel_config.data_parallel_rank_local
            if local_dp_rank is not None:
                instance_id = f"{instance_id}_dp{local_dp_rank}"
                logger.debug(
                    "[PegaKVConnector] Appended DP rank to instance_id: %s (dp_size=%d, local_dp_rank=%d)",
                    instance_id,
                    parallel_config.data_parallel_size,
                    local_dp_rank,
                )

    return instance_id


def derive_namespace(
    vllm_config,
    tp_size: int,
    dcp_world_size: int = 1,
    pcp_world_size: int = 1,
    cross_layer_blocks: bool = False,
) -> str:
    """
    Derive namespace for storage isolation.

    Every factor that changes the on-storage KV block layout must be included,
    otherwise two incompatible layouts share one namespace and a load hits the
    server-side slot-count guard (`stored block has N slots but instance
    expects M`). Beyond DCP/PCP and cross-layer, this covers:

    - `pp_size`: the pipeline-parallel degree decides how the model's layers
      are split across stages, so a given server registers a different layer
      subset (and slot count) per degree.
    - `mla_layer_split_kv_cache`: MLA layer-split registration shards each
      block's slots across ranks, a different per-block layout than the
      default full-slot registration.
    - `is_hma_enabled`: vLLM's hybrid cache manager changes whether hybrid
      cache layouts can share one logical block namespace.
    """
    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    additional_config = getattr(vllm_config, "additional_config", None) or {}

    factors = {
        "model": model_config.model,
        "dtype": str(model_config.dtype),
        "tp_size": tp_size,
        "pp_size": vllm_config.parallel_config.pipeline_parallel_size,
        "num_kv_heads": model_config.get_total_num_kv_heads(),
        "head_size": model_config.get_head_size(),
        "num_hidden_layers": model_config.get_total_num_hidden_layers(),
        "cache_dtype": str(cache_config.cache_dtype),
        "is_hma_enabled": not getattr(
            getattr(vllm_config, "scheduler_config", None),
            "disable_hybrid_kv_cache_manager",
            False,
        ),
        "dcp_world_size": dcp_world_size,
        "pcp_world_size": pcp_world_size,
        "cross_layer_blocks": cross_layer_blocks,
        "mla_layer_split_kv_cache": bool(additional_config.get("mla_layer_split_kv_cache", False)),
    }

    factor_str = str(sorted(factors.items()))
    hash_suffix = hashlib.sha256(factor_str.encode()).hexdigest()[:8]
    return f"{hash_suffix}"


def detect_mla(vllm_config) -> bool:
    """Detect if the model uses Multi-head Latent Attention (e.g. DeepSeek V2/V3)."""
    hf_config = vllm_config.model_config.hf_text_config
    return getattr(hf_config, "kv_lora_rank", None) is not None


_TRANSFER_BACKENDS = ("direct", "kernel", "ascend_direct")


def resolve_transfer_backend(
    is_mla: bool,
    override: str | None,
    is_npu: bool | None = None,
) -> str:
    """Pick the engine's H2D/D2H backend for this model.

    MLA models save/load many small, highly fragmented slots where the kernel
    backend's single launch beats one cuMemcpyAsync per slot; everything else
    defaults to direct (best bandwidth for few/large transfers). A non-empty
    `override` (from `pegaflow.transfer_backend`) wins, and an unknown value is
    rejected rather than silently falling back.

    On Ascend NPU, the `kernel` backend is **not** available (CUDA-only), so
    MLA defaults to ``"direct"`` on NPU (mapped to ``AscendMemcpyBackend``
    by the engine).  ``is_npu`` is auto-detected via ``torch.npu`` when not
    explicitly provided.
    """
    if is_npu is None:
        is_npu = _is_npu_available()

    if override is None:
        if is_mla:
            return "direct" if is_npu else "kernel"
        return "direct"
    normalized = override.strip().lower()
    if normalized not in _TRANSFER_BACKENDS:
        allowed = ", ".join(_TRANSFER_BACKENDS)
        raise ValueError(
            f"Unsupported pegaflow.transfer_backend {override!r}; expected one of: {allowed}"
        )
    return normalized


__all__ = [
    "ConnectorContext",
    "LoadIntent",
    "PegaConnectorMode",
    "PegaConnectorMetadata",
    "PegaKVConnectorStats",
    "PegaPromMetrics",
    "RecurrentLoadHold",
    "SaveIntent",
    "TpShardTopology",
    "derive_namespace",
    "detect_mla",
    "logger",
    "parse_env_int",
    "reconcile_hybrid_hit",
    "resolve_instance_id",
    "resolve_transfer_backend",
]


def _is_npu_available() -> bool:
    """True when torch.npu is importable and reports at least one device."""
    try:
        import torch
        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


def _hmma_accept_spec(spec) -> bool:
    """True when a KV group spec is supported by PegaFlow HMA.

    DeepSeek-V4 groups are UniformTypeKVCacheSpecs wrappers (SWA layers and
    MLA layers are packed separately) — inspect the inner layer specs.
    """
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MambaSpec,
        SlidingWindowSpec,
        UniformTypeKVCacheSpecs,
    )

    if isinstance(spec, UniformTypeKVCacheSpecs):
        inner = list(spec.kv_cache_specs.values())
        return all(
            isinstance(x, (FullAttentionSpec, SlidingWindowSpec, MambaSpec))
            for x in inner
        )
    return isinstance(spec, (FullAttentionSpec, SlidingWindowSpec, MambaSpec))


def _hmma_is_full_attention_group(spec) -> bool:
    """True when a KV group spec contains a dense (FullAttention-family) layer.

    DeepSeek-V4 packs MLA layers into a UniformTypeKVCacheSpecs wrapper —
    MLAAttentionSpec subclasses FullAttentionSpec — so a wrapper whose inner
    layers are all MLA counts as the dense hash group.
    """
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        UniformTypeKVCacheSpecs,
    )

    if isinstance(spec, UniformTypeKVCacheSpecs):
        inner = list(spec.kv_cache_specs.values())
        return bool(inner) and all(isinstance(x, FullAttentionSpec) for x in inner)
    return isinstance(spec, FullAttentionSpec)

