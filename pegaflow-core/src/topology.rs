//! Lightweight global cache for NUMA↔device mapping.
//!
//! Populated during engine initialisation and read by Ascend pinned-memory pool
//! creation to determine which NPU device to activate before `aclrtMallocHost`.

use pegaflow_common::NumaNode;
use std::collections::HashMap;
use std::sync::OnceLock;

static DEVICE_NUMA_MAP: OnceLock<HashMap<i32, NumaNode>> = OnceLock::new();

/// Store the device→NUMA mapping discovered during topology detection.
pub(crate) fn init_device_numa_map(map: HashMap<i32, NumaNode>) {
    let _ = DEVICE_NUMA_MAP.set(map);
}

/// Resolve the best NPU device for a given NUMA node.
///
/// Returns the first device found on that NUMA node, or 0 as a safe fallback
/// (single-device systems are by far the most common deployment).
pub(crate) fn resolve_device_for_numa(node: NumaNode) -> i32 {
    DEVICE_NUMA_MAP
        .get()
        .and_then(|map| map.iter().find(|&(_, n)| *n == node).map(|(&dev, _)| dev))
        .unwrap_or(0)
}
