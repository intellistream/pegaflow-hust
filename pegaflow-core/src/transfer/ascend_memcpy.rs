//! Ascend DMA copy-engine backend: batch-submit all copies to
//! ``aclrtMemcpyBatchAsync`` (CANN 8.5+) or fall back to per-copy
//! ``aclrtMemcpyAsync`` on older runtimes.
//!
//! This matches the vllm-ascend ``swap_blocks_batch`` pattern: one
//! kernel launch for the entire transfer batch instead of per-block
//! DMA overhead. Contiguous ranges are still coalesced to minimise
//! the number of entries in the batch.
//!
//! API mappings (CANN ← CUDA):
//! - ``aclrtMemcpyBatchAsync(HOST_TO_DEVICE)`` ← ``cuMemcpyHtoDAsync_v2`` loop
//! - ``aclrtMemcpyBatchAsync(DEVICE_TO_HOST)`` ← ``cuMemcpyDtoHAsync_v2`` loop

use std::sync::Arc;

use crate::device::{DeviceStream, ascend};

use super::{CopyDesc, TransferBackend};

/// Ascend DMA copy-engine transfer backend.
///
/// Uses ``aclrtMemcpyBatchAsync`` (CANN 8.5+) for H2D/D2H transfers,
/// submitting all copies in a single batch call.  On older CANN runtimes
/// the batch call falls back to per-entry ``aclrtMemcpyAsync`` inside
/// ``ascend::memcpy_h2d_batch_async`` / ``ascend::memcpy_d2h_batch_async``.
pub struct AscendMemcpyBackend;

impl TransferBackend for AscendMemcpyBackend {
    fn h2d(&self, copies: &[CopyDesc], stream: &Arc<DeviceStream>) -> Result<(), String> {
        let ascend_stream = match stream.as_ref() {
            DeviceStream::Ascend(s) => s,
            _ => return Err("AscendMemcpyBackend::h2d called with non-Ascend stream".into()),
        };
        if copies.is_empty() {
            return Ok(());
        }
        let batch: Vec<ascend::BatchCopyDesc> = copies
            .iter()
            .map(|c| ascend::BatchCopyDesc {
                dst: c.device,
                dst_max: c.size,
                src: c.host_device,
                size: c.size,
            })
            .collect();
        ascend::memcpy_h2d_batch_async(&batch, ascend_stream)
    }

    fn d2h(&self, copies: &[CopyDesc], stream: &Arc<DeviceStream>) -> Result<(), String> {
        let ascend_stream = match stream.as_ref() {
            DeviceStream::Ascend(s) => s,
            _ => return Err("AscendMemcpyBackend::d2h called with non-Ascend stream".into()),
        };
        if copies.is_empty() {
            return Ok(());
        }
        // For D2H: device is the source, host is the destination.
        // In BatchCopyDesc: dst=host, src=device.
        let batch: Vec<ascend::BatchCopyDesc> = copies
            .iter()
            .map(|c| ascend::BatchCopyDesc {
                dst: c.host_device,
                dst_max: c.size,
                src: c.device,
                size: c.size,
            })
            .collect();
        ascend::memcpy_d2h_batch_async(&batch, ascend_stream)
    }

    fn name(&self) -> &'static str { "ascend_batch" }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ascend_backend_name() {
        assert_eq!(AscendMemcpyBackend.name(), "ascend_batch");
    }
}
