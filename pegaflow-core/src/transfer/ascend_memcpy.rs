//! Ascend DMA copy-engine backend: coalesce contiguous copies, then
//! one directional `aclrtMemcpyAsync` per merged range (H2D or D2H).
//!
//! API mappings (CANN ← CUDA):
//! - `aclrtMemcpyAsync(HOST_TO_DEVICE)` ← `cuMemcpyHtoDAsync_v2`
//! - `aclrtMemcpyAsync(DEVICE_TO_HOST)` ← `cuMemcpyDtoHAsync_v2`

use std::sync::Arc;

use crate::device::{DeviceStream, ascend};

use super::{CopyDesc, TransferBackend};

/// A run of input copies that are contiguous on both the device and the host
/// side, submitted as a single `aclrtMemcpyAsync`.
struct Merged {
    device: u64,
    host: *mut u8,
    size: usize,
}

/// Coalesce copies that are adjacent on both sides into larger ranges.
/// Same merge logic as the CUDA memcpy backend.
fn merge(copies: &[CopyDesc]) -> Vec<Merged> {
    let mut out = Vec::with_capacity(copies.len());
    let mut i = 0;
    while i < copies.len() {
        let start = copies[i];
        let mut size = start.size;
        let mut j = i + 1;
        while j < copies.len() {
            let next = copies[j];
            let device_contiguous = start.device + size as u64 == next.device;
            // SAFETY: pointer arithmetic used only for an address-equality check;
            // the result is never dereferenced.
            let host_contiguous = unsafe { start.host.add(size) } == next.host;
            if device_contiguous && host_contiguous {
                size += next.size;
                j += 1;
            } else {
                break;
            }
        }
        out.push(Merged {
            device: start.device,
            host: start.host,
            size,
        });
        i = j;
    }
    out
}

/// Ascend DMA copy-engine transfer backend.
///
/// Uses `aclrtMemcpyAsync` for H2D/D2H transfers, coalescing
/// contiguous ranges into minimal submissions.
pub struct AscendMemcpyBackend;

impl TransferBackend for AscendMemcpyBackend {
    fn h2d(&self, copies: &[CopyDesc], stream: &Arc<DeviceStream>) -> Result<(), String> {
        let ascend_stream = match stream.as_ref() {
            DeviceStream::Ascend(s) => s,
            _ => return Err("AscendMemcpyBackend::h2d called with non-Ascend stream".into()),
        };
        for m in merge(copies) {
            ascend::memcpy_h2d_async(m.device, m.host, m.size, ascend_stream)?;
        }
        Ok(())
    }

    fn d2h(&self, copies: &[CopyDesc], stream: &Arc<DeviceStream>) -> Result<(), String> {
        let ascend_stream = match stream.as_ref() {
            DeviceStream::Ascend(s) => s,
            _ => return Err("AscendMemcpyBackend::d2h called with non-Ascend stream".into()),
        };
        for m in merge(copies) {
            ascend::memcpy_d2h_async(m.host, m.device, m.size, ascend_stream)?;
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "ascend_direct"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascend_backend_name() {
        assert_eq!(AscendMemcpyBackend.name(), "ascend_direct");
    }
}