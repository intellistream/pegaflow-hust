//! CUDA device backend adapter.
//!
//! Wraps `cudarc` types (`CudaContext`, `CudaStream`) into the
//! [`DeviceContext`] / [`DeviceStream`] enum variants.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, CudaEvent};

/// CUDA device handle wrapping a `cudarc::driver::CudaContext`.
#[derive(Debug, Clone)]
pub struct CudaDevice {
    pub(crate) ctx: Arc<CudaContext>,
    pub device_id: i32,
}

impl CudaDevice {
    /// Initialize CUDA context for the given device ordinal.
    pub fn new(device_id: i32) -> Result<Self, String> {
        let ctx = CudaContext::new(device_id as usize)
            .map_err(|e| format!("CudaContext::new(device {device_id}) failed: {e:?}"))?;
        Ok(Self {
            ctx: Arc::new(ctx),
            device_id,
        })
    }

    /// Access the underlying CUDA context.
    pub(crate) fn inner(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Create a new stream on this device.
    pub fn create_stream(&self) -> Result<CudaDeviceStream, String> {
        let stream = self
            .ctx
            .new_stream()
            .map_err(|e| format!("CudaContext::new_stream failed: {e:?}"))?;
        Ok(CudaDeviceStream { stream })
    }
}

/// CUDA stream handle wrapping a `cudarc::driver::CudaStream`.
#[derive(Debug, Clone)]
pub struct CudaDeviceStream {
    pub(crate) stream: Arc<CudaStream>,
}

impl CudaDeviceStream {
    /// Synchronize the stream.
    pub fn synchronize(&self) -> Result<(), String> {
        self.stream
            .synchronize()
            .map_err(|e| format!("cuda stream synchronize failed: {e:?}"))
    }

    /// Return the raw `CUstream` handle for driver API calls.
    pub(crate) fn cu_stream(&self) -> cudarc::driver::sys::CUstream {
        self.stream.cu_stream()
    }

    /// Access the underlying `Arc<CudaStream>`.
    pub(crate) fn inner(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Record an event after all previously enqueued work.
    pub(crate) fn record_event(&self) -> Result<CudaEvent, String> {
        self.stream
            .record_event(None)
            .map_err(|e| format!("cuda record_event failed: {e:?}"))
    }
}