//! Device abstraction layer for CUDA and Ascend backends.
//!
//! Provides [`DeviceContext`] and [`DeviceStream`] enums that wrap
//! CUDA or Ascend-specific handles behind a unified interface for
//! GPU worker pools and transfer backends.

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "ascend")]
pub mod ascend;

/// Unified device context handle.
///
/// Wraps either a CUDA `CudaContext` or an Ascend device id.
/// The Ascend variant is lightweight because `aclrtIpcMemImportByKey`
/// pointers are process-wide — no per-context handle is required.
#[derive(Debug, Clone)]
pub enum DeviceContext {
    /// CUDA device context.
    #[cfg(feature = "cuda")]
    Cuda(Box<cuda::CudaDevice>),
    /// Ascend NPU device handle (device id only, no explicit context).
    #[cfg(feature = "ascend")]
    Ascend(ascend::AscendDevice),
}

impl DeviceContext {
    /// Return the numeric device id.
    pub fn device_id(&self) -> i32 {
        match self {
            #[cfg(feature = "cuda")]
            DeviceContext::Cuda(d) => d.device_id,
            #[cfg(feature = "ascend")]
            DeviceContext::Ascend(d) => d.device_id,
        }
    }

    /// Create a new stream on this device context.
    pub fn create_stream(&self) -> Result<DeviceStream, String> {
        match self {
            #[cfg(feature = "cuda")]
            DeviceContext::Cuda(d) => d.create_stream().map(DeviceStream::Cuda),
            #[cfg(feature = "ascend")]
            DeviceContext::Ascend(d) => d.create_stream().map(DeviceStream::Ascend),
        }
    }
}

/// Unified device stream handle.
///
/// All GPU worker operations (memcpy, kernel launch, synchronize)
/// are enqueued on this stream.
#[derive(Debug)]
pub enum DeviceStream {
    /// CUDA stream backed by `cudarc::driver::CudaStream`.
    #[cfg(feature = "cuda")]
    Cuda(cuda::CudaDeviceStream),
    /// Ascend stream backed by an `aclrtStream` handle.
    #[cfg(feature = "ascend")]
    Ascend(ascend::AscendDeviceStream),
}

impl DeviceStream {
    /// Synchronize the stream, blocking until all previously enqueued
    /// operations complete.
    pub fn synchronize(&self) -> Result<(), String> {
        match self {
            #[cfg(feature = "cuda")]
            DeviceStream::Cuda(s) => s.synchronize(),
            #[cfg(feature = "ascend")]
            DeviceStream::Ascend(s) => s.synchronize(),
        }
    }

    /// Create a new device-side allocation of `len` zeros of type `T`.
    /// Returns an opaque handle that can be used with backend-specific APIs.
    #[cfg(feature = "cuda")]
    pub fn alloc_zeros<T: 'static + Default + Clone + bytemuck::Pod>(
        &self,
        len: usize,
    ) -> Result<Box<dyn std::any::Any + Send>, String> {
        use cudarc::driver::CudaSlice;
        match self {
            DeviceStream::Cuda(s) => {
                // SAFETY: the CudaSlice allocation is isolated to CUDA memory.
                unsafe {
                    let slice: CudaSlice<T> = s.stream.alloc_zeros::<T>(len)
                        .map_err(|e| format!("alloc_zeros failed: {e:?}"))?;
                    Ok(Box::new(slice))
                }
            }
            #[cfg(feature = "ascend")]
            DeviceStream::Ascend(_) => {
                Err("alloc_zeros: Ascend backend does not support scratch allocations".into())
            }
        }
    }

    /// Record an event on this stream. Returns an opaque event handle.
    #[cfg(feature = "cuda")]
    pub fn record_event(&self) -> Result<Box<dyn std::any::Any + Send>, String> {
        match self {
            DeviceStream::Cuda(s) => s.record_event(),
            #[cfg(feature = "ascend")]
            DeviceStream::Ascend(_) => {
                Err("record_event: Ascend backend not yet supported".into())
            }
        }
    }
}