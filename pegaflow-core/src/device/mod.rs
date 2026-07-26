//! Device abstraction layer for CUDA and Ascend backends.
//!
//! Provides [`DeviceContext`] and [`DeviceStream`] enums that wrap
//! CUDA or Ascend-specific handles behind a unified interface for
//! GPU worker pools, transfer backends, and event-based
//! synchronization.

#[cfg(feature = "ascend")]
pub mod ascend;
#[cfg(feature = "cuda")]
pub mod cuda;

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
    ///
    /// On Ascend, this uses `aclrtMalloc` + a synchronous `aclrtMemcpy` zero
    /// fill, which is less efficient than `cuMemAlloc` zero-init.  Callers
    /// that do not need zeroed memory should pre-allocate and reuse buffers
    /// instead of relying on `alloc_zeros`.
    #[cfg(any(feature = "cuda", feature = "ascend"))]
    #[allow(unused)]
    pub fn alloc_zeros<T: 'static + Default + Clone>(
        &self,
        len: usize,
    ) -> Result<Box<dyn std::any::Any + Send>, String> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::CudaSlice;
            if let DeviceStream::Cuda(s) = self {
                // SAFETY: CudaSlice allocation is isolated to CUDA memory.
                // cudarc requires bytemuck::Pod, which is transitively
                // available when the "cuda" feature is active.
                let slice: CudaSlice<u8> = unsafe {
                    s.stream
                        .alloc_zeros::<u8>(len * std::mem::size_of::<T>())
                        .map_err(|e| format!("alloc_zeros failed: {e:?}"))?
                };
                return Ok(Box::new(slice));
            }
        }
        #[cfg(feature = "ascend")]
        {
            if let DeviceStream::Ascend(_) = self {
                use crate::device::ascend;
                use std::alloc::{Layout, alloc, dealloc};

                let size = len.checked_mul(std::mem::size_of::<T>())
                    .ok_or_else(|| "alloc_zeros: size overflow".to_string())?;
                if size == 0 {
                    return Err("alloc_zeros: size must be > 0".into());
                }
                let dev_ptr = ascend::malloc_device(size, 0)
                    .map_err(|e| format!("alloc_zeros(Ascend): aclrtMalloc failed: {e}"))?;

                // Zero-fill via host-side buffer -> sync H2D
                let host_layout = Layout::array::<T>(len)
                    .map_err(|e| format!("alloc_zeros: layout error: {e}"))?;
                let host_ptr = unsafe { alloc(host_layout) };
                if host_ptr.is_null() {
                    ascend::free_device(dev_ptr).ok();
                    return Err("alloc_zeros: host alloc failed".into());
                }
                unsafe { std::ptr::write_bytes(host_ptr, 0, size) };
                let result = ascend::memcpy_h2d_sync(dev_ptr, host_ptr as *const u8, size);
                unsafe { dealloc(host_ptr, host_layout) };
                if let Err(e) = result {
                    ascend::free_device(dev_ptr).ok();
                    return Err(format!("alloc_zeros(Ascend): zero-fill H2D failed: {e}"));
                }

                return Ok(Box::new(dev_ptr));
            }
        }
        Err("alloc_zeros: no device backend available".into())
    }

    /// Record an event on this stream. Returns an opaque event handle.
    ///
    /// The returned handle can be passed to [`DeviceStream::wait_event`] to
    /// block until all preceding work on this stream has completed.
    #[cfg(any(feature = "cuda", feature = "ascend"))]
    pub fn record_event(&self) -> Result<Box<dyn std::any::Any + Send>, String> {
        match self {
            #[cfg(feature = "cuda")]
            DeviceStream::Cuda(s) => s
                .record_event()
                .map(|e| Box::new(e) as Box<dyn std::any::Any + Send>),
            #[cfg(feature = "ascend")]
            DeviceStream::Ascend(s) => s
                .record_event()
                .map(|e| Box::new(e) as Box<dyn std::any::Any + Send>),
        }
    }

    /// Block until a previously recorded event completes.
    ///
    /// `event` must have been obtained from [`DeviceStream::record_event`]
    /// on a stream of the same backend type.
    #[cfg(any(feature = "cuda", feature = "ascend"))]
    pub fn wait_event(event: &Box<dyn std::any::Any + Send>) -> Result<(), String> {
        #[cfg(feature = "cuda")]
        if let Some(e) = event.downcast_ref::<cudarc::driver::CudaEvent>() {
            return e
                .synchronize()
                .map_err(|e| format!("cuda event synchronize failed: {e:?}"));
        }
        #[cfg(feature = "ascend")]
        if let Some(e) = event.downcast_ref::<ascend::AscendEvent>() {
            return e.synchronize();
        }
        Err("wait_event: unrecognized event type".into())
    }
}
