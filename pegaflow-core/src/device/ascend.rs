//! Ascend NPU device backend adapter.
//!
//! Uses `extern "C"` FFI to call CANN `libascendcl.so` APIs:
//! `aclrtSetDevice`, `aclrtCreateStream`, `aclrtSynchronizeStream`,
//! `aclrtMemcpyAsync` (HOST_TO_DEVICE / DEVICE_TO_HOST).
//!
//! # Safety
//!
//! All FFI calls are unsafe. Wrappers validate arguments and translate
//! error codes into `String` errors.

use std::ffi::c_void;
use std::sync::Once;

// ---------------------------------------------------------------------------
// CANN FFI bindings (extern "C")
// ---------------------------------------------------------------------------

#[allow(non_camel_case_types)]
type aclError = i32;

#[allow(non_camel_case_types)]
type aclrtStream = *mut c_void;

#[allow(non_camel_case_types, dead_code)]
type aclrtContext = *mut c_void;

/// Success return code for all CANN ACL APIs.
pub(crate) const ACL_ERROR_NONE: i32 = 0;

/// ACL memcpy direction: host to device.
const ACL_MEMCPY_HOST_TO_DEVICE: i32 = 1;
/// ACL memcpy direction: device to host.
const ACL_MEMCPY_DEVICE_TO_HOST: i32 = 2;

// Configure the 64-byte alignment requirement for Ascend pinned memory.
pub(crate) const ASCEND_HOST_ALIGNMENT: usize = 64;

unsafe extern "C" {
    /// Initialize the Ascend CL runtime. Must be called once before any
    /// other ACL API.
    fn aclInit(config_path: *const i8) -> aclError;

    /// Set the current device for the calling thread.
    fn aclrtSetDevice(device_id: i32) -> aclError;

    /// Create a stream on the current device. `priority=0` is default.
    fn aclrtCreateStream(stream: *mut aclrtStream) -> aclError;

    /// Destroy a stream.
    fn aclrtDestroyStream(stream: aclrtStream) -> aclError;

    /// Block until all operations on the stream complete.
    fn aclrtSynchronizeStream(stream: aclrtStream) -> aclError;

    /// Asynchronous memory copy.
    /// `kind`: 1 = HOST_TO_DEVICE, 2 = DEVICE_TO_HOST.
    fn aclrtMemcpyAsync(
        dst: *mut c_void,
        dst_max: usize,
        src: *const c_void,
        count: usize,
        kind: i32,
        stream: aclrtStream,
    ) -> aclError;

    /// Synchronous memory copy.
    /// `kind`: 1 = HOST_TO_DEVICE, 2 = DEVICE_TO_HOST, 3 = HOST_TO_HOST, 4 = DEVICE_TO_DEVICE.
    fn aclrtMemcpy(
        dst: *mut c_void,
        dst_max: usize,
        src: *const c_void,
        count: usize,
        kind: i32,
    ) -> aclError;

    /// Allocate device memory on the current device.
    fn aclrtMalloc(ptr: *mut *mut c_void, size: usize, policy: i32) -> aclError;

    /// Free device memory.
    fn aclrtFree(ptr: *mut c_void) -> aclError;

    /// Allocate pinned host memory (DMA-capable).
    fn aclrtMallocHost(ptr: *mut *mut c_void, size: usize) -> aclError;

    /// Free pinned host memory.
    fn aclrtFreeHost(ptr: *mut c_void) -> aclError;

    /// Get C_ANN runtime version.
    fn aclrtGetVersion(major: *mut i32, minor: *mut i32, patch: *mut i32) -> aclError;
}

// ---------------------------------------------------------------------------
// ACL Runtime Initialization (lazy, thread-safe)
// ---------------------------------------------------------------------------

static ACL_INIT: Once = Once::new();

/// Ensure the Ascend CL runtime is initialized (idempotent, thread-safe).
///
/// Called automatically before any ACL API use; safe to call explicitly
/// at process startup for eager initialization.
pub fn ensure_acl_initialized() -> Result<(), String> {
    let mut result = Ok(());
    ACL_INIT.call_once(|| {
        let config_path = std::ptr::null::<i8>();
        let ret = unsafe { aclInit(config_path) };
        if ret != ACL_ERROR_NONE {
            result = Err(format!("aclInit failed: error code {ret}"));
        }
    });
    result
}

/// Deprecated alias kept for backward compatibility.
#[deprecated(note = "use ensure_acl_initialized() instead")]
pub fn init_acl() -> Result<(), String> {
    ensure_acl_initialized()
}

/// Query CANN runtime version as `(major, minor, patch)`.
pub fn get_acl_version() -> Result<(i32, i32, i32), String> {
    ensure_acl_initialized()?;
    let mut major = 0i32;
    let mut minor = 0i32;
    let mut patch = 0i32;
    let ret = unsafe { aclrtGetVersion(&mut major, &mut minor, &mut patch) };
    if ret != ACL_ERROR_NONE {
        return Err(format!("aclrtGetVersion failed: error code {ret}"));
    }
    Ok((major, minor, patch))
}

// ---------------------------------------------------------------------------
// AscendDevice
// ---------------------------------------------------------------------------

/// Ascend NPU device handle.
///
/// The Ascend variant is lightweight: `aclrtIpcMemImportByKey` pointers
/// are process-wide, so no per-context CUDA-like handle is required.
/// Only the device id and stream creation are tracked.
#[derive(Debug, Clone)]
pub struct AscendDevice {
    pub device_id: i32,
}

impl AscendDevice {
    /// Create an Ascend device handle for the given device ordinal.
    ///
    /// This does **not** call `aclrtSetDevice` — that is done per-thread
    /// in the worker threads. This handle is purely a marker.
    pub fn new(device_id: i32) -> Result<Self, String> {
        if device_id < 0 {
            return Err(format!("Ascend device_id {device_id} must be >= 0"));
        }
        Ok(Self { device_id })
    }

    /// Set this device as the active device for the calling thread.
    pub fn set_current(&self) -> Result<(), String> {
        ensure_acl_initialized()?;
        let ret = unsafe { aclrtSetDevice(self.device_id) };
        if ret != ACL_ERROR_NONE {
            return Err(format!(
                "aclrtSetDevice({}) failed: error code {ret}",
                self.device_id
            ));
        }
        Ok(())
    }

    /// Create a new stream on this device.
    pub fn create_stream(&self) -> Result<AscendDeviceStream, String> {
        ensure_acl_initialized()?;
        let mut stream: aclrtStream = std::ptr::null_mut();
        let ret = unsafe { aclrtCreateStream(&mut stream) };
        if ret != ACL_ERROR_NONE {
            return Err(format!("aclrtCreateStream failed: error code {ret}"));
        }
        if stream.is_null() {
            return Err("aclrtCreateStream returned null stream".into());
        }
        Ok(AscendDeviceStream { stream })
    }
}

impl Drop for AscendDevice {
    fn drop(&mut self) {
        // No resources to free — Ascend device handle is process-wide.
    }
}

// ---------------------------------------------------------------------------
// AscendDeviceStream
// ---------------------------------------------------------------------------

/// Ascend stream handle wrapping an `aclrtStream`.
#[derive(Debug)]
pub struct AscendDeviceStream {
    pub(crate) stream: aclrtStream,
}

impl AscendDeviceStream {
    /// Synchronize the stream, blocking until all enqueued work completes.
    pub fn synchronize(&self) -> Result<(), String> {
        let ret = unsafe { aclrtSynchronizeStream(self.stream) };
        if ret != ACL_ERROR_NONE {
            return Err(format!(
                "aclrtSynchronizeStream failed: error code {ret}"
            ));
        }
        Ok(())
    }

    /// Return the raw `aclrtStream` handle.
    pub(crate) fn inner(&self) -> aclrtStream {
        self.stream
    }
}

// SAFETY: `aclrtStream` is a handle that can be sent across threads.
// All API calls are thread-safe on different streams.
unsafe impl Send for AscendDeviceStream {}
unsafe impl Sync for AscendDeviceStream {}

impl Drop for AscendDeviceStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            if ensure_acl_initialized().is_ok() {
                let ret = unsafe { aclrtDestroyStream(self.stream) };
                if ret != ACL_ERROR_NONE {
                    log::warn!("aclrtDestroyStream failed: error code {ret}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ascend Pinned Memory
// ---------------------------------------------------------------------------

/// Allocate pinned host memory via `aclrtMallocHost`.
///
/// On Ascend, `aclrtMallocHost` returns both a host pointer and an
/// associated device pointer in the same allocation, so no separate
/// "get device pointer" call is needed.
pub fn malloc_host(size: usize) -> Result<(*mut u8, *mut u8), String> {
    ensure_acl_initialized()?;
    // aclrtMallocHost requires an active device context
    let ret = unsafe { aclrtSetDevice(0) };
    if ret != ACL_ERROR_NONE {
        return Err(format!("aclrtSetDevice(0) before malloc_host({size}) failed: error code {ret}"));
    }
    if size == 0 {
        return Err("aclrtMallocHost: size must be > 0".into());
    }
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let ret = unsafe { aclrtMallocHost(&mut ptr, size) };
    if ret != ACL_ERROR_NONE {
        return Err(format!("aclrtMallocHost({size}) failed: error code {ret}"));
    }
    if ptr.is_null() {
        return Err("aclrtMallocHost returned null".into());
    }
    // On Ascend, host and device pointers are the same for malloc_host allocations.
    Ok((ptr as *mut u8, ptr as *mut u8))
}

/// Free pinned host memory allocated by `aclrtMallocHost`.
pub fn free_host(ptr: *mut u8) -> Result<(), String> {
    ensure_acl_initialized()?;
    let ret = unsafe { aclrtFreeHost(ptr as *mut c_void) };
    if ret != ACL_ERROR_NONE {
        return Err(format!("aclrtFreeHost failed: error code {ret}"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Ascend Memcpy (H2D / D2H)
// ---------------------------------------------------------------------------

/// Human-readable description for common CANN ACL error codes.
fn acl_error_description(code: i32) -> &'static str {
    match code {
        507899 => "memory not allocated via aclrtMallocPhysical (expandable_segments is not DMA-capable — use camem_allocator)",
        207001 => "invalid parameter",
        207002 => "memory allocation failed",
        207003 => "device not available",
        207004 => "stream is invalid",
        _ => "see CANN documentation for error code details",
    }
}

/// Error code returned when source/destination memory was allocated by the
/// default ``torch_npu`` ``expandable_segments`` allocator instead of
/// ``aclrtMallocPhysical``. Neither async nor synchronous ``aclrtMemcpy*`` can
/// touch this memory; the synchronous path may **crash the process**.
const ACL_ERROR_DMA_NOT_SUPPORTED: i32 = 507899;

/// Enqueue a host-to-device copy on the given stream.
///
/// **Ascend expandable_segments limitation**: when the source or destination
/// memory was allocated by the default ``torch_npu`` allocator (which uses
/// ``expandable_segments``, not ``aclrtMallocPhysical``), ``aclrtMemcpyAsync``
/// fails with error **507899**.  The synchronous fallback is **NOT** attempted
/// for this error because ``aclrtMemcpy`` may segfault on non-DMA memory.
///
/// For save/load to work, enable ``camem_allocator`` (``aclrtMallocPhysical``)
/// via ``COMPILE_CUSTOM_KERNELS=1`` and ensure ``vllm_ascend_C`` is built.
pub fn memcpy_h2d_async(
    dst_device: u64,
    src_host: *const u8,
    size: usize,
    stream: &AscendDeviceStream,
) -> Result<(), String> {
    ensure_acl_initialized()?;
    let ret = unsafe {
        aclrtMemcpyAsync(
            dst_device as *mut c_void,
            size,
            src_host as *const c_void,
            size,
            ACL_MEMCPY_HOST_TO_DEVICE,
            stream.stream,
        )
    };
    if ret == ACL_ERROR_NONE {
        return Ok(());
    }

    let desc = acl_error_description(ret);
    // Error 507899: expandable_segments memory — synchronous fallback will
    // crash the process inside the Ascend driver. Return the error cleanly
    // so the caller can log and continue.
    if ret == ACL_ERROR_DMA_NOT_SUPPORTED {
        return Err(format!(
            "aclrtMemcpyAsync(H2D) failed: error {ret} ({desc}), size={size}. \
             This memory was not allocated via aclrtMallocPhysical. \
             Enable camem_allocator (COMPILE_CUSTOM_KERNELS=1) or use \
             vllm_ascend_C for DMA-capable KV cache allocations."
        ));
    }

    // For other errors, try the synchronous fallback.
    log::warn!(
        "aclrtMemcpyAsync(H2D) failed: error {ret} ({desc}), size={size} — \
         falling back to synchronous aclrtMemcpy"
    );
    let ret = unsafe {
        aclrtMemcpy(
            dst_device as *mut c_void,
            size,
            src_host as *const c_void,
            size,
            ACL_MEMCPY_HOST_TO_DEVICE,
        )
    };
    if ret != ACL_ERROR_NONE {
        let desc = acl_error_description(ret);
        return Err(format!(
            "aclrtMemcpyAsync(H2D) and synchronous fallback both failed: \
             error {ret} ({desc}), size={size}."
        ));
    }
    stream.synchronize()?;
    log::debug!("aclrtMemcpy(H2D sync fallback) succeeded: size={size}");
    Ok(())
}

/// Enqueue a device-to-host copy on the given stream.
///
/// **Ascend expandable_segments limitation**: when the source or destination
/// memory was allocated by the default ``torch_npu`` allocator (which uses
/// ``expandable_segments``, not ``aclrtMallocPhysical``), ``aclrtMemcpyAsync``
/// fails with error **507899**.  The synchronous fallback is **NOT** attempted
/// for this error because ``aclrtMemcpy`` may segfault on non-DMA memory.
///
/// For save/load to work, enable ``camem_allocator`` (``aclrtMallocPhysical``)
/// via ``COMPILE_CUSTOM_KERNELS=1`` and ensure ``vllm_ascend_C`` is built.
pub fn memcpy_d2h_async(
    dst_host: *mut u8,
    src_device: u64,
    size: usize,
    stream: &AscendDeviceStream,
) -> Result<(), String> {
    ensure_acl_initialized()?;
    let ret = unsafe {
        aclrtMemcpyAsync(
            dst_host as *mut c_void,
            size,
            src_device as *const c_void,
            size,
            ACL_MEMCPY_DEVICE_TO_HOST,
            stream.stream,
        )
    };
    if ret == ACL_ERROR_NONE {
        return Ok(());
    }

    let desc = acl_error_description(ret);
    // Error 507899: expandable_segments memory — synchronous fallback will
    // crash the process inside the Ascend driver. Return the error cleanly
    // so the caller can log and continue.
    if ret == ACL_ERROR_DMA_NOT_SUPPORTED {
        return Err(format!(
            "aclrtMemcpyAsync(D2H) failed: error {ret} ({desc}), size={size}. \
             This memory was not allocated via aclrtMallocPhysical. \
             Enable camem_allocator (COMPILE_CUSTOM_KERNELS=1) or use \
             vllm_ascend_C for DMA-capable KV cache allocations."
        ));
    }

    // For other errors, try the synchronous fallback.
    log::warn!(
        "aclrtMemcpyAsync(D2H) failed: error {ret} ({desc}), size={size} — \
         falling back to synchronous aclrtMemcpy"
    );
    let ret = unsafe {
        aclrtMemcpy(
            dst_host as *mut c_void,
            size,
            src_device as *const c_void,
            size,
            ACL_MEMCPY_DEVICE_TO_HOST,
        )
    };
    if ret != ACL_ERROR_NONE {
        let desc = acl_error_description(ret);
        return Err(format!(
            "aclrtMemcpyAsync(D2H) and synchronous fallback both failed: \
             error {ret} ({desc}), size={size}."
        ));
    }
    stream.synchronize()?;
    log::debug!("aclrtMemcpy(D2H sync fallback) succeeded: size={size}");
    Ok(())
}

// ---------------------------------------------------------------------------
// Ascend Device Allocation (for integration tests — not used in production paths)
// ---------------------------------------------------------------------------

/// Allocate device memory via `aclrtMalloc`. Returns a raw device pointer as `u64`.
///
/// `policy` is the ACL memory policy (typically 0 for default).
pub fn malloc_device(size: usize, policy: i32) -> Result<u64, String> {
    ensure_acl_initialized()?;
    // aclrtMalloc requires an active device context
    let ret = unsafe { aclrtSetDevice(0) };
    if ret != ACL_ERROR_NONE {
        return Err(format!("aclrtSetDevice(0) before malloc_device({size}) failed: error code {ret}"));
    }
    if size == 0 {
        return Err("aclrtMalloc: size must be > 0".into());
    }
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let ret = unsafe { aclrtMalloc(&mut ptr, size, policy) };
    if ret != ACL_ERROR_NONE {
        return Err(format!("aclrtMalloc({size}) failed: error code {ret}"));
    }
    if ptr.is_null() {
        return Err("aclrtMalloc returned null".into());
    }
    Ok(ptr as u64)
}

/// Free device memory allocated by `aclrtMalloc`.
pub fn free_device(ptr: u64) -> Result<(), String> {
    ensure_acl_initialized()?;
    let ret = unsafe { aclrtFree(ptr as *mut c_void) };
    if ret != ACL_ERROR_NONE {
        return Err(format!("aclrtFree failed: error code {ret}"));
    }
    Ok(())
}

/// Synchronous host-to-device memory copy (blocking, no stream needed).
pub fn memcpy_h2d_sync(dst_device: u64, src_host: *const u8, size: usize) -> Result<(), String> {
    ensure_acl_initialized()?;
    let ret = unsafe { aclrtSetDevice(0) };
    if ret != ACL_ERROR_NONE {
        return Err(format!("aclrtSetDevice(0) before memcpy_h2d_sync failed: error code {ret}"));
    }
    let ret = unsafe {
        aclrtMemcpy(
            dst_device as *mut c_void,
            size,
            src_host as *const c_void,
            size,
            ACL_MEMCPY_HOST_TO_DEVICE,
        )
    };
    if ret != ACL_ERROR_NONE {
        return Err(format!(
            "aclrtMemcpy(H2D sync) failed: error code {ret}, size={size}"
        ));
    }
    Ok(())
}

/// Synchronous device-to-host memory copy (blocking, no stream needed).
pub fn memcpy_d2h_sync(dst_host: *mut u8, src_device: u64, size: usize) -> Result<(), String> {
    ensure_acl_initialized()?;
    let ret = unsafe {
        aclrtMemcpy(
            dst_host as *mut c_void,
            size,
            src_device as *const c_void,
            size,
            ACL_MEMCPY_DEVICE_TO_HOST,
        )
    };
    if ret != ACL_ERROR_NONE {
        return Err(format!(
            "aclrtMemcpy(D2H sync) failed: error code {ret}, size={size}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: AscendDevice can be constructed without linking.
    /// Actual CANN calls require an Ascend environment; this test
    /// verifies the struct path compiles and validates basic args.
    #[test]
    fn ascend_device_new() {
        let device = AscendDevice::new(0);
        assert!(device.is_ok());
        assert_eq!(device.unwrap().device_id, 0);
    }

    #[test]
    fn ascend_device_negative_id_rejected() {
        let device = AscendDevice::new(-1);
        assert!(device.is_err());
    }

    #[test]
    fn ascend_device_debug() {
        let device = AscendDevice::new(3).unwrap();
        let debug = format!("{device:?}");
        assert!(debug.contains("3"));
    }
}