//! Ascend NPU device backend adapter.
//!
//! Uses `extern "C"` FFI to call CANN `libascendcl.so` APIs:
//! `aclrtSetDevice`, `aclrtCreateStream`, `aclrtSynchronizeStream`,
//! `aclrtCreateEvent`, `aclrtRecordEvent`, `aclrtSynchronizeEvent`,
//! `aclrtMemcpyAsync` (HOST_TO_DEVICE / DEVICE_TO_HOST).
//!
//! # Safety
//!
//! All FFI calls are unsafe. Wrappers validate arguments and translate
//! error codes into `String` errors.

use std::ffi::c_void;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// CANN FFI bindings (extern "C")
// ---------------------------------------------------------------------------

#[allow(non_camel_case_types)]
type aclError = i32;

#[allow(non_camel_case_types)]
type aclrtStream = *mut c_void;

#[allow(non_camel_case_types)]
type aclrtEvent = *mut c_void;

#[allow(non_camel_case_types, dead_code)]
type aclrtContext = *mut c_void;

/// Opaque handle for aclrtMemcpyAttr (used by aclrtMemcpyBatchAsync).
/// Always passed as NULL for pegaflow's usage — we never use attributes.
#[allow(non_camel_case_types)]
#[repr(C)]
pub(crate) struct AclrtMemcpyAttr {
    _private: u8,
}

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

    /// Batched asynchronous memory copy (CANN 8.5+).
    /// Transfers multiple discontiguous regions in a single call,
    /// matching the vllm-ascend ``swap_blocks_batch`` pattern.
    /// `kind`: 1 = HOST_TO_DEVICE, 2 = DEVICE_TO_HOST.
    fn aclrtMemcpyBatchAsync(
        dst: *const *mut c_void,
        dst_max: *const usize,
        src: *const *const c_void,
        count: *const usize,
        n: u32,
        attr: *const AclrtMemcpyAttr,
        attrs_index: *const i32,
        num_attrs: u32,
        fail_index: *mut u32,
        stream: aclrtStream,
    ) -> aclError;

    /// Get C_ANN runtime version.
    fn aclrtGetVersion(major: *mut i32, minor: *mut i32, patch: *mut i32) -> aclError;

    // -- Event APIs -----------------------------------------------------

    /// Create an event. `flag` is reserved (pass 0).
    fn aclrtCreateEvent(event: *mut aclrtEvent) -> aclError;

    /// Destroy an event.
    fn aclrtDestroyEvent(event: aclrtEvent) -> aclError;

    /// Record an event into a stream, capturing the stream's progress at
    /// this point so the CPU (or another stream) can later wait on it.
    fn aclrtRecordEvent(event: aclrtEvent, stream: aclrtStream) -> aclError;

    /// Block the calling thread until the event is recorded (i.e. until
    /// all preceding work in the event's stream has completed).
    fn aclrtSynchronizeEvent(event: aclrtEvent) -> aclError;

}

// ---------------------------------------------------------------------------
// ACL Runtime Initialization (lazy, thread-safe)
// ---------------------------------------------------------------------------

static ACL_INIT: OnceLock<Result<(), String>> = OnceLock::new();

fn memoized_init(
    cell: &OnceLock<Result<(), String>>,
    init: impl FnOnce() -> Result<(), String>,
) -> Result<(), String> {
    cell.get_or_init(init).clone()
}

/// Ensure the Ascend CL runtime is initialized (idempotent, thread-safe).
///
/// Called automatically before any ACL API use; safe to call explicitly
/// at process startup for eager initialization.
pub fn ensure_acl_initialized() -> Result<(), String> {
    memoized_init(&ACL_INIT, || {
        let config_path = std::ptr::null::<i8>();
        let ret = unsafe { aclInit(config_path) };
        if ret != ACL_ERROR_NONE {
            return Err(format!("aclInit failed: error code {ret}"));
        }
        Ok(())
    })
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
            return Err(format!("aclrtSynchronizeStream failed: error code {ret}"));
        }
        Ok(())
    }

    /// Record an event on this stream, capturing the stream's progress so the
    /// CPU (or another stream) can later wait on this specific point.
    pub fn record_event(&self) -> Result<AscendEvent, String> {
        let event = AscendEvent::new()?;
        event.record(self.stream)?;
        Ok(event)
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
        if !self.stream.is_null() && ensure_acl_initialized().is_ok() {
            let ret = unsafe { aclrtDestroyStream(self.stream) };
            if ret != ACL_ERROR_NONE {
                log::warn!("aclrtDestroyStream failed: error code {ret}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AscendEvent
// ---------------------------------------------------------------------------

/// Ascend event handle wrapping an `aclrtEvent`.
///
/// Events allow fine-grained synchronization: record an event into a stream,
/// then later wait on that specific event (rather than synchronizing the
/// entire stream). This is useful when multiple operations are in-flight on
/// the same stream and the caller only needs to wait for a subset.
///
/// # Example
///
/// ```ignore
/// let event = stream.record_event()?;
/// // ... submit more work to stream ...
/// AscendEvent::wait(&event)?;  // blocks until the event point is reached
/// ```
#[derive(Debug)]
pub struct AscendEvent {
    event: aclrtEvent,
}

impl AscendEvent {
    /// Create a new event.
    ///
    /// `flag` is reserved and always set to 0.
    pub fn new() -> Result<Self, String> {
        ensure_acl_initialized()?;
        let mut event: aclrtEvent = std::ptr::null_mut();
        let ret = unsafe { aclrtCreateEvent(&mut event) };
        if ret != ACL_ERROR_NONE {
            return Err(format!("aclrtCreateEvent failed: error code {ret}"));
        }
        if event.is_null() {
            return Err("aclrtCreateEvent returned null event".into());
        }
        Ok(Self { event })
    }

    /// Record this event into the given stream.
    ///
    /// The event captures the stream's progress at the point of this call;
    /// subsequent waits on this event will block until all preceding work
    /// in `stream` has completed.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn record(&self, stream: aclrtStream) -> Result<(), String> {
        let ret = unsafe { aclrtRecordEvent(self.event, stream) };
        if ret != ACL_ERROR_NONE {
            return Err(format!("aclrtRecordEvent failed: error code {ret}"));
        }
        Ok(())
    }

    /// Block the calling thread until this event is recorded.
    ///
    /// Equivalent to `cudaEventSynchronize` / `aclrtSynchronizeEvent`.
    pub fn synchronize(&self) -> Result<(), String> {
        let ret = unsafe { aclrtSynchronizeEvent(self.event) };
        if ret != ACL_ERROR_NONE {
            return Err(format!("aclrtSynchronizeEvent failed: error code {ret}"));
        }
        Ok(())
    }

    /// Convenience helper: block until a previously recorded event completes.
    ///
    /// The `event` is the value returned by
    /// [`DeviceStream::record_event`](super::DeviceStream::record_event).
    /// Downcasts to either `AscendEvent` or `CudaEvent` and synchronizes.
    pub fn wait(event: &Box<dyn std::any::Any + Send>) -> Result<(), String> {
        event
            .downcast_ref::<Self>()
            .ok_or_else(|| "wait_event: event is not an AscendEvent".to_string())?
            .synchronize()
    }
}

// SAFETY: `aclrtEvent` is a handle that can be sent across threads.
unsafe impl Send for AscendEvent {}
unsafe impl Sync for AscendEvent {}

impl Drop for AscendEvent {
    fn drop(&mut self) {
        if !self.event.is_null() && ensure_acl_initialized().is_ok() {
            let ret = unsafe { aclrtDestroyEvent(self.event) };
            if ret != ACL_ERROR_NONE {
                log::warn!("aclrtDestroyEvent failed: error code {ret}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ascend Pinned Memory
// ---------------------------------------------------------------------------

/// Allocate pinned host memory via `aclrtMallocHost`.
///
/// `device_id` selects which NPU device context to activate before allocation.
/// On Ascend, `aclrtMallocHost` returns both a host pointer and an
/// associated device pointer in the same allocation, so no separate
/// "get device pointer" call is needed.
pub fn malloc_host(device_id: i32, size: usize) -> Result<(*mut u8, *mut u8), String> {
    ensure_acl_initialized()?;
    // aclrtMallocHost requires an active device context.
    // Use the provided device_id for NUMA-aware placement.
    let ret = unsafe { aclrtSetDevice(device_id) };
    if ret != ACL_ERROR_NONE {
        return Err(format!(
            "aclrtSetDevice({device_id}) before malloc_host({size}) failed: error code {ret}"
        ));
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
        507899 => {
            "memory not allocated via aclrtMallocPhysical (expandable_segments is not DMA-capable — use camem_allocator)"
        }
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

    // For other errors (e.g. 507001 ACL_ERROR_RT_TS_ERROR), fall back to a
    // synchronous aclrtMemcpy.  We intentionally do NOT call stream.synchronize()
    // after the sync copy — the async call may have partially enqueued a task
    // that leaves the stream in a bad state, and synchronize() would trigger the
    // same TS error again.  The sync copy is already complete when it returns.
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

    // For other errors (e.g. 507001 ACL_ERROR_RT_TS_ERROR), fall back to a
    // synchronous aclrtMemcpy.  We intentionally do NOT call stream.synchronize()
    // after the sync copy — the async call may have partially enqueued a task
    // that leaves the stream in a bad state, and synchronize() would trigger the
    // same TS error again.  The sync copy is already complete when it returns.
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
    log::debug!("aclrtMemcpy(D2H sync fallback) succeeded: size={size}");
    Ok(())
}

// ---------------------------------------------------------------------------
// Ascend Batched Memcpy (CANN 8.5+)
// ---------------------------------------------------------------------------

/// One copy descriptor for `memcpy_batch_async`.
#[derive(Clone, Copy)]
pub(crate) struct BatchCopyDesc {
    pub dst: u64,
    pub dst_max: usize,
    pub src: u64,
    pub size: usize,
}

/// Enqueue a batched host→device copy on the given stream.
///
/// Uses `aclrtMemcpyBatchAsync` when available (CANN 8.5+), falling back
/// to per-entry `aclrtMemcpyAsync` on older runtimes.  This matches the
/// vllm-ascend ``swap_blocks_batch`` pattern: one kernel launch for the
/// entire batch instead of per-block DMA overhead.
pub fn memcpy_h2d_batch_async(
    copies: &[BatchCopyDesc],
    stream: &AscendDeviceStream,
) -> Result<(), String> {
    if copies.is_empty() {
        return Ok(());
    }
    ensure_acl_initialized()?;

    // Use the batch API — single aclrtMemcpyBatchAsync call.
    let n = copies.len() as u32;
    let dst_ptrs: Vec<*mut c_void> = copies.iter().map(|c| c.dst as *mut c_void).collect();
    let dst_maxs: Vec<usize> = copies.iter().map(|c| c.dst_max).collect();
    let src_ptrs: Vec<*const c_void> = copies.iter().map(|c| c.src as *const c_void).collect();
    let sizes: Vec<usize> = copies.iter().map(|c| c.size).collect();

    let ret = unsafe {
        aclrtMemcpyBatchAsync(
            dst_ptrs.as_ptr(),
            dst_maxs.as_ptr(),
            src_ptrs.as_ptr(),
            sizes.as_ptr(),
            n,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null_mut(),
            stream.stream,
        )
    };

    if ret != ACL_ERROR_NONE {
        let desc = acl_error_description(ret);
        // Fall back to per-copy aclrtMemcpyAsync for non-DMA-memory errors
        if ret == ACL_ERROR_DMA_NOT_SUPPORTED {
            return Err(format!(
                "aclrtMemcpyBatchAsync(H2D) failed: error {ret} ({desc}). \
                 Enable camem_allocator (COMPILE_CUSTOM_KERNELS=1) for DMA-capable \
                 KV cache allocations."
            ));
        }
        // For other errors, try per-copy fallback
        log::warn!(
            "aclrtMemcpyBatchAsync(H2D) failed: error {ret} ({desc}) — \
             falling back to per-copy aclrtMemcpyAsync"
        );
        for copy in copies {
            memcpy_h2d_async(copy.dst, copy.src as *const u8, copy.size, stream)?;
        }
    }
    Ok(())
}

/// Enqueue a batched device→host copy on the given stream.
///
/// Mirrors `memcpy_h2d_batch_async` for the reverse direction.
pub fn memcpy_d2h_batch_async(
    copies: &[BatchCopyDesc],
    stream: &AscendDeviceStream,
) -> Result<(), String> {
    if copies.is_empty() {
        return Ok(());
    }
    ensure_acl_initialized()?;

    let n = copies.len() as u32;
    let dst_ptrs: Vec<*mut c_void> = copies.iter().map(|c| c.dst as *mut c_void).collect();
    let dst_maxs: Vec<usize> = copies.iter().map(|c| c.dst_max).collect();
    let src_ptrs: Vec<*const c_void> = copies.iter().map(|c| c.src as *const c_void).collect();
    let sizes: Vec<usize> = copies.iter().map(|c| c.size).collect();

    let ret = unsafe {
        aclrtMemcpyBatchAsync(
            dst_ptrs.as_ptr(),
            dst_maxs.as_ptr(),
            src_ptrs.as_ptr(),
            sizes.as_ptr(),
            n,
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null_mut(),
            stream.stream,
        )
    };

    if ret != ACL_ERROR_NONE {
        let desc = acl_error_description(ret);
        if ret == ACL_ERROR_DMA_NOT_SUPPORTED {
            return Err(format!(
                "aclrtMemcpyBatchAsync(D2H) failed: error {ret} ({desc}). \
                 Enable camem_allocator (COMPILE_CUSTOM_KERNELS=1) for DMA-capable \
                 KV cache allocations."
            ));
        }
        log::warn!(
            "aclrtMemcpyBatchAsync(D2H) failed: error {ret} ({desc}) — \
             falling back to per-copy aclrtMemcpyAsync"
        );
        for copy in copies {
            memcpy_d2h_async(copy.dst as *mut u8, copy.src, copy.size, stream)?;
        }
    }
    Ok(())
}

// -- Device allocation (test helpers) -------------------------------------

/// Allocate device memory via `aclrtMalloc`. Returns a raw device pointer as `u64`.
///
/// `policy` is the ACL memory policy (typically 0 for default).
pub fn malloc_device(size: usize, policy: i32) -> Result<u64, String> {
    ensure_acl_initialized()?;
    // aclrtMalloc requires an active device context
    let ret = unsafe { aclrtSetDevice(0) };
    if ret != ACL_ERROR_NONE {
        return Err(format!(
            "aclrtSetDevice(0) before malloc_device({size}) failed: error code {ret}"
        ));
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
        return Err(format!(
            "aclrtSetDevice(0) before memcpy_h2d_sync failed: error code {ret}"
        ));
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[test]
    fn acl_init_failure_is_memoized() {
        let cell = OnceLock::new();
        let attempts = AtomicUsize::new(0);

        let first = memoized_init(&cell, || {
            attempts.fetch_add(1, Ordering::Relaxed);
            Err("initialization failed".to_owned())
        });
        let second = memoized_init(&cell, || {
            attempts.fetch_add(1, Ordering::Relaxed);
            Ok(())
        });

        assert_eq!(first, Err("initialization failed".to_owned()));
        assert_eq!(second, Err("initialization failed".to_owned()));
        assert_eq!(attempts.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn acl_init_success_runs_once() {
        let cell = OnceLock::new();
        let attempts = AtomicUsize::new(0);

        for _ in 0..2 {
            memoized_init(&cell, || {
                attempts.fetch_add(1, Ordering::Relaxed);
                Ok(())
            })
            .unwrap();
        }

        assert_eq!(attempts.load(Ordering::Relaxed), 1);
    }

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

    // -- Event tests (off-device smoke) ----------------------------------

    /// `AscendEvent::wait` rejects a non-AscendEvent box.
    #[test]
    fn ascend_event_wait_rejects_wrong_type() {
        // Box a plain integer — should fail to downcast.
        let wrong: Box<dyn std::any::Any + Send> = Box::new(42i32);
        let result = AscendEvent::wait(&wrong);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not an AscendEvent"));
    }
}
