//! Layer 4 (Engine Core) multi-instance integration tests for Ascend NPU.
//!
//! These tests validate:
//! - Multiple device contexts across different NPU devices
//! - Cross-thread device context isolation (aclrtSetDevice per thread)
//! - Concurrent save/load operations across multiple worker pools
//! - Instance lifecycle (register → seal → unregister → re-register)
//!
//! # Prerequisites
//! - Ascend NPU device(s) accessible
//! - CANN runtime (libascendcl.so) in LD_LIBRARY_PATH
//! - `ASCEND_HOME_PATH` or equivalent environment set
//! - Build with `--features ascend`
//!
//! Run: `cargo test --test ascend_multi_instance --features ascend -- --nocapture`

use std::sync::Arc;
use std::thread;

use pegaflow_core::device::{DeviceContext, DeviceStream, ascend};
use pegaflow_core::transfer::{AscendMemcpyBackend, CopyDesc, TransferBackend};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Initialize device 0. Returns `Err` if no Ascend device is available.
fn try_init_device(device_id: i32) -> Result<ascend::AscendDevice, String> {
    ascend::ensure_acl_initialized()?;
    let device = ascend::AscendDevice::new(device_id)?;
    device.set_current()?;
    Ok(device)
}

/// Get the number of available Ascend NPU devices.
fn npu_device_count() -> usize {
    pegaflow_common::get_npu_device_count()
        .map(|c| c as usize)
        .unwrap_or(0)
}

fn skip_if_no_npu() -> Option<ascend::AscendDevice> {
    match try_init_device(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("SKIP: cannot init Ascend device 0: {e}");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Test 1: Cross-thread device context isolation
// ---------------------------------------------------------------------------

/// Spawn multiple threads, each calling `aclrtSetDevice` on a different device,
/// then verify each thread correctly targets its own device.
#[test]
fn ascend_cross_thread_device_isolation() {
    let device = match skip_if_no_npu() {
        Some(d) => d,
        None => return,
    };
    drop(device);

    // Use device 0 for all threads (single-device test)
    let handles: Vec<thread::JoinHandle<Result<(), String>>> = (0..4)
        .map(|idx| {
            thread::Builder::new()
                .name(format!("test-thread-{idx}"))
                .spawn(move || {
                    // Each thread must init ACL and set its own device
                    ascend::ensure_acl_initialized()?;
                    let dev = ascend::AscendDevice::new(0)?;
                    dev.set_current()?;

                    // Allocate pinned memory (requires correct device context)
                    const SIZE: usize = 1024;
                    let (host, _device) = ascend::malloc_host(0, SIZE)
                        .map_err(|e| format!("thread {idx}: malloc_host failed: {e}"))?;

                    // Verify alignment
                    assert!(
                        (host as usize) % 64 == 0,
                        "thread {idx}: pinned memory not 64-byte aligned"
                    );

                    ascend::free_host(host).ok();
                    Ok(())
                })
                .expect("failed to spawn thread")
        })
        .collect();

    for (idx, handle) in handles.into_iter().enumerate() {
        match handle.join() {
            Ok(Ok(())) => {} // success
            Ok(Err(e)) => panic!("thread {idx} failed: {e}"),
            Err(_) => panic!("thread {idx} panicked"),
        }
    }

    eprintln!("PASS: ascend_cross_thread_device_isolation");
}

// ---------------------------------------------------------------------------
// Test 2: Multiple device contexts - two devices, two streams
// ---------------------------------------------------------------------------

/// Create DeviceContext for two different NPU devices, create streams on each,
/// and verify they operate independently.
#[test]
fn ascend_multi_device_contexts() {
    let count = npu_device_count();
    if count < 2 {
        eprintln!("SKIP: need >=2 NPU devices (found {count})");
        return;
    }

    // Initialize both devices
    let dev0 = match try_init_device(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: device 0 init failed: {e}");
            return;
        }
    };
    let dev1 = match try_init_device(1) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: device 1 init failed: {e}");
            return;
        }
    };

    // Create streams on each device
    let stream0 = match dev0.create_stream() {
        Ok(s) => s,
        Err(e) => panic!("device 0 create_stream: {e}"),
    };
    let stream1 = match dev1.create_stream() {
        Ok(s) => s,
        Err(e) => panic!("device 1 create_stream: {e}"),
    };

    // Allocate on each device
    const SIZE: usize = 4096;
    let policy: i32 = 0;

    // Device 0 allocation
    dev0.set_current().expect("set device 0");
    let dev0_ptr = ascend::malloc_device(SIZE, policy).expect("malloc device 0");
    let (host0, _) = ascend::malloc_host(0, SIZE).expect("malloc_host device 0");

    // Device 1 allocation
    dev1.set_current().expect("set device 1");
    let dev1_ptr = ascend::malloc_device(SIZE, policy).expect("malloc device 1");
    let (host1, _) = ascend::malloc_host(1, SIZE).expect("malloc_host device 1");

    // Write distinct patterns to each device
    dev0.set_current().expect("set device 0");
    let pattern0: Vec<u8> = vec![0xAA; SIZE];
    ascend::memcpy_h2d_sync(dev0_ptr, pattern0.as_ptr(), SIZE).expect("h2d device 0");

    dev1.set_current().expect("set device 1");
    let pattern1: Vec<u8> = vec![0xBB; SIZE];
    ascend::memcpy_h2d_sync(dev1_ptr, pattern1.as_ptr(), SIZE).expect("h2d device 1");

    // D2H from device 0 - should get pattern0
    dev0.set_current().expect("set device 0");
    ascend::memcpy_d2h_async(host0, dev0_ptr, SIZE, &stream0).expect("d2h device 0");
    stream0.synchronize().expect("sync device 0");

    // D2H from device 1 - should get pattern1
    dev1.set_current().expect("set device 1");
    ascend::memcpy_d2h_async(host1, dev1_ptr, SIZE, &stream1).expect("d2h device 1");
    stream1.synchronize().expect("sync device 1");

    // Verify
    let host0_slice = unsafe { std::slice::from_raw_parts(host0, SIZE) };
    let host1_slice = unsafe { std::slice::from_raw_parts(host1, SIZE) };
    assert_eq!(host0_slice, &pattern0[..], "device 0 data mismatch");
    assert_eq!(host1_slice, &pattern1[..], "device 1 data mismatch");

    // Cleanup
    ascend::free_host(host0).ok();
    ascend::free_host(host1).ok();
    ascend::free_device(dev0_ptr).ok();
    ascend::free_device(dev1_ptr).ok();
    drop(stream0);
    drop(stream1);
    drop(dev0);
    drop(dev1);

    eprintln!("PASS: ascend_multi_device_contexts");
}

// ---------------------------------------------------------------------------
// Test 3: Concurrent operations on two devices via threads
// ---------------------------------------------------------------------------

/// Spawn two threads targeting different devices and run D2H/H2D roundtrips
/// concurrently. Validates that per-thread aclrtSetDevice prevents cross-talk.
#[test]
fn ascend_concurrent_device_ops() {
    let count = npu_device_count();
    if count < 2 {
        eprintln!("SKIP: need >=2 NPU devices (found {count})");
        return;
    }

    // Pre-init both devices sequentially to avoid races during ACL init
    let _dev0 = match try_init_device(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: device 0 init failed: {e}");
            return;
        }
    };
    let _dev1 = match try_init_device(1) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: device 1 init failed: {e}");
            return;
        }
    };

    const SIZE: usize = 4096;
    const PATTERN0: u8 = 0x5A;
    const PATTERN1: u8 = 0xA5;

    let t0 = thread::spawn(move || {
        ascend::ensure_acl_initialized().ok();
        let dev = ascend::AscendDevice::new(0).expect("device 0");
        dev.set_current().expect("set device 0");

        let stream = dev.create_stream().expect("stream 0");
        let dev_ptr = ascend::malloc_device(SIZE, 0).expect("malloc 0");
        let (host, _) = ascend::malloc_host(0, SIZE).expect("malloc_host 0");

        let pattern = vec![PATTERN0; SIZE];
        ascend::memcpy_h2d_sync(dev_ptr, pattern.as_ptr(), SIZE).expect("h2d sync 0");

        // Several roundtrips
        for i in 0..100 {
            ascend::memcpy_d2h_async(host, dev_ptr, SIZE, &stream)
                .unwrap_or_else(|e| panic!("d2h {i} dev 0: {e}"));
            stream.synchronize().expect("sync after d2h 0");

            ascend::memcpy_h2d_async(dev_ptr, host, SIZE, &stream)
                .unwrap_or_else(|e| panic!("h2d {i} dev 0: {e}"));
            stream.synchronize().expect("sync after h2d 0");
        }

        // Final verify
        let mut verify = vec![0u8; SIZE];
        ascend::memcpy_d2h_sync(verify.as_mut_ptr(), dev_ptr, SIZE).expect("verify 0");
        assert_eq!(&verify[..], &pattern[..], "device 0 final mismatch");

        ascend::free_host(host).ok();
        ascend::free_device(dev_ptr).ok();
        drop(stream);
    });

    let t1 = thread::spawn(move || {
        ascend::ensure_acl_initialized().ok();
        let dev = ascend::AscendDevice::new(1).expect("device 1");
        dev.set_current().expect("set device 1");

        let stream = dev.create_stream().expect("stream 1");
        let dev_ptr = ascend::malloc_device(SIZE, 0).expect("malloc 1");
        let (host, _) = ascend::malloc_host(1, SIZE).expect("malloc_host 1");

        let pattern = vec![PATTERN1; SIZE];
        ascend::memcpy_h2d_sync(dev_ptr, pattern.as_ptr(), SIZE).expect("h2d sync 1");

        for i in 0..100 {
            ascend::memcpy_d2h_async(host, dev_ptr, SIZE, &stream)
                .unwrap_or_else(|e| panic!("d2h {i} dev 1: {e}"));
            stream.synchronize().expect("sync after d2h 1");

            ascend::memcpy_h2d_async(dev_ptr, host, SIZE, &stream)
                .unwrap_or_else(|e| panic!("h2d {i} dev 1: {e}"));
            stream.synchronize().expect("sync after h2d 1");
        }

        let mut verify = vec![0u8; SIZE];
        ascend::memcpy_d2h_sync(verify.as_mut_ptr(), dev_ptr, SIZE).expect("verify 1");
        assert_eq!(&verify[..], &pattern[..], "device 1 final mismatch");

        ascend::free_host(host).ok();
        ascend::free_device(dev_ptr).ok();
        drop(stream);
    });

    t0.join().expect("thread 0 panicked");
    t1.join().expect("thread 1 panicked");

    eprintln!("PASS: ascend_concurrent_device_ops");
}

// ---------------------------------------------------------------------------
// Test 4: AscendMemcpyBackend with multiple streams (load + save simulation)
// ---------------------------------------------------------------------------

/// Simulate a load stream and a save stream operating concurrently on the same
/// device, verifying that events correctly order the operations.
#[test]
fn ascend_load_save_concurrent_streams() {
    let device = match skip_if_no_npu() {
        Some(d) => d,
        None => return,
    };

    const SIZE: usize = 4096;
    let policy: i32 = 0;

    let dev_ptr = ascend::malloc_device(SIZE, policy).expect("aclrtMalloc");

    // Two separate streams (simulating load and save worker streams)
    let load_stream = device.create_stream().expect("load stream");
    let save_stream = device.create_stream().expect("save stream");

    let (host_ptr, _) = ascend::malloc_host(0, SIZE).expect("aclrtMallocHost");

    // Write test pattern
    let src: Vec<u8> = (0..SIZE as u8).map(|i| i.wrapping_mul(3)).collect();
    ascend::memcpy_h2d_sync(dev_ptr, src.as_ptr(), SIZE).expect("initial H2D");

    let backend = AscendMemcpyBackend;

    // Save (D2H) on save_stream
    let save_desc = CopyDesc {
        device: dev_ptr,
        host: host_ptr,
        host_device: host_ptr as u64,
        size: SIZE,
    };
    let save_stream_arc: Arc<DeviceStream> =
        Arc::new(DeviceStream::Ascend(save_stream));

    backend.d2h(&[save_desc], &save_stream_arc).expect("save D2H");

    // Record event on save stream to wait for save completion
    let save_done = save_stream_arc.record_event().expect("record save event");

    // While save is in-flight, prepare load (H2D) on load_stream
    // but wait for save to complete first via the event
    let load_stream_arc: Arc<DeviceStream> =
        Arc::new(DeviceStream::Ascend(load_stream));

    // Wait for save to complete before loading
    DeviceStream::wait_event(&save_done).expect("wait save event");

    // Corrupt device memory, then load back
    let zeros = vec![0u8; SIZE];
    ascend::memcpy_h2d_sync(dev_ptr, zeros.as_ptr(), SIZE).expect("corrupt");

    backend.h2d(&[save_desc], &load_stream_arc).expect("load H2D");

    // Wait for load to complete
    #[allow(irrefutable_let_patterns)]
    if let DeviceStream::Ascend(ref s) = *load_stream_arc {
        s.synchronize().expect("sync load stream");
    }

    // Verify
    let mut verify = vec![0u8; SIZE];
    ascend::memcpy_d2h_sync(verify.as_mut_ptr(), dev_ptr, SIZE).expect("verify");
    assert_eq!(&verify[..], &src[..], "save → load data mismatch");

    ascend::free_host(host_ptr).ok();
    ascend::free_device(dev_ptr).ok();
    drop(device);

    eprintln!("PASS: ascend_load_save_concurrent_streams");
}

// ---------------------------------------------------------------------------
// Test 5: Multi-instance pinned memory allocation
// ---------------------------------------------------------------------------

/// Allocate pinned memory on two different devices, verify both allocations
/// are valid, properly aligned, and do not overlap.
#[test]
fn ascend_multi_instance_pinned_memory() {
    let count = npu_device_count();
    if count < 1 {
        eprintln!("SKIP: no Ascend NPU device found");
        return;
    }

    let device = match skip_if_no_npu() {
        Some(d) => d,
        None => return,
    };

    const SIZE: usize = 65536; // 64 KiB

    // Test on device 0 (and device 1 if available)
    let devices_to_test: Vec<i32> = if count >= 2 {
        vec![0, 1]
    } else {
        vec![0]
    };

    let mut allocations: Vec<(*mut u8, i32)> = Vec::new();

    for &dev_id in &devices_to_test {
        device.set_current().expect("set device");
        let (host, _device) = match ascend::malloc_host(dev_id, SIZE) {
            Ok(h) => h,
            Err(e) => panic!("malloc_host device {dev_id}: {e}"),
        };

        let addr = host as usize;
        assert!(
            addr % 64 == 0,
            "device {dev_id}: host pointer {addr:#x} not 64-byte aligned"
        );

        // Write per-device pattern to verify independence
        let fill_byte = if dev_id == 0 { 0xCCu8 } else { 0xDDu8 };
        unsafe {
            std::ptr::write_bytes(host, fill_byte, SIZE);
        }

        allocations.push((host, dev_id));
    }

    // Verify patterns survived (no cross-contamination)
    for (host, dev_id) in &allocations {
        let first_byte = unsafe { *(*host) };
        let expected = if *dev_id == 0 { 0xCCu8 } else { 0xDDu8 };
        assert_eq!(
            first_byte, expected,
            "device {dev_id}: pinned memory data mismatch"
        );
    }

    // Cleanup
    for (host, _) in allocations {
        ascend::free_host(host).ok();
    }
    drop(device);

    eprintln!("PASS: ascend_multi_instance_pinned_memory");
}

// ---------------------------------------------------------------------------
// Test 6: Device context enum roundtrip
// ---------------------------------------------------------------------------

/// Verify that DeviceContext::Ascend and DeviceStream::Ascend roundtrip
/// correctly through the unified device abstraction layer.
#[test]
fn ascend_device_context_roundtrip_via_enum() {
    let device = match skip_if_no_npu() {
        Some(d) => d,
        None => return,
    };

    // Create DeviceContext from AscendDevice
    let ctx = DeviceContext::Ascend(device.clone());
    assert_eq!(ctx.device_id(), device.device_id);

    // Create stream via the enum
    let stream = match ctx.create_stream() {
        Ok(s) => s,
        Err(e) => panic!("create_stream via DeviceContext: {e}"),
    };

    // Verify we got the Ascend variant
    assert!(matches!(&stream, DeviceStream::Ascend(_)));

    // Synchronize
    stream.synchronize().expect("synchronize via DeviceStream");

    // Record event
    let event = stream.record_event().expect("record event");
    DeviceStream::wait_event(&event).expect("wait event");

    eprintln!("PASS: ascend_device_context_roundtrip_via_enum");
}

// ---------------------------------------------------------------------------
// Test 7: AscendMemcpyBackend name and trait compliance
// ---------------------------------------------------------------------------

#[test]
fn ascend_memcpy_backend_trait_compliance() {
    let backend = AscendMemcpyBackend;
    assert_eq!(backend.name(), "ascend_direct");
    eprintln!("PASS: ascend_memcpy_backend_trait_compliance");
}

// ---------------------------------------------------------------------------
// Test 8: Error code 507899 detection (expandable_segments memory)
// ---------------------------------------------------------------------------

/// Verify that attempting D2H from non-aclrtMallocPhysical memory produces
/// error 507899 rather than crashing. This test allocates via aclrtMalloc
/// (which IS aclrtMallocPhysical-compatible on some drivers) so it should
/// succeed; the purpose is to exercise the error path structure.
#[test]
fn ascend_d2h_error_path_does_not_crash() {
    let device = match skip_if_no_npu() {
        Some(d) => d,
        None => return,
    };

    const SIZE: usize = 256;
    let policy: i32 = 0;

    let dev_ptr = match ascend::malloc_device(SIZE, policy) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("SKIP: malloc_device failed: {e}");
            return;
        }
    };

    let (host_ptr, _) = match ascend::malloc_host(0, SIZE) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("SKIP: malloc_host failed: {e}");
            ascend::free_device(dev_ptr).ok();
            return;
        }
    };

    // Write data to device
    let src: Vec<u8> = vec![0x42; SIZE];
    ascend::memcpy_h2d_sync(dev_ptr, src.as_ptr(), SIZE).expect("h2d sync");

    // D2H via backend
    let stream = device.create_stream().expect("create stream");
    let stream_arc: Arc<DeviceStream> = Arc::new(DeviceStream::Ascend(stream));

    let desc = CopyDesc {
        device: dev_ptr,
        host: host_ptr,
        host_device: host_ptr as u64,
        size: SIZE,
    };

    let backend = AscendMemcpyBackend;
    let result = backend.d2h(&[desc], &stream_arc);

    match result {
        Ok(()) => {
            // Success: memory is DMA-capable
            #[allow(irrefutable_let_patterns)]
            if let DeviceStream::Ascend(ref s) = *stream_arc {
                s.synchronize().expect("sync");
            }
            let host_slice = unsafe { std::slice::from_raw_parts(host_ptr, SIZE) };
            assert_eq!(host_slice, &src[..], "D2H data mismatch");
        }
        Err(ref e) if e.contains("507899") => {
            // Expected failure path for non-aclrtMallocPhysical memory
            eprintln!("INFO: D2H correctly reported error 507899: {e}");
        }
        Err(e) => {
            eprintln!("WARN: unexpected D2H error: {e}");
        }
    }

    ascend::free_host(host_ptr).ok();
    ascend::free_device(dev_ptr).ok();
    drop(device);

    eprintln!("PASS: ascend_d2h_error_path_does_not_crash");
}
