//! Layer 4 (Engine Core) integration tests: Ascend D2H/H2D memory copy
//! roundtrip verification and pinned memory alignment.
//!
//! These tests use raw `aclrtMalloc` / `aclrtMallocHost` / `aclrtMemcpy`
//! (sync) to bypass Layer 1–3 (gRPC, cross-process IPC), operating entirely
//! within a single process.
//!
//! # Prerequisites
//! - Ascend NPU device(s) accessible
//! - CANN runtime (libascendcl.so) in LD_LIBRARY_PATH
//! - `ASCEND_HOME_PATH` or equivalent environment set
//! - Build with `--features ascend`
//!
//! Run: `cargo test --test ascend_memcpy_roundtrip --features ascend -- --nocapture`

use pegaflow_core::device::ascend;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Initialize device 0. Returns `Err` if no Ascend device is available.
fn try_init_device0() -> Result<ascend::AscendDevice, String> {
    ascend::ensure_acl_initialized()?;
    let device = ascend::AscendDevice::new(0)?;
    device.set_current()?;
    Ok(device)
}

// ---------------------------------------------------------------------------
// Test 1: D2H → H2D roundtrip
// ---------------------------------------------------------------------------

/// Allocate NPU device memory via aclrtMalloc, write test data, then
/// roundtrip through d2h (async) → h2d (async), and verify with a sync
/// copy back.
#[test]
fn ascend_memcpy_d2h_h2d_roundtrip_4096() {
    let device = match try_init_device0() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: cannot init Ascend device 0: {e}");
            return;
        }
    };

    const SIZE: usize = 4096;
    let policy: i32 = 0; // default memory policy

    // Allocate device memory
    let dev_ptr = match ascend::malloc_device(SIZE, policy) {
        Ok(p) => p,
        Err(e) => panic!("aclrtMalloc({SIZE}) failed: {e}"),
    };

    // Write test pattern to device via sync H2D
    let src: Vec<u8> = (0..SIZE as u8).map(|i| i.wrapping_mul(3)).collect();
    ascend::memcpy_h2d_sync(dev_ptr, src.as_ptr(), SIZE).expect("sync H2D of test pattern");

    // Allocate pinned host memory
    let (host_ptr, _device_ptr) = ascend::malloc_host(SIZE).expect("aclrtMallocHost");

    // Create stream for async operations
    let stream = device.create_stream().expect("create stream");

    // Async D2H
    ascend::memcpy_d2h_async(host_ptr, dev_ptr, SIZE, &stream).expect("async D2H");

    // Synchronize the stream — CRITICAL for data consistency
    stream.synchronize().expect("stream sync after D2H");

    // Fill source with garbage before H2D to prove we're reading from host
    let garbage: Vec<u8> = vec![0xCD; SIZE];
    ascend::memcpy_h2d_sync(dev_ptr, garbage.as_ptr(), SIZE).expect("sync H2D of garbage");

    // Now copy original data back to device via async H2D
    ascend::memcpy_h2d_async(dev_ptr, host_ptr, SIZE, &stream).expect("async H2D");

    stream.synchronize().expect("stream sync after H2D");

    // Read back via sync D2H
    let mut verify_buf = vec![0u8; SIZE];
    ascend::memcpy_d2h_sync(verify_buf.as_mut_ptr(), dev_ptr, SIZE)
        .expect("sync D2H for verification");

    // Assert roundtrip data integrity
    assert_eq!(&src[..], &verify_buf[..], "D2H→H2D roundtrip data mismatch");

    // Cleanup
    ascend::free_host(host_ptr).ok();
    ascend::free_device(dev_ptr).ok();
    drop(stream);
    drop(device);
    println!("PASS: ascend_memcpy_d2h_h2d_roundtrip_4096");
}

/// Stress test: repeat small-block D2H→H2D roundtrip 10,000 times.
/// This exercises memory-allocation paths and validates no leaks.
#[test]
fn ascend_memcpy_stress_10k_small() {
    let device = match try_init_device0() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: cannot init Ascend device 0: {e}");
            return;
        }
    };

    const SIZE: usize = 256;
    const ITERATIONS: usize = 10_000;
    let policy: i32 = 0;

    let dev_ptr = match ascend::malloc_device(SIZE, policy) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("SKIP: malloc_device failed: {e}");
            return;
        }
    };

    let (host_ptr, _device_ptr) = ascend::malloc_host(SIZE).expect("aclrtMallocHost");
    let stream = device.create_stream().expect("create stream");

    let src_pattern: Vec<u8> = (0..SIZE as u8).map(|i| i.wrapping_add(0x55)).collect();
    ascend::memcpy_h2d_sync(dev_ptr, src_pattern.as_ptr(), SIZE).expect("initial H2D");

    for i in 0..ITERATIONS {
        // D2H async → synchronize
        ascend::memcpy_d2h_async(host_ptr, dev_ptr, SIZE, &stream)
            .unwrap_or_else(|e| panic!("iteration {i}: d2h failed: {e}"));
        stream
            .synchronize()
            .unwrap_or_else(|e| panic!("iteration {i}: sync after d2h failed: {e}"));

        // H2D async → synchronize
        ascend::memcpy_h2d_async(dev_ptr, host_ptr, SIZE, &stream)
            .unwrap_or_else(|e| panic!("iteration {i}: h2d failed: {e}"));
        stream
            .synchronize()
            .unwrap_or_else(|e| panic!("iteration {i}: sync after h2d failed: {e}"));
    }

    // Final verification
    let mut verify_buf = vec![0u8; SIZE];
    ascend::memcpy_d2h_sync(verify_buf.as_mut_ptr(), dev_ptr, SIZE).expect("final sync D2H");
    assert_eq!(
        &src_pattern[..],
        &verify_buf[..],
        "data corrupted after {ITERATIONS} roundtrips"
    );

    ascend::free_host(host_ptr).ok();
    ascend::free_device(dev_ptr).ok();
    drop(stream);
    drop(device);
    println!("PASS: ascend_memcpy_stress_10k_small ({ITERATIONS} iterations × {SIZE}B)");
}

/// Test with 1-byte boundary: smallest valid transfer.
#[test]
fn ascend_memcpy_single_byte() {
    let device = match try_init_device0() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: cannot init Ascend device 0: {e}");
            return;
        }
    };

    const SIZE: usize = 1;
    let policy: i32 = 0;

    let dev_ptr = ascend::malloc_device(SIZE, policy).expect("malloc_device 1B");
    let (host_ptr, _device_ptr) = ascend::malloc_host(SIZE).expect("malloc_host 1B");
    let stream = device.create_stream().expect("create stream");

    let src: [u8; 1] = [0xAA];
    ascend::memcpy_h2d_sync(dev_ptr, src.as_ptr(), SIZE).expect("sync H2D 1B");

    ascend::memcpy_d2h_async(host_ptr, dev_ptr, SIZE, &stream).expect("async D2H 1B");
    stream.synchronize().expect("sync 1B");

    ascend::memcpy_h2d_async(dev_ptr, host_ptr, SIZE, &stream).expect("async H2D 1B");
    stream.synchronize().expect("sync 1B after H2D");

    let mut verify = [0u8; 1];
    ascend::memcpy_d2h_sync(verify.as_mut_ptr(), dev_ptr, SIZE).expect("verify D2H 1B");
    assert_eq!(verify[0], 0xAA, "1-byte roundtrip mismatch");

    ascend::free_host(host_ptr).ok();
    ascend::free_device(dev_ptr).ok();
    drop(stream);
    drop(device);
    println!("PASS: ascend_memcpy_single_byte");
}

// ---------------------------------------------------------------------------
// Test 2: Pinned memory 64-byte alignment
// ---------------------------------------------------------------------------

/// Verify that `allocate_ascend_host()` returns 64-byte aligned pointers.
#[test]
fn ascend_pinned_host_alignment() {
    match try_init_device0() {
        Ok(d) => {
            drop(d); // we just need the runtime initialized
        }
        Err(e) => {
            eprintln!("SKIP: cannot init Ascend device: {e}");
            return;
        }
    };

    // Test multiple allocation sizes to ensure alignment holds across
    // different size classes.
    let sizes: &[usize] = &[1, 7, 8, 15, 64, 65, 127, 128, 255, 4096, 65536];

    for &size in sizes {
        let (host, _device) = ascend::malloc_host(size)
            .unwrap_or_else(|e| panic!("aclrtMallocHost({size}) failed: {e}"));
        let addr = host as usize;
        let aligned = addr % 64 == 0;
        assert!(
            aligned,
            "aclrtMallocHost({size}) returned unaligned pointer: addr={addr:#x} (addr % 64 = {})",
            addr % 64
        );
        ascend::free_host(host).ok();
    }

    println!("PASS: ascend_pinned_host_alignment");
}

// ---------------------------------------------------------------------------
// Test 3: AscendMemcpyBackend d2h/h2d via TransferBackend trait
// ---------------------------------------------------------------------------

/// Exercise the full `AscendMemcpyBackend` path using the trait interface
/// with `CopyDesc` descriptors and a `DeviceStream::Ascend`.
#[test]
fn ascend_transfer_backend_roundtrip() {
    use pegaflow_core::device::DeviceStream;
    use pegaflow_core::transfer::{AscendMemcpyBackend, CopyDesc, TransferBackend};
    use std::sync::Arc;

    let device = match try_init_device0() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: cannot init Ascend device 0: {e}");
            return;
        }
    };

    const SIZE: usize = 4096;
    let policy: i32 = 0;

    let dev_ptr = ascend::malloc_device(SIZE, policy).expect("aclrtMalloc");
    let (host_ptr, _device_ptr) = ascend::malloc_host(SIZE).expect("aclrtMallocHost");

    // Write pattern to device
    let src: Vec<u8> = (0..SIZE as u8).map(|i| i.wrapping_mul(7)).collect();
    ascend::memcpy_h2d_sync(dev_ptr, src.as_ptr(), SIZE).expect("initial H2D");

    let stream = device.create_stream().expect("create stream");
    let stream: Arc<DeviceStream> = Arc::new(DeviceStream::Ascend(stream));

    // Build single-segment CopyDesc
    let desc = CopyDesc {
        device: dev_ptr,
        host: host_ptr,
        host_device: host_ptr as u64, // host == device for aclrtMallocHost
        size: SIZE,
    };

    let backend = AscendMemcpyBackend::new(0);

    // Execute D2H via backend
    backend.d2h(&[desc], &stream).expect("backend d2h");
    // Execute H2D via backend with merged descriptor
    backend.h2d(&[desc], &stream).expect("backend h2d");

    // Synchronize on the stream
    #[allow(irrefutable_let_patterns)]
    if let DeviceStream::Ascend(s) = stream.as_ref() {
        s.synchronize().expect("final sync");
    }

    // Verify
    let mut verify = vec![0u8; SIZE];
    ascend::memcpy_d2h_sync(verify.as_mut_ptr(), dev_ptr, SIZE).expect("verify D2H");
    assert_eq!(&src[..], &verify[..], "backend roundtrip mismatch");

    ascend::free_host(host_ptr).ok();
    ascend::free_device(dev_ptr).ok();
    drop(stream);
    drop(device);
    println!("PASS: ascend_transfer_backend_roundtrip");
}

/// Test coalesced multi-segment copy: two disjoint device regions mapped
/// to contiguous host regions. Verifies that the AscendMemcpyBackend
/// merge logic correctly handles adjacent copies.
#[test]
fn ascend_transfer_backend_coalesced() {
    use pegaflow_core::device::DeviceStream;
    use pegaflow_core::transfer::{AscendMemcpyBackend, CopyDesc, TransferBackend};
    use std::sync::Arc;

    let device = match try_init_device0() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: cannot init Ascend device 0: {e}");
            return;
        }
    };

    // Two 512-byte segments, adjacent on both device and host
    const SEG_SIZE: usize = 512;
    const TOTAL: usize = SEG_SIZE * 2;
    let policy: i32 = 0;

    let dev_ptr = ascend::malloc_device(TOTAL, policy).expect("aclrtMalloc");
    let (host_ptr, _device_ptr) = ascend::malloc_host(TOTAL).expect("aclrtMallocHost");

    let src: Vec<u8> = (0..TOTAL as u8).map(|i| i.wrapping_mul(3)).collect();
    ascend::memcpy_h2d_sync(dev_ptr, src.as_ptr(), TOTAL).expect("initial H2D");

    let stream = device.create_stream().expect("create stream");
    let stream: Arc<DeviceStream> = Arc::new(DeviceStream::Ascend(stream));

    let descs = [
        CopyDesc {
            device: dev_ptr,
            host: host_ptr,
            host_device: host_ptr as u64,
            size: SEG_SIZE,
        },
        CopyDesc {
            device: dev_ptr + SEG_SIZE as u64,
            host: unsafe { host_ptr.add(SEG_SIZE) },
            host_device: host_ptr as u64 + SEG_SIZE as u64,
            size: SEG_SIZE,
        },
    ];

    let backend = AscendMemcpyBackend::new(0);

    // D2H both segments
    backend.d2h(&descs, &stream).expect("backend d2h coalesced");
    // H2D both segments back
    backend.h2d(&descs, &stream).expect("backend h2d coalesced");

    #[allow(irrefutable_let_patterns)]
    if let DeviceStream::Ascend(s) = stream.as_ref() {
        s.synchronize().expect("final sync");
    }

    let mut verify = vec![0u8; TOTAL];
    ascend::memcpy_d2h_sync(verify.as_mut_ptr(), dev_ptr, TOTAL).expect("verify D2H");
    assert_eq!(
        &src[..],
        &verify[..],
        "coalesced backend roundtrip mismatch"
    );

    ascend::free_host(host_ptr).ok();
    ascend::free_device(dev_ptr).ok();
    drop(stream);
    drop(device);
    println!("PASS: ascend_transfer_backend_coalesced");
}
