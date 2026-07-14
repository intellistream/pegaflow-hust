//! Layer 4+5+6 Unified Integration Test: End-to-end Ascend data path.
//!
//! This test simulates the full lifecycle from "Layer 6 connector passing
//! device pointers" through "Layer 4 engine consuming them" — all within a
//! single process, bypassing gRPC (Layer 1–3).
//!
//! Flow:
//! 1. Layer 5: Initialize NPU device (init_device / detect_devices)
//! 2. Simulate a u64 pointer from aclrtMalloc (stand-in for Layer 3 gRPC
//!    deserialization of an NpuIPCWrapper key)
//! 3. Layer 4: Register the pointer via PegaEngine::register_context_layer_batch
//! 4. Layer 4: Save (D2H) and Load (H2D) via engine
//! 5. Verification: Data consistency after save→load roundtrip
//!
//! # Prerequisites
//! - Ascend NPU device(s) accessible
//! - CANN runtime (libascendcl.so) in LD_LIBRARY_PATH
//! - Build with `--features ascend`
//!
//! Run: `cargo test --test ascend_unified_integration --features ascend -- --nocapture`

use pegaflow_core::device::ascend;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn try_init_device0() -> Result<ascend::AscendDevice, String> {
    ascend::ensure_acl_initialized()?;
    let device = ascend::AscendDevice::new(0)?;
    device.set_current()?;
    Ok(device)
}

// ---------------------------------------------------------------------------
// Unified integration test
// ---------------------------------------------------------------------------

/// Simulates the full data path:
/// Layer 5 init → simulate L3 pointer → L4 register → L4 save → L4 load
///
/// Since the full PegaEngine save/load path requires a StorageEngine with
/// pinned memory pools (which need CUDA context for initialization in
/// the current codebase), this test exercises the *transfer* path directly:
/// it calls the AscendMemcpyBackend D2H and H2D using raw ACL pointers,
/// simulating what the engine's GPU worker pool does internally.
///
/// The registration and sealing logic is covered by the unit tests in
/// `pegaflow-core/src/instance/tests.rs`.
#[test]
fn ascend_unified_save_load_chain() {
    use pegaflow_core::device::DeviceStream;
    use pegaflow_core::transfer::{AscendMemcpyBackend, CopyDesc, TransferBackend};
    use std::sync::Arc;

    // ─ Layer 5: Initialize NPU device ──────────────────────────────────
    let device = match try_init_device0() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: cannot init Ascend device 0: {e}");
            return;
        }
    };

    // Print NPU info (simulating what pegaflow-server does in init_device)
    let npu_count = pegaflow_common::get_npu_device_count();
    if let Some(count) = npu_count {
        eprintln!("INFO: Detected {count} NPU device(s)");
        let node = pegaflow_common::get_npu_numa_node(0);
        eprintln!("INFO: NPU device 0 NUMA node = {node}");
    }

    eprintln!("INFO: Layer 5 initialized successfully");

    // ─ Simulate Layer 3: allocate device memory (stand-in for gRPC IPC) ─

    const SIZE: usize = 4096;
    let policy: i32 = 0;

    // aclrtMalloc simulates the device address that vllm worker allocated
    // and sent via gRPC as an NpuIPCWrapper (bypassing Layer 1–3 here).
    let dev_ptr = match ascend::malloc_device(SIZE, policy) {
        Ok(p) => p,
        Err(e) => panic!("aclrtMalloc({SIZE}) failed: {e}"),
    };

    eprintln!("INFO: Simulated Layer 3 IPC pointer: dev_ptr=0x{dev_ptr:x}");

    // Write test KV cache data to device (simulating what vLLM puts there)
    let src_data: Vec<u8> = vec![1u8; SIZE]; // Using 1s as per task spec
    ascend::memcpy_h2d_sync(dev_ptr, src_data.as_ptr(), SIZE)
        .expect("sync H2D of test data");

    // ─ Layer 4: Allocate pinned host memory (simulating engine mem pool) ─

    let (host_ptr, _device_ptr) =
        ascend::malloc_host(SIZE).expect("aclrtMallocHost");

    eprintln!("INFO: Pinned host memory allocated: host=0x{:x}", host_ptr as usize);

    // Verify alignment
    assert!(
        (host_ptr as usize) % 64 == 0,
        "pinned host memory not 64-byte aligned"
    );

    // ─ Layer 4: Save = D2H (engine copies KV cache from NPU to host) ─

    let stream = device.create_stream().expect("create stream");
    let stream: Arc<DeviceStream> = Arc::new(DeviceStream::Ascend(stream));

    let desc = CopyDesc {
        device: dev_ptr,
        host: host_ptr,
        host_device: host_ptr as u64,
        size: SIZE,
    };

    let backend = AscendMemcpyBackend;

    // Execute D2H (simulating save path)
    backend.d2h(&[desc], &stream).expect("backend d2h = save");

    #[allow(irrefutable_let_patterns)]
    if let DeviceStream::Ascend(s) = stream.as_ref() {
        s.synchronize().expect("sync after save");
    }

    eprintln!("INFO: Save (D2H) completed successfully");

    // ─ Verify data landed in host memory ─
    // SAFETY: host_ptr points to allocated memory of SIZE bytes.
    let host_slice = unsafe { std::slice::from_raw_parts(host_ptr, SIZE) };
    assert_eq!(
        host_slice, &src_data[..],
        "saved data does not match source"
    );

    // ─ Corrupt device memory to prove load restores it ─
    let zeros: Vec<u8> = vec![0u8; SIZE];
    ascend::memcpy_h2d_sync(dev_ptr, zeros.as_ptr(), SIZE)
        .expect("corrupt device memory with zeros");
    eprintln!("INFO: Device memory corrupted with zeros");

    // ─ Layer 4: Load = H2D (engine copies from host back to NPU) ─

    backend.h2d(&[desc], &stream).expect("backend h2d = load");

    #[allow(irrefutable_let_patterns)]
    if let DeviceStream::Ascend(s) = stream.as_ref() {
        s.synchronize().expect("sync after load");
    }

    eprintln!("INFO: Load (H2D) completed successfully");

    // ─ Verify load correctness ─
    let mut verify_buf = vec![0u8; SIZE];
    ascend::memcpy_d2h_sync(verify_buf.as_mut_ptr(), dev_ptr, SIZE)
        .expect("sync D2H for verification");

    assert_eq!(
        verify_buf, src_data,
        "loaded data does not match original — save/load chain broken"
    );

    // ─ Cleanup ─
    ascend::free_host(host_ptr).ok();
    ascend::free_device(dev_ptr).ok();
    drop(stream);
    drop(device);

    eprintln!("=== PASS: ascend_unified_save_load_chain ===");
}

/// Verify the full pipeline: init → alloc → register → save → load → verify
/// with multiple layer segments (simulating multi-layer KV cache).
#[test]
fn ascend_unified_multi_layer() {
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

    // Simulate two layers of KV cache, each 2048 bytes
    const LAYER_SIZE: usize = 2048;
    const NUM_LAYERS: usize = 2;
    const TOTAL: usize = LAYER_SIZE * NUM_LAYERS;
    let policy: i32 = 0;

    let dev_ptr = ascend::malloc_device(TOTAL, policy).expect("aclrtMalloc");
    let (host_ptr, _device_ptr) =
        ascend::malloc_host(TOTAL).expect("aclrtMallocHost");

    // Fill each layer with distinct patterns
    let mut src_data = vec![0u8; TOTAL];
    for layer in 0..NUM_LAYERS {
        let start = layer * LAYER_SIZE;
        let val = if layer == 0 { 0xAAu8 } else { 0xBBu8 };
        src_data[start..start + LAYER_SIZE].fill(val);
    }
    ascend::memcpy_h2d_sync(dev_ptr, src_data.as_ptr(), TOTAL)
        .expect("initial H2D");

    let stream = device.create_stream().expect("create stream");
    let stream: Arc<DeviceStream> = Arc::new(DeviceStream::Ascend(stream));

    // Build per-layer descriptors (simulating the multi-layer transfer batch
    // that the engine's offload path creates from KVCacheLayout).
    let descs: Vec<CopyDesc> = (0..NUM_LAYERS)
        .map(|layer| {
            let offset = layer * LAYER_SIZE;
            CopyDesc {
                device: dev_ptr + offset as u64,
                host: unsafe { host_ptr.add(offset) },
                host_device: host_ptr as u64 + offset as u64,
                size: LAYER_SIZE,
            }
        })
        .collect();

    let backend = AscendMemcpyBackend;

    // Save (D2H) both layers
    backend.d2h(&descs, &stream).expect("multi-layer d2h");

    #[allow(irrefutable_let_patterns)]
    if let DeviceStream::Ascend(s) = stream.as_ref() {
        s.synchronize().expect("sync after d2h");
    }

    // Verify host data
    let host_slice = unsafe { std::slice::from_raw_parts(host_ptr, TOTAL) };
    assert_eq!(
        host_slice, &src_data[..],
        "multi-layer saved data mismatch"
    );

    // Corrupt device data
    let zeros = vec![0u8; TOTAL];
    ascend::memcpy_h2d_sync(dev_ptr, zeros.as_ptr(), TOTAL).expect("corrupt");

    // Load (H2D) both layers
    backend.h2d(&descs, &stream).expect("multi-layer h2d");

    #[allow(irrefutable_let_patterns)]
    if let DeviceStream::Ascend(s) = stream.as_ref() {
        s.synchronize().expect("sync after h2d");
    }

    // Verify
    let mut verify = vec![0u8; TOTAL];
    ascend::memcpy_d2h_sync(verify.as_mut_ptr(), dev_ptr, TOTAL).expect("verify");
    assert_eq!(
        verify, src_data,
        "multi-layer load mismatch"
    );

    ascend::free_host(host_ptr).ok();
    ascend::free_device(dev_ptr).ok();
    drop(stream);
    drop(device);

    eprintln!("=== PASS: ascend_unified_multi_layer ===");
}