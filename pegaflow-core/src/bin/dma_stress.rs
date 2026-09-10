//! Standalone Ascend DMA stress test.
//! Mimics PegaFlow GPU worker's load path: many rapid small aclrtMemcpyAsync
//! calls on a single stream, then synchronize.
//!
//! Usage: cargo run --no-default-features --features ascend --bin dma_stress -- <device_id>
//!    Run WHILE vLLM is computing on the same NPU to reproduce 507001.

use std::env;
use std::sync::Arc;

use pegaflow_core::transfer::{AscendMemcpyBackend, CopyDesc, TransferBackend};

fn main() -> Result<(), String> {
    pegaflow_common::logging::init_stderr("info,pegaflow_core=debug");

    let args: Vec<String> = env::args().collect();
    let device_id: i32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(6);

    println!("=== Ascend DMA Stress Test ===");
    println!("Device: {device_id}");

    // Init CANN and create device/stream
    pegaflow_core::device::ascend::ensure_acl_initialized()?;

    let device = pegaflow_core::device::ascend::AscendDevice::new(device_id)?;
    device.set_current()?;
    println!("[init] aclrtSetDevice({device_id}) OK");

    let stream = Arc::new(pegaflow_core::device::DeviceStream::Ascend(
        device.create_stream()?,
    ));
    println!("[init] stream created");

    // Allocate host memory (matching real block sizes: 128KB)
    let block_size = 131072; // 128KB — real KV block size
    let num_blocks = 100;
    let total_bytes = block_size * num_blocks;

    let (host_ptr, _device_ptr) =
        pegaflow_core::device::ascend::malloc_host(device_id, total_bytes)?;
    println!(
        "[init] aclrtMallocHost({} MB) OK",
        total_bytes / 1024 / 1024
    );

    // Allocate device memory — malloc_device hardcodes aclrtSetDevice(0),
    // so re-set to our target device afterwards.
    let dev_ptr = pegaflow_core::device::ascend::malloc_device(total_bytes, 0)?;
    device.set_current()?;
    println!(
        "[init] aclrtMalloc({} MB) OK, dev_ptr=0x{dev_ptr:x}",
        total_bytes / 1024 / 1024
    );

    // ================================================================
    // Test 1: Single large H2D copy
    // ================================================================
    println!("\n--- Test 1: Single large H2D ({total_bytes} bytes) ---");
    let t0 = std::time::Instant::now();
    let ret = pegaflow_core::device::ascend::memcpy_h2d_async(
        dev_ptr,
        host_ptr,
        total_bytes,
        match stream.as_ref() {
            pegaflow_core::device::DeviceStream::Ascend(s) => s,
            _ => unreachable!(),
        },
    );
    let dur = t0.elapsed();
    match &ret {
        Ok(()) => println!("  OK  elapsed={:.2}ms", dur.as_secs_f64() * 1000.0),
        Err(e) => println!("  FAIL: {e}  elapsed={:.2}ms", dur.as_secs_f64() * 1000.0),
    }

    // ================================================================
    // Test 2: Skip (individual copies take too long, see batch tests below)
    // ================================================================
    println!("\n--- Test 2+3: SKIPPED ---");

    // ================================================================
    // Test 4: Use batch API (aclrtMemcpyBatchAsync) — same data, one call
    // ================================================================
    println!("\n--- Test 4: Batch API — 100 copies via aclrtMemcpyBatchAsync ---");
    let batch_copies: Vec<CopyDesc> = (0..100)
        .map(|i| {
            let idx = i % num_blocks;
            let host = unsafe { host_ptr.add(idx * block_size) };
            CopyDesc {
                device: dev_ptr + (idx * block_size) as u64,
                host,
                host_device: host as u64,
                size: block_size,
            }
        })
        .collect();
    let backend = AscendMemcpyBackend::new(device_id);
    let t0 = std::time::Instant::now();
    match backend.h2d(&batch_copies, &stream) {
        Ok(()) => println!(
            "  Batch submit OK  elapsed={:.2}ms",
            t0.elapsed().as_secs_f64() * 1000.0
        ),
        Err(e) => println!("  Batch submit FAIL: {e}"),
    }
    let ts = std::time::Instant::now();
    let sync_result = stream.synchronize();
    let sync_ms = ts.elapsed().as_secs_f64() * 1000.0;
    match &sync_result {
        Ok(()) => println!("  Batch sync OK  elapsed={sync_ms:.2}ms"),
        Err(e) => println!("  Batch sync FAIL: {e}  elapsed={sync_ms:.2}ms"),
    }

    // ================================================================
    // Test 5: Batch API — 1000 copies
    // ================================================================
    println!("\n--- Test 5: Batch API — 1000 copies via aclrtMemcpyBatchAsync ---");
    let batch_copies: Vec<CopyDesc> = (0..1000)
        .map(|i| {
            let idx = i % num_blocks;
            let host = unsafe { host_ptr.add(idx * block_size) };
            CopyDesc {
                device: dev_ptr + (idx * block_size) as u64,
                host,
                host_device: host as u64,
                size: block_size,
            }
        })
        .collect();
    let t0 = std::time::Instant::now();
    match backend.h2d(&batch_copies, &stream) {
        Ok(()) => println!(
            "  Batch submit OK  elapsed={:.2}ms",
            t0.elapsed().as_secs_f64() * 1000.0
        ),
        Err(e) => println!("  Batch submit FAIL: {e}"),
    }
    let ts = std::time::Instant::now();
    let sync_result = stream.synchronize();
    let sync_ms = ts.elapsed().as_secs_f64() * 1000.0;
    match &sync_result {
        Ok(()) => println!("  Batch sync OK  elapsed={sync_ms:.2}ms"),
        Err(e) => println!("  Batch sync FAIL: {e}  elapsed={sync_ms:.2}ms"),
    }

    // Cleanup
    pegaflow_core::device::ascend::free_device(dev_ptr)?;
    pegaflow_core::device::ascend::free_host(host_ptr)?;

    println!("\n=== Done ===");
    Ok(())
}
