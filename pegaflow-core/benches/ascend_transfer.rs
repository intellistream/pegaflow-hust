//! Ascend NPU DMA transfer micro-benchmark.
//!
//! Exercises `aclrtMemcpyAsync` (H2D / D2H) throughput across a range of
//! block sizes with pinned host memory allocated via `aclrtMallocHost`.
//! Also measures pinned-host allocation latency for the sizes used in
//! production KV-cache save/load paths.
//!
//! Run (requires Ascend NPU hardware + CANN runtime):
//!   cargo bench -p pegaflow-core --no-default-features \
//!       --features ascend --bench ascend_transfer
//!
//! Optional flags:
//!   cargo bench -p pegaflow-core --bench ascend_transfer -- --alloc-mib 256
//!   cargo bench -p pegaflow-core --bench ascend_transfer -- --device 1

#[cfg(feature = "ascend")]
use std::time::Instant;

#[cfg(feature = "ascend")]
use pegaflow_core::device::ascend::{
    self, AscendDevice, AscendDeviceStream, ensure_acl_initialized,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Size of a single KV-cache block segment (bytes).
const SEG: usize = 4096;

/// Warmup iterations before each measurement.
const WARMUP: usize = 5;

/// Measurement iterations per data point.
const ITERS: usize = 50;

/// Block counts to sweep (maps to total transfer sizes).
const BLOCK_COUNTS: &[usize] = &[64, 256, 1024, 4096, 16384, 65536];

/// Allocation sizes (MiB) for pinned-memory latency benchmarks.
const ALLOC_SIZES_MIB: &[usize] = &[1, 4, 16, 64, 256, 1024];

/// Default host allocation size in bytes (256 MiB).
const DEFAULT_ALLOC_BYTES: usize = 256 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Direction {
    D2h,
    H2d,
}

impl Direction {
    fn name(self) -> &'static str {
        match self {
            Self::D2h => "D2H",
            Self::H2d => "H2D",
        }
    }
}

const DIRECTIONS: &[Direction] = &[Direction::H2d, Direction::D2h];

struct BenchConfig {
    device_id: i32,
    alloc_bytes: usize,
}

fn parse_config() -> BenchConfig {
    let mut device_id = 0i32;
    let mut alloc_bytes = DEFAULT_ALLOC_BYTES;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--device" => {
                device_id = args
                    .next()
                    .expect("--device requires an integer")
                    .parse()
                    .expect("--device must be an integer");
            }
            "--alloc-mib" => {
                let mib: usize = args
                    .next()
                    .expect("--alloc-mib requires an integer")
                    .parse()
                    .expect("--alloc-mib must be an integer");
                alloc_bytes = mib * 1024 * 1024;
            }
            "--bench" => {}
            "--help" | "-h" => {
                println!(
                    "usage: cargo bench -p pegaflow-core --bench ascend_transfer -- \
                     [--device N] [--alloc-mib N]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown argument: {other}"),
        }
    }

    BenchConfig {
        device_id,
        alloc_bytes,
    }
}

// ---------------------------------------------------------------------------
// Pinned host memory
// ---------------------------------------------------------------------------

struct PinnedHost {
    host_ptr: *mut u8,
    #[allow(dead_code)]
    device_ptr: u64,
    len: usize,
}

impl PinnedHost {
    fn alloc(device_id: i32, len: usize) -> Result<Self, String> {
        let (host_ptr, device_ptr) = ascend::malloc_host(device_id, len)?;
        Ok(Self {
            host_ptr,
            device_ptr: device_ptr as u64,
            len,
        })
    }

    fn fill_pattern(&self) {
        let slice = unsafe { std::slice::from_raw_parts_mut(self.host_ptr, self.len) };
        for (idx, byte) in slice.iter_mut().enumerate() {
            *byte = (idx.wrapping_mul(31).wrapping_add(7) & 0xFF) as u8;
        }
    }
}

impl Drop for PinnedHost {
    fn drop(&mut self) {
        if !self.host_ptr.is_null() {
            let _ = ascend::free_host(self.host_ptr);
        }
    }
}

// ---------------------------------------------------------------------------
// Device memory
// ---------------------------------------------------------------------------

struct DeviceMem {
    ptr: u64,
    #[allow(dead_code)]
    len: usize,
}

impl DeviceMem {
    fn alloc(len: usize) -> Result<Self, String> {
        let ptr = ascend::malloc_device(len, 0)?;
        Ok(Self { ptr, len })
    }
}

impl Drop for DeviceMem {
    fn drop(&mut self) {
        if self.ptr != 0 {
            let _ = ascend::free_device(self.ptr);
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark helpers
// ---------------------------------------------------------------------------

fn measure_transfer(
    direction: Direction,
    host: &PinnedHost,
    device: &DeviceMem,
    size: usize,
    stream: &AscendDeviceStream,
) -> (f64, f64) {
    // Warmup
    for _ in 0..WARMUP {
        match direction {
            Direction::H2d => {
                ascend::memcpy_h2d_async(device.ptr, host.host_ptr as *const u8, size, stream)
                    .expect("h2d warmup");
            }
            Direction::D2h => {
                ascend::memcpy_d2h_async(host.host_ptr, device.ptr, size, stream)
                    .expect("d2h warmup");
            }
        }
    }
    stream.synchronize().expect("sync after warmup");

    let start = Instant::now();
    for _ in 0..ITERS {
        match direction {
            Direction::H2d => {
                ascend::memcpy_h2d_async(device.ptr, host.host_ptr as *const u8, size, stream)
                    .expect("h2d bench");
            }
            Direction::D2h => {
                ascend::memcpy_d2h_async(host.host_ptr, device.ptr, size, stream)
                    .expect("d2h bench");
            }
        }
    }
    stream.synchronize().expect("sync after bench");

    let secs = start.elapsed().as_secs_f64();
    let avg_us = secs * 1e6 / ITERS as f64;
    let gibps = (size as f64 * ITERS as f64) / secs / (1024.0 * 1024.0 * 1024.0);
    (avg_us, gibps)
}

fn print_header() {
    println!(
        "{:>8}  {:>10}  {:>6}  {:>12}  {:>10}",
        "blocks", "total_kib", "dir", "avg_us", "gibps",
    );
}

fn print_row(blocks: usize, total_kib: usize, dir: &str, avg_us: f64, gibps: f64) {
    println!(
        "{:>8}  {:>10}  {:>6}  {:>12.1}  {:>10.2}",
        blocks, total_kib, dir, avg_us, gibps,
    );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[cfg(feature = "ascend")]
fn main() {
    let config = parse_config();

    // Initialize CANN runtime.
    ensure_acl_initialized().expect("aclInit");

    // Print version info.
    match ascend::get_acl_version() {
        Ok((maj, min, patch)) => println!("# CANN {maj}.{min}.{patch}"),
        Err(e) => eprintln!("# aclrtGetVersion: {e}"),
    }

    let device = AscendDevice::new(config.device_id).expect("AscendDevice");
    device.set_current().expect("aclrtSetDevice");
    let stream = device.create_stream().expect("create stream");

    println!(
        "# segment={SEG} B, warmup={WARMUP}, iters={ITERS}, device={}",
        config.device_id
    );
    println!("# host_alloc={} MiB\n", config.alloc_bytes / (1024 * 1024));

    // Allocate a max-sized host buffer once.
    let max_transfer_bytes = BLOCK_COUNTS.iter().copied().max().unwrap_or(65536) * SEG;
    let host_bytes = config.alloc_bytes.max(max_transfer_bytes);
    let host = PinnedHost::alloc(config.device_id, host_bytes).expect("pinned host alloc");
    host.fill_pattern();

    // -- DMA throughput benchmarks --
    println!("## DMA Throughput (aclrtMemcpyAsync)\n");
    print_header();

    for &blocks in BLOCK_COUNTS {
        let total_bytes = blocks * SEG;
        let total_kib = total_bytes / 1024;
        let device = DeviceMem::alloc(total_bytes).expect("device alloc");

        for &direction in DIRECTIONS {
            let (avg_us, gibps) = measure_transfer(direction, &host, &device, total_bytes, &stream);
            print_row(blocks, total_kib, direction.name(), avg_us, gibps);
        }
    }

    // -- Pinned memory allocation latency --
    println!("\n## Pinned Host Allocation Latency (aclrtMallocHost)\n");
    println!("{:>12}  {:>12}", "alloc_mib", "avg_us");

    for &mib in ALLOC_SIZES_MIB {
        let bytes = mib * 1024 * 1024;
        // Warmup
        for _ in 0..3 {
            let _h = PinnedHost::alloc(config.device_id, bytes).expect("alloc warmup");
        }

        let start = Instant::now();
        let alloc_iters = if mib <= 64 { 50 } else { 10 };
        for _ in 0..alloc_iters {
            let _h = PinnedHost::alloc(config.device_id, bytes).expect("alloc bench");
        }
        let avg_us = start.elapsed().as_secs_f64() * 1e6 / alloc_iters as f64;
        println!("{:>12}  {:>12.1}", mib, avg_us);
    }

    // -- Synchronous H2D fallback (for comparison) --
    println!("\n## Sync H2D Baseline (aclrtMemcpy, no stream)\n");
    print_header();

    for &blocks in BLOCK_COUNTS {
        let total_bytes = blocks * SEG;
        let total_kib = total_bytes / 1024;
        let device = DeviceMem::alloc(total_bytes).expect("device alloc");

        for _ in 0..WARMUP {
            ascend::memcpy_h2d_sync(device.ptr, host.host_ptr as *const u8, total_bytes)
                .expect("sync h2d warmup");
        }

        let start = Instant::now();
        for _ in 0..ITERS {
            ascend::memcpy_h2d_sync(device.ptr, host.host_ptr as *const u8, total_bytes)
                .expect("sync h2d bench");
        }
        let secs = start.elapsed().as_secs_f64();
        let avg_us = secs * 1e6 / ITERS as f64;
        let gibps = (total_bytes * ITERS) as f64 / secs / (1024.0 * 1024.0 * 1024.0);
        print_row(blocks, total_kib, "H2D(sync)", avg_us, gibps);
    }

    println!("\n# done");
}

#[cfg(not(feature = "ascend"))]
fn main() {
    eprintln!("This benchmark requires --features ascend and Ascend NPU hardware.");
    std::process::exit(1);
}
