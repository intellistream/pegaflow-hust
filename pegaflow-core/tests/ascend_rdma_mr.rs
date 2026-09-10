#![cfg(all(feature = "ascend", feature = "rdma"))]

use std::ptr::NonNull;

use pegaflow_core::device::ascend;
use pegaflow_transfer::{MemoryRegion, TransferEngine};

/// Prove that the Ascend host allocation strategy is simultaneously usable by
/// ACL DMA and an RDMA HCA. This test is opt-in because CI hosts need not have
/// an HCA; set PEGAFLOW_TEST_RDMA_NIC to make failures fatal.
#[test]
fn ascend_registered_host_is_rdma_registerable() {
    let nic = match std::env::var("PEGAFLOW_TEST_RDMA_NIC") {
        Ok(nic) => nic,
        Err(_) => {
            eprintln!("SKIP: PEGAFLOW_TEST_RDMA_NIC is not set");
            return;
        }
    };
    ascend::ensure_acl_initialized().expect("initialize ACL");
    let device = ascend::AscendDevice::new(0).expect("Ascend device 0");
    device.set_current().expect("set Ascend device 0");

    const SIZE: usize = 4096;
    let host = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            SIZE,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    assert_ne!(host, libc::MAP_FAILED, "mmap failed");
    let host = host.cast::<u8>();
    unsafe { std::ptr::write_bytes(host, 0xA5, SIZE) };
    ascend::register_host(0, host, SIZE).expect("aclrtHostRegister");

    let engine = TransferEngine::new(&[nic], 2).expect("open RDMA HCA");
    let ptr = NonNull::new(host).expect("non-null mmap pointer");
    engine
        .register_memory(&[MemoryRegion { ptr, len: SIZE }])
        .expect("ibv_reg_mr for aclrtHostRegister mapping");
    engine.unregister_memory(&[ptr]).expect("unregister MR");

    // Engine pools are reference-counted and may be destroyed by a worker
    // thread with no current ACL device. Exercise that exact cleanup path.
    let host_addr = host as usize;
    std::thread::spawn(move || {
        ascend::unregister_host(0, host_addr as *mut u8)
            .expect("aclrtHostUnregister on a fresh thread");
    })
    .join()
    .expect("cleanup thread panicked");
    assert_eq!(unsafe { libc::munmap(host.cast(), SIZE) }, 0);
    drop(device);
}
