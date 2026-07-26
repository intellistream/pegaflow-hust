mod engine;
mod error;
mod rc_backend;
pub mod rdma_topo;

#[cfg(not(feature = "ascend"))]
mod cuda_lib;
#[cfg(not(feature = "ascend"))]
mod cuda_sys;
#[cfg(not(feature = "ascend"))]
mod cudart_sys;
pub mod v2;

pub use engine::{
    ConnectionStatus, HandshakeMetadata, MemoryRegion, TransferDesc, TransferEngine, TransferOp,
};
pub use error::{Result, TransferError};

/// Device identifier types used across the transfer module.
///
/// Available unconditionally so v2 modules don't need to depend on feature-gated
/// `cuda_lib`. When the `ascend` feature is not active, `cuda_lib` provides its
/// own `Device` / `CudaDeviceId` types which are compatible.
pub mod device {
    use bincode::{Decode, Encode};
    use serde::{Deserialize, Serialize};

    /// CUDA device identifier.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Encode, Decode, Serialize, Deserialize)]
    pub struct CudaDeviceId(pub u8);

    /// Ascend NPU device identifier.
    #[cfg(feature = "ascend")]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Encode, Decode, Serialize, Deserialize)]
    pub struct AscendDeviceId(pub u8);

    /// Device type for memory region registration.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum Device {
        Host,
        Cuda(CudaDeviceId),
        #[cfg(feature = "ascend")]
        Ascend(AscendDeviceId),
    }
}

// Non-ascend: re-export from cuda_lib for backward compatibility.
#[cfg(not(feature = "ascend"))]
pub use cuda_lib::{CudaDeviceMemory, Device};

// Ascend: export device module types directly.
#[cfg(feature = "ascend")]
pub use device::{AscendDeviceId, CudaDeviceId, Device};

pub fn init_logging() {
    pegaflow_common::logging::init_stderr("info,pegaflow_transfer=debug");
}
