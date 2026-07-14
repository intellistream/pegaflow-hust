use std::{ffi::c_void, ptr::NonNull, sync::LazyLock};

use crate::device::{CudaDeviceId, Device};

use crate::v2::error::{FabricLibError, Result};

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum Mapping {
    Host,
    Device {
        device_id: CudaDeviceId,
        dmabuf_fd: Option<i32>,
    },
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct MemoryRegion {
    ptr: NonNull<c_void>,
    len: usize,
    mapping: Mapping,
}

impl MemoryRegion {
    pub fn new(ptr: NonNull<c_void>, len: usize, device: Device) -> Result<Self> {
        let mapping = match device {
            Device::Host => Mapping::Host,
            Device::Cuda(device_id) => {
                #[cfg(not(feature = "ascend"))]
                {
                    let attrs = crate::cuda_lib::rt::cudaPointerGetAttributes(ptr)?;
                    if attrs.type_ != crate::cuda_lib::rt::cudaMemoryTypeDevice {
                        return Err(FabricLibError::Custom("not a device pointer"));
                    }
                    let dmabuf_fd = if linux_kernel_supports_dma_buf() {
                        crate::cuda_lib::driver::cu_get_dma_buf_fd(ptr, len).ok()
                    } else {
                        None
                    };
                    Mapping::Device {
                        device_id,
                        dmabuf_fd,
                    }
                }
                #[cfg(feature = "ascend")]
                Mapping::Device {
                    device_id,
                    dmabuf_fd: None,
                }
            }
            #[cfg(feature = "ascend")]
            Device::Ascend(_device_id) => Mapping::Device {
                device_id: CudaDeviceId(0),
                dmabuf_fd: None,
            },
        };
        Ok(MemoryRegion { ptr, len, mapping })
    }

    pub fn ptr(&self) -> NonNull<c_void> {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn mapping(&self) -> &Mapping {
        &self.mapping
    }
}

impl Drop for MemoryRegion {
    fn drop(&mut self) {
        match self.mapping {
            Mapping::Host => {}
            Mapping::Device {
                dmabuf_fd: None, ..
            } => {}
            Mapping::Device {
                dmabuf_fd: Some(dmabuf_fd),
                ..
            } => unsafe {
                libc::close(dmabuf_fd);
            },
        }
    }
}

/// A local descriptor for a memory region.
/// For verbs, this is the MR LKEY.
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct MemoryRegionLocalDescriptor(pub u64);

static LINUX_KERNEL_SUPPORTS_DMA_BUF: LazyLock<bool> = LazyLock::new(|| {
    let Ok(version) = std::fs::read_to_string("/proc/sys/kernel/osrelease") else {
        return false;
    };
    let mut parts = version.split('.');
    let major: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let minor: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);

    (major, minor) >= (5, 12)
});

fn linux_kernel_supports_dma_buf() -> bool {
    *LINUX_KERNEL_SUPPORTS_DMA_BUF
}
