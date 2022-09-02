//! External memory and synchronization resources
use std::ptr::{null, self};

pub use tops_raw as driv;
use driv::{topsExternalMemory_t, topsDeviceptr_t};
pub use cust_core::_hidden::{DeviceCopy};
use uhal::external::{ExternalMemoryTrait};

use uhal::error::{DeviceResult};
use uhal::memory::DevicePointerTrait;

use crate::memory::TopsDevicePointer;
pub struct TopsExternalMemory(topsExternalMemory_t);
use crate::error::ToResult;

impl ExternalMemoryTrait for TopsExternalMemory{
    type ExternalMemoryT = TopsExternalMemory;
    // type DevicePointerT = CuDevicePointer<>;
    // type DevicePointerT = CuDevicePointer<T: DeviceCopy>;
    // Import an external memory referenced by `fd` with `size`
    #[allow(clippy::missing_safety_doc)]
    unsafe fn import(fd: i32, size: usize) -> DeviceResult<Self::ExternalMemoryT>{
        let desc = driv::topsExternalMemoryHandleDesc {
            type_: driv::topsExternalMemoryHandleType::topsExternalMemoryHandleTypeOpaqueFd,
            handle: driv::topsExternalMemoryHandleDesc_st__bindgen_ty_1 { fd },
            size: size as u64,
            flags: 0
        };

        let mut memory: driv::topsExternalMemory_t = std::ptr::null_mut();

        driv::topsImportExternalMemory(&mut memory, &desc)
            .to_result()
            .map(|_| TopsExternalMemory{0: memory})
    }

    #[allow(clippy::missing_safety_doc)]
    unsafe fn reimport(&mut self, fd: i32, size: usize) -> DeviceResult<()>{
        // import new memory - this will call drop to destroy the old one
        *self = Self::import(fd, size)?;

        Ok(())
    }

}

impl TopsExternalMemory {
    // Map a buffer from this memory with `size` and `offset`
    fn mapped_buffer<G: DeviceCopy>(
        &self,
        size_in_bytes: usize,
        offset_in_bytes: usize,
    ) -> DeviceResult<TopsDevicePointer<G>>
    {
        let buffer_desc = driv::topsExternalMemoryBufferDesc {
            offset: offset_in_bytes as u64,
            size: size_in_bytes as u64,
            flags: 0
        };

        let mut dptr:topsDeviceptr_t = ptr::null_mut();
        unsafe {
            driv::topsExternalMemoryGetMappedBuffer(&mut dptr, self.0, &buffer_desc)
                .to_result()
                .map(|_| TopsDevicePointer::from_raw(dptr))
        }
    }
}
impl Drop for TopsExternalMemory {
    fn drop(&mut self) {
        unsafe {
            driv::topsDestroyExternalMemory(self.0).to_result().unwrap();
        }
    }
}
