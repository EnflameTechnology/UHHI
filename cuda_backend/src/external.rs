//! External memory and synchronization resources
pub use cust_raw as driv;
use driv::{CUexternalMemory};
pub use cust_core::_hidden::{DeviceCopy};
use uhal::external::{ExternalMemoryTrait};
// use uhal::memory::{CuDevicePointer};
use uhal::error::{DeviceResult};
use uhal::memory::DevicePointerTrait;

use crate::memory::CuDevicePointer;
pub struct CuExternalMemory(CUexternalMemory);
use crate::error::ToResult;
// #[repr(transparent)]
// pub struct ExternalMemory<T>{
//     pub inner : T,
// }


impl ExternalMemoryTrait for CuExternalMemory{
    type ExternalMemoryT = CuExternalMemory;
    // type DevicePointerT = CuDevicePointer<>;
    // type DevicePointerT = CuDevicePointer<T: DeviceCopy>;
    // Import an external memory referenced by `fd` with `size`
    #[allow(clippy::missing_safety_doc)]
    unsafe fn import(fd: i32, size: usize) -> DeviceResult<Self::ExternalMemoryT>{
        let desc = driv::CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
            type_: driv::CUexternalMemoryHandleType_enum::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
            handle: driv::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1 { fd },
            size: size as u64,
            flags: 0,
            reserved: Default::default(),
        };

        let mut memory: driv::CUexternalMemory = std::ptr::null_mut();

        driv::cuImportExternalMemory(&mut memory, &desc)
            .to_result()
            .map(|_| CuExternalMemory{0: memory})
    }

    #[allow(clippy::missing_safety_doc)]
    unsafe fn reimport(&mut self, fd: i32, size: usize) -> DeviceResult<()>{
        // import new memory - this will call drop to destroy the old one
        *self = Self::import(fd, size)?;

        Ok(())
    }

}

impl CuExternalMemory {
    // Map a buffer from this memory with `size` and `offset`
    fn mapped_buffer<G: DeviceCopy>(
        &self,
        size_in_bytes: usize,
        offset_in_bytes: usize,
    ) -> DeviceResult<CuDevicePointer<G>>
    {
        let buffer_desc = driv::CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
            flags: 0,
            size: size_in_bytes as u64,
            offset: offset_in_bytes as u64,
            reserved: Default::default(),
        };

        let mut dptr = 0;
        unsafe {
            driv::cuExternalMemoryGetMappedBuffer(&mut dptr, self.0, &buffer_desc)
                .to_result()
                .map(|_| CuDevicePointer::from_raw(dptr))
        }
    }
}
impl Drop for CuExternalMemory {
    fn drop(&mut self) {
        unsafe {
            driv::cuDestroyExternalMemory(self.0).to_result().unwrap();
        }
    }
}
