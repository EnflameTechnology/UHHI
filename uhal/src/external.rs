//! External memory and synchronization resources

// use cust_core::DeviceCopy;

use crate::error::{DeviceResult};
// use crate::memory::{DevicePointer};
// pub use cust_core::_hidden::{DeviceCopy};

pub trait ExternalMemoryTrait {
    type ExternalMemoryT;
    // type DevicePointerT;
    // Import an external memory referenced by `fd` with `size`
    #[allow(clippy::missing_safety_doc)]
    unsafe fn import(fd: i32, size: usize) -> DeviceResult<Self::ExternalMemoryT>;

    #[allow(clippy::missing_safety_doc)]
    unsafe fn reimport(&mut self, fd: i32, size: usize) -> DeviceResult<()>;

    // Map a buffer from this memory with `size` and `offset`
    // fn mapped_buffer<G>(
    //     &self,
    //     size_in_bytes: usize,
    //     offset_in_bytes: usize,
    // ) -> Self::DevicePointerT
    // where G: ?Sized + DeviceCopy;
}