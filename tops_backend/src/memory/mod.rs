//! Access to Device's memory allocation and transfer functions.
//!
//! The memory module provides a safe wrapper around Device's memory allocation and transfer functions.
//! This includes access to device memory, unified memory, and page-locked host memory.
//!
//! # Device Memory
//!
//! Device memory is just what it sounds like - memory allocated on the device. Device memory
//! cannot be accessed from the host directly, but data can be copied to and from the device.
//! cust exposes device memory through the [`DeviceBox`](struct.DeviceBox.html) and
//! [`DeviceBuffer`](struct.DeviceBuffer.html) structures. Pointers to device memory are
//! represented by [`DevicePointer`](struct.DevicePointer.html), while slices in device memory are
//! represented by [`DeviceSlice`](struct.DeviceSlice.html).
//!
//!
//! ## Warning
//!
//! ⚠️ **On certain systems/OSes/GPUs, accessing Unified memory from the CPU while the GPU is currently
//! using it (e.g. before stream synchronization) will cause a Page Error/Segfault. For this reason,
//! we strongly suggest to treat unified memory as exclusive to the GPU while it is being used by a kernel** ⚠️
//!
//! This is not considered Undefined Behavior because the behavior is always "either works, or yields a page error/segfault",
//! doing this will never corrupt memory or cause other undesireable behavior.
//!
//!
//! # FFI Information
//!
//! The internal representations of `DevicePointer<T>` and `UnifiedPointer<T>` are guaranteed to be
//! the same as `*mut T` and they can be safely passed through an FFI boundary to code expecting
//! raw pointers (though keep in mind that device-only pointers cannot be dereferenced on the CPU).
//! This is important when launching kernels written in C.
//!
//! As with regular Rust, all other types (eg. `DeviceBuffer` or `UnifiedBox`) are not FFI-safe.
//! Their internal representations are not guaranteed to be anything in particular, and are not
//! guaranteed to be the same in different versions of cust. If you need to pass them through
//! an FFI boundary, you must convert them to FFI-safe primitives yourself. For example, with
//! `UnifiedBuffer`, use the `as_unified_ptr()` and `len()` functions to get the primitives, and
//! `mem::forget()` the Buffer so that it isn't dropped. Again, as with regular Rust, the caller is
//! responsible for reconstructing the `UnifiedBuffer` using `from_raw_parts()` and dropping it to
//! ensure that the memory allocation is safely cleaned up.

mod device;
mod malloc;
mod pointer;
pub use self::device::*;
pub use self::malloc::*;
pub use self::pointer::*;

pub use tops_raw as driv;
use std::ffi::c_void;
pub use cust_core::_hidden::{DeviceCopy};
pub use driv::{topsDeviceptr_t};
use uhal::error::{DeviceResult};
use uhal::memory::{MemCpyTrait};
use crate::error::ToResult;
pub struct TopsMemCpy;

impl MemCpyTrait for TopsMemCpy {
    type RawDevicePointerT = topsDeviceptr_t;
    #[allow(clippy::missing_safety_doc)]
    unsafe fn memcpy_htod(
        d_ptr: Self::RawDevicePointerT,
        src_ptr: *const c_void,
        size: usize,
    ) -> DeviceResult<()>
    {
        driv::topsMemcpyHtoD(d_ptr, src_ptr as *mut c_void, size as u64).to_result()?;
        Ok(())
    }
    
    /// Simple wrapper over MemcpyDtoH
    #[allow(clippy::missing_safety_doc)]
    unsafe fn memcpy_dtoh(
        d_ptr: *mut c_void,
        src_ptr: Self::RawDevicePointerT,
        size: usize,
    ) -> DeviceResult<()>
    {
        driv::topsMemcpyDtoH(d_ptr, src_ptr, size as u64).to_result()?;
        Ok(())
    }
    
    /// Get the current free and total memory.
    ///
    /// Returns in `.1` the total amount of memory available to the the current context.
    /// Returns in `.0` the amount of memory on the device that is free according to
    /// the OS. Device is not guaranteed to be able to allocate all of the memory that
    /// the OS reports as free.
    fn mem_get_info() -> DeviceResult<(u64, u64)>
    {
        let mut mem_free = 0;
        let mut mem_total = 0;
        unsafe {
            driv::topsMemGetInfo(&mut mem_free, &mut mem_total).to_result()?;
        }
        Ok((mem_free, mem_total))
    }
}

