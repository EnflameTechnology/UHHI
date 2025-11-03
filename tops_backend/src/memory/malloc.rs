use std::mem;
use std::os::raw::c_void;
use std::ptr;
pub use tops_raw as driv;
use uhal::memory::DevicePointerTrait;

pub use cust_core::_hidden::DeviceCopy;
pub use driv::topsStream_t;
use uhal::error::DeviceError;
use uhal::error::DeviceResult;
use uhal::stream::StreamTrait;

use super::TopsDevicePointer;
use crate::error::ToResult;
use crate::stream::TopsStream;

pub struct TopsMemory;

impl TopsMemory {
    // type DevicePointerT = TopsDevicePointer<T>;
    // type StreamT = CuStream;
    /// Unsafe wrapper around the `cuMemAlloc` function, which allocates some device memory and
    /// returns a [`DevicePointer`](struct.DevicePointer.html) pointing to it. The memory is not cleared.
    ///
    /// Note that `count` is in units of T; thus a `count` of 3 will allocate `3 * size_of::<T>()` bytes
    /// of memory.
    ///
    /// Memory buffers allocated using `malloc` must be freed using [`free`](fn.free.html).
    ///
    /// # Errors
    ///
    /// If allocating memory fails, returns the Device error value.
    /// If the number of bytes to allocate is zero (either because count is zero or because T is a
    /// zero-sized type), or if the size of the allocation would overflow a usize, returns InvalidValue.
    ///
    /// # Safety
    ///
    /// Since the allocated memory is not initialized, the caller must ensure that it is initialized
    /// before copying it to the host in any way. Additionally, the caller must ensure that the memory
    /// allocated is freed using free, or the memory will be leaked.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tops_backend as tops;
    /// use uhal::DriverLibraryTrait;
    /// let _context = tops::TopsApi::quick_init().unwrap();
    /// use tops::memory::*;
    /// unsafe {
    ///     let device_buffer = TopsMemory::malloc::<u64>(5).unwrap();
    ///     // Free allocated memory.
    ///     TopsMemory::free(device_buffer).unwrap();
    /// }
    /// ```
    pub unsafe fn malloc<T: DeviceCopy>(count: usize) -> DeviceResult<TopsDevicePointer<T>> {
        let size = count.checked_mul(mem::size_of::<T>()).unwrap_or(0);
        if size == 0 {
            return Err(DeviceError::InvalidMemoryAllocation);
        }

        let mut ptr = ptr::null_mut();
        driv::topsMalloc(&mut ptr as *mut *mut c_void, size).to_result()?;
        Ok(TopsDevicePointer::from_raw(ptr))
    }

    /// Unsafe wrapper around `cuMemAllocAsync` which queues a memory allocation operation on a stream.
    /// Retains all of the unsafe semantics of [`malloc`] with the extra requirement that the memory
    /// must not be used until it is allocated on the stream. Therefore, proper stream ordering semantics must be
    /// respected.
    ///
    /// # Safety
    ///
    /// The memory behind the returned pointer must not be used in any way until the
    /// allocation actually takes place in the stream.
    pub unsafe fn malloc_async<T: DeviceCopy>(
        stream: &TopsStream,
        count: usize,
    ) -> DeviceResult<TopsDevicePointer<T>> {
        let size = count.checked_mul(mem::size_of::<T>()).unwrap_or(0);
        if size == 0 {
            return Err(DeviceError::InvalidMemoryAllocation);
        }

        let mut ptr: *mut c_void = ptr::null_mut();

        driv::topsMallocAsync(
            &mut ptr as *mut *mut c_void,
            size,
            stream.as_inner(),
            0,
        )
        .to_result()?;
        let ptr = ptr as *mut T;
        Ok(TopsDevicePointer::from_raw(ptr as driv::topsDeviceptr_t))
    }

    /// Unsafe wrapper around `cuMemFreeAsync` which queues a memory allocation free operation on a stream.
    /// Retains all of the unsafe semantics of [`free`] with the extra requirement that the memory
    /// must not be used after it is dropped. Therefore, proper stream ordering semantics must be
    /// respected.
    ///
    /// # Safety
    ///
    /// The pointer must be valid.
    pub unsafe fn free_async<T: DeviceCopy>(
        stream: &TopsStream,
        p: TopsDevicePointer<T>,
    ) -> DeviceResult<()> {
        if mem::size_of::<T>() == 0 {
            return Err(DeviceError::InvalidMemoryAllocation);
        }
        driv::topsFreeAsync(p.as_raw(), stream.as_inner()).to_result()
    }

    /// Free memory allocated with [`malloc`](fn.malloc.html).
    ///
    /// # Errors
    ///
    /// If freeing memory fails, returns the Device error value. If the given pointer is null, returns
    /// InvalidValue.
    ///
    /// # Safety
    ///
    /// The given pointer must have been allocated with `malloc`, or null.
    /// The caller is responsible for ensuring that no other pointers to the deallocated buffer exist.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tops_backend as tops;
    /// let _context = tops::TopsApi::quick_init().unwrap();
    /// use tops::memory::*;
    /// use uhal::DriverLibraryTrait;
    /// use uhal::memory::MemoryTrait;
    /// unsafe {
    ///     let device_buffer = TopsMemory::malloc::<u64>(5).unwrap();
    ///     // Free allocated memory.
    ///     TopsMemory::free(device_buffer).unwrap();
    /// }
    /// ```
    pub unsafe fn free<T: DeviceCopy>(ptr: TopsDevicePointer<T>) -> DeviceResult<()> {
        if ptr.is_null() {
            return Err(DeviceError::InvalidMemoryAllocation);
        }

        return driv::topsFree(ptr.as_raw()).to_result();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use uhal::memory::MemoryTrait;
    use uhal::DriverLibraryTrait;

    #[derive(Clone, Copy, Debug)]
    struct ZeroSizedType;
    unsafe impl DeviceCopy for ZeroSizedType {}

    #[test]
    fn test_cuda_malloc() {
        let _device = crate::TopsApi::quick_init(0).unwrap();
        unsafe {
            let device_mem = TopsMemory::malloc::<usize>(1).unwrap();
            assert!(!device_mem.is_null());
            TopsMemory::free(device_mem).unwrap();
        }
    }

    #[test]
    fn test_cuda_malloc_zero_bytes() {
        let _device = crate::TopsApi::quick_init(0).unwrap();
        unsafe {
            assert_eq!(
                DeviceError::InvalidMemoryAllocation,
                TopsMemory::malloc::<usize>(0).unwrap_err()
            );
        }
    }

    #[test]
    fn test_cuda_malloc_zero_sized() {
        let _device = crate::TopsApi::quick_init(0).unwrap();
        unsafe {
            assert_eq!(
                DeviceError::InvalidMemoryAllocation,
                TopsMemory::malloc::<ZeroSizedType>(10).unwrap_err()
            );
        }
    }

    #[test]
    fn test_cuda_alloc_overflow() {
        let _device = crate::TopsApi::quick_init(0).unwrap();
        unsafe {
            assert_eq!(
                DeviceError::InvalidMemoryAllocation,
                TopsMemory::malloc::<usize>(::std::usize::MAX - 1).unwrap_err()
            );
        }
    }

    #[test]
    fn test_cuda_free_null() {
        let _device = crate::TopsApi::quick_init(0).unwrap();
        unsafe {
            assert_eq!(
                DeviceError::InvalidMemoryAllocation,
                TopsMemory::free(TopsDevicePointer::<usize>::null()).unwrap_err()
            );
        }
    }
}
