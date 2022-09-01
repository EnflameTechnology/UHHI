use cust_core::DeviceCopy;

use crate::error::*;
// use crate::memory::DevicePointer;
// use crate::prelude::Stream;

// pub struct MemoryManagement;
pub trait MemoryTrait<T> {
    type DevicePointerT;
    type StreamT;
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
    unsafe fn malloc<M: DeviceCopy>(count: usize) -> DeviceResult<Self::DevicePointerT>;

    /// Unsafe wrapper around `cuMemAllocAsync` which queues a memory allocation operation on a stream.
    /// Retains all of the unsafe semantics of [`malloc`] with the extra requirement that the memory
    /// must not be used until it is allocated on the stream. Therefore, proper stream ordering semantics must be
    /// respected.
    ///
    /// # Safety
    ///
    /// The memory behind the returned pointer must not be used in any way until the
    /// allocation actually takes place in the stream.
    unsafe fn malloc_async<M: DeviceCopy>(
        stream: &Self::StreamT,
        count: usize,
    ) -> DeviceResult<Self::DevicePointerT>;

    /// Unsafe wrapper around `cuMemFreeAsync` which queues a memory allocation free operation on a stream.
    /// Retains all of the unsafe semantics of [`free`] with the extra requirement that the memory
    /// must not be used after it is dropped. Therefore, proper stream ordering semantics must be
    /// respected.
    ///
    /// # Safety
    ///
    /// The pointer must be valid.
    unsafe fn free_async<M: DeviceCopy>(
        stream: &Self::StreamT,
        p: Self::DevicePointerT,
    ) -> DeviceResult<()>;

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
    unsafe fn free<M: DeviceCopy>(ptr: Self::DevicePointerT) -> DeviceResult<()>;
}
