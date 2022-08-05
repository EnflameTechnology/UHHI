use crate::error::{DeviceResult, DropResult};
// use crate::stream::Stream;
// use crate::memory::DevicePointer;
use crate::memory::DeviceCopy;
use bytemuck::PodCastError;
/// A pointer type for heap-allocation in Device device memory.
///
/// See the [`module-level documentation`](../memory/index.html) for more information on device memory.

pub trait DeviceVariableTrait<T: DeviceCopy> {
    type DevicePointerT;
    type DeviceVariableT;
    /// Create a new `DeviceVariable` wrapping `var`.
    ///
    /// Allocates storage on the device and copies `var` to the device.
    fn new(var: T) -> DeviceResult<Self::DeviceVariableT>;

    /// Copy the host copy of the variable to the device
    fn copy_htod(&mut self) -> DeviceResult<()>;

    /// Copy the device copy of the variable to the host
    fn copy_dtoh(&mut self) -> DeviceResult<()>;

    fn as_device_ptr(&self) -> Self::DevicePointerT;
}


#[cfg(feature = "bytemuck")]
pub trait DeviceSliceTrait {
    type StreamT;
    type DevicePointerT;
    
    // NOTE(RDambrosio016): async memsets kind of blur the line between safe and unsafe, the only
    // unsafe thing i can imagine could happen is someone allocs a buffer, launches an async memset, then
    // tries to read back the value. However, it is unclear whether this is actually UB. Even if the
    // reads get jumbled into the writes, well, we know this type is Pod, so any byte value is fine for it.
    // So currently these functions are unsafe, but we may want to reevaluate this in the future.

    /// Sets the memory range of this buffer to contiguous `8-bit` values of `value`.
    ///
    /// In total it will set `sizeof<T> * len` values of `value` contiguously.
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    fn set_8(&mut self, value: u8) -> DeviceResult<()>;

    /// Sets the memory range of this buffer to contiguous `8-bit` values of `value` asynchronously.
    ///
    /// In total it will set `sizeof<T> * len` values of `value` contiguously.
    ///
    /// # Safety
    ///
    /// This operation is async so it does not complete immediately, it uses stream-ordering semantics.
    /// Therefore you should not read/write from/to the memory range until the operation is complete.
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    unsafe fn set_8_async(&mut self, value: u8, stream: &Self::StreamT) -> DeviceResult<()>;

    /// Sets the memory range of this buffer to contiguous `16-bit` values of `value`.
    ///
    /// In total it will set `(sizeof<T> / 2) * len` values of `value` contiguously.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `self.ptr % 2 != 0` (the pointer is not aligned to at least 2 bytes).
    /// - `(size_of::<T>() * self.len) % 2 != 0` (the data size is not a multiple of 2 bytes)
    #[track_caller]
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    fn set_16(&mut self, value: u16) -> DeviceResult<()>;

    /// Sets the memory range of this buffer to contiguous `16-bit` values of `value` asynchronously.
    ///
    /// In total it will set `(sizeof<T> / 2) * len` values of `value` contiguously.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `self.ptr % 2 != 0` (the pointer is not aligned to at least 2 bytes).
    /// - `(size_of::<T>() * self.len) % 2 != 0` (the data size is not a multiple of 2 bytes)
    ///
    /// # Safety
    ///
    /// This operation is async so it does not complete immediately, it uses stream-ordering semantics.
    /// Therefore you should not read/write from/to the memory range until the operation is complete.
    #[track_caller]
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    unsafe fn set_16_async(&mut self, value: u16, stream: &Self::StreamT) -> DeviceResult<()>;

    /// Sets the memory range of this buffer to contiguous `32-bit` values of `value`.
    ///
    /// In total it will set `(sizeof<T> / 4) * len` values of `value` contiguously.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `self.ptr % 4 != 0` (the pointer is not aligned to at least 4 bytes).
    /// - `(size_of::<T>() * self.len) % 4 != 0` (the data size is not a multiple of 4 bytes)
    #[track_caller]
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    fn set_32(&mut self, value: u32) -> DeviceResult<()>;

    /// Sets the memory range of this buffer to contiguous `32-bit` values of `value` asynchronously.
    ///
    /// In total it will set `(sizeof<T> / 4) * len` values of `value` contiguously.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `self.ptr % 4 != 0` (the pointer is not aligned to at least 4 bytes).
    /// - `(size_of::<T>() * self.len) % 4 != 0` (the data size is not a multiple of 4 bytes)
    ///
    /// # Safety
    ///
    /// This operation is async so it does not complete immediately, it uses stream-ordering semantics.
    /// Therefore you should not read/write from/to the memory range until the operation is complete.
    #[track_caller]
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    unsafe fn set_32_async(&mut self, value: u32, stream: &Self::StreamT) -> DeviceResult<()>;
}


pub trait DeviceBufferTrait<T: DeviceCopy> {
    type StreamT;
    type DevicePointerT;
    // type DeviceBufferPodT;
    type DeviceBufferT;
    type DeviceSliceT;

    /// Allocate a new device buffer large enough to hold `size` `T`'s, but without
    /// initializing the contents.
    ///
    /// # Errors
    ///
    /// If the allocation fails, returns the error from CUDA. If `size` is large enough that
    /// `size * mem::sizeof::<T>()` overflows usize, then returns InvalidMemoryAllocation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the contents of the buffer are initialized before reading from
    /// the buffer.
    unsafe fn uninitialized(size: usize) -> DeviceResult<Self::DeviceBufferT>;

    /// Allocates device memory asynchronously on a stream, without initializing it.
    ///
    /// This doesn't actually allocate if `T` is zero sized.
    ///
    /// # Safety
    ///
    /// The allocated memory retains all of the unsafety of [`DeviceBuffer::uninitialized`], with
    /// the additional consideration that the memory cannot be used until it is actually allocated
    /// on the stream. This means proper stream ordering semantics must be followed, such as
    /// only enqueing kernel launches that use the memory AFTER the allocation call.
    ///
    /// You can synchronize the stream to ensure the memory allocation operation is complete.
    unsafe fn uninitialized_async(size: usize, stream: &Self::StreamT) -> DeviceResult<Self::DeviceBufferT>;

    /// Enqueues an operation to free the memory backed by this [`DeviceBuffer`] on a
    /// particular stream. The stream will free the allocation as soon as it reaches
    /// the operation in the stream. You can ensure the memory is freed by synchronizing
    /// the stream.
    ///
    /// This function uses internal memory pool semantics. Async allocations will reserve memory
    /// in the default memory pool in the stream, and async frees will release the memory back to the pool
    /// for further use by async allocations.
    ///
    /// The memory inside of the pool is all freed back to the OS once the stream is synchronized unless
    /// a custom pool is configured to not do so.
    fn drop_async(self, stream: &Self::StreamT) -> DeviceResult<()>;
    /// Creates a `DeviceBuffer<T>` directly from the raw components of another device buffer.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren't
    /// checked:
    ///
    /// * `ptr` needs to have been previously allocated via `DeviceBuffer` or
    /// [`cuda_malloc`](fn.cuda_malloc.html).
    /// * `ptr`'s `T` needs to have the same size and alignment as it was allocated with.
    /// * `capacity` needs to be the capacity that the pointer was allocated with.
    ///
    /// Violating these may cause problems like corrupting the CUDA driver's
    /// internal data structures.
    ///
    /// The ownership of `ptr` is effectively transferred to the
    /// `DeviceBuffer<T>` which may then deallocate, reallocate or change the
    /// contents of memory pointed to by the pointer at will. Ensure
    /// that nothing else uses the pointer after calling this
    /// function.
    unsafe fn from_raw_parts(ptr: Self::DevicePointerT, capacity: usize) -> Self;

    /// Destroy a `DeviceBuffer`, returning an error.
    ///
    /// Deallocating device memory can return errors from previous asynchronous work. This function
    /// destroys the given buffer and returns the error and the un-destroyed buffer on failure.
    fn drop(dev_buf: Self::DeviceBufferT) -> DropResult<Self::DeviceBufferT>;

    /// Allocate a new device buffer of the same size as `slice`, initialized with a clone of
    /// the data in `slice`.
    ///
    /// # Errors
    ///
    /// If the allocation fails, returns the error from CUDA.
    fn from_slice(slice: &[T]) -> DeviceResult<Self::DeviceBufferT>;

    /// Asynchronously allocate a new buffer of the same size as `slice`, initialized
    /// with a clone of the data in `slice`.
    ///
    /// # Safety
    ///
    /// For why this function is unsafe, see [AsyncCopyDestination](trait.AsyncCopyDestination.html)
    ///
    /// # Errors
    ///
    /// If the allocation fails, returns the error from CUDA.
    unsafe fn from_slice_async(slice: &[T], stream: &Self::StreamT) -> DeviceResult<Self::DeviceBufferT>;

    /// Explicitly creates a [`DeviceSlice`] from this buffer.
    fn as_slice(&self) -> &Self::DeviceSliceT;
}


pub trait DeviceBoxTrait<T: DeviceCopy> {
    type DeviceBoxT;
    type StreamT;
    type RawDeviceT;
    type DevicePointerT;

    /// Allocate device memory and place val into it.
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Errors
    ///
    /// If a CUDA error occurs, return the error.
    fn new(val: &T) -> DeviceResult<Self::DeviceBoxT>;

    /// Allocates device memory asynchronously and asynchronously copies `val` into it.
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// If the memory behind `val` is not page-locked (pinned), a staging buffer
    /// will be allocated using a worker thread. If you are going to be making
    /// many asynchronous copies, it is generally a good idea to keep the data as a [`crate::memory::LockedBuffer`]
    /// or [`crate::memory::LockedBox`]. This will ensure the driver does not have to allocate a staging buffer
    /// on its own.
    ///
    /// However, don't keep all of your data as page-locked, doing so might slow down
    /// the OS because it is unable to page out that memory to disk.
    ///
    /// # Safety
    ///
    /// This method enqueues two operations on the stream: An async allocation
    /// and an async memcpy. Because of this, you must ensure that:
    /// - The memory is not used in any way before it is actually allocated on the stream. You
    /// can ensure this happens by synchronizing the stream explicitly or using events.
    /// - `val` is still valid when the memory copy actually takes place.
    unsafe fn new_async(val: &T, stream: &Self::StreamT) -> DeviceResult<Self::DeviceBoxT>;

    /// Enqueues an operation to free the memory backed by this [`DeviceBox`] on a
    /// particular stream. The stream will free the allocation as soon as it reaches
    /// the operation in the stream. You can ensure the memory is freed by synchronizing
    /// the stream.
    ///
    /// This function uses internal memory pool semantics. Async allocations will reserve memory
    /// in the default memory pool in the stream, and async frees will release the memory back to the pool
    /// for further use by async allocations.
    ///
    /// The memory inside of the pool is all freed back to the OS once the stream is synchronized unless
    /// a custom pool is configured to not do so.
    fn drop_async(self, stream: &Self::StreamT) -> DeviceResult<()>;

    /// Allocate device memory, but do not initialize it.
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Safety
    ///
    /// Since the backing memory is not initialized, this function is not safe. The caller must
    /// ensure that the backing memory is set to a valid value before it is read, else undefined
    /// behavior may occur.
    unsafe fn uninitialized() -> DeviceResult<Self::DeviceBoxT>;

    /// Allocates device memory asynchronously on a stream, without initializing it.
    ///
    /// This doesn't actually allocate if `T` is zero sized.
    ///
    /// # Safety
    ///
    /// The allocated memory retains all of the unsafety of [`DeviceBox::uninitialized`], with
    /// the additional consideration that the memory cannot be used until it is actually allocated
    /// on the stream. This means proper stream ordering semantics must be followed, such as
    /// only enqueing kernel launches that use the memory AFTER the allocation call.
    ///
    /// You can synchronize the stream to ensure the memory allocation operation is complete.
    unsafe fn uninitialized_async(stream: &Self::StreamT) -> DeviceResult<Self::DeviceBoxT>;

    /// Constructs a DeviceBox from a raw pointer.
    ///
    /// After calling this function, the raw pointer and the memory it points to is owned by the
    /// DeviceBox. The DeviceBox destructor will free the allocated memory, but will not call the destructor
    /// of `T`. This function may accept any pointer produced by the `cuMemAllocManaged` CUDA API
    /// call.
    ///
    /// # Safety
    ///
    /// This function is unsafe because improper use may lead to memory problems. For example, a
    /// double free may occur if this function is called twice on the same pointer, or a segfault
    /// may occur if the pointer is not one returned by the appropriate API call.
    unsafe fn from_raw(ptr: Self::RawDeviceT) -> Self::DeviceBoxT;

    /// Constructs a DeviceBox from a DevicePointer.
    ///
    /// After calling this function, the pointer and the memory it points to is owned by the
    /// DeviceBox. The DeviceBox destructor will free the allocated memory, but will not call the destructor
    /// of `T`. This function may accept any pointer produced by the `cuMemAllocManaged` CUDA API
    /// call, such as one taken from `DeviceBox::into_device`.
    ///
    /// # Safety
    ///
    /// This function is unsafe because improper use may lead to memory problems. For example, a
    /// double free may occur if this function is called twice on the same pointer, or a segfault
    /// may occur if the pointer is not one returned by the appropriate API call.
    unsafe fn from_device(ptr: Self::DevicePointerT) -> Self::DeviceBoxT;

    /// Consumes the DeviceBox, returning the wrapped DevicePointer.
    ///
    /// After calling this function, the caller is responsible for the memory previously managed by
    /// the DeviceBox. In particular, the caller should properly destroy T and deallocate the memory.
    /// The easiest way to do so is to create a new DeviceBox using the `DeviceBox::from_device` function.
    ///
    /// Note: This is an associated function, which means that you have to all it as
    /// `DeviceBox::into_device(b)` instead of `b.into_device()` This is so that there is no conflict with
    /// a method on the inner type.
    // #[allow(clippy::wrong_self_convention)]
    fn into_device(b: Self::DeviceBoxT) -> Self::DevicePointerT;

    /// Returns the contained device pointer without consuming the box.
    ///
    /// This is useful for passing the box to a kernel launch.
    fn as_device_ptr(&self) -> Self::DevicePointerT;

    /// Destroy a `DeviceBox`, returning an error.
    ///
    /// Deallocating device memory can return errors from previous asynchronous work. This function
    /// destroys the given box and returns the error and the un-destroyed box on failure.
    fn drop(dev_box: Self::DeviceBoxT) -> DropResult<Self::DeviceBoxT>;
}

// /// Sealed trait implemented by types which can be the source or destination when copying data
// /// to/from the device or from one device allocation to another.
// pub trait CopyDestination<O: ?Sized>: crate::private::Sealed {
//     /// Copy data from `source`. `source` must be the same size as `self`.
//     ///
//     /// # Errors
//     ///
//     /// If a Device error occurs, return the error.
//     fn copy_from(&mut self, source: &O) -> DeviceResult<()>;

//     /// Copy data to `dest`. `dest` must be the same size as `self`.
//     ///
//     /// # Errors
//     ///
//     /// If a Device error occurs, return the error.
//     fn copy_to(&self, dest: &mut O) -> DeviceResult<()>;
// }

// /// Sealed trait implemented by types which can be the source or destination when copying data
// /// asynchronously to/from the device or from one device allocation to another.
// ///
// /// # Safety
// ///
// /// The functions of this trait are unsafe because they return control to the calling code while
// /// the copy operation could still be occurring in the background. This could allow calling code
// /// to read, modify or deallocate the destination buffer, or to modify or deallocate the source
// /// buffer resulting in a data race and undefined behavior.
// ///
// /// Thus to enforce safety, the following invariants must be upheld:
// /// - The source and destination are not deallocated
// /// - The source is not modified
// /// - The destination is not written or read by any other operation
// ///
// /// These invariants must be preserved until the stream is synchronized or an event queued after
// /// the copy is triggered.
// ///
// pub trait AsyncCopyDestination<O: ?Sized>: crate::private::Sealed {
//     type StreamT;
//     /// Asynchronously copy data from `source`. `source` must be the same size as `self`.
//     ///
//     /// Host memory used as a source or destination must be page-locked.
//     ///
//     /// # Safety
//     ///
//     /// For why this function is unsafe, see [AsyncCopyDestination](trait.AsyncCopyDestination.html)
//     ///
//     /// # Errors
//     ///
//     /// If a Device error occurs, return the error.
//     unsafe fn async_copy_from(&mut self, source: &O, stream: &Self::StreamT) -> DeviceResult<()>;

//     /// Asynchronously copy data to `dest`. `dest` must be the same size as `self`.
//     ///
//     /// Host memory used as a source or destination must be page-locked.
//     ///
//     /// # Safety
//     ///
//     /// For why this function is unsafe, see [AsyncCopyDestination](trait.AsyncCopyDestination.html)
//     ///
//     /// # Errors
//     ///
//     /// If a Device error occurs, return the error.
//     unsafe fn async_copy_to(&self, dest: &mut O, stream: &Self::StreamT) -> DeviceResult<()>;
// }
