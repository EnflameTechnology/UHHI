pub use cust_raw as driv;
use driv::CUstream;
use uhal::error::{DeviceResult, DropResult};
use uhal::memory::{DeviceBufferTrait, DevicePointerTrait, MemoryTrait};
// use uhal::stream::{Stream};
use crate::error::ToResult;
#[cfg(feature = "bytemuck")]
pub use bytemuck;
#[cfg(feature = "bytemuck")]
use bytemuck::{Pod, PodCastError, Zeroable};
pub use cust_core::_hidden::DeviceCopy;
use std::mem::{self, align_of, size_of, transmute, ManuallyDrop};
use std::ops::{Deref, DerefMut};
use uhal::stream::StreamTrait;

use crate::memory::{CuDevicePointer, CuMemCpy, CuMemory};
use crate::stream::CuStream;

use super::{AsyncCopyDestination, CopyDestination, CuDeviceSlice};

/// Fixed-size device-side buffer. Provides basic access to device memory.
#[derive(Debug)]
#[repr(C)]
pub struct CuDeviceBuffer<T: DeviceCopy> {
    buf: CuDevicePointer<T>,
    len: usize,
}

unsafe impl<T: Send + DeviceCopy> Send for CuDeviceBuffer<T> {}
unsafe impl<T: Sync + DeviceCopy> Sync for CuDeviceBuffer<T> {}

impl<T: DeviceCopy> DeviceBufferTrait<T> for CuDeviceBuffer<T> {
    type DeviceBufferT = CuDeviceBuffer<T>;
    type DevicePointerT = CuDevicePointer<T>;
    type DeviceSliceT = CuDeviceSlice<T>;
    type StreamT = CuStream;
    // type DeviceBufferPodT = ;
    /// Allocate a new device buffer large enough to hold `size` `T`'s, but without
    /// initializing the contents.
    ///
    /// # Errors
    ///
    /// If the allocation fails, returns the error from Device. If `size` is large enough that
    /// `size * mem::sizeof::<T>()` overflows usize, then returns InvalidMemoryAllocation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the contents of the buffer are initialized before reading from
    /// the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::cuda_backend as cuda;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait};
    /// use uhal::DriverLibraryTrait;
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::memory::*;
    /// let mut buffer = unsafe { CuDeviceBuffer::uninitialized(5).unwrap() };
    /// buffer.copy_from(&[0u64, 1, 2, 3, 4]).unwrap();
    /// ```
    unsafe fn uninitialized(size: usize) -> DeviceResult<Self::DeviceBufferT> {
        let ptr = if size > 0 && size_of::<T>() > 0 {
            CuMemory::malloc(size)?
        } else {
            // FIXME (AL): Do we /really/ want to allow creating an invalid buffer?
            CuDevicePointer::null()
        };
        Ok(CuDeviceBuffer {
            buf: ptr,
            len: size,
        })
    }

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
    unsafe fn uninitialized_async(
        size: usize,
        stream: &Self::StreamT,
    ) -> DeviceResult<Self::DeviceBufferT> {
        let ptr = if size > 0 && size_of::<T>() > 0 {
            CuMemory::malloc_async(stream, size)?
        } else {
            CuDevicePointer::null()
        };
        Ok(CuDeviceBuffer {
            buf: ptr,
            len: size,
        })
    }

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
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::cuda_backend as cuda;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait};
    /// use uhal::DriverLibraryTrait;
    /// use uhal::stream::{StreamFlags, StreamTrait};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::{memory::*, stream::*};
    /// let stream = CuStream::new(StreamFlags::DEFAULT, None)?;
    /// let mut host_vals = [1, 2, 3];
    /// unsafe {
    ///     let mut allocated = CuDeviceBuffer::from_slice_async(&[4u8, 5, 6], &stream)?;
    ///     allocated.async_copy_to(&mut host_vals, &stream)?;
    ///     allocated.drop_async(&stream)?;
    /// }
    /// // ensure all async ops are done before trying to access the value
    /// stream.synchronize()?;
    /// assert_eq!(host_vals, [4, 5, 6]);
    /// # Ok(())
    /// # }
    /// ```
    fn drop_async(self, stream: &Self::StreamT) -> DeviceResult<()> {
        if self.buf.is_null() {
            return Ok(());
        }
        // make sure we dont run the normal destructor, otherwise a double drop will happen
        let me = ManuallyDrop::new(self);
        // SAFETY: we consume the box so its not possible to use the box past its drop point unless
        // you keep around a pointer, but in that case, we cannot guarantee safety.
        unsafe { CuMemory::free_async(stream, me.buf) }
    }

    /// Creates a `DeviceBuffer<T>` directly from the raw components of another device buffer.
    ///
    /// # Safety
    ///
    /// This is highly unsafe, due to the number of invariants that aren't
    /// checked:
    ///
    /// * `ptr` needs to have been previously allocated via `DeviceBuffer` or
    /// [`malloc`](fn.malloc.html).
    /// * `ptr`'s `T` needs to have the same size and alignment as it was allocated with.
    /// * `capacity` needs to be the capacity that the pointer was allocated with.
    ///
    /// Violating these may cause problems like corrupting the Device driver's
    /// internal data structures.
    ///
    /// The ownership of `ptr` is effectively transferred to the
    /// `DeviceBuffer<T>` which may then deallocate, reallocate or change the
    /// contents of memory pointed to by the pointer at will. Ensure
    /// that nothing else uses the pointer after calling this
    /// function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::cuda_backend as cuda;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait};
    /// use uhal::DriverLibraryTrait;
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use std::mem;
    /// use cuda::memory::*;
    ///
    /// let mut buffer = CuDeviceBuffer::from_slice(&[0u64; 5]).unwrap();
    /// let ptr = buffer.as_device_ptr();
    /// let size = buffer.len();
    ///
    /// mem::forget(buffer);
    ///
    /// let buffer = unsafe { CuDeviceBuffer::from_raw_parts(ptr, size) };
    /// ```
    unsafe fn from_raw_parts(ptr: Self::DevicePointerT, capacity: usize) -> Self::DeviceBufferT {
        CuDeviceBuffer {
            buf: ptr,
            len: capacity,
        }
    }

    /// Destroy a `DeviceBuffer`, returning an error.
    ///
    /// Deallocating device memory can return errors from previous asynchronous work. This function
    /// destroys the given buffer and returns the error and the un-destroyed buffer on failure.
    ///
    /// # Example
    ///
    /// ```
    /// # use crate::cuda_backend as cuda;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait};
    /// use uhal::DriverLibraryTrait;
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::memory::*;
    /// let x = CuDeviceBuffer::<u32>::from_slice(&[10, 20, 30]).unwrap();
    /// match CuDeviceBuffer::drop(x) {
    ///     Ok(()) => println!("Successfully destroyed"),
    ///     Err((e, buf)) => {
    ///         println!("Failed to destroy buffer: {:?}", e);
    ///         // Do something with buf
    ///     },
    /// }
    /// ```
    fn drop(mut dev_buf: Self::DeviceBufferT) -> DropResult<Self::DeviceBufferT> {
        if dev_buf.buf.is_null() {
            return Ok(());
        }

        if dev_buf.len > 0 && size_of::<T>() > 0 {
            let capacity = dev_buf.len;
            let ptr = mem::replace(&mut dev_buf.buf, CuDevicePointer::null());
            unsafe {
                match CuMemory::free(ptr) {
                    Ok(()) => {
                        mem::forget(dev_buf);
                        Ok(())
                    }
                    Err(e) => Err((e, CuDeviceBuffer::from_raw_parts(ptr, capacity))),
                }
            }
        } else {
            Ok(())
        }
    }

    /// Allocate a new device buffer of the same size as `slice`, initialized with a clone of
    /// the data in `slice`.
    ///
    /// # Errors
    ///
    /// If the allocation fails, returns the error from Device.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::cuda_backend as cuda;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait};
    /// use uhal::DriverLibraryTrait;
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::memory::*;
    /// let values = [0u64; 5];
    /// let mut buffer = CuDeviceBuffer::from_slice(&values).unwrap();
    /// ```
    fn from_slice(slice: &[T]) -> DeviceResult<Self::DeviceBufferT> {
        unsafe {
            let mut uninit = CuDeviceBuffer::uninitialized(slice.len())?;
            uninit.copy_from(slice)?;
            Ok(uninit)
        }
    }

    /// Asynchronously allocate a new buffer of the same size as `slice`, initialized
    /// with a clone of the data in `slice`.
    ///
    /// # Safety
    ///
    /// For why this function is unsafe, see [AsyncCopyDestination](trait.AsyncCopyDestination.html)
    ///
    /// # Errors
    ///
    /// If the allocation fails, returns the error from Device.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::cuda_backend as cuda;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait};
    /// use uhal::DriverLibraryTrait;
    /// use uhal::stream::{StreamFlags, StreamTrait};
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::memory::*;
    /// use cuda::stream::{CuStream};
    ///
    /// let stream = CuStream::new(StreamFlags::NON_BLOCKING, None).unwrap();
    /// let values = [0u64; 5];
    /// unsafe {
    ///     let mut buffer = CuDeviceBuffer::from_slice_async(&values, &stream).unwrap();
    ///     stream.synchronize();
    ///     // Perform some operation on the buffer
    /// }
    /// ```
    unsafe fn from_slice_async(
        slice: &[T],
        stream: &Self::StreamT,
    ) -> DeviceResult<Self::DeviceBufferT> {
        let mut uninit = CuDeviceBuffer::uninitialized_async(slice.len(), stream)?;
        uninit.async_copy_from(slice, stream)?;
        Ok(uninit)
    }

    /// Explicitly creates a [`DeviceSlice`] from this buffer.
    fn as_slice(&self) -> &Self::DeviceSliceT {
        self
    }
}

#[cfg(feature = "bytemuck")]
impl<T: DeviceCopy + Zeroable> CuDeviceBuffer<T> {
    // type DeviceBufferT = CuDeviceBuffer<T>;
    // type StreamT = CuStream;
    /// Allocate device memory and fill it with zeroes (`0u8`).
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::cuda_backend as cuda;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait};
    /// use uhal::DriverLibraryTrait;
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::memory::*;
    /// let mut zero = CuDeviceBuffer::zeroed(4).unwrap();
    /// let mut values = [1u8, 2, 3, 4];
    /// zero.copy_to(&mut values).unwrap();
    /// assert_eq!(values, [0; 4]);
    /// ```
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    pub fn zeroed(size: usize) -> DeviceResult<CuDeviceBuffer<T>> {
        unsafe {
            let new_buf = CuDeviceBuffer::uninitialized(size)?;
            if size_of::<T>() != 0 {
                driv::cuMemsetD8_v2(new_buf.as_device_ptr().as_raw(), 0, size_of::<T>() * size)
                    .to_result()?;
            }
            Ok(new_buf)
        }
    }

    /// Allocates device memory asynchronously and asynchronously fills it with zeroes (`0u8`).
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Safety
    ///
    /// This method enqueues two operations on the stream: An async allocation
    /// and an async memset. Because of this, you must ensure that:
    /// - The memory is not used in any way before it is actually allocated on the stream. You
    /// can ensure this happens by synchronizing the stream explicitly or using events.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::cuda_backend as cuda;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait};
    /// use uhal::DriverLibraryTrait;
    /// use uhal::stream::{StreamFlags, StreamTrait};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::{memory::*, stream::*};
    /// let stream = CuStream::new(StreamFlags::DEFAULT, None)?;
    /// let mut values = [1u8, 2, 3, 4];
    /// unsafe {
    ///     let mut zero = CuDeviceBuffer::zeroed_async(4, &stream)?;
    ///     zero.async_copy_to(&mut values, &stream)?;
    ///     zero.drop_async(&stream)?;
    /// }
    /// stream.synchronize()?;
    /// assert_eq!(values, [0; 4]);
    /// # Ok(())
    /// # }
    /// ```
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    pub unsafe fn zeroed_async(size: usize, stream: &CuStream) -> DeviceResult<CuDeviceBuffer<T>> {
        let new_buf = CuDeviceBuffer::uninitialized_async(size, stream)?;
        if size_of::<T>() != 0 {
            driv::cuMemsetD8Async(
                new_buf.as_device_ptr().as_raw(),
                0,
                size_of::<T>() * size,
                stream.as_inner(),
            )
            .to_result()?;
        }
        Ok(new_buf)
    }
}

#[cfg(feature = "bytemuck")]
fn casting_went_wrong(src: &str, err: PodCastError) -> ! {
    panic!("{}>{:?}", src, err);
}

#[cfg(feature = "bytemuck")]
impl<A: DeviceCopy + Pod> CuDeviceBuffer<A> {
    /// Same as [`DeviceBuffer::try_cast`] but panics if the cast fails.
    ///
    /// # Panics
    ///
    /// See [`DeviceBuffer::try_cast`].
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    pub fn cast<B: Pod + DeviceCopy>(self) -> CuDeviceBuffer<B> {
        match Self::try_cast(self) {
            Ok(b) => b,
            Err(e) => casting_went_wrong("cast", e),
        }
    }

    /// Tries to convert a [`DeviceBuffer`] of type `A` to a [`DeviceBuffer`] of type `B`. Returning
    /// an error if it failed.
    ///
    /// The length of the buffer after the conversion may have changed.
    ///
    /// # Failure
    ///
    /// - If the target type has a greater alignment requirement.
    /// - If the target element type is a different size and the output buffer wouldn't have a
    /// whole number of elements. Such as `3` x [`u16`] -> `1.5` x [`u32`].
    /// - If either type is a ZST (but not both).
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    pub fn try_cast<B: Pod + DeviceCopy>(self) -> Result<CuDeviceBuffer<B>, PodCastError> {
        if align_of::<B>() > align_of::<A>() && (self.buf.as_raw() as usize) % align_of::<B>() != 0
        {
            Err(PodCastError::TargetAlignmentGreaterAndInputNotAligned)
        } else if size_of::<B>() == size_of::<A>() {
            // SAFETY: we made sure sizes were compatible, and DeviceBuffer is repr(C)
            Ok(unsafe { transmute::<_, CuDeviceBuffer<B>>(self) })
        } else if size_of::<A>() == 0 || size_of::<B>() == 0 {
            Err(PodCastError::SizeMismatch)
        } else if (size_of::<A>() * self.len) % size_of::<B>() == 0 {
            let new_len = (size_of::<A>() * self.len) / size_of::<B>();
            Ok(CuDeviceBuffer::<B> {
                buf: self.buf.cast(),
                len: new_len,
            })
        } else {
            Err(PodCastError::OutputSliceWouldHaveSlop)
        }
    }
}

impl<T: DeviceCopy> Deref for CuDeviceBuffer<T> {
    type Target = CuDeviceSlice<T>;

    fn deref(&self) -> &CuDeviceSlice<T> {
        unsafe { &*(self as *const _ as *const CuDeviceSlice<T>) }
    }
}

impl<T: DeviceCopy> DerefMut for CuDeviceBuffer<T> {
    fn deref_mut(&mut self) -> &mut CuDeviceSlice<T> {
        unsafe { &mut *(self as *mut _ as *mut CuDeviceSlice<T>) }
    }
}

// impl<T: DeviceCopy> Drop for CuDeviceBuffer<T> {
//     fn drop(&mut self) {
//         if self.buf.is_null() {
//             return;
//         }

//         if self.len > 0 && size_of::<T>() > 0 {
//             let ptr = mem::replace(&mut self.buf, CuDevicePointer::null());
//             unsafe {
//                 let _ = CuMemory::free(ptr);
//             }
//         }
//         self.len = 0;
//     }
// }

#[cfg(test)]
mod test_device_buffer {
    use super::*;
    use crate::stream::CuStream;
    use uhal::memory::{DeviceBufferTrait, DevicePointerTrait};
    use uhal::stream::{StreamFlags, StreamTrait};
    use uhal::DriverLibraryTrait;

    #[derive(Clone, Copy, Debug)]
    struct ZeroSizedType;
    unsafe impl DeviceCopy for ZeroSizedType {}

    #[test]
    fn test_from_slice_drop() {
        let _context = crate::CuApi::quick_init().unwrap();
        let buf = CuDeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        drop(buf);
    }

    #[test]
    fn test_copy_to_from_device() {
        let _context = crate::CuApi::quick_init().unwrap();
        let start = [0u64, 1, 2, 3, 4, 5];
        let mut end = [0u64, 0, 0, 0, 0, 0];
        let buf = CuDeviceBuffer::from_slice(&start).unwrap();
        buf.copy_to(&mut end).unwrap();
        assert_eq!(start, end);
    }

    #[test]
    fn test_async_copy_to_from_device() {
        let _context = crate::CuApi::quick_init().unwrap();
        let stream = CuStream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        let start = [0u64, 1, 2, 3, 4, 5];
        let mut end = [0u64, 0, 0, 0, 0, 0];
        unsafe {
            let buf = CuDeviceBuffer::from_slice_async(&start, &stream).unwrap();
            buf.async_copy_to(&mut end, &stream).unwrap();
        }
        stream.synchronize().unwrap();
        assert_eq!(start, end);
    }

    #[test]
    #[should_panic]
    fn test_copy_to_d2h_wrong_size() {
        let _context = crate::CuApi::quick_init().unwrap();
        let buf = CuDeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        let mut end = [0u64, 1, 2, 3, 4];
        let _ = buf.copy_to(&mut end);
    }

    #[test]
    #[should_panic]
    fn test_async_copy_to_d2h_wrong_size() {
        let _context = crate::CuApi::quick_init().unwrap();
        let stream = CuStream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        unsafe {
            let buf = CuDeviceBuffer::from_slice_async(&[0u64, 1, 2, 3, 4, 5], &stream).unwrap();
            let mut end = [0u64, 1, 2, 3, 4];
            let _ = buf.async_copy_to(&mut end, &stream);
        }
    }

    #[test]
    #[should_panic]
    fn test_copy_from_h2d_wrong_size() {
        let _context = crate::CuApi::quick_init().unwrap();
        let start = [0u64, 1, 2, 3, 4];
        let mut buf = CuDeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        let _ = buf.copy_from(&start);
    }

    #[test]
    #[should_panic]
    fn test_async_copy_from_h2d_wrong_size() {
        let _context = crate::CuApi::quick_init().unwrap();
        let stream = CuStream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        let start = [0u64, 1, 2, 3, 4];
        unsafe {
            let mut buf =
                CuDeviceBuffer::from_slice_async(&[0u64, 1, 2, 3, 4, 5], &stream).unwrap();
            let _ = buf.async_copy_from(&start, &stream);
        }
    }

    #[test]
    #[should_panic]
    fn test_copy_to_d2d_wrong_size() {
        let _context = crate::CuApi::quick_init().unwrap();
        let buf = CuDeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        let mut end = CuDeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4]).unwrap();
        let _ = buf.copy_to(&mut end);
    }

    #[test]
    #[should_panic]
    fn test_async_copy_to_d2d_wrong_size() {
        let _context = crate::CuApi::quick_init().unwrap();
        let stream = CuStream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        unsafe {
            let buf = CuDeviceBuffer::from_slice_async(&[0u64, 1, 2, 3, 4, 5], &stream).unwrap();
            let mut end = CuDeviceBuffer::from_slice_async(&[0u64, 1, 2, 3, 4], &stream).unwrap();
            let _ = buf.async_copy_to(&mut end, &stream);
        }
    }

    #[test]
    #[should_panic]
    fn test_copy_from_d2d_wrong_size() {
        let _context = crate::CuApi::quick_init().unwrap();
        let mut buf = CuDeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
        let start = CuDeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4]).unwrap();
        let _ = buf.copy_from(&start);
    }

    #[test]
    #[should_panic]
    fn test_async_copy_from_d2d_wrong_size() {
        let _context = crate::CuApi::quick_init().unwrap();
        let stream = CuStream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        unsafe {
            let mut buf =
                CuDeviceBuffer::from_slice_async(&[0u64, 1, 2, 3, 4, 5], &stream).unwrap();
            let start = CuDeviceBuffer::from_slice_async(&[0u64, 1, 2, 3, 4], &stream).unwrap();
            let _ = buf.async_copy_from(&start, &stream);
        }
    }
}
