pub use cust_raw as driv;
use driv::CUdeviceptr;
use uhal::memory::{DeviceBoxTrait, DevicePointerTrait, MemoryTrait};
use uhal::error::{DeviceResult, DropResult};
// use uhal::stream::{Stream};

pub use cust_core::_hidden::{DeviceCopy};
use uhal::stream::StreamTrait;
use crate::error::ToResult;
use std::fmt::{self, Pointer};
use std::mem::{self, ManuallyDrop};

use std::os::raw::c_void;
use crate::driv::{CUstream};
use crate::memory::{CuDevicePointer, CuMemory};
use crate::stream::CuStream;

use super::{CopyDestination, AsyncCopyDestination};


#[derive(Debug)]
pub struct CuDeviceBox<T: DeviceCopy> {
    pub(crate) ptr: CuDevicePointer<T>,
}

unsafe impl<T: Send + DeviceCopy> Send for CuDeviceBox<T> {}
unsafe impl<T: Sync + DeviceCopy> Sync for CuDeviceBox<T> {}

impl<T: DeviceCopy + Default> CuDeviceBox<T> {
    /// Read the data back from the GPU into host memory.
    fn as_host_value(&self) -> DeviceResult<T> {
        let mut val = T::default();
        self.copy_to(&mut val)?;
        Ok(val)
    }
}

#[cfg(feature = "bytemuck")]
impl<T: DeviceCopy + bytemuck::Zeroable> CuDeviceBox<T>{
    // type DeviceBoxT = CuDeviceBox<T>;
    // type StreamT = CuStream;
    /// Allocate device memory and fill it with zeroes (`0u8`).
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::cuda_backend as cuda;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait, DeviceBoxTrait};
    /// use uhal::DriverLibraryTrait;
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::memory::*;
    /// let mut zero = CuDeviceBox::zeroed().unwrap();
    /// let mut value = 5u64;
    /// zero.copy_to(&mut value).unwrap();
    /// assert_eq!(0, value);
    /// ```
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    pub fn zeroed() -> DeviceResult<CuDeviceBox<T>>
    {
        unsafe {
            let new_box = CuDeviceBox::uninitialized()?;
            if mem::size_of::<T>() != 0 {
                driv::cuMemsetD8_v2(new_box.as_device_ptr().as_raw(), 0, mem::size_of::<T>())
                    .to_result()?;
            }
            Ok(new_box)
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
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait, DeviceBoxTrait};
    /// use uhal::stream::{StreamFlags, StreamTrait};
    /// use uhal::DriverLibraryTrait;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::{memory::*, stream::*};
    /// let stream = CuStream::new(StreamFlags::DEFAULT, None)?;
    /// let mut value = 5u64;
    /// unsafe {
    ///     let mut zero = CuDeviceBox::zeroed_async(&stream)?;
    ///     zero.async_copy_to(&mut value, &stream)?;
    ///     zero.drop_async(&stream)?;
    /// }
    /// stream.synchronize()?;
    /// assert_eq!(value, 0);
    /// # Ok(())
    /// # }
    /// ```
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    pub unsafe fn zeroed_async(stream: &CuStream) -> DeviceResult<CuDeviceBox<T>>
    {
        let new_box = CuDeviceBox::uninitialized_async(stream)?;
        if mem::size_of::<T>() != 0 {
            driv::cuMemsetD8Async(
                new_box.as_device_ptr().as_raw(),
                0,
                mem::size_of::<T>(),
                stream.as_inner(),
            )
            .to_result()?;
        }
        Ok(new_box)
    }
}


impl<T: DeviceCopy> DeviceBoxTrait<T> for CuDeviceBox<T> {
    type DeviceBoxT = CuDeviceBox<T>;
    type DevicePointerT = CuDevicePointer<T>;
    type RawDeviceT = CUdeviceptr;
    type StreamT = CuStream;
    /// Allocate device memory and place val into it.
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Errors
    ///
    /// If a Device error occurs, return the error.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::cuda_backend as cuda;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait, DeviceBoxTrait};
    /// use uhal::DriverLibraryTrait;
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::memory::*;
    /// let five = CuDeviceBox::new(&5).unwrap();
    /// ```
    fn new(val: &T) -> DeviceResult<Self::DeviceBoxT> {
        let mut dev_box = unsafe { CuDeviceBox::uninitialized()? };
        dev_box.copy_from(val)?;
        Ok(dev_box)
    }

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
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::cuda_backend as cuda;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait, DeviceBoxTrait};
    /// use uhal::DriverLibraryTrait;
    /// use uhal::stream::{StreamFlags, StreamTrait};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::{memory::*, stream::*};
    /// let stream = CuStream::new(StreamFlags::DEFAULT, None)?;
    /// let mut host_val = 0;
    /// unsafe {
    ///     let mut allocated = CuDeviceBox::new_async(&5u8, &stream)?;
    ///     allocated.async_copy_to(&mut host_val, &stream)?;
    ///     allocated.drop_async(&stream)?;
    /// }
    /// // ensure all async ops are done before trying to access the value
    /// stream.synchronize()?;
    /// assert_eq!(host_val, 5);
    /// # Ok(())
    /// # }
    unsafe fn new_async(val: &T, stream: &Self::StreamT) -> DeviceResult<Self::DeviceBoxT> {
        let mut dev_box = CuDeviceBox::uninitialized_async(stream)?;
        dev_box.async_copy_from(val, stream)?;
        Ok(dev_box)
    }

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
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::cuda_backend as cuda;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait, DeviceBoxTrait};
    /// use uhal::DriverLibraryTrait;
    /// use uhal::stream::{StreamFlags, StreamTrait};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::{memory::*, stream::*};
    /// let stream = CuStream::new(StreamFlags::DEFAULT, None)?;
    /// let mut host_val = 0;
    /// unsafe {
    ///     let mut allocated = CuDeviceBox::new_async(&5u8, &stream)?;
    ///     allocated.async_copy_to(&mut host_val, &stream)?;
    ///     allocated.drop_async(&stream)?;
    /// }
    /// // ensure all async ops are done before trying to access the value
    /// stream.synchronize()?;
    /// assert_eq!(host_val, 5);
    /// # Ok(())
    /// # }
    fn drop_async(self, stream: &Self::StreamT) -> DeviceResult<()> {
        if self.ptr.is_null() {
            return Ok(());
        }
        // make sure we dont run the normal destructor, otherwise a double drop will happen
        let me = ManuallyDrop::new(self);
        // SAFETY: we consume the box so its not possible to use the box past its drop point unless
        // you keep around a pointer, but in that case, we cannot guarantee safety.
        unsafe { CuMemory::free_async(stream, me.ptr) }
    }

    /// Allocate device memory, but do not initialize it.
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Safety
    ///
    /// Since the backing memory is not initialized, this function is not safe. The caller must
    /// ensure that the backing memory is set to a valid value before it is read, else undefined
    /// behavior may occur.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::cuda_backend as cuda;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait, DeviceBoxTrait};
    /// use uhal::DriverLibraryTrait;
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::memory::*;
    /// let mut five = unsafe { CuDeviceBox::uninitialized().unwrap() };
    /// five.copy_from(&5u64).unwrap();
    /// ```
    unsafe fn uninitialized() -> DeviceResult<Self::DeviceBoxT> {
        if mem::size_of::<T>() == 0 {
            Ok(CuDeviceBox {
                ptr: CuDevicePointer::null(),
            })
        } else {
            let ptr = CuMemory::malloc(1)?;
            Ok(CuDeviceBox { ptr })
        }
    }

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
    unsafe fn uninitialized_async(stream: &Self::StreamT) -> DeviceResult<Self::DeviceBoxT> {
        if mem::size_of::<T>() == 0 {
            Ok(CuDeviceBox {
                ptr: CuDevicePointer::null(),
            })
        } else {
            let ptr = CuMemory::malloc_async(stream, 1)?;
            Ok(CuDeviceBox { ptr })
        }
    }

    /// Constructs a DeviceBox from a raw pointer.
    ///
    /// After calling this function, the raw pointer and the memory it points to is owned by the
    /// DeviceBox. The DeviceBox destructor will free the allocated memory, but will not call the destructor
    /// of `T`. This function may accept any pointer produced by the `cuMemAllocManaged` Device API
    /// call.
    ///
    /// # Safety
    ///
    /// This function is unsafe because improper use may lead to memory problems. For example, a
    /// double free may occur if this function is called twice on the same pointer, or a segfault
    /// may occur if the pointer is not one returned by the appropriate API call.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::cuda_backend as cuda;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait, DeviceBoxTrait};
    /// use uhal::DriverLibraryTrait;
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::memory::*;
    /// let x = CuDeviceBox::<i32>::new(&5).unwrap();
    /// let ptr = CuDeviceBox::into_device(x).as_raw();
    /// let x = unsafe { CuDeviceBox::<i32>::from_raw(ptr) };
    /// ```
    unsafe fn from_raw(ptr: Self::RawDeviceT) -> Self::DeviceBoxT {
        CuDeviceBox {
            ptr: CuDevicePointer::from_raw(ptr),
        }
    }

    /// Constructs a DeviceBox from a DevicePointer.
    ///
    /// After calling this function, the pointer and the memory it points to is owned by the
    /// DeviceBox. The DeviceBox destructor will free the allocated memory, but will not call the destructor
    /// of `T`. This function may accept any pointer produced by the `cuMemAllocManaged` Device API
    /// call, such as one taken from `DeviceBox::into_device`.
    ///
    /// # Safety
    ///
    /// This function is unsafe because improper use may lead to memory problems. For example, a
    /// double free may occur if this function is called twice on the same pointer, or a segfault
    /// may occur if the pointer is not one returned by the appropriate API call.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::cuda_backend as cuda;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait, DeviceBoxTrait};
    /// use uhal::DriverLibraryTrait;
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::memory::*;
    /// let x = CuDeviceBox::<i32>::new(&5).unwrap();
    /// let ptr = CuDeviceBox::into_device(x);
    /// let x = unsafe { CuDeviceBox::<i32>::from_device(ptr) };
    /// ```
    unsafe fn from_device(ptr: Self::DevicePointerT) -> Self::DeviceBoxT {
        CuDeviceBox { ptr }
    }

    /// Consumes the DeviceBox, returning the wrapped DevicePointer.
    ///
    /// After calling this function, the caller is responsible for the memory previously managed by
    /// the DeviceBox. In particular, the caller should properly destroy T and deallocate the memory.
    /// The easiest way to do so is to create a new DeviceBox using the `DeviceBox::from_device` function.
    ///
    /// Note: This is an associated function, which means that you have to all it as
    /// `DeviceBox::into_device(b)` instead of `b.into_device()` This is so that there is no conflict with
    /// a method on the inner type.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::cuda_backend as cuda;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait, DeviceBoxTrait};
    /// use uhal::DriverLibraryTrait;
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::memory::*;
    /// let x = CuDeviceBox::<i32>::new(&5).unwrap();
    /// let ptr = CuDeviceBox::into_device(x);
    /// # unsafe { CuDeviceBox::<i32>::from_device(ptr) };
    /// ```
    #[allow(clippy::wrong_self_convention)]
    fn into_device(mut b: Self::DeviceBoxT) -> Self::DevicePointerT {
        let ptr = mem::replace(&mut b.ptr, CuDevicePointer::null());
        mem::forget(b);
        ptr
    }

    /// Returns the contained device pointer without consuming the box.
    ///
    /// This is useful for passing the box to a kernel launch.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::cuda_backend as cuda;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait, DeviceBoxTrait};
    /// use uhal::DriverLibraryTrait;
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::memory::*;
    /// let mut x = CuDeviceBox::<i32>::new(&5).unwrap();
    /// let ptr = x.as_device_ptr();
    /// println!("{:p}", ptr);
    /// ```
    fn as_device_ptr(&self) -> Self::DevicePointerT {
        self.ptr
    }

    /// Destroy a `DeviceBox`, returning an error.
    ///
    /// Deallocating device memory can return errors from previous asynchronous work. This function
    /// destroys the given box and returns the error and the un-destroyed box on failure.
    ///
    /// # Example
    ///
    /// ```
    /// # use crate::cuda_backend as cuda;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait, DeviceBoxTrait};
    /// use uhal::DriverLibraryTrait;
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::memory::*;
    /// let x = CuDeviceBox::<i32>::new(&5).unwrap();
    /// match CuDeviceBox::drop(x) {
    ///     Ok(()) => println!("Successfully destroyed"),
    ///     Err((e, dev_box)) => {
    ///         println!("Failed to destroy box: {:?}", e);
    ///         // Do something with dev_box
    ///     },
    /// }
    /// ```
    fn drop(mut dev_box: Self::DeviceBoxT) -> DropResult<Self::DeviceBoxT> {
        if dev_box.ptr.is_null() {
            return Ok(());
        }

        let ptr = mem::replace(&mut dev_box.ptr, CuDevicePointer::null());
        unsafe {
            match CuMemory::free(ptr) {
                Ok(()) => {
                    mem::forget(dev_box);
                    Ok(())
                }
                Err(e) => Err((e, CuDeviceBox { ptr })),
            }
        }
    }
}

// impl<T: DeviceCopy> Drop for CuDeviceBox<T> {
//     fn drop(&mut self) {
//         if self.ptr.is_null() {
//             return;
//         }

//         let ptr = mem::replace(&mut self.ptr, CuDevicePointer::null());
//         unsafe {
//             let _ = CuMemory::free(ptr);
//         }
//     }
// }

impl<T: DeviceCopy> Pointer for CuDeviceBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ptr = self.ptr.as_raw() as *const c_void;
        fmt::Pointer::fmt(&ptr, f)
    }
}

//CUDA Implementation
// impl<T: DeviceCopy> crate::memory::private::Sealed for DeviceBox<T> {}
impl<T: DeviceCopy> CopyDestination<T> for CuDeviceBox<T> {
    fn copy_from(&mut self, val: &T) -> DeviceResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                driv::cuMemcpyHtoD_v2(self.ptr.as_raw(), val as *const T as *const c_void, size)
                    .to_result()?
            }
        }
        Ok(())
    }

    fn copy_to(&self, val: &mut T) -> DeviceResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                driv::cuMemcpyDtoH_v2(
                    val as *const T as *mut c_void,
                    self.ptr.as_raw() as u64,
                    size,
                )
                .to_result()?
            }
        }
        Ok(())
    }
}

impl<T: DeviceCopy> CopyDestination<CuDeviceBox<T>> for CuDeviceBox<T> {
    fn copy_from(&mut self, val: &CuDeviceBox<T>) -> DeviceResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe { driv::cuMemcpyDtoD_v2(self.ptr.as_raw(), val.ptr.as_raw(), size).to_result()? }
        }
        Ok(())
    }

    fn copy_to(&self, val: &mut CuDeviceBox<T>) -> DeviceResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe { driv::cuMemcpyDtoD_v2(val.ptr.as_raw(), self.ptr.as_raw(), size).to_result()? }
        }
        Ok(())
    }
}

impl<T: DeviceCopy> AsyncCopyDestination<T> for CuDeviceBox<T> {
    type StreamT = CuStream;
    unsafe fn async_copy_from(&mut self, val: &T, stream: &Self::StreamT) -> DeviceResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            driv::cuMemcpyHtoDAsync_v2(
                self.ptr.as_raw(),
                val as *const _ as *const c_void,
                size,
                stream.as_inner(),
            )
            .to_result()?
        }
        Ok(())
    }

    unsafe fn async_copy_to(&self, val: &mut T, stream: &Self::StreamT) -> DeviceResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            driv::cuMemcpyDtoHAsync_v2(
                val as *mut _ as *mut c_void,
                self.ptr.as_raw() as u64,
                size,
                stream.as_inner(),
            )
            .to_result()?
        }
        Ok(())
    }
}

impl<T: DeviceCopy> AsyncCopyDestination<CuDeviceBox<T>> for CuDeviceBox<T> {
    type StreamT = CuStream;
    unsafe fn async_copy_from(&mut self, val: &CuDeviceBox<T>, stream: &Self::StreamT) -> DeviceResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            driv::cuMemcpyDtoDAsync_v2(self.ptr.as_raw(), val.ptr.as_raw(), size, stream.as_inner())
                .to_result()?
        }
        Ok(())
    }

    unsafe fn async_copy_to(&self, val: &mut CuDeviceBox<T>, stream: &Self::StreamT) -> DeviceResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            driv::cuMemcpyDtoDAsync_v2(val.ptr.as_raw(), self.ptr.as_raw(), size, stream.as_inner())
                .to_result()?
        }
        Ok(())
    }
}

