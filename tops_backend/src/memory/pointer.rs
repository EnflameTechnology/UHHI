use core::{
    fmt::{self, Debug, Pointer},
    hash::Hash,
    ptr,
};
use std::{ffi::c_void, os::raw::c_ulonglong};
use std::marker::PhantomData;
use std::mem::size_of;
use uhal::memory::{DevicePointerTrait};
pub use cust_core::_hidden::{DeviceCopy};
pub use tops_raw as driv;
use driv::topsDeviceptr_t;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct TopsDevicePointer<T: ?Sized + DeviceCopy> {
    ptr: topsDeviceptr_t,
    marker: PhantomData<*mut T>,
}

unsafe impl<T: ?Sized + DeviceCopy> DeviceCopy for TopsDevicePointer<T> {}

impl<T: DeviceCopy> Pointer for TopsDevicePointer<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ptr = self.ptr as *const c_void;
        fmt::Pointer::fmt(&ptr, f)
    }
}

impl<T: DeviceCopy> DevicePointerTrait<T> for TopsDevicePointer<T>{
    type DevicePointerT = TopsDevicePointer<T>;
    type RawDevicePointerT = topsDeviceptr_t;
    /// Returns a rust [`pointer`] created from this pointer, meant for FFI purposes.
    /// **The pointer is not dereferenceable from the CPU!**
    fn as_ptr(&self) -> *const T
    {
        self.ptr as *const T
    }

    /// Returns a rust [`pointer`] created from this pointer, meant for FFI purposes.
    /// **The pointer is not dereferenceable from the CPU!**
    fn as_mut_ptr(&self) -> *mut T
    {
        self.ptr as *mut T
    }

    /// Returns the contained CUdeviceptr.
    fn as_raw(&self) -> Self::RawDevicePointerT
    {
        self.ptr
    }

    /// Create a DevicePointer from a raw Device pointer
    fn from_raw(ptr: Self::RawDevicePointerT) -> Self
    {
        Self {
            ptr,
            marker: PhantomData,
        }
    }

    /// Returns true if the pointer is null.
    /// # Examples
    ///
    /// ```
    /// # use cuda_backend as cuda;
    /// use uhal::DriverLibraryTrait;
    /// use uhal::memory::{DevicePointerTrait};
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::memory::*;
    /// use std::ptr;
    /// unsafe {
    ///     let null : *mut u64 = ptr::null_mut();
    ///     assert!(CuDevicePointer::wrap(null).is_null());
    /// }
    /// ```
    fn is_null(self) -> bool
    {
        self.ptr == ptr::null_mut()
    }

    /// Returns a null device pointer.
    ///
    // TODO (AL): do we even want this?
    fn null() -> Self
    where
        T: Sized
    {
        Self {
            ptr: ptr::null_mut(),
            marker: PhantomData,
        }
    }

    /// Calculates the offset from a device pointer.
    ///
    /// `count` is in units of T; eg. a `count` of 3 represents a pointer offset of
    /// `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined
    /// Behavior:
    ///
    /// * Both the starting and resulting pointer must be either in bounds or one
    ///   byte past the end of *the same* allocated object.
    ///
    /// * The computed offset, **in bytes**, cannot overflow an `isize`.
    ///
    /// * The offset being in bounds cannot rely on "wrapping around" the address
    ///   space. That is, the infinite-precision sum, **in bytes** must fit in a usize.
    ///
    /// Consider using `wrapping_offset` instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cuda_backend as cuda;
    /// use uhal::DriverLibraryTrait;
    /// use uhal::memory::{DevicePointerTrait};
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::memory::*;
    /// unsafe {
    ///     let mut dev_ptr = CuMemory::malloc::<u64>(5).unwrap();
    ///     let offset = dev_ptr.offset(1); // Points to the 2nd u64 in the buffer
    ///     CuMemory::free(dev_ptr); // Must free the buffer using the original pointer
    /// }
    /// ```
    unsafe fn offset(self, count: isize) -> Self
    where
        T: Sized
    {
        let mut ptr = self.ptr as u64 + (count as usize * size_of::<T>()) as u64;
        Self {
            ptr : ptr as *mut c_void,
            marker: PhantomData,
        }
    }

    /// Calculates the offset from a device pointer using wrapping arithmetic.
    ///
    /// `count` is in units of T; eg. a `count` of 3 represents a pointer offset of
    /// `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// The resulting pointer does not need to be in bounds, but it is
    /// potentially hazardous to dereference (which requires `unsafe`).
    /// In particular, the resulting pointer may *not* be used to access a
    /// different allocated object than the one `self` points to. In other
    /// words, `x.wrapping_offset(y.wrapping_offset_from(x))` is
    /// *not* the same as `y`, and dereferencing it is undefined behavior
    /// unless `x` and `y` point into the same allocated object.
    ///
    /// Always use `.offset(count)` instead when possible, because `offset`
    /// allows the compiler to optimize better.  If you need to cross object
    /// boundaries, cast the pointer to an integer and do the arithmetic there.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cuda_backend as cuda;
    /// use uhal::DriverLibraryTrait;
    /// use uhal::memory::{DevicePointerTrait};
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::memory::*;
    /// unsafe {
    ///     let mut dev_ptr = CuMemory::malloc::<u64>(5).unwrap();
    ///     let offset = dev_ptr.wrapping_offset(1); // Points to the 2nd u64 in the buffer
    ///     CuMemory::free(dev_ptr); // Must free the buffer using the original pointer
    /// }
    /// ```
    fn wrapping_offset(self, count: isize) -> Self
    where
        T: Sized
    {
        let ptr = self
            .ptr
            .wrapping_add((count as usize * size_of::<T>()) as usize);
        Self {
            ptr,
            marker: PhantomData,
        }
    }

    /// Calculates the offset from a pointer (convenience for `.offset(count as isize)`).
    ///
    /// `count` is in units of T; e.g. a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined
    /// Behavior:
    ///
    /// * Both the starting and resulting pointer must be either in bounds or one
    ///   byte past the end of an allocated object.
    ///
    /// * The computed offset, **in bytes**, cannot overflow an `isize`.
    ///
    /// * The offset being in bounds cannot rely on "wrapping around" the address
    ///   space. That is, the infinite-precision sum must fit in a `usize`.
    ///
    /// Consider using `wrapping_offset` instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cuda_backend as cuda;
    /// use uhal::DriverLibraryTrait;
    /// use uhal::memory::{DevicePointerTrait};
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::memory::*;
    /// unsafe {
    ///     let mut dev_ptr = CuMemory::malloc::<u64>(5).unwrap();
    ///     let offset = dev_ptr.add(1); // Points to the 2nd u64 in the buffer
    ///     CuMemory::free(dev_ptr); // Must free the buffer using the original pointer
    /// }
    /// ```
    #[allow(clippy::should_implement_trait)]
    unsafe fn add(self, count: usize) -> Self
    where
        T: Sized
    {
        self.offset(count as isize)
    }

    /// Calculates the offset from a pointer (convenience for
    /// `.offset((count as isize).wrapping_neg())`).
    ///
    /// `count` is in units of T; e.g. a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// If any of the following conditions are violated, the result is Undefined
    /// Behavior:
    ///
    /// * Both the starting and resulting pointer must be either in bounds or one
    ///   byte past the end of an allocated object.
    ///
    /// * The computed offset, **in bytes**, cannot overflow an `isize`.
    ///
    /// * The offset being in bounds cannot rely on "wrapping around" the address
    ///   space. That is, the infinite-precision sum must fit in a `usize`.
    ///
    /// Consider using `wrapping_offset` instead if these constraints are
    /// difficult to satisfy. The only advantage of this method is that it
    /// enables more aggressive compiler optimizations.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cuda_backend as cuda;
    /// use uhal::DriverLibraryTrait;
    /// use uhal::memory::{DevicePointerTrait};
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::memory::*;
    /// unsafe {
    ///     let mut dev_ptr = CuMemory::malloc::<u64>(5).unwrap();
    ///     let offset = dev_ptr.add(4).sub(3); // Points to the 2nd u64 in the buffer
    ///     CuMemory::free(dev_ptr); // Must free the buffer using the original pointer
    /// }
    #[allow(clippy::should_implement_trait)]
    unsafe fn sub(self, count: usize) -> Self
    where
        T: Sized
    {
        self.offset((count as isize).wrapping_neg())
    }

    /// Calculates the offset from a pointer using wrapping arithmetic.
    /// (convenience for `.wrapping_offset(count as isize)`)
    ///
    /// `count` is in units of T; e.g. a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// The resulting pointer does not need to be in bounds, but it is
    /// potentially hazardous to dereference.
    ///
    /// Always use `.add(count)` instead when possible, because `add`
    /// allows the compiler to optimize better.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cuda_backend as cuda;
    /// use uhal::DriverLibraryTrait;
    /// use uhal::memory::{DevicePointerTrait};
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::memory::*;
    /// unsafe {
    ///     let mut dev_ptr = CuMemory::malloc::<u64>(5).unwrap();
    ///     let offset = dev_ptr.wrapping_add(1); // Points to the 2nd u64 in the buffer
    ///     CuMemory::free(dev_ptr); // Must free the buffer using the original pointer
    /// }
    /// ```
    fn wrapping_add(self, count: usize) -> Self
    where
        T: Sized
    {
        self.wrapping_offset(count as isize)
    }

    /// Calculates the offset from a pointer using wrapping arithmetic.
    /// (convenience for `.wrapping_offset((count as isize).wrapping_sub())`)
    ///
    /// `count` is in units of T; e.g. a `count` of 3 represents a pointer
    /// offset of `3 * size_of::<T>()` bytes.
    ///
    /// # Safety
    ///
    /// The resulting pointer does not need to be in bounds, but it is
    /// potentially hazardous to dereference (which requires `unsafe`).
    ///
    /// Always use `.sub(count)` instead when possible, because `sub`
    /// allows the compiler to optimize better.
    ///
    /// # Examples
    ///
    /// ```
    /// # use cuda_backend as cuda;
    /// use uhal::DriverLibraryTrait;
    /// use uhal::memory::{DevicePointerTrait};
    /// # let _context = cuda::CuApi::quick_init().unwrap();
    /// use cuda::memory::*;
    /// unsafe {
    ///     let mut dev_ptr = CuMemory::malloc::<u64>(5).unwrap();
    ///     let offset = dev_ptr.wrapping_add(4).wrapping_sub(3); // Points to the 2nd u64 in the buffer
    ///     CuMemory::free(dev_ptr); // Must free the buffer using the original pointer
    /// }
    /// ```
    fn wrapping_sub(self, count: usize) -> Self
    where
        T: Sized
    {
        self.wrapping_offset((count as isize).wrapping_neg())
    }
}

impl<T: DeviceCopy> TopsDevicePointer<T> {
    /// Casts this device pointer to another type.
    pub fn cast<U: DeviceCopy>(self) -> TopsDevicePointer<U>
    {
        TopsDevicePointer::from_raw(self.ptr)
    }
}


