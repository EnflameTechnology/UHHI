use core::{
    fmt::{self, Debug, Pointer},
    hash::Hash,
};
use std::marker::PhantomData;
pub type deviceptr = ::std::os::raw::c_ulonglong;

/// A pointer to device memory.
///
/// `DevicePointer` cannot be dereferenced by the CPU, as it is a pointer to a memory allocation in
/// the device. It can be safely copied to the device (eg. as part of a kernel launch) and either
/// unwrapped or transmuted to an appropriate pointer.
///
/// `DevicePointer` is guaranteed to have an equivalent internal representation to a raw pointer.
/// Thus, it can be safely reinterpreted or transmuted to `*mut T`. It is safe to pass a
/// `DevicePointer` through an FFI boundary to C code expecting a `*mut T`, so long as the code on
/// the other side of that boundary does not attempt to dereference the pointer on the CPU. It is
/// thus possible to pass a `DevicePointer` to a Device kernel written in C.
use crate::memory::DeviceCopy;

pub trait DevicePointerTrait<T: ?Sized + DeviceCopy> {
    type DevicePointerT;
    type RawDevicePointerT;
    /// Returns a rust [`pointer`] created from this pointer, meant for FFI purposes.
    /// **The pointer is not dereferenceable from the CPU!**
    fn as_ptr(&self) -> *const T;

    /// Returns a rust [`pointer`] created from this pointer, meant for FFI purposes.
    /// **The pointer is not dereferenceable from the CPU!**
    fn as_mut_ptr(&self) -> *mut T;

    fn as_mut(&mut self) -> &mut Self::RawDevicePointerT;
    fn as_ref(&self) -> &Self::RawDevicePointerT;
    /// Returns the contained CUdeviceptr.
    fn as_raw(&self) -> Self::RawDevicePointerT;

    /// Create a DevicePointer from a raw Device pointer
    fn from_raw(ptr: Self::RawDevicePointerT) -> Self;

    /// Returns true if the pointer is null.
    fn is_null(self) -> bool;

    /// Returns a null device pointer.
    ///
    // TODO (AL): do we even want this?
    fn null() -> Self
    where
        T: Sized;

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
    unsafe fn offset(self, count: isize) -> Self
    where
        T: Sized;

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
    fn wrapping_offset(self, count: isize) -> Self
    where
        T: Sized;

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
    #[allow(clippy::should_implement_trait)]
    unsafe fn add(self, count: usize) -> Self
    where
        T: Sized;

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
    #[allow(clippy::should_implement_trait)]
    unsafe fn sub(self, count: usize) -> Self
    where
        T: Sized;

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
    fn wrapping_add(self, count: usize) -> Self
    where
        T: Sized;

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
    fn wrapping_sub(self, count: usize) -> Self
    where
        T: Sized;

    // /// Casts this device pointer to another type.
    // fn cast<U>(self) -> Self
    // where U: ?Sized + DeviceCopy;
}
