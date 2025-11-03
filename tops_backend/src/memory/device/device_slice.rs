pub use tops_raw as driv;
use uhal::memory::DevicePointerTrait;
// use uhal::stream::{Stream};
pub use cust_core::_hidden::DeviceCopy;
use uhal::error::DeviceResult;
use uhal::stream::StreamTrait;
// use std::ops::{Deref, DerefMut};
#[cfg(feature = "bytemuck")]
use bytemuck::{Pod, Zeroable};
use driv::topsMemcpyKind;
// use std::mem::{self, size_of};
use std::mem;
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
use std::os::raw::c_void;

use super::{AsyncCopyDestination, CopyDestination, TopsDeviceBuffer};
use crate::error::ToResult;
use crate::memory::TopsDevicePointer;
use crate::stream::TopsStream;

/// Fixed-size device-side slice.
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct TopsDeviceSlice<T: DeviceCopy> {
    pub(crate) ptr: TopsDevicePointer<T>,
    len: usize,
}

unsafe impl<T: Send + DeviceCopy> Send for TopsDeviceSlice<T> {}
unsafe impl<T: Sync + DeviceCopy> Sync for TopsDeviceSlice<T> {}

impl<T: DeviceCopy + Default + Clone> TopsDeviceSlice<T> {
    pub fn as_host_vec(&self) -> DeviceResult<Vec<T>> {
        let mut vec = vec![T::default(); self.len()];
        self.copy_to(&mut vec)?;
        Ok(vec)
    }
}

// This works by faking a regular slice out of the device raw-pointer and the length and transmuting
// I have no idea if this is safe or not. Probably not, though I can't imagine how the compiler
// could possibly know that the pointer is not de-referenceable. I'm banking that we get proper
// Dynamicaly-sized Types before the compiler authors break this assumption.
impl<T: DeviceCopy> TopsDeviceSlice<T> {
    // type DevicePointerT = CuDevicePointer<T>;
    // type StreamT = CuStream;
    /// Returns the number of elements in the slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::tops_backend as tops;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait};
    /// use uhal::DriverLibraryTrait;
    /// # let _context = tops::TopsApi::quick_init().unwrap();
    /// use tops::memory::*;
    /// let a = TopsDeviceBuffer::from_slice(&[1, 2, 3]).unwrap();
    /// assert_eq!(a.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the slice has a length of 0.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::tops_backend as tops;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait};
    /// use uhal::DriverLibraryTrait;
    /// # let _context = tops::TopsApi::quick_init().unwrap();
    /// use tops::memory::*;
    /// let a : TopsDeviceBuffer<u64> = unsafe { TopsDeviceBuffer::uninitialized(0).unwrap() };
    /// assert!(a.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return a raw device-pointer to the slice's buffer.
    ///
    /// The caller must ensure that the slice outlives the pointer this function returns, or else
    /// it will end up pointing to garbage. The caller must also ensure that the pointer is not
    /// dereferenced by the CPU.
    ///
    /// Examples:
    ///
    /// ```
    /// # use crate::tops_backend as tops;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait};
    /// use uhal::DriverLibraryTrait;
    /// let _context = tops::TopsApi::quick_init().unwrap();
    /// use tops::memory::*;
    /// let a = TopsDeviceBuffer::<i32>::from_slice(&[1, 2, 3]).unwrap();
    /// println!("{:p}", a.as_device_ptr());
    /// ```
    pub fn as_device_ptr(&self) -> TopsDevicePointer<T> {
        self.ptr
    }

    pub fn as_device_ptr_mut(&mut self) -> &mut TopsDevicePointer<T> {
        &mut self.ptr
    }

    pub fn as_device_ptr_ref(&self) -> &TopsDevicePointer<T> {
        &self.ptr
    }
    /* TODO (AL): keep these?
    /// Divides one DeviceSlice into two at a given index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding the index `mid` itself) and
    /// the second will contain all indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// # Panics
    ///
    /// Panics if `min > len`.
    ///
    /// Examples:
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let buf = DeviceBuffer::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
    /// let (left, right) = buf.split_at(3);
    /// let mut left_host = [0u64, 0, 0];
    /// let mut right_host = [0u64, 0, 0];
    /// left.copy_to(&mut left_host).unwrap();
    /// right.copy_to(&mut right_host).unwrap();
    /// assert_eq!([0u64, 1, 2], left_host);
    /// assert_eq!([3u64, 4, 5], right_host);
    /// ```
    pub fn split_at(&self, mid: usize) -> (&DeviceSlice<T>, &DeviceSlice<T>) {
        let (left, right) = self.0.split_at(mid);
        unsafe {
            (
                DeviceSlice::from_slice(left),
                DeviceSlice::from_slice(right),
            )
        }
    }

    /// Divides one mutable DeviceSlice into two at a given index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding the index `mid` itself) and
    /// the second will contain all indices from `[mid, len)` (excluding the index `len` itself).
    ///
    /// # Panics
    ///
    /// Panics if `min > len`.
    ///
    /// Examples:
    ///
    /// ```
    /// # let _context = cust::quick_init().unwrap();
    /// use cust::memory::*;
    /// let mut buf = DeviceBuffer::from_slice(&[0u64, 0, 0, 0, 0, 0]).unwrap();
    ///
    /// {
    ///     let (left, right) = buf.split_at_mut(3);
    ///     let left_host = [0u64, 1, 2];
    ///     let right_host = [3u64, 4, 5];
    ///     left.copy_from(&left_host).unwrap();
    ///     right.copy_from(&right_host).unwrap();
    /// }
    ///
    /// let mut host_full = [0u64; 6];
    /// buf.copy_to(&mut host_full).unwrap();
    /// assert_eq!([0u64, 1, 2, 3, 4, 5], host_full);
    /// ```
    pub fn split_at_mut(&mut self, mid: usize) -> (&mut DeviceSlice<T>, &mut DeviceSlice<T>) {
        let (left, right) = self.0.split_at_mut(mid);
        unsafe {
            (
                DeviceSlice::from_slice_mut(left),
                DeviceSlice::from_slice_mut(right),
            )
        }
    }
    */

    /// Forms a slice from a `DevicePointer` and a length.
    ///
    /// The `len` argument is the number of _elements_, not the number of bytes.
    ///
    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given pointer is valid for `len`
    /// elements, nor whether the lifetime inferred is a suitable lifetime for the returned slice.
    ///
    /// # Caveat
    ///
    /// The lifetime for the returned slice is inferred from its usage. To prevent accidental misuse,
    /// it's suggested to tie the lifetime to whatever source lifetime is safe in the context, such
    /// as by providing a helper function taking the lifetime of a host value for the slice or
    /// by explicit annotation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::tops_backend as tops;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait};
    /// use uhal::DriverLibraryTrait;
    /// # let _context = tops::TopsApi::quick_init().unwrap();
    /// use tops::memory::*;
    /// let mut x = TopsDeviceBuffer::<u64>::from_slice(&[0u64, 1, 2, 3, 4, 5]).unwrap();
    /// // Manually slice the buffer (this is not recommended!)
    /// let ptr = unsafe { x.as_device_ptr().offset(1) };
    /// let slice = unsafe { TopsDeviceSlice::from_raw_parts(ptr, 2) };
    /// let mut host_buf = [0u64, 0];
    /// slice.copy_to(&mut host_buf).unwrap();
    /// assert_eq!([1u64, 2], host_buf);
    /// ```
    #[allow(clippy::needless_pass_by_value)]
    pub unsafe fn from_raw_parts(ptr: TopsDevicePointer<T>, len: usize) -> Self {
        TopsDeviceSlice { ptr, len }
    }

    /// Performs the same functionality as `from_raw_parts`, except that a
    /// mutable slice is returned.
    ///
    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given pointer is valid for `len`
    /// elements, nor whether the lifetime inferred is a suitable lifetime for the returned slice.
    ///
    /// This function is unsafe as there is no guarantee that the given pointer is valid for `len`
    /// elements, not whether the lifetime inferred is a suitable lifetime for the returned slice,
    /// as well as not being able to provide a non-aliasing guarantee of the returned
    /// mutable slice. `data` must be non-null and aligned even for zero-length
    /// slices as with `from_raw_parts`.
    ///
    /// See the documentation of `from_raw_parts` for more details.
    pub unsafe fn from_raw_parts_mut(ptr: TopsDevicePointer<T>, len: usize) -> Self {
        TopsDeviceSlice { ptr, len }
    }
}

#[cfg(feature = "bytemuck")]
impl<T: DeviceCopy + Pod> DeviceSliceTrait for TopsDeviceSlice<T> {
    type DevicePointerT = TopsDevicePointer<T>;
    type StreamT = TopsStream;
    // NOTE(RDambrosio016): async memsets kind of blur the line between safe and unsafe, the only
    // unsafe thing i can imagine could happen is someone allocs a buffer, launches an async memset, then
    // tries to read back the value. However, it is unclear whether this is actually UB. Even if the
    // reads get jumbled into the writes, well, we know this type is Pod, so any byte value is fine for it.
    // So currently these functions are unsafe, but we may want to reevaluate this in the future.

    /// Sets the memory range of this buffer to contiguous `8-bit` values of `value`.
    ///
    /// In total it will set `sizeof<T> * len` values of `value` contiguously.
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    fn set_8(&mut self, value: u8) -> DeviceResult<()> {
        if self.ptr.is_null() {
            return Ok(());
        }

        // SAFETY: We know T can hold any value because it is `Pod`, and
        // sub-byte alignment isn't a thing so we know the alignment is right.
        unsafe {
            driv::topsMemsetD8(self.ptr.as_raw(), value, size_of::<T>() * self.len).to_result()
        }
    }

    /// Sets the memory range of this buffer to contiguous `8-bit` values of `value` asynchronously.
    ///
    /// In total it will set `sizeof<T> * len` values of `value` contiguously.
    ///
    /// # Safety
    ///
    /// This operation is async so it does not complete immediately, it uses stream-ordering semantics.
    /// Therefore you should not read/write from/to the memory range until the operation is complete.
    #[cfg_attr(docsrs, doc(cfg(feature = "bytemuck")))]
    unsafe fn set_8_async(&mut self, value: u8, stream: &Self::StreamT) -> DeviceResult<()> {
        if self.ptr.is_null() {
            return Ok(());
        }

        driv::topsMemsetD8Async(
            self.ptr.as_raw(),
            value,
            size_of::<T>() * self.len,
            stream.as_inner(),
        )
        .to_result()
    }
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
    fn set_16(&mut self, value: u16) -> DeviceResult<()> {
        if self.ptr.is_null() {
            return Ok(());
        }
        let data_len = size_of::<T>() * self.len;
        assert_eq!(
            data_len % 2,
            0,
            "Buffer length is not a multiple of 2 bytes!"
        );
        assert_eq!(
            self.ptr.as_raw() % 2,
            0,
            "Buffer pointer is not aligned to at least 2 bytes!"
        );
        unsafe { driv::topsMemsetD16(self.ptr.as_raw(), value, data_len / 2).to_result() }
    }

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
    unsafe fn set_16_async(&mut self, value: u16, stream: &Self::StreamT) -> DeviceResult<()> {
        if self.ptr.is_null() {
            return Ok(());
        }
        let data_len = size_of::<T>() * self.len;
        assert_eq!(
            data_len % 2,
            0,
            "Buffer length is not a multiple of 2 bytes!"
        );
        assert_eq!(
            self.ptr.as_raw() % 2,
            0,
            "Buffer pointer is not aligned to at least 2 bytes!"
        );
        driv::topsMemsetD16Async(self.ptr.as_raw(), value, data_len / 2, stream.as_inner())
            .to_result()
    }

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
    fn set_32(&mut self, value: u32) -> DeviceResult<()> {
        if self.ptr.is_null() {
            return Ok(());
        }
        let data_len = size_of::<T>() * self.len;
        assert_eq!(
            data_len % 4,
            0,
            "Buffer length is not a multiple of 4 bytes!"
        );
        assert_eq!(
            self.ptr.as_raw() % 4,
            0,
            "Buffer pointer is not aligned to at least 4 bytes!"
        );
        unsafe { driv::topsMemsetD32(self.ptr.as_raw(), value, data_len / 4).to_result() }
    }

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
    unsafe fn set_32_async(&mut self, value: u32, stream: &Self::StreamT) -> DeviceResult<()> {
        if self.ptr.is_null() {
            return Ok(());
        }
        let data_len = size_of::<T>() * self.len;
        assert_eq!(
            data_len % 4,
            0,
            "Buffer length is not a multiple of 4 bytes!"
        );
        assert_eq!(
            self.ptr.as_raw() % 4,
            0,
            "Buffer pointer is not aligned to at least 4 bytes!"
        );
        unsafe { driv::topsMemsetD32(self.ptr.as_raw(), value, data_len / 4).to_result() }
    }
}

#[cfg(feature = "bytemuck")]
impl<T: DeviceCopy + Zeroable + Pod> TopsDeviceSlice<T> {
    // type DevicePointerT = TopsDevicePointer<T>;
    // type StreamT = TopsStream;
    /// Sets this slice's data to zero.
    fn set_zero(&mut self) -> DeviceResult<()> {
        if self.ptr.is_null() {
            return Ok(());
        }
        // SAFETY: this is fine because Zeroable guarantees a zero byte-pattern is safe
        // for this type. And a slice of bytes can represent any type.
        let mut erased = TopsDeviceSlice {
            ptr: self.ptr.cast::<u8>(),
            len: size_of::<T>() * self.len,
        };
        erased.set_8(0)
    }

    /// Sets this slice's data to zero asynchronously.
    ///
    /// # Safety
    ///
    /// This operation is async so it does not complete immediately, it uses stream-ordering semantics.
    /// Therefore you should not read/write from/to the memory range until the operation is complete.
    unsafe fn set_zero_async(&mut self, stream: &TopsStream) -> DeviceResult<()> {
        if self.ptr.is_null() {
            return Ok(());
        }
        // SAFETY: this is fine because Zeroable guarantees a zero byte-pattern is safe
        // for this type. And a slice of bytes can represent any type.
        let mut erased = TopsDeviceSlice {
            ptr: self.ptr.cast::<u8>(),
            len: size_of::<T>() * self.len,
        };
        erased.set_8_async(0, stream)
    }
}

#[inline(never)]
#[cold]
#[track_caller]
fn slice_start_index_len_fail(index: usize, len: usize) -> ! {
    panic!(
        "range start index {} out of range for slice of length {}",
        index, len
    );
}

#[inline(never)]
#[cold]
#[track_caller]
fn slice_end_index_len_fail(index: usize, len: usize) -> ! {
    panic!(
        "range end index {} out of range for slice of length {}",
        index, len
    );
}

#[inline(never)]
#[cold]
#[track_caller]
fn slice_index_order_fail(index: usize, end: usize) -> ! {
    panic!("slice index starts at {} but ends at {}", index, end);
}

#[inline(never)]
#[cold]
#[track_caller]
fn slice_end_index_overflow_fail() -> ! {
    panic!("attempted to index slice up to maximum usize");
}

pub trait DeviceSliceIndex<T: DeviceCopy> {
    // type DeviceSliceT;
    /// Indexes into this slice without checking if it is in-bounds.
    ///
    /// # Safety
    ///
    /// The range must be in-bounds of the slice.
    unsafe fn get_unchecked(self, slice: &TopsDeviceSlice<T>) -> TopsDeviceSlice<T>;
    fn index(self, slice: &TopsDeviceSlice<T>) -> TopsDeviceSlice<T>;
}

impl<T: DeviceCopy> DeviceSliceIndex<T> for usize {
    // type DeviceSliceT = TopsDeviceSlice<T>;

    unsafe fn get_unchecked(self, slice: &TopsDeviceSlice<T>) -> TopsDeviceSlice<T> {
        (self..self + 1).get_unchecked(slice)
    }
    fn index(self, slice: &TopsDeviceSlice<T>) -> TopsDeviceSlice<T> {
        slice.index(self..self + 1)
    }
}

impl<T: DeviceCopy> DeviceSliceIndex<T> for Range<usize> {
    // type DeviceSliceT = CuDeviceSlice<T>;

    unsafe fn get_unchecked(self, slice: &TopsDeviceSlice<T>) -> TopsDeviceSlice<T> {
        TopsDeviceSlice::from_raw_parts(
            slice.as_device_ptr().add(self.start),
            self.end - self.start,
        )
    }
    fn index(self, slice: &TopsDeviceSlice<T>) -> TopsDeviceSlice<T> {
        if self.start > self.end {
            slice_index_order_fail(self.start, self.end);
        } else if self.end > slice.len() {
            slice_end_index_len_fail(self.end, slice.len());
        }
        // SAFETY: `self` is checked to be valid and in bounds above.
        unsafe { self.get_unchecked(slice) }
    }
}

impl<T: DeviceCopy> DeviceSliceIndex<T> for RangeTo<usize> {
    // type DeviceSliceT = TopsDeviceSlice<T>;
    unsafe fn get_unchecked(self, slice: &TopsDeviceSlice<T>) -> TopsDeviceSlice<T> {
        (0..self.end).get_unchecked(slice)
    }
    fn index(self, slice: &TopsDeviceSlice<T>) -> TopsDeviceSlice<T> {
        (0..self.end).index(slice)
    }
}

impl<T: DeviceCopy> DeviceSliceIndex<T> for RangeFrom<usize> {
    // type DeviceSliceT = TopsDeviceSlice<T>;

    unsafe fn get_unchecked(self, slice: &TopsDeviceSlice<T>) -> TopsDeviceSlice<T> {
        (self.start..slice.len()).get_unchecked(slice)
    }
    fn index(self, slice: &TopsDeviceSlice<T>) -> TopsDeviceSlice<T> {
        if self.start > slice.len() {
            slice_start_index_len_fail(self.start, slice.len());
        }
        // SAFETY: `self` is checked to be valid and in bounds above.
        unsafe { self.get_unchecked(slice) }
    }
}

impl<T: DeviceCopy> DeviceSliceIndex<T> for RangeFull {
    // type DeviceSliceT = TopsDeviceSlice<T>;
    unsafe fn get_unchecked(self, slice: &TopsDeviceSlice<T>) -> TopsDeviceSlice<T> {
        *slice
    }
    fn index(self, slice: &TopsDeviceSlice<T>) -> TopsDeviceSlice<T> {
        *slice
    }
}

fn into_slice_range(range: RangeInclusive<usize>) -> Range<usize> {
    let exclusive_end = range.end() + 1;
    let start = if range.is_empty() {
        exclusive_end
    } else {
        *range.start()
    };
    start..exclusive_end
}

impl<T: DeviceCopy> DeviceSliceIndex<T> for RangeInclusive<usize> {
    // type DeviceSliceT = CuDeviceSlice<T>;
    unsafe fn get_unchecked(self, slice: &TopsDeviceSlice<T>) -> TopsDeviceSlice<T> {
        into_slice_range(self).get_unchecked(slice)
    }
    fn index(self, slice: &TopsDeviceSlice<T>) -> TopsDeviceSlice<T> {
        if *self.end() == usize::MAX {
            slice_end_index_overflow_fail();
        }
        into_slice_range(self).index(slice)
    }
}

impl<T: DeviceCopy> DeviceSliceIndex<T> for RangeToInclusive<usize> {
    // type DeviceSliceT = CuDeviceSlice<T>;
    unsafe fn get_unchecked(self, slice: &TopsDeviceSlice<T>) -> TopsDeviceSlice<T> {
        (0..=self.end).get_unchecked(slice)
    }
    fn index(self, slice: &TopsDeviceSlice<T>) -> TopsDeviceSlice<T> {
        (0..=self.end).index(slice)
    }
}

impl<T: DeviceCopy> TopsDeviceSlice<T> {
    pub fn index<Idx: DeviceSliceIndex<T>>(&self, idx: Idx) -> TopsDeviceSlice<T> {
        idx.index(self)
    }
}

//Tops Implementation
// impl<T: DeviceCopy> crate::memory::private::Sealed for TopsDeviceSlice<T> {}
impl<T: DeviceCopy, I: AsRef<[T]> + AsMut<[T]> + ?Sized> CopyDestination<I> for TopsDeviceSlice<T> {
    fn copy_from(&mut self, val: &I) -> DeviceResult<()> {
        let val = val.as_ref();
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = (mem::size_of::<T>() * self.len()) as usize;
        if size != 0 {
            unsafe {
                //Memcpy in tops is different from CUDA
                //Use HostAlloc function in tops to create a buffer for data transfer
                // if size % 4096 != 0 {
                //     // let mut ptr = alighed_alloc(size, 4096).unwrap();
                //     // std::ptr::copy(val.as_ptr() as *mut c_void, ptr.as_ptr() as *mut c_void, size);
                //     // let ret = driv::topsMemcpy(self.ptr.as_raw(), ptr.as_ptr() as *mut c_void, size, driv::topsMemcpyKind::topsMemcpyHostToDevice).to_result();
                //     // alighed_free(Some(ptr), size, 4096);
                //     // return ret;
                //     let mut ptr = std::ptr::null_mut();
                //     driv::topsHostMalloc(&mut ptr as *mut *mut c_void, size, 0);
                //     std::ptr::copy(val.as_ptr() as *mut c_void, ptr, size);
                //     let ret = driv::topsMemcpyHtoD(self.ptr.as_raw(), ptr as *mut c_void, size).to_result();
                //     driv::topsHostFree(ptr);
                //     return ret;
                // } else {
                return driv::topsMemcpyHtoD(self.ptr.as_raw(), val.as_ptr() as *mut c_void, size)
                    .to_result();
                // }
                // driv::topsDeviceSynchronize().to_result()?;;
                // driv::topsHostFree(ptr);
            }
        }
        Ok(())
    }

    fn copy_to(&self, val: &mut I) -> DeviceResult<()> {
        let val = val.as_mut();
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = (mem::size_of::<T>() * self.len()) as usize;
        if size != 0 {
            unsafe {
                driv::topsMemcpy(
                    val.as_mut_ptr() as *mut c_void,
                    self.as_device_ptr().as_raw(),
                    size,
                    topsMemcpyKind::topsMemcpyDeviceToHost,
                )
                .to_result()?
            }
        }
        Ok(())
    }
}

impl<T: DeviceCopy> TopsDeviceSlice<T> {
    pub fn copy_from_pointer<M>(&mut self, pointer: *const M, length: usize) -> DeviceResult<()> {
        let size = mem::size_of::<M>() * length;
        if size != 0 {
            unsafe {
                //Memcpy in tops is different from CUDA
                //Use HostAlloc function in tops to create a buffer for data transfer
                // let mut ptr = std::ptr::null_mut();
                // driv::topsHostMalloc(&mut ptr as *mut *mut c_void, size, 0);
                // std::ptr::copy(pointer as *mut c_void, ptr, size);
                // driv::topsMemcpyHtoD(self.ptr.as_raw(), ptr as *mut c_void, size)
                //     .to_result()?;
                return driv::topsMemcpyHtoD(
                    self.ptr.as_raw(),
                    pointer as *mut c_void,
                    size as usize,
                )
                .to_result();
                // if size % 4096 != 0 {
                //     let mut ptr = alighed_alloc(size, 4096).unwrap();
                //     std::ptr::copy(pointer as *mut c_void, ptr.as_ptr() as *mut c_void, size);
                //     let ret = driv::topsMemcpy(self.ptr.as_raw(), ptr.as_ptr() as *mut c_void, size as u64, driv::topsMemcpyKind::topsMemcpyHostToDevice).to_result();
                //     alighed_free(Some(ptr), size, 4096);
                //     return ret;
                // } else {
                //     return driv::topsMemcpy(self.ptr.as_raw(), pointer as *mut c_void, size as u64, driv::topsMemcpyKind::topsMemcpyHostToDevice).to_result();
                // // driv::topsDeviceSynchronize().to_result()?;;
                // }

                // driv::topsHostFree(ptr);
            }
        }
        Ok(())
    }
}
impl<T: DeviceCopy> CopyDestination<TopsDeviceSlice<T>> for TopsDeviceSlice<T> {
    fn copy_from(&mut self, val: &TopsDeviceSlice<T>) -> DeviceResult<()> {
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = (mem::size_of::<T>() * self.len()) as usize;
        if size != 0 {
            unsafe {
                driv::topsMemcpyDtoD(self.ptr.as_raw(), val.as_device_ptr().as_raw(), size)
                    .to_result()?
            }
        }
        Ok(())
    }

    fn copy_to(&self, val: &mut TopsDeviceSlice<T>) -> DeviceResult<()> {
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = (mem::size_of::<T>() * self.len()) as usize;
        if size != 0 {
            unsafe {
                driv::topsMemcpyDtoD(
                    val.as_device_ptr().as_raw(),
                    self.as_device_ptr().as_raw(),
                    size,
                )
                .to_result()?
            }
        }
        Ok(())
    }
}

impl<T: DeviceCopy> CopyDestination<TopsDeviceBuffer<T>> for TopsDeviceSlice<T> {
    fn copy_from(&mut self, val: &TopsDeviceBuffer<T>) -> DeviceResult<()> {
        self.copy_from(val as &TopsDeviceSlice<T>)
    }

    fn copy_to(&self, val: &mut TopsDeviceBuffer<T>) -> DeviceResult<()> {
        self.copy_to(val as &mut TopsDeviceSlice<T>)
    }
}

impl<T: DeviceCopy, I: AsRef<[T]> + AsMut<[T]> + ?Sized> AsyncCopyDestination<I>
    for TopsDeviceSlice<T>
{
    type StreamT = TopsStream;
    unsafe fn async_copy_from(&mut self, val: &I, stream: &Self::StreamT) -> DeviceResult<()> {
        let val = val.as_ref();
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = (mem::size_of::<T>() * self.len()) as usize;
        if size != 0 {
            driv::topsMemcpyHtoDAsync(
                self.ptr.as_raw(),
                val.as_ptr() as *mut c_void,
                size,
                stream.as_inner(),
            )
            .to_result()?
        }
        Ok(())
    }

    unsafe fn async_copy_to(&self, val: &mut I, stream: &Self::StreamT) -> DeviceResult<()> {
        let val = val.as_mut();
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = (mem::size_of::<T>() * self.len()) as usize;
        if size != 0 {
            driv::topsMemcpyDtoHAsync(
                val.as_mut_ptr() as *mut c_void,
                self.as_device_ptr().as_raw(),
                size,
                stream.as_inner(),
            )
            .to_result()?
        }
        Ok(())
    }
}

impl<T: DeviceCopy> AsyncCopyDestination<TopsDeviceSlice<T>> for TopsDeviceSlice<T> {
    type StreamT = TopsStream;

    unsafe fn async_copy_from(
        &mut self,
        val: &TopsDeviceSlice<T>,
        stream: &Self::StreamT,
    ) -> DeviceResult<()> {
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = (mem::size_of::<T>() * self.len()) as usize;
        if size != 0 {
            driv::topsMemcpyDtoDAsync(
                self.as_device_ptr().as_raw(),
                val.as_device_ptr().as_raw(),
                size,
                stream.as_inner(),
            )
            .to_result()?
        }
        Ok(())
    }

    unsafe fn async_copy_to(
        &self,
        val: &mut TopsDeviceSlice<T>,
        stream: &Self::StreamT,
    ) -> DeviceResult<()> {
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = (mem::size_of::<T>() * self.len()) as usize;
        if size != 0 {
            driv::topsMemcpyDtoDAsync(
                val.as_device_ptr().as_raw(),
                self.as_device_ptr().as_raw(),
                size,
                stream.as_inner(),
            )
            .to_result()?
        }
        Ok(())
    }
}

impl<T: DeviceCopy> AsyncCopyDestination<TopsDeviceBuffer<T>> for TopsDeviceSlice<T> {
    type StreamT = TopsStream;

    unsafe fn async_copy_from(
        &mut self,
        val: &TopsDeviceBuffer<T>,
        stream: &Self::StreamT,
    ) -> DeviceResult<()> {
        self.async_copy_from(val as &TopsDeviceSlice<T>, stream)
    }

    unsafe fn async_copy_to(
        &self,
        val: &mut TopsDeviceBuffer<T>,
        stream: &Self::StreamT,
    ) -> DeviceResult<()> {
        self.async_copy_to(val as &mut TopsDeviceSlice<T>, stream)
    }
}
