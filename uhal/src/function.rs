//! Functions and types for working with Device kernels.

use crate::context::{CacheConfig, SharedMemoryConfig};
use crate::error::DeviceResult;
// use crate::module::Module;
// use crate::driv::{CUfunction, CUmodule};
// use std::marker::PhantomData;
// use std::mem::{transmute, MaybeUninit};

/// Dimensions of a grid, or the number of thread blocks in a kernel launch.
///
/// Each component of a `GridSize` must be at least 1. The maximum size depends on your device's
/// compute capability, but maximums of `x = (2^31)-1, y = 65535, z = 65535` are common. Launching
/// a kernel with a grid size greater than these limits will cause an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridSize {
    /// Width of grid in blocks
    pub x: u32,
    /// Height of grid in blocks
    pub y: u32,
    /// Depth of grid in blocks
    pub z: u32,
}
impl GridSize {
    /// Create a one-dimensional grid of `x` blocks
    #[inline]
    pub fn x(x: u32) -> GridSize {
        GridSize { x, y: 1, z: 1 }
    }

    /// Create a two-dimensional grid of `x * y` blocks
    #[inline]
    pub fn xy(x: u32, y: u32) -> GridSize {
        GridSize { x, y, z: 1 }
    }

    /// Create a three-dimensional grid of `x * y * z` blocks
    #[inline]
    pub fn xyz(x: u32, y: u32, z: u32) -> GridSize {
        GridSize { x, y, z }
    }
}
impl From<u32> for GridSize {
    fn from(x: u32) -> GridSize {
        GridSize::x(x)
    }
}
impl From<(u32, u32)> for GridSize {
    fn from((x, y): (u32, u32)) -> GridSize {
        GridSize::xy(x, y)
    }
}
impl From<(u32, u32, u32)> for GridSize {
    fn from((x, y, z): (u32, u32, u32)) -> GridSize {
        GridSize::xyz(x, y, z)
    }
}
impl<'a> From<&'a GridSize> for GridSize {
    fn from(other: &GridSize) -> GridSize {
        *other
    }
}
#[cfg(feature = "vek")]
impl From<vek::Vec2<u32>> for GridSize {
    fn from(vec: vek::Vec2<u32>) -> Self {
        GridSize::xy(vec.x, vec.y)
    }
}
#[cfg(feature = "vek")]
impl From<vek::Vec3<u32>> for GridSize {
    fn from(vec: vek::Vec3<u32>) -> Self {
        GridSize::xyz(vec.x, vec.y, vec.z)
    }
}
#[cfg(feature = "vek")]
impl From<vek::Vec2<usize>> for GridSize {
    fn from(vec: vek::Vec2<usize>) -> Self {
        GridSize::xy(vec.x as u32, vec.y as u32)
    }
}
#[cfg(feature = "vek")]
impl From<vek::Vec3<usize>> for GridSize {
    fn from(vec: vek::Vec3<usize>) -> Self {
        GridSize::xyz(vec.x as u32, vec.y as u32, vec.z as u32)
    }
}

/// Dimensions of a thread block, or the number of threads in a block.
///
/// Each component of a `BlockSize` must be at least 1. The maximum size depends on your device's
/// compute capability, but maximums of `x = 1024, y = 1024, z = 64` are common. In addition, the
/// limit on total number of threads in a block (`x * y * z`) is also defined by the compute
/// capability, typically 1024. Launching a kernel with a block size greater than these limits will
/// cause an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockSize {
    /// X dimension of each thread block
    pub x: u32,
    /// Y dimension of each thread block
    pub y: u32,
    /// Z dimension of each thread block
    pub z: u32,
}
impl BlockSize {
    /// Create a one-dimensional block of `x` threads
    #[inline]
    pub fn x(x: u32) -> BlockSize {
        BlockSize { x, y: 1, z: 1 }
    }

    /// Create a two-dimensional block of `x * y` threads
    #[inline]
    pub fn xy(x: u32, y: u32) -> BlockSize {
        BlockSize { x, y, z: 1 }
    }

    /// Create a three-dimensional block of `x * y * z` threads
    #[inline]
    pub fn xyz(x: u32, y: u32, z: u32) -> BlockSize {
        BlockSize { x, y, z }
    }
}
impl From<u32> for BlockSize {
    fn from(x: u32) -> BlockSize {
        BlockSize::x(x)
    }
}
impl From<(u32, u32)> for BlockSize {
    fn from((x, y): (u32, u32)) -> BlockSize {
        BlockSize::xy(x, y)
    }
}
impl From<(u32, u32, u32)> for BlockSize {
    fn from((x, y, z): (u32, u32, u32)) -> BlockSize {
        BlockSize::xyz(x, y, z)
    }
}
impl<'a> From<&'a BlockSize> for BlockSize {
    fn from(other: &BlockSize) -> BlockSize {
        *other
    }
}
#[cfg(feature = "vek")]
impl From<vek::Vec2<u32>> for BlockSize {
    fn from(vec: vek::Vec2<u32>) -> Self {
        BlockSize::xy(vec.x, vec.y)
    }
}
#[cfg(feature = "vek")]
impl From<vek::Vec3<u32>> for BlockSize {
    fn from(vec: vek::Vec3<u32>) -> Self {
        BlockSize::xyz(vec.x, vec.y, vec.z)
    }
}
#[cfg(feature = "vek")]
impl From<vek::Vec2<usize>> for BlockSize {
    fn from(vec: vek::Vec2<usize>) -> Self {
        BlockSize::xy(vec.x as u32, vec.y as u32)
    }
}
#[cfg(feature = "vek")]
impl From<vek::Vec3<usize>> for BlockSize {
    fn from(vec: vek::Vec3<usize>) -> Self {
        BlockSize::xyz(vec.x as u32, vec.y as u32, vec.z as u32)
    }
}

/// All supported function attributes for [Function::get_attribute](struct.Function.html#method.get_attribute)
#[repr(u32)]
#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum FunctionAttribute {
    /// The maximum number of threads per block, beyond which a launch would fail. This depends on
    /// both the function and the device.
    MaxThreadsPerBlock = 0,

    /// The size in bytes of the statically-allocated shared memory required by this function.
    SharedMemorySizeBytes = 1,

    /// The size in bytes of the constant memory required by this function
    ConstSizeBytes = 2,

    /// The size in bytes of local memory used by each thread of this function
    LocalSizeBytes = 3,

    /// The number of registers used by each thread of this function
    NumRegisters = 4,

    /// The PTX virtual architecture version for which the function was compiled. This value is the
    /// major PTX version * 10 + the minor PTX version, so version 1.3 would return the value 13.
    PtxVersion = 5,

    /// The binary architecture version for which the function was compiled. Encoded the same way as
    /// PtxVersion.
    BinaryVersion = 6,

    /// The attribute to indicate whether the function has been compiled with user specified
    /// option "-Xptxas --dlcm=ca" set.
    CacheModeCa = 7,
}

pub trait FunctionTrait {
    type FunctionT;
    type ModuleT;
    type RawFunctionT;

    fn new(inner: Self::RawFunctionT, _module: &Self::ModuleT) -> Self::FunctionT;

    /// Returns information about a function.
    fn get_attribute(&self, attr: FunctionAttribute) -> DeviceResult<i32>;

    /// Sets the preferred cache configuration for this function.
    ///
    /// On devices where L1 cache and shared memory use the same hardware resources, this sets the
    /// preferred cache configuration for this function. This is only a preference. The
    /// driver will use the requested configuration if possible, but is free to choose a different
    /// configuration if required to execute the function. This setting will override the
    /// context-wide setting.
    ///
    /// This setting does nothing on devices where the size of the L1 cache and shared memory are
    /// fixed.
    fn set_cache_config(&mut self, config: CacheConfig) -> DeviceResult<()>;

    /// Sets the preferred shared memory configuration for this function.
    ///
    /// On devices with configurable shared memory banks, this function will set this function's
    /// shared memory bank size which is used for subsequent launches of this function. If not set,
    /// the context-wide setting will be used instead.
    fn set_shared_memory_config(&mut self, cfg: SharedMemoryConfig) -> DeviceResult<()>;

    /// Retrieves a raw handle to this function.
    fn to_raw(&self) -> Self::RawFunctionT;

    /// The amount of dynamic shared memory available per block when launching `blocks` on
    /// a streaming multiprocessor.
    fn available_dynamic_shared_memory_per_block(
        &self,
        blocks: GridSize,
        block_size: BlockSize,
    ) -> DeviceResult<usize>;

    /// The maximum number of active blocks per streaming multiprocessor when this function
    /// is launched with a specific `block_size` with some amount of dynamic shared memory.
    fn max_active_blocks_per_multiprocessor(
        &self,
        block_size: BlockSize,
        dynamic_smem_size: usize,
    ) -> DeviceResult<u32>;

    // TODO(RDambrosio016): Figure out a way to safely wrap a rust closure to pass it to Device for blockSizeToDynamicSMemSize.
    // It is an issue because we need to prevent unwinding but the no-unwinding wrapper cannot capture the function from its scope.

    /// Returns a reasonable block and grid size to achieve the maximum capacity for the launch (the max number
    /// of active warps with the fewest blocks per multiprocessor).
    ///
    /// # Params
    ///
    /// `dynamic_smem_size` is the amount of dynamic shared memory required by this function. We currently do not expose
    /// a way of determining this dynamically based on block size due to safety concerns.
    ///
    /// `block_size_limit` is the maximum block size that this function is designed to handle. if this is `0` Device will use the maximum
    /// block size permitted by the device/function instead.
    ///
    /// Note: all panics by `dynamic_smem_size` will be ignored and the function will instead use `0`.
    fn suggested_launch_configuration(
        &self,
        dynamic_smem_size: usize,
        block_size_limit: BlockSize,
    ) -> DeviceResult<(u32, u32)>;
}

/// Launch a kernel function asynchronously.
///
/// # Syntax:
///
/// The format of this macro is designed to resemble the triple-chevron syntax used to launch
/// kernels in Device C. There are two forms available:
///
/// ```ignore
/// let result = launch!(module.function_name<<<grid, block, shared_memory_size, stream>>>(parameter1, parameter2...));
/// ```
///
/// This will load a kernel called `function_name` from the module `module` and launch it with
/// the given grid/block size on the given stream. Unlike in Device C, the shared memory size and
/// stream parameters are not optional. The shared memory size is a number of bytes per thread for
/// dynamic shared memory (Note that this uses `extern __shared__ int x[]` in Device C, not the
/// fixed-length arrays created by `__shared__ int x[64]`. This will usually be zero.).
/// `stream` must be the name of a [`Stream`](stream/struct.Stream.html) value.
/// `grid` can be any value which implements [`Into<GridSize>`](function/struct.GridSize.html) (such as
/// `u32` values, tuples of up to three `u32` values, and GridSize structures) and likewise `block`
/// can be any value that implements [`Into<BlockSize>`](function/struct.BlockSize.html).
///
/// NOTE: due to some limitations of Rust's macro system, `module` and `stream` must be local
/// variable names. Paths or function calls will not work.
///
/// The second form is similar:
///
/// ```ignore
/// let result = launch!(function<<<grid, block, shared_memory_size, stream>>>(parameter1, parameter2...));
/// ```
///
/// In this variant, the `function` parameter must be a variable. Use this form to avoid looking up
/// the kernel function for each call.
///
/// # Safety
///
/// Launching kernels must be done in an `unsafe` block. Calling a kernel is similar to calling a
/// foreign-language function, as the kernel itself could be written in C or unsafe Rust. The kernel
/// must accept the same number and type of parameters that are passed to the `launch!` macro. The
/// kernel must not write invalid data (for example, invalid enums) into areas of memory that can
/// be copied back to the host. The programmer must ensure that the host does not access device or
/// unified memory that the kernel could write to until after calling `stream.synchronize()`.
///
#[macro_export]
macro_rules! launch {
    ($module:ident . $function:ident <<<$grid:expr, $block:expr, $shared:expr, $stream:ident>>>( $( $arg:expr),* $(,)?)) => {
        {
            let function = $module.get_function(stringify!($function));
            match function {
                Ok(f) => launch!(f<<<$grid, $block, $shared, $stream>>>( $($arg),* ) ),
                Err(e) => Err(e),
            }
        }
    };
    ($function:ident <<<$grid:expr, $block:expr, $shared:expr, $stream:ident>>>( $( $arg:expr),* $(,)?)) => {
        {
            fn assert_impl_devicecopy<T: $crate::memory::DeviceCopy>(_val: T) {}
            if false {
                $(
                    assert_impl_devicecopy($arg);
                )*
            };

            $stream.launch(&$function, $grid, $block, $shared,
                &[
                    $(
                        &$arg as *const _ as *mut ::std::ffi::c_void,
                    )*
                ]
            )
        }
    };
}
