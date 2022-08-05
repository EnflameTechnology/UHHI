//! Functions and types for working with Device kernels.
pub use cust_raw as driv;
use uhal::context::{CacheConfig, SharedMemoryConfig};
use uhal::function::{FunctionAttribute, GridSize, BlockSize, FunctionTrait};
// use uhal::module::{Module};
use uhal::error::{DeviceResult};

use crate::driv::{CUfunction};
use crate::module::CuModule;
use std::marker::PhantomData;
use std::mem::{transmute, MaybeUninit};
use crate::error::ToResult;

/// Handle to a global kernel function.
#[derive(Debug)]
pub struct CuFunction<'a> {
    pub inner: CUfunction,
    pub module: PhantomData<&'a CuModule>,
}


unsafe impl<'a> Send for CuFunction<'a> {}
unsafe impl<'a> Sync for CuFunction<'a> {}

impl<'a> FunctionTrait for CuFunction<'a> {
    type FunctionT = CuFunction<'a>;
    type ModuleT = CuModule;
    type RawFunctionT = CUfunction;
    fn new(inner: Self::RawFunctionT, _module: &Self::ModuleT) -> Self::FunctionT{
        Self::FunctionT {
            inner,
            module: PhantomData,
        }
    }

    /// Returns information about a function.
    fn get_attribute(&self, attr: FunctionAttribute) -> DeviceResult<i32>{
        unsafe {
            let mut val = 0i32;
            driv::cuFuncGetAttribute(
                &mut val as *mut i32,
                // This should be safe, as the repr and values of FunctionAttribute should match.
                ::std::mem::transmute(attr),
                self.inner,
            )
            .to_result()?;
            Ok(val)
        }
    }

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
    fn set_cache_config(&mut self, config: CacheConfig) -> DeviceResult<()>{
        unsafe { driv::cuFuncSetCacheConfig(self.inner, transmute(config)).to_result() }

    }

    /// Sets the preferred shared memory configuration for this function.
    ///
    /// On devices with configurable shared memory banks, this function will set this function's
    /// shared memory bank size which is used for subsequent launches of this function. If not set,
    /// the context-wide setting will be used instead.
    fn set_shared_memory_config(&mut self, cfg: SharedMemoryConfig) -> DeviceResult<()>{
        unsafe { driv::cuFuncSetSharedMemConfig(self.inner, transmute(cfg)).to_result() }

    }

    /// Retrieves a raw handle to this function.
    fn to_raw(&self) -> Self::RawFunctionT {
        self.inner
    }

    /// The amount of dynamic shared memory available per block when launching `blocks` on
    /// a streaming multiprocessor.
    fn available_dynamic_shared_memory_per_block(
        &self,
        blocks: GridSize,
        block_size: BlockSize,
    ) -> DeviceResult<usize>{
        let num_blocks = blocks.x * blocks.y * blocks.z;
        let total_block_size = block_size.x * block_size.y * block_size.z;

        let mut result = MaybeUninit::uninit();
        unsafe {
            driv::cuOccupancyAvailableDynamicSMemPerBlock(
                result.as_mut_ptr(),
                self.to_raw(),
                num_blocks as i32,
                total_block_size as i32,
            )
            .to_result()?;
            Ok(result.assume_init())
        }
    }

    /// The maximum number of active blocks per streaming multiprocessor when this function
    /// is launched with a specific `block_size` with some amount of dynamic shared memory.
    fn max_active_blocks_per_multiprocessor(
        &self,
        block_size: BlockSize,
        dynamic_smem_size: usize,
    ) -> DeviceResult<u32>{
        let total_block_size = block_size.x * block_size.y * block_size.z;

        let mut num_blocks = MaybeUninit::uninit();
        unsafe {
            driv::cuOccupancyMaxActiveBlocksPerMultiprocessor(
                num_blocks.as_mut_ptr(),
                self.to_raw(),
                total_block_size as i32,
                dynamic_smem_size,
            )
            .to_result()?;
            Ok(num_blocks.assume_init() as u32)
        }
    }

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
    ) -> DeviceResult<(u32, u32)>{
        let mut min_grid_size = MaybeUninit::uninit();
        let mut block_size = MaybeUninit::uninit();

        let total_block_size_limit = block_size_limit.x * block_size_limit.y * block_size_limit.z;

        unsafe {
            driv::cuOccupancyMaxPotentialBlockSize(
                min_grid_size.as_mut_ptr(),
                block_size.as_mut_ptr(),
                self.to_raw(),
                None,
                dynamic_smem_size,
                total_block_size_limit as i32,
            )
            .to_result()?;
            Ok((
                min_grid_size.assume_init() as u32,
                block_size.assume_init() as u32,
            ))
        }
    }
}
