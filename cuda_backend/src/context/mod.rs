//! Device Context handling.
//! The Device Driver API has two main ways of creating contexts. The "legacy" (legacy
//! meaning this is what the de-facto way of doing it in cust was) context handling,
//! and the "new" primary context handling. In the legacy way of handling contexts,
//! a thread could posess multiple contexts inside of a stack, and users would explicitly
//! create entire new contexts and set them as the current context at the top of the stack.
//!
//! This is great for control, but it causes a myriad of issues when trying to interoperate
//! with runtime API based libraries such as cuBLAS or cuFFT. Explicitly making and destroying
//! contexts causes a lot of problems with the runtime API because the runtime API will implicitly
//! use any context the driver API set as current. This sometimes causes segfaults and odd
//! behavior that is overall hard to manage if trying to use other Device libraries.
//!
//! The "new" primary context handling uses the same handling as the Runtime API. Instead
//! of context stacks, only a single context exists for every device, and this context
//! is reference-counted. Users can retain a handle to the primary context, increasing the
//! reference count, and release the context once they are done using it. Because this is
//! the same handling that the Runtime API uses, it is directly compatible with libraries
//! such as cuBLAS.
//!
//! Primary contexts also simplify the context API greatly, making new contexts on the same
//! device will just use the same context. This means there is no need for unowned contexts
//! when using multithreading. Users can simply make new contexts for every thread with no concern
//! that the context will be prematurely destroyed.
//!
//! So overall, we reccomend everyone use the new primary context handling, and avoid
//! the old legacy handling. Doing so will make your use of cust more compatible with
//! libraries like cuBLAS or cuFFT, as well as avoid potentially confusing context-based bugs.
pub use cust_raw as driv;
// use crate::{
//     private::Sealed,
// };

use uhal::device::DeviceTrait;
// use uhal::{ApiVersion};
use uhal::context::{CacheConfig, ResourceLimit, StreamPriorityRange, ContextHandle, ContextTrait, ContextFlags, SharedMemoryConfig, CurrentContextTrait};
// use uhal::device::{Device};
use uhal::error::{DeviceResult, DropResult};
use crate::device::CuDevice;
use crate::error::ToResult;
use std::{
    mem::{self, transmute, MaybeUninit},
    ptr,
};
use crate::{driv::*, CuApi};

#[derive(Debug)]
pub struct CuContext {
    pub inner: CUcontext,
    pub device: CUdevice,
}

impl crate::memory::private::Sealed for CuContext {}
impl ContextHandle for CuContext {
    type RawContextT = CUcontext;
    fn get_inner(&self) -> Self::RawContextT {
        self.inner
    }
}

unsafe impl Send for CuContext {}
unsafe impl Sync for CuContext {}

impl Clone for CuContext {
    fn clone(&self) -> Self {
        // because we already retained a context on this device successfully (self), it is
        // exceedingly rare that this function would fail, therefore a silent panic
        // is mostly okay
        Self::new(CuDevice::new(self.device))
        .expect("Failed to clone context")
    }
}


impl ContextTrait for CuContext {
    type ContextT = CuContext;
    type DeviceT = CuDevice;
    type RawContextT = CUcontext;
    type ApiVersionT = CuApi;
    /// Retains the primary context for this device and makes it current, incrementing the internal reference cycle
    /// that Device keeps track of. There is only one primary context associated with a device, multiple
    /// calls to this function with the same device will return the same internal context.
    ///
    /// This will **NOT** push the context to the stack, primary contexts do not interoperate
    /// with the context stack.
    fn new(device: Self::DeviceT) -> DeviceResult<Self::ContextT>
    {
        let mut inner = MaybeUninit::uninit();
        unsafe {
            driv::cuDevicePrimaryCtxRetain(inner.as_mut_ptr(), device.as_raw()).to_result()?;
            let inner = inner.assume_init();
            driv::cuCtxSetCurrent(inner);
            Ok(Self::ContextT {
                inner: inner,
                device: device.as_raw(),
            })
        }
    }

    /// Resets the primary context associated with the device, freeing all allocations created
    /// inside of the context. You must make sure that nothing else is using the context or using
    /// Device on the device in general. For this reason, it is usually highly advised to not use
    /// this function.
    ///
    /// # Safety
    ///
    /// Nothing else should be using the primary context for this device, otherwise,
    /// spurious errors or segfaults will occur.
    unsafe fn reset(device: &Self::DeviceT) -> DeviceResult<()>
    {
        driv::cuDevicePrimaryCtxReset_v2(device.as_raw()).to_result()
    }

    /// Sets the flags for the device context, these flags will apply to any user of the primary
    /// context associated with this device.
    fn set_flags(&self, flags: ContextFlags) -> DeviceResult<()>
    {
        unsafe { driv::cuDevicePrimaryCtxSetFlags_v2(self.device, flags.bits()).to_result() }
    }

    /// Returns the raw handle to this context.
    fn as_raw(&self) -> Self::RawContextT
    {
        self.inner
    }

    /// Get the API version used to create this context.
    ///
    /// This is not necessarily the latest version supported by the driver.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::{Context, ContextFlags};
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// cust::init(cust::Flags::empty())?;
    /// let device = Device::get_device(0)?;
    /// let context = Context::new(device)?;
    /// let version = context.get_api_version()?;
    /// # Ok(())
    /// # }
    /// ```
    fn get_api_version(&self) -> DeviceResult<Self::ApiVersionT>
    {
        unsafe {
            let mut api_version = 0u32;
            driv::cuCtxGetApiVersion(self.inner, &mut api_version as *mut u32).to_result()?;
            Ok(CuApi {
                0: api_version as i32,
            })
        }
    }

    /// Destroy a `Context`, returning an error.
    ///
    /// Destroying a context can return errors from previous asynchronous work. This function
    /// destroys the given context and returns the error and the un-destroyed context on failure.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::{Context, ContextFlags};
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::Flags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::new(device)?;
    /// match Context::drop(context) {
    ///     Ok(()) => println!("Successfully destroyed"),
    ///     Err((e, ctx)) => {
    ///         println!("Failed to destroy context: {:?}", e);
    ///         // Do something with ctx
    ///     },
    /// }
    /// # Ok(())
    /// # }
    /// ```
    fn drop(mut ctx: Self::ContextT) -> DropResult<Self::ContextT>
    {
        if ctx.inner.is_null() {
            return Ok(());
        }

        unsafe {
            let inner = mem::replace(&mut ctx.inner, ptr::null_mut());
            match driv::cuDevicePrimaryCtxRelease_v2(ctx.device).to_result() {
                Ok(()) => {
                    mem::forget(ctx);
                    Ok(())
                }
                Err(e) => Err((
                    e,
                    Self::ContextT {
                        inner,
                        device: ctx.device,
                    },
                )),
            }
        }
    }
}

impl Drop for CuContext {
    fn drop(&mut self) {
        if self.inner.is_null() {
            return;
        }

        unsafe {
            self.inner = ptr::null_mut();
            driv::cuDevicePrimaryCtxRelease_v2(self.device);
        }
    }
}
pub struct CuCurrentContext;

impl CurrentContextTrait for CuCurrentContext {
    type DeviceT = CuDevice;
    type ContextT = CuContext;
    /// Returns the preferred cache configuration for the current context.
    ///
    /// On devices where the L1 cache and shared memory use the same hardware resources, this
    /// function returns the preferred cache configuration for the current context. For devices
    /// where the size of the L1 cache and shared memory are fixed, this will always return
    /// `CacheConfig::PreferNone`.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::{ Context, ContextFlags, CurrentContext };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # init(Flags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::new(device)?;
    /// let cache_config = CurrentContext::get_cache_config()?;
    /// # Ok(())
    /// # }
    /// ```
    fn get_cache_config() -> DeviceResult<CacheConfig>
    {
        unsafe {
            let mut config = CacheConfig::PreferNone;
            driv::cuCtxGetCacheConfig(&mut config as *mut CacheConfig as *mut driv::CUfunc_cache)
                .to_result()?;
            Ok(config)
        }
    }

    /// Return the device ID for the current context.
    ///
    /// # Example
    ///
    /// ```
    /// # use crate::cuda_backend::device::CuDevice;
    /// # use crate::cuda_backend::context::{ CuContext, ContextFlags, CuCurrentContext };
    /// # use std::error::Error;
    /// # use crate::cuda_backend::CuApi;
    /// # use uhal::Flags;
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # CuApi::init(Flags::empty())?;
    /// # let device = CuDevice::get_device(0)?;
    /// let context = CuContext::new(device)?;
    /// let device = CuCurrentContext::get_device()?;
    /// # Ok(())
    /// # }
    /// ```
    fn get_device() -> DeviceResult<Self::DeviceT>
    {
        unsafe {
            let mut device = CuDevice::new(0);
            driv::cuCtxGetDevice(&mut device.as_raw() as *mut CUdevice).to_result()?;
            Ok(device)
        }
    }

    /// Return the context flags for the current context.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::{ Context, ContextFlags, CurrentContext };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::DeviceFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::new(device)?;
    /// let flags = CurrentContext::get_flags()?;
    /// # Ok(())
    /// # }
    /// ```
    fn get_flags() -> DeviceResult<ContextFlags>
    {
        unsafe {
            let mut flags = 0u32;
            driv::cuCtxGetFlags(&mut flags as *mut u32).to_result()?;
            Ok(ContextFlags::from_bits_truncate(flags))
        }
    }

    /// Return resource limits for the current context.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::{ Context, ContextFlags, CurrentContext, ResourceLimit };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::DeviceFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::new(device)?;
    /// let stack_size = CurrentContext::get_resource_limit(ResourceLimit::StackSize)?;
    /// # Ok(())
    /// # }
    /// ```
    fn get_resource_limit(resource: ResourceLimit) -> DeviceResult<usize>
    {
        unsafe {
            let mut limit: usize = 0;
            driv::cuCtxGetLimit(&mut limit as *mut usize, transmute(resource)).to_result()?;
            Ok(limit)
        }
    }

    /// Return resource limits for the current context.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::{ Context, ContextFlags, CurrentContext, ResourceLimit };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::DeviceFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::new(device)?;
    /// let shared_mem_config = CurrentContext::get_shared_memory_config()?;
    /// # Ok(())
    /// # }
    /// ```
    fn get_shared_memory_config() -> DeviceResult<SharedMemoryConfig>
    {
        unsafe {
            let mut cfg = SharedMemoryConfig::DefaultBankSize;
            driv::cuCtxGetSharedMemConfig(
                &mut cfg as *mut SharedMemoryConfig as *mut driv::CUsharedconfig,
            )
            .to_result()?;
            Ok(cfg)
        }
    }

    /// Return the least and greatest stream priorities.
    ///
    /// If the program attempts to create a stream with a priority outside of this range, it will be
    /// automatically clamped to within the valid range. If the device does not support stream
    /// priorities, the returned range will contain zeroes.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::{ Context, ContextFlags, CurrentContext};
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::DeviceFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::new(device)?;
    /// let priority_range = CurrentContext::get_stream_priority_range()?;
    /// # Ok(())
    /// # }
    /// ```
    fn get_stream_priority_range() -> DeviceResult<StreamPriorityRange>
    {
        unsafe {
            let mut range = StreamPriorityRange {
                least: 0,
                greatest: 0,
            };
            driv::cuCtxGetStreamPriorityRange(
                &mut range.least as *mut i32,
                &mut range.greatest as *mut i32,
            )
            .to_result()?;
            Ok(range)
        }
    }

    /// Sets the preferred cache configuration for the current context.
    ///
    /// On devices where L1 cache and shared memory use the same hardware resources, this sets the
    /// preferred cache configuration for the current context. This is only a preference. The
    /// driver will use the requested configuration if possible, but is free to choose a different
    /// configuration if required to execute the function.
    ///
    /// This setting does nothing on devices where the size of the L1 cache and shared memory are
    /// fixed.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::{ Context, ContextFlags, CurrentContext, CacheConfig };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::DeviceFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::new(device)?;
    /// CurrentContext::set_cache_config(CacheConfig::PreferL1)?;
    /// # Ok(())
    /// # }
    /// ```
    fn set_cache_config(cfg: CacheConfig) -> DeviceResult<()>
    {
        unsafe { driv::cuCtxSetCacheConfig(transmute(cfg)).to_result() }
    }
    /// Sets a requested resource limit for the current context.
    ///
    /// Note that this is only a request; the driver is free to modify the requested value to meet
    /// hardware requirements. Each limit has some specific restrictions.
    ///
    /// * `StackSize`: Controls the stack size in bytes for each GPU thread
    /// * `PrintfFifoSize`: Controls the size in bytes of the FIFO used by the `printf()` device
    ///   system call. This cannot be changed after a kernel has been launched which uses the
    ///   `printf()` function.
    /// * `MallocHeapSize`: Controls the size in bytes of the heap used by the `malloc()` and `free()`
    ///   device system calls. This cannot be changed aftr a kernel has been launched which uses the
    ///   `malloc()` and `free()` system calls.
    /// * `DeviceRuntimeSyncDepth`: Controls the maximum nesting depth of a grid at which a thread
    ///   can safely call `DeviceDeviceSynchronize()`. This cannot be changed after a kernel has been
    ///   launched which uses the device runtime. When setting this limit, keep in mind that
    ///   additional levels of sync depth require the driver to reserve large amounts of device
    ///   memory which can no longer be used for device allocations.
    /// * `DeviceRuntimePendingLaunchCount`: Controls the maximum number of outstanding device
    ///    runtime launches that can be made from the current context. A grid is outstanding from
    ///    the point of the launch up until the grid is known to have completed. Keep in mind that
    ///    increasing this limit will require the driver to reserve larger amounts of device memory
    ///    which can no longer be used for device allocations.
    /// * `MaxL2FetchGranularity`: Controls the L2 fetch granularity. This is purely a performance
    ///    hint and it can be ignored or clamped depending on the platform.
    ///
    /// # Example
    ///
    /// ```
    /// # use cust::device::Device;
    /// # use cust::context::{ Context, ContextFlags, CurrentContext, ResourceLimit };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cust::init(cust::DeviceFlags::empty())?;
    /// # let device = Device::get_device(0)?;
    /// let context = Context::new(device)?;
    /// CurrentContext::set_resource_limit(ResourceLimit::StackSize, 2048)?;
    /// # Ok(())
    /// # }
    /// ```
    fn set_resource_limit(resource: ResourceLimit, limit: usize) -> DeviceResult<()>
    {
        unsafe {
            driv::cuCtxSetLimit(transmute(resource), limit).to_result()?;
            Ok(())
        }
    }

    /// Sets the preferred shared memory configuration for the current context.
    ///
    /// On devices with configurable shared memory banks, this function will set the context's
    /// shared memory bank size which is used for subsequent kernel launches.
    ///
    /// # Example
    ///
    /// ```
    /// # use crate::cuda_backend as cuda;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait};
    /// use uhal::DriverLibraryTrait;
    /// # use cuda::device::CuDevice;
    /// # use cuda::context::{ CuContext, ContextFlags, CuCurrentContext, SharedMemoryConfig };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cuda::init(cuda::Flags::empty())?;
    /// # let device = CuDevice::get_device(0)?;
    /// let context = CuContext::new(device)?;
    /// CuCurrentContext::set_shared_memory_config(SharedMemoryConfig::DefaultBankSize)?;
    /// # Ok(())
    /// # }
    /// ```
    fn set_shared_memory_config(cfg: SharedMemoryConfig) -> DeviceResult<()>
    {
        unsafe { driv::cuCtxSetSharedMemConfig(transmute(cfg)).to_result() }
    }

    /// Set the given context as the current context for this thread.
    ///
    /// # Example
    ///
    /// ```
    /// # use crate::cuda_backend as cuda;
    /// use uhal::memory::{DeviceBufferTrait, MemoryTrait, DevicePointerTrait};
    /// use uhal::DriverLibraryTrait;
    /// # use cuda::device::CuDevice;
    /// # use cuda::context::{ CuContext, ContextFlags, CuCurrentContext };
    /// # use std::error::Error;
    /// #
    /// # fn main () -> Result<(), Box<dyn Error>> {
    /// # cuda::init(cuda::Flags::empty())?;
    /// # let device = CuDevice::get_device(0)?;
    /// let context = CuContext::new(device)?;
    /// CuCurrentContext::set_current(&context)?;
    /// # Ok(())
    /// # }
    /// ```
    fn set_current(c: &Self::ContextT) -> DeviceResult<()>
    {
        
        unsafe {
            driv::cuCtxSetCurrent(c.get_inner()).to_result()?;
            Ok(())
        }
    }

    /// Block to wait for a context's tasks to complete.
    fn synchronize() -> DeviceResult<()>
    {
        unsafe {
            driv::cuCtxSynchronize().to_result()?;
            Ok(())
        }
    }
}
