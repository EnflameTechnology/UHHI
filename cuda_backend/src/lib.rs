//! Safe, Fast, and user-friendly wrapper around the Device Driver API.
//!
//! # Low level Device interop
//!
//! Because additions to Device and libraries that use Device are everchanging, this library
//! provides unsafe functions for retrieving and setting handles to raw Device_sys objects.
//! This allows advanced users to embed libraries that rely on Device, such as OptiX. We
//! also re-export as a [`driv`] module for convenience.
//!
//! # Device Terminology:
//!
//! ## Devices and Hosts:
//!
//! This crate and its documentation uses the terms "device" and "host" frequently, so it's worth
//! explaining them in more detail. A device refers to a Device-capable GPU or similar device and its
//! associated external memory space. The host is the CPU and its associated memory space. Data
//! must be transferred from host memory to device memory before the device can use it for
//! computations, and the results must then be transferred back to host memory.
//!
//! ## Contexts, Modules, Streams and Functions:
//!
//! A Device context is akin to a process on the host - it contains all of the state for working with
//! a device, all memory allocations, etc. Each context is associated with a single device.
//!
//! A Module is similar to a shared-object library - it is a piece of compiled code which exports
//! functions and global values. Functions can be loaded from modules and launched on a device as
//! one might load a function from a shared-object file and call it. Functions are also known as
//! kernels and the two terms will be used interchangeably.
//!
//! A Stream is akin to a thread - asynchronous work such as kernel execution can be queued into a
//! stream. Work within a single stream will execute sequentially in the order that it was
//! submitted, and may interleave with work from other streams.
//!
//! ## Grids, Blocks and Threads:
//!
//! Device devices typically execute kernel functions on many threads in parallel. These threads can
//! be grouped into thread blocks, which share an area of fast hardware memory known as shared
//! memory. Thread blocks can be one-, two-, or three-dimensional, which is helpful when working
//! with multi-dimensional data such as images. Thread blocks are then grouped into grids, which
//! can also be one-, two-, or three-dimensional.
//!
//! Device devices often contain multiple separate processors. Each processor is capable of excuting
//! many threads simultaneously, but they must be from the same thread block. Thus, it is important
//! to ensure that the grid size is large enough to provide work for all processors. On the other
//! hand, if the thread blocks are too small each processor will be under-utilized and the
//! code will be unable to make effective use of shared memory.
//!
//! # Usage:
//!
//! Before using cust, you must install the Device development libraries for your system. Version
//! 9.0 or newer is required. You must also have a Device-capable GPU installed with the appropriate
//! drivers.
//!
//! Cust will try to find the Device libraries automatically, if it is unable to find it, you can set
//! `CUDA_LIBRARY_PATH` to some path manually.

// pub use crate::uhal::*;

use context::CuContext;
// #![cfg_attr(docsrs, feature(doc_cfg))]
pub use cust_raw as driv;
pub mod device;
pub mod error;
pub mod event;
pub mod external;
pub mod function;
// WIP
pub mod context;
#[allow(warnings)]
pub mod memory;
pub mod module;
pub mod prelude;
pub mod stream;

use uhal::device::DeviceTrait;
// pub use cust_raw as driv;
use uhal::{Flags, DriverLibraryTrait};
use uhal::error::{DeviceResult};
// use uhal::device::{DeviceTrait};
use uhal::context::{ContextFlags, ContextTrait};
use crate::device::CuDevice;
use error::ToResult;

/// Struct representing the API version number.
#[derive(Debug, Hash, Eq, PartialEq, Ord, PartialOrd, Copy, Clone)]
pub struct CuApi(i32);
pub use driv::{CUcontext, CUdevice};

impl DriverLibraryTrait for CuApi {
    type ContextT = CuContext;
    type ApiVersionT = CuApi;
    /// Initialize the Driver API.
    ///
    /// This must be called before any other function is called. Typically, this
    /// should be at the start of your program. All other functions will fail unless the API is
    /// initialized first.
    fn init(flags: Flags) -> DeviceResult<()> {
        unsafe { driv::cuInit(flags.bits()).to_result() }
    }

    /// Shortcut for initializing the Driver API and creating a context with default settings
    /// for the first device.
    ///
    /// **You must keep this context alive while you do further operations or you will get an InvalidContext
    /// error**. e.g. using `let _ctx = quick_init()?;`.
    ///
    /// This is useful for testing or just setting up a basic context quickly. Users with more
    /// complex needs (multiple devices, custom flags, etc.) should use `init` and create their own
    /// context.
    #[must_use = "The Context must be kept alive or errors will be issued for any function that is run"]
    fn quick_init() -> DeviceResult<Self::ContextT>
    {
        CuApi::init(Flags::empty())?;
        let device = CuDevice::get_device(0)?;
        let ctx = CuContext::new(device)?;
        ctx.set_flags(ContextFlags::SCHED_AUTO)?;
        Ok(ctx)
    } 

    /// Returns the latest version supported by the driver.
    fn get_api_version(self) -> DeviceResult<Self::ApiVersionT> {
        unsafe {
            let mut version: i32 = 0;
            driv::cuDriverGetVersion(&mut version as *mut i32).to_result()?;
            Ok(CuApi(version))
        }
    }

    /// Return the major version number - eg. the 9 in version 9.2
    #[inline]
    fn major_api_version(self) -> i32 {
        self.0 / 1000
    }

    /// Return the minor version number - eg. the 2 in version 9.2
    #[inline]
    fn minor_api_version(self) -> i32 {
        self.0 / 10
    }
}
