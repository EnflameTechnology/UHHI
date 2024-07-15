//! Streams of work for the device to perform.
//!
//! In Device, most work is performed asynchronously. Even tasks such as memory copying can be
//! scheduled by the host and performed when ready. Scheduling this work is done using a Stream.
//!
//! A stream is required for all asynchronous tasks in Device, such as kernel launches and
//! asynchronous memory copying. Each task in a stream is performed in the order it was scheduled,
//! and tasks within a stream cannot overlap. Tasks scheduled in multiple streams may interleave or
//! execute concurrently. Sequencing between multiple streams can be achieved using events, which
//! are not currently supported by cust. Finally, the host can wait for all work scheduled in
//! a stream to be completed.
pub use tops_raw as driv;
pub use driv::{topsStream_t, topsEvent_t, topsFunction_t, topsModule_t, dim3};
use uhal::event::EventTrait;
use uhal::function::{BlockSize, GridSize, FunctionTrait};
// use uhal::event::{Event};
use uhal::stream::{StreamTrait, StreamFlags, StreamWaitEventFlags};
use uhal::error::{DeviceResult, DropResult, DeviceError};

use std::ffi::c_void;
use std::mem;
use std::ptr;

use crate::event::TopsEvent;
use crate::function::TopsFunction;
use crate::error::ToResult;

#[derive(Debug)]
pub struct TopsStream(topsStream_t);

unsafe impl Send for TopsStream {}
unsafe impl Sync for TopsStream {}

// unsafe extern "C" fn callback_wrapper<T>(callback: *mut c_void)
// where
//     T: FnOnce() + Send,
// {
//     // Stop panics from unwinding across the FFI
//     let _ = panic::catch_unwind(|| {
//         let callback: Box<T> = Box::from_raw(callback as *mut T);
//         callback();
//     });
// }


impl<'a> StreamTrait<'a> for TopsStream {
    type RawStreamT = topsStream_t;
    type StreamT = TopsStream;
    type EventT = TopsEvent;
    type FunctionT = TopsFunction<'a>;
    /// Create a new stream with the given flags and optional priority.
    ///
    /// By convention, `priority` follows a convention where lower numbers represent greater
    /// priorities. That is, work in a stream with a lower priority number may pre-empt work in
    /// a stream with a higher priority number. `Context::get_stream_priority_range` can be used
    /// to get the range of valid priority values; if priority is set outside that range, it will
    /// be automatically clamped to the lowest or highest number in the range.
    fn new(_flags: StreamFlags, _priority: Option<i32>) -> DeviceResult<Self::StreamT>{
        unsafe {
            let mut stream = Self::StreamT {
                0: ptr::null_mut(),
            };
            driv::topsStreamCreate(
                &mut stream.0 as *mut Self::RawStreamT,
                // flags.bits(),
                // priority.unwrap_or(0),
            )
            .to_result()?;
            Ok(stream)
        }
    }

    /// Return the flags which were used to create this stream.
    fn get_flags(&self) -> DeviceResult<StreamFlags>{
        // unsafe {
        //     let mut bits = 0u32;
        //     driv::topsStreamGetFlags(self.0, &mut bits as *mut u32).to_result()?;
        //     Ok(StreamFlags::from_bits_truncate(bits))
        // }
        Ok(StreamFlags::DEFAULT)
    }

    /// Return the priority of this stream.
    ///
    /// If this stream was created without a priority, returns the default priority.
    /// If the stream was created with a priority outside the valid range, returns the clamped
    /// priority.
    fn get_priority(&self) -> DeviceResult<i32>{
        // unsafe {
        //     let mut priority = 0i32;
        //     driv::topsStreamGetPriority(self.0, &mut priority as *mut i32).to_result()?;
        //     Ok(priority)
        // }
        Ok(0)
    }

    /// Add a callback to a stream.
    ///
    /// The callback will be executed after all previously queued
    /// items in the stream have been completed. Subsequently queued
    /// items will not execute until the callback is finished.
    ///
    /// Callbacks must not make any Device API calls.
    fn add_callback<T>(&self, _callback: Box<T>) -> DeviceResult<()>
    where
        T: FnOnce() + Send
    {
        // unsafe {
        //     driv::topsStreamAddCallback(self.0, callback, 0, 0).to_result();

        //     // driv::topsLaunchHostFunc(
        //     //     self.0,
        //     //     Some(callback_wrapper::<T>),
        //     //     Box::into_raw(callback) as *mut c_void,
        //     // )
        //     // .to_result()
        // }
        //TODO
        Err(DeviceError::InvalidValue)
    }

    /// Wait until a stream's tasks are completed.
    ///
    /// Waits until the device has completed all operations scheduled for this stream.
    fn synchronize(&self) -> DeviceResult<()>{
        unsafe { driv::topsStreamSynchronize(self.0).to_result() }
    }

    /// Make the stream wait on an event.
    ///
    /// All future work submitted to the stream will wait for the event to
    /// complete. Synchronization is performed on the device, if possible. The
    /// event may originate from different context or device than the stream.
    fn wait_event(&self, event: Self::EventT, flags: StreamWaitEventFlags) -> DeviceResult<()>{
        unsafe { driv::topsStreamWaitEvent(self.0, event.as_inner(), flags.bits()).to_result() }
    }

    // Hidden implementation detail function. Highly unsafe. Use the `launch!` macro instead.
    #[doc(hidden)]
    unsafe fn launch<G, B>(
        &self,
        func: &Self::FunctionT,
        grid_size: G,
        block_size: B,
        shared_mem_bytes: u32,
        args: &[*mut c_void],
    ) -> DeviceResult<()>
    where
        G: Into<GridSize>,
        B: Into<BlockSize>,

    {
        let grid_size: GridSize = grid_size.into();
        let block_size: BlockSize = block_size.into();

        //Layout of parameters for launchkernel in tops is different from CUDA!!!
        // let mut args_ = Vec::new();
        // for i in 0..args.len(){
        //     let vaddress = std::mem::transmute::<*mut c_void, *mut *mut c_void>((*args)[i]);
        //     unsafe {args_.push(*vaddress);}
            
        // }

        // let mut size :usize = (std::mem::size_of::<c_ulonglong>() * (args.len() - 1) + std::mem::size_of::<usize>()) as usize;
        // let mut config = vec![0x1 as *const c_void, args_.as_mut_ptr() as *const _ as *mut c_void, 0x2 as *const c_void, &mut size as *const _ as *mut c_void, 0x3 as *const c_void];

        let nul = ptr::null_mut();

        driv::topsModuleLaunchKernel(
            func.to_raw(), grid_size.x, grid_size.y, grid_size.z,
            block_size.x, block_size.y, block_size.z,
            shared_mem_bytes as u32,
            self.0,
            args.as_ptr() as *mut *mut c_void,    
            nul as *mut *mut c_void,
            // config.as_mut_ptr() as *mut *mut c_void            
        )
        .to_result()
    }

    // Get the inner `CUstream` from the `Stream`. If you use this handle elsewhere,
    // make sure not to use it after the stream has been dropped. Or ManuallyDrop the struct to be safe.
    fn as_inner(&self) -> Self::RawStreamT{
        self.0
    }

    /// Destroy a `Stream`, returning an error.
    ///
    /// Destroying a stream can return errors from previous asynchronous work. This function
    /// destroys the given stream and returns the error and the un-destroyed stream on failure.
    fn drop(mut stream: Self::StreamT) -> DropResult<Self::StreamT>{
        if stream.0.is_null() {
            return Ok(());
        }

        unsafe {
            let inner = mem::replace(&mut stream.0, ptr::null_mut());
            match driv::topsStreamDestroy(inner).to_result() {
                Ok(()) => {
                    mem::forget(stream);
                    println!("Stream destroy!");
                    Ok(())
                }
                Err(e) => Err((e, Self::StreamT { 0:inner })),
            }
        }
    }
}

impl Drop for TopsStream{
    fn drop(&mut self) {
        if self.0.is_null() {
            return;
        }
        unsafe {
            let inner = mem::replace(&mut self.0, ptr::null_mut());
            driv::topsStreamDestroy(inner);
        }
    }
}

