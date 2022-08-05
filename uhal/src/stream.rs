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

use crate::error::{DeviceResult, DropResult};
// use crate::event::Event;
use crate::function::{BlockSize, GridSize};
use std::ffi::c_void;

bitflags::bitflags! {
    /// Bit flags for configuring a Device Stream.
    pub struct StreamFlags: u32 {
        /// No flags set.
        const DEFAULT = 0x00;

        /// This stream does not synchronize with the NULL stream.
        /// Note that the name is chosen to correspond to Device documentation, but is nevertheless
        /// misleading. All work within a single stream is ordered and asynchronous regardless
        /// of whether this flag is set. All streams in cust may execute work concurrently,
        /// regardless of the flag. However, for legacy reasons, Device has a notion of a NULL stream,
        /// which is used as the default when no other stream is provided. Work on other streams
        /// may not be executed concurrently with work on the NULL stream unless this flag is set.
        /// Since cust does not provide access to the NULL stream, this flag has no effect in
        /// most circumstances. However, it is recommended to use it anyway, as some other crate
        /// in this binary may be using the NULL stream directly.
        const NON_BLOCKING = 0x01;
    }
}

bitflags::bitflags! {
    /// Bit flags for configuring a Device Stream waiting on an Device Event.
    ///
    /// Current versions of Device support only the default flag.
    pub struct StreamWaitEventFlags: u32 {
        /// No flags set.
        const DEFAULT = 0x0;
    }
}

/// A stream of work for the device to perform.
///
/// See the module-level documentation for more information.
// #[derive(Debug)]
// pub struct Stream<T> {
//     pub inner: T,
// }
pub trait StreamTrait<'a> {
    type RawStreamT;
    type StreamT;
    type EventT;
    type FunctionT;
    /// Create a new stream with the given flags and optional priority.
    ///
    /// By convention, `priority` follows a convention where lower numbers represent greater
    /// priorities. That is, work in a stream with a lower priority number may pre-empt work in
    /// a stream with a higher priority number. `Context::get_stream_priority_range` can be used
    /// to get the range of valid priority values; if priority is set outside that range, it will
    /// be automatically clamped to the lowest or highest number in the range.
    fn new(flags: StreamFlags, priority: Option<i32>) -> DeviceResult<Self::StreamT>;

    /// Return the flags which were used to create this stream.
    fn get_flags(&self) -> DeviceResult<StreamFlags>;

    /// Return the priority of this stream.
    ///
    /// If this stream was created without a priority, returns the default priority.
    /// If the stream was created with a priority outside the valid range, returns the clamped
    /// priority.
    fn get_priority(&self) -> DeviceResult<i32>;

    /// Add a callback to a stream.
    ///
    /// The callback will be executed after all previously queued
    /// items in the stream have been completed. Subsequently queued
    /// items will not execute until the callback is finished.
    ///
    /// Callbacks must not make any Device API calls.
    fn add_callback<G>(&self, callback: Box<G>) -> DeviceResult<()>
    where
        G: FnOnce() + Send;

    /// Wait until a stream's tasks are completed.
    ///
    /// Waits until the device has completed all operations scheduled for this stream.
    fn synchronize(&self) -> DeviceResult<()>;

    /// Make the stream wait on an event.
    ///
    /// All future work submitted to the stream will wait for the event to
    /// complete. Synchronization is performed on the device, if possible. The
    /// event may originate from different context or device than the stream.
    fn wait_event(&self, event: Self::EventT, flags: StreamWaitEventFlags) -> DeviceResult<()>;

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
        B: Into<BlockSize>;

    // Get the inner `CUstream` from the `Stream`. If you use this handle elsewhere,
    // make sure not to use it after the stream has been dropped. Or ManuallyDrop the struct to be safe.
    fn as_inner(&self) -> Self::RawStreamT;

    /// Destroy a `Stream`, returning an error.
    ///
    /// Destroying a stream can return errors from previous asynchronous work. This function
    /// destroys the given stream and returns the error and the un-destroyed stream on failure.
    fn drop(stream: Self::StreamT) -> DropResult<Self::StreamT>;
}

