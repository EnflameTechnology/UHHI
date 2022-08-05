//! Events can be used to track status and dependencies, as well as to measure
//! the duration of work submitted to a Device stream.
//!
//! In Device, most work is performed asynchronously. Events help to manage tasks
//! scheduled on an asynchronous stream. This includes waiting for a task (or
//! multiple tasks) to complete, and measuring the time duration it takes to
//! complete a task. Events can also be used to sequence tasks on multiple
//! streams within the same context by specifying dependent tasks (not supported
//! yet by cust).
//!
//! Events may be reused multiple times.

// TODO: I'm not sure that these events are/can be safe by Rust's model of safety; they inherently
// create state which can be mutated even while an immutable borrow is held.

pub use cust_raw as driv;
use driv::{CUevent};

use uhal::error::{DeviceResult, DropResult};
use uhal::event::{EventTrait, EventFlags, EventStatus};
// use uhal::stream::{Stream};
use uhal::error::{DeviceError};
use uhal::stream::StreamTrait;
use crate::error::ToResult;
use std::mem;
use std::ptr;
use std::time::Duration;
use crate::stream::CuStream;

#[derive(Debug)]
pub struct CuEvent(CUevent);

unsafe impl Send for CuEvent {}
unsafe impl Sync for CuEvent {}

impl EventTrait for CuEvent{
    type RawEventT = CUevent;
    type EventT = CuEvent;
    type StreamT = CuStream;
    /// Create a new event with the specified flags.
    fn new(flags: EventFlags) -> DeviceResult<Self::EventT>
    {
        unsafe {
            let mut event: CUevent = mem::zeroed();
            driv::cuEventCreate(&mut event, flags.bits()).to_result()?;
            Ok(Self::EventT{0: event})
        }
    }

    /// Add the event to the given stream of work. The event will be completed when the stream
    /// completes all previously-submitted work and reaches the event in the queue.
    ///
    /// This function is used together with `query`, `synchronize`, and
    /// `elapsed_time_f32`. See the respective functions for more information.
    ///
    /// If the event is created with `EventFlags::BLOCKING_SYNC`, then `record`
    /// blocks until the event has actually been recorded.
    ///
    /// # Errors
    ///
    /// If the event and stream are not from the same context, an error is
    /// returned.
    fn record(&self, stream: &Self::StreamT) -> DeviceResult<()>{
        unsafe {
            driv::cuEventRecord(self.0, stream.as_inner()).to_result()?;
            Ok(())
        }
    }

    /// Return whether the stream this event was recorded on (see `record`) has processed this event
    /// yet or not. A return value of `EventStatus::Ready` indicates that all work submitted before
    /// the event has been completed.
    fn query(&self) -> DeviceResult<EventStatus>{
        let result = unsafe { driv::cuEventQuery(self.0).to_result() };

        match result {
            Ok(()) => Ok(EventStatus::Ready),
            Err(DeviceError::NotReady) => Ok(EventStatus::NotReady),
            Err(other) => Err(other),
        }
    }

    /// Wait for an event to complete.
    ///
    /// Blocks thread execution until all work submitted before the event was
    /// recorded has completed. `EventFlags::BLOCKING_SYNC` controls the mode of
    /// blocking. If the flag is set on event creation, the thread will sleep.
    /// Otherwise, the thread will busy-wait.
    fn synchronize(&self) -> DeviceResult<()>{
        unsafe {
            driv::cuEventSynchronize(self.0).to_result()?;
            Ok(())
        }
    }

    /// Return the duration between two events.
    ///
    /// The duration is computed in milliseconds with a resolution of
    /// approximately 0.5 microseconds. This can be used to measure the duration of work
    /// queued in between the two events.
    ///
    /// # Errors
    ///
    /// `DeviceError::NotReady` is returned if either event is not yet complete.
    ///
    /// `DeviceError::InvalidHandle` is returned if
    /// - the two events are not from the same context, or if
    /// - `record` has not been called on either event, or if
    /// - the `DISABLE_TIMING` flag is set on either event.
    fn elapsed_time_f32(&self, start: &Self) -> DeviceResult<f32>{
        unsafe {
            let mut millis: f32 = 0.0;
            driv::cuEventElapsedTime(&mut millis, start.0, self.0).to_result()?;
            Ok(millis)
        }
    }

    /// Same as [`elapsed_time_f32`](Self::elapsed_time_f32) except returns the time as a [`Duration`].
    fn elapsed(&self, start: &Self) -> DeviceResult<Duration>{
        let time_f32 = self.elapsed_time_f32(start)?;
        // multiply to nanos to preserve as much precision as possible
        Ok(Duration::from_nanos((time_f32 * 1e6) as u64))
    }

    // Get the inner `event` from the `Event`.
    //
    // Necessary for certain Device functions outside of this
    // module that expect a bare `event`.
    fn as_inner(&self) -> Self::RawEventT{
        self.0
    }

    /// Destroy an `Event` returning an error.
    ///
    /// Destroying an event can return errors from previous asynchronous work.
    /// This function destroys the given event and returns the error and the
    /// un-destroyed event on failure.
    fn drop(mut event: Self::EventT) -> DropResult<Self::EventT>{
        if event.0.is_null() {
            return Ok(());
        }

        unsafe {
            let inner = mem::replace(&mut event.0, ptr::null_mut());
            match driv::cuEventDestroy_v2(inner).to_result() {
                Ok(()) => {
                    mem::forget(event);
                    Ok(())
                }
                Err(e) => Err((e, Self::EventT{0: inner})),
            }
        }
    }
}

impl Drop for CuEvent {
    fn drop(&mut self) {
        unsafe { driv::cuEventDestroy_v2(self.0) };
    }
}