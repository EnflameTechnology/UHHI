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

use crate::error::{DeviceResult, DropResult};
// use crate::stream::Stream;
// use std::mem;
// use std::ptr;
use std::time::Duration;

bitflags::bitflags! {
    /// Bit flags for configuring a Device Event.
    ///
    /// The Device documentation claims that setting `DISABLE_TIMING` and `BLOCKING_SYNC` provides
    /// the best performance for `query()` and `stream.wait_event()`.
    pub struct EventFlags: u32 {
        /// The default event creation flag.
        const DEFAULT = 0x0;

        /// Specify that the created event should busy-wait on blocking
        /// function calls.
        const BLOCKING_SYNC = 0x1;

        /// Specify that the created event does not need to record timing data.
        const DISABLE_TIMING = 0x2;

        /// Specify that the created event may be used as an interprocess event.
        /// (not supported yet by cust). This flag requires
        /// `DISABLE_TIMING` to be set as well.
        const INTERPROCESS = 0x4;
    }
}

/// Status enum that represents the current status of an event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EventStatus {
    /// Ready indicates that all work captured by the event has been completed.
    ///
    /// The Device documentation states that for Unified Memory, `EventStatus::Ready` is
    /// equivalent to having called `Event::synchronize`.
    Ready,

    /// `EventStatus::NotReady` indicates that the work captured by the event is still
    /// incomplete.
    NotReady,
}

/// An event to track work submitted to a stream.
///
/// See the module-level documentation for more information.
// #[derive(Debug)]
// pub struct Event<T> {
//     pub e : T,
// }

pub trait EventTrait {
    type RawEventT;
    type EventT;
    type StreamT;
    /// Create a new event with the specified flags.
    fn new(flags: EventFlags) -> DeviceResult<Self::EventT>;

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
    fn record(&self, stream: &Self::StreamT) -> DeviceResult<()>;

    /// Return whether the stream this event was recorded on (see `record`) has processed this event
    /// yet or not. A return value of `EventStatus::Ready` indicates that all work submitted before
    /// the event has been completed.
    fn query(&self) -> DeviceResult<EventStatus>;

    /// Wait for an event to complete.
    ///
    /// Blocks thread execution until all work submitted before the event was
    /// recorded has completed. `EventFlags::BLOCKING_SYNC` controls the mode of
    /// blocking. If the flag is set on event creation, the thread will sleep.
    /// Otherwise, the thread will busy-wait.
    fn synchronize(&self) -> DeviceResult<()>;

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
    fn elapsed_time_f32(&self, start: &Self) -> DeviceResult<f32>;

    /// Same as [`elapsed_time_f32`](Self::elapsed_time_f32) except returns the time as a [`Duration`].
    fn elapsed(&self, start: &Self) -> DeviceResult<Duration>;

    // Get the inner `event` from the `Event`.
    //
    // Necessary for certain Device functions outside of this
    // module that expect a bare `event`.
    fn as_inner(&self) -> Self::RawEventT;

    /// Destroy an `Event` returning an error.
    ///
    /// Destroying an event can return errors from previous asynchronous work.
    /// This function destroys the given event and returns the error and the
    /// un-destroyed event on failure.
    fn drop(event: Self::EventT) -> DropResult<Self::EventT>;
}
