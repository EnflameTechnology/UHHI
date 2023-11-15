//! Types for error handling
//!
//! # Error handling in Device:
//!
//! cust uses the [`DeviceError`](enum.DeviceError.html) enum to represent the errors returned by
//! the Device API. It is important to note that nearly every function in Device (and therefore
//! cust) can fail. Even those functions which have no normal failure conditions can return
//! errors related to previous asynchronous launches.
use std::error::Error;
// use std::error::Error;
use std::ffi::CStr;
use std::fmt;
use std::os::raw::c_char;
use std::ptr;
/// Error enum which represents all the potential errors returned by the driver API.
#[repr(u32)]
#[allow(missing_docs)]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceError {
    NoError = 0,
    // Device errors
    InvalidValue = 1,
    OutOfMemory = 2,
    NotInitialized = 3,
    Deinitialized = 4,
    ProfilerDisabled = 5,
    ProfilerNotInitialized = 6,
    ProfilerAlreadyStarted = 7,
    ProfilerAlreadyStopped = 8,
    NoDevice = 100,
    InvalidDevice = 101,
    InvalidImage = 200,
    InvalidContext = 201,
    ContextAlreadyCurrent = 202,
    MapFailed = 205,
    UnmapFailed = 206,
    ArrayIsMapped = 207,
    AlreadyMapped = 208,
    NoBinaryForGpu = 209,
    AlreadyAcquired = 210,
    NotMapped = 211,
    NotMappedAsArray = 212,
    NotMappedAsPointer = 213,
    EccUncorrectable = 214,
    UnsupportedLimit = 215,
    ContextAlreadyInUse = 216,
    PeerAccessUnsupported = 217,
    InvalidPtx = 218,
    InvalidGraphicsContext = 219,
    NvlinkUncorrectable = 220,
    InvalidSource = 300,
    FileNotFound = 301,
    SharedObjectSymbolNotFound = 302,
    SharedObjectInitFailed = 303,
    OperatingSystemError = 304,
    InvalidHandle = 400,
    NotFound = 500,
    NotReady = 600,
    IllegalAddress = 700,
    LaunchOutOfResources = 701,
    LaunchTimeout = 702,
    LaunchIncompatibleTexturing = 703,
    PeerAccessAlreadyEnabled = 704,
    PeerAccessNotEnabled = 705,
    PrimaryContextActive = 708,
    ContextIsDestroyed = 709,
    AssertError = 710,
    TooManyPeers = 711,
    HostMemoryAlreadyRegistered = 712,
    HostMemoryNotRegistered = 713,
    HardwareStackError = 714,
    IllegalInstruction = 715,
    MisalignedAddress = 716,
    InvalidAddressSpace = 717,
    InvalidProgramCounter = 718,
    LaunchFailed = 719,
    NotPermitted = 800,
    NotSupported = 801,
    UnknownError = 999,

    // Self-defined device error
    InvalidMemoryAllocation = 100_100,
    OptixError = 100_101,
}
// impl Error for DeviceError {}
/// Result type for most Device functions.
pub type DeviceResult<T> = Result<T, DeviceError>;

/// Special result type for `drop` functions which includes the un-dropped value with the error.
pub type DropResult<T> = Result<(), (DeviceError, T)>;

impl Error for DeviceError {}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            DeviceError::OutOfMemory => write!(f, "Invalid memory allocation"),
            DeviceError::UnknownError => write!(f, "OptiX error"),
            other if (other as u32) <= 999 => {
                // let value = other as u32;
                // let mut ptr: *const c_char = ptr::null();
                // unsafe {
                    
                    // driv::cuGetErrorString(mem::transmute(value), &mut ptr as *mut *const c_char)
                    //     .to_result()
                    //     .map_err(|_| fmt::Error)?;
                    // let cstr = CStr::from_ptr(ptr);
                    write!(f, "{:?}", other)
                // }
            }
            // This shouldn't happen
            _ => write!(f, "Unknown error"),
        }
    }
}
