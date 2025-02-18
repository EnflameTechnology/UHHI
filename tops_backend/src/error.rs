//! Types for error handling
//!
//! # Error handling in Device:
//!
//! cust uses the [`DeviceError`](enum.DeviceError.html) enum to represent the errors returned by
//! the Device API. It is important to note that nearly every function in Device (and therefore
//! cust) can fail. Even those functions which have no normal failure conditions can return
//! errors related to previous asynchronous launches.
pub use tops_raw as driv;

use driv::topsError_t;
use uhal::error::{DeviceError, DeviceResult};

pub trait ToResult {
    fn to_result(self) -> DeviceResult<()>;
}

impl ToResult for topsError_t {
    fn to_result(self) -> DeviceResult<()> {
        match self {
            topsError_t::topsSuccess => Ok(()),
            topsError_t::topsErrorInvalidValue => Err(DeviceError::InvalidValue),
            topsError_t::topsErrorOutOfMemory => Err(DeviceError::OutOfMemory),
            topsError_t::topsErrorNotInitialized => Err(DeviceError::NotInitialized),
            topsError_t::topsErrorDeinitialized => Err(DeviceError::Deinitialized),
            topsError_t::topsErrorProfilerDisabled => Err(DeviceError::ProfilerDisabled),
            topsError_t::topsErrorProfilerNotInitialized => {
                Err(DeviceError::ProfilerNotInitialized)
            }

            topsError_t::topsErrorProfilerAlreadyStarted => {
                Err(DeviceError::ProfilerAlreadyStarted)
            }
            topsError_t::topsErrorProfilerAlreadyStopped => {
                Err(DeviceError::ProfilerAlreadyStopped)
            }
            topsError_t::topsErrorNoDevice => Err(DeviceError::NoDevice),
            topsError_t::topsErrorInvalidDevice => Err(DeviceError::InvalidDevice),
            topsError_t::topsErrorInvalidImage => Err(DeviceError::InvalidImage),
            topsError_t::topsErrorInvalidContext => Err(DeviceError::InvalidContext),
            topsError_t::topsErrorContextAlreadyCurrent => Err(DeviceError::ContextAlreadyCurrent),
            topsError_t::topsErrorMapFailed => Err(DeviceError::MapFailed),
            topsError_t::topsErrorUnmapFailed => Err(DeviceError::UnmapFailed),
            topsError_t::topsErrorArrayIsMapped => Err(DeviceError::ArrayIsMapped),
            topsError_t::topsErrorAlreadyMapped => Err(DeviceError::AlreadyMapped),
            topsError_t::topsErrorNoBinaryForGcu => Err(DeviceError::NoBinaryForGpu),
            topsError_t::topsErrorAlreadyAcquired => Err(DeviceError::AlreadyAcquired),
            topsError_t::topsErrorNotMapped => Err(DeviceError::NotMapped),
            topsError_t::topsErrorNotMappedAsArray => Err(DeviceError::NotMappedAsArray),
            topsError_t::topsErrorNotMappedAsPointer => Err(DeviceError::NotMappedAsPointer),
            topsError_t::topsErrorECCNotCorrectable => Err(DeviceError::EccUncorrectable),
            topsError_t::topsErrorUnsupportedLimit => Err(DeviceError::UnsupportedLimit),
            topsError_t::topsErrorContextAlreadyInUse => Err(DeviceError::ContextAlreadyInUse),
            topsError_t::topsErrorPeerAccessUnsupported => Err(DeviceError::PeerAccessUnsupported),
            topsError_t::topsErrorInvalidKernelFile => Err(DeviceError::InvalidPtx),
            topsError_t::topsErrorInvalidGraphicsContext => {
                Err(DeviceError::InvalidGraphicsContext)
            }
            // topsError_t::CUDA_ERROR_NVLINK_UNCORRECTABLE => Err(DeviceError::NvlinkUncorrectable),
            topsError_t::topsErrorInvalidSource => Err(DeviceError::InvalidSource),
            topsError_t::topsErrorFileNotFound => Err(DeviceError::FileNotFound),

            topsError_t::topsErrorSharedObjectSymbolNotFound => {
                Err(DeviceError::SharedObjectSymbolNotFound)
            }
            topsError_t::topsErrorSharedObjectInitFailed => {
                Err(DeviceError::SharedObjectInitFailed)
            }
            topsError_t::topsErrorOperatingSystem => Err(DeviceError::OperatingSystemError),
            topsError_t::topsErrorInvalidHandle => Err(DeviceError::InvalidHandle),
            topsError_t::topsErrorNotFound => Err(DeviceError::NotFound),
            topsError_t::topsErrorNotReady => Err(DeviceError::NotReady),
            topsError_t::topsErrorIllegalAddress => Err(DeviceError::IllegalAddress),
            topsError_t::topsErrorLaunchOutOfResources => Err(DeviceError::LaunchOutOfResources),
            topsError_t::topsErrorLaunchTimeOut => Err(DeviceError::LaunchTimeout),
            // topsError_t::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => {
            //     Err(DeviceError::LaunchIncompatibleTexturing)
            // }
            topsError_t::topsErrorPeerAccessAlreadyEnabled => {
                Err(DeviceError::PeerAccessAlreadyEnabled)
            }
            topsError_t::topsErrorPeerAccessNotEnabled => Err(DeviceError::PeerAccessNotEnabled),
            // topsError_t::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE => {
            //     Err(DeviceError::PrimaryContextActive)
            // }
            topsError_t::topsErrorContextIsDestroyed => Err(DeviceError::ContextIsDestroyed),
            topsError_t::topsErrorAssert => Err(DeviceError::AssertError),
            // topsError_t::CUDA_ERROR_TOO_MANY_PEERS => Err(DeviceError::TooManyPeers),
            topsError_t::topsErrorHostMemoryAlreadyRegistered => {
                Err(DeviceError::HostMemoryAlreadyRegistered)
            }
            topsError_t::topsErrorHostMemoryNotRegistered => {
                Err(DeviceError::HostMemoryNotRegistered)
            }
            // topsError_t::CUDA_ERROR_HARDWARE_STACK_ERROR => Err(DeviceError::HardwareStackError),
            // topsError_t::CUDA_ERROR_ILLEGAL_INSTRUCTION => Err(DeviceError::IllegalInstruction),
            // topsError_t::CUDA_ERROR_MISALIGNED_ADDRESS => Err(DeviceError::MisalignedAddress),
            // topsError_t::CUDA_ERROR_INVALID_ADDRESS_SPACE => Err(DeviceError::InvalidAddressSpace),
            // topsError_t::CUDA_ERROR_INVALID_PC => Err(DeviceError::InvalidProgramCounter),
            topsError_t::topsErrorLaunchFailure => Err(DeviceError::LaunchFailed),
            // topsError_t::CUDA_ERROR_NOT_PERMITTED => Err(DeviceError::NotPermitted),
            topsError_t::topsErrorNotSupported => Err(DeviceError::NotSupported),
            _ => Err(DeviceError::UnknownError),
        }
    }
}

// topsErrorInvalidConfiguration = 9,
// topsErrorInvalidPitchValue = 12,
// topsErrorInvalidSymbol = 13,
// topsErrorInvalidDevicePointer = 17,
// topsErrorInvalidMemcpyDirection = 21,
// topsErrorInsufficientDriver = 35,
// topsErrorMissingConfiguration = 52,
// topsErrorPriorLaunchFailure = 53,
// topsErrorInvalidDeviceFunction = 98,

// topsErrorIllegalState = 401,
// topsErrorSetOnActiveProcess = 708,
// topsErrorCooperativeLaunchTooLarge = 720,

// topsErrorStreamCaptureUnsupported = 900,
// topsErrorStreamCaptureInvalidated = 901,
// topsErrorStreamCaptureMerge = 902,
// topsErrorStreamCaptureUnmatched = 903,
// topsErrorStreamCaptureUnjoined = 904,
// topsErrorStreamCaptureIsolation = 905,
// topsErrorStreamCaptureImplicit = 906,
// topsErrorCapturedEvent = 907,
// topsErrorStreamCaptureWrongThread = 908,
// topsErrorGraphExecUpdateFailure = 910,
// topsErrorUnknown = 999,
// topsErrorRuntimeMemory = 1052,
// topsErrorRuntimeOther = 1053,
// topsErrorTbd = 1054,
