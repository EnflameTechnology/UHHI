//! Types for error handling
//!
//! # Error handling in Device:
//!
//! cust uses the [`DeviceError`](enum.DeviceError.html) enum to represent the errors returned by
//! the Device API. It is important to note that nearly every function in Device (and therefore
//! cust) can fail. Even those functions which have no normal failure conditions can return
//! errors related to previous asynchronous launches.
pub use cust_raw as driv;

use driv::cudaError_enum;
use uhal::error::{DeviceError, DeviceResult};

pub(crate) trait ToResult {
    fn to_result(self) -> DeviceResult<()>;
}
// pub trait ToResult {
//     fn to_result(self) -> DeviceResult<()>;
// }

impl ToResult for cudaError_enum {
    fn to_result(self) -> DeviceResult<()> {
        match self {
            cudaError_enum::CUDA_SUCCESS => Ok(()),
            cudaError_enum::CUDA_ERROR_INVALID_VALUE => Err(DeviceError::InvalidValue),
            cudaError_enum::CUDA_ERROR_OUT_OF_MEMORY => Err(DeviceError::OutOfMemory),
            cudaError_enum::CUDA_ERROR_NOT_INITIALIZED => Err(DeviceError::NotInitialized),
            cudaError_enum::CUDA_ERROR_DEINITIALIZED => Err(DeviceError::Deinitialized),
            cudaError_enum::CUDA_ERROR_PROFILER_DISABLED => Err(DeviceError::ProfilerDisabled),
            cudaError_enum::CUDA_ERROR_PROFILER_NOT_INITIALIZED => {
                Err(DeviceError::ProfilerNotInitialized)
            }
            cudaError_enum::CUDA_ERROR_PROFILER_ALREADY_STARTED => {
                Err(DeviceError::ProfilerAlreadyStarted)
            }
            cudaError_enum::CUDA_ERROR_PROFILER_ALREADY_STOPPED => {
                Err(DeviceError::ProfilerAlreadyStopped)
            }
            cudaError_enum::CUDA_ERROR_NO_DEVICE => Err(DeviceError::NoDevice),
            cudaError_enum::CUDA_ERROR_INVALID_DEVICE => Err(DeviceError::InvalidDevice),
            cudaError_enum::CUDA_ERROR_INVALID_IMAGE => Err(DeviceError::InvalidImage),
            cudaError_enum::CUDA_ERROR_INVALID_CONTEXT => Err(DeviceError::InvalidContext),
            cudaError_enum::CUDA_ERROR_CONTEXT_ALREADY_CURRENT => {
                Err(DeviceError::ContextAlreadyCurrent)
            }
            cudaError_enum::CUDA_ERROR_MAP_FAILED => Err(DeviceError::MapFailed),
            cudaError_enum::CUDA_ERROR_UNMAP_FAILED => Err(DeviceError::UnmapFailed),
            cudaError_enum::CUDA_ERROR_ARRAY_IS_MAPPED => Err(DeviceError::ArrayIsMapped),
            cudaError_enum::CUDA_ERROR_ALREADY_MAPPED => Err(DeviceError::AlreadyMapped),
            cudaError_enum::CUDA_ERROR_NO_BINARY_FOR_GPU => Err(DeviceError::NoBinaryForGpu),
            cudaError_enum::CUDA_ERROR_ALREADY_ACQUIRED => Err(DeviceError::AlreadyAcquired),
            cudaError_enum::CUDA_ERROR_NOT_MAPPED => Err(DeviceError::NotMapped),
            cudaError_enum::CUDA_ERROR_NOT_MAPPED_AS_ARRAY => Err(DeviceError::NotMappedAsArray),
            cudaError_enum::CUDA_ERROR_NOT_MAPPED_AS_POINTER => {
                Err(DeviceError::NotMappedAsPointer)
            }
            cudaError_enum::CUDA_ERROR_ECC_UNCORRECTABLE => Err(DeviceError::EccUncorrectable),
            cudaError_enum::CUDA_ERROR_UNSUPPORTED_LIMIT => Err(DeviceError::UnsupportedLimit),
            cudaError_enum::CUDA_ERROR_CONTEXT_ALREADY_IN_USE => {
                Err(DeviceError::ContextAlreadyInUse)
            }
            cudaError_enum::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => {
                Err(DeviceError::PeerAccessUnsupported)
            }
            cudaError_enum::CUDA_ERROR_INVALID_PTX => Err(DeviceError::InvalidPtx),
            cudaError_enum::CUDA_ERROR_INVALID_GRAPHICS_CONTEXT => {
                Err(DeviceError::InvalidGraphicsContext)
            }
            cudaError_enum::CUDA_ERROR_NVLINK_UNCORRECTABLE => {
                Err(DeviceError::NvlinkUncorrectable)
            }
            cudaError_enum::CUDA_ERROR_INVALID_SOURCE => Err(DeviceError::InvalidSource),
            cudaError_enum::CUDA_ERROR_FILE_NOT_FOUND => Err(DeviceError::FileNotFound),
            cudaError_enum::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND => {
                Err(DeviceError::SharedObjectSymbolNotFound)
            }
            cudaError_enum::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => {
                Err(DeviceError::SharedObjectInitFailed)
            }
            cudaError_enum::CUDA_ERROR_OPERATING_SYSTEM => Err(DeviceError::OperatingSystemError),
            cudaError_enum::CUDA_ERROR_INVALID_HANDLE => Err(DeviceError::InvalidHandle),
            cudaError_enum::CUDA_ERROR_NOT_FOUND => Err(DeviceError::NotFound),
            cudaError_enum::CUDA_ERROR_NOT_READY => Err(DeviceError::NotReady),
            cudaError_enum::CUDA_ERROR_ILLEGAL_ADDRESS => Err(DeviceError::IllegalAddress),
            cudaError_enum::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => {
                Err(DeviceError::LaunchOutOfResources)
            }
            cudaError_enum::CUDA_ERROR_LAUNCH_TIMEOUT => Err(DeviceError::LaunchTimeout),
            cudaError_enum::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => {
                Err(DeviceError::LaunchIncompatibleTexturing)
            }
            cudaError_enum::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => {
                Err(DeviceError::PeerAccessAlreadyEnabled)
            }
            cudaError_enum::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => {
                Err(DeviceError::PeerAccessNotEnabled)
            }
            cudaError_enum::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE => {
                Err(DeviceError::PrimaryContextActive)
            }
            cudaError_enum::CUDA_ERROR_CONTEXT_IS_DESTROYED => Err(DeviceError::ContextIsDestroyed),
            cudaError_enum::CUDA_ERROR_ASSERT => Err(DeviceError::AssertError),
            cudaError_enum::CUDA_ERROR_TOO_MANY_PEERS => Err(DeviceError::TooManyPeers),
            cudaError_enum::CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => {
                Err(DeviceError::HostMemoryAlreadyRegistered)
            }
            cudaError_enum::CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED => {
                Err(DeviceError::HostMemoryNotRegistered)
            }
            cudaError_enum::CUDA_ERROR_HARDWARE_STACK_ERROR => Err(DeviceError::HardwareStackError),
            cudaError_enum::CUDA_ERROR_ILLEGAL_INSTRUCTION => Err(DeviceError::IllegalInstruction),
            cudaError_enum::CUDA_ERROR_MISALIGNED_ADDRESS => Err(DeviceError::MisalignedAddress),
            cudaError_enum::CUDA_ERROR_INVALID_ADDRESS_SPACE => {
                Err(DeviceError::InvalidAddressSpace)
            }
            cudaError_enum::CUDA_ERROR_INVALID_PC => Err(DeviceError::InvalidProgramCounter),
            cudaError_enum::CUDA_ERROR_LAUNCH_FAILED => Err(DeviceError::LaunchFailed),
            cudaError_enum::CUDA_ERROR_NOT_PERMITTED => Err(DeviceError::NotPermitted),
            cudaError_enum::CUDA_ERROR_NOT_SUPPORTED => Err(DeviceError::NotSupported),
            _ => Err(DeviceError::UnknownError),
        }
    }
}
