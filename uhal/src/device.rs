//! Functions and types for enumerating Device devices and retrieving information about them.

use crate::error::DeviceResult;
// use std::ffi::CStr;
// use std::ops::Range;

/// All supported device attributes for [Device::get_attribute](struct.Device.html#method.get_attribute)
#[repr(u32)]
#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DeviceAttribute {
    /// Maximum number of threads per block
    MaxThreadsPerBlock = 1,
    /// Maximum x-dimension of a block
    MaxBlockDimX = 2,
    /// Maximum y-dimension of a block
    MaxBlockDimY = 3,
    /// Maximum z-dimension of a block
    MaxBlockDimZ = 4,
    /// Maximum x-dimension of a grid
    MaxGridDimX = 5,
    /// Maximum y-dimension of a grid
    MaxGridDimY = 6,
    /// Maximum z-dimension of a grid
    MaxGridDimZ = 7,
    /// Maximum amount of shared memory available to a thread block in bytes
    MaxSharedMemoryPerBlock = 8,
    /// Memory available on device for constant variables in a kernel in bytes
    TotalConstantMemory = 9,
    /// Warp size in threads
    WarpSize = 10,
    /// Maximum pitch in bytes allowed by the memory copy functions that involve memory regions
    /// allocated through cuMemAllocPitch()
    MaxPitch = 11,
    /// Maximum number of 32-bit registers available to a thread block
    MaxRegistersPerBlock = 12,
    /// Typical clock frequency in kilohertz
    ClockRate = 13,
    /// Alignment requirement for textures
    TextureAlignment = 14,
    //GpuOverlap = 15, - Deprecated.
    /// Number of multiprocessors on device.
    MultiprocessorCount = 16,
    /// Specifies whether there is a run time limit on kernels
    KernelExecTimeout = 17,
    /// Device is integrated with host memory
    Integrated = 18,
    /// Device can map host memory into CUDA address space
    CanMapHostMemory = 19,
    /// Compute Mode
    ComputeMode = 20,
    /// Maximum 1D texture width
    MaximumTexture1DWidth = 21,
    /// Maximum 2D texture width
    MaximumTexture2DWidth = 22,
    /// Maximum 2D texture height
    MaximumTexture2DHeight = 23,
    /// Maximum 3D texture width
    MaximumTexture3DWidth = 24,
    /// Maximum 3D texture height
    MaximumTexture3DHeight = 25,
    /// Maximum 3D texture depth
    MaximumTexture3DDepth = 26,
    /// Maximum 2D layered texture width
    MaximumTexture2DLayeredWidth = 27,
    /// Maximum 2D layered texture height
    MaximumTexture2DLayeredHeight = 28,
    /// Maximum layers in a 2D layered texture
    MaximumTexture2DLayeredLayers = 29,
    /// Alignment requirement for surfaces
    SurfaceAlignment = 30,
    /// Device can possibly execute multiple kernels concurrently
    ConcurrentKernels = 31,
    /// Device has ECC support enabled
    EccEnabled = 32,
    /// PCI bus ID of the device
    PciBusId = 33,
    /// PCI device ID of the device
    PciDeviceId = 34,
    /// Device is using TCC driver model
    TccDriver = 35,
    /// Peak memory clock frequency in kilohertz
    MemoryClockRate = 36,
    /// Global memory bus width in bits
    GlobalMemoryBusWidth = 37,
    /// Size of L2 cache in bytes.
    L2CacheSize = 38,
    /// Maximum resident threads per multiprocessor
    MaxThreadsPerMultiprocessor = 39,
    /// Number of asynchronous engines
    AsyncEngineCount = 40,
    /// Device shares a unified address space with the host
    UnifiedAddressing = 41,
    /// Maximum 1D layered texture width
    MaximumTexture1DLayeredWidth = 42,
    /// Maximum layers in a 1D layered texture
    MaximumTexture1DLayeredLayers = 43,
    //CanTex2DGather = 44, deprecated
    /// Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set
    MaximumTexture2DGatherWidth = 45,
    /// Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set
    MaximumTexture2DGatherHeight = 46,
    /// Alternate maximum 3D texture width
    MaximumTexture3DWidthAlternate = 47,
    /// Alternate maximum 3D texture height
    MaximumTexture3DHeightAlternate = 48,
    /// Alternate maximum 3D texture depth
    MaximumTexture3DDepthAlternate = 49,
    /// PCI domain ID of the device
    PciDomainId = 50,
    /// Pitch alignment requirement for textures
    TexturePitchAlignment = 51,
    /// Maximum cubemap texture width/height
    MaximumTextureCubemapWidth = 52,
    /// Maximum cubemap layered texture width/height
    MaximumTextureCubemapLayeredWidth = 53,
    /// Maximum layers in a cubemap layered texture
    MaximumTextureCubemapLayeredLayers = 54,
    /// Maximum 1D surface width
    MaximumSurface1DWidth = 55,
    /// Maximum 2D surface width
    MaximumSurface2DWidth = 56,
    /// Maximum 2D surface height
    MaximumSurface2DHeight = 57,
    /// Maximum 3D surface width
    MaximumSurface3DWidth = 58,
    /// Maximum 3D surface height
    MaximumSurface3DHeight = 59,
    /// Maximum 3D surface depth
    MaximumSurface3DDepth = 60,
    /// Maximum 1D layered surface width
    MaximumSurface1DLayeredWidth = 61,
    /// Maximum layers in a 1D layered surface
    MaximumSurface1DLayeredLayers = 62,
    /// Maximum 2D layered surface width
    MaximumSurface2DLayeredWidth = 63,
    /// Maximum 2D layered surface height
    MaximumSurface2DLayeredHeight = 64,
    /// Maximum layers in a 2D layered surface
    MaximumSurface2DLayeredLayers = 65,
    /// Maximum cubemap surface width
    MaximumSurfacecubemapWidth = 66,
    /// Maximum cubemap layered surface width
    MaximumSurfacecubemapLayeredWidth = 67,
    /// Maximum layers in a cubemap layered surface
    MaximumSurfacecubemapLayeredLayers = 68,
    /// Maximum 1D linear texture width
    MaximumTexture1DLinearWidth = 69,
    /// Maximum 2D linear texture width
    MaximumTexture2DLinearWidth = 70,
    /// Maximum 2D linear texture height
    MaximumTexture2DLinearHeight = 71,
    /// Maximum 2D linear texture pitch in bytes
    MaximumTexture2DLinearPitch = 72,
    /// Maximum mipmapped 2D texture height
    MaximumTexture2DMipmappedWidth = 73,
    /// Maximum mipmapped 2D texture width
    MaximumTexture2DMipmappedHeight = 74,
    /// Major compute capability version number
    ComputeCapabilityMajor = 75,
    /// Minor compute capability version number
    ComputeCapabilityMinor = 76,
    /// Maximum mipammed 1D texture width
    MaximumTexture1DMipmappedWidth = 77,
    /// Device supports stream priorities
    StreamPrioritiesSupported = 78,
    /// Device supports caching globals in L1
    GlobalL1CacheSupported = 79,
    /// Device supports caching locals in L1
    LocalL1CacheSupported = 80,
    /// Maximum shared memory available per multiprocessor in bytes
    MaxSharedMemoryPerMultiprocessor = 81,
    /// Maximum number of 32-bit registers available per multiprocessor
    MaxRegistersPerMultiprocessor = 82,
    /// Device can allocate managed memory on this system
    ManagedMemory = 83,
    /// Device is on a multi-GPU board
    MultiGpuBoard = 84,
    /// Unique ID for a group of devices on the same multi-GPU board
    MultiGpuBoardGroupId = 85,
    /// Link between the device and the host supports native atomic operations (this is a
    /// placeholder attribute and is not supported on any current hardware)
    HostNativeAtomicSupported = 86,
    /// Ratio of single precision performance (in floating-point operations per second) to double
    /// precision performance
    SingleToDoublePrecisionPerfRatio = 87,
    /// Device supports coherently accessing pageable memory without calling cudaHostRegister on it.
    PageableMemoryAccess = 88,
    /// Device can coherently access managed memory concurrently with the CPU
    ConcurrentManagedAccess = 89,
    /// Device supports compute preemption
    ComputePreemptionSupported = 90,
    /// Device can access host registered memory at the same virtual address as the CPU
    CanUseHostPointerForRegisteredMem = 91,
}

pub trait DeviceTrait {
    type DeviceT;
    type DevicesT;
    type RawDeviceT;
    /// Get the number of devices.
    /// Returns the number of devices with compute-capability 2.0 or greater which are available
    /// for execution.
    fn num_devices() -> DeviceResult<u32>;

    /// Get a handle to the `ordinal`'th device.
    /// Ordinal must be in the range `0..num_devices()`. If not, an error will be returned.
    fn get_device(ordinal: u32) -> DeviceResult<Self::DeviceT>;

    fn select_device(ordinal: u32) -> DeviceResult<()>;
    /// Return an iterator over all devices.
    fn devices() -> DeviceResult<Self::DevicesT>;

    /// Returns the total amount of memory available on the device in bytes.
    fn total_memory(self) -> DeviceResult<u64>;

    /// Returns the name of this device.
    fn name(self) -> DeviceResult<String>;

    /// Returns the UUID of this device.
    fn uuid(self) -> DeviceResult<[u8; 16]>;

    /// Returns information about this device.
    fn get_attribute(self, attr: DeviceAttribute) -> DeviceResult<i32>;

    /// Returns a raw handle to this device, not handing over ownership, meaning that dropping
    /// this device will try to drop the underlying device.
    fn as_raw(&self) -> Self::RawDeviceT;
}
