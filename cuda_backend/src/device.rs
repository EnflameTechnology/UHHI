//! Functions and types for enumerating Device devices and retrieving information about them.

// use uhal::{DeviceResult, ToResult, Device, Devices, DeviceTrait, DeviceAttribute};
pub use cust_raw as driv;

use crate::error::ToResult;
use driv::{CUdevice, CUuuid};
use std::ffi::CStr;
use std::ops::Range;
use uhal::device::{DeviceAttribute, DeviceTrait};
use uhal::error::DeviceResult;

/// Opaque handle to a Device device.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct CuDevice(CUdevice);
/// Iterator over all available devices. See
/// [the Device::devices function](./struct.Device.html#method.devices) for more information.
#[derive(Debug, Clone)]
pub struct CuDevices {
    pub range: Range<u32>,
}

impl Iterator for CuDevices {
    type Item = DeviceResult<CuDevice>;

    fn next(&mut self) -> Option<DeviceResult<CuDevice>> {
        self.range.next().map(CuDevice::get_device)
    }
}
impl CuDevice {
    pub fn new(d: CUdevice) -> Self {
        CuDevice { 0: d }
    }
}
impl DeviceTrait for CuDevice {
    type DeviceT = CuDevice;
    type DevicesT = CuDevices;
    type RawDeviceT = CUdevice;
    /// Get the number of devices.
    /// Returns the number of devices with compute-capability 2.0 or greater which are available
    /// for execution.
    fn num_devices() -> DeviceResult<u32> {
        unsafe {
            let mut num_devices = 0i32;
            driv::cuDeviceGetCount(&mut num_devices as *mut i32).to_result()?;
            Ok(num_devices as u32)
        }
    }

    /// Get a handle to the `ordinal`'th device.
    /// Ordinal must be in the range `0..num_devices()`. If not, an error will be returned.
    fn get_device(ordinal: u32) -> DeviceResult<Self::DeviceT> {
        unsafe {
            let mut device = Self::DeviceT { 0: 0 };
            driv::cuDeviceGet(&mut device.0 as *mut Self::RawDeviceT, ordinal as i32)
                .to_result()?;
            Ok(device)
        }
    }

    /// Return an iterator over all devices.
    fn devices() -> DeviceResult<Self::DevicesT> {
        CuDevice::num_devices().map(|num_devices| Self::DevicesT {
            range: 0..num_devices,
        })
    }

    /// Returns the total amount of memory available on the device in bytes.
    fn total_memory(self) -> DeviceResult<u64> {
        unsafe {
            let mut memory: usize = 0;
            driv::cuDeviceTotalMem_v2(&mut memory as *mut usize, self.0).to_result()?;
            Ok(memory as u64)
        }
    }

    /// Returns the name of this device.
    fn name(self) -> DeviceResult<String> {
        unsafe {
            let mut name = [0u8; 128]; // Hopefully this is big enough...
            driv::cuDeviceGetName(
                &mut name[0] as *mut u8 as *mut ::std::os::raw::c_char,
                128,
                self.0,
            )
            .to_result()?;
            let nul_index = name
                .iter()
                .cloned()
                .position(|byte| byte == 0)
                .expect("Expected device name to fit in 128 bytes and be nul-terminated.");
            let cstr = CStr::from_bytes_with_nul_unchecked(&name[0..=nul_index]);
            Ok(cstr.to_string_lossy().into_owned())
        }
    }

    /// Returns the UUID of this device.
    fn uuid(self) -> DeviceResult<[u8; 16]> {
        let mut cu_uuid = CUuuid { bytes: [0i8; 16] };
        unsafe {
            driv::cuDeviceGetUuid(&mut cu_uuid, self.0).to_result()?;
        }
        let uuid: [u8; 16] = cu_uuid.bytes.map(|byte| byte as u8);
        Ok(uuid)
    }

    /// Returns information about this device.
    fn get_attribute(self, attr: DeviceAttribute) -> DeviceResult<i32> {
        unsafe {
            let mut val = 0i32;
            driv::cuDeviceGetAttribute(
                &mut val as *mut i32,
                // This should be safe, as the repr and values of DeviceAttribute should match.
                ::std::mem::transmute(attr),
                self.0,
            )
            .to_result()?;
            Ok(val)
        }
    }

    /// Returns a raw handle to this device, not handing over ownership, meaning that dropping
    /// this device will try to drop the underlying device.
    fn as_raw(&self) -> Self::RawDeviceT {
        self.0
    }

    fn select_device(ordinal: u32) -> DeviceResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::error::Error;
    use uhal::DriverLibraryTrait;
    use uhal::Flags;
    fn test_init() -> Result<(), Box<dyn Error>> {
        crate::CuApi::init(Flags::empty())?;
        Ok(())
    }

    #[test]
    fn test_num_devices() -> Result<(), Box<dyn Error>> {
        test_init()?;
        let num_devices = CuDevice::num_devices()?;
        assert!(num_devices > 0);
        Ok(())
    }

    #[test]
    fn test_devices() -> Result<(), Box<dyn Error>> {
        test_init()?;
        let num_devices = CuDevice::num_devices()?;
        let all_devices: DeviceResult<Vec<_>> = CuDevice::devices()?.collect();
        let all_devices = all_devices?;
        assert_eq!(num_devices as usize, all_devices.len());
        Ok(())
    }

    #[test]
    fn test_get_name() -> Result<(), Box<dyn Error>> {
        test_init()?;
        let device_name = CuDevice::get_device(0)?.name()?;
        println!("{}", device_name);
        assert!(device_name.len() < 127);
        Ok(())
    }

    #[test]
    fn test_get_memory() -> Result<(), Box<dyn Error>> {
        test_init()?;
        let memory = CuDevice::get_device(0)?.total_memory()?;
        println!("{}", memory);
        Ok(())
    }
}
