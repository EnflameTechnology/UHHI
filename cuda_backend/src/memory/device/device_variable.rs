pub use cust_core::_hidden::DeviceCopy;
use std::ops::{Deref, DerefMut};
use uhal::error::DeviceResult;
use uhal::memory::{DeviceBoxTrait, DeviceVariableTrait};

use crate::memory::CuDevicePointer;

use super::{CopyDestination, CuDeviceBox};
/// Wrapper around a variable on the host and a [`DeviceBox`] holding the
/// variable on the device, allowing for easy synchronization and storage.
#[derive(Debug)]
pub struct CuDeviceVariable<T: DeviceCopy> {
    mem: CuDeviceBox<T>,
    var: T,
}

impl<T: DeviceCopy> DeviceVariableTrait<T> for CuDeviceVariable<T> {
    type DevicePointerT = CuDevicePointer<T>;
    type DeviceVariableT = CuDeviceVariable<T>;
    /// Create a new `DeviceVariable` wrapping `var`.
    ///
    /// Allocates storage on the device and copies `var` to the device.
    fn new(var: T) -> DeviceResult<Self::DeviceVariableT> {
        let mem = CuDeviceBox::new(&var)?;
        Ok(CuDeviceVariable { mem, var })
    }

    /// Copy the host copy of the variable to the device
    fn copy_htod(&mut self) -> DeviceResult<()> {
        self.mem.copy_from(&self.var)
    }

    /// Copy the device copy of the variable to the host
    fn copy_dtoh(&mut self) -> DeviceResult<()> {
        self.mem.copy_to(&mut self.var)
    }

    fn as_device_ptr(&self) -> Self::DevicePointerT {
        self.mem.as_device_ptr()
    }
}

impl<T: DeviceCopy> Deref for CuDeviceVariable<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.var
    }
}

impl<T: DeviceCopy> DerefMut for CuDeviceVariable<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.var
    }
}
