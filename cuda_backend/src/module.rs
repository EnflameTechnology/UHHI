//! Functions and types for working with Device modules.
pub use cust_core::_hidden::DeviceCopy;
pub use cust_raw as driv;
pub use driv::{CUjit_option, CUmodule};

use uhal::function::FunctionTrait;
use uhal::memory::DevicePointerTrait;
use uhal::module::{ModuleJitOption, ModuleTrait};
// use uhal::function::{Function};
use uhal::error::{DeviceResult, DropResult};
// use uhal::memory::{CopyDestination};

use std::ffi::{c_void, CStr, CString};
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::os::raw::c_uint;
use std::path::Path;
use std::ptr;

use crate::error::ToResult;
use crate::function::CuFunction;
use crate::memory::{CopyDestination, CuDevicePointer};
///TODO
pub struct CuModuleJitOption {}

#[derive(Debug)]
pub struct CuModule(CUmodule);

unsafe impl Sync for CuModule {}

#[cfg(unix)]
fn path_to_bytes<P: AsRef<Path>>(path: P) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    path.as_ref().as_os_str().as_bytes().to_vec()
}

#[cfg(not(unix))]
fn path_to_bytes<P: AsRef<Path>>(path: P) -> Vec<u8> {
    path.as_ref().to_string_lossy().to_string().into_bytes()
}

impl CuModuleJitOption {
    pub fn into_raw(opts: &[ModuleJitOption]) -> (Vec<CUjit_option>, Vec<*mut c_void>) {
        // And here we stumble across one of the most horrific things i have ever seen in my entire
        // journey of working with many parts of Device. As a background, Device usually wants an array
        // of pointers to values when it takes void**, after all, this is what is expected by anyone.
        // However, there is a SINGLE exception in the entire driver API, and that is cuModuleLoadDataEx,
        // it actually wants you to pass values by value instead of by ref if they fit into pointer length.
        // Therefore something like MaxRegisters should be passed as `u32 as usize as *mut c_void`.
        // This is completely undocumented. I initially brought this up to an nvidia developer,
        // who eventually was able to figure out this issue, currently it appears to be labeled "not a bug",
        // however this will likely be changed in the future, or at least get documented better. (hopefully)
        let mut raw_opts = Vec::with_capacity(opts.len());
        let mut raw_vals = Vec::with_capacity(opts.len());

        for opt in opts {
            match opt {
                ModuleJitOption::MaxRegisters(regs) => {
                    raw_opts.push(CUjit_option::CU_JIT_MAX_REGISTERS);
                    raw_vals.push(*regs as usize as *mut c_void);
                }
                ModuleJitOption::OptLevel(level) => {
                    raw_opts.push(CUjit_option::CU_JIT_OPTIMIZATION_LEVEL);
                    raw_vals.push(*level as usize as *mut c_void);
                }
                ModuleJitOption::DetermineTargetFromContext => {
                    raw_opts.push(CUjit_option::CU_JIT_TARGET_FROM_CUCONTEXT);
                }
                ModuleJitOption::Target(target) => {
                    raw_opts.push(CUjit_option::CU_JIT_TARGET);
                    raw_vals.push(*target as usize as *mut c_void);
                }
                ModuleJitOption::Fallback(fallback) => {
                    raw_opts.push(CUjit_option::CU_JIT_FALLBACK_STRATEGY);
                    raw_vals.push(*fallback as usize as *mut c_void);
                }
                ModuleJitOption::GenenerateDebugInfo(gen) => {
                    raw_opts.push(CUjit_option::CU_JIT_GENERATE_DEBUG_INFO);
                    raw_vals.push(*gen as usize as *mut c_void);
                }
                ModuleJitOption::GenerateLineInfo(gen) => {
                    raw_opts.push(CUjit_option::CU_JIT_GENERATE_LINE_INFO);
                    raw_vals.push(*gen as usize as *mut c_void)
                }
                _ => {}
            }
        }
        (raw_opts, raw_vals)
    }
}

pub struct CuSymbol<'a, T: DeviceCopy> {
    pub ptr: CuDevicePointer<T>,
    pub module: PhantomData<&'a CuModule>,
}

impl ModuleTrait for CuModule {
    type ModuleT = CuModule;
    // type SymbolT = CuSymbol<'a>;
    // type FunctionT = CuFunction<'a>;
    /// Load a module from the given path into the current context.
    ///
    /// The given path should be either a cubin file, a ptx file, or a fatbin file such as
    /// those produced by `nvcc`.
    fn from_file<P: AsRef<Path>>(path: P) -> DeviceResult<Self::ModuleT> {
        unsafe {
            let mut bytes = path_to_bytes(path);
            if !bytes.contains(&0) {
                bytes.push(0);
            }
            let mut module = CuModule { 0: ptr::null_mut() };
            driv::cuModuleLoad(&mut module.0 as *mut CUmodule, bytes.as_ptr() as *const _)
                .to_result()?;
            Ok(module)
        }
    }

    /// Creates a new module by loading a fatbin (fat binary) file.
    ///
    /// Fatbinary files are files that contain multiple ptx or cubin files. The driver will choose already-built
    /// cubin if it is present, and otherwise JIT compile any PTX in the file to cubin.
    fn from_fatbin<G: AsRef<[u8]>>(
        bytes: G,
        options: &[ModuleJitOption],
    ) -> DeviceResult<Self::ModuleT> {
        // fatbins can be loaded just like cubins, we just use different methods so it's explicit.
        // please don't use from_cubin for fatbins, that is pure chaos and ferris will come to your house
        Self::from_cubin(bytes, options)
    }

    /// Creates a new module by loading a cubin (Device Binary) file.
    ///
    /// Cubins are architecture/compute-capability specific files generated as the final step of the Device compilation
    /// process. They cannot be interchanged across compute capabilities unlike PTX (to some degree). You can create one
    /// using the PTX compiler APIs, the cust [`Linker`](crate::link::Linker), or nvcc (`nvcc a.ptx --cubin -arch=sm_XX`).
    fn from_cubin<G: AsRef<[u8]>>(
        bytes: G,
        options: &[ModuleJitOption],
    ) -> DeviceResult<Self::ModuleT> {
        // it is very unclear whether cuda wants or doesn't want a null terminator. The method works
        // whether you have one or not. So for safety we just add one. In theory you can figure out the
        // length of an ELF image without a null terminator. But the docs are confusing, so we add one just
        // to be sure.
        let mut bytes = bytes.as_ref().to_vec();
        bytes.push(0);
        // SAFETY: the image is known to be dereferenceable
        unsafe { Self::load_module(bytes.as_ptr() as *const c_void, options) }
    }

    unsafe fn load_module(
        image: *const c_void,
        options: &[ModuleJitOption],
    ) -> DeviceResult<Self::ModuleT> {
        let mut module = CuModule { 0: ptr::null_mut() };
        let (mut options, mut option_values) = CuModuleJitOption::into_raw(options);
        driv::cuModuleLoadDataEx(
            &mut module.0 as *mut CUmodule,
            image,
            options.len() as c_uint,
            options.as_mut_ptr(),
            option_values.as_mut_ptr(),
        )
        .to_result()?;
        Ok(module)
    }

    /// Creates a new module from a [`CStr`] pointing to PTX code.
    ///
    /// The driver will JIT the PTX into arch-specific cubin or pick already-cached cubin if available.
    fn from_ptx_cstr(cstr: &CStr, options: &[ModuleJitOption]) -> DeviceResult<Self::ModuleT> {
        // SAFETY: the image is known to be dereferenceable
        unsafe { Self::load_module(cstr.as_ptr() as *const c_void, options) }
    }

    /// Creates a new module from a PTX string, allocating an intermediate buffer for the [`CString`].
    ///
    /// The driver will JIT the PTX into arch-specific cubin or pick already-cached cubin if available.
    ///
    /// # Panics
    ///
    /// Panics if `string` contains a nul.
    fn from_ptx<G: AsRef<str>>(
        string: G,
        options: &[ModuleJitOption],
    ) -> DeviceResult<Self::ModuleT> {
        let cstr = CString::new(string.as_ref())
            .expect("string given to Module::from_str contained nul bytes");
        Self::from_ptx_cstr(cstr.as_c_str(), options)
    }

    /// Load a module from a normal (rust) string, implicitly making it into
    /// a cstring.
    // #[deprecated(
    //     since = "0.3.0",
    //     note = "from_str was too generic of a name, use from_ptx instead, passing an empty slice of options (usually)"
    // )]
    #[allow(clippy::should_implement_trait)]
    fn from_str<G: AsRef<str>>(string: G) -> DeviceResult<Self::ModuleT> {
        let cstr = CString::new(string.as_ref())
            .expect("string given to Module::from_str contained nul bytes");
        #[allow(deprecated)]
        Self::load_from_string(cstr.as_c_str())
    }

    /// Load a module from a CStr.
    ///
    /// This is useful in combination with `include_str!`, to include the device code into the
    /// compiled executable.
    ///
    /// The given CStr must contain the bytes of a cubin file, a ptx file or a fatbin file such as
    /// those produced by `nvcc`.
    // #[deprecated(
    //     since = "0.3.0",
    //     note = "load_from_string was an inconsistent name with inconsistent params, use from_ptx/from_ptx_cstr, passing
    // an empty slice of options (usually)
    // "
    // )]
    fn load_from_string(image: &CStr) -> DeviceResult<Self::ModuleT> {
        unsafe {
            let mut module = CuModule { 0: ptr::null_mut() };
            driv::cuModuleLoadData(
                &mut module.0 as *mut CUmodule,
                image.as_ptr() as *const c_void,
            )
            .to_result()?;
            Ok(module)
        }
    }

    /// Destroy a `Module`, returning an error.
    ///
    /// Destroying a module can return errors from previous asynchronous work. This function
    /// destroys the given module and returns the error and the un-destroyed module on failure.
    fn drop(mut module: Self::ModuleT) -> DropResult<Self::ModuleT> {
        if module.0.is_null() {
            return Ok(());
        }

        unsafe {
            let inner = mem::replace(&mut module.0, ptr::null_mut());
            match driv::cuModuleUnload(inner).to_result() {
                Ok(()) => {
                    mem::forget(module.0);
                    Ok(())
                }
                Err(e) => Err((e, CuModule { 0: inner })),
            }
        }
    }
}

impl CuModule {
    /// Get a reference to a global symbol, which can then be copied to/from.
    ///
    /// # Panics:
    ///
    /// This function panics if the size of the symbol is not the same as the `mem::sizeof<T>()`.
    pub fn get_global<'a, T: DeviceCopy>(&'a self, name: &CStr) -> DeviceResult<CuSymbol<'a, T>> {
        unsafe {
            let mut ptr: CuDevicePointer<T> = CuDevicePointer::null();
            let mut size: usize = 0;

            driv::cuModuleGetGlobal_v2(
                &mut ptr as *mut CuDevicePointer<T> as *mut driv::CUdeviceptr,
                &mut size as *mut usize,
                self.0,
                name.as_ptr(),
            )
            .to_result()?;
            assert_eq!(size, mem::size_of::<T>());
            Ok(CuSymbol {
                ptr,
                module: PhantomData,
            })
        }
    }

    /// Get a reference to a kernel function which can then be launched.
    pub fn get_function<'a, P: AsRef<str>>(&'a self, name: P) -> DeviceResult<CuFunction> {
        unsafe {
            let name = name.as_ref();
            let cstr = CString::new(name).expect("Argument to get_function had a nul");
            let mut func: driv::CUfunction = ptr::null_mut();

            driv::cuModuleGetFunction(&mut func as *mut driv::CUfunction, self.0, cstr.as_ptr())
                .to_result()?;
            Ok(CuFunction::new(func, self))
        }
    }
}

impl Drop for CuModule {
    fn drop(&mut self) {
        if self.0.is_null() {
            return;
        }
        unsafe {
            // No choice but to panic if this fails...
            let module = mem::replace(&mut self.0, ptr::null_mut());
            driv::cuModuleUnload(module);
        }
    }
}

impl<'a, T: DeviceCopy> crate::memory::private::Sealed for CuSymbol<'a, T> {}
impl<'a, T: DeviceCopy> fmt::Pointer for CuSymbol<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, f)
    }
}

impl<'a, T: DeviceCopy> CopyDestination<T> for CuSymbol<'a, T> {
    fn copy_from(&mut self, val: &T) -> DeviceResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                driv::cuMemcpyHtoD_v2(self.ptr.as_raw(), val as *const T as *const c_void, size)
                    .to_result()?
            }
        }
        Ok(())
    }

    fn copy_to(&self, val: &mut T) -> DeviceResult<()> {
        let size = mem::size_of::<T>();
        if size != 0 {
            unsafe {
                driv::cuMemcpyDtoH_v2(
                    val as *const T as *mut c_void,
                    self.ptr.as_raw() as u64,
                    size,
                )
                .to_result()?
            }
        }
        Ok(())
    }
}
