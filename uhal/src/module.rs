//! Functions and types for working with Device modules.

// use cust_core::DeviceCopy;

use cust_core::DeviceCopy;

use crate::error::{DeviceResult, DropResult};
// use crate::function::Function;
// use crate::memory::{DevicePointer};
use std::ffi::{c_void, CStr};
// use std::marker::PhantomData;
use std::path::Path;

// /// A compiled module, loaded into a context.
// #[derive(Debug)]
// pub struct Module<T> {
//     pub inner: T,
// }

/// The possible optimization levels when JIT compiling a PTX module. `O4` by default (most optimized).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptLevel {
    O0 = 0,
    O1 = 1,
    O2 = 2,
    O3 = 3,
    O4 = 4,
}

/// The possible targets when JIT compiling a PTX module.
#[non_exhaustive]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JitTarget {
    Compute20 = 20,
    Compute21 = 21,
    Compute30 = 30,
    Compute32 = 32,
    Compute35 = 35,
    Compute37 = 37,
    Compute50 = 50,
    Compute52 = 52,
    Compute53 = 53,
    Compute60 = 60,
    Compute61 = 61,
    Compute62 = 62,
    Compute70 = 70,
    Compute72 = 72,
    Compute75 = 75,
    Compute80 = 80,
    Compute86 = 86,
}

/// How to handle cases where a loaded module's data does not contain an exact match for the
/// specified architecture.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JitFallback {
    /// Prefer to compile PTX if present if an exact binary match is not found.
    PreferPtx = 0,
    /// Prefer to fall back to a compatible binary code match if exact match is not found.
    /// This means the driver may pick binary code for `7.0` if your device is `7.2` for example.
    PreferCompatibleBinary = 1,
}

/// Different options that could be applied when loading a module.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModuleJitOption {
    /// Specifies the maximum amount of registers any compiled PTX is allowed to use.
    MaxRegisters(u32),
    /// Specifies the optimization level for the JIT compiler.
    OptLevel(OptLevel),
    /// Determines the PTX target from the current context's architecture. Cannot be combined with
    /// [`ModuleJitOption::Target`].
    DetermineTargetFromContext,
    /// Specifies the target for the JIT compiler. Cannot be combined with [`ModuleJitOption::DetermineTargetFromContext`].
    Target(JitTarget),
    /// Specifies how to handle cases where a loaded module's data does not have an exact match for the specified
    /// architecture.
    Fallback(JitFallback),
    /// Generates debug info in the compiled binary.
    GenenerateDebugInfo(bool),
    /// Generates line info in the compiled binary.
    GenerateLineInfo(bool),
}

pub trait ModuleTrait {
    type ModuleT;
    // type SymbolT;
    // type FunctionT;
    /// Load a module from the given path into the current context.
    ///
    /// The given path should be either a cubin file, a ptx file, or a fatbin file such as
    /// those produced by `nvcc`.
    fn from_file<P: AsRef<Path>>(path: P) -> DeviceResult<Self::ModuleT>;

    /// Creates a new module by loading a fatbin (fat binary) file.
    ///
    /// Fatbinary files are files that contain multiple ptx or cubin files. The driver will choose already-built
    /// cubin if it is present, and otherwise JIT compile any PTX in the file to cubin.
    fn from_fatbin<G: AsRef<[u8]>>(
        bytes: G,
        options: &[ModuleJitOption],
    ) -> DeviceResult<Self::ModuleT>;

    /// Creates a new module by loading a cubin (Device Binary) file.
    ///
    /// Cubins are architecture/compute-capability specific files generated as the final step of the Device compilation
    /// process. They cannot be interchanged across compute capabilities unlike PTX (to some degree). You can create one
    /// using the PTX compiler APIs, the cust [`Linker`](crate::link::Linker), or nvcc (`nvcc a.ptx --cubin -arch=sm_XX`).
    fn from_cubin<G: AsRef<[u8]>>(bytes: G, options: &[ModuleJitOption]) -> DeviceResult<Self::ModuleT>;

    unsafe fn load_module(image: *const c_void, options: &[ModuleJitOption]) -> DeviceResult<Self::ModuleT>;

    /// Creates a new module from a [`CStr`] pointing to PTX code.
    ///
    /// The driver will JIT the PTX into arch-specific cubin or pick already-cached cubin if available.
    fn from_ptx_cstr(cstr: &CStr, options: &[ModuleJitOption]) -> DeviceResult<Self::ModuleT>;

    /// Creates a new module from a PTX string, allocating an intermediate buffer for the [`CString`].
    ///
    /// The driver will JIT the PTX into arch-specific cubin or pick already-cached cubin if available.
    ///
    /// # Panics
    ///
    /// Panics if `string` contains a nul.
    fn from_ptx<G: AsRef<str>>(string: G, options: &[ModuleJitOption]) -> DeviceResult<Self::ModuleT>;

    /// Load a module from a normal (rust) string, implicitly making it into
    /// a cstring.
    #[deprecated(
        since = "0.3.0",
        note = "from_str was too generic of a name, use from_ptx instead, passing an empty slice of options (usually)"
    )]
    #[allow(clippy::should_implement_trait)]
    fn from_str<G: AsRef<str>>(string: G) -> DeviceResult<Self::ModuleT>;

    /// Load a module from a CStr.
    ///
    /// This is useful in combination with `include_str!`, to include the device code into the
    /// compiled executable.
    ///
    /// The given CStr must contain the bytes of a cubin file, a ptx file or a fatbin file such as
    /// those produced by `nvcc`.
    #[deprecated(
        since = "0.3.0",
        note = "load_from_string was an inconsistent name with inconsistent params, use from_ptx/from_ptx_cstr, passing 
    an empty slice of options (usually)
    "
    )]
    fn load_from_string(image: &CStr) -> DeviceResult<Self::ModuleT>;

    /// Get a reference to a global symbol, which can then be copied to/from.
    ///
    /// # Panics:
    ///
    /// This function panics if the size of the symbol is not the same as the `mem::sizeof<T>()`.
    // fn get_global<'a, T: DeviceCopy>(&'a self, name: &CStr) -> DeviceResult<Self::SymbolT>;

    /// Get a reference to a kernel function which can then be launched.
    // fn get_function<'a, G: AsRef<str>>(&'a self, name: G) -> DeviceResult<Self::FunctionT>;

    /// Destroy a `Module`, returning an error.
    ///
    /// Destroying a module can return errors from previous asynchronous work. This function
    /// destroys the given module and returns the error and the un-destroyed module on failure.
    fn drop(module: Self::ModuleT) -> DropResult<Self::ModuleT>;
}

// /// Handle to a symbol defined within a Device module.
// #[derive(Debug)]
// pub struct Symbol<'a, T: DeviceCopy, M> {
//     pub ptr: DevicePointer<T>,
//     pub module: PhantomData<&'a Module<M>>,
// }
