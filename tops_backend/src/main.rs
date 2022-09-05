use tops_backend as tops;
use uhal::error::DeviceError;
use uhal::{DriverLibraryTrait};
use uhal::memory::DeviceBufferTrait;
use uhal::stream::{StreamTrait, StreamFlags};
use uhal::module::{ModuleTrait};
use tops::module::TopsModule;
use tops::memory::TopsDeviceBuffer;
use tops::stream::TopsStream;
// use std::error::Error;
use tops::memory::CopyDestination;
use uhal::launch;
use tops_raw as driv;
use std::ptr;
use std::os::raw::c_void;
use std::ffi::CString;
use driv::{topsError_t};
use std::path::Path;
use uhal::error::{DeviceResult};


#[cfg(unix)]
fn to_bytes<P: AsRef<Path>>(path: P) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    path.as_ref().as_os_str().as_bytes().to_vec()
}

fn legacy() -> DeviceResult<()>{
    let ptx = "./resources/copy_d2d.o".to_string();

    let mut bytes = to_bytes(ptx);
    if !bytes.contains(&0) {
        bytes.push(0);
    }
    let mut module : driv::topsModule_t = ptr::null_mut();
    unsafe {
        driv::topsModuleLoad(
            &mut module as *mut driv::topsModule_t,
            bytes.as_ptr() as *const _,
        );
    }

    let name = "copy_d2d";
    let cstr = CString::new(name).expect("Argument to get_function had a nul");
    let mut func: driv::topsFunction_t = ptr::null_mut();

    unsafe {
        driv::topsModuleGetFunction(
            &mut func as *mut driv::topsFunction_t,
            module,
            cstr.as_ptr(),
        );
    }

    let mut stream : driv::topsStream_t = ptr::null_mut();

    unsafe {
        driv::topsStreamCreateWithPriority(
            &mut stream,
            0,
            0
        );
    }


    struct Params {
        a_ : driv::topsDeviceptr_t,
        b_ : driv::topsDeviceptr_t,
        N: usize
    }
    
    const N:usize = 100000;
    let mut val = [0.5f32; N];
    const Nbytes:usize = N * 4;

    let mut host_ptr = std::ptr::null_mut();
    let mut device_ptr = ptr::null_mut();
    let mut device_ptr_dst = ptr::null_mut();
    let mut device_ptr_dst2 = ptr::null_mut();

    unsafe {
        println!("info: copy Host2Device\n");
        driv::topsHostAlloc(&mut host_ptr as *mut *mut c_void, Nbytes, 0);
        std::ptr::copy(val.as_ptr() as *mut c_void, host_ptr, Nbytes);
        
        driv::topsMalloc(&mut device_ptr as *mut *mut c_void, Nbytes);
        driv::topsMalloc(&mut device_ptr_dst as *mut *mut c_void, Nbytes);
        driv::topsMalloc(&mut device_ptr_dst2 as *mut *mut c_void, Nbytes);

        driv::topsMemcpyHtoD(device_ptr, host_ptr as *mut c_void, Nbytes);
    }

    let mut args = Params {a_ : device_ptr, b_: device_ptr_dst, N :  Nbytes};
    let mut size = std::mem::size_of::<Params>();
    let mut config = vec![0x1 as *const c_void, &args as *const _ as *mut c_void, 0x2 as *const c_void, &mut size as *const _ as *mut c_void, 0x3 as *const c_void];

    let nul = ptr::null_mut();
    let mut host_ptr_out = ptr::null_mut();

    unsafe {
        println!("info: Launch kernel\n");
        driv::topsModuleLaunchKernel(
            func, 1, 1, 1,
            1, 1, 1,
            0,
            stream,
            nul as *mut *mut c_void,
            config.as_mut_ptr() as *mut *mut c_void            
        );

        println!("info: copy Device2Host\n");
        driv::topsHostAlloc(&mut host_ptr_out as *mut *mut c_void, Nbytes, 0);
        driv::topsMemcpy(host_ptr_out, device_ptr_dst, Nbytes, driv::topsMemcpyKind::topsMemcpyDeviceToHost);
    }

    let mut args = Params {a_ : device_ptr, b_: device_ptr_dst2, N :  Nbytes};
    let mut size = std::mem::size_of::<Params>();
    let mut config = vec![0x1 as *const c_void, &args as *const _ as *mut c_void, 0x2 as *const c_void, &mut size as *const _ as *mut c_void, 0x3 as *const c_void];

    let mut host_ptr_out2 = ptr::null_mut();

    unsafe {
        println!("info: Launch kernel again!\n");
        driv::topsModuleLaunchKernel(
            func, 1, 1, 1,
            1, 1, 1,
            0,
            stream,
            nul as *mut *mut c_void,
            config.as_mut_ptr() as *mut *mut c_void            
        );

        println!("info: copy Device2Host again!\n");
        driv::topsHostAlloc(&mut host_ptr_out2 as *mut *mut c_void, Nbytes, 0);
        driv::topsMemcpy(host_ptr_out2, device_ptr_dst2, Nbytes, driv::topsMemcpyKind::topsMemcpyDeviceToHost);
    }

    unsafe {
        println!("info: stream synchronization...\n");
        driv::topsStreamSynchronize(stream);
    }


    let mut out_host = [0.0f32; N];
    let mut out_host2 = [0.0f32; N];

    unsafe {
        std::ptr::copy(host_ptr_out, out_host.as_mut_ptr() as *mut c_void, Nbytes);
        std::ptr::copy(host_ptr_out2, out_host2.as_mut_ptr()  as *mut c_void, Nbytes);

    }

    unsafe {
        driv::topsFree(host_ptr);
        driv::topsFree(host_ptr_out);
        driv::topsFree(host_ptr_out2);

        driv::topsFree(device_ptr);
        driv::topsFree(device_ptr_dst);
        driv::topsFree(device_ptr_dst2);
    }

    println!("info: cheking results...\n");
    for x in out_host.iter() {
        assert_eq!(0.5 as u32, *x as u32);
    }
    for x in out_host2.iter() {
        assert_eq!(0.5 as u32, *x as u32);
    }
    println!("Launch kernel success for legacy mode!");
    Ok(())
}

fn main() -> DeviceResult<()> {
    println!("******************\ninfo: start legacy test!\n");
    let ret = legacy();

    println!("\n\n\n******************\ninfo: start uhal tops_backend test!\n");

    // Set up the context, load the module, and create a stream to run kernels in.
    let _ctx = tops::TopsApi::quick_init()?;

    let ptx = "./resources/copy_d2d.o".to_string();
    let module = TopsModule::from_file(&ptx)?;
    let stream = TopsStream::new(StreamFlags::NON_BLOCKING, None)?;

    const N:usize = 100000;
    let Nbytes : usize = N * 4;

    println!("info: implicit Host2Device memory copy\n");
    let mut src = TopsDeviceBuffer::from_slice(&[2.0f32; N])?;
    let mut dst = TopsDeviceBuffer::from_slice(&[0.0f32; N])?;
    let mut dst2 = TopsDeviceBuffer::from_slice(&[0.0f32; N])?;

    println!("info: launching kernel!\n");

    // This kernel copies elements from `src` to `dst`.
    unsafe {
        // Launch the kernel with one block of one thread, no dynamic shared memory on `stream`.
        let result = launch!(module.copy_d2d<<<1, 1, 0, stream>>>(
            src.as_device_ptr(),
            dst.as_device_ptr(),
            Nbytes
        ));
        result?;

        // Launch the kernel again using the `function` form:
        let function_name = "copy_d2d".to_string();
        let copy_d2d = module.get_function(&function_name)?;
        // Launch with 1x1x1 blocks of 1x1x1 threads, to show that you can use tuples to
        // configure grid and block size.
        let result = launch!(copy_d2d<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
            src.as_device_ptr(),
            dst2.as_device_ptr(),
            Nbytes
        ));
        result?;
    }

    println!("info: stream synchronization...\n");
    // Kernel launches are asynchronous, so we wait for the kernels to finish executing.
    stream.synchronize()?;   

    println!("info: copy back to host memory!\n");
    // Copy the results back to host memory
    let mut out_host = [0.0f32; N*2];
    dst.copy_to(&mut out_host[0..N])?;
    dst2.copy_to(&mut out_host[N..2*N])?;

    println!("info: cheking results...\n");
    for x in out_host.iter() {
        assert_eq!(2.0 as u32, *x as u32);
    }

    println!("Launch kernel success for uhal tops!");

    println!("\n\nPASSED!\n\n");
    Ok(())
}