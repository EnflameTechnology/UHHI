use tops_backend as tops;
use uhal::error::DeviceError;
use uhal::{DriverLibraryTrait};
use uhal::memory::{DeviceBufferTrait, DevicePointerTrait};
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
use std::os::raw::{c_void, c_int};
use std::ffi::CString;
use driv::{topsError_t};
use std::path::Path;
use uhal::error::{DeviceResult};
use cust_core::DeviceCopy;
use std::collections::HashMap;

#[cfg(unix)]
fn to_bytes<P: AsRef<Path>>(path: P) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    path.as_ref().as_os_str().as_bytes().to_vec()
}

fn legacy() -> DeviceResult<()>{

    unsafe {
        let mut device : driv::topsDevice_t = 0;
        driv::topsDeviceGet(&mut device, 0);
        driv::topsSetDevice(device as c_int);
    }
    let ptx = "./resources/copy_d2c.o".to_string();

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

    let name = "copy_d2c";
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
    
    const N:usize = 10000;
    let mut val = vec![0.5f32; N];
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


    let mut out_host = vec![0.0f32; N];
    let mut out_host2 = vec![0.0f32; N];

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

fn load_module<'a>(name : &str) -> DeviceResult<TopsModule>{
    let ptx = format!("./resources/{}.o",name).to_string();
    TopsModule::from_file(&ptx)
}
struct Layer<'a, T: DeviceCopy> {
    op : &'a str,
    weight : Option<TopsDeviceBuffer<T>>,
    input_size : (usize, usize),
    output_size : (usize, usize),
    out_ref : Option<&'a TopsDeviceBuffer<T>>
}

fn network_test() -> DeviceResult<()> {
    let _ctx = tops::TopsApi::quick_init()?;
    let stream = TopsStream::new(StreamFlags::NON_BLOCKING, None)?;

    const N : usize = 16;
    const K : usize = 3;

    //Neural network layers: matmul(tanh act) -> matmul(relu act) -> matmul(tanh act) -> convolution(3x3 kernel, tanh act) -> matmul(tanh act) -> matmul(leaky act)
    let layers = vec![
        Layer::<f32> {op : "matmul", weight: Some(TopsDeviceBuffer::from_slice(&[0.01f32; N * N])?), input_size : (N, N), output_size : (N, N), out_ref : None}, //weight is N x N matric for next layer
        //Layer::<f32> {op : "tanh", weight : None, input_size : (N, N), output_size : (N, N), out_ref : None}, //out N x N

        Layer::<f32> {op : "matmul", weight: Some(TopsDeviceBuffer::from_slice(&[0.02f32; N * N])?), input_size : (N, N), output_size : (N, N), out_ref : None}, //weight is N x N matric for next layer
        //Layer::<f32> {op : "relu", weight : None, input_size : (N, N), output_size : (N, N), out_ref : None}, //out N x N

        Layer::<f32> {op : "matmul", weight: Some(TopsDeviceBuffer::from_slice(&[0.02f32; N * N])?), input_size : (N, N), output_size : (N, N), out_ref : None}, //weight is N x N matric for next layer

        //Layer::<f32> {op : "matmul", weight: Some(CuDeviceBuffer::from_slice(&[0.5f32; K * K])?), input_size : (N, N), output_size : (N, N), out_ref : None}, //weight is convolution kernel for next layer
        //Layer::<f32> {op : "tanh", weight : None, input_size : (N, N), output_size : (N, N), out_ref : None}, //out N x N

        //Layer::<f32> {op : "convolution", weight: Some(CuDeviceBuffer::from_slice(&[0.2f32; (N - K + 1) * (N - K + 1)])?), input_size : (N, N), output_size : (N - K + 1, N - K + 1), out_ref : None}, //weight is (N - K + 1) * (N - K + 1) matric for next layer
       // Layer::<f32> {op : "tanh", weight : None, input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None},  //out (N - K + 1) x (N - K + 1)
        
        //Layer::<f32> {op : "matmul", weight: Some(CuDeviceBuffer::from_slice(&[0.2f32; (N - K + 1) * (N - K + 1)])?), input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None}, //weight is (N - K + 1) * (N - K + 1) matric for next layer
        //Layer::<f32> {op : "tanh", weight : None, input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None}, //output shape (N - K + 1) * (N - K + 1)

        //Layer::<f32> {op : "matmul", weight: None, input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None}, // no weight in the last layer
       // Layer::<f32> {op : "leaky", weight : None, input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None}, //output shape (N - K + 1) * (N - K + 1)
    ];

    let mut matA = TopsDeviceBuffer::from_slice(&[0.5f32; N * N])?;
    let mut matB = TopsDeviceBuffer::from_slice(&[0.1f32; N * N])?;
    let mut matOut = TopsDeviceBuffer::from_slice(&[0.0f32; N * N])?;
    let mut matConvOut = TopsDeviceBuffer::from_slice(&[0.0f32; (N - K + 1) * (N - K + 1)])?;

    let map_act = HashMap::from([("relu", 0), ("elu", 1), ("leaky", 2), ("tanh", 3)]);

    let mut out_ref : Option<&TopsDeviceBuffer<f32>> = None;
    for layer in layers {
        if ["relu", "elu", "leaky", "tanh"].contains(&layer.op) {
            let function_name = "activation_array_kernel";
            match load_module("activation") {
                Ok(module) => {
                    let kernel = module.get_function(&function_name)?;
                    unsafe {
                        let result = launch!(kernel<<<(1, 1, 1), (layer.input_size.0 as u32, layer.input_size.1 as u32, 1), 0, stream>>>(
                            matA.as_device_ptr(),
                            layer.output_size.0,
                            map_act[layer.op]
                        ));
                        result?;
                    }
                    out_ref = Some(&matA);
                }
                _ => { println!("Failed to load kernel!"); break;}
            }
        } else if layer.op == "matmul" {
            match load_module(layer.op) {
                Ok(module) => {
                    let kernel = module.get_function(&layer.op)?;
                    let mut inputShapeA = TopsDeviceBuffer::from_slice(&[layer.input_size.0 as i32, layer.input_size.1 as i32, 1i32, 1i32])?;
                    let mut inputShapeB = TopsDeviceBuffer::from_slice(&[layer.input_size.0 as i32, layer.input_size.1 as i32, 1i32, 1i32])?;

                    unsafe {
                        let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                            matA.as_device_ptr(),
                            matB.as_device_ptr(),
                            matOut.as_device_ptr(),
                            inputShapeA.as_device_ptr(),
                            inputShapeB.as_device_ptr()
                        ));
                        result?;
                    }
                    std::mem::swap(&mut matA, &mut matOut);
                    match layer.weight {
                        Some(w) => { matB = w;}
                        _ => { 
                            // if idx < len - 1 { println!("Failed to get weight!"); break; }
                        }
                    }
                    out_ref = Some(&matA);
                }
                _ => { println!("Failed to load kernel!"); break; }
            }
        } else {
            match load_module(layer.op) {
                Ok(module) => {
                    let kernel = module.get_function(&layer.op)?;
                    unsafe {
                        let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                            matA.as_device_ptr(),
                            matConvOut.as_device_ptr(),
                            matB.as_device_ptr(),
                            layer.input_size.0 as u32, layer.input_size.1 as u32,
                            layer.output_size.0 as u32, layer.output_size.1 as u32,
                            K,
                            K
                        ));
                        result?;
                    }

                    std::mem::swap(&mut matA, &mut matConvOut);
                    match layer.weight {
                        Some(w) => { matB = w;}
                        _ => { 
                            // if idx < len - 1 { println!("Failed to get weight!"); break; }
                        }
                    }
                    out_ref = Some(&matA);

                }
                _ => { println!("Failed to load kernel!"); break; }
            }
        }
    }
    // Wait asynchronous kernels to finish.
    stream.synchronize()?;

    match out_ref {
        Some(out) => {
            let mut conv_out_host = vec![0.0f32; out.len()];
            out.copy_to(&mut conv_out_host[0..out.len()])?;
            println!("\n\nResults after convolution******************");
            for x in 0..N {
                for y in 0..N {
                    print!("{:.5} ", conv_out_host[x * N + y]);
                }
                println!("{}", "");
            }
        }
        _ => { println!("Unable to obtain compute result!")}
    }

    println!("\nLaunched compute kernel successfully.");

    Ok(())
}

fn main() -> DeviceResult<()> {
    println!("******************\ninfo: start legacy test!\n");
    legacy();

    println!("******************\ninfo: start uhal tops_backend network test!\n");

    network_test();

    println!("\n\n\n******************\ninfo: start uhal tops_backend test!\n");
    // Set up the context, load the module, and create a stream to run kernels in.
    let _ctx = tops::TopsApi::quick_init()?;
    let stream = TopsStream::new(StreamFlags::NON_BLOCKING, None)?;

    // matmul of 2 x 5 and 5 x 3  -> 2 x 3
    const M : usize = 2;
    const N : usize = 3;

    const Nbytes : usize = M * N * 4;
    let mut matA = TopsDeviceBuffer::from_slice(&[1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 1.0f32, 2.0f32, 3.0f32, 4.0f32])?;
    let mut matB = TopsDeviceBuffer::from_slice(&[1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32])?;
    let mut matOut = TopsDeviceBuffer::from_slice(&[0.0f32; M * N])?;

    let mut out_ref : Option<&TopsDeviceBuffer<f32>> = None;

    let mut inputShapeA = TopsDeviceBuffer::from_slice(&[2i32,5i32,1i32,1i32])?;
    let mut inputShapeB = TopsDeviceBuffer::from_slice(&[5i32,3i32,1i32,1i32])?;

    let mut outputShape = TopsDeviceBuffer::from_slice(&[2i32,3i32,1i32,1i32])?;
    let mut layout = TopsDeviceBuffer::from_slice(&[1i32,0i32,2i32,3i32])?;

    println!("info: launching kernel!\n");

    let fnname = "matmul";
    let module = load_module(fnname)?;
    let kernel = module.get_function(&fnname)?;
    unsafe {
        let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
            matA.as_device_ptr(),
            matB.as_device_ptr(),
            matOut.as_device_ptr(),
            inputShapeA.as_device_ptr(),
            inputShapeB.as_device_ptr(),
        ));
        result?;
    }

    println!("info: stream synchronization...\n");
    // Kernel launches are asynchronous, so we wait for the kernels to finish executing.
    stream.synchronize()?;  

    let mut host_ptr = std::ptr::null_mut();
    let mut out_host = vec![0.0f32; M * N];
    unsafe {
        driv::topsHostAlloc(&mut host_ptr as *mut *mut c_void, Nbytes, 0);
        driv::topsMemcpy(host_ptr, matOut.as_device_ptr().as_raw(), Nbytes, driv::topsMemcpyKind::topsMemcpyDeviceToHost);
        std::ptr::copy(host_ptr, out_host.as_mut_ptr() as *mut c_void, Nbytes);
        println!("\n\nResults of forward pass******************");
        for x in 0..M {
            for y in 0..N {
                print!("{:.5} ", out_host[x * N + y]);
            }
            println!("{}", "");
        }
    }

    println!("Launch kernel success for uhal tops!");

    println!("\n\nPASSED!\n\n");
    Ok(())
}