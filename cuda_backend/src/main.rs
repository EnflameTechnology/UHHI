use cuda::function::CuFunction;
use cuda::memory::CopyDestination;
use cuda::memory::CuDeviceBuffer;
use cuda::module::CuModule;
use cuda::stream::CuStream;
use cuda_backend as cuda;
use cust_core::DeviceCopy;
use uhal::error::DeviceResult;
use uhal::function::FunctionTrait;
use std::borrow::Borrow;
use std::marker::PhantomData;
use uhal::launch;
use uhal::memory::DeviceBufferTrait;
use uhal::module::ModuleTrait;
use uhal::stream::{StreamFlags, StreamTrait};
use uhal::DriverLibraryTrait;
use std::collections::HashMap;

struct Layer<'a, T: DeviceCopy> {
    op : &'a str,
    weight : Option<CuDeviceBuffer<T>>,
    input_size : (usize, usize),
    output_size : (usize, usize),
    out_ref : Option<&'a CuDeviceBuffer<T>>
}
pub fn get_block_grid(shape1:usize, shape0:usize) -> (usize, usize, usize) {
    let grid_a : usize = (shape1 + 16 - 1) / 16;
    let grid_b : usize = (shape0 + 16 - 1) / 16;
    return (16, grid_a, grid_b)
}
fn load_module<'a>(name : &str) -> DeviceResult<CuModule>{
    let ptx = format!("./resources/{}.ptx",name).to_string();
    CuModule::from_file(&ptx)
}

fn network_test() -> DeviceResult<()> {
    let _ctx = cuda::CuApi::quick_init()?;
    let stream = CuStream::new(StreamFlags::NON_BLOCKING, None)?;

    const N : usize = 16;
    const K : usize = 3;

    //Neural network layers: matmul(tanh act) -> matmul(relu act) -> matmul(tanh act) -> convolution(3x3 kernel, tanh act) -> matmul(tanh act) -> matmul(leaky act)
    let layers = vec![
        Layer::<f32> {op : "matmul", weight: Some(CuDeviceBuffer::from_slice(&[0.01f32; N * N])?), input_size : (N, N), output_size : (N, N), out_ref : None}, //weight is N x N matric for next layer
        Layer::<f32> {op : "tanh", weight : None, input_size : (N, N), output_size : (N, N), out_ref : None}, //out N x N

        Layer::<f32> {op : "matmul", weight: Some(CuDeviceBuffer::from_slice(&[0.02f32; N * N])?), input_size : (N, N), output_size : (N, N), out_ref : None}, //weight is N x N matric for next layer
        Layer::<f32> {op : "relu", weight : None, input_size : (N, N), output_size : (N, N), out_ref : None}, //out N x N

        Layer::<f32> {op : "matmul", weight: Some(CuDeviceBuffer::from_slice(&[0.5f32; K * K])?), input_size : (N, N), output_size : (N, N), out_ref : None}, //weight is convolution kernel for next layer
        Layer::<f32> {op : "tanh", weight : None, input_size : (N, N), output_size : (N, N), out_ref : None}, //out N x N

        Layer::<f32> {op : "convolution", weight: Some(CuDeviceBuffer::from_slice(&[0.2f32; (N - K + 1) * (N - K + 1)])?), input_size : (N, N), output_size : (N - K + 1, N - K + 1), out_ref : None}, //weight is (N - K + 1) * (N - K + 1) matric for next layer
        Layer::<f32> {op : "tanh", weight : None, input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None},  //out (N - K + 1) x (N - K + 1)
        
        Layer::<f32> {op : "matmul", weight: Some(CuDeviceBuffer::from_slice(&[0.2f32; (N - K + 1) * (N - K + 1)])?), input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None}, //weight is (N - K + 1) * (N - K + 1) matric for next layer
        Layer::<f32> {op : "tanh", weight : None, input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None}, //output shape (N - K + 1) * (N - K + 1)

        Layer::<f32> {op : "matmul", weight: None, input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None}, // no weight in the last layer
        Layer::<f32> {op : "gelu", weight : None, input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None}, //output shape (N - K + 1) * (N - K + 1)
    ];

    let mut matA = CuDeviceBuffer::from_slice(&[0.5f32; N * N])?;
    let mut matB = CuDeviceBuffer::from_slice(&[0.1f32; N * N])?;
    let mut matOut = CuDeviceBuffer::from_slice(&[0.0f32; N * N])?;
    let mut matConvOut = CuDeviceBuffer::from_slice(&[0.0f32; (N - K + 1) * (N - K + 1)])?;

    let map_act = HashMap::from([("relu", 0), ("gelu", 1), ("leaky", 2), ("tanh", 3)]);

    let mut out_ref : Option<&CuDeviceBuffer<f32>> = None;
    let mut out_size : Option<(usize, usize)> = None;
    for layer in layers {
        if ["relu", "gelu", "leaky", "tanh"].contains(&layer.op) {
            let function_name = "activation";
            match load_module(function_name) {
                Ok(module) => {
                    let kernel = module.get_function(&function_name)?;
                    let (block_size, grid_a, grid_b) = get_block_grid(layer.input_size.1, layer.input_size.0);
                    unsafe {
                        let result = launch!(kernel<<<(grid_a as u32, grid_b as u32), (block_size as u32, block_size as u32), 0, stream>>>(
                            matA.as_device_ptr(),
                            layer.input_size.0 as u32,
                            layer.input_size.1 as u32,
                            map_act[layer.op]
                        ));
                        result?;
                    }
                    out_ref = Some(&matA);
                    out_size = Some(layer.output_size);
                }
                _ => { println!("Failed to load kernel!"); break;}
            }
        } else if layer.op == "matmul" {
            match load_module(layer.op) {
                Ok(module) => {
                    let kernel = module.get_function(&layer.op)?;
                    let (block_size, grid_a, grid_b) = get_block_grid(layer.input_size.1, layer.input_size.0);
                    unsafe {
                        let result = launch!(kernel<<<(grid_a as u32, grid_b as u32), (block_size as u32, block_size as u32), 0, stream>>>(
                            matA.as_device_ptr(),
                            matB.as_device_ptr(),
                            matOut.as_device_ptr(),
                            layer.input_size.0 as u32,
                            layer.input_size.1 as u32,
                            layer.output_size.1 as u32
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
                    out_size = Some(layer.output_size);
                }
                _ => { println!("Failed to load kernel!"); break; }
            }
        } else if layer.op == "convolution" {
            match load_module(layer.op) {
                Ok(module) => {
                    let kernel = module.get_function(&layer.op)?;
                    unsafe {
                        let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                            matA.as_device_ptr(),
                            matB.as_device_ptr(),
                            matConvOut.as_device_ptr(),
                            layer.input_size.0 as i32, layer.input_size.1 as i32,
                            K as i32,
                            K as i32
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
                    out_size = Some(layer.output_size);

                }
                _ => { println!("Failed to load kernel!"); break; }
            }
        } else {
            println!("Operation {} not supported!", layer.op); 
            break;
        }
    }
    // Wait asynchronous kernels to finish.
    stream.synchronize()?;

    match out_ref {
        Some(out) => {
            let mut conv_out_host = vec![0.0f32; out.len()];
            out.copy_to(&mut conv_out_host[0..out.len()])?;
            match out_size {
                Some(sz) => {
                    let W = sz.0;
                    let H = sz.1;
                    println!("\n\nResults of forward pass******************");
                    for x in 0..H {
                        for y in 0..W {
                            print!("{:.5} ", conv_out_host[x * W + y]);
                        }
                        println!("{}", "");
                    }
                }
                _ => { println!("Unable to obtain compute result!") }
            }

        }
        _ => { println!("Unable to obtain compute result!")}
    }

    println!("\nLaunched compute kernel successfully.");

    Ok(())
}

fn test() -> DeviceResult<()> {
    println!("\n\n\n******************\ninfo: start uhal cuda_backend test!\n");
    // Set up the context, load the module, and create a stream to run kernels in.
    let _ctx = cuda::CuApi::quick_init()?;
    let stream = CuStream::new(StreamFlags::NON_BLOCKING, None)?;

    // matmul of 3 x 3 and 3 x 3  -> 3 x 3
    const M : usize = 3;
    const N : usize = 3;

    const Nbytes : usize = M * N * 4;
    let mut matA = CuDeviceBuffer::from_slice(&[1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 1.0f32, 2.0f32, 3.0f32])?;
    let mut matB = CuDeviceBuffer::from_slice(&[4.0f32, 5.0f32, 1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 1.0f32, 2.0f32])?;
    let mut matOut = CuDeviceBuffer::from_slice(&[0.0f32; M * N])?;

    println!("info: launching kernel!\n");

    let fnname = "matmul";
    let module = load_module(fnname)?;
    let kernel = module.get_function(&fnname)?;
    unsafe {
        let result = launch!(kernel<<<(1, 1, 1), (3, 3, 1), 0, stream>>>(
            matA.as_device_ptr(),
            matB.as_device_ptr(),
            matOut.as_device_ptr(),
            N
        ));
        result?;
    }

    println!("info: stream synchronization...\n");
    // Kernel launches are asynchronous, so we wait for the kernels to finish executing.
    stream.synchronize()?;  


    let mut out_host = vec![0.0f32; M * N];
    matOut.copy_to(&mut out_host[0..M * N])?;

    println!("\n\nResults of forward pass******************");
    for x in 0..M {
        for y in 0..N {
            print!("{:.5} ", out_host[x * N + y]);
        }
        println!("{}", "");
    }

    println!("Launch kernel success for uhal CUDA!");

    println!("\n\nPASSED!\n\n");
    Ok(())
}


fn main() -> DeviceResult<()> {
    println!("******************\ninfo: start uhal cuda_backend network test!\n");

    match network_test() {
        Ok(()) => {
            println!("\nLaunched network_test successfully.");
        }
        Err(e) => {
            println!("\nLaunch network_test failed.");
            return Err(e);
        }
    }
    println!("\n\nPASSED!\n\n");
    Ok(())
}