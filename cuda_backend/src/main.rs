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
        Layer::<f32> {op : "leaky", weight : None, input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None}, //output shape (N - K + 1) * (N - K + 1)
    ];

    let mut matA = CuDeviceBuffer::from_slice(&[0.5f32; N * N])?;
    let mut matB = CuDeviceBuffer::from_slice(&[0.1f32; N * N])?;
    let mut matOut = CuDeviceBuffer::from_slice(&[0.0f32; N * N])?;
    let mut matConvOut = CuDeviceBuffer::from_slice(&[0.0f32; (N - K + 1) * (N - K + 1)])?;

    let map_act = HashMap::from([("relu", 0), ("elu", 1), ("leaky", 2), ("tanh", 3)]);

    let mut out_ref : Option<&CuDeviceBuffer<f32>> = None;
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
                    unsafe {
                        let result = launch!(kernel<<<(1, 1, 1), (layer.input_size.0 as u32, layer.input_size.1 as u32, 1), 0, stream>>>(
                            matA.as_device_ptr(),
                            matB.as_device_ptr(),
                            matOut.as_device_ptr(),
                            layer.output_size.0
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
                        let result = launch!(kernel<<<(1, 1, 1), (layer.input_size.0 as u32, layer.input_size.1 as u32, 1), 0, stream>>>(
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
            for x in 0..(N - K + 1) {
                for y in 0..(N - K + 1) {
                    print!("{:.5} ", conv_out_host[x * (N - K + 1) + y]);
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
    network_test();
    // Set up the context, load the module, and create a stream to run kernels in.
    let _ctx = cuda::CuApi::quick_init()?;

    let ptx = "./resources/add.ptx".to_string();
    let module = CuModule::from_file(&ptx)?;
    let stream = CuStream::new(StreamFlags::NON_BLOCKING, None)?;

    // Create buffers for data
    let mut in_x = CuDeviceBuffer::from_slice(&[1.0f32; 10])?;
    let mut in_y = CuDeviceBuffer::from_slice(&[2.0f32; 10])?;
    let mut out_1 = CuDeviceBuffer::from_slice(&[0.0f32; 10])?;
    let mut out_2 = CuDeviceBuffer::from_slice(&[0.0f32; 10])?;

    // This kernel adds each element in `in_x` and `in_y` and writes the result into `out`.
    unsafe {
        // Launch the kernel with one block of one thread, no dynamic shared memory on `stream`.
        let result = launch!(module.sum<<<1, 1, 0, stream>>>(
            in_x.as_device_ptr(),
            in_y.as_device_ptr(),
            out_1.as_device_ptr(),
            out_1.len()
        ));
        result?;

        // Launch the kernel again using the `function` form:
        let function_name = "sum".to_string();
        let sum = module.get_function(&function_name)?;
        // Launch with 1x1x1 (1) blocks of 10x1x1 (10) threads, to show that you can use tuples to
        // configure grid and block size.
        let result = launch!(sum<<<(1, 1, 1), (10, 1, 1), 0, stream>>>(
            in_x.as_device_ptr(),
            in_y.as_device_ptr(),
            out_2.as_device_ptr(),
            out_2.len()
        ));
        result?;
    }

    // Kernel launches are asynchronous, so we wait for the kernels to finish executing.
    stream.synchronize()?;

    // Copy the results back to host memory
    let mut out_host = [0.0f32; 20];
    out_1.copy_to(&mut out_host[0..10])?;
    out_2.copy_to(&mut out_host[10..20])?;

    for x in out_host.iter() {
        assert_eq!(3.0 as u32, *x as u32);
    }

    println!("Launched kernel successfully.");
    Ok(())
}
