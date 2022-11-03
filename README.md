<div align="center">
<h1 align="center">Unified Heterogeneous Hardware Interface</h1>
<br />
<img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg" /><br>
<br>
Unified heterogeneous hardware interface for deep learning, that enables you to build robust and performant runtime system for heterogeneous deep learning workloads with minimal efforts
</div>

***

# Usage
## For cuda backend:

1) Make sure you have NVIDIA card, driver and cuda 11.3 installed

2) Run the following command to build & run cuda backend under the main folder
```
cargo run --bin cuda_backend
```
######Tested under CUDA 11.3 and NVIDIA 2080Ti.

## For tops backend:
1) Make sure you have Enflame T20 card, driver installed
2) Download TopsAPI library (libtops_api64.so) to "lib" directory under the main folder
3) Run the following command to build & run tops backend under the main folder
```
cargo run --bin tops_backend
```
######Tested under driver/sdk package "TopsRider_t2x_2.0.20220826_deb_internal" and Enflame T20 Rev 1

# Sample code

``` rust
//Example of UHAL for neural network forward pass (on NV GPU & Enflame GCU)
use cust_core::DeviceCopy;
use std::collections::HashMap;

//Import UHAL for common computing interfaces
use uhal::launch;
use uhal::error::{DeviceResult};
use uhal::{DriverLibraryTrait};
use uhal::module::{ModuleTrait};
use uhal::memory::{DeviceBufferTrait};
use uhal::stream::{StreamTrait, StreamFlags};

//Tops backend
#[cfg(feature = "tops_backend")]
use tops_backend as tops;
#[cfg(feature = "tops_backend")]
use tops::memory::TopsDeviceBuffer as DeviceBuffer;
#[cfg(feature = "tops_backend")]
use tops::memory::CopyDestination;
#[cfg(feature = "tops_backend")]
use tops::stream::TopsStream as Stream;
#[cfg(feature = "tops_backend")]
use tops::module::TopsModule as Module;
#[cfg(feature = "tops_backend")]
use tops::TopsApi as Api;

//Cuda backend
#[cfg(feature = "cuda_backend")]
use cuda_backend as cuda;
#[cfg(feature = "cuda_backend")]
use cuda::memory::CuDeviceBuffer as DeviceBuffer;
#[cfg(feature = "cuda_backend")]
use cuda::memory::CopyDestination;
#[cfg(feature = "cuda_backend")]
use cuda::stream::CuStream as Stream;
#[cfg(feature = "cuda_backend")]
use cuda::module::CuModule as Module;
#[cfg(feature = "cuda_backend")]
use cuda::CuApi as Api;

//Load kernel module
fn load_module<'a>(name : &str) -> DeviceResult<Module>{
    #[cfg(feature = "tops_backend")]
    let ptx = format!("{}/resources/{}.o", env!("CARGO_MANIFEST_DIR"), name).to_string();

    #[cfg(feature = "cuda_backend")]
    let ptx = format!("{}/resources/{}.ptx", env!("CARGO_MANIFEST_DIR"), name).to_string();

    Module::from_file(&ptx)
}

//Neural network layer definition
struct Layer<'a, T: DeviceCopy> {
    op : &'a str,
    weight : Option<DeviceBuffer<T>>,
    input_size : (usize, usize),
    output_size : (usize, usize),
    out_ref : Option<&'a DeviceBuffer<T>>
}

//A 6-layer neural network forward pass
//Unified interface (UHAL) for CUDA and Tops backend
#[allow(non_snake_case)]
fn network_test() -> DeviceResult<()> {
    let _ctx = Api::quick_init()?;

    //The entire workflow computed on this stream without copy back & forth between GPU/GCU memory and host memory.
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    const N : usize = 16;
    const K : usize = 3;

    //Neural network layers: matmul(tanh act) -> matmul(relu act) -> matmul(tanh act) -> convolution(3x3 kernel, tanh act) -> matmul(tanh act) -> matmul(leaky act)
    let layers = vec![
        Layer::<f32> {op : "matmul", weight: Some(DeviceBuffer::from_slice(&[0.01f32; N * N])?), input_size : (N, N), output_size : (N, N), out_ref : None}, //weight is N x N matric for next layer
        Layer::<f32> {op : "tanh", weight : None, input_size : (N, N), output_size : (N, N), out_ref : None}, //out N x N

        Layer::<f32> {op : "matmul", weight: Some(DeviceBuffer::from_slice(&[0.02f32; N * N])?), input_size : (N, N), output_size : (N, N), out_ref : None}, //weight is N x N matric for next layer
        Layer::<f32> {op : "relu", weight : None, input_size : (N, N), output_size : (N, N), out_ref : None}, //out N x N

        Layer::<f32> {op : "matmul", weight: Some(DeviceBuffer::from_slice(&[0.5f32; K * K])?), input_size : (N, N), output_size : (N, N), out_ref : None}, //weight is convolution kernel for next layer
        Layer::<f32> {op : "tanh", weight : None, input_size : (N, N), output_size : (N, N), out_ref : None}, //out N x N

        Layer::<f32> {op : "convolution", weight: Some(DeviceBuffer::from_slice(&[0.2f32; (N - K + 1) * (N - K + 1)])?), input_size : (N, N), output_size : (N - K + 1, N - K + 1), out_ref : None}, //weight is (N - K + 1) * (N - K + 1) matric for next layer
        Layer::<f32> {op : "tanh", weight : None, input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None},  //out (N - K + 1) x (N - K + 1)
        
        Layer::<f32> {op : "matmul", weight: Some(DeviceBuffer::from_slice(&[0.2f32; (N - K + 1) * (N - K + 1)])?), input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None}, //weight is (N - K + 1) * (N - K + 1) matric for next layer
        Layer::<f32> {op : "tanh", weight : None, input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None}, //output shape (N - K + 1) * (N - K + 1)

        Layer::<f32> {op : "matmul", weight: None, input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None}, // no weight in the last layer
        Layer::<f32> {op : "gelu", weight : None, input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None}, //output shape (N - K + 1) * (N - K + 1)
    ];

    //Buffers on device (GPU/GCU), initialized with values
    let mut matA = DeviceBuffer::from_slice(&[0.5f32; N * N])?;
    let mut matB = DeviceBuffer::from_slice(&[0.1f32; N * N])?;
    let mut matOut = DeviceBuffer::from_slice(&[0.0f32; N * N])?;
    let mut matConvOut = DeviceBuffer::from_slice(&[0.0f32; (N - K + 1) * (N - K + 1)])?;

    //For activation type mapping
    let map_act = HashMap::from([("relu", 0), ("gelu", 1), ("leaky", 2), ("tanh", 3)]);

    //Reference to output
    let mut out_ref : Option<&DeviceBuffer<f32>> = None;
    let mut out_size : Option<(usize, usize)> = None;

    //Forward computing
    for layer in layers {
        if ["relu", "gelu", "leaky", "tanh"].contains(&layer.op) {
            let function_name = "activation";
            match load_module(function_name) {
                Ok(module) => {
                    let kernel = module.get_function(&function_name)?;
                    unsafe {
                        //Slightly difference calling parameter for GCU and GPU.
                        #[cfg(feature = "tops_backend")]
                        let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                            matA.as_device_ptr(),
                            (layer.input_size.0 * layer.input_size.1) as i32,
                            map_act[layer.op] as i32
                        ));

                        #[cfg(feature = "cuda_backend")]
                        let result = launch!(kernel<<<(1, 1, 1), (layer.input_size.0 as u32, layer.input_size.1 as u32, 1), 0, stream>>>(
                            matA.as_device_ptr(),
                            layer.output_size.0,
                            map_act[layer.op]
                        ));

                        result?;
                    }
                    out_ref = Some(&matA);
                    out_size = Some(layer.output_size);
                }
                _ => { panic!("Failed to load kernel!"); }
            }
        } else if layer.op == "matmul" {
            match load_module(layer.op) {
                Ok(module) => {
                    let kernel = module.get_function(&layer.op)?;

                    #[cfg(feature = "tops_backend")]
                    let inputShapeA = DeviceBuffer::from_slice(&[layer.input_size.0 as i32, layer.input_size.1 as i32, 1i32, 1i32])?;
                    #[cfg(feature = "tops_backend")]
                    let inputShapeB = DeviceBuffer::from_slice(&[layer.input_size.0 as i32, layer.input_size.1 as i32, 1i32, 1i32])?;

                    unsafe {
                        #[cfg(feature = "tops_backend")]
                        let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                            matA.as_device_ptr(),
                            matB.as_device_ptr(),
                            matOut.as_device_ptr(),
                            inputShapeA.as_device_ptr(),
                            inputShapeB.as_device_ptr()
                        ));

                        #[cfg(feature = "cuda_backend")]
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
                    out_size = Some(layer.output_size);
                }
                _ => { panic!("\nFailed to load kernel (matmul)!"); }
            }
        } else if layer.op == "convolution" {
            match load_module(layer.op) {
                Ok(module) => {
                    let kernel = module.get_function(&layer.op)?;

                    #[cfg(feature = "tops_backend")]
                    let inputShapeA = DeviceBuffer::from_slice(&[layer.input_size.0 as i32, layer.input_size.1 as i32, 1i32, 1i32])?;
                    #[cfg(feature = "tops_backend")]
                    let inputShapeB = DeviceBuffer::from_slice(&[K as i32, K as i32, 1i32, 1i32])?;
                    #[cfg(feature = "tops_backend")]
                    let channelInfo = DeviceBuffer::from_slice(&[1i32, 1i32, 1i32, 1i32])?;

                    unsafe {
                        
                        #[cfg(feature = "tops_backend")]
                        let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                            matA.as_device_ptr(),
                            matB.as_device_ptr(),
                            matConvOut.as_device_ptr(),
                            inputShapeA.as_device_ptr(),
                            inputShapeB.as_device_ptr(),
                            channelInfo.as_device_ptr()
                        ));

                        #[cfg(feature = "cuda_backend")]
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
                _ => { panic!("\nFailed to load kernel (convolution)!"); }
            }
        } else {
            panic!("Operation {} not supported!", layer.op); 
        }
    }
    // Wait asynchronous kernels to finish.
    stream.synchronize()?;

    //Obtain results and print
    match out_ref {
        Some(out) => {
            let mut out_host = vec![0.0f32; out.len()];
            out.copy_to(&mut out_host[0..out.len()])?;
            match out_size {
                Some(sz) => {
                    let W = sz.0;
                    let H = sz.1;
                    println!("\n\nResults of forward pass******************");
                    for x in 0..H {
                        for y in 0..W {
                            print!("{:.5} ", out_host[x * W + y]);
                        }
                        println!("{}", "");
                    }
                }
                _ => { panic!("Unable to obtain compute result!") }
            }

        }
        _ => { panic!("Unable to obtain compute result!")}
    }

    println!("\nLaunched compute kernel successfully.");

    Ok(())
}

fn main() -> DeviceResult<()> {
    println!("******************\ninfo: start uhal network test!\n");
    
    match network_test() {
        Ok(()) => {
            println!("\nLaunched network_test successfully.");
        }
        Err(e) => {
            println!("\nLaunche network_test failed.");
            return Err(e);
        }
    }
    println!("\n\nPASSED!\n\n");
    Ok(())
}
```

# Results
The output of the forward pass for a 6-layer neural net should be:

**(Same on Nvidia GPU and Enflame GCU)**
```
Results of forward pass******************
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
```

# License
This project is licensed under the MIT license
# Show your support
Leave a ⭐ if you like this project
