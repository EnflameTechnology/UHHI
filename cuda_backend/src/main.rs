use cuda::memory::CopyDestination;
use cuda::memory::CuDeviceBuffer;
use cuda::module::CuModule;
use cuda::stream::CuStream;
use cuda_backend as cuda;
use uhal::error::DeviceResult;
use std::error::Error;
use uhal::launch;
use uhal::memory::DeviceBufferTrait;
use uhal::module::ModuleTrait;
use uhal::stream::{StreamFlags, StreamTrait};
use uhal::DriverLibraryTrait;

fn matmul_test() -> DeviceResult<()> {
    let _ctx = cuda::CuApi::quick_init()?;

    let ptx = "./resources/matmul.ptx".to_string();
    let module = CuModule::from_file(&ptx)?;
    let stream = CuStream::new(StreamFlags::NON_BLOCKING, None)?;

    const N : usize = 16;

    let mut matA = CuDeviceBuffer::from_slice(&[0.5f32; N * N])?;
    let mut matB = CuDeviceBuffer::from_slice(&[0.1f32; N * N])?;
    let mut matOut = CuDeviceBuffer::from_slice(&[0.0f32; N * N])?;

    // This kernel perform matric multiplication of two tensors `matA` and `matB` and writes the result into `matOut`.
    unsafe {
        // Launch the kernel using the `function` form:
        let function_name = "matmul".to_string();
        let matmul = module.get_function(&function_name)?;
        let result = launch!(matmul<<<(1, 1, 1), (N as u32, N as u32, 1), 0, stream>>>(
            matA.as_device_ptr(),
            matB.as_device_ptr(),
            matOut.as_device_ptr(),
            N
        ));
        result?;
    }

    // Kernel launches are asynchronous, so we wait for the kernels to finish executing.
    stream.synchronize()?;

    // Copy the results back to host memory
    let mut out_host = [0.0f32; N * N];
    matOut.copy_to(&mut out_host[0..N * N])?;

    println!("Results******************");
    for x in 0..N {
        for y in 0..N {
            print!("{:.2} ", out_host[x * N + y]);
        }
        println!("{}", "");
    }

    println!("Launched compute kernel successfully.");

    Ok(())
}

fn main() -> DeviceResult<()> {
    matmul_test();
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
