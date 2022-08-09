use cuda_backend as cuda;
use uhal::{DriverLibraryTrait};
use uhal::memory::DeviceBufferTrait;
use uhal::stream::{StreamTrait, StreamFlags};
use uhal::module::{ModuleTrait};
use cuda::module::CuModule;
use cuda::memory::CuDeviceBuffer;
use cuda::stream::CuStream;
use std::error::Error;
use cuda::memory::CopyDestination;
use uhal::launch;

fn main() -> Result<(), Box<dyn Error>> {
    // Set up the context, load the module, and create a stream to run kernels in.
    let _ctx = cuda::CuApi::quick_init()?;

    let ptx = "../../add.ptx".to_string();
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