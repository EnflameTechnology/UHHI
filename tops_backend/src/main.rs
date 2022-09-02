use tops_backend as tops;
use uhal::{DriverLibraryTrait};
use uhal::memory::DeviceBufferTrait;
use uhal::stream::{StreamTrait, StreamFlags};
use uhal::module::{ModuleTrait};
use tops::module::TopsModule;
use tops::memory::TopsDeviceBuffer;
use tops::stream::TopsStream;
use std::error::Error;
use tops::memory::CopyDestination;
use uhal::launch;

fn main() -> Result<(), Box<dyn Error>> {
    // Set up the context, load the module, and create a stream to run kernels in.
    let _ctx = tops::TopsApi::quick_init()?;

    let ptx = "./resources/copy_d2d.o".to_string();
    let module = TopsModule::from_file(&ptx)?;
    let stream = TopsStream::new(StreamFlags::NON_BLOCKING, None)?;

    // Create buffers for data
    let mut src = TopsDeviceBuffer::from_slice(&[2.0f32; 10])?;
    let mut dst1 = TopsDeviceBuffer::from_slice(&[0.0f32; 10])?;
    let mut dst2 = TopsDeviceBuffer::from_slice(&[0.0f32; 10])?;

    // This kernel copies elements from `src` to `dst`.
    unsafe {
        // Launch the kernel with one block of one thread, no dynamic shared memory on `stream`.
        let result = launch!(module.copy_d2d<<<1, 1, 0, stream>>>(
            src.as_device_ptr(),
            dst1.as_device_ptr(),
            src.len()
        ));
        result?;

        // Launch the kernel again using the `function` form:
        // let function_name = "copy_d2d".to_string();
        // let sum = module.get_function(&function_name)?;
        // // Launch with 1x1x1 (1) blocks of 10x1x1 (10) threads, to show that you can use tuples to
        // // configure grid and block size.
        // let result = launch!(sum<<<(1, 1, 1), (10, 1, 1), 0, stream>>>(
        //     src.as_device_ptr(),
        //     dst2.as_device_ptr(),
        //     src.len()
        // ));
        // result?;
    }

    // Kernel launches are asynchronous, so we wait for the kernels to finish executing.
    stream.synchronize()?;

    // Copy the results back to host memory
    let mut out_host = [0.0f32; 10];
    dst1.copy_to(&mut out_host[0..10])?;
    // dst2.copy_to(&mut out_host[10..20])?;

    for x in out_host.iter() {
        assert_eq!(2.0 as u32, *x as u32);
    }

    println!("Launched kernel successfully.");
    Ok(())
}