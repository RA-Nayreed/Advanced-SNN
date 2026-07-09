use anyhow::Result;

#[cfg(feature = "cuda")]
pub struct CudaRuntime {
    _context: cust::context::Context,
    pub device_name: String,
}

#[cfg(feature = "cuda")]
pub fn initialize_device() -> Result<CudaRuntime> {
    use cust::prelude::*;

    cust::init(CudaFlags::empty()).map_err(|error| {
        crate::gpu::cuda::unavailable(format!("failed to initialize CUDA driver: {error}"))
    })?;
    let device = Device::get_device(0).map_err(|error| {
        crate::gpu::cuda::unavailable(format!("failed to select CUDA device 0: {error}"))
    })?;
    let device_name = device.name().map_err(|error| {
        crate::gpu::cuda::unavailable(format!("failed to read CUDA device name: {error}"))
    })?;
    let context =
        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
            .map_err(|error| {
                crate::gpu::cuda::unavailable(format!("failed to create CUDA context: {error}"))
            })?;

    Ok(CudaRuntime {
        _context: context,
        device_name,
    })
}

#[cfg(not(feature = "cuda"))]
pub struct CudaRuntime {
    pub device_name: String,
}

#[cfg(not(feature = "cuda"))]
pub fn initialize_device() -> Result<CudaRuntime> {
    Err(crate::gpu::cuda::unavailable(
        "this binary was built without the cuda feature",
    ))
}
