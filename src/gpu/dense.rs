use anyhow::{bail, Result};

use crate::config::DenseSimulationConfig;
use crate::gpu::buffers::{DenseInputCase, DenseStepOutput};
use crate::metrics::DenseGpuMetrics;
#[cfg(feature = "cuda")]
use crate::neuron::state::NeuronState;
#[cfg(feature = "cuda")]
use crate::types::external_input_applies;

#[cfg(not(feature = "cuda"))]
pub fn run_dense(_config: &DenseSimulationConfig) -> Result<DenseGpuMetrics> {
    Err(crate::gpu::cuda::unavailable(
        "this binary was built without the cuda feature",
    ))
}

#[cfg(feature = "cuda")]
pub fn run_dense(config: &DenseSimulationConfig) -> Result<DenseGpuMetrics> {
    use std::ffi::CString;
    use std::time::Instant;

    use cust::launch;
    use cust::prelude::*;

    config.validate()?;
    let runtime = crate::gpu::device::initialize_device()?;
    let ptx = std::fs::read_to_string(crate::gpu::cuda::ptx_path("lif_dense.ptx")?)?;
    let module = Module::from_ptx(CString::new(ptx)?, &[])?;
    let function = module.get_function("lif_dense_update")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let mut host = NeuronState::new(config.neurons);
    let mut cpu_reference = NeuronState::new(config.neurons);
    let mut spike_flags = vec![0_u8; config.neurons];
    let mut device_voltage = DeviceBuffer::from_slice(&host.voltage)?;
    let mut device_input = DeviceBuffer::from_slice(&host.input_current)?;
    let mut device_refractory = DeviceBuffer::from_slice(&host.refractory_left)?;
    let mut device_spikes = DeviceBuffer::from_slice(&spike_flags)?;

    let block_size = 256_u32;
    let grid_size = (config.neurons as u32 + block_size - 1) / block_size;
    let mut total_spikes = 0_u64;
    let mut cpu_gpu_match = true;
    let mut kernel_elapsed_seconds = 0.0_f64;

    for step in 0..config.steps {
        for neuron_id in 0..config.neurons {
            host.input_current[neuron_id] = if external_input_applies(
                config.seed,
                step,
                neuron_id,
                config.external_prob,
            ) {
                config.external_current
            } else {
                0.0
            };
            cpu_reference.input_current[neuron_id] = host.input_current[neuron_id];
        }

        device_input.copy_from(&host.input_current)?;
        let started = Instant::now();
        unsafe {
            // The launch uses buffers sized to at least `neurons`, and the kernel bounds-checks ids.
            launch!(function<<<grid_size, block_size, 0, stream>>>(
                device_voltage.as_device_ptr(),
                device_input.as_device_ptr(),
                device_refractory.as_device_ptr(),
                device_spikes.as_device_ptr(),
                config.neurons as u32,
                config.lif.decay,
                config.lif.threshold,
                config.lif.reset,
                config.lif.refractory
            ))?;
        }
        stream.synchronize()?;
        kernel_elapsed_seconds += started.elapsed().as_secs_f64();

        device_voltage.copy_to(&mut host.voltage)?;
        device_refractory.copy_to(&mut host.refractory_left)?;
        device_spikes.copy_to(&mut spike_flags)?;
        total_spikes += spike_flags.iter().map(|&flag| flag as u64).sum::<u64>();

        for neuron_id in 0..config.neurons {
            let spiked = crate::neuron::lif::update_lif_neuron(
                &mut cpu_reference.voltage[neuron_id],
                cpu_reference.input_current[neuron_id],
                &mut cpu_reference.refractory_left[neuron_id],
                config.lif,
            );
            if spike_flags[neuron_id] != u8::from(spiked) {
                cpu_gpu_match = false;
            }
        }
        let voltage_matches = host
            .voltage
            .iter()
            .zip(&cpu_reference.voltage)
            .all(|(gpu_value, cpu_value)| (gpu_value - cpu_value).abs() <= 1.0e-6);
        if !voltage_matches || host.refractory_left != cpu_reference.refractory_left {
            cpu_gpu_match = false;
        }
    }

    Ok(DenseGpuMetrics {
        selected_cuda_device: runtime.device_name,
        neurons: config.neurons,
        steps: config.steps,
        total_spikes,
        kernel_elapsed_seconds: Some(kernel_elapsed_seconds),
        cpu_gpu_match,
    })
}

#[cfg(not(feature = "cuda"))]
pub fn run_dense_single_step(_case: &DenseInputCase) -> Result<DenseStepOutput> {
    Err(crate::gpu::cuda::unavailable(
        "this binary was built without the cuda feature",
    ))
}

#[cfg(feature = "cuda")]
pub fn run_dense_single_step(case: &DenseInputCase) -> Result<DenseStepOutput> {
    use std::ffi::CString;

    use cust::launch;
    use cust::prelude::*;

    validate_dense_case(case)?;
    let _runtime = crate::gpu::device::initialize_device()?;
    let ptx = std::fs::read_to_string(crate::gpu::cuda::ptx_path("lif_dense.ptx")?)?;
    let module = Module::from_ptx(CString::new(ptx)?, &[])?;
    let function = module.get_function("lif_dense_update")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let mut voltage = case.voltage.clone();
    let mut refractory = case.refractory_left.clone();
    let mut spike_flags = vec![0_u8; case.len()];
    let mut device_voltage = DeviceBuffer::from_slice(&voltage)?;
    let mut device_input = DeviceBuffer::from_slice(&case.input_current)?;
    let mut device_refractory = DeviceBuffer::from_slice(&refractory)?;
    let mut device_spikes = DeviceBuffer::from_slice(&spike_flags)?;

    let block_size = 256_u32;
    let grid_size = (case.len() as u32 + block_size - 1) / block_size;
    unsafe {
        // The case vectors have matching lengths, and the kernel bounds-checks neuron ids.
        launch!(function<<<grid_size, block_size, 0, stream>>>(
            device_voltage.as_device_ptr(),
            device_input.as_device_ptr(),
            device_refractory.as_device_ptr(),
            device_spikes.as_device_ptr(),
            case.len() as u32,
            case.lif.decay,
            case.lif.threshold,
            case.lif.reset,
            case.lif.refractory
        ))?;
    }
    stream.synchronize()?;

    device_voltage.copy_to(&mut voltage)?;
    device_refractory.copy_to(&mut refractory)?;
    device_spikes.copy_to(&mut spike_flags)?;

    Ok(DenseStepOutput {
        voltage,
        refractory_left: refractory,
        spike_flags,
    })
}

pub fn validate_dense_single_step(case: &DenseInputCase) -> Result<bool> {
    validate_dense_case(case)?;
    let gpu = run_dense_single_step(case)?;
    let cpu = case.cpu_step();
    Ok(dense_outputs_match(&gpu, &cpu, 1.0e-6))
}

fn dense_outputs_match(gpu: &DenseStepOutput, cpu: &DenseStepOutput, tolerance: f32) -> bool {
    gpu.spike_flags == cpu.spike_flags
        && gpu.refractory_left == cpu.refractory_left
        && gpu.voltage.len() == cpu.voltage.len()
        && gpu
            .voltage
            .iter()
            .zip(&cpu.voltage)
            .all(|(gpu_value, cpu_value)| (gpu_value - cpu_value).abs() <= tolerance)
}

fn validate_dense_case(case: &DenseInputCase) -> Result<()> {
    case.lif.validate()?;
    if case.is_empty() {
        bail!("dense validation case must contain at least one neuron");
    }
    if case.input_current.len() != case.voltage.len()
        || case.refractory_left.len() != case.voltage.len()
    {
        bail!("dense validation buffers must have matching lengths");
    }
    Ok(())
}
