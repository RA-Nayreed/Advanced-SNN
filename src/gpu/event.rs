use anyhow::Result;

use crate::config::SimulationConfig;
use crate::graph::csr::CsrGraph;
use crate::metrics::EventGpuMetrics;

#[cfg(not(feature = "cuda"))]
pub fn run_event(_config: &SimulationConfig, _graph: &CsrGraph) -> Result<EventGpuMetrics> {
    Err(crate::gpu::cuda::unavailable(
        "this binary was built without the cuda feature",
    ))
}

#[cfg(feature = "cuda")]
pub fn run_event(config: &SimulationConfig, graph: &CsrGraph) -> Result<EventGpuMetrics> {
    use std::ffi::CString;
    use std::time::Instant;

    use cust::launch;
    use cust::prelude::*;

    config.validate()?;
    graph.validate()?;
    let runtime = crate::gpu::device::initialize_device()?;
    let ptx = std::fs::read_to_string(crate::gpu::cuda::ptx_path("event_snn.ptx")?)?;
    let module = Module::from_ptx(CString::new(ptx)?, &[])?;
    let clear_input = module.get_function("clear_input_kernel")?;
    let apply_external = module.get_function("apply_external_input_kernel")?;
    let process_spikes = module.get_function("process_spikes_kernel")?;
    let update_neurons = module.get_function("update_neurons_kernel")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let mut voltage = vec![0.0_f32; config.neurons];
    let input_current = vec![0.0_f32; config.neurons];
    let mut refractory_left = vec![0_u16; config.neurons];
    let mut active_spikes = vec![0_u32; config.neurons];
    let mut next_spikes = vec![0_u32; config.neurons];
    let mut active_count = [0_u32];
    let mut next_count = [0_u32];
    let mut event_count = [0_u64];

    let d_row_ptr = DeviceBuffer::from_slice(graph.row_ptr())?;
    let d_targets = DeviceBuffer::from_slice(graph.targets())?;
    let d_weights = DeviceBuffer::from_slice(graph.weights())?;
    let mut d_voltage = DeviceBuffer::from_slice(&voltage)?;
    let mut d_input_current = DeviceBuffer::from_slice(&input_current)?;
    let mut d_refractory_left = DeviceBuffer::from_slice(&refractory_left)?;
    let mut d_active_spikes = DeviceBuffer::from_slice(&active_spikes)?;
    let mut d_active_count = DeviceBuffer::from_slice(&active_count)?;
    let mut d_next_spikes = DeviceBuffer::from_slice(&next_spikes)?;
    let mut d_next_count = DeviceBuffer::from_slice(&next_count)?;
    let mut d_event_count = DeviceBuffer::from_slice(&event_count)?;

    let block_size = 256_u32;
    let neuron_grid = (config.neurons as u32 + block_size - 1) / block_size;
    let mut total_spikes = 0_u64;
    let mut active_sum = 0_u64;
    let mut spikes_per_step = Vec::with_capacity(config.steps);
    let started = Instant::now();
    let mut kernel_elapsed_seconds = 0.0_f64;

    for step in 0..config.steps {
        active_sum += active_count[0] as u64;
        next_count[0] = 0;
        d_next_count.copy_from(&next_count)?;

        let kernel_started = Instant::now();
        unsafe {
            // All device buffers are allocated for the configured neuron and graph sizes.
            launch!(clear_input<<<neuron_grid, block_size, 0, stream>>>(
                d_input_current.as_device_ptr(),
                config.neurons as u32
            ))?;
            launch!(apply_external<<<neuron_grid, block_size, 0, stream>>>(
                d_input_current.as_device_ptr(),
                config.neurons as u32,
                config.seed,
                step as u64,
                config.external_prob,
                config.external_current
            ))?;
        }

        let spike_grid = if active_count[0] == 0 {
            1
        } else {
            (active_count[0] + block_size - 1) / block_size
        };
        unsafe {
            // Active spike ids are produced by the update kernel and bounded by neuron count.
            launch!(process_spikes<<<spike_grid, block_size, 0, stream>>>(
                d_row_ptr.as_device_ptr(),
                d_targets.as_device_ptr(),
                d_weights.as_device_ptr(),
                d_active_spikes.as_device_ptr(),
                d_active_count.as_device_ptr(),
                d_input_current.as_device_ptr(),
                d_event_count.as_device_ptr()
            ))?;
            launch!(update_neurons<<<neuron_grid, block_size, 0, stream>>>(
                d_voltage.as_device_ptr(),
                d_input_current.as_device_ptr(),
                d_refractory_left.as_device_ptr(),
                d_next_spikes.as_device_ptr(),
                d_next_count.as_device_ptr(),
                config.neurons as u32,
                config.lif.decay,
                config.lif.threshold,
                config.lif.reset,
                config.lif.refractory
            ))?;
        }
        stream.synchronize()?;
        kernel_elapsed_seconds += kernel_started.elapsed().as_secs_f64();

        d_next_count.copy_to(&mut next_count)?;
        total_spikes += next_count[0] as u64;
        spikes_per_step.push(next_count[0] as usize);

        std::mem::swap(&mut d_active_spikes, &mut d_next_spikes);
        active_count[0] = next_count[0];
        d_active_count.copy_from(&active_count)?;
    }

    d_voltage.copy_to(&mut voltage)?;
    d_refractory_left.copy_to(&mut refractory_left)?;
    d_active_spikes.copy_to(&mut active_spikes)?;
    d_next_spikes.copy_to(&mut next_spikes)?;
    d_event_count.copy_to(&mut event_count)?;

    let elapsed_seconds = started.elapsed().as_secs_f64();
    Ok(EventGpuMetrics {
        selected_cuda_device: runtime.device_name,
        neurons: config.neurons,
        synapses: graph.synapses(),
        steps: config.steps,
        total_spikes,
        mean_active_spikes_per_step: active_sum as f64 / config.steps as f64,
        synapse_events_processed: event_count[0],
        synapse_events_per_second: if elapsed_seconds > 0.0 {
            event_count[0] as f64 / elapsed_seconds
        } else {
            0.0
        },
        elapsed_seconds,
        kernel_elapsed_seconds: Some(kernel_elapsed_seconds),
        spikes_per_step,
    })
}
