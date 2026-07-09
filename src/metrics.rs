use std::fmt;

#[derive(Clone, Debug)]
pub struct SimulationMetrics {
    pub neurons: usize,
    pub synapses: usize,
    pub steps: usize,
    pub total_spikes: u64,
    pub mean_spikes_per_step: f64,
    pub final_active_spikes: usize,
    pub elapsed_seconds: f64,
    pub synapse_events_processed: u64,
    pub synapse_events_per_second: f64,
    pub spikes_per_step: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct DenseGpuMetrics {
    pub selected_cuda_device: String,
    pub neurons: usize,
    pub steps: usize,
    pub total_spikes: u64,
    pub kernel_elapsed_seconds: Option<f64>,
    pub cpu_gpu_match: bool,
}

#[derive(Clone, Debug)]
pub struct EventGpuMetrics {
    pub selected_cuda_device: String,
    pub neurons: usize,
    pub synapses: usize,
    pub steps: usize,
    pub total_spikes: u64,
    pub mean_active_spikes_per_step: f64,
    pub synapse_events_processed: u64,
    pub synapse_events_per_second: f64,
    pub elapsed_seconds: f64,
    pub kernel_elapsed_seconds: Option<f64>,
    pub spikes_per_step: Vec<usize>,
}

impl SimulationMetrics {
    pub fn deterministic_eq(&self, other: &Self) -> bool {
        self.neurons == other.neurons
            && self.synapses == other.synapses
            && self.steps == other.steps
            && self.total_spikes == other.total_spikes
            && self.final_active_spikes == other.final_active_spikes
            && self.synapse_events_processed == other.synapse_events_processed
            && self.spikes_per_step == other.spikes_per_step
    }
}

impl fmt::Display for SimulationMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "neurons={}", self.neurons)?;
        writeln!(f, "synapses={}", self.synapses)?;
        writeln!(f, "steps={}", self.steps)?;
        writeln!(f, "total_spikes={}", self.total_spikes)?;
        writeln!(
            f,
            "mean_spikes_per_step={:.6}",
            self.mean_spikes_per_step
        )?;
        writeln!(f, "final_active_spikes={}", self.final_active_spikes)?;
        writeln!(f, "elapsed_seconds={:.6}", self.elapsed_seconds)?;
        writeln!(
            f,
            "synapse_events_processed={}",
            self.synapse_events_processed
        )?;
        writeln!(
            f,
            "synapse_events_per_second={:.6}",
            self.synapse_events_per_second
        )
    }
}

impl fmt::Display for DenseGpuMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "selected_cuda_device={}", self.selected_cuda_device)?;
        writeln!(f, "neurons={}", self.neurons)?;
        writeln!(f, "steps={}", self.steps)?;
        writeln!(f, "total_spikes={}", self.total_spikes)?;
        match self.kernel_elapsed_seconds {
            Some(seconds) => writeln!(f, "kernel_elapsed_seconds={seconds:.6}")?,
            None => writeln!(f, "kernel_elapsed_seconds=unavailable")?,
        }
        writeln!(f, "cpu_gpu_match={}", self.cpu_gpu_match)
    }
}

impl fmt::Display for EventGpuMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "selected_cuda_device={}", self.selected_cuda_device)?;
        writeln!(f, "neurons={}", self.neurons)?;
        writeln!(f, "synapses={}", self.synapses)?;
        writeln!(f, "steps={}", self.steps)?;
        writeln!(f, "total_spikes={}", self.total_spikes)?;
        writeln!(
            f,
            "mean_active_spikes_per_step={:.6}",
            self.mean_active_spikes_per_step
        )?;
        writeln!(
            f,
            "synapse_events_processed={}",
            self.synapse_events_processed
        )?;
        writeln!(
            f,
            "synapse_events_per_second={:.6}",
            self.synapse_events_per_second
        )?;
        writeln!(f, "elapsed_seconds={:.6}", self.elapsed_seconds)?;
        match self.kernel_elapsed_seconds {
            Some(seconds) => writeln!(f, "kernel_elapsed_seconds={seconds:.6}")?,
            None => writeln!(f, "kernel_elapsed_seconds=unavailable")?,
        }
        Ok(())
    }
}
