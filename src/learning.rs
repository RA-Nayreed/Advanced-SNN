use std::collections::HashSet;
use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Write};

use anyhow::{bail, Result};

use crate::config::SimulationConfig;
use crate::graph::csr::CsrGraph;
use crate::neuron::lif::update_lif_neuron;
use crate::neuron::state::NeuronState;
use crate::snapshot::{
    SimulationSnapshot, SnapshotLayout, SnapshotMetrics, SnapshotNeuron, SnapshotOptions,
    SnapshotSynapse,
};
use crate::types::{external_input_applies, NeuronId};

#[derive(Clone, Copy, Debug)]
pub struct StdpConfig {
    pub enabled: bool,
    pub potentiation: f32,
    pub depression: f32,
    pub window_steps: usize,
    pub min_weight: f32,
    pub max_weight: f32,
}

impl Default for StdpConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            potentiation: 0.004,
            depression: 0.003,
            window_steps: 8,
            min_weight: 0.001,
            max_weight: 0.2,
        }
    }
}

impl StdpConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.potentiation.is_finite() || self.potentiation < 0.0 {
            bail!("stdp potentiation must be finite and nonnegative");
        }
        if !self.depression.is_finite() || self.depression < 0.0 {
            bail!("stdp depression must be finite and nonnegative");
        }
        if self.window_steps == 0 {
            bail!("stdp window_steps must be greater than zero");
        }
        if self.min_weight <= 0.0 || self.max_weight <= self.min_weight {
            bail!("stdp weight clamp must satisfy 0 < min_weight < max_weight");
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct LearningResult {
    pub neurons: usize,
    pub synapses: usize,
    pub steps: usize,
    pub total_spikes: u64,
    pub potentiated: u64,
    pub depressed: u64,
    pub mean_weight: f32,
}

impl fmt::Display for LearningResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "neurons={}", self.neurons)?;
        writeln!(f, "synapses={}", self.synapses)?;
        writeln!(f, "steps={}", self.steps)?;
        writeln!(f, "total_spikes={}", self.total_spikes)?;
        writeln!(f, "stdp_potentiated={}", self.potentiated)?;
        writeln!(f, "stdp_depressed={}", self.depressed)?;
        writeln!(f, "mean_weight={:.6}", self.mean_weight)
    }
}

pub fn run_stdp_learning(
    config: &SimulationConfig,
    graph: &mut CsrGraph,
    layout: SnapshotLayout,
    stdp: StdpConfig,
    snapshots: Option<SnapshotOptions>,
) -> Result<LearningResult> {
    config.validate()?;
    graph.validate()?;
    stdp.validate()?;
    layout.validate(config.neurons)?;
    if graph.neurons() != config.neurons {
        bail!("graph neuron count does not match config");
    }

    let mut writer = match snapshots {
        Some(options) => Some(LearningSnapshotWriter::new(config, graph, layout.clone(), options)?),
        None => None,
    };

    let mut state = NeuronState::new(config.neurons);
    let mut active_spikes = Vec::with_capacity(config.neurons);
    let mut next_spikes = Vec::with_capacity(config.neurons);
    let mut last_spike_step = vec![None; config.neurons];
    let original_signs = graph
        .weights()
        .iter()
        .map(|weight| if *weight < 0.0 { -1.0 } else { 1.0 })
        .collect::<Vec<_>>();

    let mut total_spikes = 0_u64;
    let mut synapse_events_processed = 0_u64;
    let mut potentiated = 0_u64;
    let mut depressed = 0_u64;

    for step in 0..config.steps {
        state.clear_input();

        for neuron_id in 0..config.neurons {
            if external_input_applies(config.seed, step, neuron_id, config.external_prob) {
                state.input_current[neuron_id] += config.external_current;
            }
        }

        for &source in &active_spikes {
            let source_index = source as usize;
            let range = graph.outgoing_range(source_index);
            synapse_events_processed += range.len() as u64;
            for edge in range {
                let target = graph.targets()[edge] as usize;
                state.input_current[target] += graph.weights()[edge];
            }
        }

        next_spikes.clear();
        for neuron_id in 0..config.neurons {
            let spiked = update_lif_neuron(
                &mut state.voltage[neuron_id],
                state.input_current[neuron_id],
                &mut state.refractory_left[neuron_id],
                config.lif,
            );
            if spiked {
                next_spikes.push(neuron_id as NeuronId);
            }
        }

        let output_spike_set = next_spikes.iter().copied().collect::<HashSet<_>>();
        if stdp.enabled {
            apply_stdp(
                graph,
                &original_signs,
                &active_spikes,
                &output_spike_set,
                &last_spike_step,
                step,
                stdp,
                &mut potentiated,
                &mut depressed,
            );
        }

        for &spike in &next_spikes {
            last_spike_step[spike as usize] = Some(step);
        }

        total_spikes += next_spikes.len() as u64;
        if let Some(writer) = writer.as_mut() {
            writer.observe(
                step,
                &state,
                &active_spikes,
                &next_spikes,
                graph,
                total_spikes,
                synapse_events_processed,
                potentiated,
                depressed,
            )?;
        }
        std::mem::swap(&mut active_spikes, &mut next_spikes);
    }

    Ok(LearningResult {
        neurons: config.neurons,
        synapses: graph.synapses(),
        steps: config.steps,
        total_spikes,
        potentiated,
        depressed,
        mean_weight: mean_abs_weight(graph.weights()),
    })
}

fn apply_stdp(
    graph: &mut CsrGraph,
    original_signs: &[f32],
    active_spikes: &[NeuronId],
    output_spikes: &HashSet<NeuronId>,
    last_spike_step: &[Option<usize>],
    step: usize,
    stdp: StdpConfig,
    potentiated: &mut u64,
    depressed: &mut u64,
) {
    let source_step = step.saturating_sub(1);
    let row_ptr = graph.row_ptr().to_vec();
    let targets = graph.targets().to_vec();
    let weights = graph.weights_mut();

    for &source in active_spikes {
        let source_index = source as usize;
        let range = row_ptr[source_index] as usize..row_ptr[source_index + 1] as usize;
        for edge in range {
            let target = targets[edge] as usize;
            let sign = original_signs[edge];
            if output_spikes.contains(&(target as NeuronId)) {
                weights[edge] = clamp_signed_weight(
                    weights[edge] + sign * stdp.potentiation,
                    sign,
                    stdp.min_weight,
                    stdp.max_weight,
                );
                *potentiated += 1;
            } else if let Some(last_post) = last_spike_step[target] {
                if last_post <= source_step && source_step - last_post <= stdp.window_steps {
                    weights[edge] = clamp_signed_weight(
                        weights[edge] - sign * stdp.depression,
                        sign,
                        stdp.min_weight,
                        stdp.max_weight,
                    );
                    *depressed += 1;
                }
            }
        }
    }
}

fn clamp_signed_weight(value: f32, sign: f32, min_abs: f32, max_abs: f32) -> f32 {
    let magnitude = value.abs().clamp(min_abs, max_abs);
    sign * magnitude
}

fn mean_abs_weight(weights: &[f32]) -> f32 {
    if weights.is_empty() {
        0.0
    } else {
        weights.iter().map(|weight| weight.abs()).sum::<f32>() / weights.len() as f32
    }
}

struct LearningSnapshotWriter {
    writer: BufWriter<File>,
    every: usize,
    neuron_ids: Vec<usize>,
    synapse_edges: Vec<(usize, usize)>,
    layout: SnapshotLayout,
    neurons_total: usize,
    synapses_total: usize,
}

impl LearningSnapshotWriter {
    fn new(
        config: &SimulationConfig,
        graph: &CsrGraph,
        layout: SnapshotLayout,
        options: SnapshotOptions,
    ) -> Result<Self> {
        options.validate()?;
        let file = File::create(&options.output)?;
        Ok(Self {
            writer: BufWriter::new(file),
            every: options.every,
            neuron_ids: sampled_indices(config.neurons, options.neuron_sample),
            synapse_edges: sampled_synapse_edges(graph, options.synapse_sample),
            layout,
            neurons_total: config.neurons,
            synapses_total: graph.synapses(),
        })
    }

    fn observe(
        &mut self,
        step: usize,
        state: &NeuronState,
        input_spikes: &[NeuronId],
        output_spikes: &[NeuronId],
        graph: &CsrGraph,
        total_spikes: u64,
        synapse_events_processed: u64,
        potentiated: u64,
        depressed: u64,
    ) -> Result<()> {
        if step % self.every != 0 {
            return Ok(());
        }

        let mut voltage_sum = 0.0_f32;
        let neurons = self
            .neuron_ids
            .iter()
            .map(|&id| {
                let voltage = state.voltage[id];
                voltage_sum += voltage;
                SnapshotNeuron {
                    id: id as u32,
                    region_id: self.layout.region_ids[id],
                    kind: self.layout.kinds[id].clone(),
                    position: self.layout.positions[id],
                    voltage,
                    input_current: state.input_current[id],
                    refractory_left: state.refractory_left[id],
                    spiked: output_spikes.binary_search(&(id as u32)).is_ok(),
                }
            })
            .collect::<Vec<_>>();

        let synapses = self
            .synapse_edges
            .iter()
            .map(|&(source, edge)| SnapshotSynapse {
                source: source as u32,
                target: graph.targets()[edge],
                weight: graph.weights()[edge],
            })
            .collect::<Vec<_>>();

        let mean_sample_voltage = if neurons.is_empty() {
            0.0
        } else {
            voltage_sum / neurons.len() as f32
        };

        let snapshot = SimulationSnapshot {
            schema_version: 3,
            step,
            neurons_total: self.neurons_total,
            synapses_total: self.synapses_total,
            regions: self.layout.regions.clone(),
            neurons,
            synapses,
            metrics: SnapshotMetrics {
                total_spikes,
                active_input_spikes: input_spikes.len(),
                active_output_spikes: output_spikes.len(),
                synapse_events_processed,
                mean_sample_voltage,
                stdp_potentiated: Some(potentiated),
                stdp_depressed: Some(depressed),
                mean_abs_weight: Some(mean_abs_weight(graph.weights())),
            },
        };

        serde_json::to_writer(&mut self.writer, &snapshot)?;
        self.writer.write_all(b"\n")?;
        Ok(())
    }
}

fn sampled_indices(total: usize, max_items: usize) -> Vec<usize> {
    if max_items >= total {
        return (0..total).collect();
    }
    let stride = (total / max_items) + usize::from(total % max_items != 0);
    (0..total).step_by(stride).take(max_items).collect()
}

fn sampled_synapse_edges(graph: &CsrGraph, max_items: usize) -> Vec<(usize, usize)> {
    if max_items == 0 {
        return Vec::new();
    }
    let mut edges = Vec::with_capacity(max_items.min(graph.synapses()));
    for source in 0..graph.neurons() {
        for edge in graph.outgoing_range(source) {
            edges.push((source, edge));
            if edges.len() >= max_items {
                return edges;
            }
        }
    }
    edges
}
