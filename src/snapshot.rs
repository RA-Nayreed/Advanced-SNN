use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use anyhow::{bail, Result};
use serde::Serialize;

use crate::config::SimulationConfig;
use crate::cpu::reference::{run_reference_observed, ReferenceResult, StepTrace};
use crate::graph::csr::CsrGraph;
use crate::types::deterministic_unit_f32;

#[derive(Clone, Debug)]
pub struct SnapshotOptions {
    pub output: PathBuf,
    pub every: usize,
    pub neuron_sample: usize,
    pub synapse_sample: usize,
}

impl SnapshotOptions {
    pub fn validate(&self) -> Result<()> {
        if self.every == 0 {
            bail!("snapshot_every must be greater than zero");
        }
        if self.neuron_sample == 0 {
            bail!("snapshot_neurons must be greater than zero");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SnapshotSynapse {
    pub source: u32,
    pub target: u32,
    pub weight: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct SnapshotNeuron {
    pub id: u32,
    pub position: [f32; 3],
    pub voltage: f32,
    pub input_current: f32,
    pub refractory_left: u16,
    pub spiked: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct SnapshotMetrics {
    pub total_spikes: u64,
    pub active_input_spikes: usize,
    pub active_output_spikes: usize,
    pub synapse_events_processed: u64,
    pub mean_sample_voltage: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct SimulationSnapshot {
    pub schema_version: u32,
    pub step: usize,
    pub neurons_total: usize,
    pub synapses_total: usize,
    pub neurons: Vec<SnapshotNeuron>,
    pub synapses: Vec<SnapshotSynapse>,
    pub metrics: SnapshotMetrics,
}

pub fn run_reference_with_snapshots(
    config: &SimulationConfig,
    graph: &CsrGraph,
    options: SnapshotOptions,
) -> Result<ReferenceResult> {
    options.validate()?;
    let mut writer = SnapshotWriter::create(config, graph, options)?;
    run_reference_observed(config, graph, &[], |trace| writer.observe(trace))
}

struct SnapshotWriter {
    writer: BufWriter<File>,
    every: usize,
    neuron_ids: Vec<usize>,
    positions: Vec<[f32; 3]>,
    synapses: Vec<SnapshotSynapse>,
    neurons_total: usize,
    synapses_total: usize,
}

impl SnapshotWriter {
    fn create(config: &SimulationConfig, graph: &CsrGraph, options: SnapshotOptions) -> Result<Self> {
        let file = File::create(&options.output)?;
        let neuron_ids = sampled_indices(config.neurons, options.neuron_sample);
        let positions = (0..config.neurons)
            .map(|id| synthetic_position(config.seed, config.neurons, id))
            .collect();
        let synapses = sample_synapses(graph, options.synapse_sample);

        Ok(Self {
            writer: BufWriter::new(file),
            every: options.every,
            neuron_ids,
            positions,
            synapses,
            neurons_total: config.neurons,
            synapses_total: graph.synapses(),
        })
    }

    fn observe(&mut self, trace: StepTrace<'_>) -> Result<()> {
        if trace.step % self.every != 0 {
            return Ok(());
        }

        let mut voltage_sum = 0.0_f32;
        let neurons = self
            .neuron_ids
            .iter()
            .map(|&id| {
                let voltage = trace.state.voltage[id];
                voltage_sum += voltage;
                SnapshotNeuron {
                    id: id as u32,
                    position: self.positions[id],
                    voltage,
                    input_current: trace.state.input_current[id],
                    refractory_left: trace.state.refractory_left[id],
                    spiked: trace.output_spikes.binary_search(&(id as u32)).is_ok(),
                }
            })
            .collect::<Vec<_>>();

        let mean_sample_voltage = if neurons.is_empty() {
            0.0
        } else {
            voltage_sum / neurons.len() as f32
        };

        let snapshot = SimulationSnapshot {
            schema_version: 1,
            step: trace.step,
            neurons_total: self.neurons_total,
            synapses_total: self.synapses_total,
            neurons,
            synapses: self.synapses.clone(),
            metrics: SnapshotMetrics {
                total_spikes: trace.total_spikes,
                active_input_spikes: trace.input_spikes.len(),
                active_output_spikes: trace.output_spikes.len(),
                synapse_events_processed: trace.synapse_events_processed,
                mean_sample_voltage,
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

fn sample_synapses(graph: &CsrGraph, max_items: usize) -> Vec<SnapshotSynapse> {
    if max_items == 0 {
        return Vec::new();
    }

    let mut synapses = Vec::with_capacity(max_items.min(graph.synapses()));
    for source in 0..graph.neurons() {
        for edge in graph.outgoing_range(source) {
            synapses.push(SnapshotSynapse {
                source: source as u32,
                target: graph.targets()[edge],
                weight: graph.weights()[edge],
            });
            if synapses.len() >= max_items {
                return synapses;
            }
        }
    }
    synapses
}

pub fn synthetic_position(seed: u64, total: usize, neuron_id: usize) -> [f32; 3] {
    let u = deterministic_unit_f32(seed ^ 0xB10B_0001, 0, neuron_id);
    let v = deterministic_unit_f32(seed ^ 0xB10B_0002, 1, neuron_id);
    let w = deterministic_unit_f32(seed ^ 0xB10B_0003, 2, neuron_id);
    let theta = std::f32::consts::TAU * u;
    let z = 2.0 * v - 1.0;
    let ring = (1.0 - z * z).max(0.0).sqrt();
    let density = if total <= 1 {
        1.0
    } else {
        (neuron_id as f32 / (total - 1) as f32).mul_add(0.15, 0.85)
    };
    let radius = w.powf(1.0 / 3.0) * density;

    [
        theta.cos() * ring * radius * 1.45,
        theta.sin() * ring * radius * 0.9,
        z * radius * 0.75,
    ]
}
