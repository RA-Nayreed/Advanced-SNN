use std::time::Instant;

use anyhow::{bail, Result};

use crate::config::SimulationConfig;
use crate::graph::csr::CsrGraph;
use crate::metrics::SimulationMetrics;
use crate::neuron::lif::update_lif_neuron;
use crate::neuron::state::NeuronState;
use crate::types::{external_input_applies, NeuronId};

#[derive(Clone, Debug)]
pub struct ReferenceResult {
    pub metrics: SimulationMetrics,
    pub state: NeuronState,
    pub active_spikes: Vec<NeuronId>,
}

pub fn run_reference(config: &SimulationConfig, graph: &CsrGraph) -> Result<ReferenceResult> {
    run_reference_with_initial_spikes(config, graph, &[])
}

pub fn run_reference_with_initial_spikes(
    config: &SimulationConfig,
    graph: &CsrGraph,
    initial_active_spikes: &[NeuronId],
) -> Result<ReferenceResult> {
    run_reference_from_state(
        config,
        graph,
        NeuronState::new(config.neurons),
        initial_active_spikes,
    )
}

pub fn run_reference_from_state(
    config: &SimulationConfig,
    graph: &CsrGraph,
    mut state: NeuronState,
    initial_active_spikes: &[NeuronId],
) -> Result<ReferenceResult> {
    config.validate()?;
    graph.validate()?;
    state.validate_len(config.neurons)?;
    if graph.neurons() != config.neurons {
        bail!("graph neuron count does not match config");
    }

    let mut active_spikes = Vec::with_capacity(config.neurons);
    for &spike in initial_active_spikes {
        if spike as usize >= config.neurons {
            bail!("initial spike neuron id {spike} is out of range");
        }
        active_spikes.push(spike);
    }

    let mut next_spikes = Vec::with_capacity(config.neurons);
    let mut total_spikes = 0_u64;
    let mut synapse_events_processed = 0_u64;
    let mut spikes_per_step = Vec::with_capacity(config.steps);

    let started = Instant::now();
    for step in 0..config.steps {
        state.clear_input();

        for neuron_id in 0..config.neurons {
            if external_input_applies(config.seed, step, neuron_id, config.external_prob) {
                state.input_current[neuron_id] += config.external_current;
            }
        }

        for &source in &active_spikes {
            let source_index = source as usize;
            if source_index >= config.neurons {
                bail!("active spike neuron id {source} is out of range");
            }
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

        total_spikes += next_spikes.len() as u64;
        spikes_per_step.push(next_spikes.len());
        std::mem::swap(&mut active_spikes, &mut next_spikes);
    }

    let elapsed_seconds = started.elapsed().as_secs_f64();
    let metrics = SimulationMetrics {
        neurons: config.neurons,
        synapses: graph.synapses(),
        steps: config.steps,
        total_spikes,
        mean_spikes_per_step: total_spikes as f64 / config.steps as f64,
        final_active_spikes: active_spikes.len(),
        elapsed_seconds,
        synapse_events_processed,
        synapse_events_per_second: if elapsed_seconds > 0.0 {
            synapse_events_processed as f64 / elapsed_seconds
        } else {
            0.0
        },
        spikes_per_step,
    };

    Ok(ReferenceResult {
        metrics,
        state,
        active_spikes,
    })
}
