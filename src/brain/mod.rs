use anyhow::{bail, Result};

use crate::graph::csr::CsrGraph;
use crate::snapshot::{SnapshotLayout, SnapshotRegion};
use crate::types::{deterministic_unit_f32, splitmix64};

#[derive(Clone, Debug)]
pub struct BrainBlobConfig {
    pub neurons: usize,
    pub fanout: usize,
    pub seed: u64,
    pub inhibitory_fraction: f32,
}

impl BrainBlobConfig {
    pub fn new(neurons: usize, fanout: usize, seed: u64) -> Self {
        Self {
            neurons,
            fanout,
            seed,
            inhibitory_fraction: 0.2,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.neurons == 0 {
            bail!("neurons must be greater than zero");
        }
        if self.fanout == 0 {
            bail!("fanout must be greater than zero");
        }
        if !(0.0..=1.0).contains(&self.inhibitory_fraction) {
            bail!("inhibitory_fraction must be in [0, 1]");
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct BrainBlob {
    pub regions: Vec<BrainRegion>,
    pub neurons: Vec<BrainNeuron>,
    pub graph: CsrGraph,
}

impl BrainBlob {
    pub fn snapshot_layout(&self) -> SnapshotLayout {
        SnapshotLayout::new(
            self.regions
                .iter()
                .map(|region| SnapshotRegion {
                    id: region.id,
                    name: region.name.clone(),
                    center: region.center,
                    radius: region.radius,
                    color: region.color,
                })
                .collect(),
            self.neurons.iter().map(|neuron| neuron.position).collect(),
            self.neurons.iter().map(|neuron| neuron.region_id).collect(),
            self.neurons
                .iter()
                .map(|neuron| neuron.kind.as_str().to_string())
                .collect(),
        )
    }
}

#[derive(Clone, Debug)]
pub struct BrainRegion {
    pub id: usize,
    pub name: String,
    pub center: [f32; 3],
    pub radius: f32,
    pub color: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrainNeuronKind {
    Excitatory,
    Inhibitory,
}

impl BrainNeuronKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Excitatory => "excitatory",
            Self::Inhibitory => "inhibitory",
        }
    }
}

#[derive(Clone, Debug)]
pub struct BrainNeuron {
    pub id: usize,
    pub region_id: usize,
    pub kind: BrainNeuronKind,
    pub position: [f32; 3],
}

pub fn generate_brain_blob(config: &BrainBlobConfig) -> Result<BrainBlob> {
    config.validate()?;
    let synapses = config
        .neurons
        .checked_mul(config.fanout)
        .ok_or_else(|| anyhow::anyhow!("synapse count overflow"))?;
    if synapses > u32::MAX as usize {
        bail!("synapse count exceeds u32 CSR index capacity");
    }

    let regions = default_regions();
    let neurons = generate_neurons(config, &regions);
    let graph = generate_synapses(config, &regions, &neurons, synapses)?;

    Ok(BrainBlob {
        regions,
        neurons,
        graph,
    })
}

fn default_regions() -> Vec<BrainRegion> {
    vec![
        BrainRegion {
            id: 0,
            name: String::from("sensory"),
            center: [-0.72, -0.18, 0.08],
            radius: 0.42,
            color: [0.18, 0.72, 1.0],
        },
        BrainRegion {
            id: 1,
            name: String::from("association"),
            center: [-0.18, 0.18, 0.02],
            radius: 0.5,
            color: [0.62, 0.92, 0.35],
        },
        BrainRegion {
            id: 2,
            name: String::from("memory"),
            center: [0.32, -0.24, -0.08],
            radius: 0.44,
            color: [1.0, 0.67, 0.23],
        },
        BrainRegion {
            id: 3,
            name: String::from("motor"),
            center: [0.74, 0.12, 0.1],
            radius: 0.4,
            color: [1.0, 0.28, 0.42],
        },
        BrainRegion {
            id: 4,
            name: String::from("core"),
            center: [0.02, 0.0, -0.28],
            radius: 0.34,
            color: [0.72, 0.55, 1.0],
        },
    ]
}

fn generate_neurons(config: &BrainBlobConfig, regions: &[BrainRegion]) -> Vec<BrainNeuron> {
    (0..config.neurons)
        .map(|id| {
            let region_id = region_for_neuron(id, config.neurons, regions.len());
            let kind_noise = deterministic_unit_f32(config.seed ^ 0x1A2B_3C4D, 0, id);
            let kind = if kind_noise < config.inhibitory_fraction {
                BrainNeuronKind::Inhibitory
            } else {
                BrainNeuronKind::Excitatory
            };
            BrainNeuron {
                id,
                region_id,
                kind,
                position: position_in_region(config.seed, id, &regions[region_id]),
            }
        })
        .collect()
}

fn generate_synapses(
    config: &BrainBlobConfig,
    regions: &[BrainRegion],
    neurons: &[BrainNeuron],
    synapses: usize,
) -> Result<CsrGraph> {
    let mut row_ptr = Vec::with_capacity(config.neurons + 1);
    let mut targets = Vec::with_capacity(synapses);
    let mut weights = Vec::with_capacity(synapses);

    row_ptr.push(0);
    for source in 0..config.neurons {
        let next = (source + 1)
            .checked_mul(config.fanout)
            .ok_or_else(|| anyhow::anyhow!("row pointer overflow"))?;
        row_ptr.push(next as u32);

        for edge_offset in 0..config.fanout {
            let target = choose_target(config, regions, neurons, source, edge_offset);
            targets.push(target as u32);
            weights.push(weight_for_source(config.seed, neurons[source].kind, source, edge_offset));
        }
    }

    CsrGraph::new(config.neurons, row_ptr, targets, weights)
}

fn choose_target(
    config: &BrainBlobConfig,
    regions: &[BrainRegion],
    neurons: &[BrainNeuron],
    source: usize,
    edge_offset: usize,
) -> usize {
    if config.neurons == 1 {
        return 0;
    }

    let source_region = neurons[source].region_id;
    let locality = deterministic_unit_f32(config.seed ^ 0xC011_EC7, source, edge_offset);
    let requested_region = if locality < 0.74 {
        source_region
    } else if locality < 0.92 {
        (source_region + 1 + hashed_index(config.seed, source, edge_offset, 0x25, regions.len() - 1))
            % regions.len()
    } else {
        hashed_index(config.seed, source, edge_offset, 0x5A, regions.len())
    };

    let mut target = pick_target_in_region(
        config.neurons,
        regions.len(),
        requested_region,
        config.seed,
        source,
        edge_offset,
    );
    if target == source {
        target = (target + 1) % config.neurons;
    }
    target
}

fn pick_target_in_region(
    neurons: usize,
    region_count: usize,
    requested_region: usize,
    seed: u64,
    source: usize,
    edge_offset: usize,
) -> usize {
    for offset in 0..region_count {
        let region_id = (requested_region + offset) % region_count;
        let (start, end) = region_bounds(neurons, region_id, region_count);
        if end > start {
            let index = hashed_index(seed, source, edge_offset, 0xA7 + offset as u64, end - start);
            return start + index;
        }
    }
    hashed_index(seed, source, edge_offset, 0xFF, neurons)
}

fn region_for_neuron(neuron_id: usize, neurons: usize, regions: usize) -> usize {
    let region = neuron_id.saturating_mul(regions) / neurons.max(1);
    region.min(regions - 1)
}

fn region_bounds(neurons: usize, region_id: usize, regions: usize) -> (usize, usize) {
    (
        neurons.saturating_mul(region_id) / regions,
        neurons.saturating_mul(region_id + 1) / regions,
    )
}

fn position_in_region(seed: u64, neuron_id: usize, region: &BrainRegion) -> [f32; 3] {
    let u = deterministic_unit_f32(seed ^ 0xB10B_1001, 0, neuron_id);
    let v = deterministic_unit_f32(seed ^ 0xB10B_1002, 1, neuron_id);
    let w = deterministic_unit_f32(seed ^ 0xB10B_1003, 2, neuron_id);
    let theta = std::f32::consts::TAU * u;
    let z = 2.0 * v - 1.0;
    let ring = (1.0 - z * z).max(0.0).sqrt();
    let radius = region.radius * w.powf(1.0 / 3.0);

    [
        region.center[0] + theta.cos() * ring * radius * 1.25,
        region.center[1] + theta.sin() * ring * radius * 0.82,
        region.center[2] + z * radius * 0.7,
    ]
}

fn weight_for_source(seed: u64, kind: BrainNeuronKind, source: usize, edge_offset: usize) -> f32 {
    let unit = deterministic_unit_f32(seed ^ 0x517A_7E5, source, edge_offset);
    match kind {
        BrainNeuronKind::Excitatory => 0.018 + 0.052 * unit,
        BrainNeuronKind::Inhibitory => -(0.02 + 0.06 * unit),
    }
}

fn hashed_index(seed: u64, source: usize, edge_offset: usize, salt: u64, modulo: usize) -> usize {
    if modulo == 0 {
        return 0;
    }
    let key = seed
        ^ (source as u64).wrapping_mul(0x9E37_79B1_85EB_CA87)
        ^ (edge_offset as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ salt.wrapping_mul(0xA076_1D64_78BD_642F);
    (splitmix64(key) % modulo as u64) as usize
}
