use anyhow::{bail, Result};

use crate::distributed::partition::NeuronPartition;
use crate::types::splitmix64;

#[derive(Clone, Debug, PartialEq)]
pub struct DistributedCsrGraph {
    global_neurons: usize,
    local_start: usize,
    local_neurons: usize,
    fanout: usize,
    row_ptr: Vec<u32>,
    targets_global: Vec<u64>,
    weights: Vec<f32>,
}

impl DistributedCsrGraph {
    pub fn new(
        global_neurons: usize,
        local_start: usize,
        local_neurons: usize,
        fanout: usize,
        row_ptr: Vec<u32>,
        targets_global: Vec<u64>,
        weights: Vec<f32>,
    ) -> Result<Self> {
        let graph = Self {
            global_neurons,
            local_start,
            local_neurons,
            fanout,
            row_ptr,
            targets_global,
            weights,
        };
        graph.validate()?;
        Ok(graph)
    }

    pub fn global_neurons(&self) -> usize {
        self.global_neurons
    }

    pub fn local_start(&self) -> usize {
        self.local_start
    }

    pub fn local_neurons(&self) -> usize {
        self.local_neurons
    }

    pub fn fanout(&self) -> usize {
        self.fanout
    }

    pub fn local_synapses(&self) -> usize {
        self.targets_global.len()
    }

    pub fn row_ptr(&self) -> &[u32] {
        &self.row_ptr
    }

    pub fn targets_global(&self) -> &[u64] {
        &self.targets_global
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn outgoing_range(&self, local_source: usize) -> std::ops::Range<usize> {
        let start = self.row_ptr[local_source] as usize;
        let end = self.row_ptr[local_source + 1] as usize;
        start..end
    }

    pub fn validate(&self) -> Result<()> {
        if self.global_neurons == 0 {
            bail!("global_neurons must be greater than zero");
        }
        if self.local_start > self.global_neurons {
            bail!("local_start is out of range");
        }
        if self.local_start + self.local_neurons > self.global_neurons {
            bail!("local partition exceeds global neuron count");
        }
        if self.fanout == 0 {
            bail!("fanout must be greater than zero");
        }
        if self.row_ptr.len() != self.local_neurons + 1 {
            bail!("row_ptr length must be local_neurons + 1");
        }
        if self.targets_global.len() != self.weights.len() {
            bail!("targets_global and weights lengths must match");
        }
        if self.row_ptr.first().copied() != Some(0) {
            bail!("row_ptr must start at zero");
        }
        for pair in self.row_ptr.windows(2) {
            if pair[0] > pair[1] {
                bail!("row_ptr must be monotonically nondecreasing");
            }
        }
        if self.row_ptr.last().copied().unwrap_or_default() as usize != self.targets_global.len() {
            bail!("last row_ptr entry must equal local synapse count");
        }
        for &target in &self.targets_global {
            if target as usize >= self.global_neurons {
                bail!("target neuron id {target} is out of range");
            }
        }
        for &weight in &self.weights {
            if !weight.is_finite() {
                bail!("synapse weights must be finite");
            }
        }
        Ok(())
    }
}

pub fn generate_distributed_random_graph(
    partition: &NeuronPartition,
    fanout: usize,
    seed: u64,
) -> Result<DistributedCsrGraph> {
    if fanout == 0 {
        bail!("fanout must be greater than zero");
    }

    let local_synapses = partition
        .count
        .checked_mul(fanout)
        .ok_or_else(|| anyhow::anyhow!("local synapse count overflow"))?;

    if local_synapses > u32::MAX as usize {
        bail!("local synapse count exceeds u32 CSR index capacity");
    }

    let mut row_ptr = Vec::with_capacity(partition.count + 1);
    let mut targets_global = Vec::with_capacity(local_synapses);
    let mut weights = Vec::with_capacity(local_synapses);

    row_ptr.push(0);

    for local_source in 0..partition.count {
        let global_source = partition.start + local_source;

        let next = (local_source + 1)
            .checked_mul(fanout)
            .ok_or_else(|| anyhow::anyhow!("row pointer overflow"))?;
        row_ptr.push(next as u32);

        for edge_offset in 0..fanout {
            let key = edge_key(seed, global_source, edge_offset);
            let target = (splitmix64(key) % partition.global_neurons as u64) as u64;
            let weight_bits = splitmix64(key ^ 0xA076_1D64_78BD_642F) >> 40;
            let unit = (weight_bits as f32) * (1.0 / 16_777_216.0);
            let weight = 0.01 + 0.04 * unit;

            targets_global.push(target);
            weights.push(weight);
        }
    }

    DistributedCsrGraph::new(
        partition.global_neurons,
        partition.start,
        partition.count,
        fanout,
        row_ptr,
        targets_global,
        weights,
    )
}

fn edge_key(seed: u64, global_source: usize, edge_offset: usize) -> u64 {
    seed ^ (global_source as u64).wrapping_mul(0x9E37_79B1_85EB_CA87)
        ^ (edge_offset as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
}
