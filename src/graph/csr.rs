use std::ops::Range;

use anyhow::{bail, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct CsrGraph {
    neurons: usize,
    row_ptr: Vec<u32>,
    targets: Vec<u32>,
    weights: Vec<f32>,
}

impl CsrGraph {
    pub fn new(
        neurons: usize,
        row_ptr: Vec<u32>,
        targets: Vec<u32>,
        weights: Vec<f32>,
    ) -> Result<Self> {
        let graph = Self {
            neurons,
            row_ptr,
            targets,
            weights,
        };
        graph.validate()?;
        Ok(graph)
    }

    pub fn neurons(&self) -> usize {
        self.neurons
    }

    pub fn synapses(&self) -> usize {
        self.targets.len()
    }

    pub fn row_ptr(&self) -> &[u32] {
        &self.row_ptr
    }

    pub fn targets(&self) -> &[u32] {
        &self.targets
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn outgoing_range(&self, neuron_id: usize) -> Range<usize> {
        let start = self.row_ptr[neuron_id] as usize;
        let end = self.row_ptr[neuron_id + 1] as usize;
        start..end
    }

    pub fn validate(&self) -> Result<()> {
        if self.neurons == 0 {
            bail!("graph must contain at least one neuron");
        }
        if self.row_ptr.len() != self.neurons + 1 {
            bail!("row_ptr length must be neurons + 1");
        }
        if self.targets.len() != self.weights.len() {
            bail!("targets and weights lengths must match");
        }
        if self.row_ptr.first().copied() != Some(0) {
            bail!("row_ptr must start at zero");
        }
        for pair in self.row_ptr.windows(2) {
            if pair[0] > pair[1] {
                bail!("row_ptr must be monotonically nondecreasing");
            }
        }
        if self.row_ptr.last().copied().unwrap_or_default() as usize != self.targets.len() {
            bail!("last row_ptr entry must equal synapse count");
        }
        for &target in &self.targets {
            if target as usize >= self.neurons {
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
