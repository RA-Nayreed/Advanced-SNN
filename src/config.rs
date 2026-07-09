use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::neuron::lif::LifParams;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub neurons: usize,
    pub fanout: usize,
    pub steps: usize,
    pub seed: u64,
    #[serde(flatten)]
    pub lif: LifParams,
    pub external_prob: f32,
    pub external_current: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DenseSimulationConfig {
    pub neurons: usize,
    pub steps: usize,
    pub seed: u64,
    #[serde(flatten)]
    pub lif: LifParams,
    pub external_prob: f32,
    pub external_current: f32,
}

impl SimulationConfig {
    pub fn new(neurons: usize, fanout: usize, steps: usize, seed: u64) -> Self {
        Self {
            neurons,
            fanout,
            steps,
            seed,
            lif: LifParams::default(),
            external_prob: 0.001,
            external_current: 1.2,
        }
    }

    pub fn validate(&self) -> Result<()> {
        validate_common(
            self.neurons,
            self.steps,
            &self.lif,
            self.external_prob,
            self.external_current,
        )?;
        if self.fanout == 0 {
            bail!("fanout must be greater than zero");
        }
        Ok(())
    }
}

impl DenseSimulationConfig {
    pub fn new(neurons: usize, steps: usize, seed: u64) -> Self {
        Self {
            neurons,
            steps,
            seed,
            lif: LifParams::default(),
            external_prob: 0.001,
            external_current: 1.2,
        }
    }

    pub fn validate(&self) -> Result<()> {
        validate_common(
            self.neurons,
            self.steps,
            &self.lif,
            self.external_prob,
            self.external_current,
        )
    }
}

fn validate_common(
    neurons: usize,
    steps: usize,
    lif: &LifParams,
    external_prob: f32,
    external_current: f32,
) -> Result<()> {
    if neurons == 0 {
        bail!("neurons must be greater than zero");
    }
    if steps == 0 {
        bail!("steps must be greater than zero");
    }
    if !(0.0..=1.0).contains(&external_prob) {
        bail!("external_prob must be in [0, 1]");
    }
    if !external_current.is_finite() {
        bail!("external_current must be finite");
    }
    lif.validate()
}
