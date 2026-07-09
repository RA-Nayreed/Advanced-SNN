use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LifParams {
    pub decay: f32,
    pub threshold: f32,
    pub reset: f32,
    pub refractory: u16,
}

impl Default for LifParams {
    fn default() -> Self {
        Self {
            decay: 0.95,
            threshold: 1.0,
            reset: 0.0,
            refractory: 0,
        }
    }
}

impl LifParams {
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.decay) {
            bail!("decay must be in [0, 1]");
        }
        if !self.threshold.is_finite() || !self.reset.is_finite() {
            bail!("threshold and reset must be finite");
        }
        if self.threshold <= self.reset {
            bail!("threshold must be greater than reset");
        }
        Ok(())
    }
}

pub fn update_lif_neuron(
    voltage: &mut f32,
    input_current: f32,
    refractory_left: &mut u16,
    params: LifParams,
) -> bool {
    if *refractory_left > 0 {
        *refractory_left -= 1;
        return false;
    }

    *voltage = *voltage * params.decay + input_current;
    if *voltage >= params.threshold {
        *voltage = params.reset;
        *refractory_left = params.refractory;
        true
    } else {
        false
    }
}
