use anyhow::{bail, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct NeuronState {
    pub voltage: Vec<f32>,
    pub input_current: Vec<f32>,
    pub refractory_left: Vec<u16>,
}

impl NeuronState {
    pub fn new(neurons: usize) -> Self {
        Self {
            voltage: vec![0.0; neurons],
            input_current: vec![0.0; neurons],
            refractory_left: vec![0; neurons],
        }
    }

    pub fn len(&self) -> usize {
        self.voltage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.voltage.is_empty()
    }

    pub fn clear_input(&mut self) {
        self.input_current.fill(0.0);
    }

    pub fn validate_len(&self, neurons: usize) -> Result<()> {
        if self.voltage.len() != neurons
            || self.input_current.len() != neurons
            || self.refractory_left.len() != neurons
        {
            bail!("neuron state buffers must match neuron count");
        }
        Ok(())
    }
}
