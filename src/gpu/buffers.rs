use crate::neuron::lif::{update_lif_neuron, LifParams};

#[derive(Clone, Debug)]
pub struct DenseInputCase {
    pub voltage: Vec<f32>,
    pub input_current: Vec<f32>,
    pub refractory_left: Vec<u16>,
    pub lif: LifParams,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DenseStepOutput {
    pub voltage: Vec<f32>,
    pub refractory_left: Vec<u16>,
    pub spike_flags: Vec<u8>,
}

impl DenseInputCase {
    pub fn len(&self) -> usize {
        self.voltage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.voltage.is_empty()
    }

    pub fn cpu_step(&self) -> DenseStepOutput {
        let mut voltage = self.voltage.clone();
        let mut refractory_left = self.refractory_left.clone();
        let mut spike_flags = vec![0_u8; self.len()];

        for neuron_id in 0..self.len() {
            let spiked = update_lif_neuron(
                &mut voltage[neuron_id],
                self.input_current[neuron_id],
                &mut refractory_left[neuron_id],
                self.lif,
            );
            spike_flags[neuron_id] = u8::from(spiked);
        }

        DenseStepOutput {
            voltage,
            refractory_left,
            spike_flags,
        }
    }
}
