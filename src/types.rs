pub type NeuronId = u32;
pub type SynapseIndex = u32;

const SPLITMIX_INCREMENT: u64 = 0x9e37_79b9_7f4a_7c15;
const SPLITMIX_MUL_1: u64 = 0xbf58_476d_1ce4_e5b9;
const SPLITMIX_MUL_2: u64 = 0x94d0_49bb_1331_11eb;

pub fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(SPLITMIX_INCREMENT);
    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(SPLITMIX_MUL_1);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(SPLITMIX_MUL_2);
    mixed ^ (mixed >> 31)
}

pub fn deterministic_unit_f32(seed: u64, step: usize, neuron_id: usize) -> f32 {
    let key = seed
        ^ (step as u64).wrapping_mul(SPLITMIX_INCREMENT)
        ^ (neuron_id as u64).wrapping_mul(SPLITMIX_MUL_1);
    let bits = splitmix64(key) >> 40;
    (bits as f32) * (1.0 / 16_777_216.0)
}

pub fn external_input_applies(seed: u64, step: usize, neuron_id: usize, probability: f32) -> bool {
    if probability <= 0.0 {
        false
    } else if probability >= 1.0 {
        true
    } else {
        deterministic_unit_f32(seed, step, neuron_id) < probability
    }
}
