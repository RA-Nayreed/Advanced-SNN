use anyhow::{bail, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::graph::csr::CsrGraph;

pub fn generate_random_graph(neurons: usize, fanout: usize, seed: u64) -> Result<CsrGraph> {
    if neurons == 0 {
        bail!("neurons must be greater than zero");
    }
    if fanout == 0 {
        bail!("fanout must be greater than zero");
    }

    let synapses = neurons
        .checked_mul(fanout)
        .ok_or_else(|| anyhow::anyhow!("synapse count overflow"))?;
    if synapses > u32::MAX as usize {
        bail!("synapse count exceeds u32 CSR index capacity");
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut row_ptr = Vec::with_capacity(neurons + 1);
    let mut targets = Vec::with_capacity(synapses);
    let mut weights = Vec::with_capacity(synapses);

    row_ptr.push(0);
    for source in 0..neurons {
        let next = (source + 1)
            .checked_mul(fanout)
            .ok_or_else(|| anyhow::anyhow!("row pointer overflow"))?;
        row_ptr.push(next as u32);

        for _ in 0..fanout {
            targets.push(rng.gen_range(0..neurons) as u32);
            weights.push(rng.gen_range(0.01_f32..0.05_f32));
        }
    }

    CsrGraph::new(neurons, row_ptr, targets, weights)
}
