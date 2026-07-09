use anyhow::{bail, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NeuronPartition {
    pub global_neurons: usize,
    pub rank: usize,
    pub size: usize,
    pub start: usize,
    pub end: usize,
    pub count: usize,
}

impl NeuronPartition {
    pub fn new(global_neurons: usize, rank: usize, size: usize) -> Result<Self> {
        if global_neurons == 0 {
            bail!("global_neurons must be greater than zero");
        }
        if size == 0 {
            bail!("world size must be greater than zero");
        }
        if rank >= size {
            bail!("rank must be smaller than world size");
        }

        let start = Self::start_for_rank(global_neurons, rank, size)?;
        let end = Self::start_for_rank(global_neurons, rank + 1, size)?;

        Ok(Self {
            global_neurons,
            rank,
            size,
            start,
            end,
            count: end - start,
        })
    }

    pub fn owner_of(&self, global_neuron_id: usize) -> Result<usize> {
        owner_of(global_neuron_id, self.global_neurons, self.size)
    }

    pub fn local_index(&self, global_neuron_id: usize) -> Option<usize> {
        if global_neuron_id >= self.start && global_neuron_id < self.end {
            Some(global_neuron_id - self.start)
        } else {
            None
        }
    }

    pub fn global_index(&self, local_index: usize) -> Result<usize> {
        if local_index >= self.count {
            bail!("local neuron id {local_index} is out of range");
        }
        Ok(self.start + local_index)
    }

    pub fn starts_per_rank(global_neurons: usize, size: usize) -> Result<Vec<usize>> {
        if global_neurons == 0 {
            bail!("global_neurons must be greater than zero");
        }
        if size == 0 {
            bail!("world size must be greater than zero");
        }

        (0..size)
            .map(|rank| Self::start_for_rank(global_neurons, rank, size))
            .collect()
    }

    pub fn counts_per_rank(global_neurons: usize, size: usize) -> Result<Vec<usize>> {
        if global_neurons == 0 {
            bail!("global_neurons must be greater than zero");
        }
        if size == 0 {
            bail!("world size must be greater than zero");
        }

        (0..size)
            .map(|rank| {
                let start = Self::start_for_rank(global_neurons, rank, size)?;
                let end = Self::start_for_rank(global_neurons, rank + 1, size)?;
                Ok(end - start)
            })
            .collect()
    }

    fn start_for_rank(global_neurons: usize, rank: usize, size: usize) -> Result<usize> {
        global_neurons
            .checked_mul(rank)
            .ok_or_else(|| anyhow::anyhow!("partition start overflow"))
            .map(|value| value / size)
    }
}

pub fn owner_of(global_neuron_id: usize, global_neurons: usize, size: usize) -> Result<usize> {
    if global_neurons == 0 {
        bail!("global_neurons must be greater than zero");
    }
    if size == 0 {
        bail!("world size must be greater than zero");
    }
    if global_neuron_id >= global_neurons {
        bail!("global neuron id {global_neuron_id} is out of range");
    }

    let mut lo = 0usize;
    let mut hi = size;

    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        let start = global_neurons
            .checked_mul(mid)
            .ok_or_else(|| anyhow::anyhow!("owner lookup overflow"))?
            / size;

        if start <= global_neuron_id {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    Ok(lo)
}
