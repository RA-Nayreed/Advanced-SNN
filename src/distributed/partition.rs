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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partitions_cover_all_neurons() {
        let neurons = 10;
        let size = 4;

        let parts: Vec<_> = (0..size)
            .map(|rank| NeuronPartition::new(neurons, rank, size).unwrap())
            .collect();

        assert_eq!(parts.first().unwrap().start, 0);
        assert_eq!(parts.last().unwrap().end, neurons);

        for pair in parts.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
        }

        let total: usize = parts.iter().map(|p| p.count).sum();
        assert_eq!(total, neurons);
    }

    #[test]
    fn owner_lookup_works_when_not_evenly_divisible() {
        let part = NeuronPartition::new(10, 0, 4).unwrap();

        let owners: Vec<_> = (0..10).map(|id| part.owner_of(id).unwrap()).collect();

        assert_eq!(owners, vec![0, 0, 1, 1, 1, 2, 2, 3, 3, 3]);
    }

    #[test]
    fn local_and_global_index_conversion_works() {
        let part = NeuronPartition::new(10, 2, 4).unwrap();

        assert_eq!(part.start, 5);
        assert_eq!(part.end, 7);
        assert_eq!(part.count, 2);

        assert_eq!(part.local_index(4), None);
        assert_eq!(part.local_index(5), Some(0));
        assert_eq!(part.local_index(6), Some(1));
        assert_eq!(part.local_index(7), None);

        assert_eq!(part.global_index(0).unwrap(), 5);
        assert_eq!(part.global_index(1).unwrap(), 6);
        assert!(part.global_index(2).is_err());
    }

    #[test]
    fn counts_and_starts_are_consistent() {
        let starts = NeuronPartition::starts_per_rank(10, 4).unwrap();
        let counts = NeuronPartition::counts_per_rank(10, 4).unwrap();

        assert_eq!(starts, vec![0, 2, 5, 7]);
        assert_eq!(counts, vec![2, 3, 2, 3]);
        assert_eq!(counts.iter().sum::<usize>(), 10);
    }
}
