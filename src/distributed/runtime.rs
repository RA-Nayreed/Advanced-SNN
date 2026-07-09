use std::env;

use anyhow::{anyhow, Result};
use mpi::traits::*;

#[derive(Clone, Debug)]
pub struct DistributedRuntime {
    pub world_rank: i32,
    pub world_size: i32,
    pub local_rank: i32,
    pub local_size: i32,
    pub processor_name: String,
}

impl DistributedRuntime {
    pub fn from_world<C>(world: &C) -> Self
    where
        C: Communicator,
    {
        let world_rank = world.rank();
        let world_size = world.size();

        let local_rank = read_i32_env(&[
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MV2_COMM_WORLD_LOCAL_RANK",
            "MPI_LOCALRANKID",
            "SLURM_LOCALID",
        ])
        .unwrap_or(world_rank);

        let local_size = read_i32_env(&["OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE"])
            .unwrap_or(1);

        let processor_name =
            mpi::environment::processor_name().unwrap_or_else(|_| String::from("unknown-host"));

        Self {
            world_rank,
            world_size,
            local_rank,
            local_size,
            processor_name,
        }
    }

    pub fn print_startup_line(&self) {
        println!(
            "rank={}/{} local_rank={}/{} host={}",
            self.world_rank,
            self.world_size,
            self.local_rank,
            self.local_size,
            self.processor_name
        );
    }
}

pub fn initialize_mpi() -> Result<mpi::environment::Universe> {
    mpi::initialize().ok_or_else(|| anyhow!("MPI was already initialized before this command"))
}

fn read_i32_env(names: &[&str]) -> Option<i32> {
    names
        .iter()
        .find_map(|name| env::var(name).ok()?.parse::<i32>().ok())
}
