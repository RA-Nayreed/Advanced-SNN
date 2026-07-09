use anyhow::Result;
use mpi::collective::SystemOperation;
use mpi::traits::*;

use crate::config::SimulationConfig;
use crate::distributed::graph::generate_distributed_random_graph;
use crate::distributed::partition::NeuronPartition;
use crate::distributed::runtime::{initialize_mpi, DistributedRuntime};

pub fn run_distributed_cpu_probe(config: &SimulationConfig) -> Result<()> {
    config.validate()?;

    let universe = initialize_mpi()?;
    let world = universe.world();

    let runtime = DistributedRuntime::from_world(&world);
    let partition = NeuronPartition::new(
        config.neurons,
        runtime.world_rank as usize,
        runtime.world_size as usize,
    )?;

    let graph = generate_distributed_random_graph(&partition, config.fanout, config.seed)?;

    runtime.print_startup_line();

    let local_neurons = partition.count as u64;
    let local_synapses = graph.local_synapses() as u64;

    let mut global_partitioned_neurons = 0_u64;
    let mut global_synapses = 0_u64;

    world.all_reduce_into(
        &local_neurons,
        &mut global_partitioned_neurons,
        SystemOperation::sum(),
    );

    world.all_reduce_into(
        &local_synapses,
        &mut global_synapses,
        SystemOperation::sum(),
    );

    if runtime.world_rank == 0 {
        println!("distributed_cpu_probe=true");
        println!("world_size={}", runtime.world_size);
        println!("neurons={}", config.neurons);
        println!("fanout={}", config.fanout);
        println!("steps={}", config.steps);
        println!("global_partitioned_neurons={global_partitioned_neurons}");
        println!("global_synapses={global_synapses}");
    }

    println!(
        "rank={} owns neurons [{}..{}) count={} local_synapses={}",
        runtime.world_rank,
        partition.start,
        partition.end,
        partition.count,
        graph.local_synapses()
    );

    Ok(())
}
