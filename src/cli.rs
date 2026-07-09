use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};

use crate::brain::{generate_brain_blob, BrainBlobConfig};
use crate::config::{DenseSimulationConfig, SimulationConfig};
use crate::cpu::reference::run_reference;
use crate::gpu;
use crate::graph::random::generate_random_graph;
use crate::learning::{run_stdp_learning, StdpConfig};
use crate::neuron::lif::LifParams;
use crate::snapshot::{run_reference_with_snapshots, SnapshotOptions};

#[derive(Debug, Parser)]
#[command(name = "advanced-snn")]
#[command(about = "Rust and CUDA spiking neural network simulator")]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Cpu(CpuArgs),
    Brain(CpuArgs),
    Learn(CpuArgs),
    GpuDense(DenseArgs),
    GpuEvent(GraphArgs),

    #[cfg(feature = "mpi")]
    DistributedCpu(GraphArgs),
}

#[derive(Debug, Args)]
struct CpuArgs {
    #[command(flatten)]
    graph: GraphArgs,
    #[command(flatten)]
    snapshot: SnapshotArgs,
}

#[derive(Debug, Args)]
struct GraphArgs {
    #[arg(long)]
    neurons: usize,
    #[arg(long)]
    fanout: usize,
    #[arg(long)]
    steps: usize,
    #[arg(long)]
    seed: u64,
    #[command(flatten)]
    lif: LifArgs,
}

#[derive(Debug, Args)]
struct DenseArgs {
    #[arg(long)]
    neurons: usize,
    #[arg(long)]
    steps: usize,
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    fanout: Option<usize>,
    #[command(flatten)]
    lif: LifArgs,
}

#[derive(Clone, Debug, Args)]
struct SnapshotArgs {
    #[arg(long)]
    snapshot_out: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    snapshot_every: usize,
    #[arg(long, default_value_t = 1000)]
    snapshot_neurons: usize,
    #[arg(long, default_value_t = 2000)]
    snapshot_synapses: usize,
}

#[derive(Clone, Copy, Debug, Args)]
struct LifArgs {
    #[arg(long, default_value_t = 0.95)]
    decay: f32,
    #[arg(long, default_value_t = 1.0)]
    threshold: f32,
    #[arg(long, default_value_t = 0.0)]
    reset: f32,
    #[arg(long, default_value_t = 0)]
    refractory: u16,
    #[arg(long, default_value_t = 0.001)]
    external_prob: f32,
    #[arg(long, default_value_t = 1.2)]
    external_current: f32,
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Cpu(args) => run_cpu(args),
        Command::Brain(args) => run_brain(args),
        Command::Learn(args) => run_learn(args),
        Command::GpuDense(args) => run_gpu_dense(args),
        Command::GpuEvent(args) => run_gpu_event(args),

        #[cfg(feature = "mpi")]
        Command::DistributedCpu(args) => run_distributed_cpu(args),
    }
}

fn run_cpu(args: CpuArgs) -> Result<()> {
    let config = args.graph.to_simulation_config();
    config.validate()?;
    let graph = generate_random_graph(config.neurons, config.fanout, config.seed)?;
    let result = if let Some(options) = args.snapshot.to_options() {
        run_reference_with_snapshots(&config, &graph, options)?
    } else {
        run_reference(&config, &graph)?
    };
    print!("{}", result.metrics);
    Ok(())
}


fn run_brain(args: CpuArgs) -> Result<()> {
    let config = args.graph.to_simulation_config();
    config.validate()?;
    let brain = generate_brain_blob(&BrainBlobConfig::new(
        config.neurons,
        config.fanout,
        config.seed,
    ))?;
    let result = if let Some(options) = args
        .snapshot
        .to_options()
        .map(|options| options.with_layout(brain.snapshot_layout()))
    {
        run_reference_with_snapshots(&config, &brain.graph, options)?
    } else {
        run_reference(&config, &brain.graph)?
    };
    print!("{}", result.metrics);
    Ok(())
}

fn run_learn(args: CpuArgs) -> Result<()> {
    let config = args.graph.to_simulation_config();
    config.validate()?;
    let mut brain = generate_brain_blob(&BrainBlobConfig::new(
        config.neurons,
        config.fanout,
        config.seed,
    ))?;
    let layout = brain.snapshot_layout();
    let snapshots = args
        .snapshot
        .to_options()
        .map(|options| options.with_layout(layout.clone()));
    let result = run_stdp_learning(
        &config,
        &mut brain.graph,
        layout,
        StdpConfig::default(),
        snapshots,
    )?;
    print!("{result}");
    Ok(())
}

fn run_gpu_dense(args: DenseArgs) -> Result<()> {
    let _ignored_fanout = args.fanout;
    let config = args.to_dense_config();
    config.validate()?;
    let metrics = gpu::dense::run_dense(&config)?;
    print!("{metrics}");
    Ok(())
}

fn run_gpu_event(args: GraphArgs) -> Result<()> {
    let config = args.to_simulation_config();
    config.validate()?;
    let graph = generate_random_graph(config.neurons, config.fanout, config.seed)?;
    let metrics = gpu::event::run_event(&config, &graph)?;
    print!("{metrics}");
    Ok(())
}

#[cfg(feature = "mpi")]
fn run_distributed_cpu(args: GraphArgs) -> Result<()> {
    let config = args.to_simulation_config();
    config.validate()?;
    crate::distributed::cpu::run_distributed_cpu_smoke(&config)
}

impl GraphArgs {
    fn to_simulation_config(&self) -> SimulationConfig {
        SimulationConfig {
            neurons: self.neurons,
            fanout: self.fanout,
            steps: self.steps,
            seed: self.seed,
            lif: self.lif.to_lif_params(),
            external_prob: self.lif.external_prob,
            external_current: self.lif.external_current,
        }
    }
}

impl DenseArgs {
    fn to_dense_config(&self) -> DenseSimulationConfig {
        DenseSimulationConfig {
            neurons: self.neurons,
            steps: self.steps,
            seed: self.seed,
            lif: self.lif.to_lif_params(),
            external_prob: self.lif.external_prob,
            external_current: self.lif.external_current,
        }
    }
}

impl SnapshotArgs {
    fn to_options(&self) -> Option<SnapshotOptions> {
        self.snapshot_out.clone().map(|output| SnapshotOptions {
            output,
            every: self.snapshot_every,
            neuron_sample: self.snapshot_neurons,
            synapse_sample: self.snapshot_synapses,
            layout: None,
        })
    }
}

impl LifArgs {
    fn to_lif_params(self) -> LifParams {
        LifParams {
            decay: self.decay,
            threshold: self.threshold,
            reset: self.reset,
            refractory: self.refractory,
        }
    }
}
