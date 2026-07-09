use anyhow::Result;
use clap::{Args, Parser, Subcommand};

use crate::config::{DenseSimulationConfig, SimulationConfig};
use crate::cpu::reference::run_reference;
use crate::gpu;
use crate::graph::random::generate_random_graph;
use crate::neuron::lif::LifParams;

#[derive(Debug, Parser)]
#[command(name = "advanced-snn")]
#[command(about = "Rust and CUDA spiking neural network simulator")]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Cpu(GraphArgs),
    GpuDense(DenseArgs),
    GpuEvent(GraphArgs),

    #[cfg(feature = "mpi")]
    DistributedCpu(GraphArgs),
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
        Command::GpuDense(args) => run_gpu_dense(args),
        Command::GpuEvent(args) => run_gpu_event(args),

        #[cfg(feature = "mpi")]
        Command::DistributedCpu(args) => run_distributed_cpu(args),
    }
}

fn run_cpu(args: GraphArgs) -> Result<()> {
    let config = args.to_simulation_config();
    config.validate()?;
    let graph = generate_random_graph(config.neurons, config.fanout, config.seed)?;
    let result = run_reference(&config, &graph)?;
    print!("{}", result.metrics);
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
