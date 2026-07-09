use advanced_snn::config::SimulationConfig;
use advanced_snn::cpu::reference::run_reference;
use advanced_snn::gpu::buffers::DenseInputCase;
use advanced_snn::gpu::cuda::is_cuda_unavailable;
use advanced_snn::gpu::dense::validate_dense_single_step;
use advanced_snn::gpu::event::run_event;
use advanced_snn::graph::csr::CsrGraph;
use advanced_snn::graph::random::generate_random_graph;
use advanced_snn::neuron::lif::LifParams;
use anyhow::Result;

#[test]
fn tiny_cpu_gpu_lif_match_or_skip() -> Result<()> {
    let case = DenseInputCase {
        voltage: vec![0.0, 0.4, 0.9],
        input_current: vec![0.5, 0.7, 0.2],
        refractory_left: vec![0, 0, 0],
        lif: LifParams::default(),
    };

    assert_gpu_match_or_skip(validate_dense_single_step(&case))
}

#[test]
fn gpu_voltage_reset_matches_cpu_or_skip() -> Result<()> {
    let case = DenseInputCase {
        voltage: vec![0.95],
        input_current: vec![0.1],
        refractory_left: vec![0],
        lif: LifParams {
            reset: -0.25,
            ..LifParams::default()
        },
    };

    assert_gpu_match_or_skip(validate_dense_single_step(&case))
}

#[test]
fn gpu_refractory_behavior_matches_cpu_or_skip() -> Result<()> {
    let case = DenseInputCase {
        voltage: vec![2.0],
        input_current: vec![4.0],
        refractory_left: vec![1],
        lif: LifParams::default(),
    };

    assert_gpu_match_or_skip(validate_dense_single_step(&case))
}

#[test]
fn tiny_event_graph_cpu_gpu_match_or_skip() -> Result<()> {
    let graph = CsrGraph::new(2, vec![0, 1, 1], vec![1], vec![1.2]).unwrap();
    let mut config = SimulationConfig::new(2, 1, 4, 7);
    config.external_prob = 1.0;
    config.external_current = 1.2;

    let cpu = run_reference(&config, &graph)?;
    match run_event(&config, &graph) {
        Ok(gpu) => {
            assert_eq!(cpu.metrics.spikes_per_step, gpu.spikes_per_step);
            Ok(())
        }
        Err(error) if is_cuda_unavailable(&error) => {
            eprintln!("skipping CUDA event test: {error}");
            Ok(())
        }
        Err(error) => Err(error),
    }
}

#[test]
fn deterministic_random_event_graph_cpu_gpu_match_or_skip() -> Result<()> {
    let mut config = SimulationConfig::new(8, 2, 6, 5);
    config.external_prob = 0.4;
    config.external_current = 1.2;
    let graph = generate_random_graph(config.neurons, config.fanout, config.seed)?;

    let cpu = run_reference(&config, &graph)?;
    match run_event(&config, &graph) {
        Ok(gpu) => {
            assert_eq!(cpu.metrics.spikes_per_step, gpu.spikes_per_step);
            Ok(())
        }
        Err(error) if is_cuda_unavailable(&error) => {
            eprintln!("skipping CUDA event test: {error}");
            Ok(())
        }
        Err(error) => Err(error),
    }
}

fn assert_gpu_match_or_skip(result: Result<bool>) -> Result<()> {
    match result {
        Ok(matches) => {
            assert!(matches);
            Ok(())
        }
        Err(error) if is_cuda_unavailable(&error) => {
            eprintln!("skipping CUDA dense test: {error}");
            Ok(())
        }
        Err(error) => Err(error),
    }
}
