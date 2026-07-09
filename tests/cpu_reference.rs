use advanced_snn::config::SimulationConfig;
use advanced_snn::cpu::reference::{run_reference, run_reference_with_initial_spikes};
use advanced_snn::graph::csr::CsrGraph;
use advanced_snn::graph::random::generate_random_graph;
use advanced_snn::neuron::lif::{update_lif_neuron, LifParams};

#[test]
fn neuron_spikes_when_voltage_crosses_threshold() {
    let params = LifParams::default();
    let mut voltage = 0.5;
    let mut refractory_left = 0;

    let spiked = update_lif_neuron(&mut voltage, 0.6, &mut refractory_left, params);

    assert!(spiked);
}

#[test]
fn voltage_resets_after_spike() {
    let params = LifParams {
        reset: -0.1,
        ..LifParams::default()
    };
    let mut voltage = 0.9;
    let mut refractory_left = 0;

    let spiked = update_lif_neuron(&mut voltage, 0.2, &mut refractory_left, params);

    assert!(spiked);
    assert_eq!(voltage, params.reset);
}

#[test]
fn refractory_period_prevents_immediate_respike() {
    let params = LifParams::default();
    let mut voltage = 2.0;
    let mut refractory_left = 1;

    let spiked = update_lif_neuron(&mut voltage, 10.0, &mut refractory_left, params);

    assert!(!spiked);
    assert_eq!(refractory_left, 0);
    assert_eq!(voltage, 2.0);
}

#[test]
fn tiny_hand_built_graph_has_deterministic_spike_counts() {
    let graph = CsrGraph::new(2, vec![0, 1, 1], vec![1], vec![1.2]).unwrap();
    let mut config = SimulationConfig::new(2, 1, 2, 1);
    config.external_prob = 0.0;

    let first = run_reference_with_initial_spikes(&config, &graph, &[0]).unwrap();
    let second = run_reference_with_initial_spikes(&config, &graph, &[0]).unwrap();

    assert_eq!(first.metrics.total_spikes, 1);
    assert_eq!(first.metrics.spikes_per_step, vec![1, 0]);
    assert!(first.metrics.deterministic_eq(&second.metrics));
}

#[test]
fn same_config_twice_has_identical_deterministic_metrics() {
    let mut config = SimulationConfig::new(16, 4, 12, 99);
    config.external_prob = 0.2;
    config.external_current = 1.2;
    let graph = generate_random_graph(config.neurons, config.fanout, config.seed).unwrap();

    let first = run_reference(&config, &graph).unwrap();
    let second = run_reference(&config, &graph).unwrap();

    assert!(first.metrics.deterministic_eq(&second.metrics));
}

#[test]
fn csr_graph_row_pointers_are_valid() {
    let graph = CsrGraph::new(3, vec![0, 2, 3, 3], vec![1, 2, 0], vec![0.1, 0.2, 0.3])
        .unwrap();

    assert_eq!(graph.outgoing_range(0), 0..2);
    assert_eq!(graph.outgoing_range(1), 2..3);
    assert_eq!(graph.outgoing_range(2), 3..3);
    assert!(graph.validate().is_ok());
}

#[test]
fn generated_graph_has_expected_number_of_synapses() {
    let graph = generate_random_graph(10, 7, 123).unwrap();

    assert_eq!(graph.synapses(), 70);
    assert_eq!(graph.row_ptr().len(), 11);
    assert_eq!(graph.row_ptr().last().copied(), Some(70));
}
