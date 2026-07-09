# Advanced-SNN

Advanced-SNN is a Rust-based spiking neural network simulator with a CPU reference path and CUDA backends for single-GPU experiments. The current model is leaky integrate-and-fire neurons connected by an outgoing CSR-style synapse graph.

## Current Status

- CPU reference simulator: implemented as a deterministic event-driven baseline.
- Brain blob CPU mode: implemented with clustered regions, excitatory/inhibitory neurons, and visualization snapshots.
- Dense GPU baseline: implemented for CUDA LIF neuron updates only.
- Event-driven GPU backend: implemented for single-GPU outgoing CSR traversal with `atomicAdd`.
- MPI and multi-GPU execution are intentionally not implemented yet.

## Build

CPU-only development works with a normal Cargo build:

```bash
cargo build
```

The build script attempts to compile CUDA kernels to PTX with `nvcc` and writes PTX files into Cargo's output directory. If `nvcc` is not available, the build continues and GPU commands return a clear CUDA-unavailable error.

The Rust CUDA launcher is behind the `cuda` feature so machines without CUDA can still build and test the CPU implementation:

```bash
cargo build --features cuda
```

## Run

CPU reference:

```bash
cargo run --release -- cpu --neurons 1000 --fanout 32 --steps 100 --seed 1
```

Dense GPU baseline:

```bash
cargo run --release --features cuda -- gpu-dense --neurons 10000 --steps 100 --seed 1
```

Event-driven GPU backend:

```bash
cargo run --release --features cuda -- gpu-event --neurons 100000 --fanout 64 --steps 100 --seed 1
```

Brain blob CPU mode:

```bash
cargo run --release -- brain --neurons 2000 --fanout 32 --steps 200 --seed 7 \
  --snapshot-out brain_blob.ndjson --snapshot-every 2 --snapshot-neurons 1500 --snapshot-synapses 4000
```

Learning brain mode:

```bash
cargo run --release -- learn --neurons 2000 --fanout 32 --steps 300 --seed 7 \
  --external-prob 0.006 --external-current 1.2 \
  --snapshot-out brain_learning.ndjson --snapshot-every 2 --snapshot-neurons 1500 --snapshot-synapses 4000
```

Script wrappers are available under `scripts/`.

CPU snapshot export for visualization:

```bash
cargo run --release -- cpu --neurons 512 --fanout 16 --steps 100 --seed 1 \
  --snapshot-out snapshots.ndjson --snapshot-every 1 --snapshot-neurons 512 --snapshot-synapses 2000
```

Snapshots are written as newline-delimited JSON. Each line contains sampled neuron positions, voltage, spike state, sampled synapses, and cumulative metrics for one timestep.

## Viewer

A static Three.js snapshot viewer lives in `viewer/`. Serve it locally and load `.ndjson` snapshots from the `brain`, `learn`, or `cpu` commands:

```bash
cd viewer
python3 -m http.server 5173
```

Then open `http://127.0.0.1:5173`. The viewer can also apply offline region stimulation overlays before live streaming is enabled.

## Simulation Model

Each neuron stores voltage, input current, and remaining refractory steps. On each timestep, the simulator clears input current, applies deterministic external input, processes outgoing synapses from active spikes, updates all neurons with the LIF rule, and swaps the active spike buffers.

Default parameters:

- `decay = 0.95`
- `threshold = 1.0`
- `reset = 0.0`
- `refractory = 0`
- `external_prob = 0.001`
- `external_current = 1.2`

## Metrics

CPU metrics include neuron count, synapse count, steps, total spikes, mean spikes per step, final active spike count, elapsed seconds, processed synapse events, and synapse events per second.

Dense GPU metrics include selected CUDA device, neurons, steps, total spikes, kernel elapsed seconds when available, and whether the GPU result matched the CPU LIF update validation.

Event GPU metrics include selected CUDA device, neurons, synapses, steps, total spikes, mean active spikes per step, processed synapse events, synapse events per second, elapsed seconds, and kernel elapsed seconds when available.

## Tests

Run all CPU and graceful-skip GPU tests:

```bash
cargo test
```

If CUDA is unavailable, GPU tests skip after receiving the CUDA-unavailable error. CPU tests still run.

## Current Limitations

- brain blob mode is biologically inspired, not an anatomically exact brain model
- no MPI yet
- no multi-GPU execution yet
- no synaptic delays yet
- STDP is currently CPU-only and intended for visual/experimental runs
- event backend uses `atomicAdd`
- graph storage is explicit in memory

## Future Direction

- MPI rank per GPU
- distributed sparse spike and event exchange
- delay queues
- richer plasticity rules and GPU learning kernels
- profiling and scaling studies
