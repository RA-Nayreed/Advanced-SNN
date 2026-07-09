# Advanced-SNN

Advanced-SNN is a Rust-based spiking neural network simulator aimed at HPC-scale synthetic brain experiments on Roihu. It includes a CPU reference path, CUDA backends for single-GPU experiments, a biologically inspired brain-blob mode, STDP learning, snapshot export, and a Three.js inspection viewer for Roihu-generated snapshots.

## Current Status

- CPU reference simulator: deterministic event-driven baseline.
- Brain blob mode: clustered regions, excitatory/inhibitory neurons, and visualization snapshots.
- Learning mode: CPU STDP weight updates with learning metrics.
- Dense GPU backend: CUDA LIF neuron-update baseline.
- Event GPU backend: CUDA outgoing CSR traversal with `atomicAdd`.
- MPI code: distributed partition/probe scaffolding, not full distributed spike exchange.

## Build

CPU build:

```bash
cargo build --release
```

CUDA build:

```bash
cargo build --release --features cuda
```

MPI build:

```bash
cargo build --release --features mpi
```

The build script attempts to compile CUDA kernels to PTX with `nvcc`. If `nvcc` is unavailable, CUDA commands return a CUDA-unavailable error.

## Run Commands

CPU reference:

```bash
cargo run --release -- cpu --neurons 1000 --fanout 32 --steps 100 --seed 1
```

Brain blob snapshot run:

```bash
cargo run --release -- brain --neurons 20000 --fanout 64 --steps 500 --seed 7 \
  --snapshot-out brain_blob.ndjson --snapshot-every 5 --snapshot-neurons 5000 --snapshot-synapses 20000
```

Learning snapshot run:

```bash
cargo run --release -- learn --neurons 20000 --fanout 64 --steps 500 --seed 7 \
  --external-prob 0.006 --external-current 1.2 \
  --snapshot-out brain_learning.ndjson --snapshot-every 5 --snapshot-neurons 5000 --snapshot-synapses 20000
```

CUDA event backend:

```bash
cargo run --release --features cuda -- gpu-event --neurons 100000 --fanout 64 --steps 200 --seed 7
```

## Roihu Validation

Validation is Roihu-only and is intended after substantial updates. The repository does not track workstation validation fixtures snapshots.

Detailed instructions are in `docs/roihu.md`. The Slurm entry point is:

```bash
sbatch scripts/roihu/validate_big_update.sbatch
```

Before submitting, edit the script placeholders for the active Roihu account, partition, GPU request, wall time, and module names.

## Viewer

The static viewer in `viewer/` inspects `.ndjson` snapshots produced by Roihu jobs. It is not a validation target. Copy selected Roihu output snapshots to a workstation when visual inspection is needed, then load the file in the viewer.

The viewer supports:

- neuron and synapse playback
- selected-neuron biological microscope view
- region stimulation overlays
- sampled and aggregate scale views
- regional spike raster and throughput counters

## Simulation Model

Each neuron stores voltage, input current, and remaining refractory steps. On each timestep, the simulator clears input current, applies deterministic external input, processes outgoing synapses from active spikes, updates all neurons with the LIF rule, and swaps active spike buffers.

Default parameters:

- `decay = 0.95`
- `threshold = 1.0`
- `reset = 0.0`
- `refractory = 0`
- `external_prob = 0.001`
- `external_current = 1.2`

## Metrics

CPU metrics include neuron count, synapse count, steps, total spikes, mean spikes per step, final active spike count, elapsed seconds, processed synapse events, and synapse events per second.

Learning metrics include potentiation/depression counts and mean absolute weight.

GPU metrics include selected CUDA device, neuron/synapse counts, total spikes, kernel elapsed seconds when available, and event throughput.

## Current Limitations

- brain blob mode is biologically inspired, not an anatomically exact brain model
- STDP is CPU-only
- no full multi-node spike exchange yet
- no multi-GPU orchestration yet
- no synaptic delays yet
- CUDA event backend uses `atomicAdd`
- graph storage is explicit in memory

## Direction

- Roihu-scale profiling and throughput studies
- MPI rank-per-GPU execution
- distributed sparse spike/event exchange
- delay queues
- richer plasticity rules and GPU learning kernels
