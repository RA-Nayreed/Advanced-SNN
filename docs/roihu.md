# Roihu HPC Validation

This repository treats Roihu as the validation environment. Do not add workstation validation checks. Validation is run on Roihu after a substantial simulator, CUDA, distributed, snapshot, or viewer-data change.

These Slurm profiles follow the CSC Roihu documentation checked on 2026-07-09.

## Roihu Resource Model

- Build and submit CPU jobs from `roihu-cpu.csc.fi`; CPU nodes are x86.
- Build and submit GPU jobs from `roihu-gpu.csc.fi`; GPU nodes are ARM/GH200.
- A Roihu billing project is mandatory. Pass it with `--account=<project>` or use the submit wrapper below.
- CPU validation uses `small`, a shared-node CPU partition with allocation type `R`.
- GPU validation uses `gpumedium`, a GPU allocation type `G` partition for one GPU node and up to 4 GH200 GPUs per job.
- Each reserved GH200 GPU maps to 72 CPU cores. A full 4-GPU node therefore uses `--ntasks-per-node=4 --cpus-per-task=72`.

## Validation Scripts

| Mode | Script | Partition | Nodes | Tasks and cores | GPU request | Purpose |
| --- | --- | --- | --- | --- | --- | --- |
| CPU | `scripts/roihu/validate_big_update.sbatch` | `small` | 1 | `--ntasks=1 --cpus-per-task=16` | none | CPU build, required MPI feature build, brain snapshot, learning snapshot, distributed MPI probe |
| Single GPU | `scripts/roihu/validate_gpu_single.sbatch` | `gpumedium` | 1 | `--ntasks-per-node=1 --cpus-per-task=72` | `--gres=gpu:gh200:1` | CUDA build and one GH200 event-backend run |
| Multi GPU smoke | `scripts/roihu/validate_gpu_multi.sbatch` | `gpumedium` | 1 | `--ntasks-per-node=4 --cpus-per-task=72` | `--gres=gpu:gh200:4` | Four independent rank-local GPU event runs, one per reserved GH200 |

The multi-GPU script validates Slurm allocation, CPU-core mapping, thread settings, and rank-to-GPU visibility. It does not implement cross-GPU spike exchange; the simulator still lists full multi-GPU orchestration as future work.

## Submitting Jobs

Use the wrapper so the project account stays outside version-controlled `#SBATCH` headers:

```bash
scripts/roihu/submit_validation.sh --account <project> cpu
scripts/roihu/submit_validation.sh --account <project> gpu
scripts/roihu/submit_validation.sh --account <project> multi-gpu
```

You can also set the project once:

```bash
export ROIHU_ACCOUNT=<project>
scripts/roihu/submit_validation.sh cpu
```

Direct `sbatch` is also valid:

```bash
sbatch --account=<project> scripts/roihu/validate_big_update.sbatch
sbatch --account=<project> scripts/roihu/validate_gpu_single.sbatch
sbatch --account=<project> scripts/roihu/validate_gpu_multi.sbatch
```

## Modules

The scripts do not hard-code Rust, CUDA, GCC, MPI, or LLVM/Clang module names. CPU validation requires a working MPI build environment because distributed MPI support is a core project target. Set `ROIHU_MODULES` to the current Roihu module names for your environment:

```bash
export ROIHU_MODULES="gcc/15.2.0 openmpi/5.0.10 <llvm-or-clang-module>"
scripts/roihu/submit_validation.sh --account <project> cpu

export ROIHU_MODULES="rust gcc cuda"
scripts/roihu/submit_validation.sh --account <project> gpu
```

If `ROIHU_MODULES` is unset, the scripts use whatever `cargo`, `nvcc`, `srun`, `mpicc`, and `mpirun` are already available in the batch shell. CPU validation fails if the MPI feature cannot build. The Rust `mpi-sys` crate uses bindgen, so Roihu must also expose `libclang` through an LLVM/Clang module or `LIBCLANG_PATH`.

## Thread and Core Settings

Each script derives thread counts from `SLURM_CPUS_PER_TASK`:

- `OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}`
- `RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK}`
- `CARGO_BUILD_JOBS=${SLURM_CPUS_PER_TASK}`
- `OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}`
- `MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}`

The scripts also set `OMP_PLACES=cores` and `OMP_PROC_BIND=spread`, matching CSC's OpenMP placement guidance.

## Multi-GPU Binding

`validate_gpu_multi.sbatch` reserves all 4 GH200 GPUs on one node and starts four Slurm tasks. `scripts/roihu/run_visible_gpu_rank.sh` maps `SLURM_LOCALID` to one entry from `CUDA_VISIBLE_DEVICES`, then runs the CUDA event backend. Each rank writes `gpu_event_rank_<rank>.log` under the job output directory.

For future multi-node GPU work, switch to `gpularge`, keep `--ntasks-per-node=4 --cpus-per-task=72 --gres=gpu:gh200:4`, and set `--nodes=<N>`. That should only be enabled after the simulator has a real cross-rank GPU exchange path.

## Big-Update Validation Flow

From the appropriate Roihu login node:

```bash
git clone git@github.com:RA-Nayreed/Advanced-SNN.git
cd Advanced-SNN
scripts/roihu/submit_validation.sh --account <project> cpu
```

Monitor with:

```bash
squeue -u "$USER"
```

Inspect output after completion:

```bash
ls -lah roihu_outputs/
```

The CPU validation job performs:

1. Environment capture: host, Slurm job id, partition, node count, core/thread settings, module set, and output directory.
2. Release CPU build.
3. Required MPI feature build.
4. Brain snapshot run.
5. STDP learning snapshot run.
6. Required MPI/distributed probe.
7. Output size checks for generated `.ndjson` files.

GPU jobs additionally capture `nvidia-smi` and CUDA visibility.

## Output Handling

Roihu validation writes artifacts under `roihu_outputs/${SLURM_JOB_ID}*`. Keep large snapshots out of git. Copy selected `.ndjson` files to a workstation only when visual inspection is needed.

## Viewer Use

The viewer is an inspection tool, not a validation tool. After Roihu generates an `.ndjson` snapshot, copy it next to your browser session and load it with `viewer/index.html`.
