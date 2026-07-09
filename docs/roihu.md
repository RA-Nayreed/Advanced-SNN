# Roihu HPC Validation

This repository treats Roihu as the validation environment. Do not add workstation validation checks. Validation is run on Roihu after a substantial simulator, CUDA, distributed, snapshot, or viewer-data change.

Public web search did not expose a Roihu-specific manual with stable queue/module names. Keep the script placeholders below aligned with the current Roihu user documentation and the output of `module avail`, `sinfo`, and project/account instructions on the cluster.

## Policy

- No workstation validation fixtures are tracked in the repository.
- Snapshot data is produced by Roihu jobs and kept out of git.
- Validation jobs are Slurm jobs submitted on Roihu.
- Run validation after a large update, not for every small wording or style change.
- Record the Slurm job id, git commit, module set, node type, and output directory in experiment notes.

## Expected Roihu Inputs

Before submitting, set these values in `scripts/roihu/validate_big_update.sbatch`:

- `--account`: your Roihu project/account.
- `--partition`: the correct CPU or GPU partition.
- `--time`: wall time accepted by the selected partition.
- `module load`: Roihu module names for Rust, CUDA, GCC, and MPI.
- GPU request: use the Roihu-documented GPU syntax, commonly a Slurm `--gres` or partition-specific GPU option.

The scripts intentionally keep these fields explicit because incorrect partition or module names are site-specific and should come from Roihu documentation, not from this repository.

## Big-Update Validation Flow

From a Roihu login node:

```bash
git clone git@github.com:RA-Nayreed/Advanced-SNN.git
cd Advanced-SNN
sbatch scripts/roihu/validate_big_update.sbatch
```

Monitor with:

```bash
squeue -u "$USER"
```

Inspect output after completion:

```bash
ls -lah roihu_outputs/
```

The validation job performs:

1. Environment capture: host, Slurm job id, loaded modules, compiler versions, CUDA visibility.
2. Release CPU build.
3. Brain snapshot run.
4. STDP learning snapshot run.
5. CUDA build and event backend run when CUDA is available in the allocation.
6. MPI/distributed probe when MPI support is available.
7. Output size checks for generated `.ndjson` files.

## Output Handling

Roihu validation writes artifacts under `roihu_outputs/${SLURM_JOB_ID}`. Keep large snapshots out of git. Copy selected `.ndjson` files to a workstation only when visual inspection is needed.

## Viewer Use

The viewer is an inspection tool, not a validation tool. After Roihu generates an `.ndjson` snapshot, copy it next to your browser session and load it with `viewer/index.html`.

## Slurm Notes

This workflow uses Slurm batch scripts. Slurm `sbatch` submits a script to the scheduler, and `#SBATCH` lines declare resource requests. `srun` launches commands inside the allocation. Keep all resource requests aligned with the active Roihu documentation and project policy.
