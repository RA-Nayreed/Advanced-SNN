#!/usr/bin/env bash
#SBATCH --job-name=advanced-snn-multigpu
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:4
#SBATCH --time=02:00:00
#SBATCH --output=advanced-snn-multigpu-%j.out
#SBATCH --error=advanced-snn-multigpu-%j.err

set -euo pipefail

JOB_ID="${SLURM_JOB_ID:?Run this script through sbatch on Roihu}"
OUT_DIR="roihu_outputs/${JOB_ID}_multigpu"
mkdir -p "${OUT_DIR}"
export OUT_DIR

{
  echo "commit=$(git rev-parse HEAD)"
  echo "job_id=${JOB_ID}"
  echo "host=$(hostname)"
  echo "date=$(date --iso-8601=seconds)"
  echo "workdir=$(pwd)"
  echo "roihu_profile=gpu-gh200-four-rank-smoke"
  echo "slurm_partition=${SLURM_JOB_PARTITION:-${SLURM_PARTITION:-unset}}"
  echo "slurm_nodes=${SLURM_JOB_NUM_NODES:-unset}"
  echo "slurm_ntasks=${SLURM_NTASKS:-unset}"
  echo "slurm_ntasks_per_node=${SLURM_NTASKS_PER_NODE:-unset}"
  echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-unset}"
  echo "slurm_gpus=${SLURM_GPUS:-unset}"
  echo "slurm_gpus_on_node=${SLURM_GPUS_ON_NODE:-unset}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
} | tee "${OUT_DIR}/environment.txt"

# Submit with:
#   sbatch --account=<project> scripts/roihu/validate_gpu_multi.sbatch
#
# This is a multi-GPU binding smoke test: it launches one independent
# gpu-event process per reserved GH200. It does not implement cross-GPU
# spike exchange; the simulator still needs that algorithmic work.
if command -v module >/dev/null 2>&1; then
  module purge || true
  if [[ -n "${ROIHU_MODULES:-}" ]]; then
    # shellcheck disable=SC2086
    module load ${ROIHU_MODULES}
  else
    echo "ROIHU_MODULES unset; using modules already available on PATH." | tee -a "${OUT_DIR}/environment.txt"
  fi
  module list 2>&1 | tee -a "${OUT_DIR}/environment.txt"
else
  echo "Environment modules are unavailable in this shell." | tee -a "${OUT_DIR}/environment.txt"
fi

THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${THREADS}"
export RAYON_NUM_THREADS="${THREADS}"
export CARGO_BUILD_JOBS="${THREADS}"
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"

nvidia-smi 2>&1 | tee "${OUT_DIR}/nvidia-smi.log"

cargo build --release --features cuda 2>&1 | tee "${OUT_DIR}/build_cuda.log"

srun --ntasks="${SLURM_NTASKS:-4}" --ntasks-per-node=4 scripts/roihu/run_visible_gpu_rank.sh

find "${OUT_DIR}" -maxdepth 1 -type f -printf "%f %s bytes\n" | sort | tee "${OUT_DIR}/artifact_sizes.txt"
