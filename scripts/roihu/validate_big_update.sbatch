#!/usr/bin/env bash
#SBATCH --job-name=advanced-snn-validation
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=advanced-snn-%j.out
#SBATCH --error=advanced-snn-%j.err

set -euo pipefail

JOB_ID="${SLURM_JOB_ID:?Run this script through sbatch on Roihu}"
OUT_DIR="roihu_outputs/${JOB_ID}"
mkdir -p "${OUT_DIR}"

{
  echo "commit=$(git rev-parse HEAD)"
  echo "job_id=${JOB_ID}"
  echo "host=$(hostname)"
  echo "date=$(date --iso-8601=seconds)"
  echo "workdir=$(pwd)"
  echo "roihu_profile=cpu-small-shared-node"
  echo "slurm_partition=${SLURM_JOB_PARTITION:-${SLURM_PARTITION:-unset}}"
  echo "slurm_nodes=${SLURM_JOB_NUM_NODES:-unset}"
  echo "slurm_ntasks=${SLURM_NTASKS:-unset}"
  echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-unset}"
  echo "slurm_mem_per_node=${SLURM_MEM_PER_NODE:-unset}"
} | tee "${OUT_DIR}/environment.txt"

# Submit with:
#   sbatch --account=<project> scripts/roihu/validate_big_update.sbatch
#
# Load modules through ROIHU_MODULES so site-specific module names are not
# hard-coded in the repository, for example:
#   sbatch --account=<project> --export=ALL,ROIHU_MODULES="rust gcc openmpi" ...
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

mpi_build_available() {
  if [[ -n "${MPI_PKG_CONFIG:-}" || -n "${CRAY_MPICH_DIR:-}" ]]; then
    return 0
  fi
  if command -v mpicc >/dev/null 2>&1; then
    return 0
  fi
  if command -v pkg-config >/dev/null 2>&1; then
    pkg-config --exists mpich || pkg-config --exists ompi
    return $?
  fi
  return 1
}

cargo build --release 2>&1 | tee "${OUT_DIR}/build_cpu.log"

./target/release/advanced-snn brain \
  --neurons 20000 \
  --fanout 64 \
  --steps 500 \
  --seed 7 \
  --snapshot-out "${OUT_DIR}/brain_blob.ndjson" \
  --snapshot-every 5 \
  --snapshot-neurons 5000 \
  --snapshot-synapses 20000 \
  2>&1 | tee "${OUT_DIR}/brain_blob.log"

./target/release/advanced-snn learn \
  --neurons 20000 \
  --fanout 64 \
  --steps 500 \
  --seed 7 \
  --external-prob 0.006 \
  --external-current 1.2 \
  --snapshot-out "${OUT_DIR}/brain_learning.ndjson" \
  --snapshot-every 5 \
  --snapshot-neurons 5000 \
  --snapshot-synapses 20000 \
  2>&1 | tee "${OUT_DIR}/brain_learning.log"

if mpi_build_available; then
  if cargo build --release --features mpi 2>&1 | tee "${OUT_DIR}/build_mpi.log"; then
    if command -v srun >/dev/null 2>&1; then
      srun --ntasks=1 ./target/release/advanced-snn distributed-cpu \
        --neurons 20000 \
        --fanout 64 \
        --steps 10 \
        --seed 7 \
        2>&1 | tee "${OUT_DIR}/distributed_probe.log"
    else
      mpirun -np 1 ./target/release/advanced-snn distributed-cpu \
        --neurons 20000 \
        --fanout 64 \
        --steps 10 \
        --seed 7 \
        2>&1 | tee "${OUT_DIR}/distributed_probe.log"
    fi
  else
    {
      echo "MPI feature build failed; distributed probe skipped."
      echo "See ${OUT_DIR}/build_mpi.log."
      echo "Common cause: bindgen requires libclang; load a Roihu LLVM/Clang module or set LIBCLANG_PATH."
    } | tee "${OUT_DIR}/distributed_probe.log"
  fi
else
  {
    echo "MPI build environment not available; distributed probe skipped."
    echo "Set ROIHU_MODULES to include Roihu OpenMPI, or otherwise provide mpicc/MPI_PKG_CONFIG."
  } | tee "${OUT_DIR}/distributed_probe.log"
fi

wc -l "${OUT_DIR}"/*.ndjson | tee "${OUT_DIR}/snapshot_line_counts.txt"
find "${OUT_DIR}" -maxdepth 1 -type f -printf "%f %s bytes\n" | sort | tee "${OUT_DIR}/artifact_sizes.txt"
