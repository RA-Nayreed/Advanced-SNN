#!/usr/bin/env bash
#SBATCH --job-name=advanced-snn-validation
#SBATCH --account=REPLACE_WITH_ROIHU_ACCOUNT
#SBATCH --partition=REPLACE_WITH_ROIHU_PARTITION
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
  echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-unset}"
} | tee "${OUT_DIR}/environment.txt"

# Replace these with the exact Roihu module names from the current Roihu documentation.
module purge || true
# module load rust
# module load cuda
# module load gcc
# module load openmpi
module list 2>&1 | tee -a "${OUT_DIR}/environment.txt"

export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

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

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi 2>&1 | tee "${OUT_DIR}/nvidia-smi.log"
  cargo build --release --features cuda 2>&1 | tee "${OUT_DIR}/build_cuda.log"
  ./target/release/advanced-snn gpu-event \
    --neurons 100000 \
    --fanout 64 \
    --steps 200 \
    --seed 7 \
    2>&1 | tee "${OUT_DIR}/gpu_event.log"
else
  echo "CUDA device not visible in this allocation; GPU backend validation skipped by allocation state." | tee "${OUT_DIR}/gpu_event.log"
fi

if command -v mpirun >/dev/null 2>&1 || command -v srun >/dev/null 2>&1; then
  cargo build --release --features mpi 2>&1 | tee "${OUT_DIR}/build_mpi.log"
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
  echo "MPI launcher not available in this allocation." | tee "${OUT_DIR}/distributed_probe.log"
fi

wc -l "${OUT_DIR}"/*.ndjson | tee "${OUT_DIR}/snapshot_line_counts.txt"
find "${OUT_DIR}" -maxdepth 1 -type f -printf "%f %s bytes\n" | sort | tee "${OUT_DIR}/artifact_sizes.txt"
