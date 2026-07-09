#!/usr/bin/env bash
set -euo pipefail

: "${OUT_DIR:?OUT_DIR must be exported by the parent Slurm script}"

original_visible="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=, read -r -a visible_devices <<< "${original_visible}"
local_rank="${SLURM_LOCALID:-0}"
world_rank="${SLURM_PROCID:-0}"

if (( ${#visible_devices[@]} == 1 )); then
  selected_device="${visible_devices[0]}"
elif (( local_rank < ${#visible_devices[@]} )); then
  selected_device="${visible_devices[local_rank]}"
else
  selected_device="${local_rank}"
fi
export CUDA_VISIBLE_DEVICES="${selected_device}"

rank_log="${OUT_DIR}/gpu_event_rank_${world_rank}.log"
seed=$((7 + world_rank))

{
  echo "rank=${world_rank}"
  echo "local_rank=${local_rank}"
  echo "host=$(hostname)"
  echo "original_cuda_visible_devices=${original_visible}"
  echo "selected_cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "omp_num_threads=${OMP_NUM_THREADS:-unset}"
  ./target/release/advanced-snn gpu-event \
    --neurons 100000 \
    --fanout 64 \
    --steps 200 \
    --seed "${seed}"
} 2>&1 | tee "${rank_log}"
