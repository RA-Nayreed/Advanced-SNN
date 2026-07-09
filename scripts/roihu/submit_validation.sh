#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/roihu/submit_validation.sh --account <project> [cpu|gpu|multi-gpu]

Environment:
  ROIHU_MODULES="rust gcc cuda openmpi"  Optional modules to load in the job.

Examples:
  scripts/roihu/submit_validation.sh --account project_2000000 cpu
  scripts/roihu/submit_validation.sh --account project_2000000 gpu
  scripts/roihu/submit_validation.sh --account project_2000000 multi-gpu
USAGE
}

account="${ROIHU_ACCOUNT:-}"
mode="cpu"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account)
      account="${2:-}"
      shift 2
      ;;
    --account=*)
      account="${1#--account=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    cpu|gpu|single-gpu|multi-gpu)
      mode="$1"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${account}" ]]; then
  echo "Missing Roihu project account. Use --account <project> or ROIHU_ACCOUNT." >&2
  exit 2
fi

case "${mode}" in
  cpu)
    script="scripts/roihu/validate_big_update.sbatch"
    ;;
  gpu|single-gpu)
    script="scripts/roihu/validate_gpu_single.sbatch"
    ;;
  multi-gpu)
    script="scripts/roihu/validate_gpu_multi.sbatch"
    ;;
  *)
    echo "Unsupported validation mode: ${mode}" >&2
    exit 2
    ;;
esac

sbatch --account="${account}" "${script}"
