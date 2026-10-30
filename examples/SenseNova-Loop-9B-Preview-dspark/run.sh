#!/usr/bin/env bash
# Train SenseNova Loop DSpark on one H800 node (4 target + 4 training GPUs).
# Usage: bash examples/SenseNova-Loop-9B-Preview-dspark/run.sh [--dry-run|--print-config-only] [key=value ...]
# Environment: SENSESPEC_ENV, TARGET_MODEL_PATH, TRAIN_DATA_PATH,
# OUTPUT_DIR, CACHE_DIR, LOG_DIR, CUDA_VISIBLE_DEVICES (default: 0-7).

set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  sed -n '2,/^$/p' "${BASH_SOURCE[0]}"
  exit 0
fi
dry_run=false
if [[ "${1:-}" == --dry-run ]]; then
  dry_run=true
  shift
fi
example_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$example_dir/../.." && pwd)
cd "$repo_dir"
timestamp=$(date +%Y%m%d_%H%M%S)
export TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-/mnt/afs_agents/yankuihao/model_ckpts/SenseNova-6.8-Loop-9B-Preview-mtp13200}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export RAY_ADDRESS="${RAY_ADDRESS:-local}"
OUTPUT_DIR="${OUTPUT_DIR:-$repo_dir/outputs/SenseNova-Loop-9B-Preview-dspark_$timestamp}"
overrides=("output_dir=$OUTPUT_DIR" "cache_dir=${CACHE_DIR:-$repo_dir/cache/SenseNova-Loop-9B-Preview-dspark}")
if [[ -n "${TRAIN_DATA_PATH:-}" ]]; then
  overrides+=("dataset.train_data_path=$TRAIN_DATA_PATH")
fi
pipeline_python="${SENSESPEC_ENV:+${SENSESPEC_ENV}/bin/}python"
launch=(
  "$pipeline_python" -m sensespec.train_entry --config "$example_dir/config.yaml"
  "${overrides[@]}" "$@"
)
if [[ "$dry_run" == true ]]; then
  printf 'VLLM_USE_V2_MODEL_RUNNER=1 '
  printf '%q ' "${launch[@]}"
  printf '\n'
  exit 0
fi
export VLLM_USE_V2_MODEL_RUNNER=1
export SENSESPEC_LOG_LEVEL="${SENSESPEC_LOG_LEVEL:-INFO}"
log_dir="${LOG_DIR:-$repo_dir/running_logs}"
mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/SenseNova-Loop-9B-Preview-dspark_$timestamp.log") 2>&1
echo "SenseNova Loop: TP=4 target + 4 FSDP ranks; 56 target execution layers; five DSpark layers"
echo "Output: $OUTPUT_DIR"
"${launch[@]}"
