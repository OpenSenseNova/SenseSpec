#!/usr/bin/env bash
# Train SenseNova Loop DFlash2 on one H800 node (4 target + 4 training GPUs).
# Usage: bash examples/SenseNova-Loop-9B-Preview-dflah2/run.sh [--dry-run] [--smoke] [--print-config-only] [key=value ...]
# --smoke: two optimizer steps, 8K context, 32 anchors, checkpoint every step.
# Environment: OUTPUT_DIR, CACHE_DIR, TRAIN_DATA_PATH, TARGET_MODEL_PATH,
# LOAD_PATH (complete checkpoints directory), CKPT_STEP (default: latest), LOG_DIR.
# Uses an active Conda/uv environment; SENSESPEC_ENV can select an environment prefix.

set -euo pipefail
example_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$example_dir/../.." && pwd)
cd "$repo_dir"

dry_run=false
smoke=false
print_config_only=false
args=()
for arg in "$@"; do
  case "$arg" in
    --help|-h) sed -n '2,/^$/p' "${BASH_SOURCE[0]}"; exit 0 ;;
    --dry-run) dry_run=true ;;
    --smoke) smoke=true ;;
    --print-config-only) print_config_only=true; args+=("$arg") ;;
    *) args+=("$arg") ;;
  esac
done

timestamp=$(date +%Y%m%d_%H%M%S)
export TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-/mnt/afs_agents/yankuihao/model_ckpts/SenseNova-6.8-Loop-9B-Preview-mtp13200}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export RAY_ADDRESS="${RAY_ADDRESS:-local}"
OUTPUT_DIR="${OUTPUT_DIR:-$repo_dir/outputs/SenseNova-Loop-9B-Preview-dflah2_$timestamp}"
overrides=("output_dir=$OUTPUT_DIR" "cache_dir=${CACHE_DIR:-$repo_dir/cache/SenseNova-Loop-9B-Preview-dflah2}")
if [[ "$smoke" == true ]]; then
  overrides+=(
    "dataset.train_data_path=$repo_dir/examples/data/sample_sensenova_loop_text.jsonl"
    dataset.num_proc=4 dataset.shuffle_dataset=false
    training.num_train_steps=2 training.lr_total_steps=5
    training.max_seq_length=8192 training.dflash_num_anchors=32
    training.save_interval=1 training.save_per_epoch=false
    inference.vllm.extra_args.max_model_len=8192
    inference.vllm.extra_args.max_num_batched_tokens=4096
    mooncake.global_segment_size=4GB mooncake.local_buffer_size=1GB
    mooncake.host_buffer_size=1GB
    debug.verify_parameter_updates=true
  )
fi
if [[ -n "${TRAIN_DATA_PATH:-}" ]]; then
  overrides+=("dataset.train_data_path=$TRAIN_DATA_PATH")
fi
if [[ -n "${LOAD_PATH:-}" ]]; then
  resume_step="${CKPT_STEP:-$(cat "$LOAD_PATH/latest_checkpointed_iteration.txt")}"
  if [[ ! "$resume_step" =~ ^[0-9]+$ ]]; then
    echo "Invalid checkpoint iteration: $resume_step" >&2
    exit 1
  fi
  resume_step=$((10#$resume_step))
  printf -v checkpoint_name 'iter_%07d' "$resume_step"
  for component in model optimizer lr_scheduler rng.pt meta.json; do
    if [[ ! -e "$LOAD_PATH/$checkpoint_name/$component" ]]; then
      echo "Incomplete checkpoint: $LOAD_PATH/$checkpoint_name/$component" >&2
      exit 1
    fi
  done
  if [[ "$(realpath -m "$OUTPUT_DIR/checkpoints")" == "$(realpath -m "$LOAD_PATH")" ]]; then
    echo "Restore into a new OUTPUT_DIR to preserve the original run." >&2
    exit 1
  fi
  overrides+=("training.load_path=$LOAD_PATH" "training.ckpt_step=$resume_step")
elif [[ -n "${CKPT_STEP:-}" ]]; then
  echo 'CKPT_STEP requires LOAD_PATH.' >&2
  exit 1
fi

pipeline_python="${SENSESPEC_ENV:+${SENSESPEC_ENV}/bin/}python"
launch=("$pipeline_python" -u -m sensespec.train_entry --config "$example_dir/config.yaml" "${overrides[@]}" "${args[@]}")
if [[ "$dry_run" == true ]]; then
  printf 'VLLM_USE_V2_MODEL_RUNNER=1 '
  printf '%q ' "${launch[@]}"
  printf '\n'
  exit 0
fi

export VLLM_USE_V2_MODEL_RUNNER=1
export SENSESPEC_LOG_LEVEL="${SENSESPEC_LOG_LEVEL:-INFO}"

if [[ "$print_config_only" == false ]]; then
  log_dir="${LOG_DIR:-$repo_dir/running_logs}"
  mkdir -p "$log_dir"
  log_file="$log_dir/SenseNova-Loop-9B-Preview-dflah2_$timestamp.log"
  printf 'Training log: %s\nOutput: %s\n' "$log_file" "$OUTPUT_DIR"
  exec >> "$log_file" 2>&1
  echo "Python: $pipeline_python"
  echo "SenseNova Loop DFlash2: default TP=4 target + 4 FSDP ranks; CE + selector loss"
fi
exec "${launch[@]}"
