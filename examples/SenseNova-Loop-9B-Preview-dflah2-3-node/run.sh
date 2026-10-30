#!/usr/bin/env bash
# Train SenseNova Loop DFlash2 on three H800 nodes (8 FSDP + two TP=8 targets).
#
# Usage: bash examples/SenseNova-Loop-9B-Preview-dflah2-3-node/run.sh [--dry-run] [--smoke] [--print-config-only] [key=value ...]
# Run on the head of an existing three-node Ray cluster.
# --smoke: two optimizer steps, 8K context, 32 anchors, accumulation=1.
# SENSESPEC_ENV selects the Python environment; otherwise use the active python.
#
# Optional environment variables:
#   TRAIN_NODE_IPS      One training node IP; set together with INFERENCE_NODE_IPS
#   INFERENCE_NODE_IPS  Two comma-separated inference node IPs
#   TRAIN_DATA_PATH     JSONL dataset (default: bundled sample conversations)
#   TARGET_MODEL_PATH   SenseNova checkpoint (default: SenseNova-6.8-Loop-9B-Preview-mtp13200)
#   LOAD_PATH           Complete checkpoints directory; unset starts a new run
#   CKPT_STEP           Checkpoint iteration; unset selects latest
#   OUTPUT_DIR          Defaults to a new timestamped output directory
#   CACHE_DIR           Defaults to cache/SenseNova-Loop-9B-Preview-dflah2-3-node
#   LOG_DIR             Defaults to running_logs
#   MOONCAKE_PROTOCOL   tcp (default), or rdma when configured on all nodes
#   SENSESPEC_TRITON_CACHE_DIR  Node-local cache (default: /tmp/sensespec-triton-$UID)
#   RAY_ADDRESS         Existing Ray address (default: auto)
#
# --dry-run prints the command without initializing the runtime or contacting Ray.
# --print-config-only prints the resolved config without starting training.
# Training logs go directly to LOG_DIR; use tail -f to follow them.

set -euo pipefail

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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"
export TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-/mnt/afs_agents/yankuihao/model_ckpts/SenseNova-6.8-Loop-9B-Preview-mtp13200}"
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/SenseNova-Loop-9B-Preview-dflah2-3-node_$timestamp}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/cache/SenseNova-Loop-9B-Preview-dflah2-3-node}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/running_logs}"
triton_cache_dir="${SENSESPEC_TRITON_CACHE_DIR:-/tmp/sensespec-triton-${UID}}"

overrides=("output_dir=$OUTPUT_DIR" "cache_dir=$CACHE_DIR")
if [[ "$smoke" == true ]]; then
  overrides+=(
    "dataset.train_data_path=$ROOT_DIR/examples/data/sample_sensenova_loop_text.jsonl"
    dataset.num_proc=4 dataset.shuffle_dataset=false
    training.num_train_steps=2 training.lr_total_steps=5
    training.draft_accumulation_steps=1
    training.max_seq_length=8192 training.dflash_num_anchors=32
    training.save_interval=1 training.save_per_epoch=false
    inference.max_sample_pool_size=16
    inference.vllm.extra_args.max_model_len=8192
    inference.vllm.extra_args.max_num_batched_tokens=4096
    mooncake.global_segment_size=16GB mooncake.local_buffer_size=1GB
    mooncake.host_buffer_size=1GB
    debug.verify_parameter_updates=true
  )
fi
if [[ -n "${TRAIN_DATA_PATH:-}" ]]; then
  overrides+=("dataset.train_data_path=$TRAIN_DATA_PATH")
fi
if [[ -n "${MOONCAKE_PROTOCOL:-}" ]]; then
  overrides+=("mooncake.protocol=$MOONCAKE_PROTOCOL")
fi
if [[ -n "${LOAD_PATH:-}" ]]; then
  resume_step="${CKPT_STEP:-}"
  if [[ -z "$resume_step" ]]; then
    tracker="$LOAD_PATH/latest_checkpointed_iteration.txt"
    if [[ ! -f "$tracker" ]]; then
      echo "ERROR: missing checkpoint tracker: $tracker; set CKPT_STEP explicitly if needed." >&2
      exit 1
    fi
    resume_step="$(<"$tracker")"
  fi
  if [[ ! "$resume_step" =~ ^[0-9]+$ ]]; then
    echo "ERROR: checkpoint iteration must be a non-negative integer." >&2
    exit 1
  fi
  resume_step="$((10#$resume_step))"
  printf -v checkpoint_name 'iter_%07d' "$resume_step"
  for component in model optimizer lr_scheduler rng.pt meta.json; do
    if [[ ! -e "$LOAD_PATH/$checkpoint_name/$component" ]]; then
      echo "ERROR: incomplete checkpoint: $LOAD_PATH/$checkpoint_name/$component" >&2
      exit 1
    fi
  done
  if [[ "$(realpath -m "$OUTPUT_DIR/checkpoints")" == "$(realpath -m "$LOAD_PATH")" ]]; then
    echo "ERROR: restore into a new OUTPUT_DIR to preserve the original run." >&2
    exit 1
  fi
  overrides+=("training.load_path=$LOAD_PATH" "training.ckpt_step=$resume_step")
elif [[ -n "${CKPT_STEP:-}" ]]; then
  echo "ERROR: CKPT_STEP requires LOAD_PATH." >&2
  exit 1
fi

format_ip_list() {
  local raw="$1" expected="$2" name="$3" ip result="["
  local -a ips
  IFS=',' read -r -a ips <<< "$raw"
  if (( ${#ips[@]} != expected )) || [[ "$raw" == *, ]]; then
    echo "ERROR: $name must contain $expected comma-separated node IP(s)." >&2
    return 1
  fi
  for ip in "${ips[@]}"; do
    if [[ -z "$ip" || "$ip" == *[[:space:]\"\\]* ]]; then
      echo "ERROR: invalid entry in $name: '$ip'" >&2
      return 1
    fi
    [[ "$result" == "[" ]] || result+=","
    result+="\"$ip\""
  done
  printf '%s]' "$result"
}
if [[ -n "${TRAIN_NODE_IPS:-}" || -n "${INFERENCE_NODE_IPS:-}" ]]; then
  train_ips="$(format_ip_list "${TRAIN_NODE_IPS:-}" 1 TRAIN_NODE_IPS)"
  inference_ips="$(format_ip_list "${INFERENCE_NODE_IPS:-}" 2 INFERENCE_NODE_IPS)"
  overrides+=(
    "training.placement_strategy=custom"
    "training.training_node_ips=$train_ips"
    "inference.inference_node_ips=$inference_ips"
  )
fi

# Keep the Loop capture contract even when generic CLI overrides are supplied.
pipeline_python="${SENSESPEC_ENV:+${SENSESPEC_ENV}/bin/}python"
launch=(
  "$pipeline_python" -u -m sensespec.train_entry --config "$SCRIPT_DIR/config.yaml"
  "${overrides[@]}" "${args[@]}"
  training.train_with_decode=false
  inference.vllm.direct_hidden_state_capture=true
  inference.store_last_hidden_states=false
  inference.last_hidden_states_prenorm=false
  decode.target_only=false
  decode.speculative_algorithm=null
  decode.speculative_draft_model_path=null
  decode.weight_sync_enabled=false
  decode.weight_sync_interval=0
)
if [[ "$dry_run" == true ]]; then
  printf 'TARGET_MODEL_PATH=%q RAY_ADDRESS=%q SENSESPEC_TRITON_CACHE_DIR=%q TRITON_CACHE_DIR=%q VLLM_USE_V2_MODEL_RUNNER=1 ' "$TARGET_MODEL_PATH" "$RAY_ADDRESS" "$triton_cache_dir" "$triton_cache_dir"
  printf '%q ' "${launch[@]}"
  printf '\n'
  exit 0
fi

# Override the shared cache inherited from Conda and existing Ray nodes.
# get_sensespec_env_vars forwards this path to each actor before Triton imports.
export SENSESPEC_TRITON_CACHE_DIR="$triton_cache_dir"
export TRITON_CACHE_DIR="$triton_cache_dir"
export VLLM_USE_V2_MODEL_RUNNER=1
export SENSESPEC_LOG_LEVEL="${SENSESPEC_LOG_LEVEL:-INFO}"

if [[ "$print_config_only" == false ]]; then
  if [[ "$RAY_ADDRESS" == local ]] || ! "$pipeline_python" -m sensespec.runtime -- ray status --address "$RAY_ADDRESS" >/dev/null 2>&1; then
    echo "ERROR: start or join an existing Ray cluster with three 8-GPU nodes before launching." >&2
    exit 1
  fi
  mkdir -p "$LOG_DIR"
  log_file="$LOG_DIR/SenseNova-Loop-9B-Preview-dflah2-3-node_$timestamp.log"
  printf 'Training logs: %s\n' "$log_file"
  printf 'Follow logs from another terminal: tail -f %q\n' "$log_file"
  exec >> "$log_file" 2>&1
fi
echo "SenseNova Loop DFlash2: V2 direct target capture; CE + selector loss"
if [[ "$smoke" == true ]]; then
  echo "Smoke defaults: 8 FSDP + 2 x TP=8 targets; 8K; global batch 8; two updates"
else
  echo "Defaults: 8 FSDP + 2 x TP=8 targets; 128K; global batch 32; one epoch"
fi
echo "Output: $OUTPUT_DIR"

"${launch[@]}"
