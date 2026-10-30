import os

# Environment variables that should be forwarded to all Ray actors.
# NOTE: TORCHINDUCTOR_CACHE_DIR is intentionally excluded — each node should
# use its own node-local default (/tmp/torchinductor_$USER/) to avoid
# cross-node triton kernel cache corruption over NFS.
_SENSESPEC_ENV_KEYS = [
    "CUDA_LAUNCH_BLOCKING",
    "GLOO_SOCKET_IFNAME",
    "HF_HOME",
    "HF_TOKEN",
    "MC_LOG_LEVEL",
    "MODELOPT_MAX_TOKENS_PER_EXPERT",
    "NCCL_DEBUG",
    "NCCL_SOCKET_IFNAME",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN",
    "SGLANG_DISABLE_CUDNN_CHECK",
    "SGLANG_VLM_CACHE_SIZE_MB",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
    "TORCHINDUCTOR_FX_GRAPH_CACHE",
    "SENSESPEC_LOG_DIR",
    "SENSESPEC_LOG_LEVEL",
    "SENSESPEC_TRITON_CACHE_DIR",
    "TP_SOCKET_IFNAME",
    "CUTE_DSL_CACHE_DIR",
    "SENSESPEC_FLASH_ATTN_OPT_LEVEL",
    "VLLM_SENSENOVA_LOOP_MHC_BACKEND",
    "VLLM_SENSENOVA_LOOP_MHC_FUSION",
    "VLLM_USE_V2_MODEL_RUNNER",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "TORCH_CUDA_ARCH_LIST",
    "OMP_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
]

_SENSESPEC_DEFAULT_ENV_VARS = {
    # SenseSpec's vLLM hidden-state extraction path requires Model Runner V2.
    # An explicit process environment value still takes precedence below.
    "VLLM_USE_V2_MODEL_RUNNER": "1",
}

# Prevent Ray from overriding VISIBLE_DEVICES so actors manage GPU assignment themselves.
# Reference: https://github.com/ray-project/ray/blob/161849364/python/ray/_private/accelerators/
_RAY_NOSET_VISIBLE_DEVICES_KEYS = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
]


def get_sensespec_env_vars() -> dict[str, str]:
    """Return common environment variables for all Ray actors.

    Includes:
    - SENSESPEC_* variables (e.g. log level) from the current process
    - RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES = "1" to prevent Ray from
      overriding device visibility

    Intended for use with ``ray.remote(runtime_env={"env_vars": ...})``.
    Call-site env vars merged after this dict take higher priority.
    """
    env = {
        **_SENSESPEC_DEFAULT_ENV_VARS,
        **{k: "1" for k in _RAY_NOSET_VISIBLE_DEVICES_KEYS},
    }
    env.update({k: os.environ[k] for k in _SENSESPEC_ENV_KEYS if k in os.environ})
    # Opt in explicitly so an inherited Conda cache does not override the
    # node-specific cache settings of unrelated training jobs.
    if cache_dir := env.get("SENSESPEC_TRITON_CACHE_DIR"):
        env["TRITON_CACHE_DIR"] = cache_dir
    return env
