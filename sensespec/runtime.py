"""Prepare the current Python environment before loading CUDA extensions.

Used automatically by train_entry. For other commands, use
``python -m sensespec.runtime -- ray start ...``. No shell activation hooks
or environment-specific paths are installed into site-packages.
"""

import argparse
import json
import os
import site
import subprocess
import sys
import sysconfig
from pathlib import Path


def _prepend(env: dict[str, str], key: str, paths: list[Path]) -> None:
    values = [str(path) for path in paths if path.is_dir()]
    values.extend(part for part in env.get(key, "").split(os.pathsep) if part)
    if values:
        env[key] = os.pathsep.join(dict.fromkeys(values))


def runtime_environment() -> dict[str, str]:
    """Discover paths from this interpreter, including inherited venv packages."""
    env = dict(os.environ)
    package_dirs = list(
        dict.fromkeys(
            Path(path) for path in [sysconfig.get_path("purelib"), *site.getsitepackages()]
        )
    )
    cuda_home = env.get("CUDA_HOME") or env.get("CUDA_PATH")
    if not cuda_home:
        cuda_home = next(
            (str(path / "nvidia/cu13") for path in package_dirs if (path / "nvidia/cu13").is_dir()),
            None,
        )
    library_dirs = []
    executable_dirs = [Path(sys.executable).parent]
    if cuda_home:
        cuda = Path(cuda_home).expanduser().absolute()
        env["CUDA_HOME"] = env["CUDA_PATH"] = str(cuda)
        executable_dirs.append(cuda / "bin")
        library_dirs.extend([cuda / "lib", cuda / "lib64"])
    for path in package_dirs:
        library_dirs.extend([path / "torch/lib", path / "nvidia/cuda_runtime/lib"])
    _prepend(env, "PATH", executable_dirs)
    _prepend(env, "LD_LIBRARY_PATH", library_dirs)

    root = Path(__file__).resolve().parents[1]
    if (root / "pyproject.toml").is_file():
        source_paths = [root]
        if (root / "vllm4spec/vllm/__init__.py").is_file():
            source_paths.append(root / "vllm4spec")
        _prepend(env, "PYTHONPATH", source_paths)

    # Merge both spellings and keep the caller's existing proxy exclusions.
    bypass = [
        value.strip()
        for key in ("no_proxy", "NO_PROXY")
        for value in env.get(key, "").split(",")
        if value.strip()
    ]
    bypass.extend(["localhost", "127.0.0.1", "::1"])
    try:
        addresses = subprocess.check_output(
            ["hostname", "-I"], text=True, stderr=subprocess.DEVNULL, timeout=2
        )
        bypass.extend(addresses.split())
    except (OSError, subprocess.SubprocessError):
        pass
    env["no_proxy"] = env["NO_PROXY"] = ",".join(dict.fromkeys(bypass))
    env["PIPELINE_PYTHON"] = sys.executable
    env.setdefault("DEBUG_PYTHON", sys.executable)
    for key, value in {
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "TORCH_CUDA_ARCH_LIST": "9.0",
        "OMP_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }.items():
        env.setdefault(key, value)
    return env


def ensure_runtime() -> None:
    """Re-exec only when needed: the ELF loader reads library paths at startup."""
    env = runtime_environment()
    if env != dict(os.environ):
        os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check CUDA, local vLLM and Mooncake")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after --")
    args = parser.parse_args()
    if args.command:
        if args.check:
            parser.error("--check does not accept a command")
        command = args.command[1:] if args.command[0] == "--" else args.command
        if not command:
            parser.error("provide a command after --")
        os.execvpe(command[0], command, runtime_environment())
    elif args.check:
        ensure_runtime()
        import torch
        import vllm
        import vllm._C_stable_libtorch  # noqa: F401
        from mooncake.store import MooncakeDistributedStore  # noqa: F401

        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU is visible; check the driver and CUDA_VISIBLE_DEVICES")
        if not (Path(vllm.__file__).parent / "model_executor/models/sensenova_loop.py").is_file():
            raise RuntimeError("The installed vLLM does not contain the required Loop/mHC model")
        print(
            json.dumps(
                {
                    "python": sys.executable,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "cuda_home": os.environ.get("CUDA_HOME"),
                    "vllm_source": vllm.__file__,
                    "visible_gpus": torch.cuda.device_count(),
                    "gpu_0": torch.cuda.get_device_name(0),
                    "native_imports": "vLLM and Mooncake OK",
                },
                indent=2,
            )
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
