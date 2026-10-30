"""Runtime setup must work before ML imports and without shell activation hooks."""

import json
import os
import subprocess
import sys
from pathlib import Path

from sensespec import runtime


def test_runtime_import_needs_no_third_party_packages():
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys, sensespec; from sensespec import runtime; "
            "assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_moved_environment_paths_and_explicit_settings(tmp_path, monkeypatch):
    prefix = tmp_path / "new environment"
    packages = prefix / "lib/python3.12/site-packages"
    for relative in ("torch/lib", "nvidia/cu13/bin", "nvidia/cu13/lib"):
        (packages / relative).mkdir(parents=True)
    (prefix / "bin").mkdir()
    monkeypatch.setattr(runtime.sys, "executable", str(prefix / "bin/python"))
    monkeypatch.setattr(runtime.sysconfig, "get_path", lambda name: str(packages))
    monkeypatch.setattr(runtime.site, "getsitepackages", lambda: [str(packages)])
    for name in ("CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "forkserver")
    monkeypatch.setenv("no_proxy", "example.test")
    monkeypatch.setenv("NO_PROXY", "another.test")
    original = dict(os.environ)
    env = runtime.runtime_environment()
    assert env["CUDA_HOME"] == str(packages / "nvidia/cu13")
    assert str(packages / "torch/lib") in env["LD_LIBRARY_PATH"].split(os.pathsep)
    assert env["PIPELINE_PYTHON"] == str(prefix / "bin/python")
    assert env["OMP_NUM_THREADS"] == "3"
    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "forkserver"
    assert {"example.test", "another.test", "localhost", "127.0.0.1"} <= set(
        env["no_proxy"].split(",")
    )
    assert env["no_proxy"] == env["NO_PROXY"]
    assert dict(os.environ) == original
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert runtime.runtime_environment() == env


def test_command_child_receives_runtime_without_shell_setup():
    env = dict(os.environ)
    for name in (
        "CUDA_HOME",
        "CUDA_PATH",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "VLLM_USE_V2_MODEL_RUNNER",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "PIPELINE_PYTHON",
    ):
        env.pop(name, None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sensespec.runtime",
            "--",
            sys.executable,
            "-c",
            "import json, os, sys; print(json.dumps({'env':dict(os.environ), 'python':sys.executable}))",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["env"]["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert data["env"]["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert data["env"]["PIPELINE_PYTHON"] == data["python"]


def test_command_exit_code_is_preserved():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sensespec.runtime",
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(7)",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 7, result.stderr


def test_public_model_exports_keep_their_identity():
    import sensespec
    from sensespec.models.draft import AutoDraftModelConfig
    from sensespec.models.dspark import DSparkModel

    assert sensespec.AutoDraftModelConfig is AutoDraftModelConfig
    assert sensespec.DSparkModel is DSparkModel
    assert set(sensespec.__all__) <= set(dir(sensespec))


def test_portable_runtime_settings_reach_ray_actors(monkeypatch):
    from sensespec.utils.env import get_sensespec_env_vars

    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("VLLM_SENSENOVA_LOOP_MHC_BACKEND", "torch")
    monkeypatch.setenv("VLLM_SENSENOVA_LOOP_MHC_FUSION", "none")
    env = get_sensespec_env_vars()
    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert env["OMP_NUM_THREADS"] == "2"
    assert env["TOKENIZERS_PARALLELISM"] == "false"
    assert env["VLLM_SENSENOVA_LOOP_MHC_BACKEND"] == "torch"
    assert env["VLLM_SENSENOVA_LOOP_MHC_FUSION"] == "none"
    assert "CUDA_HOME" not in env  # Each Ray node discovers its own local library paths.
