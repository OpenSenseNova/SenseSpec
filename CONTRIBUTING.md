# Contributing to SenseSpec

Thank you for your interest in contributing to SenseSpec! This guide will help you get started.

## Development Setup

1. Follow [Setup in the README](README.md#setup) to create a Conda or uv
environment, build the bundled vLLM with `bash tools/vllm_build.sh`, install
the runtime requirements and verify the environment. Keep that environment
activated and run the following commands from the repository root.

2. Install the dev dependencies:

```bash
python -m pip install -e ".[dev]"
```

3. (Optional) Install Flash Attention for full feature support:

```bash
pip install -e ".[fa]"
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Pre-commit hooks enforce this automatically:

```bash
pre-commit install
```

Key style rules:
- Line length: 100 characters
- Python 3.12+
- Absolute imports only (no relative imports)
- Imports sorted with `isort` (via Ruff)

Run the linter manually:

```bash
ruff check .
ruff format --check .
```

## Running Tests

Run the retained DSpark, DSpark metrics and DFlash2 tests:

```bash
./tools/run_all_tests.sh
```

Or run individual tests with pytest:

```bash
python -m pytest tests/test_dspark.py -v
```

## Pull Requests

1. Fork the repository and create a feature branch from `main`.
2. Make your changes and ensure all tests pass.
3. Run `ruff check .` and `ruff format .` before committing.
4. Write a clear PR description explaining what changed and why.
5. Link any related issues.

## Reporting Issues

When filing a bug report, please include:
- Python version and OS
- GPU type and driver version
- Steps to reproduce the issue
- Full error traceback
