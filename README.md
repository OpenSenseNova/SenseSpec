# SenseSpec

Train **DSpark** and **DFlash2** draft models for speculative decoding, using a
frozen vLLM target model, FSDP training, and Mooncake hidden-state transfer.
The included recipes target **SenseNova 6.8 Loop 9B Preview**.

Follow [Setup](#setup) and [Quick Start](#quick-start) to train a draft model
on text conversations using one eight-GPU node. All included training recipes
default to text-only input with the vision encoder disabled. See
[Examples](#examples) for DSpark and multi-node training.

## Setup

### Requirements

| Component | Requirement |
| --- | --- |
| Platform | Linux x86_64 |
| GPUs | 8 GPUs on one node: 4 for the target model + 4 for draft training |
| GPU architecture | H800 / H100 (SM90), 80 GB per GPU |
| Python | 3.12 |
| PyTorch / CUDA | 2.13.0 / 13.0, installed by the build script |
| Target weights | A compatible `SenseNova-6.8-Loop-9B-Preview-mtp13200` checkpoint, obtained separately |

Obtain the target checkpoint separately, including its configuration, tokenizer,
and safetensors weights. Set its local path in the quick start below.

Use an NVIDIA driver compatible with CUDA 13. The build script installs the CUDA
compiler/toolkit into the Python environment; a system CUDA Toolkit is not
required. Install the system build tools, for example on Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git curl tar protobuf-compiler
nvidia-smi
protoc --version
```

### 1. Clone the repository

```bash
git clone --recurse-submodules https://github.com/OpenSenseNova/SenseSpec.git
cd SenseSpec
```

**Run all commands below in Bash from the repository root.** The installation
uses the bundled [vllm4spec](https://github.com/OpenSenseNova/vllm4spec) fork for
SenseNova Loop support. For an existing checkout, initialize it with
`git submodule update --init --recursive`.

### 2. Create one Python environment

Choose **Conda or uv**, then continue to step 3. Keep that environment activated
for installation and training.

**Conda** — with Conda installed and initialized in your shell:

```bash
conda env create -f environment.yml
conda activate sensespec
export VLLM_BUILD_ENV="$CONDA_PREFIX"
```

`environment.yml` installs Python 3.12, pip, and uv.

**uv** — install [uv](https://docs.astral.sh/uv/getting-started/installation/)
first if needed, then create an isolated environment:

```bash
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
export VLLM_BUILD_ENV="$VIRTUAL_ENV"
```

### 3. Build vLLM, then install SenseSpec

Keep the environment from step 2 activated and `VLLM_BUILD_ENV` set:

```bash
bash tools/vllm_build.sh
uv pip install --python "$VLLM_BUILD_ENV/bin/python" -r requirements.txt
```

`VLLM_BUILD_ENV` selects the environment where vLLM is built and installed.
`requirements.txt` installs SenseSpec and its runtime dependencies.

Build logs are in `logs/vllm-build/`. Use local SSD storage when possible. To
reduce compilation parallelism, replace the build command above with:

```bash
MAX_JOBS=16 NVCC_THREADS=2 bash tools/vllm_build.sh
```

For build options, see [tools/vllm_build.sh](tools/vllm_build.sh).

## Quick Start

Train a DFlash2 draft model on your text dataset using eight GPUs. W&B is
disabled by default; no account or API key is needed.

### 1. Set the model, data, and output paths

Prepare a local JSONL file of text conversations. Each line contains a record
with string-valued message content:

```json
{"conversations":[{"role":"user","content":"Explain how a plant uses sunlight."},{"role":"assistant","content":"Plants use sunlight to turn water and carbon dioxide into sugars, storing energy that supports their growth and releasing oxygen in the process."}]}
```

The whole conversation provides context; only the last assistant answer is
supervised. Keep tokenized conversations shorter than the configured context
limit. Very short answers may be filtered by `dataset.min_loss_tokens`.

After [Setup](#setup), use the same activated shell. Set your model and dataset
paths, then choose the GPUs and output directory:

```bash
export TARGET_MODEL_PATH="/absolute/path/to/SenseNova-6.8-Loop-9B-Preview-mtp13200"
export TRAIN_DATA_PATH="/absolute/path/to/train.jsonl"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_ADDRESS=local
export OUTPUT_DIR="$PWD/outputs/dflash2-$(date +%Y%m%d_%H%M%S)"
export LOG_DIR="$OUTPUT_DIR/logs"
```

Select eight available GPUs with `CUDA_VISIBLE_DEVICES`. Ray and Mooncake
start automatically for single-node training. Use a new `OUTPUT_DIR` for each
fresh run.

### 2. Start training

```bash
bash examples/SenseNova-Loop-9B-Preview-dflah2/run.sh
```

Training uses the settings in the recipe's
[config.yaml](examples/SenseNova-Loop-9B-Preview-dflah2/config.yaml), including
the learning rate, batch size, sequence length, and epoch count. The default
schedule is one epoch. Edit the configuration or append `key=value` options
to the command to change these settings.

The launcher prints `Training log:` and `Output:`, then writes subsequent output
to the log file. To follow progress in a second terminal, use the printed log path:

```bash
tail -f /absolute/path/to/the/printed.log
```

The first launch includes model loading and kernel compilation.

### 3. Find the checkpoints

Checkpoints are saved under `$OUTPUT_DIR/checkpoints/`. Select the latest saved
checkpoint in the original terminal:

```bash
checkpoint_step="$(cat "$OUTPUT_DIR/checkpoints/latest_checkpointed_iteration.txt")"
printf -v checkpoint_name 'iter_%07d' "$checkpoint_step"
export CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints/$checkpoint_name"
ls "$CHECKPOINT_DIR"
```

The output directory contains:

```text
<OUTPUT_DIR>/
  config.yaml
  logs/
    SenseNova-Loop-9B-Preview-dflah2_<timestamp>.log
  checkpoints/
    latest_checkpointed_iteration.txt
    iter_<step>/
```

## Examples

Choose a recipe and set `TRAIN_DATA_PATH` to your text dataset. All recipes
disable the vision encoder by default:

| Recipe | Topology | Instructions |
| --- | --- | --- |
| DFlash2 | One node, 4 target + 4 training GPUs | [Single-node guide](examples/SenseNova-Loop-9B-Preview-dflah2/README.md) |
| DSpark | One node, 4 target + 4 training GPUs | [Single-node guide](examples/SenseNova-Loop-9B-Preview-dspark/README.md) |
| DFlash2 | Three nodes, 16 target + 8 training GPUs | [Ray cluster and training guide](examples/SenseNova-Loop-9B-Preview-dflah2-3-node/README.md) |

Each recipe owns its `config.yaml`. Extra `key=value` arguments override its
settings; see the [configuration guide](configs/README.md) and
[draft model configurations](configs/draft_models/README.md). For three nodes,
install the same environment and source build on every node, use paths accessible
to the relevant workers, and follow the Ray setup before launching training.

### Configure training

Set `training.num_epochs` or `training.num_train_steps` for the training
schedule, and `training.save_interval` for checkpoint frequency.
The recipes default to a 131072-token context limit. Choose a length suited to
your dataset and GPU memory, setting `training.max_seq_length` and
`inference.vllm.extra_args.max_model_len` to the same value. Tokenized inputs
must be shorter than that limit.

For large datasets, `dataset.tokenization_batch_size=1000` reduces temporary
preprocessing memory. The final processed dataset still stays in memory.

### Optional image input

Image input requires explicitly setting `inference.vllm.enable_multimodal=true`
and `inference.vllm.extra_args.language_model_only=false`, supplying an image
dataset through `TRAIN_DATA_PATH`, and choosing a new output directory.
See the [data guide](examples/data/README.md) for the image format and launch
command. Expanded image tokens count toward the context limit.

To run the target model as a text/image service or generate responses, use the
separate [target inference example](vllm4spec/examples/SenseNova-Loop-9B-Preview-mhc/README.md).

## Training Modes

### Resume training

To resume an interrupted run, keep its model, dataset, and training settings,
and restore into a new output directory:

```bash
previous_output_dir="/absolute/path/to/previous-run"
LOAD_PATH="$previous_output_dir/checkpoints" \
OUTPUT_DIR="${previous_output_dir}-resumed" \
LOG_DIR="${previous_output_dir}-resumed/logs" \
bash examples/SenseNova-Loop-9B-Preview-dflah2/run.sh
```

The DFlash2 launcher reads the latest step from
`latest_checkpointed_iteration.txt`; optionally set `CKPT_STEP` to select an
existing complete checkpoint. `LOAD_PATH` is the `checkpoints/` directory,
not the individual `iter_*` directory. The total step limit must exceed the
restored step. Resume restores model, optimizer, scheduler, RNG, and step state.

### Start a new run from existing weights

For a fresh training schedule from compatible weights, use
`training.load_path=/path/to/checkpoints` and
`training.continual_training=true` as CLI overrides or YAML settings. This
restores weights without continuing the previous optimizer/scheduler/step state.
Use a new output directory and keep the target architecture and draft shape
compatible.

For a compatible external draft checkpoint, see
`python -m sensespec.runtime -- python tools/convert_to_sensespec.py --help`.
The conversion tool creates a weights-only initialization directory for
`training.load_path` with `training.continual_training=true`.

## Checkpoint Conversion

Use `CHECKPOINT_DIR` selected above, or set it to the checkpoint directory you
want to export. The output contains `model.safetensors` and `config.json`:

```bash
python -m sensespec.runtime -- python tools/convert_to_hf.py \
  --input-dir "$CHECKPOINT_DIR" \
  --config configs/draft_models/sensenova_loop_dflash2.json \
  --output-dir "$OUTPUT_DIR/draft-hf" \
  --vllm
```

Use the matching draft configuration with `--config`. For a DSpark checkpoint,
use `configs/draft_models/sensenova_loop_dspark.json`. `--vllm` selects the
vLLM export format.

DFlash2 requires the full target vocabulary; do not set `draft_vocab_size` or
use `--prune-vocab` for it. See
`python -m sensespec.runtime -- python tools/convert_to_hf.py --help` for the
other conversion options.

## Metrics Reporting

All included recipes default to `logging.report_to=none`. Local logs and
checkpoints work without a W&B account.

To enable W&B, run `wandb login` in the training environment (or supply your
`WANDB_API_KEY`), then append these overrides to your recipe command:

```text
logging.report_to=wandb logging.wandb_project=sensespec logging.wandb_mode=online
```

For a team project, also set `logging.wandb_team` to your entity. Follow the
printed W&B run URL to view `train/avg_loss` against `train/step`.
For local W&B recording, use `logging.report_to=wandb logging.wandb_mode=offline`;
this needs no login during training. To upload later, authenticate and run
`wandb sync /path/to/wandb/offline-run-*`.

## Troubleshooting

| Symptom | Check or action |
| --- | --- |
| Missing `sensenova_loop.py` or unknown target architecture | Initialize the matching `vllm4spec` source and check the model architecture. A published vLLM wheel or standard Qwen checkpoint cannot replace these. |
| Imports fail | Activate the environment used for installation and run `python -m sensespec.runtime --check`. |
| `libcudart` or native extension load errors | Run `python -m sensespec.runtime --check`; verify the pinned dependencies and source build. Wrap standalone tools with `python -m sensespec.runtime -- ...` so they receive CUDA library paths. |
| Cannot find model or data | Set `TARGET_MODEL_PATH` and `TRAIN_DATA_PATH` to accessible absolute paths. |
| Training waits for GPUs or runs out of memory | Check `nvidia-smi`: single-node training needs eight available GPUs. Set `RAY_ADDRESS=local` and adjust sequence length and batch size for the available memory. |
| DFlash2 terminal becomes quiet | Follow the file printed after `Training log:`. Subsequent output is redirected there. |
| Vision configuration conflict | Set both `inference.vllm.enable_multimodal=false` and `inference.vllm.extra_args.language_model_only=true` for text-only mode. Reverse both for image input. |
| Samples exceed the context limit | Adjust both sequence-length settings together. For image data, include expanded image tokens in the budget; resize images or split conversations. |
| Mooncake startup or transfer failure | Check the training log and `python -m sensespec.runtime --check`. The single-node recipe uses TCP and the pinned `mooncake-transfer-engine==0.3.13.post1`; verify this setup before changing transport or packages. |
| Resume immediately finishes | Set `training.num_train_steps` above the restored step and use a new `OUTPUT_DIR`. |

For more detail, set `SENSESPEC_LOG_LEVEL=DEBUG` before launching. `LOG_DIR`
controls the launcher log directory; `CACHE_DIR` controls the dataset cache.
To also collect per-actor files, set `SENSESPEC_LOG_DIR` to an absolute writable
directory accessible from all nodes. Without that setting, Ray captures actor
stdout/stderr.

## License

SenseSpec is released under the [Apache License 2.0](LICENSE).
The project is built upon TorchSpec by LightSeek Foundation. Inherited TorchSpec
code retains its original MIT copyright and permission notices; see
[LICENSES/TorchSpec-MIT.txt](LICENSES/TorchSpec-MIT.txt). The bundled `vllm4spec/`
source and other third-party components retain their own licenses and notices.
