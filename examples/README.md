# Training examples

Train DSpark and DFlash2 draft models for SenseNova 6.8 Loop 9B Preview.
All recipes use text conversations and keep the vision encoder disabled by default.

Complete [Setup](../README.md#setup), activate the installed Conda or uv
environment, and run commands in Bash from the repository root.

| Recipe | Resources | Guide |
| --- | --- | --- |
| DFlash2 | One node: 4 target + 4 training GPUs | [Single-node training](SenseNova-Loop-9B-Preview-dflah2/README.md) |
| DSpark | One node: 4 target + 4 training GPUs | [Single-node training](SenseNova-Loop-9B-Preview-dspark/README.md) |
| DFlash2 | Three nodes: 16 target + 8 training GPUs | [Multi-node training](SenseNova-Loop-9B-Preview-dflah2-3-node/README.md) |

To start single-node DFlash2 training, set your model and dataset paths:

```bash
export TARGET_MODEL_PATH="/absolute/path/to/SenseNova-6.8-Loop-9B-Preview-mtp13200"
export TRAIN_DATA_PATH="/absolute/path/to/train.jsonl"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_ADDRESS=local
export OUTPUT_DIR="$PWD/outputs/dflash2-$(date +%Y%m%d_%H%M%S)"
export LOG_DIR="$OUTPUT_DIR/logs"
bash examples/SenseNova-Loop-9B-Preview-dflah2/run.sh
```

Each recipe owns its `config.yaml`; shared draft model configurations are in
[`configs/draft_models/`](../configs/draft_models/README.md). Append `key=value`
options to a launch command to override settings.

See the [data guide](data/README.md) for text formatting and optional image input.
The [root README](../README.md) covers checkpoint export and metrics reporting.
Three-node training requires an existing Ray cluster; follow the multi-node
setup before launching that recipe.
