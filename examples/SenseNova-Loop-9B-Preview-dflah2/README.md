# SenseNova Loop DFlash2 training

Train a DFlash2 draft model for `SenseNova-6.8-Loop-9B-Preview-mtp13200` on one
node with eight GPUs. Four GPUs run the target model with tensor parallelism;
four GPUs train the draft with FSDP. Training uses text input by default and
keeps the vision encoder disabled.

Complete [Setup](../../README.md#setup) and activate the installed Conda or uv
environment. Run all commands below in Bash from the repository root.

## Start training

Replace the model and dataset paths with your own. See the
[data guide](../data/README.md) for the JSONL conversation format.

```bash
export TARGET_MODEL_PATH="/absolute/path/to/SenseNova-6.8-Loop-9B-Preview-mtp13200"
export TRAIN_DATA_PATH="/absolute/path/to/train.jsonl"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_ADDRESS=local
export OUTPUT_DIR="$PWD/outputs/dflash2-$(date +%Y%m%d_%H%M%S)"
export LOG_DIR="$OUTPUT_DIR/logs"
bash examples/SenseNova-Loop-9B-Preview-dflah2/run.sh
```

Select eight available GPUs. The launcher starts local Ray and Mooncake services
automatically. The target model remains frozen while the draft is trained.
The full conversation provides context; only the last assistant answer is supervised.

## Configuration

Training settings are in [config.yaml](config.yaml). The draft architecture is
specified by [sensenova_loop_dflash2.json](../../configs/draft_models/sensenova_loop_dflash2.json).

| Setting | Default |
| --- | --- |
| Training schedule | One epoch |
| Global batch size | 4 |
| Context limit | 131072 tokens |
| Checkpoint saving | Every 2 optimizer steps and at the end of each epoch |
| Checkpoint retention | Latest 2 complete checkpoints |
| W&B | Disabled |

Edit the YAML or append `key=value` options to the launch command. Set
`training.num_epochs` or `training.num_train_steps` for the training schedule.
When adjusting context length, set `training.max_seq_length` and
`inference.vllm.extra_args.max_model_len` to the same value. Tokenized inputs
must be shorter than that limit.

## Logs and checkpoints

The launcher prints the training log path and output directory. Subsequent
output goes to that log file. Follow it from a second terminal:

```bash
tail -f /absolute/path/to/the/printed.log
```

The resolved configuration is saved to `$OUTPUT_DIR/config.yaml`, and checkpoints
are saved under `$OUTPUT_DIR/checkpoints/`. Use a new output directory for each
fresh run. `CACHE_DIR` can select a different dataset cache directory.

## Resume training

Keep the same model, dataset, and training settings, and restore into a new
output directory:

```bash
previous_output_dir="/absolute/path/to/previous-run"
LOAD_PATH="$previous_output_dir/checkpoints" \
OUTPUT_DIR="${previous_output_dir}-resumed" \
LOG_DIR="${previous_output_dir}-resumed/logs" \
bash examples/SenseNova-Loop-9B-Preview-dflah2/run.sh
```

The launcher selects the latest complete checkpoint from the tracker. Set
`CKPT_STEP` to select another saved step. The training step limit must exceed
the restored step.

For export, follow [Checkpoint Conversion](../../README.md#checkpoint-conversion)
and use `configs/draft_models/sensenova_loop_dflash2.json` with `--config`.
DFlash2 uses the full target vocabulary; keep vocabulary pruning disabled.
Image input requires explicitly enabling it as described in the
[data guide](../data/README.md).
