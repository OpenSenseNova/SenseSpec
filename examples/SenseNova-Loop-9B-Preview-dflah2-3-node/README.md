# SenseNova Loop DFlash2 multi-node training

Train a DFlash2 draft model for `SenseNova-6.8-Loop-9B-Preview-mtp13200` across
three eight-GPU nodes. Training uses text input by default and keeps the vision
encoder disabled.

| Role | Resources |
| --- | --- |
| Training | One node with 8 GPUs running FSDP |
| Target inference | Two nodes, each running an independent target model with TP=8 |
| Total | 3 nodes, 24 GPUs |

## Prepare the nodes

Complete [Setup](../../README.md#setup) on every node. Use the same source and
package versions, and make the repository and Python environment available at
the same absolute paths. All nodes must be able to read the model and dataset
and access the output directory.

Activate the installed Conda or uv environment on each node. Run all commands
below in Bash from the repository root. Select eight available GPUs on every node:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

If a specific network interface is required, set `NCCL_SOCKET_IFNAME`,
`GLOO_SOCKET_IFNAME`, and `TP_SOCKET_IFNAME` before starting Ray. When using an
HTTP proxy, include all three node IPs in `no_proxy` and `NO_PROXY`.

## Start the Ray cluster

Use the training node as the head. Replace the addresses below with your node IPs.
On the head node:

```bash
export HEAD_IP=192.0.2.10
python -m sensespec.runtime -- ray start \
  --head --node-ip-address "$HEAD_IP" --port 6379 \
  --num-gpus 8 --disable-usage-stats
```

On each inference node, set `NODE_IP` to that node's own address. Use
`192.0.2.11` for the first node and `192.0.2.12` for the second in this example:

```bash
export HEAD_IP=192.0.2.10
export NODE_IP=192.0.2.11
python -m sensespec.runtime -- ray start \
  --address "$HEAD_IP:6379" --node-ip-address "$NODE_IP" \
  --num-gpus 8 --disable-usage-stats
```

On the head node, confirm the cluster reports all 24 GPUs:

```bash
python -m sensespec.runtime -- ray status --address "$HEAD_IP:6379"
```

An existing cluster with the required resources can also be used.

## Start training

Run on the head node. Replace the model and dataset paths, and set both
`TRAIN_NODE_IPS` and `INFERENCE_NODE_IPS` to the actual node addresses:

```bash
export TARGET_MODEL_PATH="/absolute/path/to/SenseNova-6.8-Loop-9B-Preview-mtp13200"
export TRAIN_DATA_PATH="/absolute/path/to/train.jsonl"
export TRAIN_NODE_IPS="$HEAD_IP"
export INFERENCE_NODE_IPS=192.0.2.11,192.0.2.12
export RAY_ADDRESS="$HEAD_IP:6379"
export OUTPUT_DIR="$PWD/outputs/dflash2-3-node-$(date +%Y%m%d_%H%M%S)"
export LOG_DIR="$OUTPUT_DIR/logs"
bash examples/SenseNova-Loop-9B-Preview-dflah2-3-node/run.sh
```

Provide one training IP and two comma-separated inference IPs. The launcher
connects to the existing Ray cluster and starts training. TCP is the default
Mooncake transport. See the [data guide](../data/README.md) for the JSONL format.

## Configuration

Training settings are in [config.yaml](config.yaml). The draft architecture is
specified by [sensenova_loop_dflash2.json](../../configs/draft_models/sensenova_loop_dflash2.json).

| Setting | Default |
| --- | --- |
| Training schedule | One epoch |
| Global batch size | 32: 8 training ranks, microbatch 1, gradient accumulation 4 |
| Context limit | 131072 tokens |
| Checkpoint saving | Every 500 optimizer steps and at the end of each epoch |
| Checkpoint retention | Latest 2 complete checkpoints |
| W&B | Disabled |

Edit the YAML or append `key=value` options to the launch command. Set
`training.max_seq_length` and `inference.vllm.extra_args.max_model_len` together
when adjusting context length. Tokenized inputs must be shorter than that limit.

Size the Mooncake buffers for the available host memory and workload. Defaults
are `mooncake.global_segment_size=512GB` per target producer,
`mooncake.local_buffer_size=16GB`, and `mooncake.host_buffer_size=10GB`.
`CACHE_DIR` selects the dataset cache directory. `SENSESPEC_TRITON_CACHE_DIR`
can select a node-local directory for compiled kernels.

## Logs and checkpoints

The launcher prints the log path. Follow that file from another terminal:

```bash
tail -f /absolute/path/to/the/printed.log
```

The resolved configuration is saved to `$OUTPUT_DIR/config.yaml`, and checkpoints
are saved under `$OUTPUT_DIR/checkpoints/`. To also collect logs from each actor,
set `SENSESPEC_LOG_DIR` to an absolute directory writable from all nodes.

## Resume training

Use the same cluster, model, dataset, and training settings. Restore into a new
output directory accessible from every node:

```bash
previous_output_dir="/absolute/path/to/previous-run"
LOAD_PATH="$previous_output_dir/checkpoints" \
OUTPUT_DIR="${previous_output_dir}-resumed" \
LOG_DIR="${previous_output_dir}-resumed/logs" \
bash examples/SenseNova-Loop-9B-Preview-dflah2-3-node/run.sh
```

The checkpoint tracker selects the latest saved step. Set `CKPT_STEP` to select
another complete checkpoint. The training step limit must exceed the restored step.

See [Checkpoint Conversion](../../README.md#checkpoint-conversion) for export.
Use `configs/draft_models/sensenova_loop_dflash2.json` with `--config`.
For optional image input, follow the [data guide](../data/README.md) and ensure
both inference nodes can read every image.
