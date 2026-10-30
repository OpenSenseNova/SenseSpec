# Loop/mHC configurations

Training settings live in the example directories:

| Recipe | Training config | Draft config |
| --- | --- | --- |
| SenseNova Loop DFlash2, one node | [config.yaml](../examples/SenseNova-Loop-9B-Preview-dflah2/config.yaml) | [DFlash2](draft_models/sensenova_loop_dflash2.json) |
| SenseNova Loop DFlash2, three nodes | [config.yaml](../examples/SenseNova-Loop-9B-Preview-dflah2-3-node/config.yaml) | [DFlash2](draft_models/sensenova_loop_dflash2.json) |
| SenseNova Loop DSpark, one node | [config.yaml](../examples/SenseNova-Loop-9B-Preview-dspark/config.yaml) | [DSpark](draft_models/sensenova_loop_dspark.json) |

Defaults come from the Python configuration dataclasses in
`sensespec/config/`; no default YAML file is required. The loader merges those
defaults, the selected recipe YAML, and command-line `key=value` overrides.

```bash
bash examples/SenseNova-Loop-9B-Preview-dspark/run.sh --dry-run training.num_train_steps=5
```

The target backend is vLLM with V2 direct capture. SenseNova Loop recipes capture
execution layers `[3, 27, 51, 53, 54]` from the 56-layer schedule. DSpark consumes the
independent post-norm final state. DFlash2 uses CE and selector loss.
