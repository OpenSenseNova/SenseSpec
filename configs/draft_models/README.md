# Loop/mHC draft model configurations

| File | Algorithm | Consumers |
| --- | --- | --- |
| [sensenova_loop_dspark.json](sensenova_loop_dspark.json) | DSpark | SenseNova Loop 9B Preview recipe |
| [sensenova_loop_dflash2.json](sensenova_loop_dflash2.json) | DFlash2 | SenseNova Loop 9B Preview one-node and three-node recipes |

Both drafts use five layers, hidden size 4096, Q/KV heads 16/4, and head dimension
256. SenseNova uses target execution layers `[3, 27, 51, 53, 54]` from 56 execution
layers. The target has 32 physical layers; capture indices use execution layers.

These JSON files are loaded explicitly by each recipe's `model.draft_model_config`.
