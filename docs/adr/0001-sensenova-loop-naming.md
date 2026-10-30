# Use SenseNovaLoop for the custom Loop architecture

Use `SenseNovaLoop*`, `sensenova_loop*`, and `VLLM_SENSENOVA_LOOP_MHC_*` for custom Loop and mHC components throughout SenseSpec and its vllm4spec dependency, including maintained examples, tests, benchmarks, and documentation. Keep the names of reused upstream Qwen3.5 components and independent draft architectures, since those names identify their own architectures and configuration protocols.

Retire the old Python import paths and environment-variable names without compatibility aliases, and update all in-repository callers together; external callers must migrate to the new names. This gives the custom architecture one naming scheme instead of maintaining parallel interfaces.

Accept `SenseNovaLoopForConditionalGeneration` as the sole custom Loop architecture identifier and remove the old architecture registration and recognition paths. The selected `SenseNova-6.8-Loop-9B-Preview-mtp13200` checkpoint already declares that identifier, so its weights and configuration do not need a naming migration.

Preserve historical logs, experiment outputs, validation records, and their recorded source hashes as evidence of the versions that actually ran; future records use the new names. Historical checkpoints declaring the old architecture identifier are no longer supported through that identifier.

Migrate the maintained inference example and its benchmark capture configuration to the selected checkpoint's 56 execution layers and two residual streams. Keep the four-stream Triton performance results explicitly historical; the two-stream model uses the existing PyTorch fallback.

Treat an omitted `loop_settings.use_gate` as disabled, matching the selected checkpoint, while continuing to reject explicitly enabled gates. This compatibility correction belongs with the migration so the newly named entry point accepts its documented checkpoint without editing external weights or configuration.
