"""Queue consumer that saves inference outputs instead of training on them."""

from __future__ import annotations

import dataclasses

import ray
import torch

from sensespec.offline.dataset import OfflineDataset
from sensespec.ray.ray_actor import RayActor
from sensespec.training.data_fetcher import MooncakeDataset
from sensespec.transfer.mooncake.eagle_store import EagleMooncakeStore


@ray.remote(num_cpus=1, num_gpus=0)
class OfflineSavingActor:
    """Consume the normal training queues and persist their tensors."""

    def __init__(self, output_dir, mooncake_config) -> None:
        self.dataset = OfflineDataset(output_dir)
        config = dataclasses.replace(
            mooncake_config,
            local_hostname=RayActor.get_node_ip(),
            global_segment_size=0,
            async_put_pool_size=0,
            enable_gpu_direct=False,
        )
        self.store = EagleMooncakeStore(config)
        self.store.setup(device=torch.device("cpu"))
        self.reader = MooncakeDataset(None, self.store, torch.device("cpu"))
        self.queues = {}

    def set_queues(self, train_queue, eval_queue) -> None:
        self.queues = {"train": train_queue, "eval": eval_queue}

    def save_from_queue(self, split: str, count: int = 1) -> int:
        """Save exactly ``count`` items and return the number newly written."""
        written = 0
        for _ in range(count):
            sample = self.queues[split].get()
            if not sample.data_id:
                raise ValueError("Offline saving requires TrainSample.data_id")
            # Direct capture also publishes per-layer fragments with PP=1.
            # Share live training's assembly, token checks, and cleanup.
            tensors = self.reader._load_from_mooncake(sample)
            metadata = dict(sample.metadata or {})
            # Replay publishes assembled tensors under a new base key.
            metadata.pop("vllm_pp_layer_manifest", None)
            metadata.pop("vllm_pp_complete", None)
            written += self.dataset.append(
                split,
                data_id=sample.data_id,
                tensors=tensors,
                packed_loss_mask=sample.packed_loss_mask,
                metadata=metadata,
            )
        return written

    def counts(self) -> dict[str, int]:
        return {split: self.dataset.count(split) for split in ("train", "eval")}

    def close(self) -> None:
        self.store.close()
