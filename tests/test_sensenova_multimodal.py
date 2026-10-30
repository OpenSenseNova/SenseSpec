"""SenseNova image payload and post-expansion supervision contracts."""

import base64
from pathlib import Path
from queue import SimpleQueue
from types import SimpleNamespace

import pytest

from sensespec.config.train_config import config_to_flat_args, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples/SenseNova-Loop-9B-Preview-dspark/config.yaml"


@pytest.mark.parametrize("example", ["dspark", "dflah2", "dflah2-3-node"])
@pytest.mark.parametrize("enable_images", [False, True])
def test_training_examples_default_to_text_and_allow_image_opt_in(example, enable_images):
    overrides = (
        [
            "inference.vllm.enable_multimodal=true",
            "inference.vllm.extra_args.language_model_only=false",
        ]
        if enable_images
        else None
    )
    config = load_config(
        str(ROOT / f"examples/SenseNova-Loop-9B-Preview-{example}/config.yaml"),
        cli_args=overrides,
    )
    args = config_to_flat_args(config)
    assert args.vllm_enable_multimodal is enable_images
    assert args.vllm_extra_args["language_model_only"] is (not enable_images)
    assert args.renderer == "sensenova-loop"
    assert args.dynamic_loss_mask
    assert not args.defer_tokenization


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ("inference.vllm.extra_args.language_model_only=true", "conflicts"),
        ("dataset.renderer=null", "recomputes loss masks"),
        ("dataset.renderer=kimi-k3", "recomputes loss masks"),
    ],
)
def test_multimodal_prefill_rejects_incompatible_configuration(override, error):
    with pytest.raises(ValueError, match=error):
        load_config(
            str(CONFIG),
            cli_args=[
                "inference.vllm.enable_multimodal=true",
                "inference.vllm.extra_args.language_model_only=false",
                override,
            ],
        )


def test_image_payload_supports_paths_urls_and_loaded_images(tmp_path):
    from PIL import Image

    from sensespec.inference.engine.vllm_engine import VllmEngine

    path = tmp_path / "image with spaces.png"
    source = Image.new("RGB", (8, 8), (42, 80, 120))
    source.save(path)
    data_url = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
    payload = VllmEngine._to_vllm_multi_modal_data(
        {"images": [str(path), path.as_uri(), data_url, source, None]}
    )
    assert len(payload["image"]) == 4
    assert all(image.size == (8, 8) for image in payload["image"])
    assert all(image.getpixel((0, 0)) == (42, 80, 120) for image in payload["image"])
    assert VllmEngine._to_vllm_multi_modal_data({"images": [None]}) is None


@pytest.mark.parametrize("mismatched_tokens", [False, True])
def test_offline_capture_assembles_layers_before_replay(tmp_path, mismatched_tokens):
    import torch

    from sensespec.offline.dataset import OfflineDataset
    from sensespec.offline.saving_actor import OfflineSavingActor
    from sensespec.training.data_fetcher import MooncakeDataset, TrainSample

    class Store:
        def __init__(self):
            self.removed = []
            self.buffer = torch.empty(4, 2, dtype=torch.bfloat16)

        def get(self, *, key, **kwargs):
            index = int(key[-1])
            self.buffer.fill_(index)
            ids = torch.arange(4) + int(mismatched_tokens and index == 1)
            return SimpleNamespace(
                to_tensor_dict=lambda: {"hidden_states": self.buffer, "input_ids": ids}
            )

        def remove_eagle3_tensors(self, key, **kwargs):
            self.removed.append(key)

    actor_type = OfflineSavingActor.__ray_metadata__.modified_class
    saver = actor_type.__new__(actor_type)
    saver.dataset = OfflineDataset(tmp_path, create=True, last_hidden_states_prenorm=False)
    saver.store = Store()
    saver.reader = MooncakeDataset(None, saver.store, torch.device("cpu"))
    queue = SimpleQueue()
    saver.queues = {"train": queue}
    manifest = [
        {"mooncake_key": f"layer{i}", "role": role}
        for i, role in enumerate(["hidden_states", "hidden_states", "last_hidden_states"])
    ]
    queue.put(
        TrainSample(
            mooncake_key="assembled",
            data_id="image",
            tensor_shapes={
                "input_ids": (4,),
                "hidden_states": (4, 4),
                "last_hidden_states": (4, 2),
            },
            packed_loss_mask="2,2",
            metadata={
                "vllm_pp_complete": True,
                "vllm_pp_layer_manifest": manifest,
                "source": "image",
            },
        )
    )
    if mismatched_tokens:
        with pytest.raises(RuntimeError, match="input_ids mismatch"):
            saver.save_from_queue("train")
        assert saver.dataset.count("train") == 0
        return

    assert saver.save_from_queue("train") == 1
    saved = saver.dataset.load("image")
    assert saved["hidden_states"].tolist() == [[0, 0, 1, 1]] * 4
    assert saved["last_hidden_states"].tolist() == [[2, 2]] * 4
    assert saved["input_ids"].tolist() == list(range(4))
    assert saved["packed_loss_mask"] == "2,2"
    assert saved["metadata"] == {"source": "image"}
    assert saver.store.removed == ["layer0", "layer1", "layer2"]
