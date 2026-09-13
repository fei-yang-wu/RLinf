# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native GR00T N1.7 supervised loss and dataset adapters for RLinf SFT."""

from collections.abc import Mapping
from typing import Any

import torch
from gr00t.configs.base_config import get_default_config
from gr00t.configs.data.data_config import SingleDatasetConfig
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from rlinf.models.embodiment.base_policy import ForwardType


class Gr00tN17ForSFT(Gr00tN1d7):
    """Keep native checkpoint keys while exposing RLinf's supervised forward."""

    _no_split_modules = [
        "Qwen3VLTextDecoderLayer",
        "Qwen3VLVisionBlock",
        "BasicTransformerBlock",
    ]

    def forward(self, forward_type=ForwardType.SFT, data=None, **kwargs):
        if forward_type != ForwardType.SFT:
            raise ValueError("This GR00T adapter supports supervised training only")
        if not isinstance(data, Mapping) or "inputs" not in data:
            raise ValueError("GR00T SFT expects the native collator's inputs mapping")
        return super().forward(data["inputs"])


def get_sft_model(cfg: DictConfig, torch_dtype: torch.dtype) -> Gr00tN17ForSFT:
    """Load native weights without replacing the action head with the RL head."""
    settings = cfg.gr00t_sft
    loading = {"trust_remote_code": True}
    if settings.get("backbone_revision"):
        loading["revision"] = settings.backbone_revision
    model, diagnostics = Gr00tN17ForSFT.from_pretrained(
        cfg.model_path,
        torch_dtype=torch_dtype,
        tune_llm=settings.get("tune_llm", False),
        tune_visual=settings.get("tune_visual", False),
        tune_projector=settings.get("tune_projector", True),
        tune_diffusion_model=settings.get("tune_diffusion_model", True),
        tune_vlln=settings.get("tune_vlln", True),
        load_bf16=True,
        state_dropout_prob=settings.get("state_dropout_prob", 0.0),
        transformers_loading_kwargs=loading,
        output_loading_info=True,
    )
    problems = {
        key: diagnostics.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if diagnostics.get(key)
    }
    if problems:
        raise ValueError(f"GR00T checkpoint mismatch: {problems}")
    # FSDP1 requires each flattened group to have a common parameter dtype.
    return model.to(torch_dtype)


class Gr00tSftLoader:
    """A declared number of batches per epoch over the native sharded stream.

    GR00T partitions shards itself across distributed ranks. Do not add a
    DistributedSampler. Empty rank streams fail rather than hanging an update.
    Exact iterator resume is not supported by the native stream.
    """

    def __init__(self, dataset, collator, batch_size: int, batches_per_epoch: int):
        if batch_size < 1 or batches_per_epoch < 1:
            raise ValueError("batch_size and batches_per_epoch must be positive")
        self.dataset = dataset
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        self.batches_per_epoch = batches_per_epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        iterator = iter(self.loader)
        for _ in range(self.batches_per_epoch):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self.loader)
                try:
                    batch = next(iterator)
                except StopIteration as error:
                    raise RuntimeError(
                        "GR00T rank has no complete batch; increase shards/data"
                    ) from error
            yield batch


def build_sft_dataloader(
    cfg: DictConfig, world_size: int, data_paths: Any, eval_dataset: bool = False
):
    """Build the native processor and dataset using RLinf batch/epoch settings."""
    if eval_dataset:
        raise ValueError("GR00T sharded SFT does not support validation datasets yet")
    if cfg.runner.get("resume_dir"):
        raise ValueError(
            "GR00T native stream cannot resume an exact RLinf data iterator yet"
        )
    settings = cfg.actor.model.gr00t_sft
    processor = Gr00tN1d7Processor.from_pretrained(settings.processor_path)
    tag = EmbodimentTag(cfg.actor.model.embodiment_tag)
    modalities = processor.modality_configs[tag.value]
    MODALITY_CONFIGS[tag.value] = modalities
    config = get_default_config()
    config.data.modality_configs = {tag.value: modalities}
    paths = [data_paths] if isinstance(data_paths, str) else list(data_paths)
    config.data.datasets = [
        SingleDatasetConfig(dataset_paths=paths, embodiment_tag=tag.value)
    ]
    config.data.shard_size = settings.get("shard_size", 1024)
    config.data.num_shards_per_epoch = settings.get("num_shards_per_epoch", world_size)
    if config.data.num_shards_per_epoch < world_size:
        raise ValueError("num_shards_per_epoch must be at least the actor world size")
    config.data.episode_sampling_rate = settings.get("episode_sampling_rate", 1.0)
    config.data.seed = cfg.actor.get("seed", 42)
    dataset, _ = DatasetFactory(config).build(processor)
    loader = Gr00tSftLoader(
        dataset,
        processor.collator,
        cfg.actor.micro_batch_size,
        settings.batches_per_epoch,
    )
    return loader, processor
