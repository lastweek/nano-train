"""ZeRO checkpoint format v2 coverage (hard break from v1)."""

from __future__ import annotations

import torch
import torch.nn as nn

import pytest

from src.distributed.zero import DistributedOptimizerConfig
from src.distributed.zero import MegatronZeroOptimizer


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(6, 8)
        self.linear2 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(torch.relu(self.linear1(x)))


def _build_optimizer(model: nn.Module) -> MegatronZeroOptimizer:
    return MegatronZeroOptimizer(
        model=model,
        config=DistributedOptimizerConfig(
            use_distributed_optimizer=True,
            data_parallel_sharding_strategy="optim",
        ),
        data_parallel_group=None,
        expert_data_parallel_group=None,
        expert_param_ids=set(),
    )


def test_zero_checkpoint_payloads_are_v2() -> None:
    model = _TinyModel()
    optimizer = _build_optimizer(model)

    assert int(optimizer.state_dict()["format_version"]) == 2
    assert int(optimizer.get_parameter_state_dp_zero()["format_version"]) == 2
    assert int(optimizer.build_checkpoint_manifest(rank=0, world_size=1)["format_version"]) == 2


def test_zero_rejects_old_nonparam_checkpoint_format() -> None:
    model = _TinyModel()
    optimizer = _build_optimizer(model)
    state = optimizer.state_dict()
    state["format_version"] = 1

    with pytest.raises(ValueError, match="Unsupported ZeRO nonparam checkpoint format version"):
        optimizer.load_state_dict(state)


def test_zero_rejects_old_shard_checkpoint_format() -> None:
    model = _TinyModel()
    optimizer = _build_optimizer(model)
    shard_state = optimizer.get_parameter_state_dp_zero(include_tensors=True)
    shard_state["format_version"] = 1

    with pytest.raises(ValueError, match="Unsupported ZeRO shard checkpoint format version"):
        optimizer.load_parameter_state_from_dp_zero(shard_state, strict=True)
