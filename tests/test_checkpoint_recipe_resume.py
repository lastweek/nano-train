"""Checkpoint roundtrip tests for precision recipe metadata."""

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


def _build_optimizer(model: nn.Module, **cfg_overrides) -> MegatronZeroOptimizer:
    base_cfg = {
        "use_distributed_optimizer": True,
        "data_parallel_sharding_strategy": "optim",
        "precision_recipe_name": "deepseek_v3",
        "fp8_rounding": "stochastic",
        "fp8_activation_quant_granularity": "tile_1x128",
        "fp8_weight_quant_granularity": "block_128x128",
        "fp8_comm_quant_enabled": True,
        "fp8_comm_quant_granularity": "block_128x128",
    }
    base_cfg.update(cfg_overrides)

    return MegatronZeroOptimizer(
        model=model,
        config=DistributedOptimizerConfig(**base_cfg),
        data_parallel_group=None,
        expert_data_parallel_group=None,
        expert_param_ids=set(),
    )


def test_checkpoint_recipe_metadata_roundtrip() -> None:
    model = _TinyModel()
    optimizer = _build_optimizer(model)

    nonparam_state = optimizer.state_dict()
    shard_state = optimizer.get_parameter_state_dp_zero(include_tensors=True)

    reloaded = _build_optimizer(_TinyModel())
    reloaded.load_state_dict(nonparam_state)
    reloaded.load_parameter_state_from_dp_zero(shard_state, strict=True)


def test_checkpoint_recipe_mismatch_fails_fast() -> None:
    model = _TinyModel()
    optimizer = _build_optimizer(model)

    nonparam_state = optimizer.state_dict()
    shard_state = optimizer.get_parameter_state_dp_zero(include_tensors=True)

    mismatched = _build_optimizer(
        _TinyModel(),
        fp8_rounding="nearest",
    )

    with pytest.raises(ValueError, match="precision recipe metadata mismatch"):
        mismatched.load_state_dict(nonparam_state)

    with pytest.raises(ValueError, match="precision recipe metadata mismatch"):
        mismatched.load_parameter_state_from_dp_zero(shard_state, strict=True)


def test_checkpoint_manifest_includes_recipe_metadata() -> None:
    optimizer = _build_optimizer(_TinyModel())
    manifest = optimizer.build_checkpoint_manifest(rank=0, world_size=1)
    recipe = manifest.get("precision_recipe")

    assert isinstance(recipe, dict)
    assert recipe["name"] == "deepseek_v3"
    assert recipe["fp8_rounding"] == "stochastic"
    assert recipe["fp8_activation_quant_granularity"] == "tile_1x128"
    assert recipe["fp8_weight_quant_granularity"] == "block_128x128"
    assert recipe["fp8_comm_quant_enabled"] is True
    assert recipe["fp8_comm_quant_granularity"] == "block_128x128"
