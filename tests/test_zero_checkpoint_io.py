"""Checkpoint save/load tests for ZeRO optimizer shards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.distributed.zero import DistributedOptimizerConfig
from src.distributed.zero import MegatronZeroOptimizer


class _TinyModel(nn.Module):
    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.linear1 = nn.Linear(6, hidden)
        self.linear2 = nn.Linear(hidden, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(torch.relu(self.linear1(x)))


def _build_optimizer(model: nn.Module) -> MegatronZeroOptimizer:
    return MegatronZeroOptimizer(
        model=model,
        config=DistributedOptimizerConfig(
            use_distributed_optimizer=True,
            data_parallel_sharding_strategy="optim",
            learning_rate=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        ),
        data_parallel_group=None,
        expert_data_parallel_group=None,
        expert_param_ids=set(),
    )


def test_checkpoint_shard_roundtrip(tmp_path: Path) -> None:
    """Sharded optimizer state should save and load with identical tensor contents."""
    torch.manual_seed(11)
    model = _TinyModel()
    optimizer = _build_optimizer(model)

    for _ in range(3):
        x = torch.randn(4, 6)
        y = torch.randn(4, 4)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step_with_ready_grads()

    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    nonparam = optimizer.state_dict()
    torch.save(nonparam, checkpoint_dir / "optimizer_nonparam.pt")
    optimizer.save_parameter_state(str(checkpoint_dir), rank=0, world_size=1)

    manifest = optimizer.build_checkpoint_manifest(rank=0, world_size=1)
    with open(checkpoint_dir / "optimizer_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)

    model_loaded = _TinyModel()
    optimizer_loaded = _build_optimizer(model_loaded)
    optimizer_loaded.load_state_dict(torch.load(checkpoint_dir / "optimizer_nonparam.pt"))
    optimizer_loaded.load_parameter_state(str(checkpoint_dir), rank=0, world_size=1)

    state_a = optimizer.get_parameter_state_dp_zero(include_tensors=True)
    state_b = optimizer_loaded.get_parameter_state_dp_zero(include_tensors=True)
    for name, record in state_a["parameters"].items():
        loaded_record = state_b["parameters"][name]
        assert torch.allclose(record["master_param"], loaded_record["master_param"])
        assert torch.allclose(record["exp_avg"], loaded_record["exp_avg"])
        assert torch.allclose(record["exp_avg_sq"], loaded_record["exp_avg_sq"])


def test_checkpoint_signature_mismatch_raises(tmp_path: Path) -> None:
    """Loading shard state into mismatched model should raise explicit error."""
    model = _TinyModel(hidden=8)
    optimizer = _build_optimizer(model)
    checkpoint_dir = tmp_path / "ckpt_bad"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    optimizer.save_parameter_state(str(checkpoint_dir), rank=0, world_size=1)

    model_mismatch = _TinyModel(hidden=10)
    optimizer_mismatch = _build_optimizer(model_mismatch)
    with pytest.raises(ValueError, match="signature mismatch"):
        optimizer_mismatch.load_parameter_state(str(checkpoint_dir), rank=0, world_size=1)

