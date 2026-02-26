"""Unit tests for ZeRO partition metadata and signature stability."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.distributed.zero import DistributedOptimizerConfig
from src.distributed.zero import MegatronZeroOptimizer
from src.distributed.zero import _compute_shard


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(4, 4, bias=True)
        self.expert = nn.Linear(4, 2, bias=False)


def test_compute_shard_cover_and_non_overlap() -> None:
    """Shard ranges should cover [0, numel) without overlap."""
    numel = 11
    size = 4
    ranges = [_compute_shard(numel=numel, rank=rank, size=size)[:2] for rank in range(size)]

    covered = []
    for start, end in ranges:
        covered.extend(range(start, end))

    assert sorted(set(covered)) == list(range(numel))
    assert len(covered) == len(set(covered))


def test_signature_hash_stable_for_same_model() -> None:
    """Parameter signature hash must be deterministic for same module structure."""
    model_a = _TinyModel()
    model_b = _TinyModel()

    cfg = DistributedOptimizerConfig(
        use_distributed_optimizer=True,
        data_parallel_sharding_strategy="optim",
        learning_rate=1e-3,
    )
    optim_a = MegatronZeroOptimizer(
        model=model_a,
        config=cfg,
        data_parallel_group=None,
        expert_data_parallel_group=None,
        expert_param_ids={id(model_a.expert.weight)},
    )
    optim_b = MegatronZeroOptimizer(
        model=model_b,
        config=cfg,
        data_parallel_group=None,
        expert_data_parallel_group=None,
        expert_param_ids={id(model_b.expert.weight)},
    )

    assert optim_a.parameter_signature_hash() == optim_b.parameter_signature_hash()


def test_local_shard_metadata_matches_full_tensor_on_single_rank() -> None:
    """On world=1 each shard range should map to full parameter."""
    model = _TinyModel()
    cfg = DistributedOptimizerConfig(
        use_distributed_optimizer=True,
        data_parallel_sharding_strategy="optim_grads",
        learning_rate=1e-3,
    )
    optim = MegatronZeroOptimizer(
        model=model,
        config=cfg,
        data_parallel_group=None,
        expert_data_parallel_group=None,
        expert_param_ids={id(model.expert.weight)},
    )

    payload = optim.get_parameter_state_dp_zero(include_tensors=False)
    params = payload["parameters"]
    assert isinstance(params, dict)
    for name, param in model.named_parameters():
        record = params[name]
        assert int(record["shard_start"]) == 0
        assert int(record["shard_end"]) == param.numel()
