"""Logic tests for examples/train_4d.py argument handling and gradient sync rules."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn as nn


def _load_train_4p_module() -> ModuleType:
    repo_root = Path(__file__).parent.parent
    module_path = repo_root / "examples" / "train_4d.py"
    module_name = "train_4p_example_test_module"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load examples/train_4d.py for tests")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


EP_MODULE = _load_train_4p_module()


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.non_expert = nn.Parameter(torch.ones(1))
        self.expert = nn.Parameter(torch.ones(1))


def _build_args_for_validate(
    *,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 2,
    context_parallel_size: int = 1,
    expert_tensor_parallel_size: int = 1,
    batch_size: int = 8,
    intermediate_size: int = 128,
    num_heads: int = 8,
    num_experts: int = 8,
    num_layers: int = 5,
    seq_len: int = 16,
    num_microbatches: int = 1,
    dropout: float = 0.0,
    use_distributed_optimizer: bool = False,
    data_parallel_sharding_strategy: str = "no_shard",
    num_distributed_optimizer_instances: int = 1,
) -> argparse.Namespace:
    return argparse.Namespace(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        context_parallel_size=context_parallel_size,
        expert_tensor_parallel_size=expert_tensor_parallel_size,
        batch_size=batch_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_experts=num_experts,
        num_layers=num_layers,
        seq_len=seq_len,
        num_microbatches=num_microbatches,
        dropout=dropout,
        use_distributed_optimizer=use_distributed_optimizer,
        data_parallel_sharding_strategy=data_parallel_sharding_strategy,
        num_distributed_optimizer_instances=num_distributed_optimizer_instances,
    )


def test_parse_args_requires_canonical_parallel_flags(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_4d.py"])
    with pytest.raises(SystemExit):
        EP_MODULE.parse_args()


def test_parse_args_accepts_canonical_parallel_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_4d.py",
            "--tensor-model-parallel-size",
            "1",
            "--pipeline-model-parallel-size",
            "2",
            "--expert-model-parallel-size",
            "4",
            "--use-distributed-optimizer",
            "--data-parallel-sharding-strategy",
            "optim",
        ],
    )
    args = EP_MODULE.parse_args()
    assert args.tensor_model_parallel_size == 1
    assert args.pipeline_model_parallel_size == 2
    assert args.expert_model_parallel_size == 4
    assert args.expert_tensor_parallel_size == 1
    assert args.use_distributed_optimizer is True
    assert args.data_parallel_sharding_strategy == "optim"


def test_parse_pp_layer_splits_parses_csv() -> None:
    splits = EP_MODULE.parse_pp_layer_splits("0,2,5")
    assert splits == (0, 2, 5)


def test_validate_args_accepts_valid_configuration() -> None:
    args = _build_args_for_validate(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=2,
        batch_size=8,
        num_microbatches=2,
        seq_len=16,
    )
    EP_MODULE.validate_args(args, world_size=8, pp_layer_splits=None)


def test_validate_args_world_size_divisibility() -> None:
    args = _build_args_for_validate(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        expert_model_parallel_size=2,
    )
    with pytest.raises(ValueError, match="must be divisible"):
        EP_MODULE.validate_args(args, world_size=10, pp_layer_splits=None)


def test_validate_args_num_experts_divisible_by_expert_mp() -> None:
    args = _build_args_for_validate(expert_model_parallel_size=3, num_experts=8)
    with pytest.raises(ValueError, match="num_experts must be divisible by expert_model_parallel_size"):
        EP_MODULE.validate_args(args, world_size=6, pp_layer_splits=None)


def test_validate_args_disallows_tensor_plus_expert_parallel_combo() -> None:
    args = _build_args_for_validate(
        tensor_model_parallel_size=2,
        expert_model_parallel_size=2,
    )
    with pytest.raises(
        ValueError,
        match="disallows tensor_model_parallel_size>1 with expert_model_parallel_size>1",
    ):
        EP_MODULE.validate_args(args, world_size=8, pp_layer_splits=None)


def test_validate_args_seq_len_divisible_by_tensor_mp() -> None:
    args = _build_args_for_validate(
        tensor_model_parallel_size=2,
        expert_model_parallel_size=1,
        seq_len=15,
    )
    with pytest.raises(
        ValueError,
        match="seq_len must be divisible by tensor_model_parallel_size",
    ):
        EP_MODULE.validate_args(args, world_size=4, pp_layer_splits=None)


def test_validate_args_context_parallel_not_implemented() -> None:
    args = _build_args_for_validate(context_parallel_size=2)
    with pytest.raises(ValueError, match="context_parallel_size > 1 is not yet implemented"):
        EP_MODULE.validate_args(args, world_size=4, pp_layer_splits=None)


def test_validate_args_dropout_requires_single_model_parallel() -> None:
    args = _build_args_for_validate(
        tensor_model_parallel_size=2,
        expert_model_parallel_size=1,
        dropout=0.1,
    )
    with pytest.raises(ValueError, match="dropout must be 0.0"):
        EP_MODULE.validate_args(args, world_size=4, pp_layer_splits=None)


def test_validate_args_sharded_strategy_requires_dist_opt() -> None:
    args = _build_args_for_validate(
        use_distributed_optimizer=False,
        data_parallel_sharding_strategy="optim",
    )
    with pytest.raises(
        ValueError,
        match="use_distributed_optimizer must be enabled",
    ):
        EP_MODULE.validate_args(args, world_size=4, pp_layer_splits=None)


def test_validate_args_rejects_unsupported_stage3() -> None:
    args = _build_args_for_validate(
        use_distributed_optimizer=True,
        data_parallel_sharding_strategy="optim_grads_params",
    )
    with pytest.raises(ValueError, match="ZeRO-3"):
        EP_MODULE.validate_args(args, world_size=4, pp_layer_splits=None)


def test_validate_args_rejects_dist_opt_instances_gt1() -> None:
    args = _build_args_for_validate(
        use_distributed_optimizer=True,
        data_parallel_sharding_strategy="optim",
        num_distributed_optimizer_instances=2,
    )
    with pytest.raises(
        ValueError,
        match="num_distributed_optimizer_instances must be 1",
    ):
        EP_MODULE.validate_args(args, world_size=4, pp_layer_splits=None)


def test_validate_args_rejects_dist_opt_with_no_shard() -> None:
    args = _build_args_for_validate(
        use_distributed_optimizer=True,
        data_parallel_sharding_strategy="no_shard",
    )
    with pytest.raises(
        ValueError,
        match="When use_distributed_optimizer is set",
    ):
        EP_MODULE.validate_args(args, world_size=4, pp_layer_splits=None)


def test_validate_args_pp_layer_splits_validation() -> None:
    args = _build_args_for_validate(pipeline_model_parallel_size=2, num_layers=5)
    splits = EP_MODULE.parse_pp_layer_splits("0,2,4")
    with pytest.raises(ValueError, match="pp_layer_splits must end at num_layers"):
        EP_MODULE.validate_args(args, world_size=4, pp_layer_splits=splits)


def test_synchronize_gradients_uses_dp_and_edp_domains(monkeypatch) -> None:
    model = _TinyModel()
    model.non_expert.grad = torch.ones_like(model.non_expert)
    model.expert.grad = torch.ones_like(model.expert)

    shard_info = EP_MODULE.ParamShardInfo(
        tensor_model_parallel_sharded_param_ids=set(),
        expert_model_parallel_sharded_param_ids={id(model.expert)},
    )

    data_parallel_group = object()
    expert_data_parallel_group = object()
    groups_seen: list[object] = []

    def _fake_all_reduce(tensor: torch.Tensor, op=None, group=None) -> None:
        del tensor, op
        groups_seen.append(group)

    monkeypatch.setattr(EP_MODULE.dist, "all_reduce", _fake_all_reduce)

    EP_MODULE.synchronize_gradients(
        model=model,
        shard_info=shard_info,
        data_parallel_size=2,
        expert_data_parallel_size=4,
        data_parallel_group=data_parallel_group,
        expert_data_parallel_group=expert_data_parallel_group,
    )

    assert groups_seen == [data_parallel_group, expert_data_parallel_group]
    assert torch.allclose(model.non_expert.grad, torch.tensor([0.5]))
    assert torch.allclose(model.expert.grad, torch.tensor([0.25]))
