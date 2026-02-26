"""Unit tests for runtime validation helpers."""

from __future__ import annotations

import argparse

import pytest

from src.runtime.validation import parse_pp_layer_splits
from src.runtime.validation import validate_args


def _base_args() -> argparse.Namespace:
    return argparse.Namespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=2,
        context_parallel_size=1,
        expert_tensor_parallel_size=1,
        batch_size=8,
        intermediate_size=128,
        num_heads=8,
        num_experts=8,
        num_layers=5,
        seq_len=16,
        num_microbatches=1,
        dropout=0.0,
        use_distributed_optimizer=False,
        data_parallel_sharding_strategy="no_shard",
        num_distributed_optimizer_instances=1,
        zero_debug=False,
        zero_debug_max_steps=1,
        zero_debug_max_params=8,
    )


def test_parse_pp_layer_splits_none_and_csv() -> None:
    assert parse_pp_layer_splits(None) is None
    assert parse_pp_layer_splits("") is None
    assert parse_pp_layer_splits("0,2,5") == (0, 2, 5)


def test_validate_args_accepts_valid_shape() -> None:
    args = _base_args()
    validate_args(args, world_size=8, pp_layer_splits=None)


def test_validate_args_rejects_context_parallel_gt1() -> None:
    args = _base_args()
    args.context_parallel_size = 2
    with pytest.raises(ValueError, match="context_parallel_size > 1 is not yet implemented"):
        validate_args(args, world_size=8, pp_layer_splits=None)


def test_validate_args_rejects_zero3_strategy() -> None:
    args = _base_args()
    args.use_distributed_optimizer = True
    args.data_parallel_sharding_strategy = "optim_grads_params"
    with pytest.raises(ValueError, match="ZeRO-3"):
        validate_args(args, world_size=8, pp_layer_splits=None)
