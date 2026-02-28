"""Argument validation helpers for the train_4d runtime path."""

from __future__ import annotations

import argparse
from typing import Optional


def parse_pp_layer_splits(raw_splits: Optional[str]) -> Optional[tuple[int, ...]]:
    """Parse optional comma-separated PP layer boundaries into a tuple."""
    if raw_splits is None:
        return None

    text = raw_splits.strip()
    if not text:
        return None

    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return None

    try:
        splits = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("pp_layer_splits must be a comma-separated integer list") from exc

    return splits


def _validate_parallel_factorization(args: argparse.Namespace, world_size: int) -> None:
    """Validate world-size factorization across configured parallel dimensions."""
    if (
        args.tensor_model_parallel_size < 1
        or args.pipeline_model_parallel_size < 1
        or args.expert_model_parallel_size < 1
        or args.context_parallel_size < 1
    ):
        raise ValueError("parallel sizes must be >= 1")

    model_parallel = (
        args.tensor_model_parallel_size
        * args.pipeline_model_parallel_size
        * args.expert_model_parallel_size
        * args.context_parallel_size
    )
    if model_parallel > world_size:
        raise ValueError("tensor*pipeline*expert*context cannot exceed world_size")
    if world_size % model_parallel != 0:
        raise ValueError(
            "world_size "
            f"({world_size}) must be divisible by tensor*pipeline*expert*context ({model_parallel})"
        )


def _validate_model_dimension_compatibility(args: argparse.Namespace) -> None:
    """Validate model dimensions against tensor/expert parallel requirements."""
    if args.intermediate_size % max(1, args.tensor_model_parallel_size) != 0:
        raise ValueError("intermediate_size must be divisible by tensor_model_parallel_size")

    if args.num_heads % max(1, args.tensor_model_parallel_size) != 0:
        raise ValueError("num_heads must be divisible by tensor_model_parallel_size")

    if args.num_experts % args.expert_model_parallel_size != 0:
        raise ValueError("num_experts must be divisible by expert_model_parallel_size")

    if args.seq_len < 2:
        raise ValueError("seq_len must be >= 2 for causal LM loss")
    if args.tensor_model_parallel_size > 1 and args.seq_len % args.tensor_model_parallel_size != 0:
        raise ValueError(
            "seq_len must be divisible by tensor_model_parallel_size for sequence-parallel MoE"
        )


def _validate_tutorial_guards(args: argparse.Namespace) -> None:
    """Enforce tutorial-specific constraints used for correctness and clarity."""
    if args.context_parallel_size != 1:
        raise ValueError("context_parallel_size > 1 is not yet implemented in this tutorial")

    if args.expert_tensor_parallel_size != 1:
        raise ValueError("expert_tensor_parallel_size must be 1 in this tutorial")

    if args.tensor_model_parallel_size > 1 and args.expert_model_parallel_size > 1:
        raise ValueError(
            "This tutorial disallows tensor_model_parallel_size>1 with "
            "expert_model_parallel_size>1 when expert_tensor_parallel_size=1. "
            "That setup replicates expert shards across TP lanes. "
            "Use tensor_model_parallel_size=1 for MoE EP runs."
        )

    if args.dropout > 0.0 and (
        args.tensor_model_parallel_size > 1 or args.expert_model_parallel_size > 1
    ):
        raise ValueError(
            "dropout must be 0.0 when tensor_model_parallel_size>1 or "
            "expert_model_parallel_size>1 in this tutorial. This keeps replicated "
            "non-sharded parameter paths deterministic without extra model-parallel "
            "gradient synchronization."
        )


def _validate_zero_optimizer_settings(args: argparse.Namespace) -> None:
    """Validate Megatron-style distributed optimizer flags."""
    valid_strategies = {"no_shard", "optim", "optim_grads", "optim_grads_params"}
    strategy = args.data_parallel_sharding_strategy
    zero_debug = bool(getattr(args, "zero_debug", False))
    zero_debug_max_steps = int(getattr(args, "zero_debug_max_steps", 1))
    zero_debug_max_params = int(getattr(args, "zero_debug_max_params", 8))
    if strategy not in valid_strategies:
        raise ValueError(
            "data_parallel_sharding_strategy must be one of "
            "no_shard, optim, optim_grads, optim_grads_params"
        )

    if strategy == "optim_grads_params":
        raise ValueError(
            "optim_grads_params (ZeRO-3) is out of scope in this tutorial. "
            "Use no_shard, optim, or optim_grads."
        )

    if args.num_distributed_optimizer_instances != 1:
        raise ValueError(
            "num_distributed_optimizer_instances must be 1 in this tutorial implementation"
        )

    if strategy != "no_shard" and not args.use_distributed_optimizer:
        raise ValueError(
            "use_distributed_optimizer must be enabled when sharding strategy is not no_shard"
        )

    if args.use_distributed_optimizer and strategy == "no_shard":
        raise ValueError(
            "When use_distributed_optimizer is set, select sharding strategy optim or optim_grads"
        )

    if zero_debug and not args.use_distributed_optimizer:
        raise ValueError("zero_debug requires use_distributed_optimizer=True")

    if zero_debug_max_steps < 1:
        raise ValueError("zero_debug_max_steps must be >= 1")

    if zero_debug_max_params < 1:
        raise ValueError("zero_debug_max_params must be >= 1")


def _validate_pipeline_batching(
    args: argparse.Namespace,
    pp_layer_splits: Optional[tuple[int, ...]],
) -> None:
    """Validate microbatching and optional pipeline layer split boundaries."""
    if args.num_microbatches < 1:
        raise ValueError("num_microbatches must be >= 1")

    if args.batch_size % args.num_microbatches != 0:
        raise ValueError(
            "batch_size must be divisible by num_microbatches for fixed-shape 1F1B"
        )

    if pp_layer_splits is not None:
        if len(pp_layer_splits) != args.pipeline_model_parallel_size + 1:
            raise ValueError("pp_layer_splits length must equal pipeline_model_parallel_size + 1")
        if pp_layer_splits[0] != 0:
            raise ValueError("pp_layer_splits must start with 0")
        if pp_layer_splits[-1] != args.num_layers:
            raise ValueError("pp_layer_splits must end at num_layers")
        for idx in range(1, len(pp_layer_splits)):
            if pp_layer_splits[idx] <= pp_layer_splits[idx - 1]:
                raise ValueError("pp_layer_splits must be strictly increasing")


def _validate_precision_args(args: argparse.Namespace) -> None:
    """
    Precision-flag validation is centralized in runtime precision resolution.

    This validator intentionally remains topology-only to avoid duplicated flag
    checks and error-message drift across runtime paths.
    """
    del args


def validate_args(
    args: argparse.Namespace,
    world_size: int,
    pp_layer_splits: Optional[tuple[int, ...]],
) -> None:
    """Validate args that depend on global world size and topology choices."""
    _validate_parallel_factorization(args, world_size)
    _validate_model_dimension_compatibility(args)
    _validate_tutorial_guards(args)
    _validate_zero_optimizer_settings(args)
    _validate_pipeline_batching(args, pp_layer_splits)
    _validate_precision_args(args)
