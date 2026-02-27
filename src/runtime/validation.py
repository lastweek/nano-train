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
    """Validate mixed-precision CLI flags and dtype-control invariants."""
    mode_flags = [
        bool(getattr(args, "bf16", False)),
        bool(getattr(args, "fp16", False)),
        bool(getattr(args, "fp8", False)),
        bool(getattr(args, "fp4", False)),
    ]
    if sum(1 for enabled in mode_flags if enabled) > 1:
        raise ValueError("At most one of --bf16/--fp16/--fp8/--fp4 may be set")

    valid_dtypes = {"fp32", "bf16", "fp16"}
    dtype_flag_map = {
        "params_dtype": getattr(args, "params_dtype", "fp32"),
        "main_params_dtype": getattr(args, "main_params_dtype", "fp32"),
        "main_grads_dtype": getattr(args, "main_grads_dtype", "fp32"),
        "exp_avg_dtype": getattr(args, "exp_avg_dtype", "fp32"),
        "exp_avg_sq_dtype": getattr(args, "exp_avg_sq_dtype", "fp32"),
    }
    for flag_name, value in dtype_flag_map.items():
        if value is None:
            continue
        if str(value) not in valid_dtypes:
            raise ValueError(f"{flag_name} must be one of: fp32, bf16, fp16")

    if bool(getattr(args, "fp8", False)):
        if str(getattr(args, "fp8_backend", "transformer_engine")) not in (
            "transformer_engine",
            "emulated",
        ):
            raise ValueError("fp8_backend must be transformer_engine or emulated")
        if str(getattr(args, "fp8_format", "e4m3")) not in ("e4m3", "hybrid"):
            raise ValueError("fp8_format must be e4m3 or hybrid")
        if str(getattr(args, "fp8_amax_compute_algo", "most_recent")) not in (
            "most_recent",
            "max",
        ):
            raise ValueError("fp8_amax_compute_algo must be most_recent or max")
        if int(getattr(args, "fp8_amax_history_len", 16)) < 1:
            raise ValueError("fp8_amax_history_len must be >= 1")

    if bool(getattr(args, "fp4", False)):
        if str(getattr(args, "fp4_backend", "emulated")) != "emulated":
            raise ValueError("fp4_backend currently supports only emulated")

    fp8_param = bool(getattr(args, "fp8_param", False))
    fp4_param = bool(getattr(args, "fp4_param", False))
    if fp8_param and fp4_param:
        raise ValueError("At most one of --fp8-param/--fp4-param may be set")

    module_pattern_type = str(getattr(args, "module_pattern_type", "regex"))
    if module_pattern_type not in ("regex", "glob"):
        raise ValueError("module_pattern_type must be regex or glob")

    compute_lowbit_mode = getattr(args, "compute_lowbit_mode", None)
    if compute_lowbit_mode is not None and str(compute_lowbit_mode) not in ("fp8", "fp4"):
        raise ValueError("compute_lowbit_mode must be fp8 or fp4")

    persistent_lowbit_mode = str(getattr(args, "persistent_lowbit_mode", "off"))
    if persistent_lowbit_mode not in ("off", "fp8", "fp4"):
        raise ValueError("persistent_lowbit_mode must be off, fp8, or fp4")

    fp4_param_format = str(getattr(args, "fp4_param_format", "nf4"))
    if fp4_param_format != "nf4":
        raise ValueError("fp4_param_format currently supports only nf4")

    persistent_scale_granularity = str(
        getattr(args, "persistent_scale_granularity", "per_channel")
    )
    if persistent_scale_granularity not in ("per_tensor", "per_channel"):
        raise ValueError("persistent_scale_granularity must be per_tensor or per_channel")

    for attr_name in (
        "compute_lowbit_include",
        "compute_lowbit_exclude",
        "persistent_lowbit_include",
        "persistent_lowbit_exclude",
    ):
        value = getattr(args, attr_name, None)
        if value is None:
            continue
        if isinstance(value, str):
            patterns = [value]
        elif isinstance(value, (list, tuple)):
            patterns = [str(item) for item in value]
        else:
            raise ValueError(f"{attr_name} must be a string or list of strings")
        if any(not pattern.strip() for pattern in patterns):
            raise ValueError(f"{attr_name} entries must be non-empty")

    if float(getattr(args, "loss_scale_init", 65536.0)) <= 0.0:
        raise ValueError("loss_scale_init must be > 0")
    if float(getattr(args, "loss_scale_growth_factor", 2.0)) <= 1.0:
        raise ValueError("loss_scale_growth_factor must be > 1")
    if not (0.0 < float(getattr(args, "loss_scale_backoff_factor", 0.5)) < 1.0):
        raise ValueError("loss_scale_backoff_factor must be in (0, 1)")
    if int(getattr(args, "loss_scale_growth_interval", 2000)) < 1:
        raise ValueError("loss_scale_growth_interval must be >= 1")
    if float(getattr(args, "loss_scale_min", 1.0)) <= 0.0:
        raise ValueError("loss_scale_min must be > 0")
    if float(getattr(args, "loss_scale_max", 16777216.0)) < float(
        getattr(args, "loss_scale_min", 1.0)
    ):
        raise ValueError("loss_scale_max must be >= loss_scale_min")


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
