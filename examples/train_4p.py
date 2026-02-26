#!/usr/bin/env python3
"""
Canonical TP+PP+EP+DP tutorial script with Megatron-style naming.

Primary tutorial modes:
1) single:         world=1, tensor=1, pipeline=1, expert=1, data=1
2) ep_only:        tensor=1, pipeline=1, expert>1, data>=1
3) tp_ep_dp:       tensor>1 and expert>1 and data>1, with pipeline=1
4) tp_pp_ep_dp:    pipeline>1 with optional tensor/expert/data

Model used:
    Small DeepSeek-style LM by default (5 decoder layers, 8 routed experts,
    top-k=2 routing), with optional TP/PP/EP behavior enabled via model context.

Run examples:
    # single rank
    python3 examples/train_4p.py \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --expert-model-parallel-size 1 \
        --max_steps 2

    # EP-only (world=2, tensor=1, pipeline=1, expert=2, data=1)
    python3 examples/launch.py --world-size 2 --backend gloo \
        --script examples/train_4p.py --script-args \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --expert-model-parallel-size 2 \
        --max_steps 2

    # PP-only 1F1B (world=2, tensor=1, pipeline=2, expert=1, data=1)
    python3 examples/launch.py --world-size 2 --backend gloo \
        --script examples/train_4p.py --script-args \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 2 \
        --expert-model-parallel-size 1 \
        --num_microbatches 2 \
        --max_steps 1

    # PP+EP+DP (world=16, tensor=1, pipeline=2, expert=4, data=2)
    python3 examples/launch.py --world-size 16 --backend gloo \
        --script examples/train_4p.py --script-args \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 2 \
        --expert-model-parallel-size 4 \
        --num_microbatches 2 \
        --max_steps 1

    # ZeRO-1 with PP+EP+DP (world=8, tensor=1, pipeline=2, expert=2, data=2)
    python3 examples/launch.py --world-size 8 --backend gloo \
        --script examples/train_4p.py --script-args \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 2 \
        --expert-model-parallel-size 2 \
        --num_microbatches 2 \
        --use-distributed-optimizer \
        --data-parallel-sharding-strategy optim \
        --max_steps 1

    # ZeRO-2 with PP+EP+DP (world=8, tensor=1, pipeline=2, expert=2, data=2)
    python3 examples/launch.py --world-size 8 --backend gloo \
        --script examples/train_4p.py --script-args \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 2 \
        --expert-model-parallel-size 2 \
        --num_microbatches 2 \
        --use-distributed-optimizer \
        --data-parallel-sharding-strategy optim_grads \
        --max_steps 1

    # Add ZeRO debug logs (first step + first 12 params)
    #   --zero-debug --zero-debug-max-steps 1 --zero-debug-max-params 12

    # ZeRO-1 with TP+PP+DP (EP disabled: expert=1), world=8
    python3 examples/launch.py --world-size 8 --backend gloo \
        --script examples/train_4p.py --script-args \
        --tensor-model-parallel-size 2 \
        --pipeline-model-parallel-size 2 \
        --expert-model-parallel-size 1 \
        --num_microbatches 2 \
        --use-distributed-optimizer \
        --data-parallel-sharding-strategy optim \
        --max_steps 1

Note:
    This tutorial currently disallows tensor-model-parallel-size>1 together
    with expert-model-parallel-size>1 (see validate_args guard). So a fully
    "TP+PP+EP+DP with TP>1 and EP>1" ZeRO command is intentionally blocked.
    Use one of the runnable commands above.

Core learning references:
    docs/ep_tp_dp_communication.md
    docs/pp_tp_ep_dp_communication.md
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import field
import os
import sys
from typing import Optional
from typing import Set

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler

# Add parent directory to path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import create_dataloader
from src.distributed.topology import ModelParallelTopology
from src.distributed.topology import initialize_model_parallel
from src.distributed.zero import DistributedOptimizerConfig
from src.distributed.zero import MegatronZeroOptimizer
from src.layers import ColumnParallelLinear
from src.layers import RowParallelLinear
from src.logging import get_logger
from src.logging import setup_logging
from src.models.deepseek import DeepSeekModel
from src.models.deepseek import DeepSeekModelConfig
from src.models.deepseek import DeepSeekParallelContext
from src.models.moe import ExpertParallelMoE
from src.models.moe import LocalRoutedMoE


logger = get_logger("ep")


@dataclass
class ParamShardInfo:
    """Parameter id sets split by sharding domain for sync logic."""

    tensor_model_parallel_sharded_param_ids: Set[int]
    expert_model_parallel_sharded_param_ids: Set[int]


class DummyTokenDataset(Dataset):
    """Deterministic token dataset for causal LM communication demos."""

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(num_samples, seq_len),
            generator=generator,
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[index]}


def build_tiny_deepseek_config(args: argparse.Namespace) -> DeepSeekModelConfig:
    """Build a small DeepSeek config for TP/PP/EP/DP learning runs."""
    return DeepSeekModelConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=max(args.max_position_embeddings, args.seq_len),
        q_lora_rank=args.q_lora_rank,
        kv_lora_rank=args.kv_lora_rank,
        qk_nope_head_dim=args.qk_nope_head_dim,
        qk_rope_head_dim=args.qk_rope_head_dim,
        v_head_dim=args.v_head_dim,
        attention_dropout=args.dropout,
        dropout=args.dropout,
        moe_intermediate_size=args.moe_intermediate_size,
        n_routed_experts=args.num_experts,
        n_shared_experts=args.n_shared_experts,
        num_experts_per_tok=args.top_k,
        first_k_dense_replace=args.first_k_dense_replace,
        moe_layer_freq=args.moe_layer_freq,
        scoring_func="sigmoid",
        n_group=args.n_group,
        topk_group=args.topk_group,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
    )


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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for TP+PP+EP+DP tutorial training."""
    parser = argparse.ArgumentParser(description="Canonical TP+PP+EP+DP tutorial script")
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        required=True,
        help="Tensor model parallel size",
    )
    parser.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        required=True,
        help="Pipeline model parallel size",
    )
    parser.add_argument(
        "--expert-model-parallel-size",
        type=int,
        required=True,
        help="Expert model parallel size",
    )
    parser.add_argument(
        "--context-parallel-size",
        type=int,
        default=1,
        help="Context parallel size (currently must be 1 in this tutorial)",
    )
    parser.add_argument(
        "--expert-tensor-parallel-size",
        type=int,
        default=1,
        help="Expert tensor parallel size (tutorial currently supports only 1)",
    )
    parser.add_argument(
        "--num-layers-per-virtual-pipeline-stage",
        type=int,
        default=None,
        help="Reserved virtual PP setting (not used in this tutorial)",
    )
    parser.add_argument(
        "--use-distributed-optimizer",
        action="store_true",
        help="Enable Megatron-style ZeRO optimizer path",
    )
    parser.add_argument(
        "--data-parallel-sharding-strategy",
        type=str,
        default="no_shard",
        help="Megatron-style DP sharding strategy: no_shard | optim | optim_grads",
    )
    parser.add_argument(
        "--num-distributed-optimizer-instances",
        type=int,
        default=1,
        help="Number of distributed optimizer instances (v1 supports only 1)",
    )
    parser.add_argument(
        "--zero-debug",
        action="store_true",
        help="Enable verbose ZeRO debug logging in distributed optimizer",
    )
    parser.add_argument(
        "--zero-debug-max-steps",
        type=int,
        default=1,
        help="Number of early optimizer steps to emit ZeRO debug counters",
    )
    parser.add_argument(
        "--zero-debug-max-params",
        type=int,
        default=8,
        help="Number of parameter shard mappings to print at ZeRO init",
    )

    parser.add_argument(
        "--num_microbatches",
        type=int,
        default=1,
        help="Pipeline microbatch count for 1F1B schedule",
    )
    parser.add_argument("--pp_layer_splits", type=str, default=None, help="Optional PP layer boundaries, e.g. '0,2,5'")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--intermediate_size", type=int, default=256)
    parser.add_argument("--moe_intermediate_size", type=int, default=128)
    parser.add_argument("--q_lora_rank", type=int, default=64)
    parser.add_argument("--kv_lora_rank", type=int, default=32)
    parser.add_argument("--qk_nope_head_dim", type=int, default=8)
    parser.add_argument("--qk_rope_head_dim", type=int, default=8)
    parser.add_argument("--v_head_dim", type=int, default=8)
    parser.add_argument("--max_position_embeddings", type=int, default=256)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--n_group", type=int, default=4)
    parser.add_argument("--topk_group", type=int, default=2)
    parser.add_argument("--n_shared_experts", type=int, default=1)
    parser.add_argument("--first_k_dense_replace", type=int, default=1)
    parser.add_argument("--moe_layer_freq", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--capacity_factor", type=float, default=1.0)
    parser.add_argument("--aux_loss_coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def setup_process_logging(rank: int) -> None:
    """Configure process-local logging using shared logging utilities."""
    level = "INFO" if rank == 0 else "WARNING"
    setup_logging(log_level=level, use_colors=False)


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


def infer_mode(parallel: ModelParallelTopology) -> str:
    """Infer which learning mode current topology corresponds to."""
    if parallel.world_size == 1:
        return "single"
    if parallel.pipeline_model_parallel_size > 1:
        return "tp_pp_ep_dp"
    if parallel.tensor_model_parallel_size == 1 and parallel.expert_model_parallel_size > 1:
        return "ep_only"
    return "tp_ep_dp"


def log_parallel_topology(
    parallel: ModelParallelTopology,
    mode: str,
) -> None:
    """Log rank topology and process groups from rank 0."""
    if parallel.rank != 0:
        return

    logger.info("=" * 84)
    logger.info("Canonical TP+PP+EP+DP Tutorial")
    logger.info("=" * 84)
    logger.info("Mode: %s", mode)
    logger.info(
        "World Size: %d | Tensor MP: %d | Pipeline MP: %d | Expert MP: %d | Data P: %d | EDP: %d",
        parallel.world_size,
        parallel.tensor_model_parallel_size,
        parallel.pipeline_model_parallel_size,
        parallel.expert_model_parallel_size,
        parallel.data_parallel_size,
        parallel.expert_data_parallel_size,
    )
    logger.info("Context Parallel Size: %d", parallel.context_parallel_size)
    logger.info(
        "Backend: %s | Device: %s",
        dist.get_backend() if parallel.world_size > 1 else "none",
        parallel.device,
    )
    logger.info(
        "Tensor MP groups: %s",
        parallel.tensor_model_parallel_group_table
        if parallel.tensor_model_parallel_group_table
        else "n/a",
    )
    logger.info(
        "Pipeline MP groups: %s",
        parallel.pipeline_model_parallel_group_table
        if parallel.pipeline_model_parallel_group_table
        else "n/a",
    )
    logger.info(
        "Expert MP groups: %s",
        parallel.expert_model_parallel_group_table
        if parallel.expert_model_parallel_group_table
        else "n/a",
    )
    logger.info(
        "Data Parallel groups: %s",
        parallel.data_parallel_group_table if parallel.data_parallel_group_table else "n/a",
    )
    logger.info(
        "Expert Data Parallel groups: %s",
        parallel.expert_data_parallel_group_table
        if parallel.expert_data_parallel_group_table
        else "n/a",
    )
    logger.info("=" * 84)


def _add_axis_sharded_param_ids(
    shard_ids: Set[int],
    module: nn.Module,
    include_bias: bool,
) -> None:
    """Add sharded param ids from a TP linear layer into a target set."""
    shard_ids.add(id(module.weight))
    if include_bias and getattr(module, "bias", None) is not None:
        shard_ids.add(id(module.bias))


def collect_param_shard_info(
    model: nn.Module,
    tensor_model_parallel_group,
) -> ParamShardInfo:
    """
    Classify parameter ids by sharding domain.

    - Tensor-model-parallel sharded: Column/Row linear params on tensor MP group.
    - Expert-model-parallel sharded: local expert weights in ExpertParallelMoE.
    """
    tensor_model_parallel_sharded_param_ids: Set[int] = set()
    expert_model_parallel_sharded_param_ids: Set[int] = set()

    for module in model.modules():
        if isinstance(module, ColumnParallelLinear):
            if module.tp_size <= 1:
                continue
            if module.tp_group is not tensor_model_parallel_group:
                raise ValueError("Unsupported ColumnParallelLinear tp_group in model")
            _add_axis_sharded_param_ids(
                tensor_model_parallel_sharded_param_ids,
                module,
                include_bias=True,
            )
        elif isinstance(module, RowParallelLinear):
            if module.tp_size <= 1:
                continue
            if module.tp_group is not tensor_model_parallel_group:
                raise ValueError("Unsupported RowParallelLinear tp_group in model")
            _add_axis_sharded_param_ids(
                tensor_model_parallel_sharded_param_ids,
                module,
                include_bias=False,
            )
        elif isinstance(module, ExpertParallelMoE):
            for expert in module.experts:
                for param in expert.parameters():
                    expert_model_parallel_sharded_param_ids.add(id(param))

    return ParamShardInfo(
        tensor_model_parallel_sharded_param_ids=tensor_model_parallel_sharded_param_ids,
        expert_model_parallel_sharded_param_ids=expert_model_parallel_sharded_param_ids,
    )


def synchronize_initial_parameters(
    model: nn.Module,
    shard_info: ParamShardInfo,
    parallel: ModelParallelTopology,
) -> None:
    """
    Synchronize parameter initialization across replica domains.

    - Tensor-model sharded params: sync across non-tensor replicas.
    - Expert-model sharded params: sync across expert-data replicas.
    - Replicated stage params: sync across fixed-pipeline stage replicas.
    """
    if parallel.world_size == 1:
        return

    tensor_src_rank = parallel.rank_from_coords(
        data_parallel_rank=0,
        pipeline_model_parallel_rank=parallel.pipeline_model_parallel_rank,
        tensor_model_parallel_rank=parallel.tensor_model_parallel_rank,
        expert_model_parallel_rank=0,
        context_parallel_rank=parallel.context_parallel_rank,
    )
    expert_src_rank = parallel.rank_from_coords(
        data_parallel_rank=0,
        pipeline_model_parallel_rank=parallel.pipeline_model_parallel_rank,
        tensor_model_parallel_rank=0,
        expert_model_parallel_rank=parallel.expert_model_parallel_rank,
        context_parallel_rank=parallel.context_parallel_rank,
    )
    stage_src_rank = parallel.rank_from_coords(
        data_parallel_rank=0,
        pipeline_model_parallel_rank=parallel.pipeline_model_parallel_rank,
        tensor_model_parallel_rank=0,
        expert_model_parallel_rank=0,
        context_parallel_rank=parallel.context_parallel_rank,
    )

    for param in model.parameters():
        param_id = id(param)
        if param_id in shard_info.tensor_model_parallel_sharded_param_ids:
            if parallel.tensor_model_parallel_size > 1:
                dist.broadcast(
                    param.data,
                    src=tensor_src_rank,
                    group=parallel.tensor_model_parallel_replica_group,
                )
        elif param_id in shard_info.expert_model_parallel_sharded_param_ids:
            if parallel.expert_data_parallel_size > 1:
                dist.broadcast(
                    param.data,
                    src=expert_src_rank,
                    group=parallel.expert_model_parallel_replica_group,
                )
        else:
            dist.broadcast(
                param.data,
                src=stage_src_rank,
                group=parallel.pipeline_stage_replica_group,
            )

    for buffer in model.buffers():
        dist.broadcast(
            buffer.data,
            src=stage_src_rank,
            group=parallel.pipeline_stage_replica_group,
        )


def synchronize_gradients(
    model: nn.Module,
    shard_info: ParamShardInfo,
    data_parallel_size: int,
    expert_data_parallel_size: int,
    data_parallel_group,
    expert_data_parallel_group,
) -> None:
    """
    Synchronize gradients for TP+PP+EP+DP.

    Rules:
    - Non-expert params: reduce over data_parallel_group.
    - Expert params: reduce over expert_data_parallel_group.
    - TP collectives stay inside TP layers; PP has no extra gradient all-reduce here.
    """
    def _ensure_grad_for_collective(
        param: nn.Parameter,
        grad: Optional[torch.Tensor],
        needs_collective: bool,
    ) -> Optional[torch.Tensor]:
        """Materialize zero grads so all ranks execute collectives in identical order."""
        if grad is not None:
            return grad
        if needs_collective:
            return torch.zeros_like(param)
        return None

    for param in model.parameters():
        param_id = id(param)
        is_expert_param = param_id in shard_info.expert_model_parallel_sharded_param_ids
        should_reduce_data_parallel = data_parallel_size > 1 and not is_expert_param
        should_reduce_expert_data_parallel = expert_data_parallel_size > 1 and is_expert_param

        grad = param.grad
        had_local_grad = grad is not None

        grad = _ensure_grad_for_collective(
            param=param,
            grad=grad,
            needs_collective=should_reduce_data_parallel or should_reduce_expert_data_parallel,
        )
        if grad is None:
            continue

        if should_reduce_data_parallel:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=data_parallel_group)
            grad.div_(data_parallel_size)
        elif should_reduce_expert_data_parallel:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=expert_data_parallel_group)
            grad.div_(expert_data_parallel_size)

        if had_local_grad or should_reduce_data_parallel or should_reduce_expert_data_parallel:
            param.grad = grad


def create_data_loader(
    dataset: Dataset,
    batch_size: int,
    dp_size: int,
    dp_rank: int,
    seed: int,
):
    """Create a dataloader with DP-aware sharding, reusing src.dataset.create_dataloader."""
    sampler: Optional[DistributedSampler] = None
    if dp_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dp_size,
            rank=dp_rank,
            shuffle=True,
            seed=seed,
        )

    loader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        sampler=sampler,
    )
    return loader, sampler


def gather_moe_metrics(model: nn.Module, device: torch.device) -> tuple[torch.Tensor, float]:
    """Collect auxiliary MoE loss and dropped fraction from all MoE modules."""
    aux_terms: list[torch.Tensor] = []
    drop_terms: list[float] = []

    for module in model.modules():
        if isinstance(module, (ExpertParallelMoE, LocalRoutedMoE)):
            aux_terms.append(module.last_aux_loss)
            drop_terms.append(module.last_dropped_fraction)

    if aux_terms:
        aux_loss = torch.stack([term.reshape(()) for term in aux_terms]).sum()
    else:
        aux_loss = torch.zeros((), device=device)

    avg_drop = float(sum(drop_terms) / len(drop_terms)) if drop_terms else 0.0
    return aux_loss, avg_drop


def _apply_optimizer_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | MegatronZeroOptimizer,
    use_distributed_optimizer: bool,
    shard_info: ParamShardInfo,
    data_parallel_size: int,
    expert_data_parallel_size: int,
    data_parallel_group,
    expert_data_parallel_group,
) -> None:
    """Apply one optimizer step using either ZeRO path or manual grad-sync path."""
    if use_distributed_optimizer:
        if not isinstance(optimizer, MegatronZeroOptimizer):
            raise TypeError("Expected MegatronZeroOptimizer when use_distributed_optimizer=True")
        optimizer.step_with_ready_grads()
        return

    synchronize_gradients(
        model=model,
        shard_info=shard_info,
        data_parallel_size=data_parallel_size,
        expert_data_parallel_size=expert_data_parallel_size,
        data_parallel_group=data_parallel_group,
        expert_data_parallel_group=expert_data_parallel_group,
    )
    optimizer.step()


# Tensor notation used by comments below:
# B: DP-local batch, S: sequence length, H: hidden size
# B_local: per-rank batch used in forward (equals B in this tutorial)
# T = B_local * S (local tokens), K = top_k, A = T * K assignments
# N_recv: assignments received by expert-owner rank after dispatch
# N_back: returned expert outputs received by source rank in combine

def train_step_non_pipeline(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | MegatronZeroOptimizer,
    use_distributed_optimizer: bool,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    data_parallel_size: int,
    expert_data_parallel_size: int,
    aux_loss_coef: float,
    shard_info: ParamShardInfo,
    data_parallel_group,
    expert_data_parallel_group,
) -> tuple[float, float, float, float]:
    """One TP+EP+DP training step without pipeline parallelism."""
    local_input_ids = batch["input_ids"].to(device)

    optimizer.zero_grad(set_to_none=True)

    logits = model(local_input_ids)
    if local_input_ids.size(0) == 0 or local_input_ids.size(1) < 2:
        task_loss = logits.sum() * 0.0
    else:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = local_input_ids[:, 1:].contiguous()
        task_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )

    moe_aux_loss, drop_fraction = gather_moe_metrics(model, device=device)
    loss = task_loss + (aux_loss_coef * moe_aux_loss)
    loss.backward()

    _apply_optimizer_step(
        model=model,
        optimizer=optimizer,
        use_distributed_optimizer=use_distributed_optimizer,
        shard_info=shard_info,
        data_parallel_size=data_parallel_size,
        expert_data_parallel_size=expert_data_parallel_size,
        data_parallel_group=data_parallel_group,
        expert_data_parallel_group=expert_data_parallel_group,
    )
    return (
        float(task_loss.item()),
        float(moe_aux_loss.item()),
        float(loss.item()),
        drop_fraction,
    )


def _activation_tag(microbatch_idx: int) -> int:
    return 10_000 + microbatch_idx


def _grad_tag(microbatch_idx: int) -> int:
    return 20_000 + microbatch_idx


def _label_tag(microbatch_idx: int) -> int:
    return 30_000 + microbatch_idx


@dataclass
class _PipelinePeers:
    """Resolved PP peer ranks for this rank's pipeline chain."""

    prev_rank: Optional[int]
    next_rank: Optional[int]
    first_stage_rank: int
    last_stage_rank: int


@dataclass
class _PipelineStepState:
    """Mutable state containers used across 1F1B forward/backward phases."""

    stage_inputs: list[Optional[torch.Tensor]] = field(default_factory=list)
    stage_outputs: list[torch.Tensor] = field(default_factory=list)
    stage_aux_losses: list[torch.Tensor] = field(default_factory=list)
    stage_labels: list[Optional[torch.Tensor]] = field(default_factory=list)
    label_send_reqs: list[dist.Work] = field(default_factory=list)
    activation_send_reqs: list[dist.Work] = field(default_factory=list)
    activation_send_buffers: list[torch.Tensor] = field(default_factory=list)
    grad_send_reqs: list[dist.Work] = field(default_factory=list)
    grad_send_buffers: list[torch.Tensor] = field(default_factory=list)
    task_loss_sum: float = 0.0
    aux_loss_sum: float = 0.0
    drop_sum: float = 0.0
    drop_count: int = 0


def _resolve_pipeline_peers(parallel: ModelParallelTopology) -> _PipelinePeers:
    """Resolve stage-neighbor ranks and stage endpoints for the local PP chain."""
    prev_pp_rank = parallel.pipeline_model_parallel_rank - 1
    next_pp_rank = parallel.pipeline_model_parallel_rank + 1

    prev_rank = None
    if prev_pp_rank >= 0:
        prev_rank = parallel.rank_from_coords(
            data_parallel_rank=parallel.data_parallel_rank,
            pipeline_model_parallel_rank=prev_pp_rank,
            tensor_model_parallel_rank=parallel.tensor_model_parallel_rank,
            expert_model_parallel_rank=parallel.expert_model_parallel_rank,
            context_parallel_rank=parallel.context_parallel_rank,
        )

    next_rank = None
    if next_pp_rank < parallel.pipeline_model_parallel_size:
        next_rank = parallel.rank_from_coords(
            data_parallel_rank=parallel.data_parallel_rank,
            pipeline_model_parallel_rank=next_pp_rank,
            tensor_model_parallel_rank=parallel.tensor_model_parallel_rank,
            expert_model_parallel_rank=parallel.expert_model_parallel_rank,
            context_parallel_rank=parallel.context_parallel_rank,
        )

    first_stage_rank = parallel.rank_from_coords(
        data_parallel_rank=parallel.data_parallel_rank,
        pipeline_model_parallel_rank=0,
        tensor_model_parallel_rank=parallel.tensor_model_parallel_rank,
        expert_model_parallel_rank=parallel.expert_model_parallel_rank,
        context_parallel_rank=parallel.context_parallel_rank,
    )
    last_stage_rank = parallel.rank_from_coords(
        data_parallel_rank=parallel.data_parallel_rank,
        pipeline_model_parallel_rank=parallel.pipeline_model_parallel_size - 1,
        tensor_model_parallel_rank=parallel.tensor_model_parallel_rank,
        expert_model_parallel_rank=parallel.expert_model_parallel_rank,
        context_parallel_rank=parallel.context_parallel_rank,
    )
    return _PipelinePeers(
        prev_rank=prev_rank,
        next_rank=next_rank,
        first_stage_rank=first_stage_rank,
        last_stage_rank=last_stage_rank,
    )


def _prepare_microbatches(
    batch: Optional[dict[str, torch.Tensor]],
    model: DeepSeekModel,
    expected_local_batch: int,
    num_microbatches: int,
    device: torch.device,
) -> list[Optional[torch.Tensor]]:
    """Prepare local microbatch list for first stage; placeholders elsewhere."""
    if not model.is_first_pp_stage:
        return [None for _ in range(num_microbatches)]

    if batch is None:
        raise ValueError("batch is required on first PP stage")

    local_input_ids = batch["input_ids"].to(device)
    if local_input_ids.size(0) != expected_local_batch:
        raise ValueError(
            "Observed local batch size differs from expected fixed pipeline shape. "
            "Adjust num_samples/batch_size to keep full batches."
        )
    return list(torch.chunk(local_input_ids, num_microbatches, dim=0))


def _pipeline_forward_microbatch(
    microbatch_idx: int,
    model: DeepSeekModel,
    microbatches: list[Optional[torch.Tensor]],
    peers: _PipelinePeers,
    state: _PipelineStepState,
    microbatch_batch_size: int,
    seq_len: int,
    hidden_size: int,
    activation_dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Run one forward microbatch and stage outputs/metadata for later backward."""
    label_tensor: Optional[torch.Tensor] = None
    stage_input: Optional[torch.Tensor] = None

    if model.is_first_pp_stage:
        local_input_ids = microbatches[microbatch_idx]
        if local_input_ids is None:
            raise RuntimeError("Missing first-stage microbatch")

        if not model.is_last_pp_stage:
            state.label_send_reqs.append(
                dist.isend(
                    tensor=local_input_ids.contiguous(),
                    dst=peers.last_stage_rank,
                    tag=_label_tag(microbatch_idx),
                )
            )

        stage_output = model.forward_stage(
            input_ids=local_input_ids,
            hidden_states=None,
            attention_mask=None,
        )
        if model.is_last_pp_stage:
            label_tensor = local_input_ids
    else:
        stage_input = torch.empty(
            (microbatch_batch_size, seq_len, hidden_size),
            device=device,
            dtype=activation_dtype,
            requires_grad=True,
        )
        if peers.prev_rank is None:
            raise RuntimeError("Missing previous PP rank")
        dist.recv(stage_input, src=peers.prev_rank, tag=_activation_tag(microbatch_idx))

        stage_output = model.forward_stage(
            input_ids=None,
            hidden_states=stage_input,
            attention_mask=None,
        )

        if model.is_last_pp_stage:
            label_tensor = torch.empty(
                (microbatch_batch_size, seq_len),
                dtype=torch.long,
                device=device,
            )
            dist.recv(label_tensor, src=peers.first_stage_rank, tag=_label_tag(microbatch_idx))

    local_aux_loss, local_drop = gather_moe_metrics(model, device=device)
    state.aux_loss_sum += float(local_aux_loss.detach().item())
    state.drop_sum += float(local_drop)
    state.drop_count += 1

    if not model.is_last_pp_stage:
        # Forward p2p of activations to next PP stage.
        if peers.next_rank is None:
            raise RuntimeError("Missing next PP rank")
        activation_payload = stage_output.contiguous()
        state.activation_send_buffers.append(activation_payload)
        state.activation_send_reqs.append(
            dist.isend(
                tensor=activation_payload,
                dst=peers.next_rank,
                tag=_activation_tag(microbatch_idx),
            )
        )

    state.stage_inputs.append(stage_input)
    state.stage_outputs.append(stage_output)
    state.stage_aux_losses.append(local_aux_loss)
    state.stage_labels.append(label_tensor)


def _pipeline_backward_microbatch(
    microbatch_idx: int,
    model: DeepSeekModel,
    peers: _PipelinePeers,
    state: _PipelineStepState,
    num_microbatches: int,
    aux_loss_coef: float,
    microbatch_batch_size: int,
    seq_len: int,
    hidden_size: int,
    activation_dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Run one backward microbatch and send dX to previous stage when needed."""
    stage_input = state.stage_inputs.pop(0)
    stage_output = state.stage_outputs.pop(0)
    aux_loss = state.stage_aux_losses.pop(0)
    labels = state.stage_labels.pop(0)

    if model.is_last_pp_stage:
        if labels is None:
            raise RuntimeError("Missing labels on last PP stage")

        if labels.size(0) == 0 or labels.size(1) < 2:
            task_loss = stage_output.sum() * 0.0
        else:
            shift_logits = stage_output[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            task_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )

        state.task_loss_sum += float(task_loss.detach().item())
        scale = 1.0 / float(num_microbatches)
        total = (task_loss * scale) + (aux_loss_coef * scale * aux_loss)
        total.backward()

        if not model.is_first_pp_stage:
            if stage_input is None or stage_input.grad is None:
                raise RuntimeError("Expected stage input gradient on last PP stage")
            if peers.prev_rank is None:
                raise RuntimeError("Missing previous PP rank")
            grad_payload = stage_input.grad.contiguous()
            state.grad_send_buffers.append(grad_payload)
            state.grad_send_reqs.append(
                dist.isend(
                    tensor=grad_payload,
                    dst=peers.prev_rank,
                    tag=_grad_tag(microbatch_idx),
                )
            )
        return

    # Non-last stages receive dY from next stage, then backprop local graph.
    grad_output = torch.empty(
        (microbatch_batch_size, seq_len, hidden_size),
        device=device,
        dtype=activation_dtype,
    )
    if peers.next_rank is None:
        raise RuntimeError("Missing next PP rank")
    dist.recv(grad_output, src=peers.next_rank, tag=_grad_tag(microbatch_idx))

    if aux_loss.requires_grad:
        aux_scale = torch.tensor(
            aux_loss_coef / float(num_microbatches),
            dtype=aux_loss.dtype,
            device=device,
        )
        torch.autograd.backward([stage_output, aux_loss], [grad_output, aux_scale])
    else:
        torch.autograd.backward(stage_output, grad_output)

    if not model.is_first_pp_stage:
        if stage_input is None or stage_input.grad is None:
            raise RuntimeError("Expected stage input gradient on middle PP stage")
        if peers.prev_rank is None:
            raise RuntimeError("Missing previous PP rank")
        grad_payload = stage_input.grad.contiguous()
        state.grad_send_buffers.append(grad_payload)
        state.grad_send_reqs.append(
            dist.isend(
                tensor=grad_payload,
                dst=peers.prev_rank,
                tag=_grad_tag(microbatch_idx),
            )
        )


def _execute_1f1b_schedule(
    parallel: ModelParallelTopology,
    num_microbatches: int,
    run_forward,
    run_backward,
) -> None:
    """Execute warmup/steady/cooldown phases for non-interleaved 1F1B."""
    # Warmup: fill pipeline bubbles with forward-only microbatches.
    num_warmup = min(
        parallel.pipeline_model_parallel_size - parallel.pipeline_model_parallel_rank - 1,
        num_microbatches,
    )
    num_remaining = num_microbatches - num_warmup

    for microbatch_idx in range(num_warmup):
        run_forward(microbatch_idx)

    # Steady state: one forward + one backward per step.
    for steady_idx in range(num_remaining):
        forward_idx = steady_idx + num_warmup
        run_forward(forward_idx)
        run_backward(steady_idx)

    # Cooldown: drain remaining backward passes.
    for cooldown_idx in range(num_warmup):
        backward_idx = num_remaining + cooldown_idx
        run_backward(backward_idx)


def _finalize_pipeline_sends(state: _PipelineStepState) -> None:
    """Wait on async send requests and clear send payload references."""
    for req in state.activation_send_reqs:
        req.wait()
    for req in state.grad_send_reqs:
        req.wait()
    for req in state.label_send_reqs:
        req.wait()

    # Keep async payloads alive until all work handles complete, then release.
    state.activation_send_buffers.clear()
    state.grad_send_buffers.clear()


def train_step_pipeline(
    model: DeepSeekModel,
    optimizer: torch.optim.Optimizer | MegatronZeroOptimizer,
    use_distributed_optimizer: bool,
    batch: Optional[dict[str, torch.Tensor]],
    parallel: ModelParallelTopology,
    num_microbatches: int,
    expected_local_batch: int,
    seq_len: int,
    aux_loss_coef: float,
    shard_info: ParamShardInfo,
) -> tuple[float, float, float, int, int]:
    """
    One TP+PP+EP+DP training step using non-interleaved 1F1B schedule.

    Communication domains used in this step:
    - PP p2p send/recv for activations and activation-gradients.
    - TP collectives stay internal to TP layers.
    - EP all-to-all stays internal to MoE layers.
    - EP/DP gradient all-reduce happens once after all microbatches.
    """
    if parallel.pipeline_model_parallel_size <= 1:
        raise ValueError("train_step_pipeline requires pipeline_model_parallel_size > 1")

    device = parallel.device
    activation_dtype = next(model.parameters()).dtype
    hidden_size = model.config.hidden_size
    microbatch_batch_size = expected_local_batch // num_microbatches
    peers = _resolve_pipeline_peers(parallel)
    state = _PipelineStepState()
    optimizer.zero_grad(set_to_none=True)

    microbatches = _prepare_microbatches(
        batch=batch,
        model=model,
        expected_local_batch=expected_local_batch,
        num_microbatches=num_microbatches,
        device=device,
    )

    _execute_1f1b_schedule(
        parallel=parallel,
        num_microbatches=num_microbatches,
        run_forward=lambda microbatch_idx: _pipeline_forward_microbatch(
            microbatch_idx=microbatch_idx,
            model=model,
            microbatches=microbatches,
            peers=peers,
            state=state,
            microbatch_batch_size=microbatch_batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            activation_dtype=activation_dtype,
            device=device,
        ),
        run_backward=lambda microbatch_idx: _pipeline_backward_microbatch(
            microbatch_idx=microbatch_idx,
            model=model,
            peers=peers,
            state=state,
            num_microbatches=num_microbatches,
            aux_loss_coef=aux_loss_coef,
            microbatch_batch_size=microbatch_batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            activation_dtype=activation_dtype,
            device=device,
        ),
    )
    _finalize_pipeline_sends(state)

    _apply_optimizer_step(
        model=model,
        optimizer=optimizer,
        use_distributed_optimizer=use_distributed_optimizer,
        shard_info=shard_info,
        data_parallel_size=parallel.data_parallel_size,
        expert_data_parallel_size=parallel.expert_data_parallel_size,
        data_parallel_group=parallel.data_parallel_group,
        expert_data_parallel_group=parallel.expert_data_parallel_group,
    )

    objective_count = num_microbatches if model.is_last_pp_stage else 0
    return (
        state.task_loss_sum,
        state.aux_loss_sum,
        state.drop_sum,
        objective_count,
        state.drop_count,
    )


def aggregate_non_pipeline_metrics(
    task_sum: float,
    aux_sum: float,
    loss_sum: float,
    drop_sum: float,
    count: int,
    device: torch.device,
    world_size: int,
) -> tuple[float, float, float, float]:
    """Aggregate average metrics across all ranks for non-pipeline logging."""
    if world_size == 1:
        denom = max(1, count)
        return task_sum / denom, aux_sum / denom, loss_sum / denom, drop_sum / denom

    stats = torch.tensor(
        [task_sum, aux_sum, loss_sum, drop_sum, float(count)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    denom = max(1.0, stats[4].item())
    return (
        float(stats[0].item() / denom),
        float(stats[1].item() / denom),
        float(stats[2].item() / denom),
        float(stats[3].item() / denom),
    )


def aggregate_pipeline_metrics(
    task_sum: float,
    aux_sum: float,
    drop_sum: float,
    objective_count: int,
    drop_count: int,
    device: torch.device,
    world_size: int,
) -> tuple[float, float, float]:
    """Aggregate pipeline metrics with stage-aware denominator handling."""
    if world_size == 1:
        obj_denom = max(1, objective_count)
        drop_denom = max(1, drop_count)
        return task_sum / obj_denom, aux_sum / obj_denom, drop_sum / drop_denom

    stats = torch.tensor(
        [
            task_sum,
            aux_sum,
            drop_sum,
            float(objective_count),
            float(drop_count),
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    obj_denom = max(1.0, stats[3].item())
    drop_denom = max(1.0, stats[4].item())
    return (
        float(stats[0].item() / obj_denom),
        float(stats[1].item() / obj_denom),
        float(stats[2].item() / drop_denom),
    )


def _initialize_runtime(
    args: argparse.Namespace,
    pp_layer_splits: Optional[tuple[int, ...]],
) -> ModelParallelTopology:
    """Validate distributed configuration and initialize model-parallel runtime."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    validate_args(args, world_size, pp_layer_splits)
    parallel = initialize_model_parallel(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
    )
    mode = infer_mode(parallel)
    log_parallel_topology(parallel=parallel, mode=mode)
    return parallel


def _build_model_stack(
    args: argparse.Namespace,
    parallel: ModelParallelTopology,
    pp_layer_splits: Optional[tuple[int, ...]],
) -> tuple[DeepSeekModel, torch.optim.Optimizer | MegatronZeroOptimizer, ParamShardInfo]:
    """Build model/context, synchronize initialization, and construct optimizer."""
    # Seed by (pp_rank, tp_rank, ep_rank) so shards initialize differently before sync.
    seed_offset = (
        parallel.pipeline_model_parallel_rank * 1_000_003
        + parallel.tensor_model_parallel_rank * 1009
        + parallel.expert_model_parallel_rank
    )
    torch.manual_seed(args.seed + seed_offset)
    if parallel.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + seed_offset)

    model_config = build_tiny_deepseek_config(args)
    parallel_context = DeepSeekParallelContext(
        tensor_model_parallel_rank=parallel.tensor_model_parallel_rank,
        tensor_model_parallel_size=parallel.tensor_model_parallel_size,
        tensor_model_parallel_group=parallel.tensor_model_parallel_group,
        expert_model_parallel_rank=parallel.expert_model_parallel_rank,
        expert_model_parallel_size=parallel.expert_model_parallel_size,
        expert_model_parallel_group=parallel.expert_model_parallel_group,
        pipeline_model_parallel_rank=parallel.pipeline_model_parallel_rank,
        pipeline_model_parallel_size=parallel.pipeline_model_parallel_size,
        pipeline_model_parallel_group=parallel.pipeline_model_parallel_group,
        context_parallel_rank=parallel.context_parallel_rank,
        context_parallel_size=parallel.context_parallel_size,
        context_parallel_group=parallel.context_parallel_group,
        pp_layer_splits=pp_layer_splits,
        capacity_factor=args.capacity_factor,
        expert_tensor_parallel_size=args.expert_tensor_parallel_size,
        sequence_parallel=True,
    )
    model = DeepSeekModel(model_config, parallel_context=parallel_context).to(parallel.device)

    shard_info = collect_param_shard_info(
        model=model,
        tensor_model_parallel_group=parallel.tensor_model_parallel_group,
    )
    synchronize_initial_parameters(
        model=model,
        shard_info=shard_info,
        parallel=parallel,
    )

    if args.use_distributed_optimizer:
        optimizer = MegatronZeroOptimizer(
            model=model,
            config=DistributedOptimizerConfig(
                use_distributed_optimizer=True,
                data_parallel_sharding_strategy=args.data_parallel_sharding_strategy,
                num_distributed_optimizer_instances=args.num_distributed_optimizer_instances,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8,
                use_reduce_scatter=True,
                debug=args.zero_debug,
                debug_max_steps=args.zero_debug_max_steps,
                debug_max_params=args.zero_debug_max_params,
            ),
            data_parallel_group=parallel.data_parallel_group,
            expert_data_parallel_group=parallel.expert_data_parallel_group,
            expert_param_ids=shard_info.expert_model_parallel_sharded_param_ids,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    return model, optimizer, shard_info


def _build_training_data(
    args: argparse.Namespace,
    parallel: ModelParallelTopology,
):
    """Build deterministic token dataset and DP-sharded dataloader."""
    dataset = DummyTokenDataset(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    return create_data_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        dp_size=parallel.data_parallel_size,
        dp_rank=parallel.data_parallel_rank,
        seed=args.seed,
    )


def _log_training_start(
    args: argparse.Namespace,
    parallel: ModelParallelTopology,
    model: DeepSeekModel,
) -> None:
    """Log rank-0 training configuration summary."""
    if parallel.rank != 0:
        return

    local_params = sum(parameter.numel() for parameter in model.parameters())
    logger.info("Local params: %d", local_params)
    logger.info(
        "DeepSeek tiny config: layers=%d hidden=%d experts=%d top_k=%d seq_len=%d",
        args.num_layers,
        args.hidden_size,
        args.num_experts,
        args.top_k,
        args.seq_len,
    )
    logger.info(
        "Model context: tensor=%d/%d pipeline=%d/%d expert=%d/%d data=%d/%d context=%d/%d",
        parallel.tensor_model_parallel_rank,
        parallel.tensor_model_parallel_size,
        parallel.pipeline_model_parallel_rank,
        parallel.pipeline_model_parallel_size,
        parallel.expert_model_parallel_rank,
        parallel.expert_model_parallel_size,
        parallel.data_parallel_rank,
        parallel.data_parallel_size,
        parallel.context_parallel_rank,
        parallel.context_parallel_size,
    )
    logger.info(
        "Distributed optimizer: enabled=%s strategy=%s dist_opt_instances=%d "
        "zero_debug=%s zero_debug_max_steps=%d zero_debug_max_params=%d",
        args.use_distributed_optimizer,
        args.data_parallel_sharding_strategy,
        args.num_distributed_optimizer_instances,
        args.zero_debug,
        args.zero_debug_max_steps,
        args.zero_debug_max_params,
    )
    logger.info("Starting training: epochs=%d, max_steps=%d", args.epochs, args.max_steps)


def _run_non_pipeline_training(
    args: argparse.Namespace,
    model: DeepSeekModel,
    optimizer: torch.optim.Optimizer | MegatronZeroOptimizer,
    train_loader,
    sampler: Optional[DistributedSampler],
    parallel: ModelParallelTopology,
    shard_info: ParamShardInfo,
) -> None:
    """Run training loop for non-PP configurations."""
    step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        task_sum = 0.0
        aux_sum = 0.0
        loss_sum = 0.0
        drop_sum = 0.0
        batch_count = 0

        for batch in train_loader:
            if step >= args.max_steps:
                break

            task_loss, aux_loss, total_loss, drop_fraction = train_step_non_pipeline(
                model=model,
                optimizer=optimizer,
                use_distributed_optimizer=args.use_distributed_optimizer,
                batch=batch,
                device=parallel.device,
                data_parallel_size=parallel.data_parallel_size,
                expert_data_parallel_size=parallel.expert_data_parallel_size,
                aux_loss_coef=args.aux_loss_coef,
                shard_info=shard_info,
                data_parallel_group=parallel.data_parallel_group,
                expert_data_parallel_group=parallel.expert_data_parallel_group,
            )

            task_sum += task_loss
            aux_sum += aux_loss
            loss_sum += total_loss
            drop_sum += drop_fraction
            batch_count += 1

            if parallel.rank == 0 and step % args.log_every == 0:
                logger.info(
                    "step=%d task=%.6f aux=%.6f total=%.6f drop=%.4f",
                    step,
                    task_loss,
                    aux_loss,
                    total_loss,
                    drop_fraction,
                )
            step += 1

        if batch_count > 0:
            avg_task, avg_aux, avg_total, avg_drop = aggregate_non_pipeline_metrics(
                task_sum=task_sum,
                aux_sum=aux_sum,
                loss_sum=loss_sum,
                drop_sum=drop_sum,
                count=batch_count,
                device=parallel.device,
                world_size=parallel.world_size,
            )
            if parallel.rank == 0:
                logger.info(
                    "epoch=%d avg_task=%.6f avg_aux=%.6f avg_total=%.6f avg_drop=%.4f",
                    epoch + 1,
                    avg_task,
                    avg_aux,
                    avg_total,
                    avg_drop,
                )

        if step >= args.max_steps:
            break


def _run_pipeline_training(
    args: argparse.Namespace,
    model: DeepSeekModel,
    optimizer: torch.optim.Optimizer | MegatronZeroOptimizer,
    train_loader,
    sampler: Optional[DistributedSampler],
    parallel: ModelParallelTopology,
    shard_info: ParamShardInfo,
) -> None:
    """Run training loop for PP-enabled configurations."""
    pipeline_epoch = 0
    if sampler is not None and model.is_first_pp_stage:
        sampler.set_epoch(pipeline_epoch)

    data_iter = iter(train_loader) if model.is_first_pp_stage else None
    for step in range(args.max_steps):
        if model.is_first_pp_stage:
            if data_iter is None:
                raise RuntimeError("Missing data iterator on first PP stage")
            try:
                step_batch = next(data_iter)
            except StopIteration:
                pipeline_epoch += 1
                if sampler is not None:
                    sampler.set_epoch(pipeline_epoch)
                data_iter = iter(train_loader)
                step_batch = next(data_iter)
        else:
            step_batch = None

        task_sum, aux_sum, drop_sum, objective_count, drop_count = train_step_pipeline(
            model=model,
            optimizer=optimizer,
            use_distributed_optimizer=args.use_distributed_optimizer,
            batch=step_batch,
            parallel=parallel,
            num_microbatches=args.num_microbatches,
            expected_local_batch=args.batch_size,
            seq_len=args.seq_len,
            aux_loss_coef=args.aux_loss_coef,
            shard_info=shard_info,
        )

        avg_task, avg_aux, avg_drop = aggregate_pipeline_metrics(
            task_sum=task_sum,
            aux_sum=aux_sum,
            drop_sum=drop_sum,
            objective_count=objective_count,
            drop_count=drop_count,
            device=parallel.device,
            world_size=parallel.world_size,
        )
        avg_total = avg_task + (args.aux_loss_coef * avg_aux)

        if parallel.rank == 0 and step % args.log_every == 0:
            logger.info(
                "step=%d task=%.6f aux=%.6f total=%.6f drop=%.4f",
                step,
                avg_task,
                avg_aux,
                avg_total,
                avg_drop,
            )


def main() -> None:
    """Run canonical TP+PP+EP+DP tutorial training with DeepSeek model."""
    args = parse_args()
    pp_layer_splits = parse_pp_layer_splits(args.pp_layer_splits)

    rank_pre = int(os.environ.get("RANK", "0"))
    setup_process_logging(rank_pre)

    parallel = _initialize_runtime(args, pp_layer_splits)
    model, optimizer, shard_info = _build_model_stack(args, parallel, pp_layer_splits)
    train_loader, sampler = _build_training_data(args, parallel)
    _log_training_start(args, parallel, model)

    model.train()

    if parallel.pipeline_model_parallel_size == 1:
        _run_non_pipeline_training(
            args=args,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            sampler=sampler,
            parallel=parallel,
            shard_info=shard_info,
        )
    else:
        _run_pipeline_training(
            args=args,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            sampler=sampler,
            parallel=parallel,
            shard_info=shard_info,
        )

    if parallel.rank == 0:
        logger.info("Training completed")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
