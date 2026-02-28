"""Synchronization helpers shared by runtime plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from typing import Set

import torch
import torch.distributed as dist
import torch.nn as nn

from src.distributed.topology import ModelParallelTopology
from src.layers import ColumnParallelLinear
from src.layers import RowParallelLinear
from src.models.moe import ExpertParallelMoE
from src.runtime.master_store import get_lowbit_master_store


@dataclass
class ParamShardInfo:
    """Parameter id sets split by sharding domain for sync logic."""

    tensor_model_parallel_sharded_param_ids: Set[int]
    expert_model_parallel_sharded_param_ids: Set[int]


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

    # Optimizer-owned master mappings can externalize ownership. Fold metadata
    # domains into shard-classification sets so sync policy stays correct.
    master_store = get_lowbit_master_store(model)
    if master_store is not None:
        for key, metadata in master_store.metadata.items():
            if key not in master_store.masters:
                continue
            param = master_store.masters[key]
            if metadata.shard_domain == "tensor_model_parallel":
                tensor_model_parallel_sharded_param_ids.add(id(param))
            elif metadata.shard_domain == "expert_model_parallel":
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
