"""Distributed topology helpers with Megatron-style parallel naming."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

import torch
import torch.distributed as dist

from src.distributed.device import get_backend
from src.distributed.device import get_device


@dataclass
class ModelParallelTopology:
    """Resolved parallel topology and process groups using Megatron naming."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    expert_model_parallel_size: int
    context_parallel_size: int
    data_parallel_size: int
    expert_data_parallel_size: int

    tensor_model_parallel_rank: int
    pipeline_model_parallel_rank: int
    expert_model_parallel_rank: int
    context_parallel_rank: int
    data_parallel_rank: int

    tensor_model_parallel_group: Optional[object]
    pipeline_model_parallel_group: Optional[object]
    expert_model_parallel_group: Optional[object]
    context_parallel_group: Optional[object]
    data_parallel_group: Optional[object]
    expert_data_parallel_group: Optional[object]

    tensor_model_parallel_replica_group: Optional[object]
    expert_model_parallel_replica_group: Optional[object]
    pipeline_stage_replica_group: Optional[object]

    tensor_model_parallel_group_table: list[list[int]]
    pipeline_model_parallel_group_table: list[list[int]]
    expert_model_parallel_group_table: list[list[int]]
    context_parallel_group_table: list[list[int]]
    data_parallel_group_table: list[list[int]]
    expert_data_parallel_group_table: list[list[int]]

    def rank_from_coords(
        self,
        data_parallel_rank: int,
        pipeline_model_parallel_rank: int,
        tensor_model_parallel_rank: int,
        expert_model_parallel_rank: int,
        context_parallel_rank: int = 0,
    ) -> int:
        """Compute global rank from model/data parallel coordinates."""
        return rank_from_coords(
            data_parallel_rank=data_parallel_rank,
            pipeline_model_parallel_rank=pipeline_model_parallel_rank,
            tensor_model_parallel_rank=tensor_model_parallel_rank,
            expert_model_parallel_rank=expert_model_parallel_rank,
            pipeline_model_parallel_size=self.pipeline_model_parallel_size,
            tensor_model_parallel_size=self.tensor_model_parallel_size,
            expert_model_parallel_size=self.expert_model_parallel_size,
            context_parallel_rank=context_parallel_rank,
            context_parallel_size=self.context_parallel_size,
        )

    # ------------------------------------------------------------------
    # Backward-compat aliases (one-release transition)
    # ------------------------------------------------------------------
    @property
    def tp_size(self) -> int:
        return self.tensor_model_parallel_size

    @property
    def pp_size(self) -> int:
        return self.pipeline_model_parallel_size

    @property
    def ep_size(self) -> int:
        return self.expert_model_parallel_size

    @property
    def dp_size(self) -> int:
        return self.data_parallel_size

    @property
    def tp_rank(self) -> int:
        return self.tensor_model_parallel_rank

    @property
    def pp_rank(self) -> int:
        return self.pipeline_model_parallel_rank

    @property
    def ep_rank(self) -> int:
        return self.expert_model_parallel_rank

    @property
    def dp_rank(self) -> int:
        return self.data_parallel_rank

    @property
    def tp_group(self):
        return self.tensor_model_parallel_group

    @property
    def pp_group(self):
        return self.pipeline_model_parallel_group

    @property
    def ep_group(self):
        return self.expert_model_parallel_group

    @property
    def dp_group(self):
        return self.data_parallel_group

    @property
    def tp_replica_group(self):
        return self.tensor_model_parallel_replica_group

    @property
    def ep_replica_group(self):
        return self.expert_model_parallel_replica_group

    @property
    def stage_replica_group(self):
        return self.pipeline_stage_replica_group

    @property
    def tp_group_table(self) -> list[list[int]]:
        return self.tensor_model_parallel_group_table

    @property
    def pp_group_table(self) -> list[list[int]]:
        return self.pipeline_model_parallel_group_table

    @property
    def ep_group_table(self) -> list[list[int]]:
        return self.expert_model_parallel_group_table

    @property
    def dp_group_table(self) -> list[list[int]]:
        return self.data_parallel_group_table

ParallelSetup4D = ModelParallelTopology


def rank_from_coords(
    data_parallel_rank: int,
    pipeline_model_parallel_rank: int,
    tensor_model_parallel_rank: int,
    expert_model_parallel_rank: int,
    pipeline_model_parallel_size: int,
    tensor_model_parallel_size: int,
    expert_model_parallel_size: int,
    context_parallel_rank: int = 0,
    context_parallel_size: int = 1,
) -> int:
    """Compute global rank from parallel coordinates using EP-fastest layout."""
    return (
        (
            (
                (
                    data_parallel_rank * pipeline_model_parallel_size
                    + pipeline_model_parallel_rank
                )
                * tensor_model_parallel_size
                + tensor_model_parallel_rank
            )
            * expert_model_parallel_size
            + expert_model_parallel_rank
        )
        * context_parallel_size
        + context_parallel_rank
    )


def coords_from_rank(
    rank: int,
    pipeline_model_parallel_size: int,
    tensor_model_parallel_size: int,
    expert_model_parallel_size: int,
    context_parallel_size: int = 1,
) -> tuple[int, int, int, int, int]:
    """Decode `(data_parallel, pipeline, tensor, expert, context)` from global rank."""
    model_parallel = (
        pipeline_model_parallel_size
        * tensor_model_parallel_size
        * expert_model_parallel_size
        * context_parallel_size
    )
    data_parallel_rank = rank // model_parallel
    rem = rank % model_parallel

    pipeline_tensor_expert = (
        tensor_model_parallel_size * expert_model_parallel_size * context_parallel_size
    )
    pipeline_model_parallel_rank = rem // pipeline_tensor_expert
    rem = rem % pipeline_tensor_expert

    tensor_expert = expert_model_parallel_size * context_parallel_size
    tensor_model_parallel_rank = rem // tensor_expert
    rem = rem % tensor_expert

    expert_model_parallel_rank = rem // context_parallel_size
    context_parallel_rank = rem % context_parallel_size
    return (
        data_parallel_rank,
        pipeline_model_parallel_rank,
        tensor_model_parallel_rank,
        expert_model_parallel_rank,
        context_parallel_rank,
    )


def _new_group_if_needed(ranks: list[int], distributed: bool):
    """Create a process group only when distributed mode is enabled."""
    if not distributed:
        return None
    return dist.new_group(ranks)


def initialize_model_parallel(
    *,
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    expert_model_parallel_size: int,
    context_parallel_size: int = 1,
) -> ModelParallelTopology:
    """
    Initialize model/data parallel groups with Megatron-style naming.

    Rank mapping (expert parallel fastest, then context parallel):
        rank = ((((dp * pp) + pp_rank) * tp + tp_rank) * ep + ep_rank) * cp + cp_rank
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    if tensor_model_parallel_size < 1:
        raise ValueError("tensor_model_parallel_size must be >= 1")
    if pipeline_model_parallel_size < 1:
        raise ValueError("pipeline_model_parallel_size must be >= 1")
    if expert_model_parallel_size < 1:
        raise ValueError("expert_model_parallel_size must be >= 1")
    if context_parallel_size < 1:
        raise ValueError("context_parallel_size must be >= 1")

    model_parallel = (
        tensor_model_parallel_size
        * pipeline_model_parallel_size
        * expert_model_parallel_size
        * context_parallel_size
    )
    if world_size % model_parallel != 0:
        raise ValueError(
            "world_size must be divisible by tensor*pipeline*expert*context sizes: "
            f"{world_size} vs {model_parallel}"
        )

    backend = get_backend()
    device_type = "cuda" if backend == "nccl" else "cpu"
    device = get_device(device_type, local_rank)
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend, world_size=world_size, rank=rank)

    data_parallel_size = world_size // model_parallel
    (
        data_parallel_rank,
        pipeline_model_parallel_rank,
        tensor_model_parallel_rank,
        expert_model_parallel_rank,
        context_parallel_rank,
    ) = coords_from_rank(
        rank=rank,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        tensor_model_parallel_size=tensor_model_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        context_parallel_size=context_parallel_size,
    )

    tensor_model_parallel_group = None
    pipeline_model_parallel_group = None
    expert_model_parallel_group = None
    context_parallel_group = None
    data_parallel_group = None
    expert_data_parallel_group = None

    tensor_model_parallel_replica_group = None
    expert_model_parallel_replica_group = None
    pipeline_stage_replica_group = None

    tensor_model_parallel_group_table: list[list[int]] = []
    pipeline_model_parallel_group_table: list[list[int]] = []
    expert_model_parallel_group_table: list[list[int]] = []
    context_parallel_group_table: list[list[int]] = []
    data_parallel_group_table: list[list[int]] = []
    expert_data_parallel_group_table: list[list[int]] = []

    # TP groups: fixed (dp, pp, ep, cp), varying tp.
    if tensor_model_parallel_size > 1:
        for dp_idx in range(data_parallel_size):
            for pp_idx in range(pipeline_model_parallel_size):
                for ep_idx in range(expert_model_parallel_size):
                    for cp_idx in range(context_parallel_size):
                        ranks = [
                            rank_from_coords(
                                data_parallel_rank=dp_idx,
                                pipeline_model_parallel_rank=pp_idx,
                                tensor_model_parallel_rank=tp_idx,
                                expert_model_parallel_rank=ep_idx,
                                pipeline_model_parallel_size=pipeline_model_parallel_size,
                                tensor_model_parallel_size=tensor_model_parallel_size,
                                expert_model_parallel_size=expert_model_parallel_size,
                                context_parallel_rank=cp_idx,
                                context_parallel_size=context_parallel_size,
                            )
                            for tp_idx in range(tensor_model_parallel_size)
                        ]
                        tensor_model_parallel_group_table.append(ranks)
                        group = _new_group_if_needed(ranks, distributed)
                        if (
                            dp_idx == data_parallel_rank
                            and pp_idx == pipeline_model_parallel_rank
                            and ep_idx == expert_model_parallel_rank
                            and cp_idx == context_parallel_rank
                        ):
                            tensor_model_parallel_group = group

    # Expert-model-parallel groups: fixed (dp, pp, tp, cp), varying ep.
    if expert_model_parallel_size > 1:
        for dp_idx in range(data_parallel_size):
            for pp_idx in range(pipeline_model_parallel_size):
                for tp_idx in range(tensor_model_parallel_size):
                    for cp_idx in range(context_parallel_size):
                        ranks = [
                            rank_from_coords(
                                data_parallel_rank=dp_idx,
                                pipeline_model_parallel_rank=pp_idx,
                                tensor_model_parallel_rank=tp_idx,
                                expert_model_parallel_rank=ep_idx,
                                pipeline_model_parallel_size=pipeline_model_parallel_size,
                                tensor_model_parallel_size=tensor_model_parallel_size,
                                expert_model_parallel_size=expert_model_parallel_size,
                                context_parallel_rank=cp_idx,
                                context_parallel_size=context_parallel_size,
                            )
                            for ep_idx in range(expert_model_parallel_size)
                        ]
                        expert_model_parallel_group_table.append(ranks)
                        group = _new_group_if_needed(ranks, distributed)
                        if (
                            dp_idx == data_parallel_rank
                            and pp_idx == pipeline_model_parallel_rank
                            and tp_idx == tensor_model_parallel_rank
                            and cp_idx == context_parallel_rank
                        ):
                            expert_model_parallel_group = group

    # PP groups: fixed (dp, tp, ep, cp), varying pp.
    if pipeline_model_parallel_size > 1:
        for dp_idx in range(data_parallel_size):
            for tp_idx in range(tensor_model_parallel_size):
                for ep_idx in range(expert_model_parallel_size):
                    for cp_idx in range(context_parallel_size):
                        ranks = [
                            rank_from_coords(
                                data_parallel_rank=dp_idx,
                                pipeline_model_parallel_rank=pp_idx,
                                tensor_model_parallel_rank=tp_idx,
                                expert_model_parallel_rank=ep_idx,
                                pipeline_model_parallel_size=pipeline_model_parallel_size,
                                tensor_model_parallel_size=tensor_model_parallel_size,
                                expert_model_parallel_size=expert_model_parallel_size,
                                context_parallel_rank=cp_idx,
                                context_parallel_size=context_parallel_size,
                            )
                            for pp_idx in range(pipeline_model_parallel_size)
                        ]
                        pipeline_model_parallel_group_table.append(ranks)
                        group = _new_group_if_needed(ranks, distributed)
                        if (
                            dp_idx == data_parallel_rank
                            and tp_idx == tensor_model_parallel_rank
                            and ep_idx == expert_model_parallel_rank
                            and cp_idx == context_parallel_rank
                        ):
                            pipeline_model_parallel_group = group

    # CP groups: fixed (dp, pp, tp, ep), varying cp.
    if context_parallel_size > 1:
        for dp_idx in range(data_parallel_size):
            for pp_idx in range(pipeline_model_parallel_size):
                for tp_idx in range(tensor_model_parallel_size):
                    for ep_idx in range(expert_model_parallel_size):
                        ranks = [
                            rank_from_coords(
                                data_parallel_rank=dp_idx,
                                pipeline_model_parallel_rank=pp_idx,
                                tensor_model_parallel_rank=tp_idx,
                                expert_model_parallel_rank=ep_idx,
                                pipeline_model_parallel_size=pipeline_model_parallel_size,
                                tensor_model_parallel_size=tensor_model_parallel_size,
                                expert_model_parallel_size=expert_model_parallel_size,
                                context_parallel_rank=cp_idx,
                                context_parallel_size=context_parallel_size,
                            )
                            for cp_idx in range(context_parallel_size)
                        ]
                        context_parallel_group_table.append(ranks)
                        group = _new_group_if_needed(ranks, distributed)
                        if (
                            dp_idx == data_parallel_rank
                            and pp_idx == pipeline_model_parallel_rank
                            and tp_idx == tensor_model_parallel_rank
                            and ep_idx == expert_model_parallel_rank
                        ):
                            context_parallel_group = group

    # DP groups: fixed (pp, tp, ep, cp), varying dp.
    if data_parallel_size > 1:
        for pp_idx in range(pipeline_model_parallel_size):
            for tp_idx in range(tensor_model_parallel_size):
                for ep_idx in range(expert_model_parallel_size):
                    for cp_idx in range(context_parallel_size):
                        ranks = [
                            rank_from_coords(
                                data_parallel_rank=dp_idx,
                                pipeline_model_parallel_rank=pp_idx,
                                tensor_model_parallel_rank=tp_idx,
                                expert_model_parallel_rank=ep_idx,
                                pipeline_model_parallel_size=pipeline_model_parallel_size,
                                tensor_model_parallel_size=tensor_model_parallel_size,
                                expert_model_parallel_size=expert_model_parallel_size,
                                context_parallel_rank=cp_idx,
                                context_parallel_size=context_parallel_size,
                            )
                            for dp_idx in range(data_parallel_size)
                        ]
                        data_parallel_group_table.append(ranks)
                        group = _new_group_if_needed(ranks, distributed)
                        if (
                            pp_idx == pipeline_model_parallel_rank
                            and tp_idx == tensor_model_parallel_rank
                            and ep_idx == expert_model_parallel_rank
                            and cp_idx == context_parallel_rank
                        ):
                            data_parallel_group = group

    # Expert-DP groups: fixed (pp, ep, cp), varying (dp, tp).
    for pp_idx in range(pipeline_model_parallel_size):
        for ep_idx in range(expert_model_parallel_size):
            for cp_idx in range(context_parallel_size):
                ranks = [
                    rank_from_coords(
                        data_parallel_rank=dp_idx,
                        pipeline_model_parallel_rank=pp_idx,
                        tensor_model_parallel_rank=tp_idx,
                        expert_model_parallel_rank=ep_idx,
                        pipeline_model_parallel_size=pipeline_model_parallel_size,
                        tensor_model_parallel_size=tensor_model_parallel_size,
                        expert_model_parallel_size=expert_model_parallel_size,
                        context_parallel_rank=cp_idx,
                        context_parallel_size=context_parallel_size,
                    )
                    for dp_idx in range(data_parallel_size)
                    for tp_idx in range(tensor_model_parallel_size)
                ]
                expert_data_parallel_group_table.append(ranks)
                group = _new_group_if_needed(ranks, distributed)
                if (
                    pp_idx == pipeline_model_parallel_rank
                    and ep_idx == expert_model_parallel_rank
                    and cp_idx == context_parallel_rank
                ):
                    expert_data_parallel_group = group
                    expert_model_parallel_replica_group = group

    expert_data_parallel_size = data_parallel_size * tensor_model_parallel_size

    # Tensor-shard replica groups: fixed (pp, tp, cp), varying (dp, ep).
    for pp_idx in range(pipeline_model_parallel_size):
        for tp_idx in range(tensor_model_parallel_size):
            for cp_idx in range(context_parallel_size):
                ranks = [
                    rank_from_coords(
                        data_parallel_rank=dp_idx,
                        pipeline_model_parallel_rank=pp_idx,
                        tensor_model_parallel_rank=tp_idx,
                        expert_model_parallel_rank=ep_idx,
                        pipeline_model_parallel_size=pipeline_model_parallel_size,
                        tensor_model_parallel_size=tensor_model_parallel_size,
                        expert_model_parallel_size=expert_model_parallel_size,
                        context_parallel_rank=cp_idx,
                        context_parallel_size=context_parallel_size,
                    )
                    for dp_idx in range(data_parallel_size)
                    for ep_idx in range(expert_model_parallel_size)
                ]
                group = _new_group_if_needed(ranks, distributed)
                if (
                    pp_idx == pipeline_model_parallel_rank
                    and tp_idx == tensor_model_parallel_rank
                    and cp_idx == context_parallel_rank
                ):
                    tensor_model_parallel_replica_group = group

    # Pipeline-stage replica groups: fixed pp, varying (dp, tp, ep, cp).
    for pp_idx in range(pipeline_model_parallel_size):
        ranks = [
            rank_from_coords(
                data_parallel_rank=dp_idx,
                pipeline_model_parallel_rank=pp_idx,
                tensor_model_parallel_rank=tp_idx,
                expert_model_parallel_rank=ep_idx,
                pipeline_model_parallel_size=pipeline_model_parallel_size,
                tensor_model_parallel_size=tensor_model_parallel_size,
                expert_model_parallel_size=expert_model_parallel_size,
                context_parallel_rank=cp_idx,
                context_parallel_size=context_parallel_size,
            )
            for dp_idx in range(data_parallel_size)
            for tp_idx in range(tensor_model_parallel_size)
            for ep_idx in range(expert_model_parallel_size)
            for cp_idx in range(context_parallel_size)
        ]
        group = _new_group_if_needed(ranks, distributed)
        if pp_idx == pipeline_model_parallel_rank:
            pipeline_stage_replica_group = group

    return ModelParallelTopology(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        context_parallel_size=context_parallel_size,
        data_parallel_size=data_parallel_size,
        expert_data_parallel_size=expert_data_parallel_size,
        tensor_model_parallel_rank=tensor_model_parallel_rank,
        pipeline_model_parallel_rank=pipeline_model_parallel_rank,
        expert_model_parallel_rank=expert_model_parallel_rank,
        context_parallel_rank=context_parallel_rank,
        data_parallel_rank=data_parallel_rank,
        tensor_model_parallel_group=tensor_model_parallel_group,
        pipeline_model_parallel_group=pipeline_model_parallel_group,
        expert_model_parallel_group=expert_model_parallel_group,
        context_parallel_group=context_parallel_group,
        data_parallel_group=data_parallel_group,
        expert_data_parallel_group=expert_data_parallel_group,
        tensor_model_parallel_replica_group=tensor_model_parallel_replica_group,
        expert_model_parallel_replica_group=expert_model_parallel_replica_group,
        pipeline_stage_replica_group=pipeline_stage_replica_group,
        tensor_model_parallel_group_table=tensor_model_parallel_group_table,
        pipeline_model_parallel_group_table=pipeline_model_parallel_group_table,
        expert_model_parallel_group_table=expert_model_parallel_group_table,
        context_parallel_group_table=context_parallel_group_table,
        data_parallel_group_table=data_parallel_group_table,
        expert_data_parallel_group_table=expert_data_parallel_group_table,
    )


def init_parallel_4d(
    *,
    tp_size: int,
    pp_size: int,
    ep_size: int,
    context_parallel_size: int = 1,
) -> ModelParallelTopology:
    """Backward-compatible wrapper around `initialize_model_parallel`."""
    return initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        expert_model_parallel_size=ep_size,
        context_parallel_size=context_parallel_size,
    )
