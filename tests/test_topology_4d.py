"""Unit tests for Megatron-style topology helpers."""

from __future__ import annotations

import torch.distributed as dist

from src.distributed.topology import coords_from_rank
from src.distributed.topology import initialize_model_parallel
from src.distributed.topology import rank_from_coords


def test_rank_coords_round_trip() -> None:
    """rank_from_coords and coords_from_rank should be exact inverses."""
    pipeline_model_parallel_size = 3
    tensor_model_parallel_size = 2
    expert_model_parallel_size = 2
    data_parallel_size = 2

    for data_parallel_rank in range(data_parallel_size):
        for pipeline_model_parallel_rank in range(pipeline_model_parallel_size):
            for tensor_model_parallel_rank in range(tensor_model_parallel_size):
                for expert_model_parallel_rank in range(expert_model_parallel_size):
                    rank = rank_from_coords(
                        data_parallel_rank=data_parallel_rank,
                        pipeline_model_parallel_rank=pipeline_model_parallel_rank,
                        tensor_model_parallel_rank=tensor_model_parallel_rank,
                        expert_model_parallel_rank=expert_model_parallel_rank,
                        pipeline_model_parallel_size=pipeline_model_parallel_size,
                        tensor_model_parallel_size=tensor_model_parallel_size,
                        expert_model_parallel_size=expert_model_parallel_size,
                    )
                    got = coords_from_rank(
                        rank=rank,
                        pipeline_model_parallel_size=pipeline_model_parallel_size,
                        tensor_model_parallel_size=tensor_model_parallel_size,
                        expert_model_parallel_size=expert_model_parallel_size,
                    )
                    assert got == (
                        data_parallel_rank,
                        pipeline_model_parallel_rank,
                        tensor_model_parallel_rank,
                        expert_model_parallel_rank,
                        0,
                    )


def test_group_membership_tables(monkeypatch) -> None:
    """Group tables should match expected rank layout for tensor=2, pipeline=2, expert=2."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "0")

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "new_group", lambda ranks: tuple(ranks))

    setup = initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        expert_model_parallel_size=2,
    )

    assert setup.data_parallel_size == 1
    assert setup.expert_data_parallel_size == 2
    assert setup.tensor_model_parallel_group_table == [[0, 2], [1, 3], [4, 6], [5, 7]]
    assert setup.expert_model_parallel_group_table == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert setup.pipeline_model_parallel_group_table == [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert setup.data_parallel_group_table == []
    assert setup.expert_data_parallel_group_table == [[0, 2], [1, 3], [4, 6], [5, 7]]
