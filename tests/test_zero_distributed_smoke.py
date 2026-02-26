"""Distributed smoke tests for ZeRO-1/2 optimizer paths."""

from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.distributed.topology import initialize_model_parallel
from src.distributed.zero import DataParallelShardingStrategy
from src.distributed.zero import DistributedOptimizerConfig
from src.distributed.zero import MegatronZeroOptimizer


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


class _ScalarModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.expert = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))


def _setup_env(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)


def _worker_world2(
    rank: int,
    world_size: int,
    port: int,
    strategy: DataParallelShardingStrategy,
) -> None:
    _setup_env(rank=rank, world_size=world_size, port=port)
    setup = initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
    )
    model = _ScalarModel()
    optimizer = MegatronZeroOptimizer(
        model=model,
        config=DistributedOptimizerConfig(
            use_distributed_optimizer=True,
            data_parallel_sharding_strategy=strategy,
            learning_rate=1e-2,
            weight_decay=0.0,
            betas=(0.0, 0.0),
            eps=1e-8,
        ),
        data_parallel_group=setup.data_parallel_group,
        expert_data_parallel_group=setup.expert_data_parallel_group,
        expert_param_ids={id(model.expert)},
    )

    model.dense.grad = torch.tensor([float(rank + 1)], dtype=torch.float32)
    model.expert.grad = torch.tensor([float(rank + 1)], dtype=torch.float32)
    optimizer.step_with_ready_grads()

    dense_val = model.dense.detach().clone()
    expert_val = model.expert.detach().clone()
    dist.all_reduce(dense_val, op=dist.ReduceOp.SUM)
    dist.all_reduce(expert_val, op=dist.ReduceOp.SUM)
    assert torch.isfinite(dense_val).all()
    assert torch.isfinite(expert_val).all()

    dist.destroy_process_group()


def _worker_world4_domain(
    rank: int,
    world_size: int,
    port: int,
    strategy: DataParallelShardingStrategy,
) -> None:
    _setup_env(rank=rank, world_size=world_size, port=port)
    setup = initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
    )
    model = _ScalarModel()
    optimizer = MegatronZeroOptimizer(
        model=model,
        config=DistributedOptimizerConfig(
            use_distributed_optimizer=True,
            data_parallel_sharding_strategy=strategy,
            learning_rate=1e-2,
            weight_decay=0.0,
            betas=(0.0, 0.0),
            eps=1e-8,
        ),
        data_parallel_group=setup.data_parallel_group,
        expert_data_parallel_group=setup.expert_data_parallel_group,
        expert_param_ids={id(model.expert)},
    )

    # Dense path reduces over DP groups: {0,2} and {1,3}.
    # Expert path reduces over EDP group: {0,1,2,3}.
    grad_val = float(rank) - 1.5
    model.dense.grad = torch.tensor([grad_val], dtype=torch.float32)
    model.expert.grad = torch.tensor([grad_val], dtype=torch.float32)
    optimizer.step_with_ready_grads()

    dense_local = model.dense.detach().clone()
    expert_local = model.expert.detach().clone()
    dense_list = [torch.zeros_like(dense_local) for _ in range(world_size)]
    expert_list = [torch.zeros_like(expert_local) for _ in range(world_size)]
    dist.all_gather(dense_list, dense_local)
    dist.all_gather(expert_list, expert_local)

    if rank == 0:
        dense_vals = [float(item.item()) for item in dense_list]
        expert_vals = [float(item.item()) for item in expert_list]
        assert abs(dense_vals[0] - dense_vals[2]) < 1e-6
        assert abs(dense_vals[1] - dense_vals[3]) < 1e-6
        assert dense_vals[0] > 1.0
        assert dense_vals[1] < 1.0
        for value in expert_vals:
            assert abs(value - 1.0) < 1e-6

    dist.destroy_process_group()


def test_zero_distributed_world2_optim() -> None:
    """ZeRO-1 smoke should run on world=2."""
    world_size = 2
    port = _free_port()
    mp.spawn(_worker_world2, args=(world_size, port, "optim"), nprocs=world_size, join=True)


def test_zero_distributed_world2_optim_grads() -> None:
    """ZeRO-2 smoke should run on world=2."""
    world_size = 2
    port = _free_port()
    mp.spawn(
        _worker_world2,
        args=(world_size, port, "optim_grads"),
        nprocs=world_size,
        join=True,
    )


def test_zero_domain_split_world4() -> None:
    """Dense DP and expert EDP domains should produce distinct update patterns."""
    world_size = 4
    port = _free_port()
    mp.spawn(
        _worker_world4_domain,
        args=(world_size, port, "optim"),
        nprocs=world_size,
        join=True,
    )
