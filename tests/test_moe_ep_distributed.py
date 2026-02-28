"""Distributed smoke test for ExpertParallelMoE collectives."""

import os
import socket
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.moe import ExpertParallelMoE
from src.runtime.contracts import PrecisionConfig
from src.runtime.mixed_precision import build_module_precision_resolver


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(100 + rank)
    moe = ExpertParallelMoE(
        hidden_size=16,
        expert_intermediate_size=32,
        num_experts=4,
        top_k=2,
        ep_rank=rank,
        ep_size=world_size,
        ep_group=dist.group.WORLD,
        param_dtype=torch.float32,
        param_device=None,
        precision_resolver=build_module_precision_resolver(PrecisionConfig(mode="fp32")),
        module_prefix=f"moe_ep.rank{rank}",
        dropout=0.0,
        n_shared_experts=1,
        scoring_func="sigmoid",
        n_group=2,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        capacity_factor=1.0,
        expert_tp_size=1,
    )

    x = torch.randn(2, 3, 16, requires_grad=True)
    out = moe(x)
    loss = out.pow(2).mean() + 0.01 * moe.last_aux_loss
    loss.backward()

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert torch.isfinite(moe.last_aux_loss)
    assert 0.0 <= moe.last_dropped_fraction <= 1.0

    has_grad = any(param.grad is not None for param in moe.parameters())
    assert has_grad

    dist.destroy_process_group()


def test_expert_parallel_moe_all_to_all_smoke() -> None:
    """ExpertParallelMoE forward/backward should run across two Gloo ranks."""
    world_size = 2
    port = _free_port()
    mp.spawn(_worker, args=(world_size, port), nprocs=world_size, join=True)
