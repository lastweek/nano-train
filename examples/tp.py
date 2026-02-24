#!/usr/bin/env python3
"""
TP/DP tutorial script that reuses nano-train building blocks.

This script demonstrates one canonical training pipeline with three modes:
1) single rank: world_size=1
2) TP-only: world_size>1 and tp_size=world_size
3) TP+DP: world_size>1 and 1<tp_size<world_size

Core learning reference:
    docs/tp_dp_communication.md
"""

import argparse
import os
import sys
from typing import Optional, Set

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DistributedSampler

# Add parent directory to path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import create_dataloader
from src.distributed.device import get_backend, get_device
from src.layers import ColumnParallelLinear, RowParallelLinear
from src.logging import get_logger, setup_logging


logger = get_logger("tp")

# TP/DP communication cheat sheet used in this file:
# - Row-parallel: forward all-reduce(sum) on Y within TP group.
# - Column-parallel: backward all-reduce(sum) on dX within TP group.
# - DP: all-reduce(avg) on parameter gradients across dp ranks with same tp_rank.


class ParallelMLP(nn.Module):
    """MLP block using shared ColumnParallelLinear and RowParallelLinear."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_rank: int,
        tp_size: int,
        tp_group,
    ):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            bias=True,
        )
        self.fc2 = RowParallelLinear(
            intermediate_size,
            hidden_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class TutorialModel(nn.Module):
    """
    Small model for demonstrating TP/DP behavior.

    Replicated input projection -> TP MLP -> replicated output projection.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        tp_rank: int,
        tp_size: int,
        tp_group,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.mlp = ParallelMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
        )
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = F.gelu(x)
        x = self.mlp(x)
        x = self.output_proj(x)
        return x


class DummyDataset(Dataset):
    """Deterministic regression dataset used to visualize TP/DP communication."""

    def __init__(self, num_samples: int, input_size: int, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.data = torch.randn(num_samples, input_size, generator=generator)
        self.target = torch.randn(num_samples, 1, generator=generator)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "data": self.data[index],
            "target": self.target[index],
        }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Canonical TP/DP tutorial script")
    parser.add_argument("--tp_size", type=int, default=None, help="TP size. Default: world_size")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--input_size", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--intermediate_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def setup_process_logging(rank: int) -> None:
    """Configure process-local logging using shared logging utilities."""
    level = "INFO" if rank == 0 else "WARNING"
    setup_logging(log_level=level, use_colors=False)


def init_parallel(tp_size: int):
    """
    Initialize distributed process groups and compute 2D ranks.

    Returns:
        rank, world_size, local_rank, device, tp_size, dp_size, tp_rank, dp_rank, tp_group, dp_group
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    if tp_size < 1:
        raise ValueError(f"tp_size must be >= 1, got {tp_size}")
    if world_size % tp_size != 0:
        raise ValueError(f"world_size ({world_size}) must be divisible by tp_size ({tp_size})")

    backend = get_backend()
    device_type = "cuda" if backend == "nccl" else "cpu"
    device = get_device(device_type, local_rank)

    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

    distributed = world_size > 1
    tp_group = None
    dp_group = None

    if distributed:
        dist.init_process_group(backend=backend, world_size=world_size, rank=rank)

    dp_size = world_size // tp_size
    tp_rank = rank % tp_size
    dp_rank = rank // tp_size

    # TP groups: fixed dp_rank, varying tp_rank (intra-replica model-shard communication).
    if distributed and tp_size > 1:
        for dp_idx in range(dp_size):
            tp_ranks = [dp_idx * tp_size + i for i in range(tp_size)]
            group = dist.new_group(tp_ranks)
            if dp_idx == dp_rank:
                tp_group = group

    # DP groups: fixed tp_rank, varying dp_rank (cross-replica gradient averaging).
    if distributed and dp_size > 1:
        for tp_idx in range(tp_size):
            dp_ranks = [i * tp_size + tp_idx for i in range(dp_size)]
            group = dist.new_group(dp_ranks)
            if tp_idx == tp_rank:
                dp_group = group

    return (
        rank,
        world_size,
        local_rank,
        device,
        tp_size,
        dp_size,
        tp_rank,
        dp_rank,
        tp_group,
        dp_group,
    )


def collect_tp_sharded_param_ids(model: nn.Module) -> Set[int]:
    """Collect parameter ids that are TP-sharded by Column/Row parallel layers."""
    sharded: Set[int] = set()
    for module in model.modules():
        if isinstance(module, ColumnParallelLinear):
            for param in module.parameters(recurse=False):
                sharded.add(id(param))
        elif isinstance(module, RowParallelLinear):
            sharded.add(id(module.weight))
    return sharded


def synchronize_initial_parameters(
    model: nn.Module,
    tp_sharded_param_ids: Set[int],
    world_size: int,
    dp_size: int,
    tp_rank: int,
    dp_group,
) -> None:
    """Synchronize initial params so TP shards and DP replicas start from canonical weights."""
    if world_size == 1:
        return

    for param in model.parameters():
        if id(param) in tp_sharded_param_ids:
            if dp_size > 1:
                # Same TP shard index across DP replicas must start identical.
                dist.broadcast(param.data, src=tp_rank, group=dp_group)
        else:
            # Replicated parameters should match across all ranks.
            dist.broadcast(param.data, src=0)

    for buffer in model.buffers():
        dist.broadcast(buffer.data, src=0)


def synchronize_gradients(model: nn.Module, dp_size: int, dp_group) -> None:
    """
    Synchronize gradients in the canonical TP+DP setup.

    TP collectives for activation-gradient correctness already happen inside TP layers:
    - Row-parallel all-reduces Y in forward.
    - Column-parallel all-reduces dX in backward.

    Cross-replica consistency is a DP concern, so here we only average parameter gradients across DP.
    """
    if dp_size <= 1:
        return

    for param in model.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=dp_group)
        param.grad.div_(dp_size)


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


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    dp_size: int,
    dp_group,
) -> float:
    """One training step with explicit TP/DP communication."""
    data = batch["data"].to(device)
    target = batch["target"].to(device)

    optimizer.zero_grad(set_to_none=True)
    output = model(data)
    loss = F.mse_loss(output, target)
    loss.backward()

    # Loss all-reduce is optional for logging only; correctness relies on gradient synchronization.
    synchronize_gradients(model=model, dp_size=dp_size, dp_group=dp_group)

    optimizer.step()
    return float(loss.item())


def aggregate_loss(loss_sum: float, count: int, device: torch.device, world_size: int) -> float:
    """Aggregate average loss across all ranks for logging."""
    if world_size == 1:
        return loss_sum / max(1, count)

    stats = torch.tensor([loss_sum, float(count)], dtype=torch.float64, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float(stats[0].item() / max(1.0, stats[1].item()))


def main() -> None:
    """Run training in single-rank, TP-only, or TP+DP mode."""
    args = parse_args()

    rank_pre = int(os.environ.get("RANK", "0"))
    setup_process_logging(rank_pre)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    tp_size = args.tp_size if args.tp_size is not None else world_size

    (
        rank,
        world_size,
        _,
        device,
        tp_size,
        dp_size,
        tp_rank,
        dp_rank,
        tp_group,
        dp_group,
    ) = init_parallel(tp_size)

    mode = "single" if world_size == 1 else ("tp_only" if dp_size == 1 else "tp_dp")

    if rank == 0:
        logger.info("=" * 60)
        logger.info("Canonical TP/DP Tutorial")
        logger.info("=" * 60)
        logger.info("Mode: %s", mode)
        logger.info("World Size: %d | TP Size: %d | DP Size: %d", world_size, tp_size, dp_size)
        logger.info("Backend: %s | Device: %s", dist.get_backend() if world_size > 1 else "none", device)
        logger.info("=" * 60)

    # Seed by tp_rank so TP shards differ, DP replicas of same shard index match.
    torch.manual_seed(args.seed + tp_rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + tp_rank)

    model = TutorialModel(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        output_size=1,
        tp_rank=tp_rank,
        tp_size=tp_size,
        tp_group=tp_group,
    ).to(device)

    tp_sharded_param_ids = collect_tp_sharded_param_ids(model)
    synchronize_initial_parameters(
        model=model,
        tp_sharded_param_ids=tp_sharded_param_ids,
        world_size=world_size,
        dp_size=dp_size,
        tp_rank=tp_rank,
        dp_group=dp_group,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    dataset = DummyDataset(
        num_samples=args.num_samples,
        input_size=args.input_size,
        seed=args.seed,
    )
    train_loader, sampler = create_data_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        dp_size=dp_size,
        dp_rank=dp_rank,
        seed=args.seed,
    )

    if rank == 0:
        local_params = sum(p.numel() for p in model.parameters())
        logger.info("Local params: %d", local_params)
        logger.info("Starting training: epochs=%d, max_steps=%d", args.epochs, args.max_steps)

    model.train()
    step = 0

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        loss_sum = 0.0
        batch_count = 0

        for batch in train_loader:
            if step >= args.max_steps:
                break

            loss = train_step(
                model=model,
                optimizer=optimizer,
                batch=batch,
                device=device,
                dp_size=dp_size,
                dp_group=dp_group,
            )
            loss_sum += loss
            batch_count += 1

            if rank == 0 and step % args.log_every == 0:
                logger.info("step=%d loss=%.6f", step, loss)

            step += 1

        if batch_count > 0:
            global_avg_loss = aggregate_loss(loss_sum, batch_count, device, world_size)
            if rank == 0:
                logger.info("epoch=%d avg_loss=%.6f", epoch + 1, global_avg_loss)

        if step >= args.max_steps:
            break

    if rank == 0:
        logger.info("Training completed")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
