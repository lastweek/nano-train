#!/usr/bin/env python3
"""
TP/DP tutorial script that reuses runtime-core orchestration.

This script demonstrates one canonical training pipeline with three primary modes:
1) single rank: world_size=1
2) TP-only: dp_size=1 and tp_size>1
3) TP+DP: dp_size>1 and tp_size>1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import sys
import warnings
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
from src.distributed.device import get_backend
from src.distributed.device import get_device
from src.layers import ColumnParallelLinear
from src.layers import RowParallelLinear
from src.logging import get_logger
from src.logging import setup_logging
from src.runtime.checkpoint import NoOpCheckpointManager
from src.runtime.context import RunConfig
from src.runtime.context import RuntimeContext
from src.runtime.contracts import OptimizerState
from src.runtime.contracts import RuntimeBootstrap
from src.runtime.contracts import RuntimeComponents
from src.runtime.contracts import ScheduleStrategy
from src.runtime.contracts import StepContext
from src.runtime.contracts import StepOutput
from src.runtime.contracts import TrainDataBundle
from src.runtime.engine import RuntimeEngine
from src.runtime.schedules.non_pipeline import NonPipelineSchedule
from src.runtime.sync import ParamShardInfo


logger = get_logger("tp")


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
        """Run TP MLP forward pass."""
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class TutorialModel(nn.Module):
    """Small model for demonstrating TP/DP behavior."""

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
        """Run tutorial model forward pass."""
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
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        default=None,
        help="Tensor model parallel size. Default: world_size",
    )
    parser.add_argument("--tp_size", type=int, default=None, help=argparse.SUPPRESS)
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
    args = parser.parse_args()

    if args.tensor_model_parallel_size is None and args.tp_size is not None:
        warnings.warn(
            "--tp_size is deprecated; use --tensor-model-parallel-size",
            DeprecationWarning,
            stacklevel=2,
        )
        args.tensor_model_parallel_size = args.tp_size
    elif args.tensor_model_parallel_size is not None and args.tp_size is not None:
        warnings.warn(
            "Both --tp_size and --tensor-model-parallel-size were provided. "
            "Using canonical --tensor-model-parallel-size.",
            DeprecationWarning,
            stacklevel=2,
        )

    return args


def setup_process_logging(rank: int) -> None:
    """Configure process-local logging using shared logging utilities."""
    level = "INFO" if rank == 0 else "WARNING"
    setup_logging(log_level=level, use_colors=False)


@dataclass
class TPParallelContext:
    """Resolved TP/DP runtime topology used by this example."""

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


def init_parallel(tp_size: int) -> TPParallelContext:
    """Initialize distributed process groups and compute TP/DP ranks."""
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
    if distributed:
        dist.init_process_group(backend=backend, world_size=world_size, rank=rank)

    dp_size = world_size // tp_size
    tp_rank = rank % tp_size
    dp_rank = rank // tp_size

    tp_group = None
    dp_group = None

    if distributed and tp_size > 1:
        for dp_idx in range(dp_size):
            tp_ranks = [dp_idx * tp_size + i for i in range(tp_size)]
            group = dist.new_group(tp_ranks)
            if dp_idx == dp_rank:
                tp_group = group

    if distributed and dp_size > 1:
        for tp_idx in range(tp_size):
            dp_ranks = [i * tp_size + tp_idx for i in range(dp_size)]
            group = dist.new_group(dp_ranks)
            if tp_idx == tp_rank:
                dp_group = group

    return TPParallelContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        context_parallel_size=1,
        data_parallel_size=dp_size,
        expert_data_parallel_size=dp_size,
        tensor_model_parallel_rank=tp_rank,
        pipeline_model_parallel_rank=0,
        expert_model_parallel_rank=0,
        context_parallel_rank=0,
        data_parallel_rank=dp_rank,
        tensor_model_parallel_group=tp_group,
        pipeline_model_parallel_group=None,
        expert_model_parallel_group=None,
        context_parallel_group=None,
        data_parallel_group=dp_group,
        expert_data_parallel_group=dp_group,
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
    dp_group: Optional[object],
) -> None:
    """Synchronize initial params so TP shards and DP replicas match deterministic startup."""
    if world_size == 1:
        return

    for param in model.parameters():
        if id(param) in tp_sharded_param_ids:
            if dp_size > 1:
                dist.broadcast(param.data, src=tp_rank, group=dp_group)
        else:
            dist.broadcast(param.data, src=0)

    for buffer in model.buffers():
        dist.broadcast(buffer.data, src=0)


def synchronize_gradients(
    model: nn.Module,
    dp_size: int,
    dp_group: Optional[object],
) -> None:
    """Average gradients across DP ranks while TP collectives stay layer-local."""
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
) -> TrainDataBundle:
    """Create a dataloader with DP-aware sharding."""
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
    return TrainDataBundle(loader=loader, sampler=sampler)


@dataclass
class TPBootstrap(RuntimeBootstrap):
    """Build runtime context for TP tutorial args."""

    def build_context(self, args: argparse.Namespace) -> RuntimeContext:
        """Resolve TP/DP topology and create runtime context."""
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        tp_size = (
            args.tensor_model_parallel_size
            if args.tensor_model_parallel_size is not None
            else world_size
        )
        parallel = init_parallel(tp_size)

        if parallel.world_size == 1:
            mode = "single"
        elif parallel.data_parallel_size == 1:
            mode = "tp_only"
        else:
            mode = "tp_dp"

        return RuntimeContext(
            parallel=parallel,
            mode=mode,
            run_config=RunConfig(args=args, pp_layer_splits=None),
        )


class TPModelProvider:
    """Build TP tutorial model and synchronize initial shards/replicas."""

    def build_model(self, ctx: RuntimeContext) -> torch.nn.Module:
        """Build and synchronize TP tutorial model parameters."""
        args = ctx.run_config.args
        parallel = ctx.parallel

        torch.manual_seed(args.seed + parallel.tensor_model_parallel_rank)
        if parallel.device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + parallel.tensor_model_parallel_rank)

        model = TutorialModel(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            output_size=1,
            tp_rank=parallel.tensor_model_parallel_rank,
            tp_size=parallel.tensor_model_parallel_size,
            tp_group=parallel.tensor_model_parallel_group,
        ).to(parallel.device)

        tp_sharded_param_ids = collect_tp_sharded_param_ids(model)
        synchronize_initial_parameters(
            model=model,
            tp_sharded_param_ids=tp_sharded_param_ids,
            world_size=parallel.world_size,
            dp_size=parallel.data_parallel_size,
            tp_rank=parallel.tensor_model_parallel_rank,
            dp_group=parallel.data_parallel_group,
        )

        if parallel.rank == 0:
            local_params = sum(p.numel() for p in model.parameters())
            logger.info("Local params: %d", local_params)
            logger.info(
                "Starting training: epochs=%d, max_steps=%d",
                args.epochs,
                args.max_steps,
            )

        return model


class TPDataProvider:
    """Build deterministic TP tutorial training data."""

    def build_train_data(self, ctx: RuntimeContext) -> TrainDataBundle:
        """Build deterministic dataset and DP-aware data loader."""
        args = ctx.run_config.args
        parallel = ctx.parallel
        dataset = DummyDataset(
            num_samples=args.num_samples,
            input_size=args.input_size,
            seed=args.seed,
        )
        return create_data_loader(
            dataset=dataset,
            batch_size=args.batch_size,
            dp_size=parallel.data_parallel_size,
            dp_rank=parallel.data_parallel_rank,
            seed=args.seed,
        )


class TPOptimizerRuntime:
    """Optimizer policy for TP tutorial."""

    def initialize(self, model: torch.nn.Module, ctx: RuntimeContext) -> OptimizerState:
        """Initialize AdamW optimizer state for TP tutorial training."""
        args = ctx.run_config.args
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        return OptimizerState(
            optimizer=optimizer,
            shard_info=ParamShardInfo(set(), set()),
        )

    def zero_grad(self, state: OptimizerState) -> None:
        """Zero optimizer gradients before each training step."""
        zero_grad = getattr(state.optimizer, "zero_grad")
        zero_grad(set_to_none=True)

    def step(
        self,
        *,
        model: torch.nn.Module,
        state: OptimizerState,
        ctx: RuntimeContext,
    ) -> None:
        """Run DP gradient synchronization and then optimizer step."""
        synchronize_gradients(
            model=model,
            dp_size=ctx.parallel.data_parallel_size,
            dp_group=ctx.parallel.data_parallel_group,
        )
        step = getattr(state.optimizer, "step")
        step()


@dataclass
class TPStepSchedule:
    """Execute one TP tutorial train step."""

    optimizer_runtime: TPOptimizerRuntime

    def run_step(self, step_ctx: StepContext) -> StepOutput:
        """Execute one MSE train step for the TP tutorial model."""
        if step_ctx.batch is None:
            raise RuntimeError("TP schedule requires a batch")

        device = step_ctx.runtime_context.parallel.device
        data = step_ctx.batch["data"].to(device)
        target = step_ctx.batch["target"].to(device)

        self.optimizer_runtime.zero_grad(step_ctx.optimizer_state)
        output = step_ctx.model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        self.optimizer_runtime.step(
            model=step_ctx.model,
            state=step_ctx.optimizer_state,
            ctx=step_ctx.runtime_context,
        )

        return StepOutput(
            task_loss=float(loss.item()),
            aux_loss=0.0,
            total_loss=float(loss.item()),
            drop_fraction=0.0,
            counters={"objective_count": 1, "drop_count": 1},
        )


@dataclass
class TPScheduleSelector:
    """Always choose non-pipeline schedule for TP tutorial."""

    schedule: ScheduleStrategy

    def select(self, ctx: RuntimeContext) -> ScheduleStrategy:
        """Return the single non-pipeline schedule used by this script."""
        del ctx
        return self.schedule


def build_tp_components() -> RuntimeComponents:
    """Build TP tutorial runtime components."""
    optimizer_runtime = TPOptimizerRuntime()
    schedule = NonPipelineSchedule(step_fn=TPStepSchedule(optimizer_runtime).run_step)
    return RuntimeComponents(
        bootstrap=TPBootstrap(),
        model_provider=TPModelProvider(),
        data_provider=TPDataProvider(),
        optimizer_runtime=optimizer_runtime,
        schedule_selector=TPScheduleSelector(schedule=schedule),
        checkpoint_manager=NoOpCheckpointManager(),
    )


def main() -> None:
    """Run training in single-rank, TP-only, or TP+DP mode."""
    args = parse_args()

    rank_pre = int(os.environ.get("RANK", "0"))
    setup_process_logging(rank_pre)

    engine = RuntimeEngine()
    engine.run(components=build_tp_components(), args=args)


if __name__ == "__main__":
    main()
