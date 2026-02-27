#!/usr/bin/env python3
"""Simple distributed training script using runtime-core orchestration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import sys
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler

# Add parent directory to path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.distributed.device import get_backend
from src.distributed.device import get_device
from src.distributed.device import get_device_info
from src.layers import Linear
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
from src.runtime.mixed_precision import apply_model_precision_plan
from src.runtime.mixed_precision import build_model_precision_plan
from src.runtime.mixed_precision import dtype_alias_to_torch
from src.runtime.mixed_precision import ensure_lowbit_compute_assignments
from src.runtime.mixed_precision import MixedPrecisionController
from src.runtime.mixed_precision import refresh_persistent_lowbit_params
from src.runtime.mixed_precision import resolve_precision_config
from src.runtime.optimizer_runtime import PrecisionAdamW
from src.runtime.schedules.non_pipeline import NonPipelineSchedule
from src.runtime.sync import ParamShardInfo


logger = get_logger(__name__)


class SimpleModel(nn.Module):
    """Simple model for testing distributed training."""

    def __init__(
        self,
        hidden_size: int = 128,
        *,
        param_dtype: torch.dtype,
        param_device: Optional[torch.device],
    ):
        super().__init__()
        self.net = nn.Sequential(
            Linear(
                10,
                hidden_size,
                param_dtype=param_dtype,
                param_device=param_device,
            ),
            nn.ReLU(),
            Linear(
                hidden_size,
                hidden_size,
                param_dtype=param_dtype,
                param_device=param_device,
            ),
            nn.ReLU(),
            Linear(
                hidden_size,
                1,
                param_dtype=param_dtype,
                param_device=param_device,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run model forward pass."""
        return self.net(x)


class DummyDataset(Dataset):
    """Deterministic dummy dataset for DDP tutorial."""

    def __init__(self, num_samples: int):
        generator = torch.Generator().manual_seed(1234)
        self.data = torch.randn(num_samples, 10, generator=generator)
        self.targets = torch.randn(num_samples, 1, generator=generator)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"data": self.data[idx], "target": self.targets[idx]}


@dataclass
class DDPParallelContext:
    """Resolved DDP runtime topology used by this example."""

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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the DDP tutorial."""
    parser = argparse.ArgumentParser(description="Simple DDP tutorial with runtime core")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--log_every", type=int, default=0)
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 mixed precision")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision")
    parser.add_argument("--fp8", action="store_true", help="Enable FP8 mixed precision")
    parser.add_argument("--fp4", action="store_true", help="Enable FP4 mixed precision (emulated)")
    parser.add_argument(
        "--fp8-backend",
        type=str,
        default="transformer_engine",
        choices=["transformer_engine", "emulated"],
        help="FP8 backend implementation",
    )
    parser.add_argument(
        "--fp8-format",
        type=str,
        default="e4m3",
        choices=["e4m3", "hybrid"],
        help="FP8 format recipe",
    )
    parser.add_argument(
        "--fp8-amax-history-len",
        type=int,
        default=16,
        help="FP8 amax history length",
    )
    parser.add_argument(
        "--fp8-amax-compute-algo",
        type=str,
        default="most_recent",
        choices=["most_recent", "max"],
        help="FP8 amax compute algorithm",
    )
    parser.add_argument(
        "--fp4-backend",
        type=str,
        default="emulated",
        choices=["emulated"],
        help="FP4 backend implementation",
    )
    parser.add_argument(
        "--params-dtype",
        type=str,
        default=None,
        choices=["fp32", "bf16", "fp16"],
        help="Model parameter storage dtype",
    )
    parser.add_argument(
        "--main-params-dtype",
        type=str,
        default=None,
        choices=["fp32", "bf16", "fp16"],
        help="Main optimizer parameter dtype",
    )
    parser.add_argument(
        "--main-grads-dtype",
        type=str,
        default=None,
        choices=["fp32", "bf16", "fp16"],
        help="Main optimizer gradient dtype",
    )
    parser.add_argument(
        "--exp-avg-dtype",
        type=str,
        default=None,
        choices=["fp32", "bf16", "fp16"],
        help="Adam exp_avg state dtype",
    )
    parser.add_argument(
        "--exp-avg-sq-dtype",
        type=str,
        default=None,
        choices=["fp32", "bf16", "fp16"],
        help="Adam exp_avg_sq state dtype",
    )
    parser.add_argument(
        "--loss-scale-init",
        type=float,
        default=65536.0,
        help="Initial dynamic loss scale value",
    )
    parser.add_argument(
        "--loss-scale-growth-factor",
        type=float,
        default=2.0,
        help="Loss scale growth factor",
    )
    parser.add_argument(
        "--loss-scale-backoff-factor",
        type=float,
        default=0.5,
        help="Loss scale backoff factor on overflow",
    )
    parser.add_argument(
        "--loss-scale-growth-interval",
        type=int,
        default=2000,
        help="Successful step interval before loss scale growth",
    )
    parser.add_argument(
        "--loss-scale-min",
        type=float,
        default=1.0,
        help="Minimum loss scale",
    )
    parser.add_argument(
        "--loss-scale-max",
        type=float,
        default=16777216.0,
        help="Maximum loss scale",
    )
    parser.add_argument(
        "--fp8-param",
        action="store_true",
        help="Enable persistent FP8 parameter storage for selected modules",
    )
    parser.add_argument(
        "--fp4-param",
        action="store_true",
        help="Enable persistent FP4 parameter storage for selected modules",
    )
    parser.add_argument(
        "--fp4-param-format",
        type=str,
        default="nf4",
        choices=["nf4"],
        help="Persistent FP4 quantization format",
    )
    parser.add_argument(
        "--persistent-scale-granularity",
        type=str,
        default="per_channel",
        choices=["per_tensor", "per_channel"],
        help="Scale granularity for persistent low-bit quantization",
    )
    parser.add_argument(
        "--module-pattern-type",
        type=str,
        default="regex",
        choices=["regex", "glob"],
        help="Pattern matcher type for module precision policies",
    )
    parser.add_argument(
        "--compute-lowbit-mode",
        type=str,
        default=None,
        choices=["fp8", "fp4"],
        help="Per-module low-bit compute mode override",
    )
    parser.add_argument(
        "--compute-lowbit-include",
        action="append",
        default=None,
        help="Repeatable include patterns for low-bit compute module selection",
    )
    parser.add_argument(
        "--compute-lowbit-exclude",
        action="append",
        default=None,
        help="Repeatable exclude patterns for low-bit compute module selection",
    )
    parser.add_argument(
        "--persistent-lowbit-mode",
        type=str,
        default="off",
        choices=["off", "fp8", "fp4"],
        help="Per-module persistent low-bit storage mode",
    )
    parser.add_argument(
        "--persistent-lowbit-include",
        action="append",
        default=None,
        help="Repeatable include patterns for persistent low-bit module selection",
    )
    parser.add_argument(
        "--persistent-lowbit-exclude",
        action="append",
        default=None,
        help="Repeatable exclude patterns for persistent low-bit module selection",
    )
    return parser.parse_args()


def setup_process_logging(rank: int) -> None:
    """Configure process-local logging using shared logging utilities."""
    level = "INFO" if rank == 0 else "WARNING"
    setup_logging(log_level=level, use_colors=False)


@dataclass
class DDPBootstrap(RuntimeBootstrap):
    """Build runtime context for DDP tutorial."""

    def build_context(self, args: argparse.Namespace) -> RuntimeContext:
        """Initialize distributed process state and construct runtime context."""
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        backend = get_backend()
        if world_size > 1:
            dist.init_process_group(backend=backend, world_size=world_size, rank=rank)

        device_info = get_device_info()
        device = get_device(device_info.device_type, local_rank)
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)

        parallel = DDPParallelContext(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            context_parallel_size=1,
            data_parallel_size=world_size,
            expert_data_parallel_size=world_size,
            tensor_model_parallel_rank=0,
            pipeline_model_parallel_rank=0,
            expert_model_parallel_rank=0,
            context_parallel_rank=0,
            data_parallel_rank=rank,
            tensor_model_parallel_group=None,
            pipeline_model_parallel_group=None,
            expert_model_parallel_group=None,
            context_parallel_group=None,
            data_parallel_group=dist.group.WORLD if dist.is_initialized() else None,
            expert_data_parallel_group=dist.group.WORLD if dist.is_initialized() else None,
        )

        if rank == 0:
            logger.info("Initializing process group...")
            logger.info("  Rank: %d/%d", rank, world_size)
            logger.info("  Local Rank: %d", local_rank)
            logger.info("  Backend: %s", backend)
            logger.info("  Device: %s", device)

        precision_config = resolve_precision_config(args, device)

        return RuntimeContext(
            parallel=parallel,
            mode="ddp" if world_size > 1 else "single",
            run_config=RunConfig(
                args=args,
                pp_layer_splits=None,
                precision_config=precision_config,
            ),
        )


class DDPModelProvider:
    """Build and wrap DDP model."""

    def build_model(self, ctx: RuntimeContext) -> torch.nn.Module:
        """Build model and wrap with DDP when world size is greater than one."""
        args = ctx.run_config.args
        parallel = ctx.parallel
        precision_config = ctx.run_config.precision_config
        if precision_config is None:
            raise RuntimeError("precision_config must be resolved in DDPBootstrap")

        model: torch.nn.Module = SimpleModel(
            hidden_size=args.hidden_size,
            param_dtype=dtype_alias_to_torch(precision_config.params_dtype),
            param_device=parallel.device,
        )
        policy = precision_config.module_precision_policy
        if policy is not None:
            precision_plan = build_model_precision_plan(model, policy)
            apply_model_precision_plan(model, precision_plan)
            ensure_lowbit_compute_assignments(
                precision_config,
                precision_plan,
                script_name="train_ddp.py",
            )
            if parallel.rank == 0:
                logger.info(
                    "DDP per-module low-bit policy: compute_modules=%d "
                    "persistent_modules=%d",
                    precision_plan.compute_lowbit_module_count,
                    precision_plan.persistent_lowbit_module_count,
                )
        refresh_persistent_lowbit_params(model)

        if parallel.world_size > 1:
            if parallel.device.type == "cuda":
                model = DDP(model, device_ids=[parallel.local_rank])
            else:
                model = DDP(model)

        if parallel.rank == 0:
            logger.info("Model wrapped with DDP")
            if hasattr(model, "device_ids"):
                logger.info("  Device IDs: %s", model.device_ids)

        return model


class DDPDataProvider:
    """Build DDP train data loader and sampler."""

    def build_train_data(self, ctx: RuntimeContext) -> TrainDataBundle:
        """Build deterministic dataset and distributed sampler-backed loader."""
        args = ctx.run_config.args
        parallel = ctx.parallel

        dataset = DummyDataset(num_samples=args.num_samples)
        sampler: Optional[DistributedSampler] = None
        if parallel.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=parallel.world_size,
                rank=parallel.rank,
                shuffle=True,
                seed=42,
            )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=0,
        )
        return TrainDataBundle(loader=dataloader, sampler=sampler)


class DDPOptimizerRuntime:
    """Optimizer policy for DDP tutorial."""

    def initialize(self, model: torch.nn.Module, ctx: RuntimeContext) -> OptimizerState:
        """Initialize AdamW optimizer state for DDP tutorial training."""
        args = ctx.run_config.args
        precision_config = ctx.run_config.precision_config
        if precision_config is None:
            raise RuntimeError("precision_config must be resolved in DDPBootstrap")

        optimizer = PrecisionAdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.0,
            betas=(0.9, 0.999),
            eps=1e-8,
            main_params_dtype=precision_config.main_params_dtype,
            main_grads_dtype=precision_config.main_grads_dtype,
            exp_avg_dtype=precision_config.exp_avg_dtype,
            exp_avg_sq_dtype=precision_config.exp_avg_sq_dtype,
        )
        precision_controller = MixedPrecisionController(
            precision_config,
            device=ctx.parallel.device,
        )
        return OptimizerState(
            optimizer=optimizer,
            shard_info=ParamShardInfo(set(), set()),
            extra_state={"precision_controller": precision_controller},
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
        """Apply one optimizer step; DDP handles gradient synchronization."""
        del ctx
        step = getattr(state.optimizer, "step")
        step()
        refresh_persistent_lowbit_params(model)


@dataclass
class DDPStepSchedule:
    """Execute one DDP tutorial train step."""

    optimizer_runtime: DDPOptimizerRuntime

    def run_step(self, step_ctx: StepContext) -> StepOutput:
        """Execute one MSE train step for the DDP tutorial model."""
        if step_ctx.batch is None:
            raise RuntimeError("DDP schedule requires a batch")

        device = step_ctx.runtime_context.parallel.device
        data = step_ctx.batch["data"].to(device)
        target = step_ctx.batch["target"].to(device)
        args = step_ctx.runtime_context.run_config.args
        precision_controller = step_ctx.optimizer_state.extra_state.get("precision_controller")
        if precision_controller is not None and not isinstance(
            precision_controller, MixedPrecisionController
        ):
            raise TypeError("precision_controller must be MixedPrecisionController when provided")

        self.optimizer_runtime.zero_grad(step_ctx.optimizer_state)
        if precision_controller is None:
            output = step_ctx.model(data)
            loss = nn.functional.mse_loss(output, target)
            loss.backward()
            should_step = True
        else:
            with precision_controller.autocast_context():
                output = step_ctx.model(data)
                loss = nn.functional.mse_loss(output, target)
            precision_controller.backward(loss)
            should_step = precision_controller.prepare_optimizer_step(step_ctx.model)

        if should_step:
            self.optimizer_runtime.step(
                model=step_ctx.model,
                state=step_ctx.optimizer_state,
                ctx=step_ctx.runtime_context,
            )

        if precision_controller is not None:
            precision_controller.update_after_step(step_applied=should_step)
            if (
                step_ctx.runtime_context.parallel.rank == 0
                and args.log_every > 0
                and step_ctx.train_state.global_step % args.log_every == 0
                and precision_controller.uses_loss_scaling
            ):
                logger.info(
                    "precision step=%d mode=%s loss_scale=%.4f skipped_steps=%d",
                    step_ctx.train_state.global_step,
                    precision_controller.config.mode,
                    precision_controller.runtime_state.loss_scale,
                    precision_controller.runtime_state.skipped_steps,
                )

        return StepOutput(
            task_loss=float(loss.item()),
            aux_loss=0.0,
            total_loss=float(loss.item()),
            drop_fraction=0.0,
            counters={"objective_count": 1, "drop_count": 1},
        )


@dataclass
class DDPScheduleSelector:
    """Always choose non-pipeline schedule for DDP tutorial."""

    schedule: ScheduleStrategy

    def select(self, ctx: RuntimeContext) -> ScheduleStrategy:
        """Return the single non-pipeline schedule used by this script."""
        del ctx
        return self.schedule


def build_ddp_components() -> RuntimeComponents:
    """Build DDP tutorial runtime components."""
    optimizer_runtime = DDPOptimizerRuntime()
    schedule = NonPipelineSchedule(step_fn=DDPStepSchedule(optimizer_runtime).run_step)
    return RuntimeComponents(
        bootstrap=DDPBootstrap(),
        model_provider=DDPModelProvider(),
        data_provider=DDPDataProvider(),
        optimizer_runtime=optimizer_runtime,
        schedule_selector=DDPScheduleSelector(schedule=schedule),
        checkpoint_manager=NoOpCheckpointManager(),
    )


def main() -> None:
    """Main training loop entrypoint."""
    args = parse_args()

    rank_pre = int(os.environ.get("RANK", "0"))
    setup_process_logging(rank_pre)

    if rank_pre == 0:
        logger.info("=" * 60)
        logger.info("Distributed DDP Training Test")
        logger.info("=" * 60)

    engine = RuntimeEngine()
    engine.run(components=build_ddp_components(), args=args)


if __name__ == "__main__":
    main()
