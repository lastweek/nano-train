"""Simple training script for MVP using runtime-core orchestration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import sys
import time
from typing import Optional

import torch
from torch.utils.data import random_split

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.dataset import TextDataset
from src.dataset import create_dataloader
from src.logging import get_logger
from src.logging import setup_logging
from src.models.transformer import TransformerModel
from src.runtime.context import RunConfig
from src.runtime.context import RuntimeContext
from src.runtime.context import TrainState
from src.runtime.checkpoint import NoOpCheckpointManager
from src.runtime.contracts import OptimizerState
from src.runtime.contracts import ResumeState
from src.runtime.contracts import RuntimeBootstrap
from src.runtime.contracts import RuntimeComponents
from src.runtime.contracts import ScheduleStrategy
from src.runtime.contracts import StepContext
from src.runtime.contracts import StepOutput
from src.runtime.contracts import TrainDataBundle
from src.runtime.engine import RuntimeEngine
from src.runtime.mixed_precision import build_module_precision_resolver
from src.runtime.mixed_precision import dtype_alias_to_torch
from src.runtime.mixed_precision import finalize_module_precision_resolver
from src.runtime.mixed_precision import MixedPrecisionController
from src.runtime.mixed_precision import refresh_persistent_lowbit_params
from src.runtime.master_store import materialize_optimizer_owned_masters
from src.runtime.precision_args import add_mixed_precision_args
from src.runtime.precision_args import normalize_and_resolve_precision
from src.runtime.schedules.non_pipeline import NonPipelineSchedule
from src.runtime.sync import ParamShardInfo
from src.trainer import Trainer
from src.trainer import _TrainTotals
from src.trainer import _TrainWindow
from src.utils import dump_model_info


# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)


def resolve_dataset_path(dataset_path: str) -> str:
    """Resolve dataset path relative to the examples directory."""
    if os.path.isabs(dataset_path):
        return dataset_path
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.path.basename(dataset_path),
    )


def print_config(config: Config) -> None:
    """Print model and training configuration."""
    logger.info("=" * 50)
    logger.info("Nano-Train MVP - First Training Cycle")
    logger.info("=" * 50)

    logger.info("Model config:")
    logger.info("  Hidden size: %d", config.model.hidden_size)
    logger.info("  Num layers: %d", config.model.num_layers)
    logger.info("  Num attention heads: %d", config.model.num_attention_heads)
    logger.info("  Intermediate size: %d", config.model.intermediate_size)
    logger.info("  Max position embeddings: %d", config.model.max_position_embeddings)

    logger.info("Training config:")
    logger.info("  Batch size: %d", config.training.batch_size)
    logger.info("  Learning rate: %g", config.training.learning_rate)
    logger.info("  Max steps: %d", config.training.max_steps)
    logger.info("  Warmup steps: %d", config.training.warmup_steps)
    logger.info("  Use BF16: %s", config.training.bf16)


@dataclass
class MVPParallelContext:
    """Single-process runtime topology for MVP script."""

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


@dataclass
class _NoProgress:
    """Tiny tqdm-like stub for Trainer logging hooks."""

    def set_postfix(self, value: object) -> None:
        """Accept tqdm postfix payload and intentionally ignore it."""
        del value


def parse_args() -> argparse.Namespace:
    """Parse optional MVP overrides."""
    parser = argparse.ArgumentParser(description="MVP training with runtime core")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    add_mixed_precision_args(parser)
    return parser.parse_args()


@dataclass
class MVPBootstrap(RuntimeBootstrap):
    """Build runtime context and datasets for MVP script."""

    def build_context(self, args: argparse.Namespace) -> RuntimeContext:
        """Resolve config, prepare loaders, and return runtime context."""
        config = Config()
        if args.max_steps is not None:
            config.training.max_steps = int(args.max_steps)
        if args.seed is not None:
            config.seed = int(args.seed)

        print_config(config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Training device: %s", device)
        precision_config = normalize_and_resolve_precision(args, device)
        precision_resolver = build_module_precision_resolver(precision_config)
        config.training.bf16 = precision_config.mode in ("bf16", "fp8", "fp4")
        config.model.param_dtype = dtype_alias_to_torch(precision_config.params_dtype)
        config.model.param_device = device
        config.model.precision_resolver = precision_resolver

        logger.info("Loading dataset...")
        dataset = TextDataset(
            resolve_dataset_path(config.data.dataset_path),
            max_seq_length=config.data.max_seq_length,
        )
        config.model.vocab_size = dataset.vocab_size

        num_total = len(dataset)
        num_train = int(num_total * float(config.data.train_split))
        num_val = max(0, num_total - num_train)
        if num_train <= 0 or num_val <= 0:
            logger.warning(
                "Dataset too small to split (total=%d, train_split=%.3f); "
                "running without validation.",
                num_total,
                float(config.data.train_split),
            )
            train_dataset = dataset
            val_dataset = None
        else:
            generator = torch.Generator().manual_seed(int(config.seed))
            train_dataset, val_dataset = random_split(
                dataset,
                [num_train, num_val],
                generator=generator,
            )
            logger.info("Dataset split: train=%d, val=%d", len(train_dataset), len(val_dataset))

        train_loader = create_dataloader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
        )
        val_loader = None
        if val_dataset is not None:
            val_loader = create_dataloader(
                val_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
            )

        args._mvp_config = config
        args._mvp_train_loader = train_loader
        args._mvp_val_loader = val_loader
        args._mvp_device = device
        args.epochs = max(1, int(config.training.max_steps))
        args.max_steps = int(config.training.max_steps)
        args.log_every = 0

        parallel = MVPParallelContext(
            rank=0,
            world_size=1,
            local_rank=0,
            device=device,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            context_parallel_size=1,
            data_parallel_size=1,
            expert_data_parallel_size=1,
            tensor_model_parallel_rank=0,
            pipeline_model_parallel_rank=0,
            expert_model_parallel_rank=0,
            context_parallel_rank=0,
            data_parallel_rank=0,
            tensor_model_parallel_group=None,
            pipeline_model_parallel_group=None,
            expert_model_parallel_group=None,
            context_parallel_group=None,
            data_parallel_group=None,
            expert_data_parallel_group=None,
        )
        return RuntimeContext(
            parallel=parallel,
            mode="single",
            run_config=RunConfig(
                args=args,
                pp_layer_splits=None,
                precision_config=precision_config,
            ),
        )


class MVPModelProvider:
    """Build MVP transformer model."""

    def build_model(self, ctx: RuntimeContext) -> torch.nn.Module:
        """Build model and emit model-report diagnostics."""
        config = ctx.run_config.args._mvp_config
        precision_config = ctx.run_config.precision_config
        if precision_config is None:
            raise RuntimeError("precision_config must be resolved in MVPBootstrap")
        logger.info("Creating model...")
        model = TransformerModel(config.model)
        precision_summary = finalize_module_precision_resolver(config.model.precision_resolver)
        master_store = materialize_optimizer_owned_masters(
            model,
            precision_config=precision_config,
        )
        logger.info(
            "MVP per-module low-bit policy: compute_modules=%d persistent_modules=%d "
            "high_precision_exceptions=%d",
            precision_summary.compute_lowbit_module_count,
            precision_summary.persistent_lowbit_module_count,
            precision_summary.high_precision_exception_module_count,
        )
        logger.info(
            "Low-bit master ownership: mode=%s bound_modules=%d",
            precision_config.lowbit_master_ownership,
            0 if master_store is None else len(master_store.metadata),
        )
        refresh_persistent_lowbit_params(model)
        logger.info("Model vocab size: %d", config.model.vocab_size)
        logger.info("Total parameters: %s", f"{model.num_parameters:,}")
        dump_model_info(model, logger=logger, plot_distributions=False)
        return model


class MVPDataProvider:
    """Return prepared train data loader."""

    def build_train_data(self, ctx: RuntimeContext) -> TrainDataBundle:
        """Return prebuilt training loader and validation metadata."""
        args = ctx.run_config.args
        return TrainDataBundle(
            loader=args._mvp_train_loader,
            sampler=None,
            metadata={"val_loader": args._mvp_val_loader},
        )


class MVPOptimizerRuntime:
    """Build Trainer-backed optimizer state for runtime schedule execution."""

    def initialize(self, model: torch.nn.Module, ctx: RuntimeContext) -> OptimizerState:
        """Create Trainer-backed optimizer state and lifecycle bookkeeping."""
        args = ctx.run_config.args
        config = args._mvp_config
        train_loader = args._mvp_train_loader
        val_loader = args._mvp_val_loader
        device = args._mvp_device

        trainer = Trainer(model, config, train_loader, device, val_loader=val_loader)
        precision_config = ctx.run_config.precision_config
        if precision_config is None:
            raise RuntimeError("precision_config must be resolved in MVPBootstrap")
        precision_controller = MixedPrecisionController(
            precision_config,
            device=ctx.parallel.device,
        )

        logger.info("=" * 50)
        logger.info("Starting training...")
        logger.info("=" * 50)
        logger.info("Starting training for %d steps...", config.training.max_steps)
        logger.info("Model parameters: %s", f"{trainer.model.num_parameters:,}")
        logger.info("Device: %s", trainer.device)
        logger.info("Batch size: %d", config.training.batch_size)
        logger.info("Learning rate: %g", config.training.learning_rate)

        extra_state: dict[str, object] = {
            "trainer": trainer,
            "totals": _TrainTotals(),
            "window": _TrainWindow(start_time=time.time()),
            "start_time": time.time(),
            "prev_step_end_time": time.time(),
            "last_loss": None,
            "progress": _NoProgress(),
            "precision_controller": precision_controller,
        }

        return OptimizerState(
            optimizer=trainer.optimizer,
            shard_info=ParamShardInfo(set(), set()),
            extra_state=extra_state,
        )

    def zero_grad(self, state: OptimizerState) -> None:
        """Clear gradients before each runtime-owned step."""
        trainer: Trainer = state.extra_state["trainer"]  # type: ignore[assignment]
        trainer.optimizer.zero_grad(set_to_none=True)

    def step(
        self,
        *,
        model: torch.nn.Module,
        state: OptimizerState,
        ctx: RuntimeContext,
    ) -> None:
        """Apply scheduler/optimizer policy for the pending runtime step."""
        del model, ctx
        trainer: Trainer = state.extra_state["trainer"]  # type: ignore[assignment]
        pending_step = int(state.extra_state.get("pending_step", 0))
        pending_step_applied = bool(state.extra_state.get("pending_step_applied", True))
        trainer.runtime_apply_optimizer_step(
            step=pending_step,
            step_applied=pending_step_applied,
        )


@dataclass
class MVPStepSchedule:
    """Execute one Trainer-equivalent MVP step inside runtime engine."""

    optimizer_runtime: MVPOptimizerRuntime

    def run_step(self, step_ctx: StepContext) -> StepOutput:
        """Run one trainer-parity step and return step-level scalar metrics."""
        if step_ctx.batch is None:
            raise RuntimeError("MVP schedule requires a batch")

        extra_state = step_ctx.optimizer_state.extra_state
        trainer: Trainer = extra_state["trainer"]  # type: ignore[assignment]
        totals: _TrainTotals = extra_state["totals"]  # type: ignore[assignment]
        window: _TrainWindow = extra_state["window"]  # type: ignore[assignment]
        progress: _NoProgress = extra_state["progress"]  # type: ignore[assignment]
        precision_controller = extra_state.get("precision_controller")
        if precision_controller is not None and not isinstance(
            precision_controller, MixedPrecisionController
        ):
            raise TypeError("precision_controller must be MixedPrecisionController when provided")
        prev_step_end_time = float(extra_state["prev_step_end_time"])

        step = int(step_ctx.train_state.global_step)
        log_this_step = step % trainer.log_steps == 0
        hist_this_step = step > 0 and (step % trainer.histogram_steps == 0)
        did_log = bool(log_this_step or hist_this_step)

        batch_ready_time = time.time()
        data_wait_seconds = max(0.0, batch_ready_time - prev_step_end_time)

        trainer._maybe_sync_cuda_timing()
        step_start = time.time()
        lr_used = float(trainer.optimizer.param_groups[0]["lr"])

        self.optimizer_runtime.zero_grad(step_ctx.optimizer_state)

        if precision_controller is None:
            loss, step_state = trainer.runtime_forward_loss(
                step_ctx.batch,
                step=step,
                log_this_step=log_this_step,
                use_internal_autocast=True,
            )
            loss.backward()
            should_step = True
        else:
            with precision_controller.autocast_context():
                loss, step_state = trainer.runtime_forward_loss(
                    step_ctx.batch,
                    step=step,
                    log_this_step=log_this_step,
                    use_internal_autocast=False,
                )
            precision_controller.backward(loss)
            should_step = precision_controller.prepare_optimizer_step(step_ctx.model)

        grad_norm = trainer.runtime_post_backward_metrics(
            step=step,
            step_state=step_state,
        )
        step_ctx.optimizer_state.extra_state["pending_step"] = step
        step_ctx.optimizer_state.extra_state["pending_step_applied"] = bool(should_step)
        self.optimizer_runtime.step(
            model=step_ctx.model,
            state=step_ctx.optimizer_state,
            ctx=step_ctx.runtime_context,
        )
        if should_step:
            refresh_persistent_lowbit_params(step_ctx.model)
        if precision_controller is not None:
            precision_controller.update_after_step(step_applied=should_step)
            if precision_controller.uses_loss_scaling and log_this_step:
                logger.info(
                    "precision step=%d mode=%s loss_scale=%.4f skipped_steps=%d",
                    step,
                    precision_controller.config.mode,
                    precision_controller.runtime_state.loss_scale,
                    precision_controller.runtime_state.skipped_steps,
                )

        trainer._maybe_sync_cuda_timing()
        step_seconds = time.time() - step_start

        loss_value = float(loss.detach().cpu().item())
        totals.update(
            tokens=int(step_state.tokens),
            effective_tokens=int(step_state.effective_tokens),
            samples=int(step_state.samples),
        )
        window.update(
            tokens=int(step_state.tokens),
            effective_tokens=int(step_state.effective_tokens),
            samples=int(step_state.samples),
            clipped=bool(trainer._is_step_clipped(grad_norm)),
        )

        loss_ema, loss_spike_ratio = trainer._update_loss_ema(loss_value)

        if log_this_step:
            trainer._log_train_step(
                progress=progress,
                step=step,
                lr_used=float(lr_used),
                loss_value=float(loss_value),
                loss_ema=float(loss_ema),
                loss_spike_ratio=float(loss_spike_ratio),
                tokens=int(step_state.tokens),
                effective_tokens=int(step_state.effective_tokens),
                step_seconds=float(step_seconds),
                data_wait_seconds=float(data_wait_seconds),
                totals=totals,
                window=window,
            )

        if (
            trainer.val_loader is not None
            and trainer.config.training.eval_steps > 0
            and step > 0
            and (step % trainer.config.training.eval_steps) == 0
        ):
            trainer.evaluate(trainer.val_loader, step=step)
            did_log = True

        if (
            trainer.config.monitoring.probe_steps > 0
            and step > 0
            and (step % int(trainer.config.monitoring.probe_steps)) == 0
        ):
            trainer._run_fixed_probe(step=step)
            did_log = True

        if hist_this_step:
            trainer._log_weight_histograms(step)
            did_log = True

        if did_log and trainer.config.monitoring.tensorboard_flush_on_log:
            trainer._flush_writer()

        extra_state["prev_step_end_time"] = time.time()
        extra_state["last_loss"] = loss_value

        return StepOutput(
            task_loss=loss_value,
            aux_loss=0.0,
            total_loss=loss_value,
            drop_fraction=0.0,
            counters={"objective_count": 1, "drop_count": 1},
        )


@dataclass
class MVPScheduleSelector:
    """Always choose non-pipeline schedule for MVP."""

    schedule: ScheduleStrategy

    def select(self, ctx: RuntimeContext) -> ScheduleStrategy:
        """Return the single non-pipeline schedule used by this script."""
        del ctx
        return self.schedule


class MVPCheckpointManager(NoOpCheckpointManager):
    """Trainer-parity checkpoint lifecycle hooks."""

    def load(
        self,
        *,
        model: torch.nn.Module,
        optimizer_state: OptimizerState,
        ctx: RuntimeContext,
    ) -> ResumeState:
        """Run initial probe hooks and return default resume position."""
        del model, ctx
        trainer: Trainer = optimizer_state.extra_state["trainer"]  # type: ignore[assignment]
        if trainer.config.monitoring.probe_steps > 0:
            trainer._run_fixed_probe(step=0)
            if trainer.config.monitoring.tensorboard_flush_on_log:
                trainer._flush_writer()
        return ResumeState()

    def on_run_end(
        self,
        *,
        model: torch.nn.Module,
        optimizer_state: OptimizerState,
        state: TrainState,
        ctx: RuntimeContext,
    ) -> None:
        """Save final checkpoint and emit trainer finalization metrics."""
        del model, ctx
        trainer: Trainer = optimizer_state.extra_state["trainer"]  # type: ignore[assignment]
        totals: _TrainTotals = optimizer_state.extra_state["totals"]  # type: ignore[assignment]
        start_time = float(optimizer_state.extra_state["start_time"])
        last_loss = optimizer_state.extra_state.get("last_loss")

        trainer.save_checkpoint(int(state.global_step), final=True)
        elapsed_seconds = time.time() - start_time
        trainer._finalize_training(
            step=int(state.global_step),
            totals=totals,
            elapsed_seconds=float(elapsed_seconds),
            final_loss=float(last_loss) if last_loss is not None else float("nan"),
        )


def build_mvp_components() -> RuntimeComponents:
    """Build MVP runtime component bundle."""
    optimizer_runtime = MVPOptimizerRuntime()
    schedule = NonPipelineSchedule(step_fn=MVPStepSchedule(optimizer_runtime).run_step)
    return RuntimeComponents(
        bootstrap=MVPBootstrap(),
        model_provider=MVPModelProvider(),
        data_provider=MVPDataProvider(),
        optimizer_runtime=optimizer_runtime,
        schedule_selector=MVPScheduleSelector(schedule=schedule),
        checkpoint_manager=MVPCheckpointManager(),
    )


def main() -> None:
    """Runtime-engine entrypoint for MVP training."""
    args = parse_args()
    engine = RuntimeEngine()
    engine.run(components=build_mvp_components(), args=args)

    logger.info("=" * 50)
    logger.info("Training completed successfully!")
    logger.info("=" * 50)
    logger.info("Next steps:")
    logger.info("1. Check the loss curve in outputs/")
    logger.info("2. Generate text from the trained model")
    logger.info("3. Increment: Add Flash Attention (Phase 2)")
    logger.info("4. Increment: Add Tensor Parallelism (Phase 3)")


if __name__ == "__main__":
    main()
