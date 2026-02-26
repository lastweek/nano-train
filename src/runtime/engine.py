"""Runtime engine for distributed training orchestration."""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist

from src.logging import get_logger
from src.runtime.context import RuntimeContext
from src.runtime.context import TrainState
from src.runtime.contracts import RuntimeComponents
from src.runtime.contracts import StepContext


logger = get_logger(__name__)


class RuntimeEngine:
    """Engine that orchestrates distributed training through runtime components."""

    def run(
        self,
        components: RuntimeComponents,
        args: argparse.Namespace,
    ) -> None:
        """Run train-loop orchestration with runtime components."""
        runtime_context = components.bootstrap.build_context(args)

        self._log_runtime_topology(runtime_context)

        model = components.model_provider.build_model(runtime_context)
        train_data = components.data_provider.build_train_data(runtime_context)
        optimizer_state = components.optimizer_runtime.initialize(model, runtime_context)
        schedule = components.schedule_selector.select(runtime_context)

        resume = components.checkpoint_manager.load(
            model=model,
            optimizer_state=optimizer_state,
            ctx=runtime_context,
        )

        model.train()
        state = TrainState(
            global_step=resume.start_global_step,
            epoch=resume.start_epoch,
            pipeline_epoch=resume.pipeline_epoch,
        )

        try:
            if runtime_context.parallel.pipeline_model_parallel_size == 1:
                self._run_non_pipeline_mode(
                    args=args,
                    model=model,
                    train_data=train_data,
                    optimizer_state=optimizer_state,
                    schedule=schedule,
                    runtime_context=runtime_context,
                    state=state,
                    components=components,
                )
            else:
                self._run_pipeline_mode(
                    args=args,
                    model=model,
                    train_data=train_data,
                    optimizer_state=optimizer_state,
                    schedule=schedule,
                    runtime_context=runtime_context,
                    state=state,
                    components=components,
                )

            components.checkpoint_manager.on_run_end(
                model=model,
                optimizer_state=optimizer_state,
                state=state,
                ctx=runtime_context,
            )

            if runtime_context.parallel.rank == 0:
                logger.info("Training completed")
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    def _run_non_pipeline_mode(
        self,
        *,
        args: argparse.Namespace,
        model: torch.nn.Module,
        train_data,
        optimizer_state,
        schedule,
        runtime_context: RuntimeContext,
        state: TrainState,
        components: RuntimeComponents,
    ) -> None:
        parallel = runtime_context.parallel
        sampler = train_data.sampler

        for epoch in range(state.epoch, args.epochs):
            state.epoch = epoch
            if sampler is not None:
                sampler.set_epoch(epoch)

            task_sum = 0.0
            aux_sum = 0.0
            total_sum = 0.0
            drop_sum = 0.0
            objective_count = 0
            drop_count = 0

            for batch in train_data.loader:
                if state.global_step >= args.max_steps:
                    break

                output = schedule.run_step(
                    StepContext(
                        model=model,
                        batch=batch,
                        optimizer_state=optimizer_state,
                        runtime_context=runtime_context,
                        train_state=state,
                    )
                )

                task_sum += output.task_loss
                aux_sum += output.aux_loss
                total_sum += output.total_loss
                drop_sum += output.drop_fraction
                objective_count += output.counters.get("objective_count", 1)
                drop_count += output.counters.get("drop_count", 1)

                components.checkpoint_manager.on_step_end(
                    model=model,
                    optimizer_state=optimizer_state,
                    state=state,
                    ctx=runtime_context,
                )

                if parallel.rank == 0 and self._should_log_step(args, state.global_step):
                    step_obj_count = max(1, output.counters.get("objective_count", 1))
                    step_drop_count = max(1, output.counters.get("drop_count", 1))
                    logger.info(
                        "step=%d task=%.6f aux=%.6f total=%.6f drop=%.4f",
                        state.global_step,
                        output.task_loss / step_obj_count,
                        output.aux_loss / step_obj_count,
                        output.total_loss / step_obj_count,
                        output.drop_fraction / step_drop_count,
                    )
                state.global_step += 1

            if objective_count > 0:
                avg_task, avg_aux, avg_total, avg_drop = self._reduce_metric_sums(
                    task_sum=task_sum,
                    aux_sum=aux_sum,
                    total_sum=total_sum,
                    drop_sum=drop_sum,
                    objective_count=objective_count,
                    drop_count=drop_count,
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

            if state.global_step >= args.max_steps:
                break

    def _run_pipeline_mode(
        self,
        *,
        args: argparse.Namespace,
        model: torch.nn.Module,
        train_data,
        optimizer_state,
        schedule,
        runtime_context: RuntimeContext,
        state: TrainState,
        components: RuntimeComponents,
    ) -> None:
        parallel = runtime_context.parallel
        sampler = train_data.sampler

        if sampler is not None and getattr(model, "is_first_pp_stage", False):
            sampler.set_epoch(state.pipeline_epoch)

        data_iter = iter(train_data.loader) if getattr(model, "is_first_pp_stage", False) else None

        for step in range(state.global_step, args.max_steps):
            state.global_step = step
            if getattr(model, "is_first_pp_stage", False):
                if data_iter is None:
                    raise RuntimeError("Missing data iterator on first PP stage")
                try:
                    step_batch = next(data_iter)
                except StopIteration:
                    state.pipeline_epoch += 1
                    if sampler is not None:
                        sampler.set_epoch(state.pipeline_epoch)
                    data_iter = iter(train_data.loader)
                    step_batch = next(data_iter)
            else:
                step_batch = None

            output = schedule.run_step(
                StepContext(
                    model=model,
                    batch=step_batch,
                    optimizer_state=optimizer_state,
                    runtime_context=runtime_context,
                    train_state=state,
                )
            )
            avg_task, avg_aux, avg_total, avg_drop = self._reduce_metric_sums(
                task_sum=output.task_loss,
                aux_sum=output.aux_loss,
                total_sum=output.total_loss,
                drop_sum=output.drop_fraction,
                objective_count=output.counters.get("objective_count", 1),
                drop_count=output.counters.get("drop_count", 1),
                device=parallel.device,
                world_size=parallel.world_size,
            )

            components.checkpoint_manager.on_step_end(
                model=model,
                optimizer_state=optimizer_state,
                state=state,
                ctx=runtime_context,
            )

            if parallel.rank == 0 and self._should_log_step(args, step):
                logger.info(
                    "step=%d task=%.6f aux=%.6f total=%.6f drop=%.4f",
                    step,
                    avg_task,
                    avg_aux,
                    avg_total,
                    avg_drop,
                )

    def _should_log_step(self, args: argparse.Namespace, step: int) -> bool:
        """Return whether periodic per-step logging should run for this step."""
        log_every = int(getattr(args, "log_every", 0))
        if log_every <= 0:
            return False
        return step % log_every == 0

    def _reduce_metric_sums(
        self,
        *,
        task_sum: float,
        aux_sum: float,
        total_sum: float,
        drop_sum: float,
        objective_count: int,
        drop_count: int,
        device: torch.device,
        world_size: int,
    ) -> tuple[float, float, float, float]:
        """Reduce local metric sums and return global averages."""
        if world_size == 1:
            obj_denom = max(1, objective_count)
            drop_denom = max(1, drop_count)
            return (
                task_sum / obj_denom,
                aux_sum / obj_denom,
                total_sum / obj_denom,
                drop_sum / drop_denom,
            )

        stats = torch.tensor(
            [
                task_sum,
                aux_sum,
                total_sum,
                drop_sum,
                float(objective_count),
                float(drop_count),
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        obj_denom = max(1.0, stats[4].item())
        drop_denom = max(1.0, stats[5].item())
        return (
            float(stats[0].item() / obj_denom),
            float(stats[1].item() / obj_denom),
            float(stats[2].item() / obj_denom),
            float(stats[3].item() / drop_denom),
        )

    def _log_runtime_topology(self, ctx: RuntimeContext) -> None:
        """Emit one-time rank-0 runtime topology summary."""
        parallel = ctx.parallel
        if parallel.rank != 0:
            return

        logger.info("=" * 84)
        logger.info("RuntimeEngine")
        logger.info("=" * 84)
        logger.info("Mode: %s", ctx.mode)
        logger.info(
            "World Size: %d | Tensor MP: %d | Pipeline MP: %d | "
            "Expert MP: %d | Data P: %d | EDP: %d",
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
        logger.info("=" * 84)
