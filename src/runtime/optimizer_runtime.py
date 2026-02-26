"""Shared optimizer-runtime helpers."""

from __future__ import annotations

from typing import Callable

import torch

from src.runtime.context import RuntimeContext
from src.runtime.contracts import OptimizerState


def zero_grad_optimizer(state: OptimizerState) -> None:
    """Zero gradients on the wrapped optimizer with `set_to_none=True`."""
    zero_grad_fn = getattr(state.optimizer, "zero_grad", None)
    if not callable(zero_grad_fn):
        raise TypeError("optimizer in OptimizerState does not implement zero_grad")
    zero_grad_fn(set_to_none=True)


def step_with_sync_policy(
    *,
    model: torch.nn.Module,
    state: OptimizerState,
    ctx: RuntimeContext,
    synchronize_gradients_fn: Callable[..., None],
) -> None:
    """
    Apply ZeRO-aware optimizer step policy.

    - If distributed optimizer is enabled, call `step_with_ready_grads()`.
    - Otherwise run gradient synchronization then call regular `step()`.
    """
    args = ctx.run_config.args
    if args.use_distributed_optimizer:
        step_with_ready_grads = getattr(state.optimizer, "step_with_ready_grads", None)
        if not callable(step_with_ready_grads):
            raise TypeError(
                "Expected optimizer with step_with_ready_grads when "
                "use_distributed_optimizer=True"
            )
        step_with_ready_grads()
        return

    synchronize_gradients_fn(
        model=model,
        shard_info=state.shard_info,
        data_parallel_size=ctx.parallel.data_parallel_size,
        expert_data_parallel_size=ctx.parallel.expert_data_parallel_size,
        data_parallel_group=ctx.parallel.data_parallel_group,
        expert_data_parallel_group=ctx.parallel.expert_data_parallel_group,
    )
    step_fn = getattr(state.optimizer, "step", None)
    if not callable(step_fn):
        raise TypeError("optimizer in OptimizerState does not implement step")
    step_fn()
