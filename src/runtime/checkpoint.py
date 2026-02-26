"""Checkpoint hook implementations for runtime engine."""

from __future__ import annotations

import torch

from src.runtime.context import RuntimeContext
from src.runtime.context import TrainState
from src.runtime.contracts import OptimizerState
from src.runtime.contracts import ResumeState


class NoOpCheckpointManager:
    """Default checkpoint manager with no-op load/save hooks."""

    def load(
        self,
        *,
        model: torch.nn.Module,
        optimizer_state: OptimizerState,
        ctx: RuntimeContext,
    ) -> ResumeState:
        del model, optimizer_state, ctx
        return ResumeState()

    def on_step_end(
        self,
        *,
        model: torch.nn.Module,
        optimizer_state: OptimizerState,
        state: TrainState,
        ctx: RuntimeContext,
    ) -> None:
        del model, optimizer_state, state, ctx

    def on_run_end(
        self,
        *,
        model: torch.nn.Module,
        optimizer_state: OptimizerState,
        state: TrainState,
        ctx: RuntimeContext,
    ) -> None:
        del model, optimizer_state, state, ctx
