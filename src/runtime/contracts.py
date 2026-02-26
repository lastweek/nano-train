"""Runtime component contracts and shared value objects."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import field
from typing import Iterable
from typing import Optional
from typing import Protocol

import torch
from torch.utils.data import DistributedSampler

from src.runtime.context import RuntimeContext
from src.runtime.context import TrainState
from src.runtime.sync import ParamShardInfo


@dataclass
class TrainDataBundle:
    """Container for training loader and optional distributed sampler."""

    loader: Iterable[dict[str, torch.Tensor]]
    sampler: Optional[DistributedSampler]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class OptimizerState:
    """Runtime optimizer state shared across schedules and checkpoint hooks."""

    optimizer: object
    shard_info: ParamShardInfo
    extra_state: dict[str, object] = field(default_factory=dict)


@dataclass
class StepContext:
    """Inputs required to execute one training step."""

    model: torch.nn.Module
    batch: Optional[dict[str, torch.Tensor]]
    optimizer_state: OptimizerState
    runtime_context: RuntimeContext
    train_state: TrainState


@dataclass
class StepOutput:
    """Step-level local metric numerators and count metadata."""

    task_loss: float
    aux_loss: float
    total_loss: float
    drop_fraction: float
    counters: dict[str, int] = field(default_factory=dict)


@dataclass
class ResumeState:
    """Checkpoint resume cursor returned by checkpoint manager."""

    start_global_step: int = 0
    start_epoch: int = 0
    pipeline_epoch: int = 0


class ModelProvider(Protocol):
    """Build model stack for a runtime context."""

    def build_model(self, ctx: RuntimeContext) -> torch.nn.Module:
        """Build and return the model for the current runtime context."""
        ...


class DataProvider(Protocol):
    """Build train-time data pipeline for a runtime context."""

    def build_train_data(self, ctx: RuntimeContext) -> TrainDataBundle:
        """Build and return the train data bundle for the current runtime context."""
        ...


class OptimizerRuntime(Protocol):
    """Initialize optimizer state and execute optimizer policy."""

    def initialize(self, model: torch.nn.Module, ctx: RuntimeContext) -> OptimizerState:
        """Build optimizer state objects bound to `model` and runtime context."""
        ...

    def zero_grad(self, state: OptimizerState) -> None:
        """Zero gradients on optimizer state before a train step."""
        ...

    def step(
        self,
        *,
        model: torch.nn.Module,
        state: OptimizerState,
        ctx: RuntimeContext,
    ) -> None:
        """Apply one optimizer step using the configured synchronization policy."""
        ...


class ScheduleStrategy(Protocol):
    """Run one mode-specific training step."""

    def run_step(self, step_ctx: StepContext) -> StepOutput:
        """Execute one training step and return local metric sums/counts."""
        ...


class ScheduleSelector(Protocol):
    """Select schedule strategy for the current runtime context."""

    def select(self, ctx: RuntimeContext) -> ScheduleStrategy:
        """Select the schedule strategy for the current runtime mode."""
        ...


class CheckpointManager(Protocol):
    """Checkpoint load/save lifecycle for runtime engine loops."""

    def load(
        self,
        *,
        model: torch.nn.Module,
        optimizer_state: OptimizerState,
        ctx: RuntimeContext,
    ) -> ResumeState:
        """Load checkpoint state and return resume cursors for train loop state."""
        ...

    def on_step_end(
        self,
        *,
        model: torch.nn.Module,
        optimizer_state: OptimizerState,
        state: TrainState,
        ctx: RuntimeContext,
    ) -> None:
        """Handle per-step checkpoint hooks after each completed train step."""
        ...

    def on_run_end(
        self,
        *,
        model: torch.nn.Module,
        optimizer_state: OptimizerState,
        state: TrainState,
        ctx: RuntimeContext,
    ) -> None:
        """Handle end-of-run checkpoint hooks after the training loop exits."""
        ...


@dataclass
class RuntimeComponents:
    """Runtime component bundle used by RuntimeEngine."""

    bootstrap: "RuntimeBootstrap"
    model_provider: ModelProvider
    data_provider: DataProvider
    optimizer_runtime: OptimizerRuntime
    schedule_selector: ScheduleSelector
    checkpoint_manager: CheckpointManager


class RuntimeBootstrap(Protocol):
    """Build runtime context from entrypoint args."""

    def build_context(self, args: argparse.Namespace) -> RuntimeContext:
        """Validate args and construct the runtime context for the current process."""
        ...
