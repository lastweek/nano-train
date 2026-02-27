"""Runtime component contracts and shared value objects."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import field
from typing import Iterable
from typing import Literal
from typing import Optional
from typing import Protocol

import torch
from torch.utils.data import DistributedSampler

from src.runtime.context import RuntimeContext
from src.runtime.context import TrainState
from src.runtime.sync import ParamShardInfo


PrecisionMode = Literal["fp32", "bf16", "fp16", "fp8", "fp4"]
PrecisionDType = Literal["fp32", "bf16", "fp16"]
LowBitBackendName = Literal["transformer_engine", "emulated"]
FP8Format = Literal["e4m3", "hybrid"]
FP8AmaxComputeAlgo = Literal["most_recent", "max"]
ModulePatternType = Literal["regex", "glob"]
LowBitComputeMode = Literal["fp8", "fp4"]
PersistentLowBitMode = Literal["off", "fp8", "fp4"]
PersistentScaleGranularity = Literal["per_tensor", "per_channel"]
FP4PersistentFormat = Literal["nf4"]


@dataclass
class ModulePrecisionPolicy:
    """Global module-pattern policy for low-bit compute and persistent params."""

    pattern_type: ModulePatternType = "regex"
    compute_lowbit_mode: Optional[LowBitComputeMode] = None
    compute_lowbit_include: tuple[str, ...] = ()
    compute_lowbit_exclude: tuple[str, ...] = ()
    persistent_lowbit_mode: PersistentLowBitMode = "off"
    persistent_lowbit_include: tuple[str, ...] = ()
    persistent_lowbit_exclude: tuple[str, ...] = ()
    persistent_scale_granularity: PersistentScaleGranularity = "per_channel"
    fp4_persistent_format: FP4PersistentFormat = "nf4"


@dataclass
class ModulePrecisionAssignment:
    """Resolved low-bit assignment for one concrete model module instance."""

    module_name: str
    module_type: str
    compute_lowbit_mode: Optional[LowBitComputeMode]
    persistent_lowbit_mode: PersistentLowBitMode
    persistent_scale_granularity: PersistentScaleGranularity
    fp4_persistent_format: FP4PersistentFormat


@dataclass
class ModelPrecisionPlan:
    """Resolved per-module precision assignments for one built model."""

    assignments: dict[str, ModulePrecisionAssignment] = field(default_factory=dict)
    compute_lowbit_module_count: int = 0
    persistent_lowbit_module_count: int = 0


@dataclass
class PrecisionConfig:
    """Resolved precision settings for a runtime training run."""

    mode: PrecisionMode = "fp32"
    params_dtype: PrecisionDType = "fp32"
    main_params_dtype: PrecisionDType = "fp32"
    main_grads_dtype: PrecisionDType = "fp32"
    exp_avg_dtype: PrecisionDType = "fp32"
    exp_avg_sq_dtype: PrecisionDType = "fp32"

    # Runtime compute/autocast dtype derived from mode and availability.
    activation_dtype: PrecisionDType = "fp32"

    fp8_backend: LowBitBackendName = "transformer_engine"
    fp8_format: FP8Format = "e4m3"
    fp8_amax_history_len: int = 16
    fp8_amax_compute_algo: FP8AmaxComputeAlgo = "most_recent"
    fp4_backend: Literal["emulated"] = "emulated"
    module_precision_policy: Optional[ModulePrecisionPolicy] = None
    fp8_param: bool = False
    fp4_param: bool = False
    fp4_param_format: FP4PersistentFormat = "nf4"
    persistent_scale_granularity: PersistentScaleGranularity = "per_channel"

    loss_scale_init: float = 65536.0
    loss_scale_growth_factor: float = 2.0
    loss_scale_backoff_factor: float = 0.5
    loss_scale_growth_interval: int = 2000
    loss_scale_min: float = 1.0
    loss_scale_max: float = 16777216.0


@dataclass
class PrecisionRuntimeState:
    """Mutable precision runtime counters emitted by mixed-precision controller."""

    loss_scale: float
    growth_tracker: int = 0
    skipped_steps: int = 0
    found_inf_steps: int = 0


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
