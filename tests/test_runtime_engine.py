"""Unit tests for runtime engine mode dispatch and lifecycle."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from src.runtime.context import RunConfig
from src.runtime.context import RuntimeContext
from src.runtime.contracts import OptimizerState
from src.runtime.contracts import ResumeState
from src.runtime.contracts import RuntimeComponents
from src.runtime.contracts import StepOutput
from src.runtime.contracts import TrainDataBundle
from src.runtime.engine import RuntimeEngine
from src.runtime.sync import ParamShardInfo


class _DummyModel(torch.nn.Module):
    def __init__(self, *, is_first_pp_stage: bool = False) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0]))
        self.is_first_pp_stage = is_first_pp_stage

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.w


@dataclass
class _DummyBootstrap:
    parallel: object

    def build_context(self, args: argparse.Namespace) -> RuntimeContext:
        return RuntimeContext(
            parallel=self.parallel,
            mode="test",
            run_config=RunConfig(args=args, pp_layer_splits=None),
        )


@dataclass
class _DummySchedule:
    calls: int = 0

    def run_step(self, step_ctx) -> StepOutput:
        del step_ctx
        self.calls += 1
        return StepOutput(
            task_loss=1.0,
            aux_loss=0.1,
            total_loss=1.1,
            drop_fraction=0.0,
            counters={"objective_count": 1, "drop_count": 1},
        )


@dataclass
class _DummyScheduleSelector:
    schedule: _DummySchedule

    def select(self, ctx):
        del ctx
        return self.schedule


@dataclass
class _DummyCheckpointManager:
    load_calls: int = 0
    step_calls: int = 0
    run_end_calls: int = 0

    def load(self, *, model, optimizer_state, ctx) -> ResumeState:
        del model, optimizer_state, ctx
        self.load_calls += 1
        return ResumeState()

    def on_step_end(self, *, model, optimizer_state, state, ctx) -> None:
        del model, optimizer_state, state, ctx
        self.step_calls += 1

    def on_run_end(self, *, model, optimizer_state, state, ctx) -> None:
        del model, optimizer_state, state, ctx
        self.run_end_calls += 1


class _DummyModelProvider:
    def __init__(self, *, pipeline: bool) -> None:
        self._pipeline = pipeline

    def build_model(self, ctx):
        del ctx
        return _DummyModel(is_first_pp_stage=self._pipeline)


class _DummyDataProvider:
    def build_train_data(self, ctx):
        batch_size = ctx.run_config.args.batch_size
        seq_len = ctx.run_config.args.seq_len
        batch = {"input_ids": torch.ones(batch_size, seq_len, dtype=torch.long)}
        return TrainDataBundle(loader=[batch], sampler=None)


class _DummyOptimizerRuntime:
    def initialize(self, model, ctx) -> OptimizerState:
        del ctx
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        return OptimizerState(optimizer=optimizer, shard_info=ParamShardInfo(set(), set()))

    def zero_grad(self, state: OptimizerState) -> None:
        del state

    def step(self, *, model, state: OptimizerState, ctx) -> None:
        del model, state, ctx


def _args(max_steps: int = 2) -> argparse.Namespace:
    return argparse.Namespace(
        batch_size=2,
        seq_len=4,
        epochs=1,
        max_steps=max_steps,
        log_every=1,
    )


def test_runtime_engine_dispatches_non_pipeline_components() -> None:
    parallel = SimpleNamespace(
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        data_parallel_size=1,
        expert_data_parallel_size=1,
        context_parallel_size=1,
        data_parallel_group=None,
        expert_data_parallel_group=None,
    )

    schedule = _DummySchedule()
    checkpoint_manager = _DummyCheckpointManager()
    components = RuntimeComponents(
        bootstrap=_DummyBootstrap(parallel=parallel),
        model_provider=_DummyModelProvider(pipeline=False),
        data_provider=_DummyDataProvider(),
        optimizer_runtime=_DummyOptimizerRuntime(),
        schedule_selector=_DummyScheduleSelector(schedule=schedule),
        checkpoint_manager=checkpoint_manager,
    )

    engine = RuntimeEngine()
    engine.run(components, _args(max_steps=2))

    assert schedule.calls == 1
    assert checkpoint_manager.load_calls == 1
    assert checkpoint_manager.step_calls == 1
    assert checkpoint_manager.run_end_calls == 1


def test_runtime_engine_dispatches_pipeline_components() -> None:
    parallel = SimpleNamespace(
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        pipeline_model_parallel_size=2,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        data_parallel_size=1,
        expert_data_parallel_size=1,
        context_parallel_size=1,
        data_parallel_group=None,
        expert_data_parallel_group=None,
    )

    args = _args(max_steps=3)

    schedule = _DummySchedule()
    checkpoint_manager = _DummyCheckpointManager()
    components = RuntimeComponents(
        bootstrap=_DummyBootstrap(parallel=parallel),
        model_provider=_DummyModelProvider(pipeline=True),
        data_provider=_DummyDataProvider(),
        optimizer_runtime=_DummyOptimizerRuntime(),
        schedule_selector=_DummyScheduleSelector(schedule=schedule),
        checkpoint_manager=checkpoint_manager,
    )

    engine = RuntimeEngine()
    engine.run(components, args)

    assert schedule.calls == 3
    assert checkpoint_manager.load_calls == 1
    assert checkpoint_manager.step_calls == 3
    assert checkpoint_manager.run_end_calls == 1


def test_runtime_engine_requires_runtime_components() -> None:
    engine = RuntimeEngine()
    with pytest.raises(AttributeError):
        engine.run(object(), _args(max_steps=1))  # type: ignore[arg-type]
