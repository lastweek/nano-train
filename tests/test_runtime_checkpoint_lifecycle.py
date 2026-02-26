"""Unit tests for runtime checkpoint load/save lifecycle ordering."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from types import SimpleNamespace

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
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0]))


@dataclass
class _DummyBootstrap:
    parallel: object

    def build_context(self, args: argparse.Namespace) -> RuntimeContext:
        return RuntimeContext(
            parallel=self.parallel,
            mode="test",
            run_config=RunConfig(args=args, pp_layer_splits=None),
        )


class _DummyModelProvider:
    def build_model(self, ctx):
        del ctx
        return _DummyModel()


class _DummyDataProvider:
    def build_train_data(self, ctx):
        batch_size = ctx.run_config.args.batch_size
        seq_len = ctx.run_config.args.seq_len
        batch = {"input_ids": torch.ones(batch_size, seq_len, dtype=torch.long)}
        return TrainDataBundle(loader=[batch, batch, batch], sampler=None)


class _DummyOptimizerRuntime:
    def initialize(self, model, ctx):
        del ctx
        return OptimizerState(
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            shard_info=ParamShardInfo(set(), set()),
        )

    def zero_grad(self, state):
        del state

    def step(self, *, model, state, ctx):
        del model, state, ctx


@dataclass
class _OrderingSchedule:
    events: list[str]

    def run_step(self, step_ctx) -> StepOutput:
        del step_ctx
        self.events.append("run_step")
        return StepOutput(
            task_loss=1.0,
            aux_loss=0.0,
            total_loss=1.0,
            drop_fraction=0.0,
            counters={"objective_count": 1, "drop_count": 1},
        )


@dataclass
class _Selector:
    schedule: _OrderingSchedule

    def select(self, ctx):
        del ctx
        return self.schedule


@dataclass
class _CheckpointManager:
    events: list[str]
    loaded_global_step: int = -1

    def load(self, *, model, optimizer_state, ctx):
        del model, optimizer_state, ctx
        self.events.append("load")
        return ResumeState(start_global_step=2, start_epoch=0, pipeline_epoch=0)

    def on_step_end(self, *, model, optimizer_state, state, ctx):
        del model, optimizer_state, ctx
        self.events.append("on_step_end")
        self.loaded_global_step = state.global_step

    def on_run_end(self, *, model, optimizer_state, state, ctx):
        del model, optimizer_state, ctx
        self.events.append("on_run_end")
        self.loaded_global_step = state.global_step


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        batch_size=2,
        seq_len=4,
        epochs=1,
        max_steps=3,
        log_every=1,
    )


def test_checkpoint_load_happens_before_first_step_and_resume_state_is_used() -> None:
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

    events: list[str] = []
    schedule = _OrderingSchedule(events=events)
    checkpoint_manager = _CheckpointManager(events=events)
    components = RuntimeComponents(
        bootstrap=_DummyBootstrap(parallel=parallel),
        model_provider=_DummyModelProvider(),
        data_provider=_DummyDataProvider(),
        optimizer_runtime=_DummyOptimizerRuntime(),
        schedule_selector=_Selector(schedule=schedule),
        checkpoint_manager=checkpoint_manager,
    )

    engine = RuntimeEngine()
    engine.run(components, _args())

    assert events[0] == "load"
    assert events.count("run_step") == 1
    assert checkpoint_manager.loaded_global_step == 3
