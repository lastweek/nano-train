"""Unit tests for RuntimeEngine bootstrap contract."""

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


@dataclass
class _OrderBootstrap:
    events: list[str]

    def build_context(self, args: argparse.Namespace) -> RuntimeContext:
        self.events.append("bootstrap")
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
        return RuntimeContext(
            parallel=parallel,
            mode="test",
            run_config=RunConfig(args=args, pp_layer_splits=None),
        )


@dataclass
class _OrderModelProvider:
    events: list[str]

    def build_model(self, ctx):
        del ctx
        self.events.append("model")
        return torch.nn.Linear(2, 2)


@dataclass
class _OrderDataProvider:
    events: list[str]

    def build_train_data(self, ctx):
        self.events.append("data")
        batch_size = ctx.run_config.args.batch_size
        batch = {"input_ids": torch.ones(batch_size, 2, dtype=torch.long)}
        return TrainDataBundle(loader=[batch], sampler=None)


@dataclass
class _OrderOptimizerRuntime:
    events: list[str]

    def initialize(self, model, ctx):
        del ctx
        self.events.append("optimizer")
        return OptimizerState(
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            shard_info=ParamShardInfo(set(), set()),
        )

    def zero_grad(self, state):
        del state

    def step(self, *, model, state, ctx):
        del model, state, ctx


class _Schedule:
    def run_step(self, step_ctx):
        del step_ctx
        return StepOutput(
            task_loss=1.0,
            aux_loss=0.0,
            total_loss=1.0,
            drop_fraction=0.0,
            counters={"objective_count": 1, "drop_count": 1},
        )


class _Selector:
    def select(self, ctx):
        del ctx
        return _Schedule()


class _Checkpoint:
    def load(self, *, model, optimizer_state, ctx):
        del model, optimizer_state, ctx
        return ResumeState()

    def on_step_end(self, *, model, optimizer_state, state, ctx):
        del model, optimizer_state, state, ctx

    def on_run_end(self, *, model, optimizer_state, state, ctx):
        del model, optimizer_state, state, ctx


def test_engine_uses_bootstrap_before_building_components() -> None:
    events: list[str] = []
    components = RuntimeComponents(
        bootstrap=_OrderBootstrap(events=events),
        model_provider=_OrderModelProvider(events=events),
        data_provider=_OrderDataProvider(events=events),
        optimizer_runtime=_OrderOptimizerRuntime(events=events),
        schedule_selector=_Selector(),
        checkpoint_manager=_Checkpoint(),
    )

    args = argparse.Namespace(batch_size=2, epochs=1, max_steps=1, log_every=0)
    RuntimeEngine().run(components, args)

    assert events[:4] == ["bootstrap", "model", "data", "optimizer"]
