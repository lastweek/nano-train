"""Unit tests for runtime optimizer policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from src.runtime.context import RunConfig
from src.runtime.context import RuntimeContext
from src.runtime.contracts import OptimizerState
from src.runtime.optimizer_runtime import step_with_sync_policy
from src.runtime.optimizer_runtime import zero_grad_optimizer
from src.runtime.sync import ParamShardInfo


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))


@dataclass
class _RegularOptimizer:
    zero_grad_calls: int = 0
    step_calls: int = 0

    def zero_grad(self, *, set_to_none: bool) -> None:
        assert set_to_none is True
        self.zero_grad_calls += 1

    def step(self) -> None:
        self.step_calls += 1


@dataclass
class _DistributedOptimizer:
    ready_step_calls: int = 0

    def zero_grad(self, *, set_to_none: bool) -> None:
        assert set_to_none is True

    def step_with_ready_grads(self) -> None:
        self.ready_step_calls += 1


def _ctx(use_distributed_optimizer: bool) -> RuntimeContext:
    args = SimpleNamespace(use_distributed_optimizer=use_distributed_optimizer)
    parallel = SimpleNamespace(
        data_parallel_size=2,
        expert_data_parallel_size=4,
        data_parallel_group="dp",
        expert_data_parallel_group="edp",
    )
    return RuntimeContext(parallel=parallel, mode="test", run_config=RunConfig(args=args, pp_layer_splits=None))


def test_zero_grad_optimizer_calls_optimizer_zero_grad() -> None:
    optimizer = _RegularOptimizer()
    state = OptimizerState(optimizer=optimizer, shard_info=ParamShardInfo(set(), set()))
    zero_grad_optimizer(state)
    assert optimizer.zero_grad_calls == 1


def test_step_with_sync_policy_non_zero_path_syncs_then_steps() -> None:
    optimizer = _RegularOptimizer()
    model = _TinyModel()
    state = OptimizerState(
        optimizer=optimizer,
        shard_info=ParamShardInfo(
            tensor_model_parallel_sharded_param_ids=set(),
            expert_model_parallel_sharded_param_ids=set(),
        ),
    )
    calls: list[dict[str, object]] = []

    def _sync(**kwargs) -> None:
        calls.append(kwargs)

    step_with_sync_policy(
        model=model,
        state=state,
        ctx=_ctx(use_distributed_optimizer=False),
        synchronize_gradients_fn=_sync,
    )

    assert len(calls) == 1
    assert calls[0]["model"] is model
    assert calls[0]["data_parallel_size"] == 2
    assert calls[0]["expert_data_parallel_size"] == 4
    assert calls[0]["data_parallel_group"] == "dp"
    assert calls[0]["expert_data_parallel_group"] == "edp"
    assert optimizer.step_calls == 1


def test_step_with_sync_policy_zero_path_uses_ready_grads_step() -> None:
    optimizer = _DistributedOptimizer()
    model = _TinyModel()
    state = OptimizerState(optimizer=optimizer, shard_info=ParamShardInfo(set(), set()))
    sync_calls = 0

    def _sync(**kwargs) -> None:
        nonlocal sync_calls
        del kwargs
        sync_calls += 1

    step_with_sync_policy(
        model=model,
        state=state,
        ctx=_ctx(use_distributed_optimizer=True),
        synchronize_gradients_fn=_sync,
    )

    assert sync_calls == 0
    assert optimizer.ready_step_calls == 1
