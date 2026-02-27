"""Pipeline runtime tests for persistent low-bit refresh callback behavior."""

from __future__ import annotations

import types

import torch

from src.runtime.pipeline import train_step_pipeline
from src.runtime.sync import ParamShardInfo


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(2, 2))
        self.config = types.SimpleNamespace(hidden_size=2)
        self.is_first_pp_stage = False
        self.is_last_pp_stage = False


class _DummyParallel:
    pipeline_model_parallel_size = 2
    pipeline_model_parallel_rank = 1
    data_parallel_rank = 0
    tensor_model_parallel_rank = 0
    expert_model_parallel_rank = 0
    context_parallel_rank = 0
    data_parallel_size = 1
    expert_data_parallel_size = 1
    data_parallel_group = None
    expert_data_parallel_group = None
    device = torch.device("cpu")
    world_size = 1

    def rank_from_coords(self, **kwargs) -> int:
        del kwargs
        return 0



def test_pipeline_calls_refresh_callback_once_when_step_runs(monkeypatch) -> None:
    model = _DummyModel()
    parallel = _DummyParallel()

    counts = {"step": 0, "refresh": 0}

    def _noop_schedule(parallel, num_microbatches, run_forward, run_backward):
        del parallel, num_microbatches, run_forward, run_backward

    monkeypatch.setattr("src.runtime.pipeline.execute_1f1b_schedule", _noop_schedule)

    def _apply_optimizer_step(**kwargs):
        del kwargs
        counts["step"] += 1

    def _refresh(_model):
        counts["refresh"] += 1

    task, aux, drop, obj_count, drop_count = train_step_pipeline(
        model=model,
        optimizer=object(),
        use_distributed_optimizer=False,
        batch=None,
        parallel=parallel,
        num_microbatches=1,
        expected_local_batch=1,
        seq_len=2,
        aux_loss_coef=0.0,
        shard_info=ParamShardInfo(set(), set()),
        gather_moe_metrics_fn=lambda *_args, **_kwargs: (torch.zeros(()), 0.0),
        apply_optimizer_step_fn=_apply_optimizer_step,
        sync_plugin=None,
        zero_grad_fn=lambda _optimizer: None,
        refresh_persistent_params_fn=_refresh,
    )

    assert (task, aux, drop, obj_count, drop_count) == (0.0, 0.0, 0.0, 0, 0)
    assert counts["step"] == 1
    assert counts["refresh"] == 1
