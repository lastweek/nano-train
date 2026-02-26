"""Unit tests for runtime schedule selection helpers."""

from __future__ import annotations

from types import SimpleNamespace

from src.runtime.context import RunConfig
from src.runtime.context import RuntimeContext
from src.runtime.schedules.selector import DefaultScheduleSelector


class _DummySchedule:
    pass


def _runtime_context(pipeline_model_parallel_size: int) -> RuntimeContext:
    parallel = SimpleNamespace(
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )
    run_config = RunConfig(args=SimpleNamespace(), pp_layer_splits=None)
    return RuntimeContext(parallel=parallel, mode="test", run_config=run_config)


def test_default_schedule_selector_chooses_non_pipeline() -> None:
    non_pipeline = _DummySchedule()
    pipeline = _DummySchedule()
    selector = DefaultScheduleSelector(
        non_pipeline_schedule=non_pipeline,
        pipeline_schedule=pipeline,
    )

    selected = selector.select(_runtime_context(pipeline_model_parallel_size=1))
    assert selected is non_pipeline


def test_default_schedule_selector_chooses_pipeline() -> None:
    non_pipeline = _DummySchedule()
    pipeline = _DummySchedule()
    selector = DefaultScheduleSelector(
        non_pipeline_schedule=non_pipeline,
        pipeline_schedule=pipeline,
    )

    selected = selector.select(_runtime_context(pipeline_model_parallel_size=2))
    assert selected is pipeline
