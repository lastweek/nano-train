"""Unit tests for runtime pipeline scheduling helpers."""

from __future__ import annotations

from types import SimpleNamespace

from src.runtime.pipeline import activation_tag
from src.runtime.pipeline import execute_1f1b_schedule
from src.runtime.pipeline import grad_tag
from src.runtime.pipeline import label_tag


def test_pipeline_tags_are_stable() -> None:
    assert activation_tag(3) == 10003
    assert grad_tag(3) == 20003
    assert label_tag(3) == 30003


def test_execute_1f1b_schedule_order() -> None:
    parallel = SimpleNamespace(
        pipeline_model_parallel_size=4,
        pipeline_model_parallel_rank=1,
    )
    events: list[str] = []

    execute_1f1b_schedule(
        parallel=parallel,
        num_microbatches=5,
        run_forward=lambda idx: events.append(f"f{idx}"),
        run_backward=lambda idx: events.append(f"b{idx}"),
    )

    assert events == ["f0", "f1", "f2", "b0", "f3", "b1", "f4", "b2", "b3", "b4"]
