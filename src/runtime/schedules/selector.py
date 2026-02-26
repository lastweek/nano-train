"""Default schedule selection utilities."""

from __future__ import annotations

from dataclasses import dataclass

from src.runtime.context import RuntimeContext
from src.runtime.contracts import ScheduleStrategy


@dataclass
class DefaultScheduleSelector:
    """Select between non-pipeline and pipeline schedules from topology."""

    non_pipeline_schedule: ScheduleStrategy
    pipeline_schedule: ScheduleStrategy

    def select(self, ctx: RuntimeContext) -> ScheduleStrategy:
        """Choose schedule based on pipeline model-parallel size."""
        if ctx.parallel.pipeline_model_parallel_size > 1:
            return self.pipeline_schedule
        return self.non_pipeline_schedule
