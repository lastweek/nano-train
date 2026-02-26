"""Pipeline 1F1B schedule strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.runtime.contracts import StepContext
from src.runtime.contracts import StepOutput


PipelineStepFn = Callable[[StepContext], StepOutput]


@dataclass
class Pipeline1F1BSchedule:
    """Strategy that executes one non-interleaved pipeline 1F1B step."""

    step_fn: PipelineStepFn

    def run_step(self, step_ctx: StepContext) -> StepOutput:
        """Run one pipeline step."""
        return self.step_fn(step_ctx)
