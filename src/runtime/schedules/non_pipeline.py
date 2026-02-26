"""Non-pipeline schedule strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.runtime.contracts import StepContext
from src.runtime.contracts import StepOutput


NonPipelineStepFn = Callable[[StepContext], StepOutput]


@dataclass
class NonPipelineSchedule:
    """Strategy that executes one non-pipeline training step."""

    step_fn: NonPipelineStepFn

    def run_step(self, step_ctx: StepContext) -> StepOutput:
        """Run one non-pipeline step."""
        return self.step_fn(step_ctx)
