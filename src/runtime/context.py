"""Runtime context objects shared across runtime engine components."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from src.distributed.topology import ModelParallelTopology


@dataclass
class RunConfig:
    """Resolved run configuration used by the runtime engine."""

    args: argparse.Namespace
    pp_layer_splits: Optional[tuple[int, ...]]


@dataclass
class RuntimeContext:
    """Runtime metadata for the current process and mode."""

    parallel: ModelParallelTopology
    mode: str
    run_config: RunConfig


@dataclass
class TrainState:
    """Mutable train loop counters used by runtime orchestration."""

    global_step: int = 0
    epoch: int = 0
    pipeline_epoch: int = 0
