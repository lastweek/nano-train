"""Runtime orchestration package for thin training scripts."""

from src.runtime.checkpoint import NoOpCheckpointManager
from src.runtime.context import RunConfig
from src.runtime.context import RuntimeContext
from src.runtime.context import TrainState
from src.runtime.contracts import CheckpointManager
from src.runtime.contracts import DataProvider
from src.runtime.contracts import ModelProvider
from src.runtime.contracts import OptimizerRuntime
from src.runtime.contracts import OptimizerState
from src.runtime.contracts import ResumeState
from src.runtime.contracts import RuntimeBootstrap
from src.runtime.contracts import RuntimeComponents
from src.runtime.contracts import ScheduleSelector
from src.runtime.contracts import ScheduleStrategy
from src.runtime.contracts import StepContext
from src.runtime.contracts import StepOutput
from src.runtime.contracts import TrainDataBundle
from src.runtime.engine import RuntimeEngine
from src.runtime.sync import ParamShardInfo

__all__ = [
    "CheckpointManager",
    "DataProvider",
    "ModelProvider",
    "NoOpCheckpointManager",
    "OptimizerRuntime",
    "OptimizerState",
    "ParamShardInfo",
    "RunConfig",
    "ResumeState",
    "RuntimeBootstrap",
    "RuntimeComponents",
    "RuntimeContext",
    "RuntimeEngine",
    "ScheduleSelector",
    "ScheduleStrategy",
    "StepContext",
    "StepOutput",
    "TrainDataBundle",
    "TrainState",
]
