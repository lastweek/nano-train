"""Runtime orchestration package for thin training scripts.

The package uses lazy exports to avoid import cycles between low-level layer modules
and runtime contracts/sync helpers.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "NoOpCheckpointManager": ("src.runtime.checkpoint", "NoOpCheckpointManager"),
    "RunConfig": ("src.runtime.context", "RunConfig"),
    "RuntimeContext": ("src.runtime.context", "RuntimeContext"),
    "TrainState": ("src.runtime.context", "TrainState"),
    "CheckpointManager": ("src.runtime.contracts", "CheckpointManager"),
    "DataProvider": ("src.runtime.contracts", "DataProvider"),
    "ModelProvider": ("src.runtime.contracts", "ModelProvider"),
    "OptimizerRuntime": ("src.runtime.contracts", "OptimizerRuntime"),
    "OptimizerState": ("src.runtime.contracts", "OptimizerState"),
    "ModulePrecisionAssignment": ("src.runtime.contracts", "ModulePrecisionAssignment"),
    "ModulePrecisionPolicy": ("src.runtime.contracts", "ModulePrecisionPolicy"),
    "ModelPrecisionPlan": ("src.runtime.contracts", "ModelPrecisionPlan"),
    "PrecisionConfig": ("src.runtime.contracts", "PrecisionConfig"),
    "PrecisionRuntimeState": ("src.runtime.contracts", "PrecisionRuntimeState"),
    "ResumeState": ("src.runtime.contracts", "ResumeState"),
    "RuntimeBootstrap": ("src.runtime.contracts", "RuntimeBootstrap"),
    "RuntimeComponents": ("src.runtime.contracts", "RuntimeComponents"),
    "ScheduleSelector": ("src.runtime.contracts", "ScheduleSelector"),
    "ScheduleStrategy": ("src.runtime.contracts", "ScheduleStrategy"),
    "StepContext": ("src.runtime.contracts", "StepContext"),
    "StepOutput": ("src.runtime.contracts", "StepOutput"),
    "TrainDataBundle": ("src.runtime.contracts", "TrainDataBundle"),
    "RuntimeEngine": ("src.runtime.engine", "RuntimeEngine"),
    "MixedPrecisionController": ("src.runtime.mixed_precision", "MixedPrecisionController"),
    "apply_model_precision_plan": ("src.runtime.mixed_precision", "apply_model_precision_plan"),
    "build_model_precision_plan": ("src.runtime.mixed_precision", "build_model_precision_plan"),
    "dtype_alias_to_torch": ("src.runtime.mixed_precision", "dtype_alias_to_torch"),
    "refresh_persistent_lowbit_params": (
        "src.runtime.mixed_precision",
        "refresh_persistent_lowbit_params",
    ),
    "resolve_precision_config": ("src.runtime.mixed_precision", "resolve_precision_config"),
    "ParamShardInfo": ("src.runtime.sync", "ParamShardInfo"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Lazily resolve runtime exports to avoid import-time dependency cycles."""
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
