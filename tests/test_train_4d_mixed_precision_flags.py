"""Mixed-precision CLI and validation coverage for train_4d script."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch


def _load_train_4d_module() -> ModuleType:
    repo_root = Path(__file__).parent.parent
    module_path = repo_root / "examples" / "train_4d.py"
    module_name = "train_4d_mixed_precision_test_module"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load examples/train_4d.py for tests")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


EP_MODULE = _load_train_4d_module()


def _build_args_for_validate(**overrides) -> argparse.Namespace:
    base = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "batch_size": 8,
        "intermediate_size": 128,
        "num_heads": 8,
        "num_experts": 8,
        "num_layers": 5,
        "seq_len": 16,
        "num_microbatches": 1,
        "dropout": 0.0,
        "use_distributed_optimizer": False,
        "data_parallel_sharding_strategy": "no_shard",
        "num_distributed_optimizer_instances": 1,
        "zero_debug": False,
        "zero_debug_max_steps": 1,
        "zero_debug_max_params": 8,
        "bf16": False,
        "fp16": False,
        "fp8": False,
        "fp4": False,
        "fp8_backend": "emulated",
        "fp8_format": "e4m3",
        "fp8_amax_history_len": 16,
        "fp8_amax_compute_algo": "most_recent",
        "fp4_backend": "emulated",
        "params_dtype": None,
        "main_params_dtype": None,
        "main_grads_dtype": None,
        "exp_avg_dtype": None,
        "exp_avg_sq_dtype": None,
        "loss_scale_init": 65536.0,
        "loss_scale_growth_factor": 2.0,
        "loss_scale_backoff_factor": 0.5,
        "loss_scale_growth_interval": 2000,
        "loss_scale_min": 1.0,
        "loss_scale_max": 16777216.0,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_parse_args_accepts_fp8_precision_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_4d.py",
            "--tensor-model-parallel-size",
            "1",
            "--pipeline-model-parallel-size",
            "1",
            "--expert-model-parallel-size",
            "1",
            "--fp8",
            "--fp8-backend",
            "emulated",
            "--fp8-format",
            "hybrid",
            "--params-dtype",
            "bf16",
        ],
    )
    args = EP_MODULE.parse_args()
    assert args.fp8 is True
    assert args.fp8_backend == "emulated"
    assert args.fp8_format == "hybrid"
    assert args.params_dtype == "bf16"


def test_validate_args_rejects_conflicting_precision_flags() -> None:
    args = _build_args_for_validate(fp16=True, fp8=True)
    with pytest.raises(ValueError, match="At most one of"):
        EP_MODULE.normalize_and_resolve_precision(args, torch.device("cpu"))


def test_validate_args_rejects_invalid_fp8_backend() -> None:
    args = _build_args_for_validate(fp8=True, fp8_backend="invalid_backend")
    with pytest.raises(ValueError, match="fp8-backend"):
        EP_MODULE.normalize_and_resolve_precision(args, torch.device("cpu"))
