"""Unit tests for runtime mixed-precision resolver and controller."""

from __future__ import annotations

import argparse

import torch

from src.runtime.contracts import PrecisionConfig
from src.runtime.mixed_precision import MixedPrecisionController
from src.runtime.mixed_precision import resolve_precision_config


def _args(**overrides) -> argparse.Namespace:
    base = {
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
        "loss_scale_init": 128.0,
        "loss_scale_growth_factor": 2.0,
        "loss_scale_backoff_factor": 0.5,
        "loss_scale_growth_interval": 2,
        "loss_scale_min": 1.0,
        "loss_scale_max": 1024.0,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_resolve_precision_auto_on_cpu_defaults_to_fp32() -> None:
    cfg = resolve_precision_config(_args(), torch.device("cpu"))
    assert cfg.mode == "fp32"
    assert cfg.activation_dtype == "fp32"


def test_resolve_precision_rejects_conflicting_flags() -> None:
    try:
        resolve_precision_config(_args(bf16=True, fp16=True), torch.device("cpu"))
    except ValueError as exc:
        assert "At most one of" in str(exc)
    else:
        raise AssertionError("Expected ValueError for conflicting precision flags")


def test_mixed_precision_controller_scales_and_backoffs_on_overflow() -> None:
    cfg = PrecisionConfig(
        mode="fp4",
        params_dtype="fp32",
        main_params_dtype="fp32",
        main_grads_dtype="fp32",
        exp_avg_dtype="fp32",
        exp_avg_sq_dtype="fp32",
        activation_dtype="fp32",
        fp4_backend="emulated",
        loss_scale_init=16.0,
        loss_scale_growth_factor=2.0,
        loss_scale_backoff_factor=0.5,
        loss_scale_growth_interval=2,
        loss_scale_min=1.0,
        loss_scale_max=128.0,
    )
    controller = MixedPrecisionController(cfg, device=torch.device("cpu"))

    model = torch.nn.Linear(2, 1)
    model.weight.grad = torch.tensor([[float("inf"), 1.0]])
    model.bias.grad = torch.tensor([0.0])

    should_step = controller.prepare_optimizer_step(model)
    assert should_step is False
    controller.update_after_step(step_applied=False)

    assert controller.runtime_state.skipped_steps == 1
    assert controller.runtime_state.found_inf_steps == 1
    assert controller.runtime_state.loss_scale == 8.0


def test_mixed_precision_controller_growth_after_success_interval() -> None:
    cfg = PrecisionConfig(
        mode="fp16",
        activation_dtype="fp16",
        loss_scale_init=4.0,
        loss_scale_growth_factor=2.0,
        loss_scale_backoff_factor=0.5,
        loss_scale_growth_interval=2,
        loss_scale_min=1.0,
        loss_scale_max=64.0,
    )
    controller = MixedPrecisionController(cfg, device=torch.device("cpu"))

    controller.update_after_step(step_applied=True)
    assert controller.runtime_state.loss_scale == 4.0
    controller.update_after_step(step_applied=True)
    assert controller.runtime_state.loss_scale == 8.0
