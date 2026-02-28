"""DeepSeek-V3 precision recipe resolution and exception coverage."""

from __future__ import annotations

import argparse
from dataclasses import replace

import pytest
import torch

from src.layers import Linear
from src.runtime.mixed_precision import build_module_precision_resolver
from src.runtime.mixed_precision import finalize_module_precision_resolver
from src.runtime.mixed_precision import resolve_precision_config


def _args(**overrides) -> argparse.Namespace:
    base = {
        "precision_recipe": "default",
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
        "loss_scale_init": 1024.0,
        "loss_scale_growth_factor": 2.0,
        "loss_scale_backoff_factor": 0.5,
        "loss_scale_growth_interval": 100,
        "loss_scale_min": 1.0,
        "loss_scale_max": 65536.0,
        "fp8_param": False,
        "fp4_param": False,
        "fp4_param_format": "nf4",
        "persistent_scale_granularity": "per_channel",
        "module_pattern_type": "regex",
        "compute_lowbit_mode": None,
        "compute_lowbit_include": None,
        "compute_lowbit_exclude": None,
        "persistent_lowbit_mode": "off",
        "persistent_lowbit_include": None,
        "persistent_lowbit_exclude": None,
        "lowbit_master_ownership": "optimizer",
        "fp8_rounding": None,
        "fp8_activation_granularity": None,
        "fp8_weight_granularity": None,
        "fp8_comm_quant": None,
        "fp8_comm_granularity": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_deepseek_recipe_defaults_force_fp8_mode() -> None:
    config = resolve_precision_config(
        _args(precision_recipe="deepseek_v3"),
        torch.device("cpu"),
    )

    assert config.mode == "fp8"
    assert config.deepseek_v3_recipe is not None
    assert config.deepseek_v3_recipe.activation_quant_granularity == "tile_1x128"
    assert config.deepseek_v3_recipe.weight_quant_granularity == "block_128x128"
    assert config.deepseek_v3_recipe.rounding_mode == "stochastic"
    assert config.deepseek_v3_recipe.comm_quant_enabled is True


def test_deepseek_recipe_allows_explicit_overrides() -> None:
    config = resolve_precision_config(
        _args(
            precision_recipe="deepseek_v3",
            fp8=True,
            fp8_rounding="nearest",
            fp8_activation_granularity="tensor",
            fp8_weight_granularity="channel",
            fp8_comm_quant=False,
            fp8_comm_granularity="tensor",
        ),
        torch.device("cpu"),
    )

    assert config.mode == "fp8"
    assert config.deepseek_v3_recipe is not None
    assert config.deepseek_v3_recipe.rounding_mode == "nearest"
    assert config.deepseek_v3_recipe.activation_quant_granularity == "tensor"
    assert config.deepseek_v3_recipe.weight_quant_granularity == "channel"
    assert config.deepseek_v3_recipe.comm_quant_enabled is False
    assert config.deepseek_v3_recipe.comm_quant_granularity == "tensor"


def test_deepseek_recipe_rejects_non_fp8_modes() -> None:
    with pytest.raises(ValueError, match="requires --fp8"):
        resolve_precision_config(
            _args(precision_recipe="deepseek_v3", fp16=True),
            torch.device("cpu"),
        )


def test_high_precision_exception_patterns_disable_lowbit_on_matching_modules() -> None:
    config = resolve_precision_config(
        _args(precision_recipe="deepseek_v3"),
        torch.device("cpu"),
    )
    resolver = build_module_precision_resolver(config)

    final_norm = Linear(
        8,
        8,
        param_dtype=torch.float32,
        param_device=None,
        module_path="decoder.final_norm",
        precision_resolver=resolver,
    )
    mlp_proj = Linear(
        8,
        8,
        param_dtype=torch.float32,
        param_device=None,
        module_path="decoder.blocks.0.mlp.proj",
        precision_resolver=resolver,
    )

    norm_assignment = final_norm._module_precision_state.assignment  # type: ignore[attr-defined]
    proj_assignment = mlp_proj._module_precision_state.assignment  # type: ignore[attr-defined]
    assert norm_assignment.compute_lowbit_mode is None
    assert proj_assignment.compute_lowbit_mode == "fp8"

    summary = finalize_module_precision_resolver(resolver)
    assert summary.high_precision_exception_module_count >= 1
    assert summary.compute_lowbit_module_count >= 1


def test_high_precision_exception_patterns_fail_fast_on_no_match() -> None:
    config = resolve_precision_config(
        _args(precision_recipe="deepseek_v3"),
        torch.device("cpu"),
    )
    assert config.deepseek_v3_recipe is not None
    config.deepseek_v3_recipe = replace(
        config.deepseek_v3_recipe,
        high_precision_module_patterns=(r"^does_not_match_any_module$",),
    )

    resolver = build_module_precision_resolver(config)
    _ = Linear(
        8,
        8,
        param_dtype=torch.float32,
        param_device=None,
        module_path="decoder.blocks.0.mlp.proj",
        precision_resolver=resolver,
    )

    with pytest.raises(ValueError, match="matched zero modules"):
        finalize_module_precision_resolver(resolver)
