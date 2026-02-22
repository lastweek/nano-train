"""
Pure-Python unit tests for monitoring math + name selection.

These tests intentionally avoid importing torch.
"""

from __future__ import annotations

import math

from src.monitoring import (
    achieved_tflops,
    block_main_weight_names,
    candidate_sentinel_param_names,
    clip_coef,
    global_update_ratio,
    mfu,
    resolve_sentinel_blocks,
    update_ratio,
)


def test_resolve_sentinel_blocks_default() -> None:
    assert resolve_sentinel_blocks(12) == (0, 6, 11)


def test_resolve_sentinel_blocks_filters_and_dedups() -> None:
    assert resolve_sentinel_blocks(12, requested=(-1, 0, 0, 11, 12)) == (0, 11)


def test_block_main_weight_names() -> None:
    assert block_main_weight_names(3) == [
        "blocks.3.attention.qkv_proj.weight",
        "blocks.3.attention.out_proj.weight",
        "blocks.3.mlp.fc1.weight",
        "blocks.3.mlp.fc2.weight",
    ]


def test_candidate_sentinel_param_names_contains_expected() -> None:
    names = candidate_sentinel_param_names(sentinel_blocks=(0, 6, 11))
    assert "token_embeddings.weight" in names
    assert "position_embeddings.weight" in names
    assert "ln_f.weight" in names
    assert "ln_f.bias" in names
    assert "blocks.0.attention.qkv_proj.weight" in names
    assert "blocks.6.mlp.fc2.weight" in names
    assert "blocks.11.attention.out_proj.weight" in names


def test_clip_coef() -> None:
    assert clip_coef(grad_norm=0.5, max_norm=1.0) == 1.0
    c = clip_coef(grad_norm=5.0, max_norm=1.0, eps=1e-6)
    assert math.isclose(c, 1.0 / (5.0 + 1e-6), rel_tol=0.0, abs_tol=1e-12)


def test_global_update_ratio() -> None:
    ratio = global_update_ratio(
        1e-3,
        grad_norm_pre_clip=10.0,
        clip_coef_value=0.1,
        param_norm=100.0,
        eps=1e-12,
    )
    assert math.isclose(ratio, 1e-5, rel_tol=0.0, abs_tol=1e-12)


def test_update_ratio() -> None:
    ratio = update_ratio(1e-3, grad_norm=2.0, weight_norm=10.0, eps=1e-12)
    assert math.isclose(ratio, 2e-4, rel_tol=0.0, abs_tol=1e-12)


def test_achieved_tflops_and_mfu() -> None:
    tflops = achieved_tflops(1000.0, 1_000_000_000, flops_multiplier=6.0)
    assert math.isclose(tflops, 6.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(mfu(tflops, peak_tflops=12.0), 0.5, rel_tol=0.0, abs_tol=1e-12)
