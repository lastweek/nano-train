"""
Monitoring helpers (torch-free).

This module contains small, deterministic utilities used by `src/trainer.py` to keep monitoring
logic testable without importing PyTorch.
"""

from __future__ import annotations

from typing import Iterable, Optional


def resolve_sentinel_blocks(
    num_layers: int,
    requested: Optional[Iterable[int]] = None,
) -> tuple[int, ...]:
    """
    Resolve sentinel block indices used for bounded monitoring.

    Default: (0, mid, last) where mid = num_layers // 2.
    """
    num_layers = int(num_layers)
    if num_layers <= 0:
        return ()

    if requested is None:
        requested = (0, num_layers // 2, num_layers - 1)

    resolved: list[int] = []
    seen: set[int] = set()
    for idx in requested:
        block_idx = int(idx)
        if block_idx < 0 or block_idx >= num_layers:
            continue
        if block_idx in seen:
            continue
        resolved.append(block_idx)
        seen.add(block_idx)

    return tuple(resolved) if resolved else (0,)


def block_main_weight_names(block_idx: int) -> list[str]:
    """Return the "main" weight tensors for a transformer block."""
    block_idx = int(block_idx)
    return [
        f"blocks.{block_idx}.attention.qkv_proj.weight",
        f"blocks.{block_idx}.attention.out_proj.weight",
        f"blocks.{block_idx}.mlp.fc1.weight",
        f"blocks.{block_idx}.mlp.fc2.weight",
    ]


def candidate_sentinel_param_names(*, sentinel_blocks: Iterable[int]) -> set[str]:
    """
    Return the deterministic sentinel parameter name set (unfiltered).

    Trainer will later filter this set to the names actually present in the model.
    """
    names: set[str] = {
        "token_embeddings.weight",
        "position_embeddings.weight",
        "ln_f.weight",
        "ln_f.bias",
        "lm_head.weight",
    }

    for block_idx in sentinel_blocks:
        for name in block_main_weight_names(int(block_idx)):
            names.add(name)

    return names


def clip_coef(grad_norm: float, max_norm: float, *, eps: float = 1e-6) -> float:
    """Compute the global gradient clipping coefficient."""
    grad_norm = float(grad_norm)
    max_norm = float(max_norm)
    if max_norm <= 0:
        return 1.0
    return min(1.0, max_norm / (grad_norm + float(eps)))


def global_update_ratio(
    lr: float,
    *,
    grad_norm_pre_clip: float,
    clip_coef_value: float,
    param_norm: float,
    eps: float = 1e-12,
) -> float:
    """
    Compute a proxy "global update ratio" scalar.

    ratio_global = (lr * (clip_coef * grad_norm_pre_clip)) / (param_norm + eps)
    """
    lr = float(lr)
    grad_norm_pre_clip = float(grad_norm_pre_clip)
    clip_coef_value = float(clip_coef_value)
    param_norm = float(param_norm)
    if param_norm <= 0:
        return 0.0
    return lr * (clip_coef_value * grad_norm_pre_clip) / (param_norm + float(eps))


def update_ratio(
    lr: float,
    *,
    grad_norm: float,
    weight_norm: float,
    eps: float = 1e-12,
) -> float:
    """Compute a proxy per-tensor update ratio scalar."""
    lr = float(lr)
    grad_norm = float(grad_norm)
    weight_norm = float(weight_norm)
    if weight_norm <= 0:
        return 0.0
    return lr * grad_norm / (weight_norm + float(eps))

