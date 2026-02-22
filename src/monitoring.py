"""
Monitoring helpers (torch-free).

This module contains small, deterministic utilities used by `src/trainer.py` to keep monitoring
logic testable without importing PyTorch.
"""

from __future__ import annotations

from hashlib import sha256
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


def achieved_tflops(
    effective_tokens_per_second: float,
    num_params: int,
    *,
    flops_multiplier: float = 6.0,
) -> float:
    """
    Approximate achieved TFLOPs for dense Transformer pre-training.

    A common proxy is ~6 * N_params FLOPs per token for forward+backward+update.
    """
    effective_tokens_per_second = float(effective_tokens_per_second)
    num_params = int(num_params)
    flops_multiplier = float(flops_multiplier)
    if effective_tokens_per_second <= 0 or num_params <= 0 or flops_multiplier <= 0:
        return 0.0
    return (effective_tokens_per_second * flops_multiplier * float(num_params)) / 1e12


def mfu(achieved_tflops_value: float, peak_tflops: float) -> float:
    """Compute model FLOPs utilization from achieved vs peak TFLOPs."""
    achieved_tflops_value = float(achieved_tflops_value)
    peak_tflops = float(peak_tflops)
    if peak_tflops <= 0:
        return 0.0
    return achieved_tflops_value / peak_tflops


def eval_artifact_hash(
    *,
    input_ids_bytes: bytes,
    labels_bytes: bytes,
    input_dtype: str,
    labels_dtype: str,
    input_shape: tuple[int, ...],
    labels_shape: tuple[int, ...],
    vocab_size: int,
    max_seq_length: int,
    seed: int,
) -> str:
    """Compute a stable hash for fixed-probe artifacts and invariants."""
    hasher = sha256()
    hasher.update(bytes(str(input_dtype), "utf-8"))
    hasher.update(bytes(str(labels_dtype), "utf-8"))
    hasher.update(bytes(str(tuple(input_shape)), "utf-8"))
    hasher.update(bytes(str(tuple(labels_shape)), "utf-8"))
    hasher.update(bytes(str(int(vocab_size)), "utf-8"))
    hasher.update(bytes(str(int(max_seq_length)), "utf-8"))
    hasher.update(bytes(str(int(seed)), "utf-8"))
    hasher.update(bytes(input_ids_bytes))
    hasher.update(bytes(labels_bytes))
    return hasher.hexdigest()
