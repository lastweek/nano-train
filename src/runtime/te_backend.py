"""Low-bit backend interfaces and backend selection for mixed precision."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional
from typing import Protocol
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from src.runtime.contracts import PrecisionConfig
    from src.runtime.contracts import PrecisionMode
else:
    PrecisionConfig = object
    PrecisionMode = str


class LowBitBackend(Protocol):
    """Backend interface used by low-bit precision-aware linear layers."""

    name: str

    def linear(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        mode: PrecisionMode,
        config: PrecisionConfig,
    ) -> torch.Tensor:
        """Run one low-bit-aware linear op and return output tensor."""
        ...


@dataclass
class ActiveLowBitContext:
    """Active low-bit backend and run-level precision settings.

    This context only provides backend objects and recipe/config data. Module-level
    `ModulePrecisionAssignment` decides whether a given layer runs low-bit and which mode to use.
    """

    backend_by_mode: dict[str, LowBitBackend]
    default_mode: PrecisionMode
    config: PrecisionConfig


_ACTIVE_LOWBIT_CONTEXT: Optional[ActiveLowBitContext] = None


def set_active_lowbit_context(context: Optional[ActiveLowBitContext]) -> None:
    """Install active low-bit backend context for precision-aware linear layers."""
    global _ACTIVE_LOWBIT_CONTEXT
    _ACTIVE_LOWBIT_CONTEXT = context


def get_active_lowbit_context() -> Optional[ActiveLowBitContext]:
    """Return active low-bit backend context if low-bit mode is enabled."""
    return _ACTIVE_LOWBIT_CONTEXT


def get_active_lowbit_backend(mode: PrecisionMode) -> Optional[LowBitBackend]:
    """Return backend for the requested low-bit mode from active context."""
    context = get_active_lowbit_context()
    if context is None:
        return None
    return context.backend_by_mode.get(str(mode))


def clear_active_lowbit_context() -> None:
    """Clear active low-bit context."""
    set_active_lowbit_context(None)


def _ste_fake_quantize(x: torch.Tensor, *, bits: int) -> torch.Tensor:
    """Fake-quantize tensor with STE to emulate low-bit numerics in forward pass."""
    if x.numel() == 0:
        return x
    if bits < 2:
        raise ValueError("bits must be >= 2")

    x_f = x.float()
    qmax = (1 << (bits - 1)) - 1
    scale = x_f.detach().abs().amax()
    if not torch.isfinite(scale) or scale <= 0:
        scale = x_f.new_tensor(1.0)
    else:
        scale = scale / float(qmax)

    q = torch.clamp(torch.round(x_f / scale), min=-qmax, max=qmax)
    dq = q * scale
    dq_cast = dq.to(dtype=x.dtype)
    return x + (dq_cast - x).detach()


class EmulatedLowBitBackend:
    """Software-emulated low-bit backend used for CI/dev and FP4 experiments."""

    name = "emulated"

    def linear(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        mode: PrecisionMode,
        config: PrecisionConfig,
    ) -> torch.Tensor:
        del config
        bits = 8 if mode == "fp8" else 4
        input_q = _ste_fake_quantize(input_tensor, bits=bits)
        weight_q = _ste_fake_quantize(weight, bits=bits)

        compute_dtype = torch.float32
        bias_compute = bias.to(dtype=compute_dtype) if bias is not None else None
        output = F.linear(
            input_q.to(dtype=compute_dtype),
            weight_q.to(dtype=compute_dtype),
            bias_compute,
        )
        return output.to(dtype=input_tensor.dtype)


class TransformerEngineLowBitBackend:
    """Optional low-bit backend that routes linear ops through TE FP8 autocast context."""

    name = "transformer_engine"

    def __init__(self) -> None:
        try:
            import transformer_engine.pytorch as te  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional dependency.
            raise RuntimeError(
                "Transformer Engine backend requested but transformer_engine is unavailable"
            ) from exc

        self._te = te

    def _fp8_autocast_context(self, config: PrecisionConfig):
        fp8_autocast = getattr(self._te, "fp8_autocast", None)
        if fp8_autocast is None:
            return nullcontext()

        fp8_recipe = None
        try:  # pragma: no cover - optional API surface depends on TE version.
            from transformer_engine.common import recipe as te_recipe  # type: ignore

            format_map = {
                "e4m3": te_recipe.Format.E4M3,
                "hybrid": te_recipe.Format.HYBRID,
            }
            fp8_recipe = te_recipe.DelayedScaling(
                fp8_format=format_map[config.fp8_format],
                amax_history_len=int(config.fp8_amax_history_len),
                amax_compute_algo=str(config.fp8_amax_compute_algo),
            )
        except Exception:
            fp8_recipe = None

        if fp8_recipe is None:
            return fp8_autocast(enabled=True)
        return fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)

    def linear(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        mode: PrecisionMode,
        config: PrecisionConfig,
    ) -> torch.Tensor:
        if mode != "fp8":
            raise ValueError("TransformerEngineLowBitBackend only supports fp8 mode")

        with self._fp8_autocast_context(config):
            return F.linear(input_tensor, weight, bias)


def build_lowbit_backend(config: PrecisionConfig) -> Optional[LowBitBackend]:
    """Build low-bit backend for the resolved precision mode."""
    return build_lowbit_backend_for_mode(config, mode=config.mode)


def build_lowbit_backend_for_mode(
    config: PrecisionConfig,
    *,
    mode: PrecisionMode,
) -> Optional[LowBitBackend]:
    """Build low-bit backend for one explicit low-bit mode."""
    if mode == "fp8":
        if config.fp8_backend == "transformer_engine":
            return TransformerEngineLowBitBackend()
        if config.fp8_backend == "emulated":
            return EmulatedLowBitBackend()
        raise ValueError(f"Unsupported fp8 backend: {config.fp8_backend}")

    if mode == "fp4":
        if config.fp4_backend == "emulated":
            return EmulatedLowBitBackend()
        raise ValueError(f"Unsupported fp4 backend: {config.fp4_backend}")

    return None
