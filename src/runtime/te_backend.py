"""Low-bit backend interfaces and backend selection for mixed precision."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional
from typing import Protocol
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from src.runtime.contracts import LowBitKernelSpec
    from src.runtime.contracts import PrecisionConfig
    from src.runtime.contracts import PrecisionMode
else:
    LowBitKernelSpec = object
    PrecisionConfig = object
    PrecisionMode = str


class LowBitBackend(Protocol):
    """Backend interface used by precision-aware low-bit linear layers."""

    name: str
    mode: PrecisionMode

    def linear(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run one low-bit-aware linear op and return output tensor."""
        ...


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


def _stochastic_round(values: torch.Tensor) -> torch.Tensor:
    noise = torch.rand_like(values)
    positive = torch.floor(values + noise)
    negative = torch.ceil(values - noise)
    return torch.where(values >= 0, positive, negative)


def _round_values(values: torch.Tensor, *, rounding_mode: str) -> torch.Tensor:
    if rounding_mode == "stochastic":
        return _stochastic_round(values)
    return torch.round(values)


def _safe_scale(scale: torch.Tensor) -> torch.Tensor:
    return torch.where(
        torch.isfinite(scale) & (scale > 0),
        scale,
        torch.ones_like(scale),
    )


def _tile_quant_dequant(
    x_f: torch.Tensor,
    *,
    bits: int,
    tile_size: int,
    rounding_mode: str,
) -> torch.Tensor:
    x_2d = x_f.reshape(-1, x_f.shape[-1])
    n_rows, width = x_2d.shape
    n_tiles = (width + tile_size - 1) // tile_size

    pad_width = n_tiles * tile_size - width
    if pad_width > 0:
        x_2d = F.pad(x_2d, (0, pad_width))

    x_tiles = x_2d.reshape(n_rows, n_tiles, tile_size)
    qmax = (1 << (bits - 1)) - 1
    scale = x_tiles.detach().abs().amax(dim=-1, keepdim=True) / float(qmax)
    scale = _safe_scale(scale)
    q = torch.clamp(
        _round_values(x_tiles / scale, rounding_mode=rounding_mode),
        min=-qmax,
        max=qmax,
    )
    dq = q * scale
    dq = dq.reshape(n_rows, n_tiles * tile_size)
    if pad_width > 0:
        dq = dq[:, :width]
    return dq.reshape_as(x_f)


def _block_quant_dequant(
    x_f: torch.Tensor,
    *,
    bits: int,
    block_h: int,
    block_w: int,
    rounding_mode: str,
) -> torch.Tensor:
    if x_f.dim() != 2:
        return _tile_quant_dequant(
            x_f,
            bits=bits,
            tile_size=block_w,
            rounding_mode=rounding_mode,
        )
    rows, cols = x_f.shape
    row_blocks = (rows + block_h - 1) // block_h
    col_blocks = (cols + block_w - 1) // block_w

    pad_rows = row_blocks * block_h - rows
    pad_cols = col_blocks * block_w - cols
    if pad_rows > 0 or pad_cols > 0:
        x_f = F.pad(x_f, (0, pad_cols, 0, pad_rows))

    x_blocks = x_f.reshape(row_blocks, block_h, col_blocks, block_w).permute(0, 2, 1, 3)
    qmax = (1 << (bits - 1)) - 1
    scale = x_blocks.detach().abs().amax(dim=(-1, -2), keepdim=True) / float(qmax)
    scale = _safe_scale(scale)
    q = torch.clamp(
        _round_values(x_blocks / scale, rounding_mode=rounding_mode),
        min=-qmax,
        max=qmax,
    )
    dq = (q * scale).permute(0, 2, 1, 3).reshape(row_blocks * block_h, col_blocks * block_w)
    if pad_rows > 0 or pad_cols > 0:
        dq = dq[:rows, :cols]
    return dq


def _quantize_dequantize_with_granularity(
    x: torch.Tensor,
    *,
    bits: int,
    granularity: str,
    rounding_mode: str,
    channel_axis: int,
) -> torch.Tensor:
    if x.numel() == 0:
        return x
    x_f = x.float()
    qmax = (1 << (bits - 1)) - 1

    if granularity == "tensor":
        scale = x_f.detach().abs().amax() / float(qmax)
        scale = _safe_scale(scale)
        q = torch.clamp(
            _round_values(x_f / scale, rounding_mode=rounding_mode),
            min=-qmax,
            max=qmax,
        )
        return (q * scale).to(dtype=x.dtype)

    if granularity == "channel":
        axis = channel_axis if channel_axis >= 0 else x_f.dim() + channel_axis
        reduce_dims = [idx for idx in range(x_f.dim()) if idx != axis]
        scale = x_f.detach().abs().amax(dim=reduce_dims, keepdim=True) / float(qmax)
        scale = _safe_scale(scale)
        q = torch.clamp(
            _round_values(x_f / scale, rounding_mode=rounding_mode),
            min=-qmax,
            max=qmax,
        )
        return (q * scale).to(dtype=x.dtype)

    if granularity == "tile_1x128":
        return _tile_quant_dequant(
            x_f,
            bits=bits,
            tile_size=128,
            rounding_mode=rounding_mode,
        ).to(dtype=x.dtype)

    if granularity == "block_128x128":
        return _block_quant_dequant(
            x_f,
            bits=bits,
            block_h=128,
            block_w=128,
            rounding_mode=rounding_mode,
        ).to(dtype=x.dtype)

    return _ste_fake_quantize(x, bits=bits)


class EmulatedLowBitBackend:
    """Software-emulated low-bit backend used for CI/dev and FP4 experiments."""

    name = "emulated"

    def __init__(
        self,
        *,
        mode: PrecisionMode,
        kernel_spec: Optional["LowBitKernelSpec"] = None,
    ) -> None:
        if mode not in ("fp8", "fp4"):
            raise ValueError("EmulatedLowBitBackend supports only fp8/fp4")
        self.mode = mode
        self._activation_granularity = "tensor"
        self._weight_granularity = "tensor"
        self._rounding_mode = "nearest"
        if kernel_spec is not None:
            self._activation_granularity = str(kernel_spec.activation_quant_granularity)
            self._weight_granularity = str(kernel_spec.weight_quant_granularity)
            self._rounding_mode = str(kernel_spec.rounding_mode)

    def linear(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bits = 8 if self.mode == "fp8" else 4
        input_dq = _quantize_dequantize_with_granularity(
            input_tensor,
            bits=bits,
            granularity=self._activation_granularity,
            rounding_mode=self._rounding_mode,
            channel_axis=-1,
        )
        weight_dq = _quantize_dequantize_with_granularity(
            weight,
            bits=bits,
            granularity=self._weight_granularity,
            rounding_mode=self._rounding_mode,
            channel_axis=0,
        )
        input_q = input_tensor + (input_dq - input_tensor).detach()
        weight_q = weight + (weight_dq - weight).detach()

        compute_dtype = torch.float32
        bias_compute = bias.to(dtype=compute_dtype) if bias is not None else None
        output = F.linear(
            input_q.to(dtype=compute_dtype),
            weight_q.to(dtype=compute_dtype),
            bias_compute,
        )
        return output.to(dtype=input_tensor.dtype)


class TransformerEngineLowBitBackend:
    """Optional backend that routes FP8 linears through TE linear kernels."""

    name = "transformer_engine"

    def __init__(
        self,
        *,
        config: PrecisionConfig,
        mode: PrecisionMode,
        kernel_spec: Optional["LowBitKernelSpec"],
    ) -> None:
        if mode != "fp8":
            raise ValueError("TransformerEngineLowBitBackend only supports fp8 mode")
        if kernel_spec is None:
            raise RuntimeError(
                "Transformer Engine backend requires a per-module LowBitKernelSpec"
            )

        try:
            import transformer_engine.pytorch as te  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency.
            raise RuntimeError(
                "Transformer Engine backend requested but transformer_engine is unavailable"
            ) from exc

        self.mode = mode
        self.config = config
        self._te = te
        self._kernel_spec = kernel_spec
        self._te_linear = te.Linear(
            in_features=int(kernel_spec.in_features),
            out_features=int(kernel_spec.out_features),
            bias=bool(kernel_spec.has_bias),
        )
        self._bound_weight: Optional[nn.Parameter] = None
        self._bound_bias: Optional[nn.Parameter] = None

    def bind_parameters(
        self,
        weight: nn.Parameter,
        bias: Optional[nn.Parameter],
    ) -> None:
        """Bind module-owned trainable parameters to the TE linear module."""
        if not isinstance(weight, nn.Parameter):
            raise TypeError("TransformerEngineLowBitBackend expects nn.Parameter weight binding")
        if tuple(weight.shape) != (
            int(self._kernel_spec.out_features),
            int(self._kernel_spec.in_features),
        ):
            raise ValueError(
                "TE backend weight shape mismatch: "
                f"expected={(self._kernel_spec.out_features, self._kernel_spec.in_features)} "
                f"got={tuple(weight.shape)}"
            )

        if self._kernel_spec.has_bias:
            if bias is None:
                raise ValueError("TE backend expected bias parameter but received None")
            if not isinstance(bias, nn.Parameter):
                raise TypeError("TransformerEngineLowBitBackend expects nn.Parameter bias binding")
            if tuple(bias.shape) != (int(self._kernel_spec.out_features),):
                raise ValueError(
                    "TE backend bias shape mismatch: "
                    f"expected={(self._kernel_spec.out_features,)} got={tuple(bias.shape)}"
                )
            self._te_linear.bias = bias
            self._bound_bias = bias
        else:
            self._bound_bias = None

        self._te_linear.weight = weight
        self._bound_weight = weight

    def _te_autocast_context(self):
        te_autocast = getattr(self._te, "autocast", None)
        if te_autocast is None:
            te_autocast = getattr(self._te, "fp8_autocast", None)
        if te_autocast is None:
            return nullcontext()

        fp8_recipe = None
        try:  # pragma: no cover - optional API surface depends on TE version.
            from transformer_engine.common import recipe as te_recipe  # type: ignore

            format_map = {
                "e4m3": te_recipe.Format.E4M3,
                "hybrid": te_recipe.Format.HYBRID,
            }
            fp8_recipe = te_recipe.DelayedScaling(
                fp8_format=format_map[self.config.fp8_format],
                amax_history_len=int(self.config.fp8_amax_history_len),
                amax_compute_algo=str(self.config.fp8_amax_compute_algo),
            )
        except Exception:
            fp8_recipe = None

        if fp8_recipe is None:
            return te_autocast(enabled=True)
        return te_autocast(enabled=True, fp8_recipe=fp8_recipe)

    def linear(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self._bound_weight is None:
            raise RuntimeError(
                "Transformer Engine backend used before parameter binding. "
                "Layer init must bind module-owned trainable parameters."
            )

        module = self._te_linear.to(device=input_tensor.device)
        if weight is self._bound_weight and (
            bias is self._bound_bias or (bias is None and self._bound_bias is None)
        ):
            with self._te_autocast_context():
                return module(input_tensor)

        # Fallback path for non-parameter tensors (for example, STE-transformed weights).
        with self._te_autocast_context():
            return F.linear(input_tensor, weight, bias)


def build_lowbit_backend(config: PrecisionConfig) -> Optional[LowBitBackend]:
    """Build low-bit backend for the resolved precision mode."""
    return build_lowbit_backend_for_mode(config, mode=config.mode, kernel_spec=None)


def build_lowbit_backend_for_mode(
    config: PrecisionConfig,
    *,
    mode: PrecisionMode,
    kernel_spec: Optional["LowBitKernelSpec"],
) -> Optional[LowBitBackend]:
    """Build low-bit backend for one explicit low-bit mode."""
    if mode == "fp8":
        if config.fp8_backend == "transformer_engine":
            return TransformerEngineLowBitBackend(
                config=config,
                mode=mode,
                kernel_spec=kernel_spec,
            )
        if config.fp8_backend == "emulated":
            return EmulatedLowBitBackend(mode=mode, kernel_spec=kernel_spec)
        raise ValueError(f"Unsupported fp8 backend: {config.fp8_backend}")

    if mode == "fp4":
        if config.fp4_backend == "emulated":
            return EmulatedLowBitBackend(mode=mode, kernel_spec=kernel_spec)
        raise ValueError(f"Unsupported fp4 backend: {config.fp4_backend}")

    return None
