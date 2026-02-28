"""
Native implementations of core neural network layers.

Each layer includes detailed visualization comments showing concrete
examples of how the operations work.

COMPARION TO ORIGINAL PYTORCH API:
Every layer shows both our native implementation and the equivalent torch.nn API.

Example:
    # NATIVE: Our implementation
    linear = Linear(in_features, out_features)

    # ORIGINAL: torch.nn API
    linear_orig = torch.nn.Linear(in_features, out_features)

Both produce identical outputs and gradients!
"""

import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.runtime.contracts import LowBitKernelSpec
    from src.runtime.contracts import LowBitComputeMode
    from src.runtime.contracts import LowBitCapableModuleType
    from src.runtime.contracts import ModulePrecisionInitState
    from src.runtime.contracts import ModulePrecisionResolver
    from src.runtime.contracts import PersistentScaleGranularity
else:
    LowBitKernelSpec = object
    LowBitComputeMode = str
    LowBitCapableModuleType = str
    ModulePrecisionInitState = object
    ModulePrecisionResolver = object
    PersistentScaleGranularity = str


_COMPUTE_DTYPE_ALIAS_TO_TORCH = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def _resolve_compute_torch_dtype(dtype_alias: str) -> torch.dtype:
    try:
        return _COMPUTE_DTYPE_ALIAS_TO_TORCH[dtype_alias]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported compute dtype override alias: {dtype_alias}"
        ) from exc


def _nf4_codebook(device: torch.device) -> torch.Tensor:
    """Return NF4 codebook values used for FP4 persistent quantization."""
    return torch.tensor(
        [
            -1.0,
            -0.6961928,
            -0.52507305,
            -0.3949175,
            -0.28444138,
            -0.18477343,
            -0.09105004,
            0.0,
            0.0795803,
            0.1609302,
            0.2461123,
            0.33791524,
            0.44070983,
            0.562617,
            0.72295684,
            1.0,
        ],
        dtype=torch.float32,
        device=device,
    )


def _safe_scale(
    tensor: torch.Tensor,
    *,
    granularity: "PersistentScaleGranularity",
) -> torch.Tensor:
    """Compute safe positive scales for per-tensor or per-channel quantization."""
    tensor_f = tensor.float()
    if granularity == "per_channel" and tensor.dim() > 1:
        reduce_dims = tuple(range(1, tensor.dim()))
        scale = tensor_f.abs().amax(dim=reduce_dims, keepdim=True)
    else:
        scale = tensor_f.abs().amax()

    one = torch.ones_like(scale, dtype=torch.float32)
    scale = torch.where(torch.isfinite(scale) & (scale > 0), scale, one)
    return scale


def _quantize_fp8_persistent(
    tensor: torch.Tensor,
    *,
    granularity: "PersistentScaleGranularity",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor into FP8 persistent representation with scales."""
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        raise RuntimeError("Persistent FP8 requires torch.float8_e4m3fn support")

    scale = _safe_scale(tensor, granularity=granularity)
    normalized = (tensor.float() / scale).clamp(min=-448.0, max=448.0)
    fp8_values = normalized.to(dtype=fp8_dtype)
    return fp8_values, scale


def _dequantize_fp8_persistent(
    fp8_values: torch.Tensor,
    scale: torch.Tensor,
    *,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize FP8 persistent representation into target dtype."""
    return (fp8_values.float() * scale).to(dtype=target_dtype)


def _pack_nibbles(indices: torch.Tensor) -> torch.Tensor:
    """Pack uint8 [0, 15] values into packed uint8 nibbles."""
    flat = indices.reshape(-1).to(dtype=torch.uint8)
    if flat.numel() % 2 == 1:
        pad = torch.zeros(1, dtype=torch.uint8, device=flat.device)
        flat = torch.cat((flat, pad), dim=0)

    low = flat[0::2]
    high = flat[1::2] << 4
    return (low | high).contiguous()


def _unpack_nibbles(
    packed: torch.Tensor,
    *,
    num_values: int,
) -> torch.Tensor:
    """Unpack packed uint8 nibbles into long indices [0, 15]."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked = torch.empty(packed.numel() * 2, dtype=torch.long, device=packed.device)
    unpacked[0::2] = low.long()
    unpacked[1::2] = high.long()
    return unpacked[:num_values]


def _quantize_nf4_persistent(
    tensor: torch.Tensor,
    *,
    granularity: "PersistentScaleGranularity",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor into NF4 packed codes plus scale metadata."""
    scale = _safe_scale(tensor, granularity=granularity)
    normalized = tensor.float() / scale
    codebook = _nf4_codebook(tensor.device)
    clipped = normalized.clamp(min=float(codebook[0]), max=float(codebook[-1]))
    distances = (clipped.reshape(-1, 1) - codebook.reshape(1, -1)).abs()
    indices = distances.argmin(dim=1).to(dtype=torch.uint8)
    packed = _pack_nibbles(indices)
    return packed, scale


def _dequantize_nf4_persistent(
    packed_codes: torch.Tensor,
    scale: torch.Tensor,
    *,
    num_values: int,
    target_shape: torch.Size,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize NF4 packed code representation into target dtype tensor."""
    codebook = _nf4_codebook(packed_codes.device)
    unpacked = _unpack_nibbles(packed_codes, num_values=num_values)
    dequant = codebook[unpacked].reshape(target_shape) * scale
    return dequant.to(dtype=target_dtype)


class _ModulePrecisionStateMixin:
    """Attach constructor-time module precision state to parameterized layers."""

    def _init_module_precision_state(
        self,
        *,
        module_path: str,
        module_type: str,
        precision_resolver: "ModulePrecisionResolver",
        lowbit_capable_type: Optional["LowBitCapableModuleType"],
        kernel_spec: Optional["LowBitKernelSpec"],
    ) -> None:
        self._module_precision_state: "ModulePrecisionInitState" = (
            precision_resolver.resolve_module_init_state(
                module_path=module_path,
                module_type=module_type,
                lowbit_capable_type=lowbit_capable_type,
                kernel_spec=kernel_spec,
            )
        )

    def _module_compute_dtype_override(self) -> Optional[torch.dtype]:
        assignment = self._module_precision_state.assignment
        dtype_alias = getattr(assignment, "compute_dtype_override", None)
        if dtype_alias is None:
            return None
        return _resolve_compute_torch_dtype(str(dtype_alias))


class _LowBitPrecisionLinearMixin(_ModulePrecisionStateMixin):
    """Reusable per-module low-bit compute and persistent parameter hooks."""

    def _init_lowbit_precision_state(
        self,
        *,
        module_path: str,
        module_type: str,
        precision_resolver: "ModulePrecisionResolver",
        lowbit_capable_type: "LowBitCapableModuleType",
        kernel_spec: "LowBitKernelSpec",
    ) -> None:
        self._init_module_precision_state(
            module_path=module_path,
            module_type=module_type,
            precision_resolver=precision_resolver,
            lowbit_capable_type=lowbit_capable_type,
            kernel_spec=kernel_spec,
        )
        self.register_buffer("_persistent_fp8_weight", torch.empty(0), persistent=True)
        self.register_buffer(
            "_persistent_fp4_codes",
            torch.empty(0, dtype=torch.uint8),
            persistent=True,
        )
        self.register_buffer("_persistent_scale", torch.empty(0), persistent=True)
        self.register_buffer(
            "_persistent_numel",
            torch.zeros((), dtype=torch.int64),
            persistent=True,
        )
        backend = self._module_precision_state.lowbit_backend
        bind_fn = getattr(backend, "bind_parameters", None)
        if callable(bind_fn):
            bind_fn(self.weight, self.bias)

    @property
    def module_precision_assignment(self):
        """Return constructor-time precision assignment resolved for this module."""
        return self._module_precision_state.assignment

    @property
    def module_path(self) -> str:
        """Return canonical module path used by precision policy matching."""
        return str(self._module_precision_state.assignment.module_name)

    @property
    def master_ownership_mode(self) -> str:
        """Return configured master ownership mode for this module."""
        return str(getattr(self._module_precision_state, "master_ownership_mode", "module"))

    def bind_optimizer_master_weight(self, master_weight: nn.Parameter) -> None:
        """Bind optimizer-owned master weight parameter to this module."""
        if not isinstance(master_weight, nn.Parameter):
            raise TypeError("master_weight must be nn.Parameter")
        if master_weight.shape != self.weight.shape:
            raise ValueError(
                "optimizer master weight shape mismatch: "
                f"expected={tuple(self.weight.shape)} got={tuple(master_weight.shape)}"
            )
        with torch.no_grad():
            master_weight.copy_(self.weight.detach().to(device=master_weight.device))
        self.weight = master_weight

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Load parameters while tolerating legacy checkpoints without low-bit buffers."""
        super()._load_from_state_dict(  # type: ignore[misc]
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        optional_suffixes = (
            "_persistent_fp8_weight",
            "_persistent_fp4_codes",
            "_persistent_scale",
            "_persistent_numel",
        )
        optional_keys = {f"{prefix}{suffix}" for suffix in optional_suffixes}
        missing_keys[:] = [key for key in missing_keys if key not in optional_keys]

    def _persistent_mode(self) -> str:
        assignment = self._module_precision_state.assignment
        return str(getattr(assignment, "persistent_lowbit_mode", "off"))

    def _persistent_granularity(self) -> str:
        assignment = self._module_precision_state.assignment
        return str(getattr(assignment, "persistent_scale_granularity", "per_channel"))

    def _compute_lowbit_mode(
        self,
    ) -> Optional["LowBitComputeMode"]:
        assignment = self._module_precision_state.assignment
        assignment_mode = getattr(assignment, "compute_lowbit_mode", None)
        if assignment_mode is None:
            return None
        mode_text = str(assignment_mode)
        if mode_text not in ("fp8", "fp4"):
            return None
        return mode_text  # type: ignore[return-value]

    def _compute_dtype_override(self) -> Optional[torch.dtype]:
        return self._module_compute_dtype_override()

    def _linear_with_dtype_override(
        self,
        *,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        compute_dtype = self._compute_dtype_override()
        if compute_dtype is None:
            return None
        x_compute = x.to(dtype=compute_dtype)
        weight_compute = weight.to(dtype=compute_dtype)
        bias_compute = bias.to(dtype=compute_dtype) if bias is not None else None
        return F.linear(x_compute, weight_compute, bias_compute)

    def _clear_persistent_lowbit_buffers(self) -> None:
        self._persistent_fp8_weight = torch.empty(0, device=self.weight.device)
        self._persistent_fp4_codes = torch.empty(0, dtype=torch.uint8, device=self.weight.device)
        self._persistent_scale = torch.empty(0, device=self.weight.device)
        self._persistent_numel = torch.zeros((), dtype=torch.int64, device=self.weight.device)

    @torch.no_grad()
    def refresh_persistent_lowbit_params(self) -> None:
        """Refresh persistent low-bit buffers from current master weight."""
        mode = self._persistent_mode()
        if mode == "off":
            self._clear_persistent_lowbit_buffers()
            return

        granularity = self._persistent_granularity()
        if granularity not in ("per_tensor", "per_channel"):
            raise ValueError(
                f"Unsupported persistent_scale_granularity={granularity!r} "
                "for low-bit persistent params"
            )

        if mode == "fp8":
            fp8_weight, scale = _quantize_fp8_persistent(
                self.weight.detach(),
                granularity=granularity,  # type: ignore[arg-type]
            )
            self._persistent_fp8_weight = fp8_weight
            self._persistent_fp4_codes = torch.empty(
                0,
                dtype=torch.uint8,
                device=self.weight.device,
            )
            self._persistent_scale = scale
            self._persistent_numel = torch.tensor(
                int(self.weight.numel()),
                dtype=torch.int64,
                device=self.weight.device,
            )
            return

        if mode == "fp4":
            assignment = self._module_precision_state.assignment
            fp4_format = str(getattr(assignment, "fp4_persistent_format", "nf4"))
            if fp4_format != "nf4":
                raise ValueError("Only nf4 is supported for FP4 persistent params")

            packed, scale = _quantize_nf4_persistent(
                self.weight.detach(),
                granularity=granularity,  # type: ignore[arg-type]
            )
            self._persistent_fp4_codes = packed
            self._persistent_fp8_weight = torch.empty(0, device=self.weight.device)
            self._persistent_scale = scale
            self._persistent_numel = torch.tensor(
                int(self.weight.numel()),
                dtype=torch.int64,
                device=self.weight.device,
            )
            return

        raise ValueError(f"Unsupported persistent low-bit mode: {mode}")

    def _persistent_weight_ste(self) -> torch.Tensor:
        """Return weight used in forward with STE bridge to master parameter."""
        mode = self._persistent_mode()
        if mode == "off":
            return self.weight

        if mode == "fp8":
            if self._persistent_fp8_weight.numel() != self.weight.numel():
                self.refresh_persistent_lowbit_params()
            dequant = _dequantize_fp8_persistent(
                self._persistent_fp8_weight,
                self._persistent_scale,
                target_dtype=self.weight.dtype,
            )
        elif mode == "fp4":
            if int(self._persistent_numel.item()) != int(self.weight.numel()):
                self.refresh_persistent_lowbit_params()
            dequant = _dequantize_nf4_persistent(
                self._persistent_fp4_codes,
                self._persistent_scale,
                num_values=int(self.weight.numel()),
                target_shape=self.weight.shape,
                target_dtype=self.weight.dtype,
            )
        else:
            raise ValueError(f"Unsupported persistent low-bit mode: {mode}")

        return self.weight + (dequant - self.weight).detach()

    def _dispatch_lowbit_linear(
        self,
        *,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Run low-bit backend if module policy selects one; otherwise return None."""
        compute_mode = self._compute_lowbit_mode()
        if compute_mode is None:
            return None

        backend = self._module_precision_state.lowbit_backend
        if backend is None:
            module_name = self._module_precision_state.assignment.module_name
            raise RuntimeError(
                f"Module {self.__class__.__name__} ({module_name}) requested compute mode "
                f"{compute_mode} without a bound low-bit backend"
            )

        return backend.linear(x, weight, bias)


# =============================================================================
# LINEAR LAYER (Fully Connected / Dense)
# =============================================================================

class Linear(_LowBitPrecisionLinearMixin, nn.Module):
    """
    Linear transformation: y = xA^T + b

    Also called: Fully Connected, Dense, Affine transformation

    COMPARISON:
        # NATIVE (this file):
        from src.layers import Linear
        layer = Linear(10, 20)

        # ORIGINAL PyTorch:
        import torch.nn as nn
        layer_orig = nn.Linear(10, 20)

    Both use identical operations under the hood and produce the same results!

    Used in:
    - QKV projections in attention
    - Output projections
    - MLP layers (fc1, fc2)
    - Final language model head

    VISUALIZED with concrete example:

    Input: Batch of 2 sequences, each with 3 features
    -------
    x (shape: 2x3):
        [[1.0, 2.0, 3.0],   ← sequence 1
         [4.0, 5.0, 6.0]]   ← sequence 2

    Weight matrix A (shape: 4x3):
        [[0.1, 0.2, 0.3],   ← output neuron 1
         [0.4, 0.5, 0.6],   ← output neuron 2
         [0.7, 0.8, 0.9],   ← output neuron 3
         [1.0, 1.1, 1.2]]   ← output neuron 4

    Bias b (shape: 4):
        [0.1, 0.2, 0.3, 0.4]

    Step 1: Matrix multiplication x @ A.T
    -------
    For sequence 1 [1.0, 2.0, 3.0]:
        output[0] = 1.0*0.1 + 2.0*0.2 + 3.0*0.3 = 1.4
        output[1] = 1.0*0.4 + 2.0*0.5 + 3.0*0.6 = 3.2
        output[2] = 1.0*0.7 + 2.0*0.8 + 3.0*0.9 = 5.0
        output[3] = 1.0*1.0 + 2.0*1.1 + 3.0*1.2 = 6.8

    Step 2: Add bias
    -------
    y[0] = 1.4 + 0.1 = 1.5
    y[1] = 3.2 + 0.2 = 3.4
    y[2] = 5.0 + 0.3 = 5.3
    y[3] = 6.8 + 0.4 = 7.2

    Final output (shape: 2x4):
        [[1.5, 3.4, 5.3, 7.2],  ← sequence 1 transformed
         [?, ?, ?, ?]]           ← sequence 2 transformed

    In a transformer, this is used for:
    - QKV: Input (hidden_size) → Query, Key, Value (3 * hidden_size)
    - MLP: Input (hidden_size) → Intermediate (4 * hidden_size)
    - Output: Input (hidden_size) → Vocab (vocab_size)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        param_dtype: torch.dtype,
        param_device: Optional[torch.device],
        module_path: str,
        precision_resolver: "ModulePrecisionResolver",
    ):
        """
        Args:
            in_features: Number of input features (last dim of input)
            out_features: Number of output features (last dim of output)
            bias: Whether to include bias term
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight matrix: (out_features, in_features)
        # Each row is weights for one output neuron
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                dtype=param_dtype,
                device=param_device,
            )
        )

        # Bias vector: (out_features,)
        # One bias value per output neuron
        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    out_features,
                    dtype=param_dtype,
                    device=param_device,
                )
            )
        else:
            self.register_parameter('bias', None)

        # Initialize weights (Xavier/Glorot initialization)
        # This helps prevent vanishing/exploding gradients in deep networks
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        from src.runtime.contracts import LowBitKernelSpec

        self._init_lowbit_precision_state(
            module_path=module_path,
            module_type=self.__class__.__name__,
            precision_resolver=precision_resolver,
            lowbit_capable_type="linear",
            kernel_spec=LowBitKernelSpec(
                module_type="linear",
                in_features=in_features,
                out_features=out_features,
                has_bias=self.bias is not None,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation.

        Formula: output = input @ weight.T + bias
                 (B, O)    (B, I)    (O, I)      (O)

        where B = batch, I = in_features, O = out_features

        Args:
            x: Input tensor of shape (*, in_features)
               The * can be any dimensions (batch, seq, etc.)

        Returns:
            Tensor of shape (*, out_features)
        """
        weight = self._persistent_weight_ste()
        lowbit_output = self._dispatch_lowbit_linear(
            x=x,
            weight=weight,
            bias=self.bias,
        )
        if lowbit_output is not None:
            return lowbit_output

        dtype_override_output = self._linear_with_dtype_override(
            x=x,
            weight=weight,
            bias=self.bias,
        )
        if dtype_override_output is not None:
            return dtype_override_output

        # F.linear is equivalent to x @ weight.T + bias
        # But more numerically stable and optimized
        if self.bias is not None:
            return F.linear(x, weight, self.bias)
        return F.linear(x, weight)


# =============================================================================
# LAYER NORMALIZATION
# =============================================================================

class LayerNorm(_ModulePrecisionStateMixin, nn.Module):
    """
    Layer Normalization: Normalize features to have zero mean, unit variance

    Unlike BatchNorm (normalizes across batch), LayerNorm normalizes
    across features for each sample independently. This is crucial for
    transformers because batch sizes vary and sequences have different lengths.

    Used in:
    - After attention residuals (attn_norm)
    - After MLP residuals (mlp_norm)
    - Final output normalization (ln_f)

    VISUALIZED with concrete example:

    Input: Hidden states for 2 positions, hidden_size = 4
    -------
    x (shape: 2x4):
        [[2.0, 4.0, 6.0, 8.0],   ← position 1
         [1.0, 3.0, 5.0, 7.0]]   ← position 2

    Step 1: Compute mean and variance FOR EACH POSITION
    -------
    Position 1: [2.0, 4.0, 6.0, 8.0]
        mean[1] = (2.0 + 4.0 + 6.0 + 8.0) / 4 = 5.0
        var[1]  = ((2.0-5)^2 + (4.0-5)^2 + (6.0-5)^2 + (8.0-5)^2) / 4 = 5.0

    Position 2: [1.0, 3.0, 5.0, 7.0]
        mean[2] = (1.0 + 3.0 + 5.0 + 7.0) / 4 = 4.0
        var[2]  = ((1.0-4)^2 + (3.0-4)^2 + (5.0-4)^2 + (7.0-4)^2) / 4 = 5.0

    Step 2: Normalize (subtract mean, divide by sqrt(var + eps))
    -------
    For position 1, feature 0 (value 2.0):
        normalized = (2.0 - 5.0) / sqrt(5.0 + 1e-5)
                 = -3.0 / 2.236
                 = -1.342

    For position 1, feature 1 (value 4.0):
        normalized = (4.0 - 5.0) / sqrt(5.0 + 1e-5)
                 = -1.0 / 2.236
                 = -0.447

    Result after normalization:
        [[-1.342, -0.447, 0.447, 1.342],   ← mean=0, std=1
         [-1.342, -0.447, 0.447, 1.342]]

    Step 3: Scale and shift (learnable parameters)
    -------
    These allow the model to "undo" normalization if needed
    output = gamma * normalized + beta

    If gamma = [1.0, 1.0, 1.0, 1.0], beta = [0.0, 0.0, 0.0, 0.0]
        Output = normalized (identity transformation)

    If gamma = [0.5, 0.5, 0.5, 0.5], beta = [1.0, 1.0, 1.0, 1.0]
        Output = 0.5 * normalized + 1.0 (half variance, shifted up)

    Why this matters in transformers:
    -------
    - Gradient flow: Prevents activations from exploding/vanishing
    - Stability: Each layer sees inputs with consistent scale
    - Training speed: Allows higher learning rates
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        *,
        param_dtype: torch.dtype,
        param_device: Optional[torch.device],
        module_path: str,
        precision_resolver: "ModulePrecisionResolver",
    ):
        """
        Args:
            normalized_shape: Number of features to normalize (usually hidden_size)
            eps: Small constant for numerical stability (prevents div by zero)
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Learnable scale (gamma): Initialize to 1.0 (identity)
        self.weight = nn.Parameter(
            torch.ones(
                normalized_shape,
                dtype=param_dtype,
                device=param_device,
            )
        )

        # Learnable shift (beta): Initialize to 0.0 (identity)
        self.bias = nn.Parameter(
            torch.zeros(
                normalized_shape,
                dtype=param_dtype,
                device=param_device,
            )
        )
        self._init_module_precision_state(
            module_path=module_path,
            module_type=self.__class__.__name__,
            precision_resolver=precision_resolver,
            lowbit_capable_type=None,
            kernel_spec=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.

        Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta

        where mean and var are computed over the last dimension (features)
        for each sample independently.

        Args:
            x: Input tensor of shape (*, normalized_shape)

        Returns:
            Normalized tensor of same shape
        """
        compute_dtype = self._module_compute_dtype_override()
        x_compute = x.to(dtype=compute_dtype) if compute_dtype is not None else x
        weight = self.weight.to(dtype=compute_dtype) if compute_dtype is not None else self.weight
        bias = self.bias.to(dtype=compute_dtype) if compute_dtype is not None else self.bias

        # Compute mean over last dimension
        # Shape: (*) - one value per sample
        mean = x_compute.mean(dim=-1, keepdim=True)

        # Compute variance over last dimension
        # Shape: (*) - one value per sample
        var = x_compute.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize: (x - mean) / sqrt(var + eps)
        # This centers at 0 and scales to unit variance
        x_normalized = (x_compute - mean) / torch.sqrt(var + self.eps)

        # Scale and shift with learnable parameters
        return weight * x_normalized + bias


# =============================================================================
# EMBEDDING LAYER
# =============================================================================

class Embedding(_ModulePrecisionStateMixin, nn.Module):
    """
    Embedding layer: Map discrete token IDs to continuous vectors

    This is how discrete tokens (characters, words, subwords) become
    continuous representations that neural networks can process.

    Used in:
    - Token embeddings: Convert token IDs to vectors
    - Position embeddings: Convert position indices to vectors

    VISUALIZED with concrete example:

    Input: Token IDs for "cat sat"
    -------
    vocab_size = 10, hidden_size = 4

    token_ids (shape: 3):
        [2, 5, 3]
         ↑  ↑  ↑
        cat sat mat

    Embedding matrix (shape: 10x4):
        Each row is the vector for one token:
        token 0: [0.1, 0.2, 0.3, 0.4]
        token 1: [0.5, 0.6, 0.7, 0.8]
        token 2: [0.9, 1.0, 1.1, 1.2]  ← "cat"
        token 3: [1.3, 1.4, 1.5, 1.6]  ← "mat"
        token 4: [1.7, 1.8, 1.9, 2.0]
        token 5: [2.1, 2.2, 2.3, 2.4]  ← "sat"
        ...

    Step 1: Look up vectors for each token ID
    -------
    token_ids[0] = 2 → lookup row 2 → [0.9, 1.0, 1.1, 1.2]
    token_ids[1] = 5 → lookup row 5 → [2.1, 2.2, 2.3, 2.4]
    token_ids[2] = 3 → lookup row 3 → [1.3, 1.4, 1.5, 1.6]

    Step 2: Stack into batch
    -------
    Output embeddings (shape: 3x4):
        [[0.9, 1.0, 1.1, 1.2],  ← "cat" as vector
         [2.1, 2.2, 2.3, 2.4],  ← "sat" as vector
         [1.3, 1.4, 1.5, 1.6]]  ← "mat" as vector

    These vectors capture semantic meaning:
    - Similar tokens have similar vectors (learned during training)
    - Dimensions might represent: tense, plurality, part-of-speech, etc.
    - The model learns these representations automatically!

    Why embeddings matter:
    -------
    - Discrete tokens → continuous representations
    - Capture semantic relationships (king - man + woman ≈ queen)
    - Enable gradient-based learning (can't backprop through discrete IDs)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        param_dtype: torch.dtype,
        param_device: Optional[torch.device],
        module_path: str,
        precision_resolver: "ModulePrecisionResolver",
    ):
        """
        Args:
            num_embeddings: Size of the dictionary (vocab_size)
            embedding_dim: Size of each embedding vector (hidden_size)
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Embedding matrix: (num_embeddings, embedding_dim)
        # Each row is the vector for one token ID
        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                dtype=param_dtype,
                device=param_device,
            )
        )

        # Initialize with normal distribution (std=0.02)
        # This is common for transformers (similar to GPT-2)
        nn.init.normal_(self.weight, std=0.02)
        self._init_module_precision_state(
            module_path=module_path,
            module_type=self.__class__.__name__,
            precision_resolver=precision_resolver,
            lowbit_capable_type=None,
            kernel_spec=None,
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for token IDs.

        Args:
            indices: Token IDs of shape (*)
                     Can be any shape (batch), (batch, seq), etc.

        Returns:
            Embeddings of shape (*, embedding_dim)
        """
        # F.embedding performs the lookup efficiently
        # It's like fancy indexing: for each ID, get that row
        # Note: F.embedding has signature (input, weight, ...)
        return F.embedding(indices, self.weight)


# =============================================================================
# DROPOUT LAYER
# =============================================================================

class Dropout(nn.Module):
    """
    Dropout: Randomly zero activations during training to prevent overfitting

    During training: Randomly set some activations to zero
    During inference: Return input unchanged (identity)

    This forces the network to learn redundant representations and
    prevents co-adaptation of neurons.

    Used in:
    - After attention (attention dropout)
    - After MLP activation (MLP dropout)
    - Embedding dropout (randomly drop token embeddings)

    VISUALIZED with concrete example:

    Input: Activations from a layer
    -------
    x (shape: 2x4):
        [[0.5, 0.8, 0.3, 0.9],
         [0.2, 0.7, 0.4, 0.6]]

    Dropout rate: p = 0.5 (50% chance to drop each activation)

    Step 1: Generate random binary mask
    -------
    For each element, sample from Bernoulli(p_keep)
    where p_keep = 1 - p = 0.5

    mask (shape: 2x4) - random during training:
        [[1, 0, 1, 0],  ← 50% zeros on average
         [0, 1, 0, 1]]

    Step 2: Apply mask and scale
    -------
    During training:
        output = x * mask / (1 - p)
                = x * mask / 0.5  ← Scale to preserve expected value

        output[0, 0] = 0.5 * 1 / 0.5 = 1.0  ← kept, scaled up
        output[0, 1] = 0.8 * 0 / 0.5 = 0.0  ← dropped
        output[0, 2] = 0.3 * 1 / 0.5 = 0.6  ← kept, scaled up
        output[0, 3] = 0.9 * 0 / 0.5 = 0.0  ← dropped

    Result:
        [[1.0, 0.0, 0.6, 0.0],
         [0.0, 1.4, 0.0, 1.2]]

    During inference:
        output = x  ← No dropout, no scaling

    Why scale by 1/(1-p)?
    -------
    Expected value during training:
        E[output] = E[x * mask / (1-p)]
                   = x / (1-p) * E[mask]
                   = x / (1-p) * (1-p)
                   = x

    So the expected output is the same as input! This preserves
    the scale of activations, so we don't need to adjust learning rate
    or other hyperparameters when changing dropout rate.

    Why dropout works:
    -------
    - Prevents overfitting: Can't rely on any specific neurons
    - Forces robustness: Each neuron must be useful independently
    - Ensemble effect: Different dropout masks ≈ training many sub-networks
    """

    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of dropping each activation (0.0 to 1.0)
               Common values: 0.1 for embeddings, 0.3-0.5 for layers
        """
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability must be in [0, 1], got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout during training.

        Args:
            x: Input tensor of any shape

        Returns:
            Tensor with dropout applied (same shape as input)
        """
        if self.p == 0:
            return x  # No dropout, return as-is

        if not self.training:
            return x  # Inference mode: no dropout

        # Generate random mask: 1 with prob (1-p), 0 with prob p
        keep_prob = 1.0 - self.p
        mask = torch.empty_like(x).bernoulli_(keep_prob)

        # Apply mask and scale to preserve expected value
        return x * mask / keep_prob


# =============================================================================
# GELU ACTIVATION
# =============================================================================

class GELU(nn.Module):
    """
    Gaussian Error Linear Unit: Smooth, non-linear activation function

    GELU is smoother than ReLU and performs better in transformers.
    It's like a "soft" ReLU with a curved transition region.

    Formula: GELU(x) = x * Φ(x) where Φ is the standard normal CDF

    Used in:
    - MLP activation (after fc1, before fc2)
    - Modern transformers (GPT-2, BERT, etc.) use GELU instead of ReLU

    VISUALIZED with concrete example:

    Input: Activations from linear layer
    -------
    x (shape: 5):
        [-3.0, -1.0, 0.0, 1.0, 3.0]

    Step 1: Compute CDF of standard normal Φ(x)
    -------
    Φ(x) ≈ 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))

    For each value:
    x = -3.0: Φ(-3.0) ≈ 0.0013  (0.13% - almost zero)
    x = -1.0: Φ(-1.0) ≈ 0.1587  (15.87% - small)
    x =  0.0: Φ(0.0)  ≈ 0.5    (50% - half)
    x =  1.0: Φ(1.0)  ≈ 0.8413  (84.13% - large)
    x =  3.0: Φ(3.0)  ≈ 0.9987  (99.87% - almost one)

    Step 2: Multiply x by Φ(x)
    -------
    output = x * Φ(x)

    x = -3.0: output = -3.0 * 0.0013 ≈ -0.004  ← Almost zero (negative)
    x = -1.0: output = -1.0 * 0.1587 ≈ -0.159  ← Small negative
    x =  0.0: output =  0.0 * 0.5    =  0.0    ← Exactly zero
    x =  1.0: output =  1.0 * 0.8413 ≈  0.841  ← Small positive
    x =  3.0: output =  3.0 * 0.9987 ≈  2.996  ← Almost unchanged

    Result:
        [-0.004, -0.159, 0.0, 0.841, 2.996]

    Comparison to ReLU:
    -------
    ReLU(x) = max(0, x):
        [-3.0, -1.0, 0.0, 1.0, 3.0] → [0, 0, 0, 1, 3]

    GELU is smoother:
        - No sharp "knee" at zero
        - Allows some negative values to pass through
        - Differentiable everywhere (ReLU has non-differentiable point at 0)

    Why GELU works better:
    -------
    - Smooth gradient: No discontinuities, easier to optimize
    - Gating behavior: Φ(x) acts like a learned gate
    - Empirical results: Outperforms ReLU in transformers
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GELU activation.

        We use the same definition as `torch.nn.GELU` (exact formulation) so that unit tests and
        reference comparisons match closely. The commonly used tanh-approximation is a valid
        alternative for speed, but it introduces small numerical differences.

        Args:
            x: Input tensor of any shape

        Returns:
            Activated tensor of same shape
        """
        return F.gelu(x)


# =============================================================================
# GRADIENT CLIPPING
# =============================================================================

def clip_grad_norm(parameters, max_norm: float):
    """
    Clip gradient norms to prevent exploding gradients.

    During training, gradients can become very large, causing:
    - Parameter updates to be too big
    - Loss to become NaN or Infinity
    - Training to diverge

    Clipping scales gradients if their norm exceeds max_norm.

    VISUALIZED with concrete example:

    Gradients from backward pass:
    -------
    3 parameters with gradients:
        grad[0] = 3.0  (weight 1)
        grad[1] = 4.0  (weight 2)
        grad[2] = 0.0  (weight 3)

    Step 1: Compute global gradient norm
    -------
    norm = √(3.0^2 + 4.0^2 + 0.0^2)
         = √(9.0 + 16.0 + 0.0)
         = √25.0
         = 5.0

    Step 2: Check if norm exceeds max_norm
    -------
    If max_norm = 1.0:
        norm (5.0) > max_norm (1.0)  → Need to clip!

    Step 3: Compute clipping coefficient
    -------
    clip_coef = max_norm / (norm + small epsilon)
              = 1.0 / 5.0
              = 0.2

    Step 4: Scale all gradients
    -------
    grad_clipped[i] = grad[i] * clip_coef

    grad_clipped[0] = 3.0 * 0.2 = 0.6
    grad_clipped[1] = 4.0 * 0.2 = 0.8
    grad_clipped[2] = 0.0 * 0.2 = 0.0

    Verify new norm:
        √(0.6^2 + 0.8^2 + 0.0^2)
        = √(0.36 + 0.64 + 0.0)
        = √1.0
        = 1.0  ✓ Now equals max_norm!

    If norm was already small:
    -------
    If grad = [0.3, 0.4, 0.0] (norm = 0.5):
        norm (0.5) < max_norm (1.0)  → No clipping!
        clip_coef = 1.0 (identity)

    Why clip at 1.0:
    -------
    - Transformers typically use max_norm = 1.0
    - Prevents gradient explosion in deep networks
    - Stabilizes training with large learning rates
    - Common in GPT-style models

    Args:
        parameters: Iterable of parameters with gradients
        max_norm: Maximum allowed gradient norm

    Returns:
        Total norm before clipping (for logging)
    """
    # Compute gradient norm (L2 norm)
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    # Clip if necessary
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)

    return total_norm

# =============================================================================
# TENSOR PARALLELISM LAYERS
# =============================================================================

class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    """
    TP reduce op used by row-parallel layers.

    Forward: all-reduce sum to materialize full activations on every TP rank.
    Backward: identity so each rank keeps its local gradient shard without
    introducing an extra cross-rank sum.
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor, tp_group):
        output = input_.clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=tp_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class _CopyToTensorParallelRegion(torch.autograd.Function):
    """
    TP copy op used by column-parallel layers.

    Forward: identity because input activations are replicated across TP ranks.
    Backward: all-reduce sum so upstream replicated layers receive full dX.
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor, tp_group):
        ctx.tp_group = tp_group
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = grad_output.clone()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=ctx.tp_group)
        return grad_input, None


class ColumnParallelLinear(_LowBitPrecisionLinearMixin, nn.Module):
    """
    Column-parallel linear (shard output features).

    Weight shape is sharded on dim 0:
        global W: [out_features, in_features]
        local  W_i: [out_features / tp_size, in_features]

    Communication pattern:
    - Forward: no TP collective. Each rank produces a local output shard.
    - Backward: TP all-reduce(sum) on dX via _CopyToTensorParallelRegion.
      This is the canonical column-parallel rule.

    Parameter gradients for local shards are computed locally. In TP+DP setups,
    cross-replica gradient averaging is handled by DP groups outside this module.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_rank: int,
        tp_size: int,
        bias: bool = True,
        tp_group=None,
        *,
        param_dtype: torch.dtype,
        param_device: Optional[torch.device],
        module_path: str,
        precision_resolver: "ModulePrecisionResolver",
    ):
        """
        Args:
            in_features: Number of input features
            out_features: Number of output features (must be divisible by tp_size)
            tp_rank: Tensor parallel rank of this process (0 to tp_size-1)
            tp_size: Total number of tensor parallel processes
            bias: Whether to include bias term
            tp_group: TP process group for collectives (defaults to WORLD)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = tp_group if tp_group is not None else dist.group.WORLD

        # Each GPU stores out_features // tp_size outputs
        if out_features % tp_size != 0:
            raise ValueError(
                f"out_features {out_features} must be divisible by tp_size {tp_size}"
            )
        self.shard_size = out_features // tp_size

        # Weight shard: (shard_size, in_features)
        # F.linear expects (out_features, in_features), so this is correct
        self.weight = nn.Parameter(
            torch.empty(
                self.shard_size,
                in_features,
                dtype=param_dtype,
                device=param_device,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    self.shard_size,
                    dtype=param_dtype,
                    device=param_device,
                )
            )
        else:
            self.register_parameter('bias', None)

        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        from src.runtime.contracts import LowBitKernelSpec

        self._init_lowbit_precision_state(
            module_path=module_path,
            module_type=self.__class__.__name__,
            precision_resolver=precision_resolver,
            lowbit_capable_type="column_parallel_linear",
            kernel_spec=LowBitKernelSpec(
                module_type="column_parallel_linear",
                in_features=in_features,
                out_features=self.shard_size,
                has_bias=self.bias is not None,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply column parallel linear transformation.

        Args:
            x: Input tensor of shape (*, in_features) - same on all GPUs

        Returns:
            Tensor of shape (*, shard_size) - each GPU's output shard
        """
        if self.tp_size > 1:
            x = _CopyToTensorParallelRegion.apply(x, self.tp_group)

        weight = self._persistent_weight_ste()
        lowbit_output = self._dispatch_lowbit_linear(
            x=x,
            weight=weight,
            bias=self.bias,
        )
        if lowbit_output is not None:
            return lowbit_output

        dtype_override_output = self._linear_with_dtype_override(
            x=x,
            weight=weight,
            bias=self.bias,
        )
        if dtype_override_output is not None:
            return dtype_override_output

        if self.bias is not None:
            return F.linear(x, weight, self.bias)
        return F.linear(x, weight)


class RowParallelLinear(_LowBitPrecisionLinearMixin, nn.Module):
    """
    Row-parallel linear (shard input features).

    Weight shape is sharded on dim 1:
        global W: [out_features, in_features]
        local  W_i: [out_features, in_features / tp_size]

    Communication pattern:
    - Forward: TP all-reduce(sum) on partial outputs to materialize full Y on
      each TP rank.
    - Backward: no extra TP collective from this autograd edge. Each rank keeps
      its local dX shard, and local shard parameter gradients are computed
      locally.

    In TP+DP setups, parameter gradients are synchronized across DP groups
    outside this module.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_rank: int,
        tp_size: int,
        bias: bool = True,
        tp_group=None,
        *,
        param_dtype: torch.dtype,
        param_device: Optional[torch.device],
        module_path: str,
        precision_resolver: "ModulePrecisionResolver",
    ):
        """
        Args:
            in_features: Number of input features (must be divisible by tp_size)
            out_features: Number of output features
            tp_rank: Tensor parallel rank of this process (0 to tp_size-1)
            tp_size: Total number of tensor parallel processes
            bias: Whether to include bias term
            tp_group: Process group for all-reduce (defaults to WORLD)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = tp_group if tp_group is not None else dist.group.WORLD

        # Each GPU processes in_features // tp_size inputs
        if in_features % tp_size != 0:
            raise ValueError(
                f"in_features {in_features} must be divisible by tp_size {tp_size}"
            )
        self.shard_size = in_features // tp_size

        # Weight shard: (out_features, shard_size)
        # F.linear expects (out_features, in_features), so this is correct
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                self.shard_size,
                dtype=param_dtype,
                device=param_device,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    out_features,
                    dtype=param_dtype,
                    device=param_device,
                )
            )
        else:
            self.register_parameter('bias', None)

        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        from src.runtime.contracts import LowBitKernelSpec

        self._init_lowbit_precision_state(
            module_path=module_path,
            module_type=self.__class__.__name__,
            precision_resolver=precision_resolver,
            lowbit_capable_type="row_parallel_linear",
            kernel_spec=LowBitKernelSpec(
                module_type="row_parallel_linear",
                in_features=self.shard_size,
                out_features=out_features,
                has_bias=self.bias is not None,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply row parallel linear transformation with all-reduce.

        Args:
            x: Input tensor of shape (*, shard_size) - sharded from column parallel

        Returns:
            Tensor of shape (*, out_features) - full output on all GPUs
        """
        weight = self._persistent_weight_ste()
        lowbit_output = self._dispatch_lowbit_linear(
            x=x,
            weight=weight,
            bias=None,
        )
        if lowbit_output is None:
            # Compute partial output from local input/weight shards.
            dtype_override_output = self._linear_with_dtype_override(
                x=x,
                weight=weight,
                bias=None,
            )
            if dtype_override_output is not None:
                partial_output = dtype_override_output
            else:
                partial_output = F.linear(x, weight, None)
        else:
            partial_output = lowbit_output

        # Sum partial outputs across TP ranks.
        if self.tp_size > 1:
            partial_output = _ReduceFromTensorParallelRegion.apply(partial_output, self.tp_group)

        # Bias is replicated, so it must be added once after reduction.
        if self.bias is not None:
            partial_output = partial_output + self.bias

        return partial_output


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Autograd-safe sequence scatter across TP ranks."""

    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        tp_rank: int,
        tp_size: int,
        tp_group,
        seq_dim: int,
    ) -> torch.Tensor:
        ctx.tp_rank = tp_rank
        ctx.tp_size = tp_size
        ctx.tp_group = tp_group
        ctx.seq_dim = seq_dim

        if tp_size == 1:
            return input_

        seq_len = input_.size(seq_dim)
        if seq_len % tp_size != 0:
            raise ValueError("sequence length must be divisible by tp_size for sequence scatter")

        chunk = seq_len // tp_size
        start = tp_rank * chunk
        return input_.narrow(seq_dim, start, chunk).contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.tp_size == 1:
            return grad_output, None, None, None, None

        gather_list = [torch.empty_like(grad_output) for _ in range(ctx.tp_size)]
        dist.all_gather(gather_list, grad_output.contiguous(), group=ctx.tp_group)
        grad_input = torch.cat(gather_list, dim=ctx.seq_dim).contiguous()
        return grad_input, None, None, None, None


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Autograd-safe sequence gather across TP ranks."""

    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        tp_rank: int,
        tp_size: int,
        tp_group,
        seq_dim: int,
    ) -> torch.Tensor:
        ctx.tp_rank = tp_rank
        ctx.tp_size = tp_size
        ctx.seq_dim = seq_dim

        if tp_size == 1:
            return input_

        gather_list = [torch.empty_like(input_) for _ in range(tp_size)]
        dist.all_gather(gather_list, input_.contiguous(), group=tp_group)
        return torch.cat(gather_list, dim=seq_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.tp_size == 1:
            return grad_output, None, None, None, None

        seq_len = grad_output.size(ctx.seq_dim)
        if seq_len % ctx.tp_size != 0:
            raise ValueError("sequence length must be divisible by tp_size for sequence gather")

        chunk = seq_len // ctx.tp_size
        start = ctx.tp_rank * chunk
        grad_input = grad_output.narrow(ctx.seq_dim, start, chunk).contiguous()
        return grad_input, None, None, None, None


def scatter_to_sequence_parallel_region(
    input_: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    tp_group=None,
    seq_dim: int = 1,
) -> torch.Tensor:
    """Scatter sequence dimension across TP ranks and keep autograd correctness."""
    group = tp_group if tp_group is not None else dist.group.WORLD
    return _ScatterToSequenceParallelRegion.apply(input_, tp_rank, tp_size, group, seq_dim)


def gather_from_sequence_parallel_region(
    input_: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    tp_group=None,
    seq_dim: int = 1,
) -> torch.Tensor:
    """Gather sequence shards from TP ranks and keep autograd correctness."""
    group = tp_group if tp_group is not None else dist.group.WORLD
    return _GatherFromSequenceParallelRegion.apply(input_, tp_rank, tp_size, group, seq_dim)
