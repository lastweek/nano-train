"""
DeepSeek-V3-like decoder model with MLA-style attention and routed MoE.

The implementation keeps core DeepSeek-V3 architectural ideas and key
configuration fields, but remains a compact approximation for local training:
- MLA-style low-rank query/KV projections with split NoPE/RoPE QK dimensions
- Dense-first then MoE decoder blocks
- Routed experts with configurable top-k behavior and optional group routing
- Optional parallel context for TP dense FFN and EP routed MoE
"""

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import Dict, Optional, Tuple
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import ColumnParallelLinear
from src.layers import Dropout
from src.layers import Embedding
from src.layers import gather_from_sequence_parallel_region
from src.layers import Linear
from src.layers import RowParallelLinear
from src.layers import scatter_to_sequence_parallel_region
from src.models.moe import ExpertMLP
from src.models.moe import ExpertParallelMoE
from src.models.moe import LocalRoutedMoE

if TYPE_CHECKING:
    from src.runtime.contracts import DeepSeekV3PrecisionRecipe
    from src.runtime.contracts import ModulePrecisionInitState
    from src.runtime.contracts import ModulePrecisionResolver
    from src.runtime.contracts import ModulePrecisionSummary
    from src.runtime.contracts import PrecisionDType
else:
    DeepSeekV3PrecisionRecipe = object
    ModulePrecisionInitState = object
    ModulePrecisionResolver = object
    ModulePrecisionSummary = object
    PrecisionDType = str


@dataclass
class DeepSeekModelConfig:
    """Configuration shaped after DeepSeek-V3 key parameters."""
    param_dtype: torch.dtype
    param_device: Optional[torch.device]
    precision_resolver: ModulePrecisionResolver
    module_compute_dtype_overrides: dict[str, "PrecisionDType"] = field(default_factory=dict)

    # Core architecture
    vocab_size: int = 129280
    hidden_size: int = 7168
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    num_key_value_heads: int = 128
    intermediate_size: int = 18432
    max_position_embeddings: int = 163840
    rms_norm_eps: float = 1e-6

    # MLA-style attention
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope_theta: float = 10000.0
    rope_scaling: Dict[str, float] = field(
        default_factory=lambda: {
            "type": "yarn",
            "factor": 40.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096.0,
        }
    )
    attention_dropout: float = 0.0
    dropout: float = 0.0

    # Routed MoE
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 3
    moe_layer_freq: int = 1
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    n_group: int = 8
    topk_group: int = 4
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.5

    # Output head
    tie_word_embeddings: bool = False

    # Backward-compat aliases (old local names)
    num_layers: Optional[int] = None
    num_kv_heads: Optional[int] = None
    num_experts: Optional[int] = None
    num_shared_experts: Optional[int] = None
    expert_intermediate_size: Optional[int] = None
    top_k: Optional[int] = None

    def __post_init__(self) -> None:
        """Map legacy field names and validate key constraints."""
        if self.num_layers is not None:
            self.num_hidden_layers = self.num_layers
        if self.num_kv_heads is not None:
            self.num_key_value_heads = self.num_kv_heads
        if self.num_experts is not None:
            self.n_routed_experts = self.num_experts
        if self.num_shared_experts is not None:
            self.n_shared_experts = self.num_shared_experts
        if self.expert_intermediate_size is not None:
            self.moe_intermediate_size = self.expert_intermediate_size
        if self.top_k is not None:
            self.num_experts_per_tok = self.top_k

        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.n_routed_experts <= 0:
            raise ValueError("n_routed_experts must be positive")
        if self.num_experts_per_tok <= 0:
            raise ValueError("num_experts_per_tok must be positive")
        if self.num_experts_per_tok > self.n_routed_experts:
            raise ValueError("num_experts_per_tok cannot exceed n_routed_experts")
        if self.qk_rope_head_dim <= 0 or self.qk_rope_head_dim % 2 != 0:
            raise ValueError("qk_rope_head_dim must be a positive even number")
        if self.n_group <= 0 or self.n_routed_experts % self.n_group != 0:
            raise ValueError("n_group must divide n_routed_experts")
        if self.topk_group <= 0 or self.topk_group > self.n_group:
            raise ValueError("topk_group must be in [1, n_group]")
        experts_per_group = self.n_routed_experts // self.n_group
        if self.topk_group * experts_per_group < self.num_experts_per_tok:
            raise ValueError("topk_group selection exposes fewer experts than num_experts_per_tok")
        if self.first_k_dense_replace < 0:
            raise ValueError("first_k_dense_replace must be non-negative")
        if self.scoring_func not in {"sigmoid", "softmax"}:
            raise ValueError("scoring_func must be 'sigmoid' or 'softmax'")
        valid_compute_dtypes = {"fp32", "bf16", "fp16"}
        for module_name, dtype_alias in self.module_compute_dtype_overrides.items():
            if not module_name:
                raise ValueError("module_compute_dtype_overrides keys must be non-empty")
            if dtype_alias not in valid_compute_dtypes:
                raise ValueError(
                    "module_compute_dtype_overrides values must be one of fp32, bf16, fp16"
                )


class _DeepSeekConfigPrecisionResolver:
    """Resolver wrapper that applies exact module-path dtype overrides from config."""

    def __init__(
        self,
        base: ModulePrecisionResolver,
        module_compute_dtype_overrides: dict[str, "PrecisionDType"],
    ) -> None:
        self._base = base
        self._overrides = dict(module_compute_dtype_overrides)
        self._matched: set[str] = set()

    def resolve_module_init_state(
        self,
        *,
        module_path: str,
        module_type: str,
        lowbit_capable_type,
        kernel_spec=None,
    ) -> "ModulePrecisionInitState":
        state = self._base.resolve_module_init_state(
            module_path=module_path,
            module_type=module_type,
            lowbit_capable_type=lowbit_capable_type,
            kernel_spec=kernel_spec,
        )
        dtype_override = self._overrides.get(module_path)
        if dtype_override is None:
            return state

        self._matched.add(module_path)
        assignment = replace(
            state.assignment,
            compute_lowbit_mode=None,
            compute_dtype_override=dtype_override,
        )
        return replace(state, assignment=assignment, lowbit_backend=None)

    def finalize(self) -> "ModulePrecisionSummary":
        summary = self._base.finalize()
        missing = [name for name in self._overrides if name not in self._matched]
        if missing:
            raise ValueError(
                "DeepSeek config module_compute_dtype_overrides matched zero modules: "
                f"{missing[:5]}"
            )
        return summary

    def deepseek_v3_recipe(self) -> Optional["DeepSeekV3PrecisionRecipe"]:
        recipe_fn = getattr(self._base, "deepseek_v3_recipe", None)
        if not callable(recipe_fn):
            return None
        return recipe_fn()


@dataclass
class DeepSeekParallelContext:
    """
    Optional model parallel context using Megatron-style parallel naming.

    Defaults keep behavior identical to the single-process baseline.
    """

    tensor_model_parallel_rank: int = 0
    tensor_model_parallel_size: int = 1
    tensor_model_parallel_group: Optional[object] = None
    expert_model_parallel_rank: int = 0
    expert_model_parallel_size: int = 1
    expert_model_parallel_group: Optional[object] = None
    pipeline_model_parallel_rank: int = 0
    pipeline_model_parallel_size: int = 1
    pipeline_model_parallel_group: Optional[object] = None
    context_parallel_rank: int = 0
    context_parallel_size: int = 1
    context_parallel_group: Optional[object] = None
    pp_layer_splits: Optional[tuple[int, ...]] = None
    capacity_factor: float = 1.0
    expert_tensor_parallel_size: int = 1
    sequence_parallel: bool = True

    def __post_init__(self) -> None:
        if self.tensor_model_parallel_size <= 0:
            raise ValueError("tensor_model_parallel_size must be positive")
        if self.expert_model_parallel_size <= 0:
            raise ValueError("expert_model_parallel_size must be positive")
        if self.pipeline_model_parallel_size <= 0:
            raise ValueError("pipeline_model_parallel_size must be positive")
        if self.context_parallel_size <= 0:
            raise ValueError("context_parallel_size must be positive")

        if (
            self.tensor_model_parallel_rank < 0
            or self.tensor_model_parallel_rank >= self.tensor_model_parallel_size
        ):
            raise ValueError(
                "tensor_model_parallel_rank must be in [0, tensor_model_parallel_size)"
            )
        if (
            self.expert_model_parallel_rank < 0
            or self.expert_model_parallel_rank >= self.expert_model_parallel_size
        ):
            raise ValueError(
                "expert_model_parallel_rank must be in [0, expert_model_parallel_size)"
            )
        if (
            self.pipeline_model_parallel_rank < 0
            or self.pipeline_model_parallel_rank >= self.pipeline_model_parallel_size
        ):
            raise ValueError(
                "pipeline_model_parallel_rank must be in [0, pipeline_model_parallel_size)"
            )
        if self.context_parallel_rank < 0 or self.context_parallel_rank >= self.context_parallel_size:
            raise ValueError("context_parallel_rank must be in [0, context_parallel_size)")

        if self.pp_layer_splits is not None:
            if len(self.pp_layer_splits) != self.pipeline_model_parallel_size + 1:
                raise ValueError("pp_layer_splits length must be pipeline_model_parallel_size + 1")
            if self.pp_layer_splits[0] != 0:
                raise ValueError("pp_layer_splits must start with 0")
            for idx in range(1, len(self.pp_layer_splits)):
                if self.pp_layer_splits[idx] <= self.pp_layer_splits[idx - 1]:
                    raise ValueError("pp_layer_splits must be strictly increasing")
        if self.capacity_factor <= 0:
            raise ValueError("capacity_factor must be positive")
        if self.expert_tensor_parallel_size != 1:
            raise ValueError(
                "DeepSeek routed experts currently require expert_tensor_parallel_size=1"
            )

    def resolve_attention_tensor_model_parallel(self) -> Tuple[int, int, Optional[object]]:
        """Resolve attention sharding to tensor model parallel domain."""
        return (
            self.tensor_model_parallel_rank,
            self.tensor_model_parallel_size,
            self.tensor_model_parallel_group,
        )

    def resolve_expert_model_parallel(self) -> Tuple[int, int, Optional[object]]:
        """Resolve routed-expert sharding to expert model parallel domain."""
        return (
            self.expert_model_parallel_rank,
            self.expert_model_parallel_size,
            self.expert_model_parallel_group,
        )

    # Backward-compat aliases (one-release transition)
    def resolve_attn_tp(self) -> Tuple[int, int, Optional[object]]:
        return self.resolve_attention_tensor_model_parallel()

    def resolve_moe_ep(self) -> Tuple[int, int, Optional[object]]:
        return self.resolve_expert_model_parallel()

    def resolve_pp_layer_boundaries(self, num_hidden_layers: int) -> tuple[int, ...]:
        """Resolve global layer boundaries for PP stage partitioning."""
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")

        if self.pipeline_model_parallel_size == 1:
            return (0, num_hidden_layers)

        if self.pp_layer_splits is not None:
            if self.pp_layer_splits[-1] != num_hidden_layers:
                raise ValueError("pp_layer_splits last value must equal num_hidden_layers")
            return self.pp_layer_splits

        base = num_hidden_layers // self.pipeline_model_parallel_size
        remainder = num_hidden_layers % self.pipeline_model_parallel_size
        boundaries = [0]
        cursor = 0
        for stage in range(self.pipeline_model_parallel_size):
            stage_size = base + (1 if stage < remainder else 0)
            cursor += stage_size
            boundaries.append(cursor)
        return tuple(boundaries)

    def resolve_pp_layer_range(self, num_hidden_layers: int) -> tuple[int, int]:
        """Return local `[start, end)` layer range for this PP rank."""
        boundaries = self.resolve_pp_layer_boundaries(num_hidden_layers)
        rank = self.pipeline_model_parallel_rank
        return boundaries[rank], boundaries[rank + 1]

    # ------------------------------------------------------------------
    # Backward-compat aliases (one-release transition)
    # ------------------------------------------------------------------
    @property
    def tp_rank(self) -> int:
        return self.tensor_model_parallel_rank

    @property
    def tp_size(self) -> int:
        return self.tensor_model_parallel_size

    @property
    def tp_group(self):
        return self.tensor_model_parallel_group

    @property
    def ep_rank(self) -> int:
        return self.expert_model_parallel_rank

    @property
    def ep_size(self) -> int:
        return self.expert_model_parallel_size

    @property
    def ep_group(self):
        return self.expert_model_parallel_group

    @property
    def pp_rank(self) -> int:
        return self.pipeline_model_parallel_rank

    @property
    def pp_size(self) -> int:
        return self.pipeline_model_parallel_size

    @property
    def pp_group(self):
        return self.pipeline_model_parallel_group

    @property
    def expert_tp_size(self) -> int:
        return self.expert_tensor_parallel_size

    @property
    def enable_moe_sequence_parallel(self) -> bool:
        return self.sequence_parallel


class RMSNorm(nn.Module):
    """RMSNorm used by DeepSeek-style decoder blocks."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        param_dtype: torch.dtype,
        param_device: Optional[torch.device],
        module_path: str,
        precision_resolver: ModulePrecisionResolver,
    ):
        super().__init__()
        self.eps = eps
        self._module_precision_state = precision_resolver.resolve_module_init_state(
            module_path=module_path,
            module_type=self.__class__.__name__,
            lowbit_capable_type=None,
            kernel_spec=None,
        )
        self.weight = nn.Parameter(
            torch.ones(
                hidden_size,
                dtype=param_dtype,
                device=param_device,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assignment = self._module_precision_state.assignment
        dtype_alias = getattr(assignment, "compute_dtype_override", None)
        if dtype_alias is None:
            x_compute = x
            weight = self.weight
        else:
            dtype_map = {
                "fp32": torch.float32,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }
            try:
                compute_dtype = dtype_map[str(dtype_alias)]
            except KeyError as exc:
                raise ValueError(
                    f"Unsupported RMSNorm compute dtype override alias: {dtype_alias}"
                ) from exc
            x_compute = x.to(dtype=compute_dtype)
            weight = self.weight.to(dtype=compute_dtype)

        var = x_compute.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_compute * torch.rsqrt(var + self.eps)
        return x_norm * weight


class GatedMLP(nn.Module):
    """SwiGLU-style feed-forward block."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float,
        param_dtype: torch.dtype,
        param_device: Optional[torch.device],
        precision_resolver: ModulePrecisionResolver,
        module_prefix: str,
        parallel_context: Optional[DeepSeekParallelContext] = None,
    ):
        super().__init__()
        self.parallel_context = parallel_context or DeepSeekParallelContext()
        self.use_tp = self.parallel_context.tensor_model_parallel_size > 1

        if self.use_tp:
            self.gate_proj = ColumnParallelLinear(
                hidden_size,
                intermediate_size,
                tp_rank=self.parallel_context.tensor_model_parallel_rank,
                tp_size=self.parallel_context.tensor_model_parallel_size,
                tp_group=self.parallel_context.tensor_model_parallel_group,
                bias=True,
                param_dtype=param_dtype,
                param_device=param_device,
                module_path=f"{module_prefix}.gate_proj",
                precision_resolver=precision_resolver,
            )
            self.up_proj = ColumnParallelLinear(
                hidden_size,
                intermediate_size,
                tp_rank=self.parallel_context.tensor_model_parallel_rank,
                tp_size=self.parallel_context.tensor_model_parallel_size,
                tp_group=self.parallel_context.tensor_model_parallel_group,
                bias=True,
                param_dtype=param_dtype,
                param_device=param_device,
                module_path=f"{module_prefix}.up_proj",
                precision_resolver=precision_resolver,
            )
            self.down_proj = RowParallelLinear(
                intermediate_size,
                hidden_size,
                tp_rank=self.parallel_context.tensor_model_parallel_rank,
                tp_size=self.parallel_context.tensor_model_parallel_size,
                tp_group=self.parallel_context.tensor_model_parallel_group,
                bias=True,
                param_dtype=param_dtype,
                param_device=param_device,
                module_path=f"{module_prefix}.down_proj",
                precision_resolver=precision_resolver,
            )
            self.dropout = Dropout(dropout)
        else:
            self.mlp = ExpertMLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dropout=dropout,
                param_dtype=param_dtype,
                param_device=param_device,
                precision_resolver=precision_resolver,
                module_prefix=f"{module_prefix}.mlp",
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_tp:
            gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
            out = self.down_proj(gated)
            return self.dropout(out)

        return self.mlp(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate final dimension by 90 degrees in pairs for RoPE."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1)
    return rotated.flatten(-2)


def _build_rope_cache(
    seq_len: int,
    dim: int,
    device: torch.device,
    theta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create cos/sin caches for rotary embeddings."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to tensor shaped (B, H, S, D)."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (_rotate_half(x) * sin)


class MultiHeadLatentAttention(nn.Module):
    """MLA-style attention approximation with low-rank Q/KV projections."""

    def __init__(
        self,
        config: DeepSeekModelConfig,
        parallel_context: Optional[DeepSeekParallelContext] = None,
        *,
        module_prefix: str,
    ):
        super().__init__()
        self.parallel_context = parallel_context or DeepSeekParallelContext()
        (
            self.attention_tensor_model_parallel_rank,
            self.attention_tensor_model_parallel_size,
            self.attention_tensor_model_parallel_group,
        ) = (
            self.parallel_context.resolve_attention_tensor_model_parallel()
        )
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        if self.num_heads % self.attention_tensor_model_parallel_size != 0:
            raise ValueError("num_attention_heads must be divisible by attention TP size")
        self.local_num_heads = self.num_heads // self.attention_tensor_model_parallel_size
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scale = self.qk_head_dim ** -0.5
        self.rope_theta = config.rope_theta

        self.q_a_proj = Linear(
            self.hidden_size,
            config.q_lora_rank,
            bias=False,
            param_dtype=config.param_dtype,
            param_device=config.param_device,
            module_path=f"{module_prefix}.q_a_proj",
            precision_resolver=config.precision_resolver,
        )
        self.q_a_norm = RMSNorm(
            config.q_lora_rank,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
            param_device=config.param_device,
            module_path=f"{module_prefix}.q_a_norm",
            precision_resolver=config.precision_resolver,
        )
        if self.attention_tensor_model_parallel_size > 1:
            self.q_b_proj = ColumnParallelLinear(
                config.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                tp_rank=self.attention_tensor_model_parallel_rank,
                tp_size=self.attention_tensor_model_parallel_size,
                tp_group=self.attention_tensor_model_parallel_group,
                bias=False,
                param_dtype=config.param_dtype,
                param_device=config.param_device,
                module_path=f"{module_prefix}.q_b_proj",
                precision_resolver=config.precision_resolver,
            )
        else:
            self.q_b_proj = Linear(
                config.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                param_dtype=config.param_dtype,
                param_device=config.param_device,
                module_path=f"{module_prefix}.q_b_proj",
                precision_resolver=config.precision_resolver,
            )

        self.kv_a_proj = Linear(
            self.hidden_size,
            config.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            param_dtype=config.param_dtype,
            param_device=config.param_device,
            module_path=f"{module_prefix}.kv_a_proj",
            precision_resolver=config.precision_resolver,
        )
        self.kv_a_norm = RMSNorm(
            config.kv_lora_rank,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
            param_device=config.param_device,
            module_path=f"{module_prefix}.kv_a_norm",
            precision_resolver=config.precision_resolver,
        )
        if self.attention_tensor_model_parallel_size > 1:
            self.kv_b_proj = ColumnParallelLinear(
                config.kv_lora_rank,
                self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
                tp_rank=self.attention_tensor_model_parallel_rank,
                tp_size=self.attention_tensor_model_parallel_size,
                tp_group=self.attention_tensor_model_parallel_group,
                bias=False,
                param_dtype=config.param_dtype,
                param_device=config.param_device,
                module_path=f"{module_prefix}.kv_b_proj",
                precision_resolver=config.precision_resolver,
            )
            self.out_proj = RowParallelLinear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                tp_rank=self.attention_tensor_model_parallel_rank,
                tp_size=self.attention_tensor_model_parallel_size,
                tp_group=self.attention_tensor_model_parallel_group,
                bias=False,
                param_dtype=config.param_dtype,
                param_device=config.param_device,
                module_path=f"{module_prefix}.out_proj",
                precision_resolver=config.precision_resolver,
            )
        else:
            self.kv_b_proj = Linear(
                config.kv_lora_rank,
                self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=False,
                param_dtype=config.param_dtype,
                param_device=config.param_device,
                module_path=f"{module_prefix}.kv_b_proj",
                precision_resolver=config.precision_resolver,
            )
            self.out_proj = Linear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=False,
                param_dtype=config.param_dtype,
                param_device=config.param_device,
                module_path=f"{module_prefix}.out_proj",
                precision_resolver=config.precision_resolver,
            )
        self.dropout = Dropout(config.attention_dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x: (B, S, H)
        batch_size, seq_len, _ = x.shape

        q_latent = self.q_a_proj(x)
        q = self.q_b_proj(self.q_a_norm(q_latent))
        # Under attention TP, local projection output is only the local head shard.
        q = q.view(batch_size, seq_len, self.local_num_heads, self.qk_head_dim)
        q = q.permute(0, 2, 1, 3)  # (B, heads, S, qk_head_dim)
        q_nope, q_rope = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        kv_a = self.kv_a_proj(x)
        kv_latent, k_rope_shared = torch.split(
            kv_a,
            [self.kv_a_norm.weight.shape[0], self.qk_rope_head_dim],
            dim=-1,
        )
        kv = self.kv_b_proj(self.kv_a_norm(kv_latent))
        kv = kv.view(
            batch_size,
            seq_len,
            self.local_num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        kv = kv.permute(0, 2, 1, 3)  # (B, heads, S, qk_nope+v)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rope = k_rope_shared.unsqueeze(1).expand(-1, self.local_num_heads, -1, -1)
        cos, sin = _build_rope_cache(
            seq_len=seq_len,
            dim=self.qk_rope_head_dim,
            device=x.device,
            theta=self.rope_theta,
        )
        q_rope = _apply_rope(q_rope, cos, sin)
        k_rope = _apply_rope(k_rope, cos, sin)

        q_final = torch.cat([q_nope, q_rope], dim=-1)
        k_final = torch.cat([k_nope, k_rope], dim=-1)

        scores = torch.matmul(q_final, k_final.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores + attention_mask
        scores_softmax = scores
        if scores_softmax.dtype not in (torch.float32, torch.float64):
            scores_softmax = scores_softmax.float()
        # Softmax in stable dtype, then align with value dtype for attention matmul.
        probs = torch.softmax(scores_softmax, dim=-1).to(dtype=v.dtype)
        probs = self.dropout(probs)

        out = torch.matmul(probs, v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_len, self.local_num_heads * self.v_head_dim)
        return self.out_proj(out)


class RoutedMoE(nn.Module):
    """Token-level routed MoE wrapper backed by reusable LocalRoutedMoE infra."""

    def __init__(
        self,
        config: DeepSeekModelConfig,
        parallel_context: Optional[DeepSeekParallelContext] = None,
        *,
        module_prefix: str,
    ):
        super().__init__()
        self.parallel_context = parallel_context or DeepSeekParallelContext()
        (
            self.expert_model_parallel_rank,
            self.expert_model_parallel_size,
            self.expert_model_parallel_group,
        ) = self.parallel_context.resolve_expert_model_parallel()
        self.use_moe_sequence_parallel = (
            self.parallel_context.sequence_parallel
            and self.parallel_context.tensor_model_parallel_size > 1
        )
        shared_kwargs = {
            "hidden_size": config.hidden_size,
            "expert_intermediate_size": config.moe_intermediate_size,
            "num_experts": config.n_routed_experts,
            "top_k": config.num_experts_per_tok,
            "param_dtype": config.param_dtype,
            "param_device": config.param_device,
            "precision_resolver": config.precision_resolver,
            "module_prefix": module_prefix,
            "dropout": config.dropout,
            "n_shared_experts": config.n_shared_experts,
            "scoring_func": config.scoring_func,
            "n_group": config.n_group,
            "topk_group": config.topk_group,
            "norm_topk_prob": config.norm_topk_prob,
            "routed_scaling_factor": config.routed_scaling_factor,
            "capacity_factor": self.parallel_context.capacity_factor,
        }

        if self.expert_model_parallel_size > 1:
            self.moe = ExpertParallelMoE(
                **shared_kwargs,
                ep_rank=self.expert_model_parallel_rank,
                ep_size=self.expert_model_parallel_size,
                ep_group=self.expert_model_parallel_group,
                expert_tp_size=self.parallel_context.expert_tensor_parallel_size,
            )
        else:
            self.moe = LocalRoutedMoE(**shared_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_moe_sequence_parallel:
            return self.moe(x)

        # Avoid duplicate MoE token compute across TP ranks by sharding sequence on TP.
        local_x = scatter_to_sequence_parallel_region(
            x,
            tp_rank=self.parallel_context.tensor_model_parallel_rank,
            tp_size=self.parallel_context.tensor_model_parallel_size,
            tp_group=self.parallel_context.tensor_model_parallel_group,
            seq_dim=1,
        )
        local_out = self.moe(local_x)
        return gather_from_sequence_parallel_region(
            local_out,
            tp_rank=self.parallel_context.tensor_model_parallel_rank,
            tp_size=self.parallel_context.tensor_model_parallel_size,
            tp_group=self.parallel_context.tensor_model_parallel_group,
            seq_dim=1,
        )


class DeepSeekDecoderBlock(nn.Module):
    """DeepSeek-like decoder block with MLA and dense/MoE FFN."""

    def __init__(
        self,
        config: DeepSeekModelConfig,
        layer_idx: int,
        parallel_context: Optional[DeepSeekParallelContext] = None,
        *,
        module_prefix: str,
    ):
        super().__init__()
        self.parallel_context = parallel_context or DeepSeekParallelContext()
        self.attn_norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
            param_device=config.param_device,
            module_path=f"{module_prefix}.attn_norm",
            precision_resolver=config.precision_resolver,
        )
        self.attn = MultiHeadLatentAttention(
            config,
            parallel_context=self.parallel_context,
            module_prefix=f"{module_prefix}.attn",
        )
        self.ffn_norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            param_dtype=config.param_dtype,
            param_device=config.param_device,
            module_path=f"{module_prefix}.ffn_norm",
            precision_resolver=config.precision_resolver,
        )

        use_dense = layer_idx < config.first_k_dense_replace
        use_moe = (
            not use_dense
            and (layer_idx - config.first_k_dense_replace) % max(1, config.moe_layer_freq) == 0
        )

        if use_moe:
            self.ffn = RoutedMoE(
                config,
                parallel_context=self.parallel_context,
                module_prefix=f"{module_prefix}.ffn",
            )
            self.ffn_type = "moe"
        else:
            self.ffn = GatedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                dropout=config.dropout,
                param_dtype=config.param_dtype,
                param_device=config.param_device,
                precision_resolver=config.precision_resolver,
                module_prefix=f"{module_prefix}.ffn",
                parallel_context=self.parallel_context,
            )
            self.ffn_type = "dense"

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, attention_mask)
        x = residual + x

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x
        return x


class DeepSeekModel(nn.Module):
    """DeepSeek-V3-like causal decoder language model."""

    def __init__(
        self,
        config: DeepSeekModelConfig,
        parallel_context: Optional[DeepSeekParallelContext] = None,
    ):
        super().__init__()
        if config.module_compute_dtype_overrides:
            config.precision_resolver = _DeepSeekConfigPrecisionResolver(
                config.precision_resolver,
                config.module_compute_dtype_overrides,
            )
        self.config = config
        self.parallel_context = parallel_context or DeepSeekParallelContext()
        _, attention_tensor_model_parallel_size, _ = (
            self.parallel_context.resolve_attention_tensor_model_parallel()
        )
        _, expert_model_parallel_size, _ = self.parallel_context.resolve_expert_model_parallel()
        self.pipeline_model_parallel_rank = self.parallel_context.pipeline_model_parallel_rank
        self.pipeline_model_parallel_size = self.parallel_context.pipeline_model_parallel_size
        self.pipeline_model_parallel_group = self.parallel_context.pipeline_model_parallel_group
        self.pipeline_layer_start, self.pipeline_layer_end = self.parallel_context.resolve_pp_layer_range(
            config.num_hidden_layers
        )
        self.pipeline_layer_boundaries = self.parallel_context.resolve_pp_layer_boundaries(
            config.num_hidden_layers
        )

        if (
            self.parallel_context.tensor_model_parallel_size > 1
            and config.intermediate_size % self.parallel_context.tensor_model_parallel_size != 0
        ):
            raise ValueError(
                "intermediate_size must be divisible by tensor_model_parallel_size for TP dense FFN"
            )
        if (
            attention_tensor_model_parallel_size > 1
            and config.num_attention_heads % attention_tensor_model_parallel_size != 0
        ):
            raise ValueError("num_attention_heads must be divisible by attention TP size")
        if expert_model_parallel_size > 1 and config.n_routed_experts % expert_model_parallel_size != 0:
            raise ValueError(
                "n_routed_experts must be divisible by expert_model_parallel_size for EP MoE"
            )
        if self.pipeline_layer_end <= self.pipeline_layer_start:
            raise ValueError("Each PP stage must own at least one decoder layer")
        if config.tie_word_embeddings and self.pipeline_model_parallel_size > 1:
            raise ValueError(
                "tie_word_embeddings is not supported when pipeline_model_parallel_size > 1"
            )

        self.token_embeddings: Optional[Embedding]
        self.dropout: Optional[Dropout]
        if self.is_first_pp_stage:
            self.token_embeddings = Embedding(
                config.vocab_size,
                config.hidden_size,
                param_dtype=config.param_dtype,
                param_device=config.param_device,
                module_path="token_embeddings",
                precision_resolver=config.precision_resolver,
            )
            self.dropout = Dropout(config.dropout)
        else:
            self.token_embeddings = None
            self.dropout = None

        # Local stage owns only [pipeline_layer_start, pipeline_layer_end) decoder blocks.
        self.blocks = nn.ModuleList(
            [
                DeepSeekDecoderBlock(
                    config,
                    i,
                    parallel_context=self.parallel_context,
                    module_prefix=f"blocks.{i}",
                )
                for i in range(self.pipeline_layer_start, self.pipeline_layer_end)
            ]
        )

        self.final_norm: Optional[RMSNorm]
        self.lm_head: Optional[Linear]
        if self.is_last_pp_stage:
            self.final_norm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                param_dtype=config.param_dtype,
                param_device=config.param_device,
                module_path="final_norm",
                precision_resolver=config.precision_resolver,
            )
            self.lm_head = Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                param_dtype=config.param_dtype,
                param_device=config.param_device,
                module_path="lm_head",
                precision_resolver=config.precision_resolver,
            )
        else:
            self.final_norm = None
            self.lm_head = None

        self.apply(self._init_weights)

        if config.tie_word_embeddings and self.token_embeddings is not None and self.lm_head is not None:
            self.lm_head.weight = self.token_embeddings.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (Linear, ColumnParallelLinear, RowParallelLinear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def _create_causal_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.view(1, 1, seq_len, seq_len).expand(batch_size, -1, -1, -1)

    @property
    def is_first_pp_stage(self) -> bool:
        """True when this rank owns the first PP stage."""
        return self.pipeline_model_parallel_rank == 0

    @property
    def is_last_pp_stage(self) -> bool:
        """True when this rank owns the last PP stage."""
        return self.pipeline_model_parallel_rank == self.pipeline_model_parallel_size - 1

    def local_layer_range(self) -> tuple[int, int]:
        """Return local global-layer range `[start, end)` owned by this PP rank."""
        return self.pipeline_layer_start, self.pipeline_layer_end

    def forward_stage(
        self,
        input_ids: Optional[torch.Tensor],
        hidden_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run local stage forward for pipeline execution.

        - First stage consumes `input_ids` and returns hidden states (or logits if pipeline size=1).
        - Middle stages consume hidden states and return hidden states.
        - Last stage consumes hidden states and returns logits.
        """
        x: torch.Tensor

        if self.is_first_pp_stage:
            if input_ids is None:
                raise ValueError("input_ids is required on the first PP stage")
            if self.token_embeddings is None or self.dropout is None:
                raise RuntimeError("First PP stage is missing embedding modules")
            batch_size, seq_len = input_ids.shape
            x = self.token_embeddings(input_ids)
            x = self.dropout(x)
        else:
            if hidden_states is None:
                raise ValueError("hidden_states is required on non-first PP stages")
            batch_size, seq_len, _ = hidden_states.shape
            x = hidden_states

        if attention_mask is None:
            attention_mask = self._create_causal_mask(batch_size, seq_len, x.device)

        for block in self.blocks:
            x = block(x, attention_mask)

        if self.is_last_pp_stage:
            if self.final_norm is None or self.lm_head is None:
                raise RuntimeError("Last PP stage is missing output modules")
            x = self.final_norm(x)
            return self.lm_head(x)

        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Keep single-stage behavior unchanged.
        if self.pipeline_model_parallel_size > 1:
            raise RuntimeError(
                "forward() is only valid for pipeline_model_parallel_size==1; "
                "use forward_stage() for PP"
            )
        return self.forward_stage(
            input_ids=input_ids,
            hidden_states=None,
            attention_mask=attention_mask,
        )

    @property
    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())
