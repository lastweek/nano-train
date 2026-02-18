"""
DeepSeek-V3-like decoder model with MLA-style attention and routed MoE.

The implementation keeps core DeepSeek-V3 architectural ideas and key
configuration fields, but remains a compact approximation for local training:
- MLA-style low-rank query/KV projections with split NoPE/RoPE QK dimensions
- Dense-first then MoE decoder blocks
- Routed experts with configurable top-k behavior and optional group routing
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.layers import Dropout, Embedding, Linear


@dataclass
class DeepSeekModelConfig:
    """Configuration shaped after DeepSeek-V3 key parameters."""

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


class RMSNorm(nn.Module):
    """RMSNorm used by DeepSeek-style decoder blocks."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return x_norm * self.weight


class GatedMLP(nn.Module):
    """SwiGLU-style feed-forward block."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        out = self.down_proj(gated)
        return self.dropout(out)


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

    def __init__(self, config: DeepSeekModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scale = self.qk_head_dim ** -0.5
        self.rope_theta = config.rope_theta

        self.q_a_proj = Linear(self.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = Linear(
            config.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
        )

        self.kv_a_proj = Linear(
            self.hidden_size,
            config.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_norm = RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = Linear(
            config.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.out_proj = Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        self.dropout = Dropout(config.attention_dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x: (B, S, H)
        batch_size, seq_len, _ = x.shape

        q_latent = self.q_a_proj(x)
        q = self.q_b_proj(self.q_a_norm(q_latent))
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
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
        kv = kv.view(batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        kv = kv.permute(0, 2, 1, 3)  # (B, heads, S, qk_nope+v)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rope = k_rope_shared.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
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
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        out = torch.matmul(probs, v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_len, self.num_heads * self.v_head_dim)
        return self.out_proj(out)


class RoutedMoE(nn.Module):
    """Token-level routed MoE with optional shared experts."""

    def __init__(self, config: DeepSeekModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.scoring_func = config.scoring_func
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        self.router = Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                GatedMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    dropout=config.dropout,
                )
                for _ in range(self.num_experts)
            ]
        )
        self.shared_experts = nn.ModuleList(
            [
                GatedMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    dropout=config.dropout,
                )
                for _ in range(config.n_shared_experts)
            ]
        )

    def _score_experts(self, logits: torch.Tensor) -> torch.Tensor:
        if self.scoring_func == "sigmoid":
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

    def _apply_group_routing(self, scores: torch.Tensor) -> torch.Tensor:
        if self.n_group <= 1 or self.topk_group >= self.n_group:
            return scores

        tokens = scores.size(0)
        experts_per_group = self.num_experts // self.n_group
        grouped = scores.view(tokens, self.n_group, experts_per_group)
        group_scores = grouped.max(dim=-1).values
        selected_groups = torch.topk(group_scores, k=self.topk_group, dim=-1).indices

        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, selected_groups, True)
        expert_mask = group_mask.unsqueeze(-1).expand(-1, -1, experts_per_group).reshape(tokens, -1)

        if self.scoring_func == "sigmoid":
            return torch.where(expert_mask, scores, torch.zeros_like(scores))
        neg_inf = torch.full_like(scores, float("-inf"))
        return torch.where(expert_mask, scores, neg_inf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, H)
        batch_size, seq_len, hidden_size = x.shape
        tokens = x.view(-1, hidden_size)

        logits = self.router(tokens)
        scores = self._score_experts(logits)
        routed_scores = self._apply_group_routing(scores)
        topk_weights, topk_indices = torch.topk(routed_scores, k=self.top_k, dim=-1)

        if self.scoring_func == "sigmoid" and self.norm_topk_prob:
            topk_weights = topk_weights / torch.clamp(topk_weights.sum(dim=-1, keepdim=True), min=1e-9)
        elif self.scoring_func == "softmax":
            topk_weights = torch.softmax(topk_weights, dim=-1)
        topk_weights = topk_weights * self.routed_scaling_factor

        output = torch.zeros_like(tokens)
        for expert_idx, expert in enumerate(self.experts):
            selected = (topk_indices == expert_idx).nonzero(as_tuple=False)
            if selected.numel() == 0:
                continue
            token_indices = selected[:, 0]
            slot_indices = selected[:, 1]
            expert_input = tokens[token_indices]
            expert_out = expert(expert_input)
            weights = topk_weights[token_indices, slot_indices].unsqueeze(-1)
            output[token_indices] += expert_out * weights

        if self.shared_experts:
            shared = torch.zeros_like(tokens)
            for expert in self.shared_experts:
                shared += expert(tokens)
            output += shared / len(self.shared_experts)

        return output.view(batch_size, seq_len, hidden_size)


class DeepSeekDecoderBlock(nn.Module):
    """DeepSeek-like decoder block with MLA and dense/MoE FFN."""

    def __init__(self, config: DeepSeekModelConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = MultiHeadLatentAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        use_dense = layer_idx < config.first_k_dense_replace
        if use_dense:
            self.ffn = GatedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                dropout=config.dropout,
            )
            self.ffn_type = "dense"
        else:
            if (layer_idx - config.first_k_dense_replace) % max(1, config.moe_layer_freq) == 0:
                self.ffn = RoutedMoE(config)
                self.ffn_type = "moe"
            else:
                self.ffn = GatedMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    dropout=config.dropout,
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

    def __init__(self, config: DeepSeekModelConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.dropout = Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [DeepSeekDecoderBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def _create_causal_mask(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.view(1, 1, seq_len, seq_len).expand(batch_size, -1, -1, -1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # input_ids: (B, S)
        batch_size, seq_len = input_ids.shape
        x = self.token_embeddings(input_ids)
        x = self.dropout(x)

        if attention_mask is None:
            attention_mask = self._create_causal_mask(batch_size, seq_len, input_ids.device)

        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.final_norm(x)
        return self.lm_head(x)

    @property
    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())
