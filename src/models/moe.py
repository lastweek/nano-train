"""
Reusable MoE building blocks for local and expert-parallel training.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
import torch.nn.functional as F

from src.layers import Dropout
from src.layers import Linear

if TYPE_CHECKING:
    from src.runtime.contracts import ModulePrecisionResolver
else:
    ModulePrecisionResolver = object


@dataclass
class RouterOutput:
    """Router outputs for routed MoE layers."""

    topk_indices: torch.Tensor
    topk_weights: torch.Tensor
    aux_loss: torch.Tensor
    dropped_fraction: float


@dataclass
class _DispatchPlan:
    """Per-destination payloads prepared before EP token dispatch."""

    send_tokens: list[torch.Tensor]
    send_token_idx: list[torch.Tensor]
    send_slot_idx: list[torch.Tensor]
    send_local_expert_idx: list[torch.Tensor]
    send_sort_weight: list[torch.Tensor]
    send_counts: list[int]


@dataclass
class _DispatchedTokens:
    """Tokens and metadata received by this rank after dispatch."""

    recv_tokens: torch.Tensor
    recv_token_idx: torch.Tensor
    recv_slot_idx: torch.Tensor
    recv_local_expert_idx: torch.Tensor
    recv_sort_weight: torch.Tensor
    recv_src_ranks: torch.Tensor


@dataclass
class _ReturnPayload:
    """Expert outputs and metadata that will be sent back to source ranks."""

    send_back_features: list[torch.Tensor]
    send_back_token_idx: list[torch.Tensor]
    send_back_slot_idx: list[torch.Tensor]
    dropped: int
    local_counts: torch.Tensor
    autograd_fallback: torch.Tensor


def _comm_stochastic_round(values: torch.Tensor) -> torch.Tensor:
    noise = torch.rand_like(values)
    positive = torch.floor(values + noise)
    negative = torch.ceil(values - noise)
    return torch.where(values >= 0, positive, negative)


def _comm_quant_dequant(
    tensor: torch.Tensor,
    *,
    granularity: str,
    rounding_mode: str,
    bits: int = 8,
) -> torch.Tensor:
    """Apply fake quant-dequant for comm payloads while preserving autograd."""
    if tensor.numel() == 0:
        return tensor

    x = tensor.float()
    qmax = (1 << (bits - 1)) - 1

    def _quantize_block(block: torch.Tensor) -> torch.Tensor:
        scale = block.detach().abs().amax()
        if not torch.isfinite(scale) or scale <= 0:
            scale = block.new_tensor(1.0)
        else:
            scale = scale / float(qmax)
        normalized = block / scale
        if rounding_mode == "stochastic":
            q = _comm_stochastic_round(normalized)
        else:
            q = torch.round(normalized)
        q = torch.clamp(q, min=-qmax, max=qmax)
        return q * scale

    if granularity == "tensor":
        dq = _quantize_block(x)
        dq = dq.to(dtype=tensor.dtype)
        return tensor + (dq - tensor).detach()

    if granularity == "channel":
        if x.dim() < 2:
            dq = _quantize_block(x)
            dq = dq.to(dtype=tensor.dtype)
            return tensor + (dq - tensor).detach()
        chunks = []
        for idx in range(x.shape[-1]):
            chunks.append(_quantize_block(x[..., idx : idx + 1]))
        dq = torch.cat(chunks, dim=-1).to(dtype=tensor.dtype)
        return tensor + (dq - tensor).detach()

    if granularity in {"tile_1x128", "block_128x128"}:
        tile = 128
        x2d = x.reshape(-1, x.shape[-1])
        width = x2d.shape[-1]
        n_tiles = (width + tile - 1) // tile
        pad = n_tiles * tile - width
        if pad > 0:
            x2d = F.pad(x2d, (0, pad))
        blocks = []
        for block_idx in range(n_tiles):
            start = block_idx * tile
            end = (block_idx + 1) * tile
            blocks.append(_quantize_block(x2d[:, start:end]))
        dq2d = torch.cat(blocks, dim=-1)
        if pad > 0:
            dq2d = dq2d[:, :width]
        dq = dq2d.reshape_as(x).to(dtype=tensor.dtype)
        return tensor + (dq - tensor).detach()

    dq = _quantize_block(x).to(dtype=tensor.dtype)
    return tensor + (dq - tensor).detach()


class ExpertMLP(nn.Module):
    """SwiGLU expert MLP used by routed MoE layers."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float,
        *,
        param_dtype: torch.dtype,
        param_device: torch.device | None,
        precision_resolver: ModulePrecisionResolver,
        module_prefix: str,
    ) -> None:
        super().__init__()
        self.gate_proj = Linear(
            hidden_size,
            intermediate_size,
            bias=False,
            param_dtype=param_dtype,
            param_device=param_device,
            module_path=f"{module_prefix}.gate_proj",
            precision_resolver=precision_resolver,
        )
        self.up_proj = Linear(
            hidden_size,
            intermediate_size,
            bias=False,
            param_dtype=param_dtype,
            param_device=param_device,
            module_path=f"{module_prefix}.up_proj",
            precision_resolver=precision_resolver,
        )
        self.down_proj = Linear(
            intermediate_size,
            hidden_size,
            bias=False,
            param_dtype=param_dtype,
            param_device=param_device,
            module_path=f"{module_prefix}.down_proj",
            precision_resolver=precision_resolver,
        )
        self.dropout = Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        out = self.down_proj(gated)
        return self.dropout(out)


class TopKRouter(nn.Module):
    """DeepSeek-style top-k router with optional group routing and aux balancing loss."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        *,
        param_dtype: torch.dtype,
        param_device: torch.device | None,
        precision_resolver: ModulePrecisionResolver,
        module_prefix: str,
        scoring_func: str = "sigmoid",
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if scoring_func not in {"sigmoid", "softmax"}:
            raise ValueError("scoring_func must be 'sigmoid' or 'softmax'")
        if n_group <= 0 or num_experts % n_group != 0:
            raise ValueError("n_group must divide num_experts")
        if topk_group <= 0 or topk_group > n_group:
            raise ValueError("topk_group must be in [1, n_group]")

        experts_per_group = num_experts // n_group
        if topk_group * experts_per_group < top_k:
            raise ValueError("topk_group exposes fewer experts than top_k")

        self.num_experts = num_experts
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor

        self.router = Linear(
            hidden_size,
            num_experts,
            bias=False,
            param_dtype=param_dtype,
            param_device=param_device,
            module_path=f"{module_prefix}.router",
            precision_resolver=precision_resolver,
        )

    def _score_experts(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute routing scores from raw logits."""
        if self.scoring_func == "sigmoid":
            return torch.sigmoid(logits)
        logits_softmax = logits
        if logits_softmax.dtype not in (torch.float32, torch.float64):
            logits_softmax = logits_softmax.float()
        return torch.softmax(logits_softmax, dim=-1).to(dtype=logits.dtype)

    def _apply_group_routing(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply DeepSeek-style group routing mask before top-k selection."""
        if self.n_group <= 1 or self.topk_group >= self.n_group:
            return scores

        num_tokens = scores.size(0)
        experts_per_group = self.num_experts // self.n_group
        grouped_scores = scores.view(num_tokens, self.n_group, experts_per_group)
        group_scores = grouped_scores.max(dim=-1).values
        selected_groups = torch.topk(group_scores, k=self.topk_group, dim=-1).indices

        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, selected_groups, True)
        expert_mask = group_mask.unsqueeze(-1).expand(-1, -1, experts_per_group)
        expert_mask = expert_mask.reshape(num_tokens, self.num_experts)

        if self.scoring_func == "sigmoid":
            return torch.where(expert_mask, scores, torch.zeros_like(scores))

        neg_inf = torch.full_like(scores, float("-inf"))
        return torch.where(expert_mask, scores, neg_inf)

    def _compute_aux_loss(self, scores: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute expert-level load-balance auxiliary loss (DeepSeekMoE-style proxy).

        This implements L_exp = N * sum_i(f_i * P_i), where:
        - f_i uses realized top-k assignments (count_i / (K*T))
        - P_i uses mean router probability for expert i

        Note: device-level balance loss is documented but not added in this tutorial code.
        """
        if scores.numel() == 0:
            return torch.zeros((), dtype=scores.dtype, device=scores.device)

        if self.scoring_func == "sigmoid":
            normed_scores = scores / torch.clamp(scores.sum(dim=-1, keepdim=True), min=1e-9)
        else:
            normed_scores = scores

        # One-hot top-k selections: [T, K, N].
        assignments = F.one_hot(topk_indices, num_classes=self.num_experts).float()
        # Per-token selection mass: selected experts get 1/K, others 0. Shape [T, N].
        assignments = assignments.mean(dim=1)

        # f_i proxy in [0, +): mean selected mass per expert over tokens.
        expert_fraction = assignments.mean(dim=0)
        # P_i proxy: mean router probability per expert over tokens.
        expert_prob = normed_scores.mean(dim=0)

        aux_loss = (expert_fraction * expert_prob).sum() * float(self.num_experts)
        return aux_loss.to(dtype=scores.dtype)

    def forward(self, tokens: torch.Tensor) -> RouterOutput:
        """Route flattened tokens to global experts with top-k selection."""
        if tokens.dim() != 2:
            raise ValueError("tokens must have shape [num_tokens, hidden_size]")

        num_tokens = tokens.size(0)
        if num_tokens == 0:
            empty_idx = torch.zeros((0, self.top_k), dtype=torch.long, device=tokens.device)
            empty_weight = torch.zeros((0, self.top_k), dtype=tokens.dtype, device=tokens.device)
            return RouterOutput(
                topk_indices=empty_idx,
                topk_weights=empty_weight,
                aux_loss=torch.zeros((), dtype=tokens.dtype, device=tokens.device),
                dropped_fraction=0.0,
            )

        logits = self.router(tokens)
        scores = self._score_experts(logits)
        routed_scores = self._apply_group_routing(scores)

        topk_weights, topk_indices = torch.topk(routed_scores, k=self.top_k, dim=-1)

        if self.scoring_func == "sigmoid" and self.norm_topk_prob:
            denom = torch.clamp(topk_weights.sum(dim=-1, keepdim=True), min=1e-9)
            topk_weights = topk_weights / denom
        elif self.scoring_func == "softmax":
            topk_weights_softmax = topk_weights
            if topk_weights_softmax.dtype not in (torch.float32, torch.float64):
                topk_weights_softmax = topk_weights_softmax.float()
            topk_weights = torch.softmax(topk_weights_softmax, dim=-1).to(dtype=topk_weights.dtype)

        topk_weights = topk_weights * self.routed_scaling_factor
        aux_loss = self._compute_aux_loss(scores, topk_indices)

        return RouterOutput(
            topk_indices=topk_indices,
            topk_weights=topk_weights,
            aux_loss=aux_loss,
            dropped_fraction=0.0,
        )


class LocalRoutedMoE(nn.Module):
    """Single-process routed MoE with top-k routing, shared experts, and capacity drops."""

    def __init__(
        self,
        hidden_size: int,
        expert_intermediate_size: int,
        num_experts: int,
        top_k: int,
        *,
        param_dtype: torch.dtype,
        param_device: torch.device | None,
        precision_resolver: ModulePrecisionResolver,
        module_prefix: str,
        dropout: float = 0.0,
        n_shared_experts: int = 0,
        scoring_func: str = "sigmoid",
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        capacity_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if capacity_factor <= 0:
            raise ValueError("capacity_factor must be positive")

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.router = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            param_dtype=param_dtype,
            param_device=param_device,
            precision_resolver=precision_resolver,
            module_prefix=f"{module_prefix}.router",
            scoring_func=scoring_func,
            n_group=n_group,
            topk_group=topk_group,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
        )
        self.experts = nn.ModuleList(
            [
                ExpertMLP(
                    hidden_size=hidden_size,
                    intermediate_size=expert_intermediate_size,
                    dropout=dropout,
                    param_dtype=param_dtype,
                    param_device=param_device,
                    precision_resolver=precision_resolver,
                    module_prefix=f"{module_prefix}.experts.{idx}",
                )
                for idx in range(num_experts)
            ]
        )
        self.shared_experts = nn.ModuleList(
            [
                ExpertMLP(
                    hidden_size=hidden_size,
                    intermediate_size=expert_intermediate_size,
                    dropout=dropout,
                    param_dtype=param_dtype,
                    param_device=param_device,
                    precision_resolver=precision_resolver,
                    module_prefix=f"{module_prefix}.shared_experts.{idx}",
                )
                for idx in range(n_shared_experts)
            ]
        )

        self._last_aux_loss = torch.zeros(())
        self._last_dropped_fraction = 0.0
        self._last_token_counts = torch.zeros(num_experts, dtype=torch.long)

    def _compute_capacity(self, num_tokens: int) -> int:
        """Compute per-expert capacity for top-k assignments."""
        if num_tokens <= 0:
            return 0

        expected = (num_tokens * self.top_k) / float(self.num_experts)
        capacity = int(math.ceil(self.capacity_factor * expected))
        return max(capacity, 1)

    @property
    def last_aux_loss(self) -> torch.Tensor:
        """Last computed router auxiliary loss."""
        return self._last_aux_loss

    @property
    def last_dropped_fraction(self) -> float:
        """Last dropped assignment fraction after capacity filtering."""
        return self._last_dropped_fraction

    @property
    def last_token_counts(self) -> torch.Tensor:
        """Per-expert assignment counts after capacity filtering."""
        return self._last_token_counts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply local routed MoE to inputs of shape [B, S, H]."""
        if x.dim() != 3:
            raise ValueError("x must have shape [batch, seq, hidden]")

        batch_size, seq_len, hidden_size = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError("hidden dimension mismatch")

        tokens = x.reshape(-1, hidden_size)
        output = torch.zeros_like(tokens)

        router_output = self.router(tokens)
        topk_indices = router_output.topk_indices
        topk_weights = router_output.topk_weights

        capacity = self._compute_capacity(tokens.size(0))

        dropped = 0
        token_counts = torch.zeros(self.num_experts, dtype=torch.long, device=tokens.device)

        for expert_idx, expert in enumerate(self.experts):
            selected = (topk_indices == expert_idx).nonzero(as_tuple=False)
            if selected.numel() == 0:
                continue

            token_indices = selected[:, 0]
            slot_indices = selected[:, 1]
            sort_weights = topk_weights[token_indices, slot_indices].detach()

            if capacity > 0 and token_indices.numel() > capacity:
                keep_rel = torch.topk(sort_weights, k=capacity, sorted=True).indices
                token_indices = token_indices[keep_rel]
                slot_indices = slot_indices[keep_rel]
                dropped += selected.size(0) - capacity

            expert_input = tokens[token_indices]
            expert_out = expert(expert_input)
            weights = topk_weights[token_indices, slot_indices].unsqueeze(-1)
            output[token_indices] += expert_out * weights
            token_counts[expert_idx] = token_indices.numel()

        if self.shared_experts:
            shared = torch.zeros_like(tokens)
            for shared_expert in self.shared_experts:
                shared += shared_expert(tokens)
            output += shared / float(len(self.shared_experts))

        total_assignments = topk_indices.numel()
        self._last_aux_loss = router_output.aux_loss
        self._last_dropped_fraction = float(dropped / max(total_assignments, 1))
        self._last_token_counts = token_counts

        return output.view(batch_size, seq_len, hidden_size)


class ExpertParallelMoE(nn.Module):
    """
    Expert-parallel routed MoE with autograd-safe all-to-all dispatch/return.

    Experts are explicitly TP=1 (no tensor-parallel sharding inside expert MLPs).

    Tensor notation used by comments in this class:
    - B: DP-local batch, B_ep: EP-local batch shard, S: sequence length, H: hidden size
    - T = B_ep * S (local flattened tokens), K = top_k, A = T * K (token-expert assignments)
    - N_r: assignments sent to destination rank r
    - N_recv: assignments received on this rank after dispatch all-to-all
    - N_back: returned expert outputs received by source rank in combine all-to-all
    """

    def __init__(
        self,
        hidden_size: int,
        expert_intermediate_size: int,
        num_experts: int,
        top_k: int,
        ep_rank: int,
        ep_size: int,
        ep_group,
        *,
        param_dtype: torch.dtype,
        param_device: torch.device | None,
        precision_resolver: ModulePrecisionResolver,
        module_prefix: str,
        dropout: float = 0.0,
        n_shared_experts: int = 0,
        scoring_func: str = "sigmoid",
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        capacity_factor: float = 1.0,
        expert_tp_size: int = 1,
    ) -> None:
        super().__init__()
        if expert_tp_size != 1:
            raise ValueError("ExpertParallelMoE enforces expert_tp_size=1")
        if ep_size <= 0:
            raise ValueError("ep_size must be positive")
        if ep_rank < 0 or ep_rank >= ep_size:
            raise ValueError("ep_rank must be in [0, ep_size)")
        if num_experts % ep_size != 0:
            raise ValueError("num_experts must be divisible by ep_size")
        if capacity_factor <= 0:
            raise ValueError("capacity_factor must be positive")

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.ep_group = ep_group
        self.capacity_factor = capacity_factor

        self.experts_per_rank = num_experts // ep_size
        self.global_expert_start = ep_rank * self.experts_per_rank

        self.router = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            param_dtype=param_dtype,
            param_device=param_device,
            precision_resolver=precision_resolver,
            module_prefix=f"{module_prefix}.router",
            scoring_func=scoring_func,
            n_group=n_group,
            topk_group=topk_group,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
        )

        self.experts = nn.ModuleList(
            [
                ExpertMLP(
                    hidden_size=hidden_size,
                    intermediate_size=expert_intermediate_size,
                    dropout=dropout,
                    param_dtype=param_dtype,
                    param_device=param_device,
                    precision_resolver=precision_resolver,
                    module_prefix=f"{module_prefix}.experts.{idx}",
                )
                for idx in range(self.experts_per_rank)
            ]
        )

        self.shared_experts = nn.ModuleList(
            [
                ExpertMLP(
                    hidden_size=hidden_size,
                    intermediate_size=expert_intermediate_size,
                    dropout=dropout,
                    param_dtype=param_dtype,
                    param_device=param_device,
                    precision_resolver=precision_resolver,
                    module_prefix=f"{module_prefix}.shared_experts.{idx}",
                )
                for idx in range(n_shared_experts)
            ]
        )

        self._last_aux_loss = torch.zeros(())
        self._last_dropped_fraction = 0.0
        self._last_local_expert_counts = torch.zeros(self.experts_per_rank, dtype=torch.long)

        recipe_fn = getattr(precision_resolver, "deepseek_v3_recipe", None)
        recipe = recipe_fn() if callable(recipe_fn) else None
        self._comm_quant_enabled = bool(recipe is not None and recipe.comm_quant_enabled)
        self._comm_quant_granularity = (
            "tensor" if recipe is None else str(recipe.comm_quant_granularity)
        )
        self._comm_quant_rounding_mode = (
            "nearest" if recipe is None else str(recipe.rounding_mode)
        )

    @property
    def last_aux_loss(self) -> torch.Tensor:
        """Last computed router auxiliary loss."""
        return self._last_aux_loss

    @property
    def last_dropped_fraction(self) -> float:
        """Last dropped assignment fraction after capacity filtering."""
        return self._last_dropped_fraction

    @property
    def last_local_expert_counts(self) -> torch.Tensor:
        """Per-local-expert assignment counts after capacity filtering."""
        return self._last_local_expert_counts

    def _require_distributed(self) -> None:
        """Validate EP distributed setup when ep_size > 1."""
        if self.ep_size == 1:
            return
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Distributed process group must be initialized for ep_size > 1")
        if self.ep_group is None:
            raise RuntimeError("ep_group must be provided for ep_size > 1")

    def _compute_capacity(self, num_local_tokens: int, device: torch.device) -> int:
        """Compute per-expert capacity using global EP token count."""
        if num_local_tokens <= 0:
            return 0

        token_count = torch.tensor([float(num_local_tokens)], device=device)
        if self.ep_size > 1:
            dist.all_reduce(token_count, op=dist.ReduceOp.SUM, group=self.ep_group)

        total_tokens = float(token_count.item())
        expected = (total_tokens * self.top_k) / float(self.num_experts)
        capacity = int(math.ceil(self.capacity_factor * expected))
        return max(capacity, 1)

    def _build_send_partitions(
        self,
        tokens: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> _DispatchPlan:
        """Build per-destination dispatch payloads from routed token assignments."""
        # topk_indices/topk_weights: [T, K] -> flatten into A = T*K assignments.
        num_tokens = tokens.size(0)
        token_indices = torch.arange(num_tokens, device=tokens.device, dtype=torch.long)
        token_indices = token_indices.repeat_interleave(self.top_k)

        slot_indices = torch.arange(self.top_k, device=tokens.device, dtype=torch.long)
        slot_indices = slot_indices.unsqueeze(0).expand(num_tokens, -1).reshape(-1)

        expert_indices = topk_indices.reshape(-1)
        sort_weights = topk_weights.detach().reshape(-1)

        # Map global expert id -> EP destination rank + local expert id on that rank.
        destination_ranks = expert_indices // self.experts_per_rank
        local_expert_indices = expert_indices % self.experts_per_rank

        # Repeat tokens so each (token, selected expert) assignment carries its own payload.
        dispatched_tokens = tokens[token_indices]

        send_tokens: list[torch.Tensor] = []
        send_token_idx: list[torch.Tensor] = []
        send_slot_idx: list[torch.Tensor] = []
        send_local_expert_idx: list[torch.Tensor] = []
        send_sort_weight: list[torch.Tensor] = []

        for dst_rank in range(self.ep_size):
            mask = destination_ranks == dst_rank
            # send_*[dst_rank] has shape [N_r, ...].
            send_tokens.append(dispatched_tokens[mask])
            send_token_idx.append(token_indices[mask])
            send_slot_idx.append(slot_indices[mask])
            send_local_expert_idx.append(local_expert_indices[mask])
            send_sort_weight.append(sort_weights[mask])

        # send_counts[dst_rank] == N_r
        send_counts = [int(tensor.size(0)) for tensor in send_tokens]
        return _DispatchPlan(
            send_tokens=send_tokens,
            send_token_idx=send_token_idx,
            send_slot_idx=send_slot_idx,
            send_local_expert_idx=send_local_expert_idx,
            send_sort_weight=send_sort_weight,
            send_counts=send_counts,
        )

    def _exchange_counts(self, send_counts: list[int], device: torch.device) -> list[int]:
        """Exchange variable-size split counts across EP ranks."""
        send_counts_tensor = torch.tensor(send_counts, device=device, dtype=torch.long)
        recv_counts_tensor = torch.zeros_like(send_counts_tensor)
        dist.all_to_all_single(recv_counts_tensor, send_counts_tensor, group=self.ep_group)
        return [int(value) for value in recv_counts_tensor.tolist()]

    def _flatten_splits(
        self,
        tensors: list[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Concatenate list of split tensors, returning an empty tensor when all are empty."""
        if not tensors:
            return torch.empty(0, dtype=dtype, device=device)
        non_empty = [tensor for tensor in tensors if tensor.numel() > 0]
        if not non_empty:
            return torch.empty(0, dtype=dtype, device=device)
        return torch.cat(non_empty, dim=0)

    def _flatten_splits_for_autograd(
        self,
        tensors: list[torch.Tensor],
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenate split tensors while preserving autograd connectivity on empty sends.

        When all splits are empty we return `fallback[:0]` so the send tensor stays attached to
        the current graph.
        """
        non_empty = [tensor for tensor in tensors if tensor.numel() > 0]
        if non_empty:
            return torch.cat(non_empty, dim=0)
        return fallback[:0]

    def _all_to_all_metadata(
        self,
        send_tensor: torch.Tensor,
        send_counts: list[int],
        recv_counts: list[int],
    ) -> torch.Tensor:
        """All-to-all exchange for metadata tensors using explicit split sizes."""
        recv_total = int(sum(recv_counts))
        recv_tensor = torch.empty(recv_total, dtype=send_tensor.dtype, device=send_tensor.device)
        dist.all_to_all_single(
            recv_tensor,
            send_tensor,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=self.ep_group,
        )
        return recv_tensor

    def _all_to_all_autograd(
        self,
        send_tensor: torch.Tensor,
        send_counts: list[int],
        recv_counts: list[int],
        output_shape_tail: tuple[int, ...],
    ) -> torch.Tensor:
        """Autograd-safe all-to-all for variable split tensors."""
        recv_total = int(sum(recv_counts))
        recv_shape = (recv_total,) + output_shape_tail
        recv_tensor = torch.empty(recv_shape, dtype=send_tensor.dtype, device=send_tensor.device)
        return dist_nn.all_to_all_single(
            recv_tensor,
            send_tensor,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=self.ep_group,
        )

    def _dispatch_tokens(
        self,
        tokens: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> _DispatchedTokens:
        """Dispatch routed token copies and metadata to expert-owner ranks."""
        plan = self._build_send_partitions(tokens, topk_indices, topk_weights)
        recv_counts = self._exchange_counts(plan.send_counts, device=tokens.device)

        # A2A #1 (dispatch): token payloads move from source EP rank -> expert-owner EP rank.
        # recv_tokens shape: [N_recv, H].
        send_tokens = self._flatten_splits(plan.send_tokens, tokens.dtype, tokens.device)
        if self._comm_quant_enabled:
            send_tokens = _comm_quant_dequant(
                send_tokens,
                granularity=self._comm_quant_granularity,
                rounding_mode=self._comm_quant_rounding_mode,
                bits=8,
            )

        recv_tokens = self._all_to_all_autograd(
            send_tensor=send_tokens,
            send_counts=plan.send_counts,
            recv_counts=recv_counts,
            output_shape_tail=(self.hidden_size,),
        )
        if self._comm_quant_enabled:
            recv_tokens = recv_tokens.to(dtype=tokens.dtype)

        # Metadata A2A mirrors payload routing; each tensor below has shape [N_recv].
        recv_token_idx = self._all_to_all_metadata(
            send_tensor=self._flatten_splits(plan.send_token_idx, torch.long, tokens.device),
            send_counts=plan.send_counts,
            recv_counts=recv_counts,
        )
        recv_slot_idx = self._all_to_all_metadata(
            send_tensor=self._flatten_splits(plan.send_slot_idx, torch.long, tokens.device),
            send_counts=plan.send_counts,
            recv_counts=recv_counts,
        )
        recv_local_expert_idx = self._all_to_all_metadata(
            send_tensor=self._flatten_splits(
                plan.send_local_expert_idx,
                torch.long,
                tokens.device,
            ),
            send_counts=plan.send_counts,
            recv_counts=recv_counts,
        )
        recv_sort_weight = self._all_to_all_metadata(
            send_tensor=self._flatten_splits(plan.send_sort_weight, tokens.dtype, tokens.device),
            send_counts=plan.send_counts,
            recv_counts=recv_counts,
        )

        src_rank_ids = torch.arange(self.ep_size, dtype=torch.long, device=tokens.device)
        recv_counts_tensor = torch.tensor(recv_counts, dtype=torch.long, device=tokens.device)
        # recv_src_ranks marks which source EP rank each received assignment came from.
        recv_src_ranks = torch.repeat_interleave(src_rank_ids, recv_counts_tensor)

        return _DispatchedTokens(
            recv_tokens=recv_tokens,
            recv_token_idx=recv_token_idx,
            recv_slot_idx=recv_slot_idx,
            recv_local_expert_idx=recv_local_expert_idx,
            recv_sort_weight=recv_sort_weight,
            recv_src_ranks=recv_src_ranks,
        )

    def _materialize_rank_splits(
        self,
        rank_parts: list[list[torch.Tensor]],
        *,
        dtype: torch.dtype,
        device: torch.device,
        feature_width: int | None = None,
    ) -> list[torch.Tensor]:
        """Materialize per-rank split lists into dense tensors."""
        result: list[torch.Tensor] = []
        for parts in rank_parts:
            if parts:
                result.append(torch.cat(parts, dim=0))
                continue

            if feature_width is None:
                result.append(torch.empty(0, dtype=dtype, device=device))
            else:
                result.append(
                    torch.empty((0, feature_width), dtype=dtype, device=device),
                )
        return result

    def _run_local_experts(
        self,
        dispatched: _DispatchedTokens,
        capacity: int,
        *,
        token_dtype: torch.dtype,
        device: torch.device,
    ) -> _ReturnPayload:
        """
        Run local experts and stage outputs for return all-to-all.

        This is the "compute" stage in the dispatch/compute/combine pipeline.
        """
        send_back_features_parts: list[list[torch.Tensor]] = [[] for _ in range(self.ep_size)]
        send_back_token_idx_parts: list[list[torch.Tensor]] = [[] for _ in range(self.ep_size)]
        send_back_slot_idx_parts: list[list[torch.Tensor]] = [[] for _ in range(self.ep_size)]

        dropped = 0
        local_counts = torch.zeros(self.experts_per_rank, dtype=torch.long, device=device)

        for local_expert_idx, expert in enumerate(self.experts):
            # selected: indices into [N_recv] assignments routed to this local expert.
            selected = (dispatched.recv_local_expert_idx == local_expert_idx).nonzero(
                as_tuple=False
            )
            if selected.numel() == 0:
                continue

            selected = selected.squeeze(-1)
            sort_weights = dispatched.recv_sort_weight[selected]

            if capacity > 0 and selected.numel() > capacity:
                # Capacity keeps highest routing weights for this expert on this rank.
                keep_rel = torch.topk(sort_weights, k=capacity, sorted=True).indices
                dropped += int(selected.numel() - keep_rel.numel())
                selected = selected[keep_rel]

            # Expert compute: [N_keep, H] -> [N_keep, H].
            expert_output = expert(dispatched.recv_tokens[selected])
            src_ranks = dispatched.recv_src_ranks[selected]
            src_token_idx = dispatched.recv_token_idx[selected]
            src_slot_idx = dispatched.recv_slot_idx[selected]

            for src_rank in range(self.ep_size):
                src_mask = src_ranks == src_rank
                if not bool(src_mask.any()):
                    continue
                # Stage outputs/metadata by source rank for return A2A.
                send_back_features_parts[src_rank].append(expert_output[src_mask])
                send_back_token_idx_parts[src_rank].append(src_token_idx[src_mask])
                send_back_slot_idx_parts[src_rank].append(src_slot_idx[src_mask])

            local_counts[local_expert_idx] = selected.numel()

        return _ReturnPayload(
            send_back_features=self._materialize_rank_splits(
                send_back_features_parts,
                dtype=token_dtype,
                device=device,
                feature_width=self.hidden_size,
            ),
            send_back_token_idx=self._materialize_rank_splits(
                send_back_token_idx_parts,
                dtype=torch.long,
                device=device,
            ),
            send_back_slot_idx=self._materialize_rank_splits(
                send_back_slot_idx_parts,
                dtype=torch.long,
                device=device,
            ),
            dropped=dropped,
            local_counts=local_counts,
            autograd_fallback=dispatched.recv_tokens,
        )

    def _combine_remote_outputs(
        self,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        payload: _ReturnPayload,
    ) -> tuple[torch.Tensor, int, int]:
        """
        Return expert outputs to source ranks and combine into final token outputs.

        This is the "combine" stage in the dispatch/compute/combine pipeline.
        """
        send_back_counts = [int(tensor.size(0)) for tensor in payload.send_back_features]
        recv_back_counts = self._exchange_counts(send_back_counts, device=tokens.device)

        # A2A #2 (return): send expert outputs back to original source ranks.
        # recv_back_features shape: [N_back, H].
        send_back_features = self._flatten_splits_for_autograd(
            payload.send_back_features,
            fallback=payload.autograd_fallback,
        )
        if self._comm_quant_enabled:
            send_back_features = _comm_quant_dequant(
                send_back_features,
                granularity=self._comm_quant_granularity,
                rounding_mode=self._comm_quant_rounding_mode,
                bits=8,
            )

        recv_back_features = self._all_to_all_autograd(
            send_tensor=send_back_features,
            send_counts=send_back_counts,
            recv_counts=recv_back_counts,
            output_shape_tail=(self.hidden_size,),
        )
        if self._comm_quant_enabled:
            recv_back_features = recv_back_features.to(dtype=tokens.dtype)

        send_back_token_idx_flat = self._flatten_splits(
            payload.send_back_token_idx,
            torch.long,
            tokens.device,
        )
        send_back_slot_idx_flat = self._flatten_splits(
            payload.send_back_slot_idx,
            torch.long,
            tokens.device,
        )

        recv_back_token_idx = self._all_to_all_metadata(
            send_tensor=send_back_token_idx_flat,
            send_counts=send_back_counts,
            recv_counts=recv_back_counts,
        )
        recv_back_slot_idx = self._all_to_all_metadata(
            send_tensor=send_back_slot_idx_flat,
            send_counts=send_back_counts,
            recv_counts=recv_back_counts,
        )

        # Combine: map each returned expert output to (token_idx, topk slot) and sum into [T, H].
        output_tokens = torch.zeros_like(tokens)
        if recv_back_features.numel() > 0:
            recv_weights = topk_weights[recv_back_token_idx, recv_back_slot_idx]
            recv_weighted = recv_back_features * recv_weights.unsqueeze(-1)
            output_tokens.index_add_(0, recv_back_token_idx, recv_weighted)

        local_total_assignments = float(topk_weights.numel())
        stats = torch.tensor(
            [float(payload.dropped), local_total_assignments],
            device=tokens.device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=self.ep_group)
        dropped = int(stats[0].item())
        total_assignments = int(stats[1].item())
        return output_tokens, dropped, total_assignments

    def _forward_multi_rank(
        self,
        tokens: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        capacity: int,
    ) -> tuple[torch.Tensor, int, int, torch.Tensor]:
        """Run EP multi-rank forward via dispatch -> local compute -> combine stages."""
        dispatched = self._dispatch_tokens(tokens, topk_indices, topk_weights)
        payload = self._run_local_experts(
            dispatched,
            capacity,
            token_dtype=tokens.dtype,
            device=tokens.device,
        )
        output_tokens, dropped, total_assignments = self._combine_remote_outputs(
            tokens,
            topk_weights,
            payload,
        )
        return output_tokens, dropped, total_assignments, payload.local_counts

    def _forward_single_rank(
        self,
        tokens: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        capacity: int,
    ) -> tuple[torch.Tensor, int, int, torch.Tensor]:
        """Fallback local execution path when ep_size == 1."""
        # Single-rank equivalent of dispatch -> compute -> combine without collectives.
        output_tokens = torch.zeros_like(tokens)
        dropped = 0
        local_counts = torch.zeros(self.experts_per_rank, dtype=torch.long, device=tokens.device)

        for local_expert_idx, expert in enumerate(self.experts):
            global_expert_idx = self.global_expert_start + local_expert_idx
            selected = (topk_indices == global_expert_idx).nonzero(as_tuple=False)
            if selected.numel() == 0:
                continue

            token_indices = selected[:, 0]
            slot_indices = selected[:, 1]
            sort_weights = topk_weights[token_indices, slot_indices].detach()

            if capacity > 0 and token_indices.numel() > capacity:
                keep_rel = torch.topk(sort_weights, k=capacity, sorted=True).indices
                token_indices = token_indices[keep_rel]
                slot_indices = slot_indices[keep_rel]
                dropped += selected.size(0) - capacity

            expert_out = expert(tokens[token_indices])
            weights = topk_weights[token_indices, slot_indices].unsqueeze(-1)
            output_tokens[token_indices] += expert_out * weights
            local_counts[local_expert_idx] = token_indices.numel()

        total_assignments = int(topk_indices.numel())
        return output_tokens, dropped, total_assignments, local_counts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply expert-parallel routed MoE to input tensors shaped `[batch, seq, hidden]`."""
        if x.dim() != 3:
            raise ValueError("x must have shape [batch, seq, hidden]")

        batch_size, seq_len, hidden_size = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError("hidden dimension mismatch")

        if self.ep_size > 1:
            self._require_distributed()

        # x: [B_ep, S, H] -> tokens: [T, H], where T = B_ep * S.
        tokens = x.reshape(-1, hidden_size)
        router_output = self.router(tokens)
        # Router outputs are per-token top-k assignments: [T, K].
        topk_indices = router_output.topk_indices
        topk_weights = router_output.topk_weights

        # capacity is per expert for this step, computed from global EP token count.
        capacity = self._compute_capacity(tokens.size(0), device=tokens.device)

        if self.ep_size == 1:
            output_tokens, dropped, total_assignments, local_counts = self._forward_single_rank(
                tokens=tokens,
                topk_indices=topk_indices,
                topk_weights=topk_weights,
                capacity=capacity,
            )
        else:
            output_tokens, dropped, total_assignments, local_counts = self._forward_multi_rank(
                tokens=tokens,
                topk_indices=topk_indices,
                topk_weights=topk_weights,
                capacity=capacity,
            )

        if self.shared_experts:
            shared_output = torch.zeros_like(tokens)
            for shared_expert in self.shared_experts:
                shared_output += shared_expert(tokens)
            output_tokens += shared_output / float(len(self.shared_experts))

        self._last_aux_loss = router_output.aux_loss
        self._last_dropped_fraction = float(dropped / max(total_assignments, 1))
        self._last_local_expert_counts = local_counts

        return output_tokens.view(batch_size, seq_len, hidden_size)
