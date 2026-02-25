"""
Reusable MoE building blocks for local and expert-parallel training.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
import torch.nn.functional as F

from src.layers import Dropout
from src.layers import Linear


@dataclass
class RouterOutput:
    """Router outputs for routed MoE layers."""

    topk_indices: torch.Tensor
    topk_weights: torch.Tensor
    aux_loss: torch.Tensor
    dropped_fraction: float


class ExpertMLP(nn.Module):
    """SwiGLU expert MLP used by routed MoE layers."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float) -> None:
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
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

        self.router = Linear(hidden_size, num_experts, bias=False)

    def _score_experts(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute routing scores from raw logits."""
        if self.scoring_func == "sigmoid":
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

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
        """Compute a simple load-balancing auxiliary loss."""
        if scores.numel() == 0:
            return torch.zeros((), dtype=scores.dtype, device=scores.device)

        if self.scoring_func == "sigmoid":
            normed_scores = scores / torch.clamp(scores.sum(dim=-1, keepdim=True), min=1e-9)
        else:
            normed_scores = scores

        assignments = F.one_hot(topk_indices, num_classes=self.num_experts).float()
        assignments = assignments.mean(dim=1)

        expert_fraction = assignments.mean(dim=0)
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
            topk_weights = torch.softmax(topk_weights, dim=-1)

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
                )
                for _ in range(num_experts)
            ]
        )
        self.shared_experts = nn.ModuleList(
            [
                ExpertMLP(
                    hidden_size=hidden_size,
                    intermediate_size=expert_intermediate_size,
                    dropout=dropout,
                )
                for _ in range(n_shared_experts)
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
                )
                for _ in range(self.experts_per_rank)
            ]
        )

        self.shared_experts = nn.ModuleList(
            [
                ExpertMLP(
                    hidden_size=hidden_size,
                    intermediate_size=expert_intermediate_size,
                    dropout=dropout,
                )
                for _ in range(n_shared_experts)
            ]
        )

        self._last_aux_loss = torch.zeros(())
        self._last_dropped_fraction = 0.0
        self._last_local_expert_counts = torch.zeros(self.experts_per_rank, dtype=torch.long)

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
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[int]]:
        """Build per-destination EP send tensors for token dispatch."""
        num_tokens = tokens.size(0)
        token_indices = torch.arange(num_tokens, device=tokens.device, dtype=torch.long)
        token_indices = token_indices.repeat_interleave(self.top_k)

        slot_indices = torch.arange(self.top_k, device=tokens.device, dtype=torch.long)
        slot_indices = slot_indices.unsqueeze(0).expand(num_tokens, -1).reshape(-1)

        expert_indices = topk_indices.reshape(-1)
        sort_weights = topk_weights.detach().reshape(-1)

        destination_ranks = expert_indices // self.experts_per_rank
        local_expert_indices = expert_indices % self.experts_per_rank

        dispatched_tokens = tokens[token_indices]

        send_tokens: list[torch.Tensor] = []
        send_token_idx: list[torch.Tensor] = []
        send_slot_idx: list[torch.Tensor] = []
        send_local_expert_idx: list[torch.Tensor] = []
        send_sort_weight: list[torch.Tensor] = []

        for dst_rank in range(self.ep_size):
            mask = destination_ranks == dst_rank
            send_tokens.append(dispatched_tokens[mask])
            send_token_idx.append(token_indices[mask])
            send_slot_idx.append(slot_indices[mask])
            send_local_expert_idx.append(local_expert_indices[mask])
            send_sort_weight.append(sort_weights[mask])

        send_counts = [int(tensor.size(0)) for tensor in send_tokens]
        return (
            send_tokens,
            send_token_idx,
            send_slot_idx,
            send_local_expert_idx,
            send_sort_weight,
            send_counts,
        )

    def _exchange_counts(self, send_counts: list[int], device: torch.device) -> list[int]:
        """Exchange variable-size split counts across EP ranks."""
        send_counts_tensor = torch.tensor(send_counts, device=device, dtype=torch.long)
        recv_counts_tensor = torch.zeros_like(send_counts_tensor)
        dist.all_to_all_single(recv_counts_tensor, send_counts_tensor, group=self.ep_group)
        return [int(value) for value in recv_counts_tensor.tolist()]

    def _flatten_splits(self, tensors: list[torch.Tensor], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Concatenate list of split tensors, returning an empty tensor when all are empty."""
        if not tensors:
            return torch.empty(0, dtype=dtype, device=device)
        non_empty = [tensor for tensor in tensors if tensor.numel() > 0]
        if not non_empty:
            return torch.empty(0, dtype=dtype, device=device)
        return torch.cat(non_empty, dim=0)

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

    def _forward_single_rank(
        self,
        tokens: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        capacity: int,
    ) -> tuple[torch.Tensor, int, int, torch.Tensor]:
        """Fallback local execution path when ep_size == 1."""
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
        """Apply expert-parallel routed MoE to input tensors shaped [B, S, H]."""
        if x.dim() != 3:
            raise ValueError("x must have shape [batch, seq, hidden]")

        batch_size, seq_len, hidden_size = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError("hidden dimension mismatch")

        if self.ep_size > 1:
            self._require_distributed()

        tokens = x.reshape(-1, hidden_size)
        router_output = self.router(tokens)
        topk_indices = router_output.topk_indices
        topk_weights = router_output.topk_weights

        capacity = self._compute_capacity(tokens.size(0), device=tokens.device)

        if self.ep_size == 1:
            output_tokens, dropped, total_assignments, local_counts = self._forward_single_rank(
                tokens=tokens,
                topk_indices=topk_indices,
                topk_weights=topk_weights,
                capacity=capacity,
            )
        else:
            (
                send_tokens,
                send_token_idx,
                send_slot_idx,
                send_local_expert_idx,
                send_sort_weight,
                send_counts,
            ) = self._build_send_partitions(tokens, topk_indices, topk_weights)

            recv_counts = self._exchange_counts(send_counts, device=tokens.device)
            send_token_flat = self._flatten_splits(send_tokens, tokens.dtype, tokens.device)
            recv_tokens = self._all_to_all_autograd(
                send_tensor=send_token_flat,
                send_counts=send_counts,
                recv_counts=recv_counts,
                output_shape_tail=(hidden_size,),
            )

            send_token_idx_flat = self._flatten_splits(send_token_idx, torch.long, tokens.device)
            send_slot_idx_flat = self._flatten_splits(send_slot_idx, torch.long, tokens.device)
            send_local_expert_flat = self._flatten_splits(send_local_expert_idx, torch.long, tokens.device)
            send_sort_weight_flat = self._flatten_splits(send_sort_weight, tokens.dtype, tokens.device)

            recv_token_idx_flat = self._all_to_all_metadata(
                send_tensor=send_token_idx_flat,
                send_counts=send_counts,
                recv_counts=recv_counts,
            )
            recv_slot_idx_flat = self._all_to_all_metadata(
                send_tensor=send_slot_idx_flat,
                send_counts=send_counts,
                recv_counts=recv_counts,
            )
            recv_local_expert_flat = self._all_to_all_metadata(
                send_tensor=send_local_expert_flat,
                send_counts=send_counts,
                recv_counts=recv_counts,
            )
            recv_sort_weight_flat = self._all_to_all_metadata(
                send_tensor=send_sort_weight_flat,
                send_counts=send_counts,
                recv_counts=recv_counts,
            )

            src_rank_ids = torch.arange(self.ep_size, dtype=torch.long, device=tokens.device)
            recv_counts_tensor = torch.tensor(recv_counts, dtype=torch.long, device=tokens.device)
            recv_src_ranks = torch.repeat_interleave(src_rank_ids, recv_counts_tensor)

            send_back_features_parts: list[list[torch.Tensor]] = [[] for _ in range(self.ep_size)]
            send_back_token_idx_parts: list[list[torch.Tensor]] = [[] for _ in range(self.ep_size)]
            send_back_slot_idx_parts: list[list[torch.Tensor]] = [[] for _ in range(self.ep_size)]

            dropped = 0
            local_counts = torch.zeros(self.experts_per_rank, dtype=torch.long, device=tokens.device)

            for local_expert_idx, expert in enumerate(self.experts):
                selected = (recv_local_expert_flat == local_expert_idx).nonzero(as_tuple=False)
                if selected.numel() == 0:
                    continue

                selected = selected.squeeze(-1)
                sort_weights = recv_sort_weight_flat[selected]

                if capacity > 0 and selected.numel() > capacity:
                    keep_rel = torch.topk(sort_weights, k=capacity, sorted=True).indices
                    selected = selected[keep_rel]
                    dropped += sort_weights.numel() - capacity

                expert_input = recv_tokens[selected]
                expert_output = expert(expert_input)

                src_ranks = recv_src_ranks[selected]
                src_token_idx = recv_token_idx_flat[selected]
                src_slot_idx = recv_slot_idx_flat[selected]

                for src_rank in range(self.ep_size):
                    src_mask = src_ranks == src_rank
                    if not bool(src_mask.any()):
                        continue
                    send_back_features_parts[src_rank].append(expert_output[src_mask])
                    send_back_token_idx_parts[src_rank].append(src_token_idx[src_mask])
                    send_back_slot_idx_parts[src_rank].append(src_slot_idx[src_mask])

                local_counts[local_expert_idx] = selected.numel()

            send_back_features = [
                torch.cat(parts, dim=0)
                if parts
                else torch.empty((0, hidden_size), dtype=tokens.dtype, device=tokens.device)
                for parts in send_back_features_parts
            ]
            send_back_token_idx = [
                torch.cat(parts, dim=0)
                if parts
                else torch.empty(0, dtype=torch.long, device=tokens.device)
                for parts in send_back_token_idx_parts
            ]
            send_back_slot_idx = [
                torch.cat(parts, dim=0)
                if parts
                else torch.empty(0, dtype=torch.long, device=tokens.device)
                for parts in send_back_slot_idx_parts
            ]

            send_back_counts = [int(tensor.size(0)) for tensor in send_back_features]
            recv_back_counts = self._exchange_counts(send_back_counts, device=tokens.device)
            non_empty_back = [tensor for tensor in send_back_features if tensor.numel() > 0]
            if non_empty_back:
                send_back_feature_flat = torch.cat(non_empty_back, dim=0)
            else:
                # Keep autograd graph connectivity even for zero-token ranks.
                send_back_feature_flat = recv_tokens[:0]
            recv_back_features = self._all_to_all_autograd(
                send_tensor=send_back_feature_flat,
                send_counts=send_back_counts,
                recv_counts=recv_back_counts,
                output_shape_tail=(hidden_size,),
            )

            send_back_token_idx_flat = self._flatten_splits(send_back_token_idx, torch.long, tokens.device)
            send_back_slot_idx_flat = self._flatten_splits(send_back_slot_idx, torch.long, tokens.device)

            recv_back_token_idx_flat = self._all_to_all_metadata(
                send_tensor=send_back_token_idx_flat,
                send_counts=send_back_counts,
                recv_counts=recv_back_counts,
            )
            recv_back_slot_idx_flat = self._all_to_all_metadata(
                send_tensor=send_back_slot_idx_flat,
                send_counts=send_back_counts,
                recv_counts=recv_back_counts,
            )

            output_tokens = torch.zeros_like(tokens)
            if recv_back_features.numel() > 0:
                recv_weights = topk_weights[recv_back_token_idx_flat, recv_back_slot_idx_flat]
                recv_weighted = recv_back_features * recv_weights.unsqueeze(-1)
                output_tokens.index_add_(0, recv_back_token_idx_flat, recv_weighted)

            local_total_assignments = float(topk_indices.numel())
            stats = torch.tensor([float(dropped), local_total_assignments], device=tokens.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=self.ep_group)
            dropped = int(stats[0].item())
            total_assignments = int(stats[1].item())

        if self.shared_experts:
            shared_output = torch.zeros_like(tokens)
            for shared_expert in self.shared_experts:
                shared_output += shared_expert(tokens)
            output_tokens += shared_output / float(len(self.shared_experts))

        self._last_aux_loss = router_output.aux_loss
        self._last_dropped_fraction = float(dropped / max(total_assignments, 1))
        self._last_local_expert_counts = local_counts

        return output_tokens.view(batch_size, seq_len, hidden_size)
