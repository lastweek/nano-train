#!/usr/bin/env python3
"""
Canonical TP+EP+DP tutorial script (experts run with TP=1 only).

Primary tutorial modes:
1) single:     world=1, tp=1, ep=1
2) ep_only:    tp=1, ep>1, dp>=1
3) tp_ep_dp:   tp>1 and ep>1 and dp>1

Model used:
    Small DeepSeek-style LM by default (5 decoder layers, 8 routed experts,
    top-k=2 routing), with optional TP/EP behavior enabled via model context.

Note:
    Any valid factorization with world_size = tp_size * ep_size * dp_size can run.
    The three modes above are the intended learning targets.

Run examples:
    # single rank
    python3 examples/ep.py --tp_size 1 --ep_size 1 --max_steps 2 --epochs 1

    # EP-only (world=2, tp=1, ep=2, dp=1)
    python3 examples/launch.py --world-size 2 --backend gloo \
        --script examples/ep.py --script-args \
        --tp_size 1 --ep_size 2 --max_steps 2 --epochs 1

    # TP+EP+DP (world=8, tp=2, ep=2, dp=2)
    python3 examples/launch.py --world-size 8 --backend gloo \
        --script examples/ep.py --script-args \
        --tp_size 2 --ep_size 2 --max_steps 1 --epochs 1

Core learning reference:
    docs/ep_tp_dp_communication.md
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import sys
from typing import Optional
from typing import Set

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler

# Add parent directory to path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import create_dataloader
from src.distributed.device import get_backend
from src.distributed.device import get_device
from src.layers import ColumnParallelLinear
from src.layers import RowParallelLinear
from src.logging import get_logger
from src.logging import setup_logging
from src.models.deepseek import DeepSeekModel
from src.models.deepseek import DeepSeekModelConfig
from src.models.deepseek import DeepSeekParallelContext
from src.models.moe import ExpertParallelMoE
from src.models.moe import LocalRoutedMoE


logger = get_logger("ep")


@dataclass
class ParallelSetup:
    """Computed distributed topology and process-group handles."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    tp_size: int
    ep_size: int
    dp_size: int
    tp_rank: int
    ep_rank: int
    dp_rank: int
    tp_group: Optional[object]
    ep_group: Optional[object]
    dp_group: Optional[object]
    tp_replica_group: Optional[object]
    ep_replica_group: Optional[object]
    tp_group_table: list[list[int]]
    ep_group_table: list[list[int]]
    dp_group_table: list[list[int]]


class DummyTokenDataset(Dataset):
    """Deterministic token dataset for causal LM communication demos."""

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(num_samples, seq_len),
            generator=generator,
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[index]}


def build_tiny_deepseek_config(args: argparse.Namespace) -> DeepSeekModelConfig:
    """Build a small DeepSeek config for EP/TP/DP learning runs."""
    return DeepSeekModelConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=max(args.max_position_embeddings, args.seq_len),
        q_lora_rank=args.q_lora_rank,
        kv_lora_rank=args.kv_lora_rank,
        qk_nope_head_dim=args.qk_nope_head_dim,
        qk_rope_head_dim=args.qk_rope_head_dim,
        v_head_dim=args.v_head_dim,
        attention_dropout=args.dropout,
        dropout=args.dropout,
        moe_intermediate_size=args.moe_intermediate_size,
        n_routed_experts=args.num_experts,
        n_shared_experts=args.n_shared_experts,
        num_experts_per_tok=args.top_k,
        first_k_dense_replace=args.first_k_dense_replace,
        moe_layer_freq=args.moe_layer_freq,
        scoring_func="sigmoid",
        n_group=args.n_group,
        topk_group=args.topk_group,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for EP tutorial training."""
    parser = argparse.ArgumentParser(description="Canonical TP+EP+DP tutorial script")
    parser.add_argument("--tp_size", type=int, default=1, help="TP size")
    parser.add_argument("--ep_size", type=int, default=1, help="EP size")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--intermediate_size", type=int, default=256)
    parser.add_argument("--moe_intermediate_size", type=int, default=128)
    parser.add_argument("--q_lora_rank", type=int, default=64)
    parser.add_argument("--kv_lora_rank", type=int, default=32)
    parser.add_argument("--qk_nope_head_dim", type=int, default=8)
    parser.add_argument("--qk_rope_head_dim", type=int, default=8)
    parser.add_argument("--v_head_dim", type=int, default=8)
    parser.add_argument("--max_position_embeddings", type=int, default=256)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--n_group", type=int, default=4)
    parser.add_argument("--topk_group", type=int, default=2)
    parser.add_argument("--n_shared_experts", type=int, default=1)
    parser.add_argument("--first_k_dense_replace", type=int, default=1)
    parser.add_argument("--moe_layer_freq", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--capacity_factor", type=float, default=1.0)
    parser.add_argument("--aux_loss_coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def setup_process_logging(rank: int) -> None:
    """Configure process-local logging using shared logging utilities."""
    level = "INFO" if rank == 0 else "WARNING"
    setup_logging(log_level=level, use_colors=False)


def init_parallel(tp_size: int, ep_size: int) -> ParallelSetup:
    """
    Initialize distributed process groups for a 3D (DP, TP, EP) layout.

    Rank mapping (EP fastest):
        rank = dp_rank * (tp_size * ep_size) + tp_rank * ep_size + ep_rank
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    if tp_size < 1 or ep_size < 1:
        raise ValueError("tp_size and ep_size must be >= 1")

    model_parallel = tp_size * ep_size
    if world_size % model_parallel != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by tp_size*ep_size ({model_parallel})"
        )

    backend = get_backend()
    device_type = "cuda" if backend == "nccl" else "cpu"
    device = get_device(device_type, local_rank)
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend=backend, world_size=world_size, rank=rank)

    dp_size = world_size // model_parallel
    dp_rank = rank // model_parallel
    rem = rank % model_parallel
    tp_rank = rem // ep_size
    ep_rank = rem % ep_size

    tp_group = None
    ep_group = None
    dp_group = None
    tp_replica_group = None
    ep_replica_group = None

    tp_group_table: list[list[int]] = []
    ep_group_table: list[list[int]] = []
    dp_group_table: list[list[int]] = []

    # TP groups: fixed (dp, ep), varying tp.
    if distributed and tp_size > 1:
        for dp_idx in range(dp_size):
            base = dp_idx * model_parallel
            for ep_idx in range(ep_size):
                ranks = [base + tp_idx * ep_size + ep_idx for tp_idx in range(tp_size)]
                tp_group_table.append(ranks)
                group = dist.new_group(ranks)
                if dp_idx == dp_rank and ep_idx == ep_rank:
                    tp_group = group

    # EP groups: fixed (dp, tp), varying ep.
    if distributed and ep_size > 1:
        for dp_idx in range(dp_size):
            base = dp_idx * model_parallel
            for tp_idx in range(tp_size):
                ranks = [base + tp_idx * ep_size + ep_idx for ep_idx in range(ep_size)]
                ep_group_table.append(ranks)
                group = dist.new_group(ranks)
                if dp_idx == dp_rank and tp_idx == tp_rank:
                    ep_group = group

    # DP groups: fixed (tp, ep), varying dp.
    if distributed and dp_size > 1:
        for tp_idx in range(tp_size):
            for ep_idx in range(ep_size):
                ranks = [
                    dp_idx * model_parallel + tp_idx * ep_size + ep_idx
                    for dp_idx in range(dp_size)
                ]
                dp_group_table.append(ranks)
                group = dist.new_group(ranks)
                if tp_idx == tp_rank and ep_idx == ep_rank:
                    dp_group = group

    # TP-shard replica groups: fixed tp, varying (dp, ep).
    if distributed:
        for tp_idx in range(tp_size):
            ranks = [
                dp_idx * model_parallel + tp_idx * ep_size + ep_idx
                for dp_idx in range(dp_size)
                for ep_idx in range(ep_size)
            ]
            group = dist.new_group(ranks)
            if tp_idx == tp_rank:
                tp_replica_group = group

        # EP-shard replica groups: fixed ep, varying (dp, tp).
        for ep_idx in range(ep_size):
            ranks = [
                dp_idx * model_parallel + tp_idx * ep_size + ep_idx
                for dp_idx in range(dp_size)
                for tp_idx in range(tp_size)
            ]
            group = dist.new_group(ranks)
            if ep_idx == ep_rank:
                ep_replica_group = group

    return ParallelSetup(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        tp_size=tp_size,
        ep_size=ep_size,
        dp_size=dp_size,
        tp_rank=tp_rank,
        ep_rank=ep_rank,
        dp_rank=dp_rank,
        tp_group=tp_group,
        ep_group=ep_group,
        dp_group=dp_group,
        tp_replica_group=tp_replica_group,
        ep_replica_group=ep_replica_group,
        tp_group_table=tp_group_table,
        ep_group_table=ep_group_table,
        dp_group_table=dp_group_table,
    )


def validate_args(args: argparse.Namespace, world_size: int) -> None:
    """Validate args that depend on global world size."""
    if args.tp_size * args.ep_size > world_size:
        raise ValueError("tp_size * ep_size cannot exceed world_size")

    if args.batch_size % args.ep_size != 0:
        raise ValueError("batch_size must be divisible by ep_size for balanced EP token shards")

    if args.intermediate_size % max(1, args.tp_size) != 0:
        raise ValueError("intermediate_size must be divisible by tp_size")

    if args.num_experts % args.ep_size != 0:
        raise ValueError("num_experts must be divisible by ep_size")

    if args.seq_len < 2:
        raise ValueError("seq_len must be >= 2 for causal LM loss")


def infer_mode(world_size: int, tp_size: int, ep_size: int) -> str:
    """Infer which learning mode current topology corresponds to."""
    if world_size == 1:
        return "single"
    if tp_size == 1 and ep_size > 1:
        return "ep_only"
    return "tp_ep_dp"


def log_parallel_topology(parallel: ParallelSetup, mode: str) -> None:
    """Log rank topology and process groups from rank 0."""
    if parallel.rank != 0:
        return

    logger.info("=" * 72)
    logger.info("Canonical TP+EP+DP Tutorial")
    logger.info("=" * 72)
    logger.info("Mode: %s", mode)
    logger.info(
        "World Size: %d | TP Size: %d | EP Size: %d | DP Size: %d",
        parallel.world_size,
        parallel.tp_size,
        parallel.ep_size,
        parallel.dp_size,
    )
    logger.info(
        "Backend: %s | Device: %s",
        dist.get_backend() if parallel.world_size > 1 else "none",
        parallel.device,
    )
    logger.info("TP groups: %s", parallel.tp_group_table if parallel.tp_group_table else "n/a")
    logger.info("EP groups: %s", parallel.ep_group_table if parallel.ep_group_table else "n/a")
    logger.info("DP groups: %s", parallel.dp_group_table if parallel.dp_group_table else "n/a")
    logger.info("=" * 72)


def collect_tp_sharded_param_ids(model: nn.Module) -> Set[int]:
    """Collect parameter ids that are TP-sharded by Column/Row parallel layers."""
    sharded: Set[int] = set()
    for module in model.modules():
        if isinstance(module, ColumnParallelLinear):
            for param in module.parameters(recurse=False):
                sharded.add(id(param))
        elif isinstance(module, RowParallelLinear):
            sharded.add(id(module.weight))
    return sharded


def collect_ep_sharded_param_ids(model: nn.Module) -> Set[int]:
    """Collect parameter ids owned by local EP experts (EP-sharded params)."""
    sharded: Set[int] = set()
    for module in model.modules():
        if isinstance(module, ExpertParallelMoE):
            for expert in module.experts:
                for param in expert.parameters():
                    sharded.add(id(param))
    return sharded


def synchronize_initial_parameters(
    model: nn.Module,
    tp_sharded_param_ids: Set[int],
    ep_sharded_param_ids: Set[int],
    world_size: int,
    tp_size: int,
    ep_size: int,
    tp_rank: int,
    ep_rank: int,
    tp_replica_group,
    ep_replica_group,
) -> None:
    """Synchronize parameters so TP/EP shards and replicas start from canonical weights."""
    if world_size == 1:
        return

    tp_src_rank = tp_rank * ep_size
    ep_src_rank = ep_rank

    for param in model.parameters():
        param_id = id(param)
        if param_id in tp_sharded_param_ids:
            if tp_size > 1:
                dist.broadcast(param.data, src=tp_src_rank, group=tp_replica_group)
        elif param_id in ep_sharded_param_ids:
            if ep_size > 1:
                dist.broadcast(param.data, src=ep_src_rank, group=ep_replica_group)
        else:
            dist.broadcast(param.data, src=0)

    for buffer in model.buffers():
        dist.broadcast(buffer.data, src=0)


def synchronize_gradients(
    model: nn.Module,
    ep_sharded_param_ids: Set[int],
    ep_size: int,
    dp_size: int,
    ep_group,
    dp_group,
) -> None:
    """
    Synchronize gradients for TP+EP+DP.

    Rules:
    - EP-sharded expert params: DP all-reduce only, then divide by ep_size to account for
      local mean-loss scaling on EP token shards.
    - Non-EP params: EP all-reduce(avg) then DP all-reduce(avg).
    - TP collectives stay inside TP layers and are not repeated here.
    """
    for param in model.parameters():
        is_ep_sharded = id(param) in ep_sharded_param_ids
        needs_ep_reduce = ep_size > 1 and not is_ep_sharded
        needs_dp_reduce = dp_size > 1
        needs_ep_scale = ep_size > 1 and is_ep_sharded

        grad = param.grad
        had_local_grad = grad is not None

        # Keep collective order identical across ranks even when local grads are missing.
        if grad is None and (needs_ep_reduce or needs_dp_reduce):
            grad = torch.zeros_like(param)
        elif grad is None:
            continue

        if needs_ep_reduce:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ep_group)
            grad.div_(ep_size)

        if needs_dp_reduce:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=dp_group)
            grad.div_(dp_size)

        if needs_ep_scale:
            grad.div_(ep_size)

        if had_local_grad or needs_ep_reduce or needs_dp_reduce:
            param.grad = grad


def create_data_loader(
    dataset: Dataset,
    batch_size: int,
    dp_size: int,
    dp_rank: int,
    seed: int,
):
    """Create a dataloader with DP-aware sharding, reusing src.dataset.create_dataloader."""
    sampler: Optional[DistributedSampler] = None
    if dp_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dp_size,
            rank=dp_rank,
            shuffle=True,
            seed=seed,
        )

    loader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        sampler=sampler,
    )
    return loader, sampler


def split_batch_for_ep(
    input_ids: torch.Tensor,
    ep_rank: int,
    ep_size: int,
) -> torch.Tensor:
    """Split a DP-local batch across EP ranks."""
    if ep_size == 1:
        return input_ids

    input_chunks = torch.chunk(input_ids, ep_size, dim=0)

    return input_chunks[ep_rank]


def gather_moe_metrics(model: nn.Module, device: torch.device) -> tuple[torch.Tensor, float]:
    """Collect auxiliary MoE loss and dropped fraction from all MoE modules."""
    aux_terms: list[torch.Tensor] = []
    drop_terms: list[float] = []

    for module in model.modules():
        if isinstance(module, (ExpertParallelMoE, LocalRoutedMoE)):
            aux_terms.append(module.last_aux_loss)
            drop_terms.append(module.last_dropped_fraction)

    if aux_terms:
        aux_loss = torch.stack([term.reshape(()) for term in aux_terms]).sum()
    else:
        aux_loss = torch.zeros((), device=device)

    avg_drop = float(sum(drop_terms) / len(drop_terms)) if drop_terms else 0.0
    return aux_loss, avg_drop


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    ep_rank: int,
    ep_size: int,
    dp_size: int,
    aux_loss_coef: float,
    ep_sharded_param_ids: Set[int],
    ep_group,
    dp_group,
) -> tuple[float, float, float, float]:
    """One TP+EP+DP training step with explicit communication."""
    input_ids = batch["input_ids"].to(device)
    local_input_ids = split_batch_for_ep(input_ids=input_ids, ep_rank=ep_rank, ep_size=ep_size)

    optimizer.zero_grad(set_to_none=True)

    logits = model(local_input_ids)
    if local_input_ids.size(0) == 0 or local_input_ids.size(1) < 2:
        task_loss = logits.sum() * 0.0
    else:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = local_input_ids[:, 1:].contiguous()
        task_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )

    moe_aux_loss, drop_fraction = gather_moe_metrics(model, device=device)
    loss = task_loss + (aux_loss_coef * moe_aux_loss)
    loss.backward()

    synchronize_gradients(
        model=model,
        ep_sharded_param_ids=ep_sharded_param_ids,
        ep_size=ep_size,
        dp_size=dp_size,
        ep_group=ep_group,
        dp_group=dp_group,
    )

    optimizer.step()
    return (
        float(task_loss.item()),
        float(moe_aux_loss.item()),
        float(loss.item()),
        drop_fraction,
    )


def aggregate_metrics(
    task_sum: float,
    aux_sum: float,
    loss_sum: float,
    drop_sum: float,
    count: int,
    device: torch.device,
    world_size: int,
) -> tuple[float, float, float, float]:
    """Aggregate average metrics across all ranks for logging."""
    if world_size == 1:
        denom = max(1, count)
        return (
            task_sum / denom,
            aux_sum / denom,
            loss_sum / denom,
            drop_sum / denom,
        )

    stats = torch.tensor(
        [task_sum, aux_sum, loss_sum, drop_sum, float(count)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    denom = max(1.0, stats[4].item())
    return (
        float(stats[0].item() / denom),
        float(stats[1].item() / denom),
        float(stats[2].item() / denom),
        float(stats[3].item() / denom),
    )


def main() -> None:
    """Run canonical EP tutorial training with optional TP and DP."""
    args = parse_args()

    rank_pre = int(os.environ.get("RANK", "0"))
    setup_process_logging(rank_pre)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    validate_args(args, world_size)

    parallel = init_parallel(tp_size=args.tp_size, ep_size=args.ep_size)
    mode = infer_mode(parallel.world_size, parallel.tp_size, parallel.ep_size)
    log_parallel_topology(parallel, mode)

    # Seed by (tp_rank, ep_rank) so TP and EP shards initialize differently before sync.
    seed_offset = parallel.tp_rank * 1009 + parallel.ep_rank
    torch.manual_seed(args.seed + seed_offset)
    if parallel.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + seed_offset)

    model_config = build_tiny_deepseek_config(args)
    parallel_context = DeepSeekParallelContext(
        tp_rank=parallel.tp_rank,
        tp_size=parallel.tp_size,
        tp_group=parallel.tp_group,
        ep_rank=parallel.ep_rank,
        ep_size=parallel.ep_size,
        ep_group=parallel.ep_group,
        capacity_factor=args.capacity_factor,
        expert_tp_size=1,
    )
    model = DeepSeekModel(model_config, parallel_context=parallel_context).to(parallel.device)

    tp_sharded_param_ids = collect_tp_sharded_param_ids(model)
    ep_sharded_param_ids = collect_ep_sharded_param_ids(model)

    synchronize_initial_parameters(
        model=model,
        tp_sharded_param_ids=tp_sharded_param_ids,
        ep_sharded_param_ids=ep_sharded_param_ids,
        world_size=parallel.world_size,
        tp_size=parallel.tp_size,
        ep_size=parallel.ep_size,
        tp_rank=parallel.tp_rank,
        ep_rank=parallel.ep_rank,
        tp_replica_group=parallel.tp_replica_group,
        ep_replica_group=parallel.ep_replica_group,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    dataset = DummyTokenDataset(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    train_loader, sampler = create_data_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        dp_size=parallel.dp_size,
        dp_rank=parallel.dp_rank,
        seed=args.seed,
    )

    if parallel.rank == 0:
        local_params = sum(parameter.numel() for parameter in model.parameters())
        logger.info("Local params: %d", local_params)
        logger.info(
            "DeepSeek tiny config: layers=%d hidden=%d experts=%d top_k=%d seq_len=%d",
            args.num_layers,
            args.hidden_size,
            args.num_experts,
            args.top_k,
            args.seq_len,
        )
        logger.info(
            "Model parallel context: tp_rank=%d/%d ep_rank=%d/%d",
            parallel.tp_rank,
            parallel.tp_size,
            parallel.ep_rank,
            parallel.ep_size,
        )
        logger.info("Starting training: epochs=%d, max_steps=%d", args.epochs, args.max_steps)

    model.train()
    step = 0

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        task_sum = 0.0
        aux_sum = 0.0
        loss_sum = 0.0
        drop_sum = 0.0
        batch_count = 0

        for batch in train_loader:
            if step >= args.max_steps:
                break

            task_loss, aux_loss, total_loss, drop_fraction = train_step(
                model=model,
                optimizer=optimizer,
                batch=batch,
                device=parallel.device,
                ep_rank=parallel.ep_rank,
                ep_size=parallel.ep_size,
                dp_size=parallel.dp_size,
                aux_loss_coef=args.aux_loss_coef,
                ep_sharded_param_ids=ep_sharded_param_ids,
                ep_group=parallel.ep_group,
                dp_group=parallel.dp_group,
            )

            task_sum += task_loss
            aux_sum += aux_loss
            loss_sum += total_loss
            drop_sum += drop_fraction
            batch_count += 1

            if parallel.rank == 0 and step % args.log_every == 0:
                logger.info(
                    "step=%d task=%.6f aux=%.6f total=%.6f drop=%.4f",
                    step,
                    task_loss,
                    aux_loss,
                    total_loss,
                    drop_fraction,
                )

            step += 1

        if batch_count > 0:
            avg_task, avg_aux, avg_total, avg_drop = aggregate_metrics(
                task_sum=task_sum,
                aux_sum=aux_sum,
                loss_sum=loss_sum,
                drop_sum=drop_sum,
                count=batch_count,
                device=parallel.device,
                world_size=parallel.world_size,
            )
            if parallel.rank == 0:
                logger.info(
                    "epoch=%d avg_task=%.6f avg_aux=%.6f avg_total=%.6f avg_drop=%.4f",
                    epoch + 1,
                    avg_task,
                    avg_aux,
                    avg_total,
                    avg_drop,
                )

        if step >= args.max_steps:
            break

    if parallel.rank == 0:
        logger.info("Training completed")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
