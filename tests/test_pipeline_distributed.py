"""Distributed smoke tests for PP and PP+EP communication paths."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.distributed.topology import initialize_model_parallel
from src.models.deepseek import DeepSeekModel
from src.models.deepseek import DeepSeekModelConfig
from src.models.deepseek import DeepSeekParallelContext


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _tiny_config() -> DeepSeekModelConfig:
    return DeepSeekModelConfig(
        vocab_size=64,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        q_lora_rank=32,
        kv_lora_rank=24,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        intermediate_size=128,
        moe_intermediate_size=96,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        n_group=2,
        topk_group=1,
        max_position_embeddings=64,
        dropout=0.0,
        attention_dropout=0.0,
    )


def _pp_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    setup = initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        expert_model_parallel_size=1,
    )
    cfg = _tiny_config()
    ctx = DeepSeekParallelContext(
        tensor_model_parallel_rank=setup.tensor_model_parallel_rank,
        tensor_model_parallel_size=setup.tensor_model_parallel_size,
        tensor_model_parallel_group=setup.tensor_model_parallel_group,
        expert_model_parallel_rank=setup.expert_model_parallel_rank,
        expert_model_parallel_size=setup.expert_model_parallel_size,
        expert_model_parallel_group=setup.expert_model_parallel_group,
        pipeline_model_parallel_rank=setup.pipeline_model_parallel_rank,
        pipeline_model_parallel_size=setup.pipeline_model_parallel_size,
        pipeline_model_parallel_group=setup.pipeline_model_parallel_group,
    )
    model = DeepSeekModel(cfg, parallel_context=ctx)

    batch_size = 2
    seq_len = 16

    if model.is_first_pp_stage:
        input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        hidden = model.forward_stage(input_ids=input_ids, hidden_states=None, attention_mask=None)
        next_rank = setup.rank_from_coords(
            data_parallel_rank=setup.data_parallel_rank,
            pipeline_model_parallel_rank=1,
            tensor_model_parallel_rank=setup.tensor_model_parallel_rank,
            expert_model_parallel_rank=setup.expert_model_parallel_rank,
            context_parallel_rank=setup.context_parallel_rank,
        )
        dist.send(hidden.contiguous(), dst=next_rank, tag=101)

        grad_hidden = torch.empty_like(hidden)
        dist.recv(grad_hidden, src=next_rank, tag=102)
        torch.autograd.backward(hidden, grad_hidden)
    else:
        prev_rank = setup.rank_from_coords(
            data_parallel_rank=setup.data_parallel_rank,
            pipeline_model_parallel_rank=0,
            tensor_model_parallel_rank=setup.tensor_model_parallel_rank,
            expert_model_parallel_rank=setup.expert_model_parallel_rank,
            context_parallel_rank=setup.context_parallel_rank,
        )
        hidden = torch.empty((batch_size, seq_len, cfg.hidden_size), requires_grad=True)
        dist.recv(hidden, src=prev_rank, tag=101)

        logits = model.forward_stage(input_ids=None, hidden_states=hidden, attention_mask=None)
        loss = logits.mean()
        loss.backward()
        if hidden.grad is None:
            raise RuntimeError("Expected hidden.grad on last PP stage")
        dist.send(hidden.grad.contiguous(), dst=prev_rank, tag=102)

    has_grad = torch.tensor(
        1 if any(param.grad is not None for param in model.parameters()) else 0,
        dtype=torch.long,
    )
    dist.all_reduce(has_grad, op=dist.ReduceOp.SUM)
    assert has_grad.item() == world_size

    dist.destroy_process_group()


def _pp_ep_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    setup = initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        expert_model_parallel_size=2,
    )

    # PP p2p smoke: forward send/recv between pp neighbors in the same (dp,tp,ep) chain.
    if setup.pipeline_model_parallel_rank == 0:
        dst = setup.rank_from_coords(
            data_parallel_rank=setup.data_parallel_rank,
            pipeline_model_parallel_rank=1,
            tensor_model_parallel_rank=setup.tensor_model_parallel_rank,
            expert_model_parallel_rank=setup.expert_model_parallel_rank,
            context_parallel_rank=setup.context_parallel_rank,
        )
        payload = torch.tensor([float(rank)])
        dist.send(payload, dst=dst, tag=201)
    else:
        src = setup.rank_from_coords(
            data_parallel_rank=setup.data_parallel_rank,
            pipeline_model_parallel_rank=0,
            tensor_model_parallel_rank=setup.tensor_model_parallel_rank,
            expert_model_parallel_rank=setup.expert_model_parallel_rank,
            context_parallel_rank=setup.context_parallel_rank,
        )
        recv = torch.empty(1)
        dist.recv(recv, src=src, tag=201)

    # Expert model parallel collective smoke: fixed (dp,pp,tp), varying expert-rank.
    ep_tensor = torch.ones(1)
    dist.all_reduce(
        ep_tensor,
        op=dist.ReduceOp.SUM,
        group=setup.expert_model_parallel_group,
    )
    assert float(ep_tensor.item()) == float(setup.expert_model_parallel_size)

    dist.destroy_process_group()


def test_pipeline_distributed_smoke_world2() -> None:
    """Pipeline forward/backward should run on world=2 with pp=2."""
    world_size = 2
    port = _free_port()
    mp.spawn(_pp_worker, args=(world_size, port), nprocs=world_size, join=True)


def test_pipeline_ep_distributed_smoke_world4() -> None:
    """PP+EP communication should run on world=4 with pp=2, ep=2."""
    world_size = 4
    port = _free_port()
    mp.spawn(_pp_ep_worker, args=(world_size, port), nprocs=world_size, join=True)
