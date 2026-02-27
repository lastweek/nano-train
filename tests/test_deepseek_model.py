"""Smoke tests for DeepSeek-style model components."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.layers import ColumnParallelLinear
from src.layers import RowParallelLinear
from src.models.deepseek import DeepSeekModel
from src.models.deepseek import DeepSeekModelConfig
from src.models.deepseek import DeepSeekParallelContext
from src.models.moe import ExpertParallelMoE


def _tiny_cfg(num_hidden_layers: int = 2) -> DeepSeekModelConfig:
    return DeepSeekModelConfig(
        param_dtype=torch.float32,
        param_device=None,
        vocab_size=64,
        hidden_size=96,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=8,
        num_key_value_heads=8,
        q_lora_rank=48,
        kv_lora_rank=32,
        qk_nope_head_dim=12,
        qk_rope_head_dim=8,
        v_head_dim=12,
        intermediate_size=192,
        moe_intermediate_size=160,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        n_group=2,
        topk_group=1,
        max_position_embeddings=64,
    )


def test_forward_shape() -> None:
    """Model forward pass returns expected logits shape."""
    cfg = _tiny_cfg(num_hidden_layers=2)
    model = DeepSeekModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 32))

    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (2, 32, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_backward_runs() -> None:
    """Backward pass works for routed MoE graph."""
    cfg = _tiny_cfg(num_hidden_layers=2)
    model = DeepSeekModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(input_ids)
    loss = logits.mean()
    loss.backward()

    has_grad = any(param.grad is not None for param in model.parameters())
    assert has_grad


def test_meta_init_avoids_storage() -> None:
    """Meta initialization keeps all parameters on meta device."""
    cfg = _tiny_cfg(num_hidden_layers=2)
    with torch.device("meta"):
        model = DeepSeekModel(cfg)
    assert all(param.device.type == "meta" for param in model.parameters())


def test_parallel_context_wires_tensor_mp_dense_and_expert_mp_moe() -> None:
    """Parallel context should build TP dense FFNs and EP MoE blocks."""
    cfg = _tiny_cfg(num_hidden_layers=3)
    parallel_context = DeepSeekParallelContext(
        tensor_model_parallel_rank=0,
        tensor_model_parallel_size=2,
        expert_model_parallel_rank=0,
        expert_model_parallel_size=2,
        capacity_factor=1.0,
    )
    model = DeepSeekModel(cfg, parallel_context=parallel_context)

    dense_blocks = [block for block in model.blocks if block.ffn_type == "dense"]
    moe_blocks = [block for block in model.blocks if block.ffn_type == "moe"]

    assert dense_blocks
    assert moe_blocks
    assert all(getattr(block.ffn, "use_tp", False) for block in dense_blocks)
    assert all(isinstance(block.ffn.moe, ExpertParallelMoE) for block in moe_blocks)


def test_moe_uses_expert_model_parallel_domain() -> None:
    """RoutedMoE should resolve expert ownership from expert-model-parallel domain."""
    cfg = _tiny_cfg(num_hidden_layers=3)
    parallel_context = DeepSeekParallelContext(
        tensor_model_parallel_rank=1,
        tensor_model_parallel_size=2,
        expert_model_parallel_rank=1,
        expert_model_parallel_size=2,
    )
    model = DeepSeekModel(cfg, parallel_context=parallel_context)
    moe_blocks = [block for block in model.blocks if block.ffn_type == "moe"]
    assert moe_blocks

    moe_module = moe_blocks[0].ffn.moe
    assert isinstance(moe_module, ExpertParallelMoE)
    assert moe_module.ep_size == 2
    assert moe_module.ep_rank == 1


def test_moe_expert_ranges_replicate_across_tensor_parallel() -> None:
    """With expert_model_parallel_size=2, each EP shard repeats across TP ranks."""
    cfg = DeepSeekModelConfig(
        param_dtype=torch.float32,
        param_device=None,
        vocab_size=64,
        hidden_size=96,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=8,
        q_lora_rank=48,
        kv_lora_rank=32,
        qk_nope_head_dim=12,
        qk_rope_head_dim=8,
        v_head_dim=12,
        intermediate_size=192,
        moe_intermediate_size=160,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        n_group=4,
        topk_group=2,
        max_position_embeddings=64,
    )

    ranges: dict[tuple[int, int], tuple[int, int]] = {}
    for tp_rank in range(2):
        for ep_rank in range(2):
            context = DeepSeekParallelContext(
                tensor_model_parallel_rank=tp_rank,
                tensor_model_parallel_size=2,
                expert_model_parallel_rank=ep_rank,
                expert_model_parallel_size=2,
            )
            model = DeepSeekModel(cfg, parallel_context=context)
            moe_blocks = [block for block in model.blocks if block.ffn_type == "moe"]
            assert moe_blocks
            moe_module = moe_blocks[0].ffn.moe
            assert isinstance(moe_module, ExpertParallelMoE)
            start = moe_module.global_expert_start
            end = start + moe_module.experts_per_rank
            ranges[(tp_rank, ep_rank)] = (start, end)

    assert ranges == {
        (0, 0): (0, 4),
        (0, 1): (4, 8),
        (1, 0): (0, 4),
        (1, 1): (4, 8),
    }


def test_attention_tp_uses_tensor_model_parallel_domain() -> None:
    """Attention TP should use tensor model parallel rank/size."""
    cfg = _tiny_cfg(num_hidden_layers=1)
    parallel_context = DeepSeekParallelContext(
        tensor_model_parallel_rank=1,
        tensor_model_parallel_size=2,
        expert_model_parallel_rank=0,
        expert_model_parallel_size=2,
    )
    model = DeepSeekModel(cfg, parallel_context=parallel_context)
    attn = model.blocks[0].attn

    assert attn.attention_tensor_model_parallel_size == 2
    assert attn.attention_tensor_model_parallel_rank == 1
    assert attn.local_num_heads == 4
    assert isinstance(attn.q_b_proj, ColumnParallelLinear)
    assert isinstance(attn.kv_b_proj, ColumnParallelLinear)
    assert isinstance(attn.out_proj, RowParallelLinear)


def test_attention_tp_head_divisibility_validation() -> None:
    """Model should reject tensor TP sizes that do not divide attention heads."""
    cfg = _tiny_cfg(num_hidden_layers=1)
    cfg.num_attention_heads = 10
    cfg.num_key_value_heads = 10
    parallel_context = DeepSeekParallelContext(
        tensor_model_parallel_rank=0,
        tensor_model_parallel_size=3,
        expert_model_parallel_rank=0,
        expert_model_parallel_size=1,
    )

    with pytest.raises(ValueError, match="num_attention_heads must be divisible"):
        DeepSeekModel(cfg, parallel_context=parallel_context)


def test_pp_auto_partition_boundaries() -> None:
    """Auto PP partition should produce contiguous local layer ranges."""
    cfg = _tiny_cfg(num_hidden_layers=5)

    stage0_ctx = DeepSeekParallelContext(
        pipeline_model_parallel_rank=0,
        pipeline_model_parallel_size=2,
    )
    stage1_ctx = DeepSeekParallelContext(
        pipeline_model_parallel_rank=1,
        pipeline_model_parallel_size=2,
    )
    stage0 = DeepSeekModel(cfg, parallel_context=stage0_ctx)
    stage1 = DeepSeekModel(cfg, parallel_context=stage1_ctx)

    assert stage0.local_layer_range() == (0, 3)
    assert stage1.local_layer_range() == (3, 5)


def test_pp_manual_partition_boundaries() -> None:
    """Manual pp_layer_splits should override auto partitioning."""
    cfg = _tiny_cfg(num_hidden_layers=5)

    stage0_ctx = DeepSeekParallelContext(
        pipeline_model_parallel_rank=0,
        pipeline_model_parallel_size=2,
        pp_layer_splits=(0, 2, 5),
    )
    stage1_ctx = DeepSeekParallelContext(
        pipeline_model_parallel_rank=1,
        pipeline_model_parallel_size=2,
        pp_layer_splits=(0, 2, 5),
    )
    stage0 = DeepSeekModel(cfg, parallel_context=stage0_ctx)
    stage1 = DeepSeekModel(cfg, parallel_context=stage1_ctx)

    assert stage0.local_layer_range() == (0, 2)
    assert stage1.local_layer_range() == (2, 5)


def test_pp_stage_module_ownership() -> None:
    """First stage owns embeddings; last stage owns norm/head."""
    cfg = _tiny_cfg(num_hidden_layers=4)

    stage0 = DeepSeekModel(
        cfg,
        parallel_context=DeepSeekParallelContext(
            pipeline_model_parallel_rank=0,
            pipeline_model_parallel_size=2,
        ),
    )
    stage1 = DeepSeekModel(
        cfg,
        parallel_context=DeepSeekParallelContext(
            pipeline_model_parallel_rank=1,
            pipeline_model_parallel_size=2,
        ),
    )

    assert stage0.token_embeddings is not None
    assert stage0.final_norm is None
    assert stage0.lm_head is None

    assert stage1.token_embeddings is None
    assert stage1.final_norm is not None
    assert stage1.lm_head is not None


def test_forward_stage_shapes_for_pp() -> None:
    """forward_stage should pass hidden states stage-to-stage and end with logits."""
    cfg = _tiny_cfg(num_hidden_layers=4)
    batch_size = 2
    seq_len = 16

    stage0 = DeepSeekModel(
        cfg,
        parallel_context=DeepSeekParallelContext(
            pipeline_model_parallel_rank=0,
            pipeline_model_parallel_size=2,
        ),
    )
    stage1 = DeepSeekModel(
        cfg,
        parallel_context=DeepSeekParallelContext(
            pipeline_model_parallel_rank=1,
            pipeline_model_parallel_size=2,
        ),
    )

    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    hidden = stage0.forward_stage(input_ids=input_ids, hidden_states=None, attention_mask=None)
    assert hidden.shape == (batch_size, seq_len, cfg.hidden_size)

    logits = stage1.forward_stage(input_ids=None, hidden_states=hidden, attention_mask=None)
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)


def test_forward_raises_when_pp_enabled() -> None:
    """forward() should guide callers to use forward_stage in PP mode."""
    cfg = _tiny_cfg(num_hidden_layers=2)
    model = DeepSeekModel(
        cfg,
        parallel_context=DeepSeekParallelContext(
            pipeline_model_parallel_rank=0,
            pipeline_model_parallel_size=2,
        ),
    )
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))

    with pytest.raises(
        RuntimeError,
        match="forward\\(\\) is only valid for pipeline_model_parallel_size==1",
    ):
        model(input_ids)
