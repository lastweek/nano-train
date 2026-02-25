"""
Smoke tests for DeepSeek-style model components.
"""

import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.deepseek import DeepSeekModel, DeepSeekModelConfig
from src.models.deepseek import DeepSeekParallelContext
from src.models.moe import ExpertParallelMoE


def test_forward_shape():
    """Model forward pass returns expected logits shape."""
    cfg = DeepSeekModelConfig(
        vocab_size=128,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        q_lora_rank=64,
        kv_lora_rank=48,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        intermediate_size=256,
        moe_intermediate_size=192,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        n_group=2,
        topk_group=1,
        max_position_embeddings=128,
    )
    model = DeepSeekModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 32))

    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (2, 32, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_backward_runs():
    """Backward pass works for routed MoE graph."""
    cfg = DeepSeekModelConfig(
        vocab_size=64,
        hidden_size=96,
        num_hidden_layers=2,
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
    model = DeepSeekModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(input_ids)
    loss = logits.mean()
    loss.backward()

    has_grad = any(param.grad is not None for param in model.parameters())
    assert has_grad


def test_meta_init_avoids_storage():
    """Meta initialization keeps all parameters on meta device (no real storage)."""
    cfg = DeepSeekModelConfig(
        vocab_size=64,
        hidden_size=96,
        num_hidden_layers=2,
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

    with torch.device("meta"):
        model = DeepSeekModel(cfg)

    assert all(param.device.type == "meta" for param in model.parameters())


def test_parallel_context_wires_tp_dense_and_ep_moe() -> None:
    """Parallel context should build TP dense FFNs and EP MoE blocks in DeepSeekModel."""
    cfg = DeepSeekModelConfig(
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
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        n_group=2,
        topk_group=1,
        max_position_embeddings=64,
    )
    parallel_context = DeepSeekParallelContext(
        tp_rank=0,
        tp_size=2,
        ep_rank=0,
        ep_size=2,
        capacity_factor=1.0,
    )
    model = DeepSeekModel(cfg, parallel_context=parallel_context)

    dense_blocks = [block for block in model.blocks if block.ffn_type == "dense"]
    moe_blocks = [block for block in model.blocks if block.ffn_type == "moe"]

    assert dense_blocks
    assert moe_blocks
    assert all(getattr(block.ffn, "use_tp", False) for block in dense_blocks)
    assert all(isinstance(block.ffn.moe, ExpertParallelMoE) for block in moe_blocks)


def run_all_tests():
    """Run local smoke tests."""
    print("Testing DeepSeek-V3-like model...")
    test_forward_shape()
    print("  ✓ Forward shape")
    test_backward_runs()
    print("  ✓ Backward pass")
    test_meta_init_avoids_storage()
    print("  ✓ Meta init (no parameter storage)")
    print("DeepSeek-V3-like tests passed.")


if __name__ == "__main__":
    run_all_tests()
