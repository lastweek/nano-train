"""
Train a DeepSeek-V3-like routed-MoE model on the local text dataset.

Usage:
    python3 examples/deepseek.py
    python3 examples/deepseek.py --no-meta-init
    python3 examples/deepseek.py --model-preset tiny --no-meta-init
"""

import argparse
import os
import sys
from typing import Optional

import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.dataset import TextDataset, create_dataloader
from src.logging import get_logger, setup_logging
from src.models.deepseek import DeepSeekModel, DeepSeekModelConfig
from src.trainer import Trainer
from src.utils import dump_model_info


setup_logging(log_level="INFO")
logger = get_logger(__name__)


def resolve_dataset_path(dataset_path: str) -> str:
    """Resolve dataset path relative to examples directory if needed."""
    if os.path.isabs(dataset_path):
        return dataset_path
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.path.basename(dataset_path),
    )


def build_deepseek_config(
    model_preset: str,
    vocab_size: Optional[int] = None,
) -> DeepSeekModelConfig:
    """
    Build model config for the selected preset.

    Presets:
    - deepseek_v3: Uses DeepSeekModelConfig defaults (official-scale shape).
    - tiny: Small runnable config for local training experiments.
    """
    if model_preset == "deepseek_v3":
        return DeepSeekModelConfig()

    if model_preset == "tiny":
        if vocab_size is None:
            raise ValueError("vocab_size is required for tiny preset")
        return DeepSeekModelConfig(
            vocab_size=vocab_size,
            hidden_size=384,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=1024,
            rms_norm_eps=1e-6,
            q_lora_rank=192,
            kv_lora_rank=96,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
            attention_dropout=0.0,
            dropout=0.1,
            intermediate_size=1536,
            moe_intermediate_size=1024,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            moe_layer_freq=1,
            scoring_func="sigmoid",
            norm_topk_prob=True,
            n_group=2,
            topk_group=1,
            routed_scaling_factor=2.5,
            tie_word_embeddings=False,
        )

    raise ValueError(f"Unknown model preset: {model_preset}")


def parse_args() -> argparse.Namespace:
    """Parse script arguments."""
    parser = argparse.ArgumentParser(description="Train or inspect a DeepSeek-V3-like model.")
    parser.set_defaults(meta_init=True)
    parser.add_argument(
        "--model-preset",
        choices=["deepseek_v3", "tiny"],
        default="deepseek_v3",
        help=(
            "Model preset. "
            "'deepseek_v3' uses official-scale config fields; "
            "'tiny' is a small runnable approximation."
        ),
    )
    parser.add_argument(
        "--meta-init",
        dest="meta_init",
        action="store_true",
        help=(
            "Initialize model on 'meta' device (no parameter storage). "
            "Enabled by default to avoid OOM with large model configs."
        ),
    )
    parser.add_argument(
        "--no-meta-init",
        dest="meta_init",
        action="store_false",
        help=(
            "Materialize model parameters and run full training flow "
            "(requires enough memory for model weights)."
        ),
    )
    return parser.parse_args()


def estimate_param_memory_mb(num_params: int, bytes_per_param: int) -> float:
    """Estimate parameter memory from count and bytes/parameter."""
    return (num_params * bytes_per_param) / (1024 ** 2)


def main() -> None:
    args = parse_args()

    config = Config()
    config.run_name = "deepseek_style_moe"
    config.training.batch_size = 8
    config.training.max_steps = 100
    config.training.warmup_steps = 10
    config.training.learning_rate = 2e-4
    config.training.save_steps = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("DeepSeek-V3-like training setup")
    logger.info(f"Model preset: {args.model_preset}")
    logger.info(f"Device: {device}")
    logger.info(f"Dataset path: {config.data.dataset_path}")

    needs_dataset = (not args.meta_init) or (args.model_preset == "tiny")
    dataset = None
    if needs_dataset:
        dataset = TextDataset(
            resolve_dataset_path(config.data.dataset_path),
            max_seq_length=config.data.max_seq_length,
        )

    train_loader = None
    if not args.meta_init:
        if dataset is None:
            raise RuntimeError("Dataset must be loaded for training.")
        train_loader = create_dataloader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
        )

    model_cfg = build_deepseek_config(
        model_preset=args.model_preset,
        vocab_size=dataset.vocab_size if dataset is not None else None,
    )
    if args.meta_init:
        logger.info("Using meta initialization: model parameters will not allocate real memory.")
        with torch.device("meta"):
            model = DeepSeekModel(model_cfg)
    else:
        model = DeepSeekModel(model_cfg)

    logger.info(f"Model parameters: {model.num_parameters:,}")
    fp8_mb = estimate_param_memory_mb(model.num_parameters, bytes_per_param=1)
    fp32_mb = estimate_param_memory_mb(model.num_parameters, bytes_per_param=4)
    bf16_mb = estimate_param_memory_mb(model.num_parameters, bytes_per_param=2)
    logger.info(
        "Estimated parameter memory: "
        f"fp8={fp8_mb / 1024:.2f} GB, bf16/fp16={bf16_mb / 1024:.2f} GB, "
        f"fp32={fp32_mb / 1024:.2f} GB"
    )
    logger.info(
        "Architecture: "
        f"layers={model_cfg.num_hidden_layers}, heads={model_cfg.num_attention_heads}, "
        f"vocab={model_cfg.vocab_size}, routed_experts={model_cfg.n_routed_experts}, "
        f"experts_per_tok={model_cfg.num_experts_per_tok}"
    )

    dump_model_info(
        model,
        logger=logger,
        report_path=os.path.join(config.output_dir, "model_reports", "deepseek_model_report.md"),
        batch_size=config.training.batch_size,
        seq_len=config.data.max_seq_length - 1,
        plot_distributions=not args.meta_init,
        plot_roofline=True,
    )

    if args.meta_init:
        logger.info("Meta-init mode: report generated; skipping training.")
        return

    trainer = Trainer(model, config, train_loader, device)
    trainer.train()


if __name__ == "__main__":
    main()
