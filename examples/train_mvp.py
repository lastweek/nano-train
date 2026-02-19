"""
Simple training script for MVP.

Usage:
    python examples/train_mvp.py

This will train a 125M parameter model on the tiny Shakespeare dataset.
The goal is to verify the loss decreases and the model learns.
"""

import os
import sys
import torch
from torch.utils.data import random_split

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.logging import setup_logging, get_logger
from src.models.transformer import TransformerModel
from src.dataset import TextDataset, create_dataloader
from src.trainer import Trainer
from src.utils import dump_model_info


# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)


def resolve_dataset_path(dataset_path: str) -> str:
    """Resolve dataset path relative to the examples directory."""
    if os.path.isabs(dataset_path):
        return dataset_path
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.path.basename(dataset_path)
    )


def print_config(config: Config) -> None:
    """Print model and training configuration."""
    logger.info("=" * 50)
    logger.info("Nano-Train MVP - First Training Cycle")
    logger.info("=" * 50)

    logger.info("Model config:")
    logger.info(f"  Hidden size: {config.model.hidden_size}")
    logger.info(f"  Num layers: {config.model.num_layers}")
    logger.info(f"  Num attention heads: {config.model.num_attention_heads}")
    logger.info(f"  Intermediate size: {config.model.intermediate_size}")
    logger.info(f"  Max position embeddings: {config.model.max_position_embeddings}")

    logger.info("Training config:")
    logger.info(f"  Batch size: {config.training.batch_size}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    logger.info(f"  Max steps: {config.training.max_steps}")
    logger.info(f"  Warmup steps: {config.training.warmup_steps}")
    logger.info(f"  Use BF16: {config.training.bf16}")


def main():

    # Configuration
    config = Config()
    print_config(config)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")

    # Load dataset first to determine vocab size
    logger.info("Loading dataset...")
    dataset = TextDataset(
        resolve_dataset_path(config.data.dataset_path),
        max_seq_length=config.data.max_seq_length
    )

    # Update config with actual vocab size
    config.model.vocab_size = dataset.vocab_size

    # Deterministic train/val split for monitoring Loss/val + PPL/val.
    num_total = len(dataset)
    num_train = int(num_total * float(config.data.train_split))
    num_val = max(0, num_total - num_train)
    if num_train <= 0 or num_val <= 0:
        logger.warning(
            "Dataset too small to split (total=%d, train_split=%.3f); running without validation.",
            num_total,
            float(config.data.train_split),
        )
        train_dataset = dataset
        val_dataset = None
    else:
        generator = torch.Generator().manual_seed(int(config.seed))
        train_dataset, val_dataset = random_split(
            dataset,
            [num_train, num_val],
            generator=generator,
        )
        logger.info("Dataset split: train=%d, val=%d", len(train_dataset), len(val_dataset))

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = create_dataloader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
        )

    # Create model after vocab is known
    logger.info("Creating model...")
    model = TransformerModel(config.model)
    logger.info(f"Model vocab size: {config.model.vocab_size}")
    logger.info(f"Total parameters: {model.num_parameters:,}")
    dump_model_info(model, logger=logger, plot_distributions=False)

    # Create trainer
    trainer = Trainer(model, config, train_loader, device, val_loader=val_loader)

    # Train
    logger.info("=" * 50)
    logger.info("Starting training...")
    logger.info("=" * 50)
    trainer.train()

    logger.info("=" * 50)
    logger.info("Training completed successfully!")
    logger.info("=" * 50)
    logger.info("Next steps:")
    logger.info("1. Check the loss curve in outputs/")
    logger.info("2. Generate text from the trained model")
    logger.info("3. Increment: Add Flash Attention (Phase 2)")
    logger.info("4. Increment: Add Tensor Parallelism (Phase 3)")


if __name__ == "__main__":
    main()
