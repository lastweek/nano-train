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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import Config
from src.models.transformer import TransformerModel
from data.dataset import TextDataset, create_dataloader
from src.training.trainer import Trainer


def main():
    # Configuration
    config = Config()

    # Print config
    print("=" * 50)
    print("Nano-Train MVP - First Training Cycle")
    print("=" * 50)
    print(f"\nModel config:")
    print(f"  Hidden size: {config.model.hidden_size}")
    print(f"  Num layers: {config.model.num_layers}")
    print(f"  Num attention heads: {config.model.num_attention_heads}")
    print(f"  Intermediate size: {config.model.intermediate_size}")
    print(f"  Max position embeddings: {config.model.max_position_embeddings}")
    print(f"\nTraining config:")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Max steps: {config.training.max_steps}")
    print(f"  Warmup steps: {config.training.warmup_steps}")
    print(f"  Use BF16: {config.training.bf16}")
    print()

    # Create model
    print("Creating model...")
    model = TransformerModel(config.model)
    print(f"Model created with {model.num_parameters:,} parameters")

    # Update vocab size based on dataset
    print("\nLoading dataset...")
    dataset = TextDataset(
        config.data.dataset_path,
        max_seq_length=config.data.max_seq_length
    )

    # Update config with actual vocab size
    config.model.vocab_size = dataset.vocab_size
    model = TransformerModel(config.model)
    print(f"Model updated with vocab size: {config.model.vocab_size}")
    print(f"Total parameters: {model.num_parameters:,}")

    # Create dataloader
    train_loader = create_dataloader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining device: {device}")

    # Create trainer
    trainer = Trainer(model, config, train_loader, device)

    # Train
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)
    trainer.train()

    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Check the loss curve in logs/")
    print("2. Generate text from the trained model")
    print("3. Increment: Add Flash Attention (Phase 2)")
    print("4. Increment: Add Tensor Parallelism (Phase 3)")


if __name__ == "__main__":
    main()
