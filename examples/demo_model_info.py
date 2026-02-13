"""
Demonstration of model information dumping.

This script shows how to use dump_model_info() to inspect
models comprehensively.

Usage:
    python examples/demo_model_info.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.logging import setup_logging, get_logger
from src.utils.model_info import dump_model_info
from src.models.transformer import TransformerModel


def create_simple_demo_models():
    """Create various demo models to inspect."""
    models = {}

    # 1. Simple Linear Model
    class SimpleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 64)
            self.fc3 = nn.Linear(64, 10)

    models['simple_linear'] = SimpleLinear()

    # 2. Model with Embedding
    class ModelWithEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 128)
            self.fc = nn.Linear(128, 10)

    models['with_embedding'] = ModelWithEmbedding()

    # 3. Small Transformer (actual GPT-style model)
    config = ModelConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=512,
        dropout=0.1
    )
    models['small_transformer'] = TransformerModel(config)

    # 4. Model with frozen layers
    class ModelWithFrozenLayers(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(128, 256)
            self.layer2 = nn.Linear(256, 128)
            self.layer3 = nn.Linear(128, 64)

            # Freeze middle layer
            for param in self.layer2.parameters():
                param.requires_grad = False

    models['frozen_layers'] = ModelWithFrozenLayers()

    return models


def demo_basic_usage():
    """Demonstrate basic usage with print output."""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Usage (Print Output)")
    print("=" * 80)

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

    # Dump info (no logger provided = uses print)
    info = dump_model_info(model, logger=None, plot_distributions=False)

    print(f"\nReturned ModelInfo object with {info.num_layers} layers")
    print(f"Total parameters: {info.total_params:,}")


def demo_with_logger():
    """Demonstrate usage with structured logging."""
    print("\n" + "=" * 80)
    print("DEMO 2: Usage with Logger")
    print("=" * 80)

    # Setup logging
    setup_logging(log_level="INFO")
    logger = get_logger("demo_model_info")

    # Create model
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Linear(512, 256)
    )

    # Dump with logger
    info = dump_model_info(model, logger=logger, plot_distributions=False)

    logger.info(f"Analyzed model with {info.total_params:,} parameters")


def demo_transformer_model():
    """Demonstrate with actual transformer model."""
    print("\n" + "=" * 80)
    print("DEMO 3: Transformer Model Analysis")
    print("=" * 80)

    setup_logging(log_level="INFO")
    logger = get_logger("demo_transformer")

    # Create small transformer
    config = ModelConfig(
        vocab_size=100,
        hidden_size=128,
        num_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=256,
        dropout=0.1
    )

    model = TransformerModel(config)
    logger.info(f"Created transformer model")

    # Dump with distribution plots
    info = dump_model_info(
        model,
        logger=logger,
        report_path="outputs/model_reports/transformer_model_report.md",
        plot_distributions=True
    )

    logger.info(f"Transformer has {info.total_params:,} total parameters")
    logger.info(f"Memory usage: {info.total_memory_mb:.2f} MB")


def demo_comparison():
    """Demonstrate comparing multiple models."""
    print("\n" + "=" * 80)
    print("DEMO 4: Model Comparison")
    print("=" * 80)

    setup_logging(log_level="WARNING")  # Only show warnings
    logger = get_logger("demo_comparison")

    models = create_simple_demo_models()

    print(f"\n{'Model':<25} {'Params':<15} {'Memory (MB)':>15}")
    print("-" * 80)

    for name, model in models.items():
        info = dump_model_info(model, logger=logger, plot_distributions=False)
        print(f"{name:<25} {info.total_params:>15,} {info.total_memory_mb:>15.2f}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("MODEL INFO DUMPING DEMONSTRATIONS")
    print("=" * 80)

    demo_basic_usage()
    demo_with_logger()
    demo_transformer_model()
    demo_comparison()

    print("\n" + "=" * 80)
    print("ALL DEMOS COMPLETED ✓")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - outputs/model_reports/transformer_model_report.md (Markdown report)")
    print("  - outputs/model_reports/transformer_model_report_weights_hist.png (weight histogram)")
    print("  - outputs/model_reports/transformer_model_report_roofline.png (roofline summary)")
    print("  - outputs/model_reports/transformer_model_report_decode_batch_roofline.png")
    print("  - outputs/model_reports/transformer_model_report_module_pie.png (param breakdown)")
    print()


if __name__ == "__main__":
    main()
