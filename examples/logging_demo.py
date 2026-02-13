"""
Demonstration of the production-grade logging system.

Run this to see different log levels and colors in action:
    python examples/logging_demo.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging import setup_logging, get_logger


def demo_basic_logging():
    """Demonstrate basic logging functionality."""
    logger = get_logger("demo.basic")

    logger.info("Starting logging demonstration")
    logger.debug("This debug message is hidden (level=INFO)")
    logger.info("This is an informational message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")


def demo_module_loggers():
    """Demonstrate module-specific loggers."""
    model_logger = get_logger("src.models.transformer")
    trainer_logger = get_logger("src.trainer")
    data_logger = get_logger("src.dataset")

    model_logger.info("Initializing TransformerModel")
    trainer_logger.info("Starting training loop")
    data_logger.info(f"Loaded dataset with 1000 sequences")


def demo_structured_logging():
    """Demonstrate structured logging with context."""
    logger = get_logger("demo.structured")

    # Training progress
    steps = 1000
    max_steps = 5000
    logger.info(f"Training step {steps}/{max_steps} ({steps/max_steps*100:.1f}%)")

    # Metrics
    loss = 2.456
    lr = 3e-4
    logger.info(f"Step {steps}: loss={loss:.4f}, lr={lr:.2e}")

    # System status
    logger.warning(f"GPU memory: 8.2GB / 24GB used (34%)")
    logger.debug(f"Batch shape: torch.Size([4, 256])")


def demo_error_scenarios():
    """Demonstrate error logging."""
    logger = get_logger("demo.errors")

    # Resource warnings
    logger.warning("High GPU memory usage: 22GB / 24GB")
    logger.warning("Slow iteration detected: 2.5s (target: <1s)")

    # Errors
    checkpoint_path = "/path/to/checkpoint.pt"
    logger.error(f"Failed to load checkpoint: {checkpoint_path} (file not found)")

    # Critical failures
    logger.critical("CUDA out of memory. Cannot continue training.")


def demo_color_levels():
    """Show all log levels and their colors."""
    logger = get_logger("demo.colors")

    logger.debug("DEBUG level - Grey color")
    logger.info("INFO level - Blue color")
    logger.warning("WARNING level - Yellow color")
    logger.error("ERROR level - Red color")
    logger.critical("CRITICAL level - Bold Red color")


def main():
    """Run all demonstrations."""
    setup_logging(log_level="INFO")
    logger = get_logger("main")

    logger.info("=" * 70)
    logger.info("Production-Grade Logging System Demonstration")
    logger.info("=" * 70)

    print()  # Add visual separation

    # Demo 1: Basic logging
    print("1. Basic Logging:")
    print("-" * 70)
    demo_basic_logging()

    print()

    # Demo 2: Module-specific loggers
    print("2. Module-Specific Loggers:")
    print("-" * 70)
    demo_module_loggers()

    print()

    # Demo 3: Structured logging
    print("3. Structured Logging with Context:")
    print("-" * 70)
    demo_structured_logging()

    print()

    # Demo 4: Error scenarios
    print("4. Error Scenarios:")
    print("-" * 70)
    demo_error_scenarios()

    print()

    # Demo 5: Color levels
    print("5. Log Level Colors:")
    print("-" * 70)
    demo_color_levels()

    print()
    logger.info("=" * 70)
    logger.info("Demonstration Complete")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
