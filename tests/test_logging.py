"""
Test the production-grade logging system.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging import setup_logging, get_logger


def test_logging():
    """Test basic logging functionality."""
    print("\n" + "=" * 70)
    print("Testing Logging System")
    print("=" * 70)

    # Setup logging
    setup_logging(log_level="INFO")
    logger = get_logger("test_logging")

    # Test different log levels
    logger.debug("This DEBUG message should NOT appear (level is INFO)")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")

    # Test with different log levels
    print("\n" + "-" * 70)
    print("Testing with DEBUG level:")
    print("-" * 70)

    setup_logging(log_level="DEBUG")
    logger_debug = get_logger("test_debug")

    logger_debug.debug("This DEBUG message SHOULD appear (level is DEBUG)")
    logger_debug.info("This is an INFO message")
    logger_debug.warning("This is a WARNING message")

    print("\n" + "=" * 70)
    print("Logging Test Completed ✓")
    print("=" * 70)


if __name__ == "__main__":
    test_logging()
