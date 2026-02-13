"""
Production-grade logging configuration for nano-train.

Features:
- Structured logging with timestamps and log levels
- Console and file output support
- Configurable log levels
- Module-specific loggers for better filtering
- Color-coded console output for readability
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for different log levels."""
    GREY = '\033[90m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD_RED = '\033[91m\033[1m'
    RESET = '\033[0m'


class ColorFormatter(logging.Formatter):
    """Color-coded formatter for console output."""

    COLORS = {
        logging.DEBUG: Colors.GREY,
        logging.INFO: Colors.BLUE,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BOLD_RED,
    }

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, use_colors: bool = True):
        """Initialize formatter with optional color support."""
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors

    def format(self, record):
        """Format log record with colors if enabled."""
        if self.use_colors and record.levelno in self.COLORS:
            levelname = record.levelname
            record.levelname = f"{self.COLORS[record.levelno]}{levelname}{Colors.RESET}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    use_colors: bool = True,
    log_format: Optional[str] = None
) -> None:
    """
    Configure root logger for nano-train.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        use_colors: Enable color-coded console output
        log_format: Custom format string (default: timestamp + level + module + message)
    """
    # Convert string level to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Default format: timestamp | level | module: message
    if log_format is None:
        log_format = '%(asctime)s | %(levelname)-8s | %(name)s: %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
    else:
        date_format = None

    # Create console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = ColorFormatter(
        log_format,
        datefmt=date_format,
        use_colors=use_colors
    )
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()  # Remove any existing handlers
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        # File output doesn't use colors
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for a module.

    Usage:
        from src.core.logging import get_logger
        logger = get_logger(__name__)

        logger.info("Training started")
        logger.debug(f"Batch shape: {batch.shape}")
        logger.warning("Low GPU memory")
        logger.error("Training failed")

    Args:
        name: Logger name (typically __name__ of module)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Convenience function for quick setup in scripts
def quick_setup(level: str = "INFO") -> None:
    """
    Quick setup for logging in scripts.

    Usage:
        from src.core.logging import quick_setup
        quick_setup("INFO")
    """
    setup_logging(log_level=level)
