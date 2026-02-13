# Logging System

## Overview

Nano-train uses a production-grade logging system built on Python's standard `logging` module with color-coded console output.

## Features

- **Color-coded output** for easy readability in terminals
- **Structured log format** with timestamps, levels, and module names
- **Configurable log levels** (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Optional file logging** for persisting logs
- **Module-specific loggers** for better filtering

## Usage

### Basic Setup

```python
from src.logging import setup_logging, get_logger

# Setup at application entry point
setup_logging(log_level="INFO")

# Get logger for your module
logger = get_logger(__name__)

# Use logger
logger.info("Training started")
logger.debug(f"Batch shape: {batch.shape}")
logger.warning("Low GPU memory")
logger.error("Training failed")
```

### Log Levels

| Level | Usage | Output |
|--------|--------|---------|
| DEBUG | Detailed debugging information | Shown only when level=DEBUG |
| INFO | General informational messages | Default level |
| WARNING | Warning messages (not critical) | Always shown |
| ERROR | Error messages | Always shown |
| CRITICAL | Critical failures | Always shown |

### Configuration Options

```python
from src.core.logging import setup_logging

setup_logging(
    log_level="INFO",              # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file="output/training.log", # Optional file path for logging
    use_colors=True,                # Enable/disable color-coded output
    log_format=None                 # Custom format (default: timestamp | level | module: message)
)
```

### Custom Log Format

```python
custom_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
setup_logging(log_level="INFO", log_format=custom_format)
```

## Examples

### In Modules

```python
# src/models/transformer.py
from src.core.logging import get_logger

logger = get_logger(__name__)

class TransformerModel(nn.Module):
    def __init__(self, config):
        logger.info(f"Initializing model with {config.num_layers} layers")
        ...
```

### In Training Scripts

```python
# examples/train_mvp.py
from src.core.logging import setup_logging, get_logger

# Setup at the start
setup_logging(log_level="INFO", log_file="output/training.log")
logger = get_logger(__name__)

def main():
    logger.info("Starting training...")
    logger.info(f"Device: {device}")
    logger.info(f"Model parameters: {model.num_parameters:,}")
```

### In Tests

```python
# tests/test_layers.py
from src.core.logging import setup_logging, get_logger

# Use DEBUG level for detailed test output
setup_logging(log_level="DEBUG")
logger = get_logger(__name__)

def test_linear():
    logger.debug("Testing Linear layer...")
    ...
```

## Log Format

Default format:
```
2026-02-13 13:58:43 | INFO     | src.trainer: Starting training for 1000 steps...
2026-02-13 13:58:43 | INFO     | src.trainer: Model parameters: 125,000,000
2026-02-13 13:58:43 | WARNING  | src.trainer: Low GPU memory: 2GB remaining
2026-02-13 13:58:43 | ERROR    | src.trainer: Training failed: CUDA out of memory
```

## Color Coding

| Level | Color |
|--------|--------|
| DEBUG | Grey |
| INFO | Blue |
| WARNING | Yellow |
| ERROR | Red |
| CRITICAL | Bold Red |

## File Logging

To persist logs to a file:

```python
setup_logging(
    log_level="INFO",
    log_file="output/training.log"
)
```

Logs will be appended to the file without color codes (for better readability in log files).

## Best Practices

1. **Use appropriate log levels**
   - DEBUG: Detailed diagnostics (shapes, values, algorithm steps)
   - INFO: High-level progress (training started, checkpoint saved)
   - WARNING: Non-critical issues (high memory usage, slow iteration)
   - ERROR: Failures (training crashed, CUDA OOM)

2. **Use structured messages**
   ```python
   logger.info(f"Training step {step}/{max_steps}")
   logger.error(f"Failed to load checkpoint: {checkpoint_path}")
   ```

3. **Avoid expensive operations in debug logs**
   ```python
   # BAD: Expensive string formatting even if not logged
   logger.debug(f"Tensor stats: {torch.expensive_computation(tensor)}")

   # GOOD: Lazy evaluation
   if logger.isEnabledFor(logging.DEBUG):
       logger.debug(f"Tensor stats: {torch.expensive_computation(tensor)}")
   ```

4. **Use module-level loggers**
   ```python
   # In each module
   logger = get_logger(__name__)  # Returns 'src.models.transformer'
   ```

## Migration from print()

Replace `print()` statements with appropriate log levels:

```python
# BEFORE
print(f"Loading data from {path}...")
print(f"Vocab size: {vocab_size}")
print(f"ERROR: {error}")

# AFTER
logger.info(f"Loading data from {path}...")
logger.info(f"Vocab size: {vocab_size}")
logger.error(f"Failed to load: {error}")
```

## Troubleshooting

### No output for DEBUG messages

Make sure you set the log level to DEBUG:
```python
setup_logging(log_level="DEBUG")
```

### Colors not showing

Ensure your terminal supports ANSI color codes. Most modern terminals do.

### Logs not writing to file

Check directory exists and is writable:
```python
import os
log_dir = "output"
os.makedirs(log_dir, exist_ok=True)
setup_logging(log_file=f"{log_dir}/training.log")
```
