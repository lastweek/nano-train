# Logging Guide

**Purpose**: Document the repo logging API and practical logging patterns for training and
infrastructure code.

**Audience**: Contributors editing `src/` modules and `examples/` scripts.

**Prerequisites**: Basic familiarity with Python's `logging` module.

**Related Docs**:
- [Training Monitoring Quickstart](training_monitoring_quickstart.md)
- [Training Monitoring Metrics Reference](training_monitoring_metrics_reference.md)

## Table of Contents

- [Overview](#overview)
- [Basic Setup](#basic-setup)
- [Configuration](#configuration)
- [Example Usage by Context](#example-usage-by-context)
- [Log Levels](#log-levels)
- [Best Practices](#best-practices)
- [Migration Pattern from `print()`](#migration-pattern-from-print)
- [Troubleshooting](#troubleshooting)

## Overview

nano-train uses a centralized logging helper in `src/logging.py`.

Core entry points:
- `setup_logging(...)`: initialize handlers and level at the program entrypoint.
- `get_logger(name)`: create module-level loggers with consistent format.

## Basic Setup

```python
from src.logging import get_logger
from src.logging import setup_logging

setup_logging(log_level="INFO")
logger = get_logger(__name__)

logger.info("Training started")
logger.warning("Low GPU memory")
```

## Configuration

```python
from src.logging import setup_logging

setup_logging(
    log_level="INFO",                 # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_file="output/training.log",   # optional file sink
    use_colors=True,                   # console colors
    log_format=None,                   # optional custom format string
)
```

Custom format example:

```python
custom_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
setup_logging(log_level="INFO", log_format=custom_format)
```

## Example Usage by Context

### Module Code

```python
from src.logging import get_logger

logger = get_logger(__name__)

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        logger.info("init layers=%d", config.num_layers)
```

### Training Entrypoint

```python
from src.logging import get_logger
from src.logging import setup_logging

setup_logging(log_level="INFO", log_file="output/train.log")
logger = get_logger(__name__)
```

### Tests

```python
from src.logging import get_logger
from src.logging import setup_logging

setup_logging(log_level="DEBUG")
logger = get_logger(__name__)
```

## Log Levels

| Level | Use Case |
|---|---|
| `DEBUG` | Detailed diagnostics and shape tracing |
| `INFO` | Normal training progress and config |
| `WARNING` | Non-fatal problems worth attention |
| `ERROR` | Operation failed, run may continue or abort |
| `CRITICAL` | Irrecoverable failure |

## Best Practices

1. Call `setup_logging(...)` once at the main entrypoint.
2. Use `logger = get_logger(__name__)` at module scope.
3. Keep `INFO` signal dense and `DEBUG` signal high-detail.
4. Use lazy guarded debug blocks for expensive computations.

```python
import logging

if logger.isEnabledFor(logging.DEBUG):
    logger.debug("Tensor stats: %s", expensive_stats(tensor))
```

5. Prefer structured wording with stable prefixes for grepability.

## Migration Pattern from `print()`

```python
# before
print(f"Loading data from {path}")
print(f"ERROR: {error}")

# after
logger.info("Loading data from %s", path)
logger.error("Failed to load: %s", error)
```

## Troubleshooting

### DEBUG messages do not appear

Set `log_level="DEBUG"` in `setup_logging(...)`.

### File logs are not created

Ensure the output directory exists and is writable.

```python
import os

os.makedirs("output", exist_ok=True)
setup_logging(log_file="output/training.log")
```

### Color codes look wrong

Set `use_colors=False` for terminals without ANSI support.
