# Model Info Dumping Utility

## Overview

The `dump_model_info()` function provides comprehensive analysis of PyTorch models, going far beyond the basic `print(model)` output.

## Features

### Information Provided

1. **Parameter Counts**
   - Total parameters
   - Trainable parameters (requires_grad=True)
   - Non-trainable parameters (frozen layers)

2. **Memory Usage**
   - Per-layer memory consumption in MB
   - Total model memory usage
   - Human-readable format (KB, MB, GB)

3. **Layer Details**
   - Full layer name (e.g., `transformer.blocks.0.attn_norm.weight`)
   - Tensor shape (e.g., `(768, 768)`)
   - Number of parameters
   - Memory footprint
   - Gradient status (✓ requires grad, ✗ frozen)

4. **Weight Statistics**
   - Mean (average weight value)
   - Standard deviation (spread of weights)
   - Minimum value
   - Maximum value

5. **Distribution Visualization**
   - Generates grid of weight distribution plots
   - Shows statistics for each layer
   - Saved as PNG file

## Usage

### Basic Usage (Print Output)

```python
from src.utils.model_info import dump_model_info
import torch.nn as nn

# Create a model
model = nn.Sequential(
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 256)
)

# Dump info to console (uses print)
info = dump_model_info(model, logger=None, plot_distributions=False)
```

### With Logger

```python
from src.utils.model_info import dump_model_info
from src.logging import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)

# Dump with structured logging
info = dump_model_info(model, logger=logger, plot_distributions=False)
```

### With Distribution Plots

```python
# Generate weight distribution visualization
info = dump_model_info(
    model,
    logger=logger,
    plot_distributions=True,
    save_path="my_model_distributions.png"
)
```

## Output Format

### Summary Section

```
SUMMARY
  Total parameters:     263,936
  Trainable params:   263,936
  Non-trainable params: 0
  Total memory:        1.01 MB
  Total layers:        6
```

### Layer Details Table

```
LAYER DETAILS
Layer Name                               Shape                Params              Memory  Grad
0.weight                                 (512, 256)           131,072               0.50 MB     ✓
0.bias                                   (512,)               512                   0.00 MB     ✓
2.weight                                 (256, 512)           131,072               0.50 MB     ✓
2.bias                                   (256,)               256                   0.00 MB     ✓
```

### Weight Statistics Table

```
WEIGHT STATISTICS
Layer                                            Mean          Std          Min          Max
0.weight                                    -0.000121     0.036043    -0.062499     0.062500
0.bias                                       -0.001222     0.036325    -0.061849     0.062166
2.weight                                     0.000131      0.025508    -0.044193     0.044193
2.bias                                       0.000104      0.025811    -0.043909     0.043641
```

## Examples

### Example 1: Quick Model Inspection

```python
from src.utils.model_info import dump_model_info
from src.models.transformer import TransformerModel
from src.config import ModelConfig

# Create model
config = ModelConfig(vocab_size=1000, hidden_size=768, ...)
model = TransformerModel(config)

# Quick inspection
dump_model_info(model)
```

### Example 2: Compare Two Models

```python
# Model A: Small transformer
config_a = ModelConfig(hidden_size=256, num_layers=4, ...)
model_a = TransformerModel(config_a)
info_a = dump_model_info(model_a, logger=logger, plot_distributions=False)

# Model B: Large transformer
config_b = ModelConfig(hidden_size=768, num_layers=12, ...)
model_b = TransformerModel(config_b)
info_b = dump_model_info(model_b, logger=logger, plot_distributions=False)

print(f"\nModel A: {info_a.total_params:,} params")
print(f"Model B: {info_b.total_params:,} params")
print(f"Ratio: {info_b.total_params / info_a.total_params:.2f}x larger")
```

### Example 3: Check Frozen Layers

```python
# Create model with frozen encoder
model = MyModel()
for param in model.encoder.parameters():
    param.requires_grad = False

# Dump info to see what's frozen
info = dump_model_info(model, logger=logger)

print(f"\nTrainable: {info.trainable_params:,}")
print(f"Frozen: {info.non_trainable_params:,}")
```

### Example 4: Analyze Weight Initialization

```python
import torch.nn.init as init

model = MyModel()

# Check initial weight distribution
info_before = dump_model_info(model, logger=logger)

# Initialize with Xavier uniform
for module in model.modules():
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)

# Check after initialization
info_after = dump_model_info(model, logger=logger)
```

## Return Value

The function returns a `ModelInfo` dataclass:

```python
@dataclass
class ModelInfo:
    total_params: int           # Total number of parameters
    trainable_params: int        # Parameters with requires_grad=True
    non_trainable_params: int    # Parameters with requires_grad=False
    total_memory_mb: float       # Total memory in MB
    layers: List[LayerInfo]     # List of layer information
    num_layers: int             # Number of layers
```

You can use this for programmatic analysis:

```python
info = dump_model_info(model, logger=logger)

# Access information programmatically
ratio = info.trainable_params / info.total_params
print(f"Trainable ratio: {ratio:.2%}")

# Find largest layer
largest_layer = max(info.layers, key=lambda l: l.num_params)
print(f"Largest layer: {largest_layer.name} ({largest_layer.num_params:,} params)")

# Total memory in GB
memory_gb = info.total_memory_mb / 1024
print(f"Model size: {memory_gb:.2f} GB")
```

## Dependencies

### Required
- PyTorch (`torch`)
- Python standard library (`dataclasses`, `math`, `typing`)

### Optional
- Matplotlib (`matplotlib`) - For distribution plots
  - Install with: `pip install matplotlib`
  - If not available, distribution plots are skipped gracefully

## Integration with Training

### Before Training

```python
# Setup logging
setup_logging(log_level="INFO", log_file="training.log")
logger = get_logger(__name__)

# Create and analyze model
model = TransformerModel(config)
logger.info("Model analysis before training:")
model_info = dump_model_info(model, logger=logger, plot_distributions=True)

# Start training
trainer = Trainer(model, config, ...)
```

### During Training (Checkpoint Analysis)

```python
# Load checkpoint
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Analyze checkpoint
logger.info("Checkpoint model analysis:")
dump_model_info(model, logger=logger)
```

### Comparing Checkpoints

```python
# Load two checkpoints
model_early = load_model("checkpoint-step-1000.pt")
model_late = load_model("checkpoint-step-5000.pt")

# Compare
info_early = dump_model_info(model_early, logger=None, plot_distributions=False)
info_late = dump_model_info(model_late, logger=None, plot_distributions=False)

# Detect weight drift
for layer_e, layer_l in zip(info_early.layers, info_late.layers):
    mean_drift = abs(layer_e.mean - layer_l.mean)
    if mean_drift > 0.1:
        print(f"Large drift in {layer_e.name}: {mean_drift:.4f}")
```

## Performance Notes

- **Large models**: For models with billions of parameters, the statistics computation may take several seconds
- **Plot generation**: Distribution plots can take 1-2 seconds for models with 100+ layers
- **Memory overhead**: Minimal - only stores layer metadata, not weight copies

## Troubleshooting

### Issue: "Matplotlib not available" warning

**Solution**: Install matplotlib
```bash
pip install matplotlib
```

### Issue: Distribution plots not generated

**Solution**: Check file write permissions
```python
import os
save_path = "/tmp/my_model_distributions.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
dump_model_info(model, logger=logger, plot_distributions=True, save_path=save_path)
```

### Issue: Too much output

**Solution**: Set plot_distributions=False
```python
# Just get statistics, no plots
info = dump_model_info(model, logger=logger, plot_distributions=False)
```

## Testing

Run the test suite:
```bash
python tests/test_model_info.py
```

Run the demonstration:
```bash
python examples/demo_model_info.py
```

## API Reference

### dump_model_info()

```python
dump_model_info(
    model: torch.nn.Module,
    logger: Optional[logging.Logger] = None,
    plot_distributions: bool = True,
    save_path: Optional[str] = None
) -> ModelInfo
```

**Parameters:**
- `model`: PyTorch model to inspect
- `logger`: Optional logger instance (uses print if None)
- `plot_distributions`: Whether to generate distribution plots (default: True)
- `save_path`: Optional path to save plots (default: "model_weight_distributions.png")

**Returns:**
- `ModelInfo`: Dataclass with all collected information

**Raises:**
- No exceptions raised (all errors caught and logged)
