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

6. **Scientific Efficiency Study (new)**
   - Explicit FLOP variants: `F_theory`, `F_tensorcore`, `F_realizable`
   - Explicit byte decomposition: weights/activations/KV/temporary
   - AI metrics: `AI_weights_only`, `AI_hbm`, `AI_total`
   - Tensor-core saturation model: `eta_tc(M_eff)=min(1, M_eff/B_sat)` where `M_eff` is an
     effective GEMM M-dimension proxy (decode: `M_eff≈B`; dense prefill/training: `M_eff≈B*S`;
     MoE experts: `M_eff≈tokens per active expert`). The report keeps the historical symbol
     `eta_tc(B)` for decode intuition, but it is interpreted as `eta_tc(M_eff)` in calculations.
   - Peak-equivalent compute cost: `F_realizable = F_tensorcore/eta_tc + (F_theory-F_tensorcore)/eta_scalar`
   - Effective compute ceiling: `P_effective = P_peak * F_theory / F_realizable`
   - MFU estimate: `MFU_est = F_theory / (P_peak * T_est)`
   - Multi-chip roofline (H200/B200 FP8) with ridge markers and regime shading
   - Sensitivity analysis full combinational grid (medium profile default)
   - Architectural limit summary for training/prefill/decode

### Execution Model Knobs (WRF, Fusion, Elementwise)

The report contains an **Execution / Kernel Assumptions** table with these knobs:

| Knob | Meaning | Why it exists |
|---|---|---|
| `WRF` | Weight Residency Factor. Effective streamed weight bytes are modeled as `W_eff = W / WRF`. | Static analysis cannot directly measure cache residency; this approximates reuse. |
| `WRF attn/dense/moe` | Separate residency factors for attention, dense linear, and MoE expert weights. | Reuse patterns differ by module type. |
| `act fusion` | Multiplier on activation/intermediate bytes. Lower means fewer HBM reads/writes from fused kernels. | Captures fusion impact without runtime profiling. |
| `elementwise` | Multiplier on elementwise-heavy bytes (softmax/norm/masking temporaries). | Captures reduced temporary traffic in optimized kernels. |
| `attention bytes` | `naive` includes score/prob materialization; `flash` removes most `SxS` temporary HBM traffic while preserving KV read/write. | Distinguishes attention kernel families. |

Rationale for defaults:

- `naive`: `WRF=1`, `act fusion=1`, `elementwise=1` is a pessimistic baseline.
- `efficient`: `WRF=4/4/2`, `act fusion=0.5`, `elementwise=0.7` is a conservative,
  sensitivity-oriented approximation of common kernel/system improvements.
- These are **not** claims of measured behavior. If you have profiling counters,
  tune these knobs to match observed bytes and regenerate the report.

### Decode Batch Sweep: Formula Chain and Interpretation

In the decode batch sweep section, the report varies only microbatch size `B`.
The decode KV length `L` is fixed to the report `seq_len` value
(for `examples/train_deepseek.py`, this is `config.data.max_seq_length - 1`,
typically `255` with the default config).

For each execution mode `m` (`naive` or `efficient`), model-level rows are aggregated as:

- `F_theory(B,L,m) = sum_i F_i(B,L,m)`
- `bytes_hbm(B,L,m) = sum_i (bytes_weights_i + bytes_activations_i + bytes_kv_i + bytes_temporary_i)`
- `bytes_net(B,L,m) = sum_i bytes_net_i`
- `AI_hbm = F_theory / bytes_hbm`
- `TF_est = F_theory / T_est / 1e12`
- `TF_roofline_hbm = min(P_peak, BW_hbm * AI_hbm / 1e12)` (HBM roofline upper bound)
- `T_comp = F_realizable / (P_peak * 1e12)`
- `T_hbm = bytes_hbm / (BW_hbm * 1e9)`
- `T_net = bytes_net / (BW_net * 1e9)`
- `T_est = max(T_comp, T_hbm, T_net)`
- `regime = argmax{T_comp, T_hbm, T_net}`

Interpretation rule:

- If `AI_hbm >= OI_knee`, the point is on the compute side of the roofline.
- `TF_roofline_hbm` can be clipped by `P_peak` because the roofline uses
  `min(P_peak, BW_hbm * AI_hbm / 1e12)`. This is an *upper bound*, not a performance claim.
- `TF_est` is the report's plotted point: it uses the full time model (`T_est`), so it can sit
  strictly below `P_peak` when tensor-core utilization is modeled as < 1.

The report includes a worked decode example (prefer `B=128` when present) so every
`AI_hbm`, `TF_hbm`, `T_est`, and regime value is traceable to formulas, not hardcoded.

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

### Scientific Report Controls

```python
info = dump_model_info(
    model,
    logger=logger,
    report_path="outputs/model_reports/model_report.md",
    include_architecture_diagrams=True,
    include_appendix_derivations=True,
    enable_tensor_core_model=True,
    tensor_core_b_sat=64,
    sensitivity_enable=True,
    sensitivity_profile="medium_full_grid",
    roofline_x_limits=(1e-1, 1e5),
    roofline_y_limits=None,
    roofline_label_mode="minimal",
)
```

### New Output Structure

The generated markdown is organized as:

1. Architecture Overview
2. Analytical Model (FLOPs, Bytes, Time)
3. Roofline Analysis
4. Sensitivity Analysis
5. Architectural Limits
6. Appendix pointers (full appendix detail remains in `docs/model_info.md`)

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

## Appendix A: Detailed Derivations

### Dense Linear Layer

- Input shape: `[B,S,In]`, weight shape: `[In,Out]`
- FLOPs:
  - `F_linear = 2 * B * S * In * Out`

### MLA Attention FLOPs (Prefill)

- Terms:
  - `F_Q = 2 * B * S * H * r_q`
  - `F_K = 2 * B * S * H * r_kv`
  - `F_attn_score = 2 * B * h * S^2 * d_eff`
  - `d_eff = d_nope + d_rope`
- For DeepSeek-style MLA, use model-config dims (`H`, `h`, `r_q`, `r_kv`, `d_nope`, `d_rope`, `d_v`).

### MoE FLOPs Per Token

- One expert MLP:
  - `F_expert = 6 * H * d_moe`
- Routed total:
  - `F_MoE = B * S * top_k * 6 * H * d_moe`

## Appendix B: Full Debugging Checklist

- Verify `T_comp` uses `F_realizable` as a peak-equivalent compute cost (i.e., `T_comp = F_realizable / P_peak`).
- Verify WRF is applied consistently in prefill and decode paths.
- Verify KV bytes are counted per layer and multiplied by layer count.
- Verify temporary buffer bytes are not double-counted as activations.
- Verify dtype byte assumptions are consistent across weights/acts/KV.
