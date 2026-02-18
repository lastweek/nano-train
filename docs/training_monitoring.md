# Training Monitoring & Stability Guide

**Purpose**: Comprehensive guide for monitoring LLM training, detecting issues early, and maintaining training stability.

**Last Updated**: 2026-02-16

---

## Table of Contents

1. [Implementation Status](#implementation-status)
2. [Implemented Features](#implemented-features)
3. [Core Monitoring Metrics](#core-monitoring-metrics)
4. [How to Use](#how-to-use)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Best Practices](#best-practices)
7. [Implementation Roadmap](#implementation-roadmap)
8. [References](#references)

---

## Implementation Status

### Completed ✅

- **Phase 1: Gradient Monitoring** (100%)
  - Per-layer gradient statistics (norm, mean, std)
  - Weight update ratio monitoring
  - Gradient flow visualization (histograms)
  - Automatic alert system for learning rate issues

### In Progress 🚧

- None currently

### Planned 📋

- Phase 2: Weight & Activation Monitoring
- Phase 3: Performance Monitoring
- Phase 4: Convergence & Stability Monitoring
- Phase 5: Advanced Diagnostics

---

## Implemented Features

### Phase 1: Gradient Monitoring (COMPLETE)

**Location**: [src/trainer.py:358-496](src/trainer.py#L358-L496)

Three methods added to the `Trainer` class:

#### 1.1 Per-Layer Gradient Statistics

**Method**: `_log_detailed_gradient_stats(step)`

**Metrics**:
- **Gradient Norm** (L2 magnitude): Overall gradient scale per layer
- **Gradient Mean**: Average gradient value (should be near zero)
- **Gradient Std**: Standard deviation (gradient variance)

**TensorBoard**: `gradients/{layer_name}/norm|mean|std`

**Console Output** (every 100 steps):
```
INFO  transformer.blocks.0.attn_norm.weight: grad_norm=0.1234, mean=0.000012, std=0.000456
INFO  transformer.blocks.0.attn.c_proj.weight: grad_norm=0.2345, mean=-0.000023, std=0.000567
```

#### 1.2 Weight Update Ratio Monitoring

**Method**: `_log_weight_update_ratios(step)`

**Formula**: `update_ratio = (learning_rate × gradient_norm) / weight_norm`

**Purpose**: Detect learning rate issues
- **Ratio > 0.1**: Learning rate too high (unstable)
- **Ratio < 1e-7**: Learning rate too low (not learning)

**TensorBoard**: `updates/{layer_name}/ratio|update_norm|weight_norm`

**Console Alerts** (every 100 steps when issues detected):
```
WARNING  Update ratio alerts (step 100):
WARNING    blocks.0.attn.c_proj.weight: 0.1234 > 0.1 (learning rate may be too high)
WARNING    blocks.5.mlp.c_fc.weight: 0.00000008 < 1e-7 (learning rate may be too low)
```

#### 1.3 Gradient Flow Visualization

**Method**: `_log_gradient_histograms(step)`

**Metrics**:
- Full gradient histograms (distribution shapes)
- Percentiles: min, p25, p50 (median), p75, p95, p99, max

**TensorBoard**: `gradients_hist/{layer_name}` (Histograms/Distributions tab)

**Console Output** (every 1000 steps):
```
INFO  Gradient distribution summary (step 1000):
INFO    blocks.0.attn.c_proj.weight: min=-0.002345, p25=-0.000456,
      median=0.000012, p75=0.000489, p95=0.001234, max=0.002567
```

**Detects**:
- Multi-modal distributions (multiple peaks)
- Outliers and heavy tails
- Dead neurons (gradients concentrated at zero)
- Distribution shifts over time

---

## Core Monitoring Metrics

### Loss Metrics

| Metric | Healthy Pattern | Red Flag |
|--------|-----------------|----------|
| **Training Loss** | Smooth decrease, occasional plateaus | Spikes, NaN, increase |
| **Validation Loss** | Decreases with train loss | Much higher than train |

**What to Expect**:
- Loss decreases in "hockey stick" pattern
- Initial rapid decrease, then gradual improvement
- Small plateaus during LR warmup/decay are normal

**When to Worry**:
- Loss spikes > 10× moving average
- Loss becomes NaN or Inf
- Validation loss increases while training decreases (overfitting)

### Gradient Metrics

| Metric | Healthy Range | Red Flag |
|--------|---------------|----------|
| **Global Gradient Norm** | 0.1 - 10.0 | < 0.01 or > 100 |
| **Per-Layer Gradient Norm** | Early layers 2-5× smaller than last | Early < 1% of last |
| **Gradient Mean** | Near zero (small values) | Large values, skewed |
| **Update Ratio** | 1e-5 to 1e-2 (depends on phase) | > 0.1 or < 1e-7 |

**Phase-Dependent Update Ratios**:
- Early training (0-1K steps): 1e-3 to 1e-2
- Mid training (1K-10K steps): 1e-4 to 1e-3
- Late training (10K+ steps): 1e-5 to 1e-4

**When to Worry**:
- **Vanishing gradients**: First layer < 1% of last layer
- **Exploding gradients**: Any layer > 100
- **Update ratio > 0.1**: Reduce LR by 5-10×
- **Update ratio < 1e-7**: Increase LR by 5-10×

### Weight Metrics

| Metric | Healthy Pattern | Red Flag |
|--------|-----------------|----------|
| **Weight Norm** | Grows slowly (10-50% over training) | Sudden spikes > 10× |
| **Per-Layer Weight Norms** | Consistent ratios across layers | Diverge by > 1000× |

### Performance Metrics

| Metric | Healthy Range | Red Flag |
|--------|---------------|----------|
| **Tokens/Second** | Consistent (±5%) | Drops > 50% |
| **GPU Utilization** | 80-95% | < 50% or > 98% |
| **Memory Usage** | 70-90% of VRAM | Spikes near limit |
| **Step Time** | Consistent (±10%) | Increases > 2× |

---

## How to Use

### Basic Training

Gradient monitoring is automatically enabled when you run training:

```bash
python3 examples/train_mvp.py
```

### TensorBoard Visualization

```bash
tensorboard --logdir=logs/
```

Navigate to `http://localhost:6006`:

**Scalars Tab**:
- `gradients/` - Per-layer gradient statistics
  - Compare layers by overlaying multiple
  - Look for smooth decreasing trends
- `updates/` - Weight update ratios
  - Check all layers in healthy range (1e-5 to 1e-2)
  - Debug layers with ratios > 0.1 or < 1e-7

**Histograms/Distributions Tab**:
- `gradients_hist/` - Full gradient distributions
  - Watch distribution shape evolution
  - Look for bell curves centered at zero
  - Identify multi-modal patterns, outliers

**Console Output** (automatic):
- Every 100 steps: Per-layer gradient stats
- Every 100 steps: Update ratio alerts (if issues)
- Every 1000 steps: Gradient distribution summary

### Interpreting Histogram Shapes

**HEALTHY**:
```
Normal Distribution:    ╭────╮
                        ╱      ╲   ← Centered at zero, smooth spread
                       ╱        ╲
```

**UNHEALTHY**:
```
Multi-modal:        ╭╮  ╭╮      ← Multiple behaviors
                    ╰╯  ╰╯         (e.g., attention vs MLP)

Heavy Tails:       ╭───╮          ← Extreme outliers
                   ╱     ╲___
                  ╱          ╲___

Shifted Center:      ╭────╮     ← Not centered at zero
                     ╱      ╲
                    ╱        ╲

Collapsed:           │            ← Dead gradients
                     │
```

---

## Troubleshooting Guide

### Problem 1: Loss Not Decreasing

**Symptoms**: Loss flat or decreasing very slowly

**Diagnosis Steps**:
1. Check gradient norms:
   ```python
   # In TensorBoard: gradients/{layer}/norm
   ```
   - Too low (< 0.01)? → Increase LR
   - Too high (> 100)? → Enable gradient clipping

2. Check update ratios:
   ```python
   # In TensorBoard: updates/{layer}/ratio
   ```
   - All low? → Increase global LR by 5-10×
   - Single layer high? → Check that layer's initialization

3. Check histograms:
   ```python
   # In TensorBoard: gradients_hist/{layer}
   ```
   - Collapsed to zero? → Vanishing gradients
   - Multi-modal? → Check layer architecture

**Solutions**:
- Increase learning rate (if gradients too small)
- Enable gradient clipping (if gradients too large)
- Check for dead ReLUs (if gradients are zero)
- Verify data is being loaded correctly

### Problem 2: Loss Spike

**Symptoms**: Sudden large increase in loss (e.g., 2.5 → 10.0)

**Diagnosis Steps**:
1. Check gradient norms:
   - Sudden spike? → Gradient explosion
   - Fix: Reduce LR, enable clipping

2. Check update ratios:
   - Sudden increase > 0.1? → LR too high
   - Fix: Reduce LR by 5-10×

3. Check histograms:
   - Heavy tails? → Outliers causing instability
   - Fix: Gradient clipping, batch normalization

**Solutions**:
- Reduce learning rate by 5-10×
- Enable gradient clipping (max_norm=1.0)
- Check for batch contamination (outliers)
- Switch from FP16 to BF16 or FP32

### Problem 3: Validation Loss Increasing

**Symptoms**: Training loss decreases but validation increases

**Diagnosis Steps**:
1. Check gradient norms (train vs val)
   - Train much higher? → Overfitting
   - Fix: Early stopping, regularization

2. Check per-layer patterns
   - Later layers much larger? → Overfitting in head
   - Fix: Weight decay, dropout

3. Check update ratios
   - Still high in late training? → LR not decayed
   - Fix: Check LR schedule

**Solutions**:
- Add/increase weight decay
- Add dropout layers
- Implement early stopping
- Reduce model capacity
- Increase training data

### Problem 4: Specific Layer Not Learning

**Symptoms**: One layer has much smaller gradients than others

**Diagnosis Steps**:
1. Check gradient norm for that layer:
   - Much smaller than others? → Vanishing
   - Fix: Residual connections, layer norm

2. Check update ratio for that layer:
   - Near zero? → Layer not receiving gradients
   - Fix: Check connectivity, skip connections

3. Check histogram:
   - Different shape than others? → Architecture issue
   - Fix: Check layer initialization, activation function

**Solutions**:
- Add residual connections
- Add layer normalization
- Check for disconnected layers
- Verify weight initialization

### Problem 5: Gradient Explosion

**Symptoms**: Gradient norms > 100, loss spikes to NaN

**Diagnosis**:
- Check global gradient norm in TensorBoard
- Look for sudden spikes in `Gradients/norm`

**Solutions**:
- Enable gradient clipping (max_norm=1.0)
- Reduce learning rate by 10×
- Check sequence length (longer = more unstable)
- Use gradient checkpointing

### Problem 6: Gradient Vanishing

**Symptoms**: Early layers have near-zero gradients

**Diagnosis**:
- Compare first layer norm to last layer norm
- Ratio < 0.01 indicates vanishing

**Solutions**:
- Increase learning rate
- Use gradient clipping with minimum threshold
- Add residual connections
- Use layer normalization
- Reduce network depth

---

## Best Practices

### Logging Configuration

**DO**:
- ✅ Log gradients every step (lightweight)
- ✅ Log weights every 100 steps (medium weight)
- ✅ Use TensorBoard for visualization
- ✅ Set up automated alerts for red conditions

**DON'T**:
- ❌ Log activations every step (too heavy)
- ❌ Use multiple logging frameworks (confusing)
- ❌ Log without timestamps (hard to debug)

### Logging Frequency Guide

| Metric | Frequency | Rationale |
|--------|-----------|-----------|
| Loss | Every step | Critical for convergence |
| Learning rate | Every step | Debug schedule issues |
| Gradient norm (global) | Every step | Detect explosion early |
| Per-layer gradients | Every log_steps | Detailed debugging |
| Per-layer weights | Every 100-1000 steps | Track evolution |
| Throughput | Every 100 steps | Detect performance issues |
| Memory usage | Every 100 steps | Prevent OOM |
| Gradient histograms | Every 100-1000 steps | Expensive operation |
| Validation loss | Every 1000 steps | Overfitting check |

### TensorBoard Organization

```
runs/
└── experiment_name/
    ├── Loss/
    │   ├── train           # Training loss
    │   └── validation      # Validation loss
    ├── Gradients/
    │   ├── norm_global     # Global gradient norm
    │   └── per_layer/      # Per-layer statistics
    │       ├── blocks.0.attn.c_proj.weight/norm
    │       ├── blocks.0.attn.c_proj.weight/mean
    │       └── ...
    ├── Gradients_Hist/     # Histogram distributions
    │   └── per_layer/
    │       └── ...
    ├── Updates/
    │   └── per_layer/      # Update ratios
    │       └── ...
    ├── Optimization/
    │   ├── lr              # Learning rate
    │   └── update_ratio    # Weight/gradient ratio
    ├── Performance/
    │   ├── tokens_per_sec  # Throughput
    │   ├── gpu_utilization # GPU % used
    │   └── memory_mb       # VRAM usage
    └── Parameters/
        └── norm_global     # Global weight norm
```

### Alert Thresholds

```python
ALERT_THRESHOLDS = {
    'loss_spike': 10.0,           # 10× increase in loss
    'grad_norm_max': 100.0,       # Maximum gradient norm
    'grad_norm_min': 0.001,       # Minimum gradient norm
    'weight_update_max': 0.1,     # Maximum update ratio
    'weight_update_min': 1e-7,    # Minimum update ratio
    'throughput_drop': 0.5,       # 50% throughput drop
    'memory_usage': 0.95,         # 95% VRAM usage
    'val_loss_increase': 0.1,    # 10% validation increase
}
```

---

## Implementation Roadmap

### Phase 1: Gradient Monitoring ✅ (COMPLETE)

- [x] Per-layer gradient statistics (norm, mean, std)
- [x] Weight update ratio monitoring
- [x] Gradient flow visualization (histograms)
- [x] Automatic alert system
- [ ] Gradient accumulation tracking
- [ ] Gradient flow quantile tracking

### Phase 2: Weight & Activation Monitoring (PLANNED)

- [ ] Per-layer weight norm logging
- [ ] Weight distribution tracking (histograms)
- [ ] Activation statistics (mean, std, min, max)
- [ ] Dead neuron detection
- [ ] Activation flow visualization
- [ ] Layer-wise feature statistics

### Phase 3: Performance Monitoring (PLANNED)

- [ ] Throughput tracking (tokens/sec)
- [ ] GPU utilization logging
- [ ] Memory usage profiling
- [ ] Step time breakdown (forward/backward/optimize)
- [ ] Communication overhead tracking
- [ ] Data loading statistics

### Phase 4: Convergence & Stability (PLANNED)

- [ ] Loss plateau detection
- [ ] Validation divergence alerts
- [ ] Learning rate schedule validation
- [ ] Training convergence prediction
- [ ] Early stopping criteria
- [ ] Overfitting detection

### Phase 5: Advanced Diagnostics (PLANNED)

- [ ] Automated health checks
- [ ] Alert system (email/slack integration)
- [ ] Anomaly detection (statistical)
- [ ] Comparative analysis (run comparison)
- [ ] Debug mode (detailed logging on failure)
- [ ] Recovery recommendations

---

## Quick Reference

### Red Alert Conditions 🚨

**Immediate Action Required** (Stop Training):
- Loss is NaN/Inf
- Loss spikes > 100×
- Gradient norm > 1000
- GPU memory OOM
- Throughput drops > 80%

**Warning Conditions** (Monitor Closely):
- Validation loss plateau (no improvement > 10K steps)
- Gradient norm decay (< 0.1× starting value)
- Per-layer gradient variance > 100× across depth
- Weight update ratio < 1e-7 or > 0.1

### Healthy Training Checklist ✅

- [ ] Loss decreases smoothly (no spikes)
- [ ] Gradient norms in 0.1-10 range
- [ ] Update ratios in phase-appropriate range
- [ ] Gradient histograms show bell curves at zero
- [ ] Per-layer gradient variance < 100×
- [ ] Throughput stable (±5%)
- [ ] Memory usage stable (70-90% VRAM)
- [ ] No update ratio alerts

### Debugging Workflow

```
Training Issue Detected
         ↓
Check Loss Trend (Spikes? NaN? Flat?)
         ↓
Check Gradient Norms (Too high? Too low?)
         ↓
Check Update Ratios (All same? Single outlier?)
         ↓
Check Histograms (Distribution shape?)
         ↓
Identify Root Cause
         ↓
Apply Fix (LR, clipping, architecture)
         ↓
Monitor Recovery
```

---

## References

### Academic Papers

- [AtPatch: Debugging Transformers via Hot-Fixing Over-Attention](https://arxiv.org/html/2601.21695v1) (2026)
- [Transformer Instability in Long Sequence Training](https://openreview.net/forum?id=hkVTFQQHBd) (2026)

### Practical Guides

- [LLM Evaluation: Frameworks, Metrics, and Best Practices (2026 Edition)](https://medium.com/@future_agi/llm-evaluation-frameworks-metrics-and-best-practices-2026-edition-162790f831f4)
- [The Complete Guide to LLM Evaluation Tools in 2026](https://futureagi.substack.com/p/the-complete-guide-to-llm-evaluation-c82)
- [The LLM Evaluation Guide: Metrics, Methods & Best Practices](https://www.comet.com/site/blog/llm-evaluation-guide/)
- [7 RL Debugging Moves When Training Looks Random](https://medium.com/@duckweave/7-rl-debugging-moves-when-training-looks-random-3d048d70616e)

### Tools & Frameworks

- [Weights & Biases (W&B)](https://wandb.ai/) - Experiment tracking
- [TensorBoard](https://www.tensorflow.org/tensorboard) - Visualization
- [MLflow](https://mlflow.org/) - Open-source tracking
- [Neptune.ai](https://neptune.ai/) - Experiment management
- [DeepEval](https://deepeval.ai/) - Production monitoring

---

**Document Status**: Living document, updated as new monitoring features are implemented.

**Next Implementation**: Phase 2 - Weight & Activation Monitoring

**For Questions**: Refer to the [Troubleshooting Guide](#troubleshooting-guide) or [Implementation Roadmap](#implementation-roadmap)
