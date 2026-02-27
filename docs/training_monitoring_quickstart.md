# EP Training Monitoring Quickstart

**Purpose**: Provide a fast path to run training, open TensorBoard, and read the first
high-signal health indicators.

**Audience**: Users running tutorial training scripts and debugging stability/perf.

**Prerequisites**: Working Python environment, TensorBoard installed, and runnable examples.

**Related Docs**:
- [Training Monitoring Metrics Reference](training_monitoring_metrics_reference.md)
- [Logging Guide](logging.md)

## Run and View

### 1) Run a training script

```bash
python examples/train_mvp.py
```

### 2) Start TensorBoard

```bash
# Option A
./scripts/start_tensorboard.sh

# Option B
tensorboard --logdir=outputs --port=6006 --host=localhost
```

Open `http://localhost:6006`.

### 3) Verify event file location

Event files are written to:

```text
{config.log_dir}/{config.run_name}/
```

Defaults from `src/config.py`:
- `Config.log_dir = "outputs"`
- `Config.run_name = "nano_train_mvp"`

## Monitoring Modes

Configured by `Config.monitoring.mode`:
- `minimal`: core metrics only.
- `standard`: bounded default mode.
- `debug`: deeper per-parameter diagnostics.

## First Checks (in order)

1. Numerical validity
   Watch `Health/non_finite_loss`, `Gradients/norm`, `Parameters/norm`.
2. Update scale
   Watch `Updates/ratio_global`, `Updates/ratio_p95`, `Updates/ratio_max`.
3. Input vs compute bottleneck
   Watch `Time/data_wait_frac`, `Time/compute_frac_est`, `Throughput/effective_tokens_per_second`.
4. Memory risk
   Watch `Memory/reserved_frac`, `Memory/max_reserved_mb`.

## Fast Triage Actions

- Divergence or non-finite values:
  reduce learning rate by 5-10x; verify grad clipping is enabled.
- Flat learning:
  check `Updates/ratio_*` and `Gradients/depth_ratio_first_last`; adjust LR.
- Slow throughput:
  check `Time/data_wait_frac`; simplify dataloader and transforms.
- OOM trend:
  reduce batch size or sequence length.

## Detailed Reference

Use the full metric definitions, formulas, and thresholds in:
[Training Monitoring Metrics Reference](training_monitoring_metrics_reference.md)
