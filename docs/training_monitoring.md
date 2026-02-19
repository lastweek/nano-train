# Training Monitoring v2: Stability/Precision vs Performance

**Purpose**: nano-train monitoring is designed to answer two different questions:

1) **Training stability / numerical precision**: are updates finite, well-scaled, and flowing
through depth?
2) **Training performance**: are we compute- or input-bound, and are we trending toward OOM or
regressions?

**Last Updated**: 2026-02-19

---

## Quickstart

### 1) Run training (writes TensorBoard event files)

```bash
python examples/train_mvp.py
```

### 2) View metrics in TensorBoard

```bash
# Option A: helper script (if present in your checkout)
./start_tensorboard.sh

# Option B: start TensorBoard directly
tensorboard --logdir=outputs --port=6006 --host=localhost
```

Open `http://localhost:6006`.

### 3) Where logs live (source of truth)

Event files are written to:

```
{config.log_dir}/{config.run_name}/
```

Defaults (see `../src/config.py`):
- `Config.log_dir = "outputs"`
- `Config.run_name = "nano_train_mvp"`

So by default: `outputs/nano_train_mvp/`.

---

## Monitoring Modes (bounded by default)

Configured via `Config.monitoring.mode` (see `../src/config.py`):

- `minimal`: performance + core scalars only (no per-parameter or histogram work).
- `standard` (default): **bounded** monitoring. The number of time series scales with
  **#layers**, not **#parameter tensors**.
- `debug`: opt-in deep dive; enables per-parameter scalars/histograms (expensive).

**Histogram cadence** is controlled by `Config.monitoring.histogram_steps` (default: `1000`).

---

## Notation / Glossary

Let:

- Step index: \(t\)
- Learning rate used for the step: \(\eta_t\) (logged as `LR`)
- Parameters before the update: \(w_t\)
- Gradients: \(g_t = \nabla_w L_t\)
- Global L2 norm: \(\lVert x \rVert_2 = \sqrt{\sum_i x_i^2}\)
- Gradient clipping threshold: \(G_{\max}\) (`Config.training.clip_grad`)
- Small epsilon: \(\epsilon \approx 10^{-6}\)

---

## Metric Table A — Training Stability / Precision (numerics + optimization)

Default cadence:
- **Core scalars**: every `Config.training.log_steps`
- **Validation**: every `Config.training.eval_steps` (if a `val_loader` exists)
- **Histograms**: every `Config.monitoring.histogram_steps` (sentinel-only in Standard)

| Metric (TensorBoard tag) | What it is | Formula | Good (heuristic) | Bad / Action threshold (heuristic) | Mode |
|---|---|---|---|---|---|
| `Loss/train` | EMA of token cross-entropy (train) | \( \bar L_t=\beta\bar L_{t-1}+(1-\beta)L_t\) | Decreasing trend | Flat for long window → LR/data issue | Standard |
| `Health/loss_spike_ratio` | Spike detector vs previous EMA | \(S_t = L_t/\bar L_{t-1}\) | \(S_t < 2\) | Warn: \(>3\); Treat as “stop + debug”: \(>10\) | Standard |
| `Loss/val` | Held-out token cross-entropy | \(L^{val}=\mathrm{mean}(-\log p_\theta(y\mid x))\) | Tracks train downward | Rising while train falls → overfit / eval mismatch | Standard |
| `PPL/val` | Perplexity on val | \(\mathrm{PPL}=\exp(L^{val})\) | Decreasing | Rapid increase or non-finite | Standard |
| `LR` | LR actually used for the update | \(\eta_t\) | Matches schedule | Unexpected discontinuities / zeros | Standard |
| `Health/non_finite_loss` | Non-finite loss flag (fail-fast) | \(1[\neg\mathrm{isfinite}(L_t)]\) | Always 0 | If 1: run is invalid (training raises) | Standard |
| `Data/ignore_frac` | Fraction of targets ignored in loss (train) | \(\#(y=-100)/\#y\) | Near 0 | Warn: \(>0.2\); Bad: \(>0.3\) | Standard |
| `Data/ignore_frac_val` | Fraction ignored in loss (val) | same | Near 0 | Warn: \(>0.2\); Bad: \(>0.3\) | Standard |
| `Gradients/norm` | Global grad norm (pre-clip) | \(\lVert g_t\rVert_2\) | \(\sim 10^{-1}\)–\(10^{1}\) | Warn: \(>10^2\) or \(<10^{-2}\); Bad: \(>10^3\) | Standard |
| `Gradients/clip_coef` | Clip coefficient (how much we scale grads) | \(c_t=\min(1,\frac{G_{\max}}{\lVert g_t\rVert_2+\epsilon})\) | \(\approx 1\) most steps | Warn if \(<0.5\); Bad if \(<0.1\) repeatedly | Standard |
| `Gradients/clipped` | Whether clipping was active | \(1[\lVert g_t\rVert_2>G_{\max}]\) | Mostly 0 | If frequent → LR too high or unstable batch | Standard |
| `Parameters/norm` | Global param norm (pre-update) | \(\lVert w_t\rVert_2\) | Slowly drifting | Non-finite → run invalid (training raises) | Standard |
| `Updates/ratio_global` | Relative update size (proxy) | \(\frac{\eta_t\cdot (c_t\lVert g_t\rVert_2)}{\lVert w_t\rVert_2+\epsilon}\) | \(10^{-5}\)–\(10^{-2}\) | Warn: \(>10^{-1}\) or \(<10^{-7}\); Bad: \(>3\times10^{-1}\) | Standard |
| `Updates/ratio_p50` | Median update ratio over **block-main weights** | quantile of ratios | Stable | Very low \(<10^{-7}\) → too-small updates | Standard |
| `Updates/ratio_p95` | Tail update ratio (robust high end) | quantile | Not near \(10^{-1}\) | Warn if \(>10^{-1}\) | Standard |
| `Updates/ratio_max` | Max update ratio (outlier detector) | max | Stable | Bad if \(>3\times10^{-1}\) | Standard |
| `Gradients/block_{i}/norm` | Block aggregate grad norm (block-main weights) | \(\sqrt{\sum_j \lVert g_{i,j}\rVert_2^2}\) | Smooth across depth | Early blocks \(\ll\) last → vanishing | Standard |
| `Updates/block_{i}/ratio` | Block update ratio (block-main weights) | \(\eta_t\lVert g_i\rVert/\lVert w_i\rVert\) | Similar order across depth | Early blocks \(\ll\) last → “only last layers move” | Standard |
| `Gradients/depth_ratio_first_last` | First/last block grad ratio | \(\lVert g_0\rVert/\lVert g_{L-1}\rVert\) | \(0.1\)–\(1.0\) | Warn: \(<0.05\); Bad: \(<0.01\) | Standard |
| `Updates/depth_ratio_first_last` | First/last block update ratio | \(r_0/r_{L-1}\) | \(0.1\)–\(1.0\) | Warn: \(<0.05\); Bad: \(<0.01\) | Standard |
| `Activations/{site}/rms` | LayerNorm output RMS (sentinel sites) | \(\sqrt{\mathrm{mean}(a^2)}\) | \(\approx 1\) | Warn: \(<0.5\) or \(>2\); Bad: \(<0.25\) or \(>4\) | Standard |
| `Activations/{site}/max_abs` | Max abs activation (sentinel) | \(\max|a|\) | Stable | Warn: \(>20\); Bad: \(>50\) | Standard |
| `gradients/{param}/norm|mean|std` | Per-parameter gradient localization | \(\lVert g_i\rVert,\mu(g_i),\sigma(g_i)\) | mean near 0, std nonzero | Large outliers vs peers | Debug (sentinel-only in Standard) |
| `updates/{param}/ratio|update_norm|weight_norm` | Per-parameter update localization | ratio as above | Stable | Outliers \(>0.1\) or \(<10^{-7}\) | Debug (sentinel-only in Standard) |
| `gradients_hist/{param}` | Gradient histogram (shape) | distribution | Symmetric, stable tails | Heavy tails growing; collapse near 0 | Debug (sentinel-only in Standard) |
| `weights/{param}/norm` | Per-parameter weight norm | \(\lVert w_i\rVert_2\) | Stable | Sudden jumps / drift | Debug (sentinel-only in Standard) |
| `Weights/block_{i}/norm` | Block aggregate weight norm | \(\sqrt{\sum_j \lVert w_{i,j}\rVert_2^2}\) | Similar scale across depth | Warn if max/min \(>1000\times\) | Standard |
| `weights_hist/{param}` | Weight histogram (shape) | distribution | Stable tails | Heavy tails / drift | Debug (sentinel-only in Standard) |

**Where this is implemented**: `../src/trainer.py` (training loop + monitoring helpers) and
`../src/config.py` (modes + thresholds).

---

## Metric Table B — Training Performance (time + throughput + memory)

| Metric (TensorBoard tag) | What it is | Formula | Good (heuristic) | Bad / Action threshold (heuristic) |
|---|---|---|---|---|
| `Time/step_seconds` | Wall time per training step | \(t_{\text{step}}\) | Stable (±10%) | Persistent \(>2\times\) jump → regression |
| `Time/data_wait_seconds` | Time waiting for next batch | \(t_{\text{wait}}\) | Small vs step time | Persistent large wait → input-bound |
| `Time/data_wait_frac` | Input-bound indicator | \(t_{\text{wait}}/t_{\text{step}}\) | < 0.3 | Warn: > 0.3; Bad: > 0.5 |
| `Throughput/steps_per_second` | Steps/sec over the current window | steps / time | Stable | Drop \(>30\%\) needs investigation |
| `Throughput/tokens_per_second` | Tokens/sec over the current window | \(\sum N_{\text{tok}}/\Delta t\) | Stable | Drop \(>50\%\) → likely regression |
| `Throughput/samples_per_second` | Samples/sec over the current window | samples / time | Stable | Same as above |
| `Memory/allocated_mb` | Active CUDA memory | MB | Stable | Upward creep suggests leak |
| `Memory/reserved_mb` | CUDA allocator reserved memory | MB | Stabilizes after warmup | Persistent climb → fragmentation/leak risk |
| `Memory/max_allocated_mb` | High-water mark allocated | MB | Stable | Approaches VRAM limit → OOM risk |
| `Memory/max_reserved_mb` | High-water mark reserved | MB | Stable | Approaches VRAM limit → OOM risk |
| `Memory/reserved_frac` | Reserved / total VRAM | reserved / total | < 0.90 | Warn: > 0.90; Bad: > 0.95 |
| `Time/checkpoint_seconds` | Checkpoint save latency | seconds | Occasional spikes ok | Persistent stalls → I/O bottleneck |

---

## Interpretation Rules (fast diagnosis)

### Rule 1: “Is this run numerically valid?”

Stop and debug immediately if any are non-finite:
- `Health/non_finite_loss` becomes 1
- `Gradients/norm` becomes non-finite
- `Parameters/norm` becomes non-finite

First actions:
- Reduce `Config.training.learning_rate` by 5–10×
- Ensure `Config.training.clip_grad` is enabled (default: 1.0)
- If on FP16: switch to BF16 (nano-train defaults to BF16 when available)

### Rule 2: “Are updates the right size?”

Look at:
- `Updates/ratio_global`
- `Updates/ratio_p95` and `Updates/ratio_max`

Interpretation:
- Too high (≈\(10^{-1}\) or above): updates are huge → instability risk.
- Too low (≈\(10^{-7}\) or below): weights barely move → training will look flat.

First actions:
- Too high: reduce LR 5–10×; check for frequent clipping (`Gradients/clipped`).
- Too low: increase LR 5–10×; confirm gradients aren’t vanishing (`Gradients/depth_ratio_first_last`).

### Rule 3: “Are gradients flowing through depth?”

Look at:
- `Gradients/block_{i}/norm`
- `Gradients/depth_ratio_first_last`

Interpretation rule:
- If early blocks are consistently < 1% of the last block, you’re effectively only training the top.

First actions:
- Increase LR slightly (if global update ratios are too low).
- Check data/labels (high `Data/ignore_frac` makes learning look dead).

### Rule 4: “Is the run input-bound or compute-bound?”

Look at:
- `Time/data_wait_frac`
- `Throughput/tokens_per_second`

Interpretation:
- If `Time/data_wait_frac` is high, you’re waiting on the dataloader/CPU, not the GPU.

First actions:
- Reduce Python overhead; simplify dataset transforms.
- Consider increasing batch size if memory allows.

### Rule 5: “Am I heading toward OOM?”

Look at:
- `Memory/reserved_frac`
- `Memory/max_reserved_mb`

First actions:
- Reduce batch size or sequence length.
- Enable BF16 (already default).

---

## Tuning Knobs (config-driven) and what to watch

All knobs live in `../src/config.py`.

- `Config.training.learning_rate`: primarily moves `Updates/*` ratios up/down. Confirm by watching
  `Updates/ratio_global` and `Updates/ratio_p95`.
- `Config.training.clip_grad`: controls clipping. Confirm with `Gradients/clip_coef` and
  `Gradients/clipped`.
- `Config.training.warmup_steps`: affects early `LR` and early update ratios. Watch `LR` and
  `Updates/ratio_global`.
- `Config.training.log_steps`: controls scalar logging cadence. Watch tags update every N steps.
- `Config.monitoring.histogram_steps`: controls histogram cadence. Watch `*_hist/*` update every N.
- `Config.training.eval_steps`: controls `Loss/val` / `PPL/val` cadence.
- `Config.monitoring.mode`: `standard` for bounded; `debug` for per-parameter deep dives.
- `Config.monitoring.sync_cuda_timing`: if `True`, `Time/step_seconds` is measured with CUDA sync
  (more accurate, more overhead).

---

## Known Limitations (current MVP)

- Update ratios are **proxies** for AdamW (moments + weight decay are not modeled).
- No GPU utilization logging yet (we only log wall time + throughput + CUDA memory).
- No activation histograms (only sentinel LN output scalars).
- No detailed time breakdown (forward/backward/optim) unless we add a future timing module.

---

## Roadmap (high-signal next additions)

- Configurable alerts (Slack/email) + structured “health check” summaries.
- More activation sentinels (expand to `mlp_norm`, etc.) and optional histograms.
- Timing breakdown (forward/backward/optimizer), optionally with CUDA events.
- Richer validation metrics (accuracy proxies, sample generation checks).

