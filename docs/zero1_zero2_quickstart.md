# ZeRO-1/2 Quickstart (Megatron-Style Flags)

**Purpose**: Run ZeRO-1/2 in nano-train using Megatron-style interface flags.

**Audience**: Engineers validating distributed optimizer behavior in `examples/train_4d.py`.

**Prerequisites**:
- Python environment with repo dependencies installed.
- Use `examples/launch.py` for local multi-process gloo runs.

**Related Docs**:
- [Megatron ZeRO-1/2 Design](megatron_zero1_zero2_design.md)
- [ZeRO-1/2 Intuitive Summary](zero1_zero2_intuitive_summary.md)
- [TP + PP + EP + DP Communication Guide](pp_tp_ep_dp_communication.md)

## Quick Commands

### 1) Baseline (no sharding)

```bash
python3 examples/launch.py --world-size 2 --backend gloo \
  --script examples/train_4d.py --script-args \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --max_steps 2 \
  --data-parallel-sharding-strategy no_shard
```

### 2) ZeRO-1 (`optim`)

```bash
python3 examples/launch.py --world-size 2 --backend gloo \
  --script examples/train_4d.py --script-args \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --use-distributed-optimizer \
  --data-parallel-sharding-strategy optim \
  --max_steps 2
```

### 3) ZeRO-2 (`optim_grads`)

```bash
python3 examples/launch.py --world-size 2 --backend gloo \
  --script examples/train_4d.py --script-args \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --use-distributed-optimizer \
  --data-parallel-sharding-strategy optim_grads \
  --max_steps 2
```

### 4) PP+EP with ZeRO

```bash
python3 examples/launch.py --world-size 4 --backend gloo \
  --script examples/train_4d.py --script-args \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 2 \
  --expert-model-parallel-size 2 \
  --num_microbatches 2 \
  --use-distributed-optimizer \
  --data-parallel-sharding-strategy optim \
  --max_steps 1
```

### 5) ZeRO Debug Mode (what the distributed optimizer is doing)

```bash
python3 examples/launch.py --world-size 4 --backend gloo \
  --script examples/train_4d.py --script-args \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 2 \
  --expert-model-parallel-size 2 \
  --num_microbatches 2 \
  --use-distributed-optimizer \
  --data-parallel-sharding-strategy optim_grads \
  --zero-debug \
  --zero-debug-max-steps 1 \
  --zero-debug-max-params 12 \
  --max_steps 1
```

### 6) Mixed Precision + ZeRO (Megatron-Style Precision Flags)

FP16 with dynamic loss scaling:

```bash
python3 examples/launch.py --world-size 2 --backend gloo \
  --script examples/train_4d.py --script-args \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --use-distributed-optimizer \
  --data-parallel-sharding-strategy optim_grads \
  --fp16 \
  --params-dtype fp16 \
  --main-params-dtype fp32 \
  --main-grads-dtype fp32 \
  --exp-avg-dtype fp32 \
  --exp-avg-sq-dtype fp32 \
  --max_steps 2
```

FP8 emulated path (for functional validation):

```bash
python3 examples/launch.py --world-size 2 --backend gloo \
  --script examples/train_4d.py --script-args \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --use-distributed-optimizer \
  --data-parallel-sharding-strategy optim \
  --fp8 \
  --fp8-backend emulated \
  --params-dtype bf16 \
  --main-params-dtype fp32 \
  --main-grads-dtype fp32 \
  --exp-avg-dtype fp32 \
  --exp-avg-sq-dtype fp32 \
  --max_steps 1
```

## Interface Contract

- Enable ZeRO path with `--use-distributed-optimizer`.
- Choose sharding strategy with `--data-parallel-sharding-strategy`:
  - `no_shard`: classic gradient sync path.
  - `optim`: ZeRO-1.
  - `optim_grads`: ZeRO-2.
- Current tutorial implementation rejects:
  - `optim_grads_params` (ZeRO-3 out of scope)
  - `--num-distributed-optimizer-instances` values other than `1`.
- ZeRO debug knobs:
  - `--zero-debug`: emit rank-aware init and per-step ZeRO counters.
  - `--zero-debug-max-steps`: how many early optimizer steps to log.
  - `--zero-debug-max-params`: how many parameter shard mappings to print at init.

## How To Confirm ZeRO Is Active

1. Start log should print `Distributed optimizer: enabled=True strategy=optim|optim_grads`.
2. With `--zero-debug`, look for lines like:
   - `[ZeRO Debug][rank=...] init strategy=...`
   - `[ZeRO Debug][rank=...] step=... all_reduce_calls=... reduce_scatter_calls=... all_gather_calls=...`
3. Strategy-specific signal:
   - `optim` (ZeRO-1): expect `all_reduce_calls > 0`, `reduce_scatter_calls = 0`
   - `optim_grads` (ZeRO-2): expect `reduce_scatter_calls > 0` (or fallback all-reduce count if backend falls back)

## Checkpoint Files (ZeRO path)

When ZeRO is enabled, each checkpoint directory writes:
- `model.pt`
- `scheduler.pt`
- `optimizer_nonparam.pt`
- `optimizer_manifest.json`
- `optimizer_shard_rank{rank}.pt`

## Troubleshooting

1. Error: `use_distributed_optimizer must be enabled...`
- Fix: add `--use-distributed-optimizer` when using `optim` or `optim_grads`.

2. Error: `optim_grads_params (ZeRO-3) is out of scope...`
- Fix: use `optim` or `optim_grads`.

3. Error: `num_distributed_optimizer_instances must be 1...`
- Fix: keep `--num-distributed-optimizer-instances 1` in v1.
