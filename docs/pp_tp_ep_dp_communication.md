# TP + PP + EP + DP Communication Guide (Megatron Naming)

**Purpose**: Explain what pipeline parallelism adds on top of TP+EP+DP in
`examples/train_4d.py`.

**Audience**: Readers who already understand TP/EP/DP and want a practical 4D model.

**Prerequisites**:
- [TP + DP Backward Flow](tp_dp_communication.md)
- [TP + EP + DP Communication Guide](ep_tp_dp_communication.md)

**Related Docs**:
- [DeepSeekMoE Auxiliary Losses](deepseek_moe_aux_losses.md)

## Table of Contents

- [1) 4D process grid](#1-4d-process-grid)
- [2) Communication domains](#2-communication-domains)
- [3) What PP changes in model execution](#3-what-pp-changes-in-model-execution)
- [4) 1F1B non-interleaved schedule](#4-1f1b-non-interleaved-schedule)
- [5) Label path in this tutorial](#5-label-path-in-this-tutorial)
- [6) Backward and gradient synchronization](#6-backward-and-gradient-synchronization)
- [7) Where extra communication appears](#7-where-extra-communication-appears)
- [8) Practical checklist](#8-practical-checklist)

## 1) 4D process grid

Each rank has coordinates:
`(data_parallel_rank, pipeline_model_parallel_rank, tensor_model_parallel_rank, expert_model_parallel_rank)`.

Rank mapping in this repo:

`rank = (((dp * pp_size + pp) * tp_size + tp) * ep_size + ep)`

(`context_parallel_size` defaults to `1` in this tutorial.)

## 2) Communication domains

`examples/train_4d.py` uses:

- `tensor_model_parallel_group`: fixed `(dp, pp, ep)`, varying `tp`
- `pipeline_model_parallel_group`: fixed `(dp, tp, ep)`, varying `pp`
- `expert_model_parallel_group`: fixed `(dp, pp, tp)`, varying `ep`
- `data_parallel_group`: fixed `(pp, tp, ep)`, varying `dp`
- `expert_data_parallel_group`: fixed `(pp, ep)`, varying `(dp, tp)`

## 3) What PP changes in model execution

With `pipeline_model_parallel_size > 1`:

- Stage 0 owns embeddings + its local decoder block range.
- Middle stages own only local decoder block ranges.
- Last stage owns local decoder block range + final norm + LM head.

Each rank runs `DeepSeekModel.forward_stage(...)` for its local stage only.

## 4) 1F1B non-interleaved schedule

This tutorial uses non-interleaved 1F1B:

1. Warmup forwards
2. Steady-state alternating forward/backward
3. Cooldown backwards

Per-microbatch PP tensors:

- Activations between stages: `[B_mb, S, H]`
- Activation gradients between stages: `[B_mb, S, H]`
- Labels sent to last stage: `[B_mb, S]`

## 5) Label path in this tutorial

- Stage 0 reads `input_ids`.
- Last stage computes language-model loss.

So labels are sent stage-0 -> last-stage per microbatch along the same
`(dp, tp, ep)` chain.

## 6) Backward and gradient synchronization

Pipeline parallelism adds p2p tensor transfer, not parameter-gradient all-reduce.

After all microbatches of a step:

- Non-expert params sync on `data_parallel_group`.
- Expert params sync on `expert_data_parallel_group`.
- TP collectives remain inside TP layers.
- MoE dispatch/return collectives remain inside MoE on `expert_model_parallel_group`.

## 7) Where extra communication appears

Compared with TP+EP+DP without PP, PP adds:

- Forward activation sends to next stage
- Backward activation-gradient sends to previous stage
- Label sends from first stage to last stage

With the `deepseek_v3` precision recipe enabled, MoE dispatch/combine payload tensors may also
use FP8 communication quantization on the expert-parallel path. Metadata tensors remain
unquantized, and collective ordering does not change.

For module-level DeepSeek precision control, see
[DeepSeek Precision Configuration](deepseek_precision_configuration.md).

All other TP/EP/DP collectives remain unchanged.

## 8) Practical checklist

1. `world_size % (tensor_model_parallel_size * pipeline_model_parallel_size * expert_model_parallel_size * context_parallel_size) == 0`
2. Current tutorial guard: do not combine `tensor_model_parallel_size > 1` with `expert_model_parallel_size > 1`
3. `num_microbatches >= 1` and `batch_size % num_microbatches == 0`
4. `num_experts % expert_model_parallel_size == 0`
5. `expert_tensor_parallel_size == 1` for routed experts in this tutorial
6. `seq_len % tensor_model_parallel_size == 0` when MoE sequence parallel is active
7. `dropout == 0.0` whenever tensor or expert model parallel is greater than 1
