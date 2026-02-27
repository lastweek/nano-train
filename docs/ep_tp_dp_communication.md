# TP + EP + DP Communication Guide (Megatron Naming)

**Purpose**: Explain communication domains and gradient synchronization rules in
`examples/train_4d.py` when `pipeline_model_parallel_size == 1`.

**Audience**: Readers learning where expert parallelism adds communication.

**Prerequisites**:
- [TP + DP Backward Flow](tp_dp_communication.md)
- Basic MoE top-k routing intuition

**Related Docs**:
- [TP + PP + EP + DP Communication Guide](pp_tp_ep_dp_communication.md)
- [DeepSeekMoE Auxiliary Losses](deepseek_moe_aux_losses.md)

## Table of Contents

- [1) 3D process grid](#1-3d-process-grid)
- [2) Communication domains](#2-communication-domains)
- [3) MoE dispatch, expert compute, combine](#3-moe-dispatch-expert-compute-combine)
- [4) Sequence-parallel behavior around MoE](#4-sequence-parallel-behavior-around-moe)
- [5) Gradient synchronization domains](#5-gradient-synchronization-domains)
- [6) Why no WORLD loss all-reduce is required](#6-why-no-world-loss-all-reduce-is-required)
- [7) Practical checklist](#7-practical-checklist)

## 1) 3D process grid

Each rank has coordinates:
`(data_parallel_rank, tensor_model_parallel_rank, expert_model_parallel_rank)`.

- Tensor model parallel: shard dense/attention linear layers.
- Expert model parallel: shard routed experts.
- Data parallel: replicate the same TP+EP layout across data shards.

## 2) Communication domains

`examples/train_4d.py` uses these groups:

- `tensor_model_parallel_group`: fixed `(dp, ep)`, varying `tp`
- `expert_model_parallel_group`: fixed `(dp, tp)`, varying `ep`
- `data_parallel_group`: fixed `(tp, ep)`, varying `dp`
- `expert_data_parallel_group`: fixed `ep`, varying `(dp, tp)`

The key distinction is:

- EP collectives inside MoE run on `expert_model_parallel_group`.
- Expert parameter gradients synchronize on `expert_data_parallel_group`.

## 3) MoE dispatch, expert compute, combine

Inside each routed MoE layer:

1. Route tokens to top-k expert ids.
2. Dispatch token assignments with all-to-all on `expert_model_parallel_group`.
3. Run local expert MLPs (`expert_tensor_parallel_size = 1`).
4. Return expert outputs with all-to-all on the same group.
5. Combine returned outputs with router weights.

## 4) Sequence-parallel behavior around MoE

When `tensor_model_parallel_size > 1` and sequence parallel is enabled, the model:

1. Scatters sequence/tokens across TP ranks before MoE.
2. Runs MoE on local sequence shard.
3. Gathers sequence back across TP ranks after MoE.

This avoids duplicate routed-expert token compute across TP ranks.

## 5) Gradient synchronization domains

After `loss.backward()`:

- Non-expert parameters: all-reduce(avg) on `data_parallel_group`.
- Expert parameters: all-reduce(avg) on `expert_data_parallel_group`.

No script-level TP all-reduce is added here; TP collectives stay inside TP layers.

## 6) Why no WORLD loss all-reduce is required

Loss all-reduce is optional for logging only. Training correctness comes from
parameter-gradient synchronization in the correct domains above.

## 7) Practical checklist

1. `world_size % (tensor_model_parallel_size * pipeline_model_parallel_size * expert_model_parallel_size * context_parallel_size) == 0`
2. Current tutorial guard: do not combine `tensor_model_parallel_size > 1` with `expert_model_parallel_size > 1`
3. `num_experts % expert_model_parallel_size == 0`
4. `num_heads % tensor_model_parallel_size == 0`
5. `seq_len % tensor_model_parallel_size == 0` when MoE sequence parallel is enabled
6. `expert_tensor_parallel_size == 1` in this tutorial implementation
7. `dropout == 0.0` whenever tensor or expert model parallel is greater than 1
