# TP + EP + DP Communication Guide

**Purpose**: Explain communication domains and gradient synchronization rules in the
`examples/ep.py` tutorial for combined TP+EP+DP training.

**Audience**: Readers learning how EP changes communication relative to TP+DP.

**Prerequisites**: Understanding of TP/DP basics and MoE top-k routing.

**Related Docs**:
- [TP + DP Backward Flow](tp_dp_communication.md)
- [DeepSeekMoE Auxiliary Losses](deepseek_moe_aux_losses.md)

## Table of Contents

- [1) 3D process grid](#1-3d-process-grid)
- [2) What TP does in this setup](#2-what-tp-does-in-this-setup)
- [3) What EP adds](#3-what-ep-adds)
- [4) EP batch semantics: split vs shared](#4-ep-batch-semantics-split-vs-shared)
- [5) Capacity and dropping](#5-capacity-and-dropping)
- [6) Where gradients are synchronized](#6-where-gradients-are-synchronized)
- [7) Why no WORLD loss all-reduce is required](#7-why-no-world-loss-all-reduce-is-required)
- [8) Practical interpretation](#8-practical-interpretation)
- [9) Router auxiliary losses](#9-router-auxiliary-losses)

This note summarizes the canonical communication domains for combined Tensor Parallelism
(TP), Expert Parallelism (EP), and Data Parallelism (DP), as implemented by `examples/ep.py`.


## 1) 3D process grid

Each rank has coordinates `(dp_rank, tp_rank, ep_rank)`.

- TP dimension: shard dense linear tensors inside a replica.
- EP dimension: shard routed experts across expert ranks.
- DP dimension: replicate the full TP+EP shard layout over data shards.

Rule of thumb:

- Dense TP collectives: fixed `(dp_rank, ep_rank)`, varying `tp_rank`.
- Attention TP collectives:
  - `--attn_tp_axis tp`: fixed `(dp_rank, ep_rank)`, varying `tp_rank`
  - `--attn_tp_axis ep`: fixed `(dp_rank, tp_rank)`, varying `ep_rank`
- EP collectives: fixed `(dp_rank, tp_rank)`, varying `ep_rank`.
- DP collectives: fixed `(tp_rank, ep_rank)`, varying `dp_rank`.

## 2) What TP does in this setup

Dense TP layers follow the same rules as TP-only training:

- Row-parallel linear: forward all-reduce(sum) on partial output `Y` in TP group.
- Column-parallel linear: backward all-reduce(sum) on `dX` in TP group.

These TP collectives stay inside TP layer implementations.

## 3) What EP adds

EP adds routed-expert communication around the MoE layer.

Within each `(dp_rank, tp_rank)` EP group:

1. Each EP rank routes its local token shard to global expert ids.
2. **Dispatch all-to-all**: send token activations to owner ranks of selected experts.
3. Local experts run on received tokens (`expert_tp=1`, no TP inside experts).
4. **Return all-to-all**: send expert outputs back to source ranks.
5. Source ranks merge expert contributions with top-k weights.

These two all-to-all collectives are the main new communication introduced by EP.

## 4) EP batch semantics: split vs shared

`examples/ep.py` supports two EP batch behaviors, inferred from `--attn_tp_axis`:

- `--attn_tp_axis tp` -> `split` mode:
  each EP rank gets a disjoint DP-local batch shard (`B -> B_ep`).
- `--attn_tp_axis ep` -> `shared` mode:
  each EP rank sees the full DP-local batch (`B` on every EP rank).
  This is required when attention heads are TP-sharded across the EP axis.

Interpretation:

- In `split` mode, EP ranks are independent for attention input ownership.
- In `shared` mode, EP ranks cooperate on attention TP, so attention input must be shared.

## 5) Capacity and dropping

Per-expert capacity is:

`capacity = ceil(capacity_factor * total_tokens * top_k / num_experts)`

Assignments above capacity are dropped deterministically by routing weight
(highest kept first), and dropped fraction is logged.

## 6) Where gradients are synchronized

After backward in TP+EP+DP:

- `split` mode:
  - EP all-reduce(avg): non-EP-axis and non-expert parameters
    (EP ranks processed different data shards).
  - DP all-reduce(avg): all parameters across DP replicas.
  - Expert params: additional `1/ep_size` scaling after DP reduce, because each rank
    used mean loss over its local shard.
- `shared` mode:
  - No EP gradient all-reduce for non-expert params (EP ranks saw the same batch).
  - DP all-reduce(avg): all parameters across DP replicas.
  - Expert params: additional `1/ep_size` scaling, because EP dispatch/return duplicates
    source-token contributions across EP ranks in this tutorial setup.

`expert_tp=1` means expert MLP weights are local to one EP rank and never TP-sharded.

## 7) Why no WORLD loss all-reduce is required

Loss all-reduce is optional for logging.
Training correctness comes from correct gradient synchronization domains above.

## 8) Practical interpretation

Compared with TP+DP, EP introduces:

- Additional forward communication: dispatch + return all-to-all around MoE.
- Additional backward pressure through those all-to-all paths.
- Additional non-expert EP gradient synchronization due token sharding by EP rank.

That is the core communication cost/benefit tradeoff of EP.

## 9) Router auxiliary losses

For DeepSeekMoE-style load-balance losses (expert-level vs device-level) and how
they map to this repo's current code, see:

- [DeepSeekMoE Aux Losses](deepseek_moe_aux_losses.md)
