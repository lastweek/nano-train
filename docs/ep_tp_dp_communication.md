# TP + EP + DP Communication Guide

This note summarizes the canonical communication domains for combined
Tensor Parallelism (TP), Expert Parallelism (EP), and Data Parallelism (DP),
as implemented by `examples/ep.py`.

## 1) 3D process grid

Each rank has coordinates `(dp_rank, tp_rank, ep_rank)`.

- TP dimension: shard dense linear tensors inside a replica.
- EP dimension: shard routed experts across expert ranks.
- DP dimension: replicate the full TP+EP shard layout over data shards.

Rule of thumb:

- TP collectives: fixed `(dp_rank, ep_rank)`, varying `tp_rank`.
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

## 4) Capacity and dropping

Per-expert capacity is:

`capacity = ceil(capacity_factor * total_tokens * top_k / num_experts)`

Assignments above capacity are dropped deterministically by routing weight
(highest kept first), and dropped fraction is logged.

## 5) Where gradients are synchronized

After backward in TP+EP+DP:

- DP all-reduce(avg): all parameters across DP group.
- EP all-reduce(avg): non-expert (non-EP-sharded) parameters across EP group,
  because each EP rank processes different token shards.
- EP expert-sharded parameters: no EP all-reduce (each rank owns distinct experts).

`expert_tp=1` means expert MLP weights are local to one EP rank and never TP-sharded.

## 6) Why no WORLD loss all-reduce is required

Loss all-reduce is optional for logging.
Training correctness comes from correct gradient synchronization domains above.

## 7) Practical interpretation

Compared with TP+DP, EP introduces:

- Additional forward communication: dispatch + return all-to-all around MoE.
- Additional backward pressure through those all-to-all paths.
- Additional non-expert EP gradient synchronization due token sharding by EP rank.

That is the core communication cost/benefit tradeoff of EP.
