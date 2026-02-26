# ZeRO-1 and ZeRO-2 (Intuitive Summary)

**Purpose**: Build intuition for ZeRO-1 and ZeRO-2 using small tensor examples.

**Audience**: Learners reading the nano-train ZeRO implementation in `src/distributed/zero.py`.

**Prerequisites**: Basic DP training knowledge and familiarity with all-reduce/all-gather.

**Related Docs**:
- [Megatron ZeRO-1/2 Design](megatron_zero1_zero2_design.md)
- [ZeRO-1/2 Quickstart](zero1_zero2_quickstart.md)
- [TP + PP + EP + DP Communication Guide](pp_tp_ep_dp_communication.md)

## Table of Contents

1. [The Three Piles of Training State](#1-the-three-piles-of-training-state)
2. [Collectives with Tiny Examples](#2-collectives-with-tiny-examples)
3. [One DP Group, One Ownership Map](#3-one-dp-group-one-ownership-map)
4. [ZeRO-1 (`optim`): Partition O Only](#4-zero-1-optim-partition-o-only)
5. [ZeRO-2 (`optim_grads`): Partition O and G](#5-zero-2-optim_grads-partition-o-and-g)
6. [Side-by-Side Cheat Sheet](#6-side-by-side-cheat-sheet)
7. [Visual Data Flow](#7-visual-data-flow)
8. [Repo Mapping: DP vs EDP Domains](#8-repo-mapping-dp-vs-edp-domains)
9. [Key Takeaways](#9-key-takeaways)

## 1) The Three Piles of Training State

For each parameter tensor, training state can be grouped as:

- `P`: parameters (weights used in forward/backward, usually bf16/fp16)
- `G`: gradients (`dP`)
- `O`: optimizer states (for Adam: `m`, `v`, and FP32 master parameter shards)

Classic DP replicates `P + G + O` on every data-parallel rank and uses all-reduce on gradients.

ZeRO saves memory by partitioning selected piles across ranks.

## 2) Collectives with Tiny Examples

Assume a DP group has two ranks: `A` and `B`.

### 2.1 All-reduce (sum/avg)

Everyone ends up with the same reduced tensor.

- `A`: `[1 1 1 1  0 0 0 0]`
- `B`: `[0 0 0 0  2 2 2 2]`

After all-reduce(sum), both ranks hold:

- `[1 1 1 1  2 2 2 2]`

### 2.2 All-gather

Everyone reconstructs full tensor from partitioned shards.

- `A` shard: `[10 11 12 13]`
- `B` shard: `[20 21 22 23]`

After all-gather, both ranks hold:

- `[10 11 12 13  20 21 22 23]`

### 2.3 Reduce-scatter

Reduce first, then scatter reduced result by shard ownership.

Using the all-reduce example:

- Reduced sum: `[1 1 1 1  2 2 2 2]`
- `A` receives: `[1 1 1 1]`
- `B` receives: `[2 2 2 2]`

### 2.4 Useful relationship

Conceptually:

`all-reduce ~= reduce-scatter + all-gather`

## 3) One DP Group, One Ownership Map

Assume one DP group `{A, B}` and a flattened parameter:

`W = [w0 w1 w2 w3 w4 w5 w6 w7]`

Ownership map:

- `A` owns indices `0..3`
- `B` owns indices `4..7`

```text
W elements: [ w0  w1  w2  w3 | w4  w5  w6  w7 ]
owner:        A   A   A   A  |  B   B   B   B
```

This ownership is optimizer responsibility within a DP/EDP group, not TP/PP sharding.

## 4) ZeRO-1 (`optim`): Partition O Only

### 4.1 Partitioned vs replicated

- Partitioned: `O`
- Replicated: `P`
- Gradients: reduced globally via all-reduce, then sliced for local update

### 4.2 Iteration flow

1. Backward produces local full grads on each rank.
2. All-reduce(avg) makes global averaged grads visible on all ranks.
3. Each rank updates only its owned shard using local optimizer state shard.
4. All-gather updated parameter shards to reconstruct full `P` for next forward.

### 4.3 Why extra communication appears

Because update responsibility is sharded, ranks must exchange updated parameter shards to
reconstruct full parameters for compute.

## 5) ZeRO-2 (`optim_grads`): Partition O and G

### 5.1 Partitioned vs replicated

- Partitioned: `O + G`
- Replicated: `P`

### 5.2 Iteration flow

1. Backward produces local full grads.
2. Reduce-scatter(avg) produces only local reduced grad shard on each rank.
3. Each rank updates only owned parameter shard using local grad shard + local optimizer shard.
4. All-gather updated parameter shards to reconstruct full `P`.

### 5.3 Main difference from ZeRO-1

- ZeRO-1: grads are all-reduced (replicated reduced view before slicing)
- ZeRO-2: grads are directly sharded by reduce-scatter

So ZeRO-2 saves both optimizer-state memory and gradient memory.

## 6) Side-by-Side Cheat Sheet

| Item | Classic DP | ZeRO-1 (`optim`) | ZeRO-2 (`optim_grads`) |
|---|---|---|---|
| Parameters `P` | replicated | replicated | replicated |
| Gradients `G` | replicated | all-reduced then sliced | partitioned after reduce-scatter |
| Optimizer states `O` | replicated | partitioned | partitioned |
| Key grad collective | all-reduce | all-reduce | reduce-scatter |
| Need param all-gather after update | no | yes | yes |
| Memory savings | baseline | optimizer state | optimizer state + gradients |

## 7) Visual Data Flow

### ZeRO-1 (`optim`)

```text
local grads
A: g^(A) --\
            +--> all-reduce(avg) --> both ranks see reduced full grad
B: g^(B) --/

sharded update
A updates owned shard (e.g., w0..w3), B updates owned shard (e.g., w4..w7)

re-replicate parameters
updated shards --> all-gather --> both ranks hold full updated W
```

### ZeRO-2 (`optim_grads`)

```text
local grads
A: g^(A) --\
            +--> reduce-scatter(avg) --> A gets grad shard 0..3
B: g^(B) --/                           B gets grad shard 4..7

sharded update
A updates owned shard, B updates owned shard

re-replicate parameters
updated shards --> all-gather --> both ranks hold full updated W
```

## 8) Repo Mapping: DP vs EDP Domains

In this repo:

- Dense parameters synchronize on DP groups.
- Expert parameters synchronize on EDP (expert-data-parallel) groups.

`src/distributed/zero.py` selects group per parameter by `is_expert_param` metadata:

- dense param -> `data_parallel_group`
- expert param -> `expert_data_parallel_group`

So the same ZeRO-1/2 logic applies, but communication domain depends on parameter family.

## 9) Key Takeaways

- All-reduce: everyone gets full reduced tensor.
- Reduce-scatter: everyone gets only owned shard of reduced tensor.
- All-gather: everyone reconstructs full tensor from shards.
- ZeRO-1 shards optimizer state (`O`) and still all-gathers updated params.
- ZeRO-2 shards optimizer state plus gradients (`O + G`) and still all-gathers updated params.

ZeRO-3 is intentionally out of scope in nano-train v1.
