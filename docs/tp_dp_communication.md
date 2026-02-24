# TP + DP Backward Flow in LLM Training (Summary)

This note summarizes how tensor parallel (TP) and data parallel (DP) interact during
forward/backward in modern LLM training, focusing on where gradients live and which
collectives are required.

---

## 1) Mental model: a 2D process grid

Think of ranks arranged as a grid:

- DP dimension: replicate the model (including TP sharding) over different data.
- TP dimension: shard tensors/weights within a layer across GPUs.

Each GPU has coordinates: `(dp_rank, tp_rank)`.

Rule of thumb:
- TP collectives happen within a DP replica (fixed `dp_rank`, varying `tp_rank`).
- DP collectives happen across replicas for the same shard (fixed `tp_rank`, varying
  `dp_rank`).

---

## 2) Anchor identities (single linear layer)

For a linear layer:

- Forward: `Y = XW`
- Backward:
  - `dW = X^T dY`
  - `dX = dY W^T`

The math is unchanged under parallelism; what changes is where pieces of
`W, Y, dY` reside.

---

## 3) Two standard TP sharding patterns

### A) Column-parallel linear (shard output features)

Shard `W` by columns (output dimension):

`W = [W_1, ..., W_T],  W_i in R^(d_in x d_out/T)`

Forward:
- Each TP rank computes local output shard: `Y_i = X W_i`
- Often no gather is needed if the next op consumes the same sharding.

Backward:
- Local parameter grad (no TP comm): `dW_i = X^T dY_i`
- Each rank computes partial input grad: `dX_i = dY_i W_i^T`
- Full `dX` needs a sum across TP:

`dX = sum_{i=1..T} dX_i`

=> TP all-reduce (sum) on `dX`

Key takeaway: Column-parallel backward requires an all-reduce on activation gradient
`dX`.

### B) Row-parallel linear (shard input features)

Shard `W` by rows (input dimension):

`W = [W_1; ...; W_T],  W_i in R^(d_in/T x d_out)`

Input activations are sharded consistently: `X = [X_1, ..., X_T]`.

Forward:
- Each TP rank computes partial output: `Y~_i = X_i W_i`
- Full output is a sum:

`Y = sum_{i=1..T} Y~_i`

=> TP all-reduce (sum) on `Y` (so every TP rank sees the same `Y`)

Backward:
- `dY` is replicated on TP ranks (because `Y` was all-reduced).
- Local parameter grad: `dW_i = X_i^T dY`
- Local input grad shard: `dX_i = dY W_i^T`
- Typically no TP reduction is needed for `dX` (it is naturally sharded).

Key takeaway: Row-parallel needs an all-reduce on forward output `Y`; backward is
mostly local.

---

## 4) How DP fits: gradients vs activation-gradients

Inside each DP replica (TP correctness):
Within a given `dp_rank`, TP collectives ensure activation-gradient correctness.

- Row-parallel: TP all-reduce on `Y` in forward -> replicated `dY` in backward.
- Column-parallel: TP all-reduce on `dX` in backward.

Across DP replicas (replica consistency):
DP is about parameter gradients.

For each parameter shard on TP rank `i`:

`dW_i <- (1/DP) * sum_{r=1..DP} dW_i^(r)`

=> DP all-reduce (sum/avg) on parameter gradients, across `dp_rank` with the same
`tp_rank`.

Important distinction:
- TP does not combine `dW` across TP ranks (they are different shards).
- DP does combine `dW` across DP ranks (same shard, different data).

---

## 5) Optimizer step (who updates what)

After DP all-reduce, each GPU holds the averaged gradients for the parameter shards it
owns.

- The optimizer (Adam/SGD/etc.) runs locally on each rank's shards.
- No broadcast is needed: DP all-reduce makes updates consistent across replicas.

With gradient accumulation, DP all-reduce is commonly done once per optimizer step
(after accumulating several microbatches), not after every microbatch.

---

## 6) Concrete TP=2, DP=2 example (world=4)

One common mapping:

- TP groups (within a DP replica):
  - DP replica 0: ranks `{0,1}`
  - DP replica 1: ranks `{2,3}`
- DP all-reduce groups (same TP index across replicas):
  - tp=0: ranks `{0,2}`
  - tp=1: ranks `{1,3}`

So during backprop:
- TP collectives occur in `{0,1}` and `{2,3}` (e.g., all-reduce `dX` for
  column-parallel).
- DP all-reduce of parameter grads occurs in `{0,2}` and `{1,3}` before the
  optimizer step.

---

## 7) Where pipeline parallel (PP) fits (quick note)

PP is different from TP/DP reductions:
- Forward: stages send activations to next stage.
- Backward: stages send activation gradients back.

Within each stage you still apply TP logic above; DP syncing still applies to parameter
 grads.

---

## 8) Who reduces what (cheat sheet)

| Parallelism | What it synchronizes | Typical collective |
|---|---|---|
| TP (row-parallel) | Build full `Y` from partials | all-reduce(sum) on `Y` |
| TP (col-parallel) | Build full `dX` from partials | all-reduce(sum) on `dX` |
| DP | Make parameter grads consistent across replicas | all-reduce(avg) on `dW`, etc. |
| PP | Move tensors between pipeline stages | send/recv activations and `dX` |

---

## 9) Practical implementation notes

- Overlap: frameworks bucket gradients and start DP all-reduce early to overlap comm
  with backprop.
- Loss scalar: DP-reducing the loss is optional (mainly for logging). Correctness
  relies on DP-sync of gradients.
- Clipping: global grad norm may require an additional reduction (often DP, sometimes
  across more dims depending on sharding).

---

If you want, extend this summary to a full Transformer block (attention + MLP) showing
exactly which projections are row/col-parallel and where TP collectives land.
