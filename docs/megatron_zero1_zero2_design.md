# Megatron ZeRO-1/2 Design (Implementation-Level)

**Purpose**: Explain how Megatron-Core implements ZeRO-1/2 in code, then map the design to
`nano-train` so implementation work can proceed without architectural ambiguity.

**Audience**: Engineers implementing distributed optimizer sharding in this repo.

**Prerequisites**: Familiarity with DP/TP/PP/EP, reduce-scatter/all-gather, and mixed-precision
training.

**Scope**:
- In-scope: ZeRO-1 and ZeRO-2 mechanics, communication domains, group semantics, checkpoint model.
- Out-of-scope: ZeRO-3 internals, offload engines, performance tuning beyond synchronous baseline.

**Related Docs**:
- [Documentation Index](README.md)
- [TP + DP Communication Guide](tp_dp_communication.md)
- [TP + EP + DP Communication Guide](ep_tp_dp_communication.md)

## Table of Contents

1. [Terminology Contract](#terminology-contract)
2. [Megatron Process-Group Topology](#megatron-process-group-topology)
3. [Core Abstractions and Responsibilities](#core-abstractions-and-responsibilities)
4. [ZeRO-1 vs ZeRO-2 in Megatron](#zero-1-vs-zero-2-in-megatron)
5. [Step-by-Step Runtime Timeline (ZeRO-2, Synchronous Baseline)](#step-by-step-runtime-timeline-zero-2-synchronous-baseline)
6. [MoE Interaction](#moe-interaction)
7. [Pipeline Interaction](#pipeline-interaction)
8. [Checkpoint Semantics](#checkpoint-semantics)
9. [Configuration Matrix](#configuration-matrix)
10. [Failure Modes and Invariants](#failure-modes-and-invariants)
11. [Nano-Train Adoption Notes](#nano-train-adoption-notes)
12. [Primary Source References](#primary-source-references)

## Terminology Contract

- `DP` (data parallel): replicated model copies over different data shards.
- `TP` (tensor model parallel): intra-layer tensor sharding.
- `PP` (pipeline model parallel): layer-depth sharding across stages.
- `CP` (context parallel): sequence-length sharding for attention context.
- `EP` (expert model parallel): MoE expert sharding domain.
- `EDP` (expert data parallel): replica domain for expert-sharded parameters.
- `distributed optimizer instance`: Megatron concept for splitting DP into optimizer subdomains when
  `num_distributed_optimizer_instances > 1`.

ZeRO stages in this doc:
- **ZeRO-1**: shard optimizer states (and optimizer-owned master states), gradients replicated.
- **ZeRO-2**: shard optimizer states + gradients; params still gathered for forward execution.

## Megatron Process-Group Topology

Megatron builds groups from `initialize_model_parallel(...)` in
`megatron/core/parallel_state.py`, with explicit dimensions:

- `tensor_model_parallel_size`
- `pipeline_model_parallel_size`
- `context_parallel_size`
- `expert_model_parallel_size`
- (derived) data-parallel sizes and expert-data-parallel sizes

Group access and assembly are centralized via:
- `megatron/core/parallel_state.py`
- `megatron/core/process_groups_config.py` (`ProcessGroupCollection`)

Important groups for ZeRO-1/2 behavior:

| Group | Typical use |
|---|---|
| `dp_group` / `dp_cp_group` | Dense parameter gradient synchronization domain |
| `expt_dp_group` | Expert-parameter gradient synchronization domain |
| `intra_dp_cp_group` | Dense gradient reduce-scatter/all-reduce domain when distributed optimizer is active |
| `intra_expt_dp_group` | Expert gradient reduce-scatter/all-reduce domain |
| `inter_dist_opt_group` | Cross-instance sync when multiple distributed optimizer instances are configured |

### Concrete small-world example (illustrative)

Assume:
- `world_size=8`, `TP=1`, `PP=1`, `CP=1`, `EP=2`, `num_distributed_optimizer_instances=1`

One valid layout has:
- Dense DP reduction domain: ranks `{0,1,2,3,4,5,6,7}`
- Expert EDP domains:
  - expert-shard-0 replicas: `{0,2,4,6}`
  - expert-shard-1 replicas: `{1,3,5,7}`

Interpretation:
- Dense params reduce over the dense DP group.
- Expert params reduce over their corresponding EDP group, not the full dense DP group.

Exact rank IDs depend on parallel-order configuration, but the communication domains above are the
invariant.

## Core Abstractions and Responsibilities

### 1) `DistributedDataParallel` (Megatron wrapper)

File:
- `megatron/core/distributed/distributed_data_parallel.py`

Responsibilities:
- Build and own contiguous param/grad buffers for dense and expert params.
- Bucket gradients and dispatch collectives.
- Register backward hooks to stage `param.main_grad`.
- Provide explicit sync entry points:
  - `DistributedDataParallel.start_grad_sync`
  - `DistributedDataParallel.finish_grad_sync`
  - `DistributedDataParallel.start_param_sync`

Key implementation behavior:
- Dense/expert params are split by `param.allreduce` flag and placed into separate buffer sets.
- Collective domain selection is parameter-type aware (dense uses DP/DP+CP, expert uses EDP).
- Supports overlap knobs, but can run purely synchronous.

### 2) `ParamAndGradBuffer` + bucket groups

Files:
- `megatron/core/distributed/param_and_grad_buffer.py`
- `partition_buckets(...)` usage in DDP wrapper

Responsibilities:
- Keep parameter and gradient storage contiguous and indexable.
- Track per-parameter ranges inside bucket/global buffers.
- Enable reduce-scatter/all-reduce/all-gather per bucket.

Practical consequence:
- Distributed optimizer can convert between model-param views and optimizer-owned shards without
  per-parameter ad-hoc copies.

### 3) `DistributedOptimizer`

File:
- `megatron/core/optimizer/distrib_optimizer.py`

Responsibilities:
- Build model-param-to-buffer/shard range maps.
- Materialize optimizer-owned main parameter shards.
- Copy reduced model grads into local main-grad shards:
  - `DistributedOptimizer._copy_model_grads_to_main_grads`
- Run local optimizer step on owned shards.
- Copy updated shard data back to param buffers.
- Trigger all-gather of model params for next forward:
  - `DistributedOptimizer.step_with_ready_grads`
- Persist and restore sharded optimizer parameter state:
  - `DistributedOptimizer.get_parameter_state_dp_zero`
  - `DistributedOptimizer.load_parameter_state_from_dp_zero`

## ZeRO-1 vs ZeRO-2 in Megatron

| Aspect | ZeRO-1 | ZeRO-2 |
|---|---|---|
| Optimizer state | Sharded | Sharded |
| Gradients | Replicated reduction result | Sharded via reduce-scatter path |
| Communication shape | Heavier full-domain gradient sync | Reduce-scatter for grad shards, then all-gather params |
| Memory pressure | Lower than DDP | Lower than ZeRO-1 |

Why reduce-scatter + all-gather appears:
1. Backward produces gradient information for full model parameters.
2. ZeRO-2 partitions reduced gradients so each rank keeps only its shard for optimizer update.
3. After local shard update, forward still needs full model-param visibility in replica domains,
   so params are all-gathered (or overlapped equivalent).

## Step-by-Step Runtime Timeline (ZeRO-2, Synchronous Baseline)

The sequence below intentionally excludes overlap knobs.

1. **Forward** runs with model params available in local param buffers.
2. **Backward hooks** accumulate grads into contiguous grad buffers (`param.main_grad` staging).
3. `DistributedDataParallel.start_grad_sync` launches gradient collective work.
4. `DistributedDataParallel.finish_grad_sync` completes gradient synchronization.
5. `DistributedOptimizer._copy_model_grads_to_main_grads` extracts local shard grads from reduced
   grad buffers to optimizer-owned shard tensors.
6. Local optimizer step updates only the shard owned by this rank.
7. Updated main-param shard is copied back into param buffer shard view.
8. `DistributedOptimizer.step_with_ready_grads` triggers parameter synchronization
   (`start_param_sync`) so next forward sees the updated distributed parameter view.
9. Grad buffers are reset for the next iteration.

### Per-iteration pseudocode (ZeRO-2 synchronous baseline)

```python
# inputs: model (Megatron DDP wrapper), dist_optim (Megatron DistributedOptimizer)

for step in training_steps:
    model.zero_grad_buffer()

    # Forward + backward
    loss = model(batch)
    loss.backward()

    # Synchronous gradient synchronization on dense + expert domains
    model.start_grad_sync()
    model.finish_grad_sync()

    # Dist-optimizer internals (conceptual)
    dist_optim._copy_model_grads_to_main_grads()
    # local shard-only optimizer update (inside step_with_ready_grads)
    dist_optim.step_with_ready_grads()
    # step_with_ready_grads calls/coordinates param sync (all-gather path)
```

## MoE Interaction

Megatron distinguishes dense vs expert parameters by marking expert params as not belonging to the
dense all-reduce path (`param.allreduce=False` pattern in wrappers/extensions).

Result:
- Dense gradients synchronize on DP/DP+CP domain.
- Expert gradients synchronize on EDP domain.

This separation is reflected directly in DDP buffer allocation and in gradient scaling logic that
keeps dense and expert reductions numerically consistent with global loss scaling.

## Pipeline Interaction

For ZeRO-1/2, pipeline-relevant knobs in Megatron DDP/distributed optimizer paths include:
- `overlap_grad_reduce`
- `overlap_param_gather`
- bucket sizing and scheduling behavior by PP rank

For this repo’s first ZeRO implementation, use **synchronous baseline**:
- No overlap required.
- Keep ordering deterministic.
- Add overlap as a separate phase after correctness parity.

## Checkpoint Semantics

Megatron separates checkpoint concerns:

1. **Non-parameter optimizer state** via optimizer `state_dict()` path.
2. **Parameter-dependent sharded optimizer state** via dedicated distributed-optimizer save/load
   paths (DP-zero style gather/scatter flows).

Relevant distributed-optimizer APIs:
- `get_parameter_state_dp_zero(...)`
- `load_parameter_state_from_dp_zero(...)`

This separation is important for large-scale resume because parameter shards are not represented as
ordinary replicated optimizer tensors.

## Configuration Matrix

Core runtime switches (Megatron):

| Flag | Meaning |
|---|---|
| `--use-distributed-optimizer` | Enable distributed optimizer path |
| `--data-parallel-sharding-strategy no_shard` | DP replication baseline |
| `--data-parallel-sharding-strategy optim` | ZeRO-1-like optimizer-state sharding |
| `--data-parallel-sharding-strategy optim_grads` | ZeRO-2-like optimizer+gradient sharding |
| `--data-parallel-sharding-strategy optim_grads_params` | ZeRO-3-like scope (out-of-scope here) |
| `--overlap-grad-reduce` | overlap gradient reduction with backward |
| `--overlap-param-gather` | overlap parameter all-gather with forward |

Related dimensions:
- `tensor-model-parallel-size`
- `pipeline-model-parallel-size`
- `context-parallel-size`
- `expert-model-parallel-size`
- `num-distributed-optimizer-instances`

## Failure Modes and Invariants

Critical invariants:
1. Collective ordering must be identical across participating ranks.
2. Dense params must not be reduced on EDP domain; expert params must not be reduced on dense DP
   domain.
3. Param-to-buffer shard ranges must be stable and correct before optimizer step.
4. Loss-scaling and gradient scaling factors must remain consistent across dense/expert paths.

Common failure modes:
- Group mismatch (wrong process group for a parameter family).
- Missing grad placeholders causing rank divergence in collective sequence.
- Wrong shard-range mapping leading to silent optimizer corruption.
- Mixing overlap behavior into an unverified baseline path.

## Nano-Train Adoption Notes

### Current Nano-Train status (v1)

The repository now includes a synchronous Megatron-style ZeRO-1/2 baseline:

- Core optimizer: `src/distributed/zero.py` (`MegatronZeroOptimizer`)
- Tutorial integration: `examples/train_4d.py` (`--use-distributed-optimizer`,
  `--data-parallel-sharding-strategy`)
- Trainer hook: `src/trainer.py` calls `step_with_ready_grads()` when available
- Checkpoint format: `optimizer_nonparam.pt` + `optimizer_manifest.json` +
  per-rank `optimizer_shard_rank{rank}.pt`

Current v1 limitations:

- `num_distributed_optimizer_instances == 1` only
- Synchronous communication only (no overlap knobs in the implementation path)
- Strategies limited to `optim` (ZeRO-1) and `optim_grads` (ZeRO-2)
- ZeRO-3 (`optim_grads_params`) is intentionally out of scope

### Minimal mapping to this repo

- Group topology and naming:
  - [src/distributed/topology.py](../src/distributed/topology.py)
- DDP-like orchestration and bucketed grad/param buffers:
  - currently no direct equivalent; implement new module under `src/distributed/`
- Optimizer integration:
  - [src/optimizer.py](../src/optimizer.py) should gain distributed-optimizer entrypoints
- Training loop integration:
  - [examples/train_4d.py](../examples/train_4d.py) and shared training paths should call explicit
    `start_grad_sync/finish_grad_sync/step_with_ready_grads` style APIs

### What to implement first (ordered)

1. **ZeRO-1 baseline**:
  - shard optimizer states in DP domain
  - keep gradients fully reduced/replicated
  - validate checkpoint save/load for sharded optimizer state
2. **ZeRO-2 baseline**:
  - add gradient shard ownership and reduce-scatter path
  - preserve synchronous ordering only
  - maintain explicit dense vs expert reduction domains
3. **Only then**:
  - overlap knobs, bucket tuning, multi-instance distributed optimizer

### Verification checklist for nano-train implementation

- Single-rank smoke remains numerically identical to non-ZeRO.
- Multi-rank ZeRO-1 and ZeRO-2 produce finite loss and stable updates.
- Resume from sharded optimizer checkpoint is correct.
- Expert and dense params reduce on their intended domains.

## Primary Source References

1. Megatron distributed optimizer guide:
   - https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/dist_optimizer.md
2. Megatron parallelism guide:
   - https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/parallelism-guide.md
3. Megatron DDP implementation:
   - https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/distributed_data_parallel.py
4. Megatron distributed optimizer implementation:
   - https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/distrib_optimizer.py
5. Megatron process-group infrastructure:
   - https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
   - https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/process_groups_config.py
6. DeepSpeed contrast references:
   - https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/zero/config.py
   - https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/zero/stage_1_and_2.py
   - https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/engine.py
