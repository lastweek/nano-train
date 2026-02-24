# Parallelism & Memory Optimization Plan

## Overview

This plan outlines the implementation of distributed training strategies and memory optimization techniques for nano-train, organized by **implementation complexity** and **impact**.

---

## Part 1: Memory Optimization (Zero & Offloading)

### Phase 1: Gradients State Sharding (ZeRO-1)

**Goal**: Shard optimizer states across GPUs

**Memory Savings**: ~4x reduction in optimizer memory

**Implementation**:
- Shard optimizer states (momentum, variance) across data parallel ranks
- Each GPU stores only `(1/world_size)` of optimizer states
- All-reduce gradients before optimizer step

**Complexity**: Medium
**Impact**: High (train 4x larger models on same GPU)

**File Structure**:
```
src/
├── parallelism/
│   ├── __init__.py
│   ├── zero.py          # ZeRO-1: Optimizer state sharding
│   └── distributed.py   # Distributed initialization (torch.distributed)
```

**Key Changes**:
- Custom optimizer wrapper that shards states
- Gradient all-reduce before optimizer step
- Distributed training setup

---

### Phase 2: Gradient Sharding (ZeRO-2)

**Goal**: Shard gradients + optimizer states

**Memory Savings**: ~8x reduction total

**Implementation**:
- Shard gradients across GPUs
- Each GPU computes gradients for local parameters
- All-gather gradients only when needed for update
- Immediately discard after use

**Complexity**: Medium-High
**Impact**: Very High (train 8x larger models)

**Key Changes**:
- Gradient sharding in backward pass
- Selective all-gather for optimizer step
- Memory-efficient backward

---

### Phase 3: Parameter Sharding (ZeRO-3)

**Goal**: Shard parameters + gradients + optimizer states

**Memory Savings**: Up to `N_gpu` x reduction (theoretical)

**Implementation**:
- Each GPU stores only a subset of model parameters
- All-gather parameters during forward (just-in-time)
- All-gather gradients during backward
- Maximum memory efficiency but higher communication

**Complexity**: High
**Impact**: Maximum (train models larger than single GPU)

**Key Changes**:
- Parameter sharding system
- JIT all-gather in forward/backward
- Careful memory management

---

### Phase 4: CPU Offloading

**Goal**: Offload optimizer states/gradients to CPU

**Memory Savings**: Additional 2-4x reduction

**Implementation**:
- Keep optimizer states on CPU
- Copy to GPU only during optimizer step
- Asynchronous transfers to overlap compute

**Complexity**: Low-Medium
**Impact**: Medium (trades speed for memory)

**Key Changes**:
- CPU-pinned memory allocator
- Async transfer in optimizer step
- Prefetch next step's states

---

## Part 2: Parallelism Strategies

### Phase 5: Distributed Data Parallel (DDP)

**Goal**: Basic data parallelism

**Scaling**: Linear with GPU count (8 GPUs = 8x throughput)

**Implementation**:
- Each GPU has full model replica
- Split batch across GPUs
- Gradient synchronization via all-reduce
- Use PyTorch DDP or custom wrapper

**Complexity**: Low
**Impact**: High (foundation for all parallelism)

**File Structure**:
```
src/parallelism/
├── data_parallel.py  # DDP wrapper
└── communicators.py  # All-reduce, all-gather wrappers
```

**Key Changes**:
- Process group initialization
- Gradient synchronization hooks
- Batch sampler for distributed training

---

### Phase 6: Tensor Parallelism (TP)

**Goal**: Split individual tensors across GPUs

**Scaling**: Enables models larger than single GPU memory

**Best for**: Very large models (100B+ params)

**Implementation**:
- Split linear layer weights across GPUs
- All-reduce for attention output
- Column-row parallelism for MLP
- No communication in backward for some ops

**Complexity**: High
**Impact**: Critical for 100B+ models

**File Structure**:
```
src/parallelism/
├── tensor_parallel.py
├── layers/
│   ├── parallel_linear.py    # Split linear layers
│   └── parallel_attention.py # Parallel attention
```

**Key Changes**:
- ColumnParallelLinear (split output dim)
- RowParallelLinear (split input dim)
- Parallel attention (QKV split, output all-reduce)
- Parallel MLP (fc1 column, fc2 row)

---

### Phase 7: Pipeline Parallelism (PP)

**Goal**: Split layers across GPUs (stage-by-stage)

**Scaling**: Reduces per-GPU memory, increases model size

**Best for**: Deep models with limited GPU memory

**Implementation**:
- Each GPU stores a subset of layers
- Micro-batches for pipeline bubbles
- Gradient accumulation across micro-batches
- 1F1B schedule for efficiency

**Complexity**: Very High
**Impact**: High for very deep models

**File Structure**:
```
src/parallelism/
├── pipeline_parallel.py
├── schedule.py    # 1F1B schedule
└── buffers.py     # Pipeline buffers
```

**Key Changes**:
- Model partitioning
- Micro-batch scheduler
- Pipeline communication
- Checkpoint/rebalancing

---

## Part 3: Combined Strategies

### Phase 8: 3D Parallelism (DDP + TP + PP)

**Goal**: Combine all strategies

**Configuration**:
```
total_gpus = 64
- 8-way data parallel (8 replicas)
- 4-way tensor parallel (split layers)
- 2-way pipeline parallel (split depth)
```

**Complexity**: Very High
**Impact**: Maximum scale (1000+ GPUs)

---

### Phase 9: Mixed Precision & Communication

**Goal**: Optimize communication

**Implementation**:
- Gradient compression (FP16/BF16)
- FP8 all-reduce (if hardware supports)
- Communication-computation overlap
- Tensor cores for faster collectives

**Complexity**: Medium
**Impact**: Medium (20-30% speedup)

---

## Implementation Priority

### Quick Wins (Do First):
1. **Phase 1 (ZeRO-1)**: 4x memory savings, medium complexity
2. **Phase 5 (DDP)**: Basic parallelism, low complexity
3. **Phase 4 (CPU Offloading)**: Easy memory savings

### High Impact (Do Second):
4. **Phase 2 (ZeRO-2)**: 8x memory savings
5. **Phase 6 (TP)**: Enables very large models

### Advanced (Do Later):
6. **Phase 3 (ZeRO-3)**: Maximum memory efficiency
7. **Phase 7 (PP)**: For very deep models
8. **Phase 8 (3D)**: Production scaling

---

## Memory Savings Summary

| Technique | Memory Saved | Speed Impact | Complexity |
|-----------|--------------|--------------|------------|
| ZeRO-1 | 4x | Minimal | Medium |
| ZeRO-2 | 8x | Small | Medium-High |
| ZeRO-3 | N_gpu x | Moderate | High |
| CPU Offload | 2-4x | Significant | Low-Medium |
| Gradient Checkpointing | 50% activations | +30% compute | Low |

**Combined**: ZeRO-2 + Gradient Checkpointing = ~16x memory reduction

---

## Recommended First Steps

1. **Start with ZeRO-1** (optimizer state sharding)
   - Highest ROI: 4x memory savings for medium complexity
   - Foundation for ZeRO-2/3

2. **Add DDP** (basic data parallel)
   - Enables multi-GPU training
   - Relatively simple

3. **Add Gradient Checkpointing**
   - 50% activation memory savings
   - Low complexity

4. **Then move to ZeRO-2**
   - 8x total memory savings
   - Enables training much larger models

5. **Finally TP** (if needed for very large models)
   - Required for 100B+ parameter models
   - More complex architecture changes

---

## Reference Implementations

- **DeepSpeed**: ZeRO-1/2/3 reference
- **Megatron-LM**: Tensor parallelism reference
- **PyTorch FSDP**: Fully sharded data parallel
- **Alpa**: Combined 3D parallelism

---

## Testing Strategy

1. **Unit tests** for each parallelism type
2. **Gradient equivalence checks** (single vs multi-GPU)
3. **Memory profiling** (verify savings)
4. **Scaling tests** (2, 4, 8 GPUs)
5. **Convergence tests** (same loss curve)
