# Nano-Train Code Style Guidelines

This document defines the coding standards for the nano-train project. All code should follow these guidelines to ensure consistency, readability, and maintainability.

## Core Principles

1. **Readability First** - Code should be easy to understand and explain
2. **Simplicity Over Cleverness** - Avoid over-engineering and premature optimization
3. **Explicit Over Implicit** - Make logic clear and visible
4. **Test-Driven** - Use tests to validate implementations
5. **Documentation** - Document non-obvious code with clear comments

## Python Code Style

### Formatting
- Use **4 spaces** for indentation (not tabs)
- Maximum **line length** of **100 characters**
- One blank line between functions
- Two blank lines between top-level classes

### Naming Conventions
- **Modules**: `lowercase_with_underscores` (e.g., `attention.py`, `tensor_parallel.py`)
- **Classes**: `PascalCase` (e.g., `TransformerModel`, `Trainer`)
- **Functions**: `lowercase_with_underscores` (e.g., `compute_loss`, `create_dataloader`)
- **Constants**: `UPPERCASE_WITH_UNDERSCORES` (e.g., `HIDDEN_SIZE`, `LEARNING_RATE`)
- **Private**: Single leading underscore (e.g., `_forward`, `_cache`)

### Imports
- One import per line
- Group standard library imports first, then third-party imports
- Use absolute imports (not relative): `from nano_train.models import TransformerBlock`
- Avoid wildcard imports: `from utils import *`

### Documentation
- **Docstrings** required for all public functions and classes
- **Type hints** recommended for function signatures
- **Inline comments** to explain non-obvious logic
- Comment why, not what

## Documentation Writing Style (Reports, Docs, Design Notes)

When writing Markdown in this repo (including generated reports), use a top-systems-paper style:
clear narrative, explicit definitions, and auditable reasoning.

- Prefer coherent paragraphs over bullet checklists. Use bullets only for true enumerations.
- Use “We …” voice in reports to explain intent and reasoning (e.g., “We define …”, “We estimate …”).
- Start each major section with a short bridge paragraph that answers:
  what this section does, why it exists, where the evidence is (table/figure), and what it does not claim.
- Term hygiene:
  define all acronyms/symbols before first use; keep notation consistent; include units.
  If a term is used repeatedly, add a small glossary table near the top.
- Make the reasoning chain explicit and stable:
  `inputs/assumptions -> FLOPs/bytes -> AI -> time model -> plotted/summary metrics`.
  Clearly label *upper bounds* (roofline ceilings) vs *estimates* (time-model throughput).
- Tables/figures:
  introduce each table/figure with 2–4 sentences explaining what it shows and how to read it.
  Keep plots uncluttered; put detailed numbers in tables; include key config in captions/titles.
- Avoid over-claiming:
  static models are not measured performance; separate algorithmic FLOPs from realizable/achievable FLOPs.
- Editing discipline:
  trim repetition, keep definitions in one canonical place, and avoid mixing alternative notations.

### Code Organization

#### Module Structure
```python
# Single responsibility: one module, one purpose
nano_train/
├── core/              # Core infrastructure (config, distributed, logging)
├── models/             # Model components (attention, MLP, transformer, MoE)
├── parallelism/         # Parallelism strategies (TP, PP, DP, SP, EP)
├── memory/             # Memory optimization (checkpointing, offloading)
├── training/            # Training loop (optimizer, scheduler, trainer)
├── data/               # Data loading and preprocessing
├── kernels/            # Custom CUDA kernels
└── utils/              # Helper utilities
```

#### File and Class Naming
- Files: `lowercase_with_underscores.py`
- Classes: `PascalCase`
- Test files: `test_*.py`

### Implementation Guidelines

#### 1. Function Design
- Keep functions **focused** - one clear purpose
- **Pure functions** - no side effects, easy to test
- **Length**: Functions should be reasonably short (< 50 lines typically)
- Return early for error conditions

#### 2. Error Handling
- Use Python exceptions, not error codes
- Log errors with context
- Provide helpful error messages

#### 3. Performance
- **Avoid premature optimization** - make it work, then make it fast
- **Profile before optimizing** - measure first
- Consider readability over micro-optimizations

#### 4. Type Safety
- Use type hints for function signatures
- Validate inputs (shapes, dtypes)
- Handle edge cases (empty tensors, None values)

#### 5. Deep Learning Specific

##### Model Implementation
- **Configurable architectures** - use config objects, not hardcoded values
- **Initialize weights properly** - use GPT-2 style initialization
- **Gradient clipping** - always use, don't make optional

##### Training
- **Checkpoint frequently** - don't wait until end of training
- **Learning rate scheduling** - use warmup + decay
- **Mixed precision** - default to BF16, not FP32
- **Logging** - log loss, throughput, and learning rate every N steps

#### 6. Testing
- **Write tests** for all components
- **Test edge cases** (empty inputs, wrong shapes)
- **Integration tests** for training loop
- Aim for >80% code coverage

#### 7. Git Workflow
- **Atomic commits** - one logical change per commit
- **Clear commit messages** - describe what and why
- **Don't commit cache/files** - use .gitignore

## Linting

This project uses standard Python linting tools. Run before committing:

```bash
# Check code style
black --check .

# Check imports
isort --check-only .

# Type checking
mypy nano_train/

# Run all checks
pytest tests/
```

### Quick Reference

| Aspect | Guideline |
|---------|-----------|
| Indentation | 4 spaces |
| Line Length | ≤ 100 chars |
| Imports | Absolute, grouped (stdlib then 3rd party) |
| Docstrings | Required for public functions |
| Type Hints | Recommended for signatures |
| Comments | Explain "why", not "what" |

---

## Examples

### Good ✅
```python
def compute_loss(logits: labels):
    """
    Compute cross-entropy loss for language modeling.

    Args:
        logits: (batch, seq_len, vocab_size) - model predictions
        labels: (batch, seq_len) - target tokens

    Returns:
        Scalar loss value
    """
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    return loss
```

### Bad ❌
```python
def f():  # Bad: meaningless name
    return [x**2 for x in xs if x%2]

# What NOT to do
from utils import *  # Bad wildcard imports
```

---

## Tutorial/Educational Code Guidelines

When creating **tutorial code** for learning distributed training concepts, the goal is to create **easy-to-understand minimal code** that showcases how things work.

### Tutorial Code Principles

1. **Minimal Working Examples** - Start with the simplest code that demonstrates the concept
2. **No Abstractions Initially** - Show raw primitives first, then layer abstractions
3. **Progressive Complexity** - Build understanding step by step
4. **Explicit Communication** - Make all-to-all, reduce, broadcast calls visible
5. **Comment the "Why"** - Explain WHY we need each distributed operation

### Tutorial Code Structure

```python
# TUTORIAL: This file demonstrates CONCEPT_X from scratch.
# Goal: Show how CONCEPT_X works without relying on PyTorch's built-in X.
# Learning: By implementing X manually, you understand the communication patterns.

# Step 1: Show the problem (what we're solving)
# Step 2: Show the solution (minimal working code)
# Step 3: Test it (verify correctness)
```

### Example: DDP from Scratch

**DO:**
```python
# TUTORIAL: Data Parallel from Scratch
# This shows how gradients are synchronized across devices manually.
# PyTorch DDP does this automatically, but we implement it here to learn.

def all_reduce_gradients(model, rank, world_size):
    """
    Manually synchronize gradients across all processes.

    Key concept: Each rank computes gradients on its data shard.
    We need to AVERAGE gradients so all ranks have the same update.

    Communication: All-to-All reduce (sum + average)
    """
    for param in model.parameters():
        if param.grad is not None:
            # 1. Sum gradients from all ranks
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            # 2. Average by dividing by world_size
            param.grad.div_(world_size)
```

**DON'T (for tutorials):**
```python
# DON'T: Just wrap with DDP and call it a day
# This hides the learning!
model = DDP(model)  # Magic! No learning happens.
```

### Tutorial vs Production Code

| Aspect | Tutorial Code | Production Code |
| :--- | :--- | :--- |
| **Goal** | Teach concepts | Optimize performance |
| **Abstractions** | Minimize, show primitives | Use high-level APIs |
| **Comments** | Explain "why" heavily | Explain non-obvious logic |
| **Error Handling** | Minimal (keep it simple) | Comprehensive |
| **Features** | One concept per file | Full-featured |

### Tutorial File Convention

Tutorial/educational files should be named clearly:

- `train_ddp_from_scratch.py` - Manual DDP implementation
- `train_ddp_torch.py` - Using PyTorch DDP
- `examples/tutorial_*` - Educational examples

Production files use standard naming:

- `src/trainer.py` - Production trainer
- `src/layers.py` - Core layer implementations

---

## File Organization Guidelines

When deciding where to place code, follow these principles:

### 1. Core Layers → `src/layers.py`

Place fundamental layer implementations here:
- `Linear` - Basic linear transformation
- `LayerNorm` - Normalization
- `Embedding` - Token embeddings
- `Dropout` - Regularization
- `GELU` - Activation function
- `ColumnParallelLinear`, `RowParallelLinear` - Tensor parallelism variants

**Key**: These are BUILDING BLOCKS that can be composed into larger components.

### 2. Model Components → `src/models/`

Place model-specific compositions here:
- `attention.py` - Attention mechanisms
- `mlp.py` - Feed-forward networks
- `transformer.py` - Transformer blocks
- `deepseek.py` - Specific model architectures

**Key**: These are COMPOSITIONS that use layers from `src/layers.py`.

### 3. Distributed Utilities → `src/distributed/`

Place distributed training infrastructure here:
- `device.py` - Device abstraction
- `launch.py` - Process launchers

**Key**: These are INFRASTRUCTURE for distributed training, not model components.

### Examples

| Code | Location | Why |
|------|----------|-----|
| `ColumnParallelLinear` | `src/layers.py` | It's a distributed variant of `Linear` layer |
| `ParallelMLP` | `src/models/mlp.py` | It's a model composition using parallel layers |
| `get_device_info()` | `src/distributed/device.py` | It's infrastructure, not a model component |

---

*Keep it simple, keep it clean.*
