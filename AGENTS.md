# Nano-Train Code Style Guidelines

This document defines the coding standards for the nano-train project. All code should follow these guidelines to ensure consistency, readability, and maintainability.

## Core Principles

1. **Readability First** - Code should be easy to understand and explain
2. **Simplicity Over Cleverness** - Avoid over-engineering and premature optimization
3. **Explicit Over Implicit** - Make logic clear and visible
4. **Test-Driven** - Use tests to validate implementations
5. **Documentation** - Document non-obvious code with clear comments
6. **Upgrade Existing Modules First** - Prefer improving reusable core modules
   over example-only retrofits so the main codebase evolves incrementally.

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

## AI Infra Python Coding Standards

This section applies to AI infrastructure code paths (distributed training,
training loops, orchestration, and model-parallel integrations). It uses a
balanced profile: strong defaults with documented exceptions when needed.

### 1. Design and Readability (Balanced)
- Prefer incremental upgrades to existing modules over example-only forks.
- Keep functions focused; extract helper units for validation, topology,
  logging, and synchronization.
- For distributed setup/results, prefer named structures (for example,
  dataclass/context objects) over long positional tuples.
- Public classes/functions must include concise docstrings, including
  shape/parallel-domain notes for distributed code paths.

### 2. Typing Policy
- Public functions/classes must include type hints.
- New or modified infra code should avoid `Any` unless justified in a short
  code comment.
- `mypy` runs in CI with progressive strictness; avoid blanket strict mandates
  across the entire repo unless explicitly adopted.

### 3. Distributed Training Safety Rules (PyTorch)
- Use `torchrun` and explicit launcher patterns; avoid ad-hoc rank/env
  assumptions.
- Ensure process-group lifecycle correctness (init once, destroy once,
  rank-safe behavior).
- Keep collective ordering identical across ranks in gradient synchronization.
- Log TP/EP/DP rank topology once from rank 0 for traceability.
- Seed strategy must be explicit and documented for parallel contexts.

### 3.1 Megatron Naming Contract
- Use Megatron-style names in code, logs, docs, and CLI:
  - `tensor_model_parallel_*`
  - `pipeline_model_parallel_*`
  - `expert_model_parallel_*`
  - `data_parallel_*`
  - `expert_data_parallel_*`
  - `context_parallel_*`
- Do not overload EP to mean batch-splitting or attention-axis switching.
- Prefer explicit domains over ad-hoc toggles; removed interfaces must not be
  reintroduced under new aliases.
- Backward-compat aliases are allowed for one release only and must emit
  deprecation warnings.

### 4. Reproducibility and Checkpointing
- Record run config and seed in logs for every training entrypoint.
- Training scripts that touch infra loops must include resumable checkpoint
  design.
- State determinism limits explicitly when full determinism is not guaranteed.

### 5. Observability and Failures
- Use structured, rank-aware logging in training loops.
- Raise explicit `ValueError`/`RuntimeError` with actionable context for
  invalid parallel configs.
- Do not swallow distributed exceptions; fail fast with clear diagnostics.

### 6. Testing Expectations
- Every new or changed distributed tutorial path must include:
  - single-rank smoke coverage
  - the smallest relevant multi-rank smoke coverage
- Model refactors in `src/models` must add at least one focused unit test to
  lock wiring/behavior.
- Keep tests minimal and targeted; avoid over-broad suites for small edits.

### Enforcement (CI + pre-commit)
- For changed AI infra code, required gates are:
  - `ruff check`
  - `ruff format --check`
  - `mypy` on affected modules (or configured target)
  - `pytest` for impacted tests/smokes
- Local pre-commit hooks are required for the same checks where applicable.
- If formatter tooling is migrated, use the equivalent configured formatter
  check command and keep CI/pre-commit parity.

### Source References
- PEP 8: https://peps.python.org/pep-0008/
- PEP 257: https://peps.python.org/pep-0257/
- PEP 484: https://peps.python.org/pep-0484/
- PyTorch DDP API:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
- PyTorch DDP design note: https://docs.pytorch.org/docs/2.9/notes/ddp.html
- `torchrun` / Elastic launcher:
  https://docs.pytorch.org/docs/2.9/elastic/run.html
- PyTorch multiprocessing best practices:
  https://docs.pytorch.org/docs/2.8/notes/multiprocessing.html
- PyTorch reproducibility:
  https://docs.pytorch.org/docs/stable/notes/randomness.html
- PyTorch AMP examples:
  https://docs.pytorch.org/docs/stable/notes/amp_examples.html
- PyTorch save/load tutorial:
  https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
- PyTorch activation checkpointing:
  https://docs.pytorch.org/docs/stable/checkpoint
- pytest good practices:
  https://docs.pytest.org/en/stable/explanation/goodpractices.html
- Ruff docs: https://docs.astral.sh/ruff/
- Ruff formatter: https://docs.astral.sh/ruff/formatter/
- mypy gradual typing guidance:
  https://mypy.readthedocs.io/en/stable/existing_code.html
- pre-commit: https://pre-commit.com/
- Hydra structured config intro:
  https://hydra.cc/docs/1.1/tutorials/structured_config/intro/
- Hydra ConfigStore API:
  https://hydra.cc/docs/1.2/tutorials/structured_config/config_store/
- Pydantic settings:
  https://docs.pydantic.dev/usage/settings/
- OpenTelemetry Python instrumentation:
  https://opentelemetry.io/docs/languages/python/instrumentation/

## Linting

This project uses CI-aligned linting and validation gates. Run before
committing:

```bash
# AI infra gate checks (required for infra-related changes)
ruff check .
ruff format --check .

# Type checking
mypy src/  # or affected modules

# Run impacted tests/smokes
pytest tests/  # scope to impacted tests when appropriate
```

If formatter/import tooling is migrated, use the equivalent configured check
command in both CI and pre-commit.

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

*Keep it simple, keep it clean.*
