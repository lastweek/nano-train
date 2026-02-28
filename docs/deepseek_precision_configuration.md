# DeepSeek Precision Configuration

**Purpose**: Explain how to configure precision for `DeepSeekModel` using the current
config-first interface.

**Audience**: Contributors and users working with `examples/train_4d.py` or building
custom DeepSeek-based runtime entrypoints.

**Prerequisites**:
- Familiarity with `examples/train_4d.py`
- Basic understanding of mixed precision and low-bit training

**Related Docs**:
- [Project README](../README.md)
- [Runtime Core Design](runtime_core_design.md)
- [ZeRO-1/2 Quickstart](zero1_zero2_quickstart.md)
- [TP + PP + EP + DP Communication Guide](pp_tp_ep_dp_communication.md)

## 1) Mental Model

DeepSeek precision control in this repo has four distinct layers:

1. Parameter storage dtype:
- Controlled by `DeepSeekModelConfig.param_dtype`
- Typical values: `torch.float32`, `torch.bfloat16`, `torch.float16`

2. Generic runtime precision policy:
- Controlled by `PrecisionConfig`
- Includes run-level mode such as `fp32`, `bf16`, `fp16`, `fp8`, `fp4`
- Includes DeepSeek-V3 recipe defaults and runtime loss-scaling behavior

3. Low-bit compute assignment:
- Applies to low-bit-capable linear-family modules
- Controlled by runtime precision resolution and module assignment
- Backed by the emulated or Transformer Engine backend

4. Exact DeepSeek module compute override:
- Controlled by `DeepSeekModelConfig.module_compute_dtype_overrides`
- Uses exact module paths such as `blocks.0.attn.q_a_norm`
- Intended for model-local control of numerically sensitive modules

## 2) Canonical Control Surface

For `DeepSeekModel`, prefer:

```python
DeepSeekModelConfig.module_compute_dtype_overrides
```

This is the canonical DeepSeek-specific interface because:

- it is exact-path based rather than regex-based
- it lives with the model config, where DeepSeek architectural choices already live
- it is easier to review than generic CLI patterns
- it avoids making DeepSeek users infer module names from runtime policy flags

Generic CLI rule fallback still exists:

```bash
--module-compute-dtype-rule '<pattern>=<fp32|bf16|fp16>'
```

Use the CLI rule when:

- you are working on non-DeepSeek scripts
- you want a quick experiment without touching model config assembly
- you want pattern-based behavior across many modules

## 3) Configuration Precedence

Current behavior is:

1. Runtime resolves generic precision policy from CLI and recipe flags.
2. `DeepSeekModelConfig` is built with:
- `param_dtype`
- `param_device`
- `precision_resolver`
- optional `module_compute_dtype_overrides`
3. `DeepSeekModel` wraps the base resolver with an exact-path override layer when
   `module_compute_dtype_overrides` is non-empty.
4. Each parameterized module resolves constructor-time precision state using its explicit
   `module_path`.

Important implications:

1. DeepSeek config override is exact-path and model-specific.
2. CLI rule is generic and pattern-based.
3. DeepSeek exact-path override disables low-bit compute for the overridden module.
4. Low-bit compute and explicit dtype override must not target the same module at the same
   time; the current implementation fails fast.

## 4) Exact Module Paths

DeepSeek exact-path overrides use the same constructor `module_path` names that modules bind
at initialization time.

Common examples:

- `blocks.0.attn.q_a_proj`
- `blocks.0.attn.q_a_norm`
- `blocks.0.ffn_norm`
- `final_norm`
- `lm_head`

The exact path is stable only if the model structure is the same. For example:

- `blocks.0.attn.q_a_norm` means block 0, attention submodule, `q_a_norm`
- `blocks.3.ffn.gate_proj` means block 3, feed-forward submodule, gate projection

If you are unsure about a path, inspect module names using `named_modules()` on the built model.

## 5) Canonical Code Example

`examples/train_4d.py` already exposes the config-first hook through
`build_tiny_deepseek_config(...)`:

```python
model_config = build_tiny_deepseek_config(
    args,
    param_dtype=param_dtype,
    param_device=parallel.device,
    precision_resolver=build_module_precision_resolver(precision_config),
    module_compute_dtype_overrides={
        "blocks.0.attn.q_a_norm": "fp16",
    },
)
```

This means:

- low-bit-capable linears still follow the resolved FP8/FP4 policy
- `blocks.0.attn.q_a_norm` runs in FP16 instead of low-bit or default autocast dtype

## 6) Common Scenarios

### A) DeepSeek-V3 recipe + `q_a_norm` in FP16

Use the DeepSeek-V3 recipe in CLI:

```bash
python3 examples/train_4d.py \
  --precision-recipe deepseek_v3 \
  --fp8 \
  --fp8-backend emulated
```

Then set:

```python
module_compute_dtype_overrides={
    "blocks.0.attn.q_a_norm": "fp16",
}
```

Result:

- low-bit linears still use the FP8 recipe path
- `q_a_norm` stays in FP16

### B) Keep `final_norm` or `lm_head` higher precision

Example:

```python
module_compute_dtype_overrides={
    "final_norm": "fp32",
    "lm_head": "bf16",
}
```

Use this when you want an explicit model-local precision exception instead of relying only on
generic recipe defaults or runtime patterns.

### C) Generic CLI fallback

If you want a fast experiment without editing DeepSeek config assembly:

```bash
python3 examples/train_4d.py \
  --precision-recipe deepseek_v3 \
  --fp8 \
  --fp8-backend emulated \
  --module-compute-dtype-rule '.*q_a_norm=fp16'
```

This works, but for DeepSeek it is a fallback rather than the preferred interface.

## 7) Failure Modes

### Exact path matches zero modules

If `module_compute_dtype_overrides` contains a path that does not exist in the built model,
the resolver finalization path fails fast.

Practical causes:

- wrong block index
- wrong submodule name
- path copied from a different model shape or layer layout

### Low-bit and dtype override conflict

If the same module is selected for both:

- low-bit compute
- explicit compute dtype override

the runtime fails fast rather than guessing precedence.

For DeepSeek exact-path overrides, the intended usage is:

- use config override for the exact sensitive module
- let all other linears follow the normal low-bit policy

### Why CLI regex is less preferred for DeepSeek

Regex/glob CLI rules are useful, but they are weaker for DeepSeek-specific control because:

- they are less explicit than exact module paths
- they can accidentally match more modules than intended
- they are harder to review in experiment configs

## 8) Where To Inspect The Implementation

If you want to trace the behavior in code:

- `examples/train_4d.py`
- `src/models/deepseek.py`
- `src/runtime/mixed_precision.py`
- `src/layers.py`

Useful places:

- `build_tiny_deepseek_config(...)` in `examples/train_4d.py`
- `DeepSeekModelConfig` and `DeepSeekModel` in `src/models/deepseek.py`
- constructor-time resolver logic in `src/runtime/mixed_precision.py`
- low-bit-capable linear execution in `src/layers.py`
