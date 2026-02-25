# Model Info Appendix

**Purpose**: Hold detailed formulas and the full debugging checklist referenced by
`dump_model_info()` reports.

**Audience**: Contributors validating FLOP/byte derivations and troubleshooting report math.

**Prerequisites**: Familiarity with model dimensions and arithmetic intensity concepts.

**Related Docs**:
- [Model Info Utility](model_info.md)

## Table of Contents

- [Appendix A: Detailed Derivations](#appendix-a-detailed-derivations)
- [Appendix B: Full Debugging Checklist](#appendix-b-full-debugging-checklist)

## Appendix A: Detailed Derivations

### Dense Linear Layer

- Input shape: `[B,S,In]`, weight shape: `[In,Out]`
- FLOPs:
  - `F_linear = 2 * B * S * In * Out`

### MLA Attention FLOPs (Prefill)

- Terms:
  - `F_Q = 2 * B * S * H * r_q`
  - `F_K = 2 * B * S * H * r_kv`
  - `F_attn_score = 2 * B * h * S^2 * d_eff`
  - `d_eff = d_nope + d_rope`
- For DeepSeek-style MLA, use model-config dims (`H`, `h`, `r_q`, `r_kv`, `d_nope`, `d_rope`, `d_v`).

### MoE FLOPs Per Token

- One expert MLP:
  - `F_expert = 6 * H * d_moe`
- Routed total:
  - `F_MoE = B * S * top_k * 6 * H * d_moe`

## Appendix B: Full Debugging Checklist

- Verify `T_comp` uses `F_realizable` as a peak-equivalent compute cost (i.e., `T_comp = F_realizable / P_peak`).
- Verify WRF is applied consistently in prefill and decode paths.
- Verify KV bytes are counted per layer and multiplied by layer count.
- Verify temporary buffer bytes are not double-counted as activations.
- Verify dtype byte assumptions are consistent across weights/acts/KV.
