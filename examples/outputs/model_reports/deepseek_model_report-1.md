# Model Report

Generated: 2026-02-18 10:57:17

## 1. How to Read This Report

### 1.1. Goal

We want a simple outcome: given a model and a chip target, predict what limits throughput or latency in training, prefill, and decode, and then identify which knobs actually move those limits. We proceed from architecture to math/bytes to roofline to regimes to sensitivity, so each later table has an explicit definition and a reason to exist.

### 1.2. Workflow (3 Steps)

First, we classify the limiting regime by comparing arithmetic intensity to the chip ridge (`OI_knee = P_peak / BW_hbm`) and by inspecting the modeled times `T_comp`, `T_hbm`, and `T_net`. Second, we locate the dominant byte term using the byte decomposition (weights, activations, KV, temporary). Third, we map the regime to an optimization family: compute-bound suggests utilization/fusion; HBM(weight)-bound suggests weight residency/compression; HBM(KV)-bound suggests KV format/dtype/layout; and network-bound suggests topology/compression/overlap.

### 1.3. Common Gotchas

`F_theory` is an algorithmic count; it is not a performance claim. We explicitly separate `F_theory` from `F_realizable` (a peak-equivalent compute cost under a utilization model) because tiny-batch decode and thin GEMMs can leave tensor cores under-saturated. Decode also has a distinct length variable (`L`, KV length), and long-context decode can trend toward KV-driven `~1/L` intensity decline. Finally, MoE dispatch becomes a first-order concern primarily when expert parallelism (`EP`) is greater than 1.

## 2. Key Terms and Units

We use a small chain of definitions throughout the report. We start from `F_theory` (symbolic FLOPs derived from operator shapes, e.g. GEMM `2*M*K*N`) and a byte decomposition (`bytes_weights`, `bytes_activations`, `bytes_kv`, `bytes_temporary`). We then form `AI_hbm = F_theory / bytes_hbm` and convert it into chip ceilings via roofline (`OI_knee = P_peak / BW_hbm`). Separately, we estimate time by combining a compute cost `F_realizable` (peak-equivalent compute cost under utilization assumptions) with HBM/network transfer times and taking `T_est = max(T_comp, T_hbm, T_net)`. Finally, we report estimated throughput as `TF_est = F_theory / T_est` and plot points at (`AI_hbm`, `TF_est`).

| Term | Definition | Units | Where Used |
|---|---|---|---|
| `F_theory` | Symbolic FLOPs from model equations before hardware mapping. | `FLOPs` | Math, KPIs |
| `F_tensorcore` | Tensor-core-eligible subset of FLOPs. | `FLOPs` | Math, KPIs |
| `F_realizable` | Peak-equivalent compute cost FLOPs after utilization model (`eta_tc`) and scalar fallback. | `FLOPs` | Math, KPIs, Roofline |
| `eta_tc(B)` | Tensor-core utilization factor as a function of effective GEMM M dimension (decode proxy: batch). | `ratio` | Math |
| `P_peak` | Chip peak compute throughput. | `TFLOPs` | Roofline |
| `P_effective` | Effective compute ceiling implied by the utilization model (`P_peak * F_theory / F_realizable`). | `TFLOPs` | KPIs |
| `WRF_attn/dense/moe` | Weight Residency Factor(s) (WRF). We model effective streamed weight bytes as `W_eff = W / WRF`, with separate factors for attention/dense/MoE families. | `ratio` | Byte model, KPIs |
| `activation_fusion_factor` | Scales activation/intermediate bytes to represent fewer HBM trips under fused kernels. | `ratio` | Byte model |
| `elementwise_bytes_factor` | Scales elementwise-heavy terms (for example softmax/norm) to represent fusion and reduced temporaries. | `ratio` | Byte model |
| `bytes_weights` | Streamed weight bytes. | `bytes` | Byte model, KPIs |
| `bytes_activations` | Activation input/output bytes. | `bytes` | Byte model, KPIs |
| `bytes_kv` | KV-cache read/write bytes. | `bytes` | Byte model, KPIs |
| `bytes_temporary` | Temporary/intermediate buffer bytes. | `bytes` | Byte model, KPIs |
| `bytes_hbm` | Total HBM bytes = weights + activations + KV + temporary. | `bytes` | Byte model |
| `bytes_net` | Interconnect bytes (for example MoE dispatch). | `bytes` | Byte model, KPIs |
| `AI_hbm` | Arithmetic intensity using HBM bytes (`FLOPs / bytes_hbm`). | `FLOPs/byte` | Roofline, KPIs |
| `AI_total` | Arithmetic intensity using HBM+network bytes. | `FLOPs/byte` | KPIs |
| `OI_knee` | Roofline ridge point (`P_peak / BW_hbm`). | `FLOPs/byte` | Roofline |
| `T_comp` | Compute time estimate. | `seconds` | KPIs |
| `T_hbm` | HBM transfer time estimate. | `seconds` | KPIs |
| `T_net` | Network transfer time estimate. | `seconds` | KPIs |
| `T_est` | Estimated step time (`max(T_comp, T_hbm, T_net)`). | `seconds` | KPIs |
| `TF_est` | Estimated throughput from the time model (`F_theory / T_est`). | `TFLOPs` | Roofline plots, sweeps |

- Term chain: `F_theory -> F_tensorcore -> F_realizable -> AI -> roofline/time limits`.

## 3. Executive Summary

### 3.1. 3 Conclusions

1. In efficient mode, training is `compute-bound` and prefill is `compute-bound` (`AI_hbm`: 8171.58, 5447.72).
2. Decode in efficient mode is `hbm-bound` at base config, with `B_crit≈61.9` to approach the HBM ridge (`OI_knee=412.3`).
3. Experts hold ~97.8% of parameters, but parameter share does not equal runtime cost share.

### 3.2. 3 Next Optimizations

1. Prioritize weight-stream reduction: raise effective residency (WRF), improve expert weight staging, and reduce streaming bytes.
2. Keep improving compute path: increase tensor-core utilization and fuse bandwidth-heavy elementwise steps.
3. For serving, optimize batching policy around `B_crit` and latency constraints instead of targeting peak TFLOPs alone.

## 4. Architecture Overview

We begin with architecture because it determines where FLOPs and bytes come from before any kernel or system tuning. We focus on MLA and MoE because they reshape attention KV traffic, parameter concentration, and (optionally) dispatch behavior. These choices largely determine whether training, prefill, and decode are compute-, HBM-, or network-limited on a given chip. In the evidence below, we focus on how parameters concentrate in experts (drives weight-residency priorities), how MLA sets KV elements per token (drives decode KV bandwidth), and how the dense-vs-MoE layer mix shifts where bytes and FLOPs concentrate.

> **Callout:** Parameter distribution != runtime cost distribution.

### 4.1. Model Fingerprint

| Property | Value |
|---|---|
| Family | `Transformer` |
| Attention | `MLA` |
| Position Encoding | `RoPE (scaled)` |
| Normalization | `RMSNorm` |
| Activation | `Unknown` |
| MoE | `Detected` |
| Weight Tying | `No` |

**Config**

| Property | Value |
|---|---|
| `hidden_size` | `7168` |
| `num_hidden_layers` | `61` |
| `num_attention_heads` | `128` |
| `num_key_value_heads` | `128` |
| `intermediate_size` | `18432` |
| `moe_intermediate_size` | `2048` |
| `n_routed_experts` | `256` |
| `n_shared_experts` | `1` |
| `num_experts_per_tok` | `8` |
| `first_k_dense_replace` | `3` |
| `moe_layer_freq` | `1` |
| `q_lora_rank` | `1536` |
| `kv_lora_rank` | `512` |
| `qk_nope_head_dim` | `128` |
| `qk_rope_head_dim` | `64` |
| `v_head_dim` | `128` |
| `scoring_func` | `sigmoid` |
| `n_group` | `8` |
| `topk_group` | `4` |
| `vocab_size` | `129280` |
| `max_position_embeddings` | `163840` |
| `dropout` | `0.0` |

### 4.2. Architecture Diagrams

```mermaid
flowchart TB
    ids["Input token ids<br/>[B,S]"] --> emb["Token embedding<br/>V=129280, H=7168"]
    emb --> dense["Dense decoder blocks x 3<br/>Block = RMSNorm -> MLA -> + -> RMSNorm -> Dense FFN -> +"]
    dense --> moe["MoE decoder blocks x 58<br/>Block = RMSNorm -> MLA -> + -> RMSNorm -> Routed MoE -> +"]
    moe --> norm["Final RMSNorm"]
    norm --> head["LM head / logits<br/>[B,S,V]"]
```

```mermaid
flowchart TB
    x0["x_l<br/>[B,S,H]"] --> n1["RMSNorm"]
    n1 --> attn["MLA attention (per-layer)<br/>Q: H->1536->h*(128+64)<br/>KV cache: [B,L,(512+64)]<br/>Out: h*128->H"]
    x0 --> add1["Residual add"]
    attn --> add1
    add1 --> n2["RMSNorm"]
    n2 --> ffn["FFN (per-layer)<br/>Dense: H->18432->H<br/>MoE: E=256, top-k=8, shared=1, d_moe=2048"]
    add1 --> add2["Residual add"]
    ffn --> add2
    add2 --> x1["x_{l+1}<br/>[B,S,H]"]
    kv["KV cache (decode)<br/>[B,L,(512+64)]"] --> attn
    attn --> kvw["Append KV_t"]
```

```mermaid
flowchart TB
    h["Token hidden<br/>[B*S,H]"] --> r["Router logits<br/>[B*S,E]"]
    r --> tk["Top-k select (k=8)"]
    tk --> ex["Selected experts (k MLPs)<br/>H->d_moe->H"]
    ex --> mix["Weighted combine"]
    mix --> out["FFN out<br/>[B*S,H]"]
    h --> sh["Shared expert(s) (n=1)"]
    sh --> out
```

We keep these diagrams schematic: an end-to-end decoder stack, a representative decoder block (residual + MLA + FFN), and the routed MoE dataflow. Dense vs MoE FFN placement is controlled by `first_k_dense_replace` and `moe_layer_freq`.

### 4.3. Parameter & Memory Summary

We summarize parameter counts and memory in two complementary views: the instantiated tensor-dtype footprint (from parameter tensors) and the assumed parameter-byte setting used consistently by the analytic byte model.

| Metric | Value |
|---|---:|
| Total parameters | `671,026,404,352` |
| Trainable parameters | `671,026,404,352` |
| Non-trainable parameters | `0` |
| Parameter memory (tensor dtypes) | `2559762.59 MB` |
| Parameter memory (assumed, 1-byte params) | `639940.65 MB` |

Note: parameters are meta-initialized, so memory is estimated from tensor shapes and dtypes.

**Parameter Category Breakdown**

![](deepseek_model_report-1_module_pie.png)

### 4.4. Module Size Breakdown

Pattern-based view groups repeated layer indices (for example, `blocks.*.attn.out_proj`) to avoid repetitive rows.

| Module Pattern | Instances | Params / Instance | Total Params | % Total | Total Memory (MB) | Example Module |
|---|---:|---:|---:|---:|---:|---|
| `blocks.*.ffn.experts.*.gate_proj` | 14848 | 14,680,064 | 217,969,590,272 | 32.48% | 831488.00 | `blocks.3.ffn.experts.0.gate_proj` |
| `blocks.*.ffn.experts.*.up_proj` | 14848 | 14,680,064 | 217,969,590,272 | 32.48% | 831488.00 | `blocks.3.ffn.experts.0.up_proj` |
| `blocks.*.ffn.experts.*.down_proj` | 14848 | 14,680,064 | 217,969,590,272 | 32.48% | 831488.00 | `blocks.3.ffn.experts.0.down_proj` |
| `blocks.*.attn.out_proj` | 61 | 117,440,512 | 7,163,871,232 | 1.07% | 27328.00 | `blocks.0.attn.out_proj` |
| `blocks.*.attn.q_b_proj` | 61 | 37,748,736 | 2,302,672,896 | 0.34% | 8784.00 | `blocks.0.attn.q_b_proj` |
| `blocks.*.attn.kv_b_proj` | 61 | 16,777,216 | 1,023,410,176 | 0.15% | 3904.00 | `blocks.0.attn.kv_b_proj` |
| `token_embeddings` | 1 | 926,679,040 | 926,679,040 | 0.14% | 3535.00 | `token_embeddings` |
| `lm_head` | 1 | 926,679,040 | 926,679,040 | 0.14% | 3535.00 | `lm_head` |
| `blocks.*.ffn.shared_experts.*.gate_proj` | 58 | 14,680,064 | 851,443,712 | 0.13% | 3248.00 | `blocks.3.ffn.shared_experts.0.gate_proj` |
| `blocks.*.ffn.shared_experts.*.up_proj` | 58 | 14,680,064 | 851,443,712 | 0.13% | 3248.00 | `blocks.3.ffn.shared_experts.0.up_proj` |
| `blocks.*.ffn.shared_experts.*.down_proj` | 58 | 14,680,064 | 851,443,712 | 0.13% | 3248.00 | `blocks.3.ffn.shared_experts.0.down_proj` |
| `blocks.*.attn.q_a_proj` | 61 | 11,010,048 | 671,612,928 | 0.10% | 2562.00 | `blocks.0.attn.q_a_proj` |
| `blocks.*.ffn.gate_proj` | 3 | 132,120,576 | 396,361,728 | 0.06% | 1512.00 | `blocks.0.ffn.gate_proj` |
| `blocks.*.ffn.up_proj` | 3 | 132,120,576 | 396,361,728 | 0.06% | 1512.00 | `blocks.0.ffn.up_proj` |
| `blocks.*.ffn.down_proj` | 3 | 132,120,576 | 396,361,728 | 0.06% | 1512.00 | `blocks.0.ffn.down_proj` |
| `blocks.*.attn.kv_a_proj` | 61 | 4,128,768 | 251,854,848 | 0.04% | 960.75 | `blocks.0.attn.kv_a_proj` |
| `blocks.*.ffn.router` | 58 | 1,835,008 | 106,430,464 | 0.02% | 406.00 | `blocks.3.ffn.router` |
| `blocks.*.attn_norm` | 61 | 7,168 | 437,248 | 0.00% | 1.67 | `blocks.0.attn_norm` |
| `blocks.*.ffn_norm` | 61 | 7,168 | 437,248 | 0.00% | 1.67 | `blocks.0.ffn_norm` |
| `blocks.*.attn.q_a_norm` | 61 | 1,536 | 93,696 | 0.00% | 0.36 | `blocks.0.attn.q_a_norm` |
... 2 more patterns

### 4.5. Weight Statistics (Top 20 Layers)

- Weight value statistics are unavailable for meta-initialized parameters (no backing storage).

## 5. Analytical Model (FLOPs, Bytes, Time)

We define an analytical model that turns architecture into symbolic FLOPs, explicit bytes, and a static time estimate. We separate algorithmic work (`F_theory`) from a utilization-aware compute cost (`F_realizable`) and combine compute, HBM, and network times into `T_est` so later TFLOPs are not misread as measured performance. All roofline points and regime tables in the report are derived from this chain, so this section is the contract and audit trail for the numbers that follow. In the evidence below, we focus on the full definition chain `F_theory -> F_tensorcore -> F_realizable -> (AI_hbm, T_est) -> TF_est`, execution assumptions (WRF, fusion, flash attention) that change bytes without changing algorithms, and byte accounting and dominance tests that drive optimization conclusions.

### 5.1. Modeling intent and scope

The model is static: it does not ingest runtime profiler timelines, scheduler queues, or overlap traces. Its role is to rank bottlenecks and optimization priorities, not to claim measured peak attainment. The core mapping is `F_theory` (symbolic FLOPs from formulas) -> `F_tensorcore` (tensor-core-eligible FLOPs) -> `F_realizable` (peak-equivalent compute cost after utilization model), combined with `bytes_hbm` (HBM byte total) decomposition and `AI_hbm` (FLOPs divided by HBM bytes) / `AI_total` (FLOPs divided by HBM+network bytes) to determine roofline and timing outcomes. We use `AI_hbm` relative to `OI_knee` (ridge intensity `P_peak / BW_hbm`) to describe which side of the roofline ridge a point lies on, but the `Regime` labels in KPI tables come from the time model (`T_est` (estimated step time from compute/memory/network times) as `max(T_comp, T_hbm, T_net)`).

**Sanity checks.** As batch `B` grows, GEMM FLOPs scale with `B` while some weight bytes are amortized, so arithmetic intensity typically rises. In decode, KV read bytes scale with `L`; once KV dominates, intensity tends toward a `~1/L` decline.

### 5.2. Notation

| Symbol | Meaning |
|---|---|
| `B` | microbatch size per GPU |
| `S` | prefill sequence length |
| `L` | decode KV-cache length |
| `H` | hidden size |
| `h` | attention head count |
| `In`, `Out` | linear input/output channel size |
| `D` | embedding width |
| `W` | raw module weight bytes |
| `W_eff = W / WRF` | effective streamed weight bytes |
| `A` | activation bytes/element |
| `A_kv` | KV-cache bytes/element |
| `C_kv` | KV-cache elements per token |
| `F_theory` | mathematical FLOPs from symbolic formulas |
| `F_tensorcore` | tensor-core-eligible FLOPs subset |
| `F_realizable` | peak-equivalent compute cost after utilization/scalar-efficiency model |
| `eta_tc(B)=min(1,M_eff/B_sat)` | tensor-core saturation factor (proxy: decode uses `M_eff≈B`; dense prefill/training uses `M_eff≈B*S`) |
| `P_effective = P_peak * F_theory / F_realizable` | effective compute ceiling implied by utilization model |
| `AI_hbm = FLOPs / bytes_hbm` | HBM arithmetic intensity |
| `AI_total = FLOPs / (bytes_hbm + bytes_net)` | end-to-end intensity |
| `T_comp = F_realizable / P_peak` | compute time estimate |
| `T_hbm = bytes_hbm / BW_hbm` | HBM time estimate |
| `T_net = bytes_net / BW_net` | network time estimate |
| `T_est = max(T_comp, T_hbm, T_net)` | step-time upper bound |

### 5.3. Execution / Kernel Assumptions

These are conservative knobs for sensitivity analysis, not claims about measured kernel reuse.

| exec_model | attention bytes | WRF attn/dense/moe | act fusion | elementwise |
|---|---|---|---:|---:|
| `naive` | `naive` | `1.00/1.00/1.00` | 1.00 | 1.00 |
| `efficient` | `flash` | `4.00/4.00/2.00` | 0.50 | 0.70 |

#### 5.3.1. Knob Semantics and Rationale

We introduce a small set of execution-mode knobs to express how kernel families change HBM traffic without changing the underlying algorithm. `WRF` (Weight Residency Factor) models effective streamed weights as `W_eff = W / WRF`, and we use separate factors for attention/dense/MoE because reuse differs by module family. `act fusion` scales activation/intermediate bytes to represent fewer HBM trips under fused kernels, while `elementwise` scales elementwise-heavy terms (softmax/norm/masking temporaries). Finally, `attention bytes` selects the attention byte path: `naive` materializes score/prob matrices in HBM, while `flash` removes most `SxS` temporary traffic while still counting explicit KV reads and writes.

These are conservative knobs for sensitivity analysis, not claims about measured kernel reuse. We treat `naive` as a pessimistic baseline (`WRF=1`, no fusion) and `efficient` as a conservative approximation of common optimizations (partial residency + fusion). If you have measured counters in your serving/training stack, tune these factors to match observed bytes and regenerate the report.

### 5.4. Per-Module Formulas

We use GEMM as the primitive building block: multiplying `[M,K] @ [K,N]` costs `2*M*K*N` FLOPs. For any module, we combine its FLOP model with its byte model to form `AI_hbm`, and we call a point compute-favorable when `AI_hbm >= OI_knee` (with `OI_knee = P_peak / BW_hbm`).

| Module | Shape Explanation | Sample Torch | FLOPs | Bytes (HBM, naive) | Native AI | Efficient AI | Note |
|---|---|---|---|---|---|---|---|
| Linear prefill | `X:[B,S,In] @ W:[In,Out] -> Y:[B,S,Out]` | `y = x @ w` | `2*B*S*In*Out` | `W + A*(B*S*In + B*S*Out)` | `F / Bytes_naive` | `F / (W/WRF_dense + A*act_fusion*(B*S*In + B*S*Out))` | `Usually modest vs native unless WRF or fusion is high` |
| Linear decode | `X:[B,In] @ W:[In,Out] -> Y:[B,Out]` | `y = x @ w` | `2*B*In*Out` | `W + A*(B*In + B*Out)` | `F / Bytes_naive` | `F / (W/WRF_dense + A*act_fusion*(B*In + B*Out))` | `Typically similar; weight reuse is limited in decode` |
| Embedding prefill | `ids:[B,S] -> out:[B,S,D]` from `table:[V,D]` | `out = table[ids]` | `B*S*D` | `W + A*(B*S*D)` | `F / Bytes_naive` | `F / (W/WRF_dense + A*act_fusion*(B*S*D))` | `Usually close; dominated by table read` |
| Embedding decode | `ids:[B] -> out:[B,D]` from `table:[V,D]` | `out = table[ids]` | `B*D` | `W + A*(B*D)` | `F / Bytes_naive` | `F / (W/WRF_dense + A*act_fusion*(B*D))` | `Close to native; little activation reuse` |
| Attention prefill | `Q,K,V:[B,h,S,d]`, score `[B,h,S,S]`, context `[B,h,S,d]` | `p=softmax(q@kT); y=p@v` | `8*B*S*H^2 + 4*B*S^2*H + B*h*S^2` | `W + A*(B*S*H + 3*B*S*H + 2*B*h*S^2 + B*S*H) + A_kv*(B*S*C_kv)` | `F / Bytes_naive` | `F / (W/WRF_attn + A*(core_terms) + A_kv*(B*S*C_kv))` (flash removes SxS HBM materialization) | `Often much higher vs native when S is large` |
| Attention decode | `Q:[B,h,1,d]`, cache `K,V:[B,h,L,d]`, score `[B,h,1,L]` | `p=softmax(q@kT); y=p@v` | `8*B*H^2 + 4*B*L*H + B*h*L` | `W + A*(B*H + 3*B*H + 2*B*h*L + B*H + 2*B*H) + A_kv*(B*L*C_kv + B*C_kv)` | `F / Bytes_naive` | `F / (W/WRF_attn + A*(core_terms) + A_kv*(B*L*C_kv + B*C_kv))` | `Usually limited by KV reads; efficient ~= native` |

Footnotes:
- `C_kv` mapping: standard MHA/GQA uses K/V cache in head space; DeepSeek-MLA uses `kv_lora_rank + qk_rope_head_dim`.
- Attention formulas above are generic dense attention references; module-level MLA estimates use detected dims (`r_q`, `r_kv`, `d_nope`, `d_rope`, `d_v`) in `_compute_attention_flops_*_mla`.
- Training maps from prefill: `F_train = F_prefill * training_flops_multiplier`, `bytes_train = bytes_prefill * training_bytes_multiplier`.

### 5.5. Tensor Core Mapping

Tensor cores deliver their advertised throughput only when GEMM shapes provide enough parallel work and pack well into hardware tiles. In decode, GEMMs often have a small effective M dimension (roughly the microbatch `B`), while dense prefill/training GEMMs have `M≈B*S` and MoE expert GEMMs have `M≈(tokens per active expert)`. We model tensor-core saturation with `eta_tc(B)=min(1, M_eff/B_sat)` (the symbol uses `B` for historical decode intuition; the implementation uses `M_eff`). We then define a peak-equivalent compute cost as `F_realizable = F_tensorcore/eta_tc + (F_theory-F_tensorcore)/eta_scalar`, so that `T_comp = F_realizable / P_peak` is consistent without double-counting utilization.

### 5.6. Byte Accounting

`bytes_hbm = bytes_weights + bytes_activations + bytes_kv + bytes_temporary`
For routed MoE, `bytes_weights` is not scaled linearly by `top_k`: we approximate expert weight traffic by the expected number of *distinct* experts activated in the microbatch (uniform routing baseline), because weight reads occur per active expert.

| Byte Term | Meaning |
|---|---|
| `bytes_weights` | Streamed weight traffic after residency factor (WRF) |
| `bytes_activations` | Input/output activation movement |
| `bytes_kv` | KV-cache read/write traffic (`L` for decode, `S` write for prefill) |
| `bytes_temporary` | Score/prob/intermediate temporary buffers |

### 5.7. Byte Dominance Test

`share(x) = bytes_x / bytes_hbm`

Decision rules: if `share(weights) > 70%`, we prioritize weight residency/compression/paging; if `share(kv) > 30%` and grows with `L`, we prioritize KV dtype/layout/cache; and if `share(temporary)` is large only in naive attention, we prioritize flash/fused attention.

| Mode | weights naive | kv naive | temp naive | weights eff | kv eff | temp eff |
|---|---:|---:|---:|---:|---:|---:|
| `training` | 38.5% | 0.1% | 11.4% | 36.0% | 0.3% | 0.0% |
| `prefill` | 38.5% | 0.1% | 11.4% | 36.0% | 0.3% | 0.0% |
| `decode` | 99.1% | 0.3% | 0.1% | 98.6% | 0.7% | 0.0% |

In this run, prefill temporary-byte share drops from `11.4%` (naive) to `0.0%` (efficient), which is the intended effect of switching from score/prob materialization to flash-style attention. Decode in efficient mode has a byte mix of weights `98.6%`, KV `0.7%`, and temporary `0.0%`. If decode appears weight-dominant here, that reflects residency assumptions (WRF/paging strategy) and can shift under continuous batching; decode is often KV-bound in literature, and this static model can show either KV or weight-stream dominance depending on those assumptions.

### 5.8. Mode Byte Decomposition (naive vs efficient)

| Mode | bytes_weights naive | bytes_activations naive | bytes_kv naive | bytes_temporary naive | bytes_hbm naive | bytes_weights eff | bytes_activations eff | bytes_kv eff | bytes_temporary eff | bytes_hbm eff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `training` | 50.91 GB | 66.20 GB | 136.71 MB | 15.13 GB | 132.38 GB | 18.73 GB | 33.10 GB | 136.71 MB | 21.27 MB | 51.98 GB |
| `prefill` | 25.46 GB | 33.10 GB | 68.36 MB | 7.57 GB | 66.19 GB | 9.36 GB | 16.55 GB | 68.36 MB | 10.63 MB | 25.99 GB |
| `decode` | 25.46 GB | 139.60 MB | 68.62 MB | 30.38 MB | 25.69 GB | 9.36 GB | 69.80 MB | 68.62 MB | 42.70 KB | 9.50 GB |

## 6. Roofline Analysis

We combine roofline ceilings with the static time model to explain regimes across training, prefill, and decode. We plot points at (`AI_hbm`, `TF_est`) and use `T_est=max(T_comp, T_hbm, T_net)` to identify the limiting resource without claiming measured performance. Sweeps over batch and prompt/KV length then show which knobs can move decode toward the ridge and which cannot under the model assumptions. In the evidence below, we focus on how the anchor workload (`B`, `S`, `L`) sets the base point for reported KPIs, whether phases are left/right of ridge and which time term dominates, and how batch and sequence/KV-length sweeps shift `AI_hbm` and `TF_est`.

### 6.1. Reading a Roofline Point

A roofline point is positioned by its arithmetic intensity and its estimated throughput. The roofline bound is the minimum of the compute ceiling (`P_peak`) and the memory ceiling (`BW_hbm * AI_hbm`). Points left of the ridge (`AI_hbm < OI_knee`) are bounded by HBM, while points right of the ridge are bounded by compute.
This ridge-side classification is a roofline diagnostic; the `Regime` labels in KPI tables come from the time model (`T_est=max(T_comp,T_hbm,T_net)`).

For this run (efficient mode, primary target `H200_SXM_FP8`), decode has `AI_hbm=58.46` compared to `OI_knee=412.29`, so it is `memory-bound` under the HBM roofline model.
Prefill has `AI_hbm=5447.72` compared to `OI_knee=412.29`, so it is `compute-bound` under the same assumptions.

### 6.2. Configuration & Assumptions

We evaluate roofline points at an anchor workload defined by the caller-provided `batch_size` and `seq_len` arguments to `dump_model_info(...)`. In `examples/train_deepseek.py`, this corresponds to the training microbatch (`config.training.batch_size`) and the dataset/training context length (`config.data.max_seq_length - 1`). We set the base point with prefill length `S=seq_len` and decode KV length `L=seq_len`, and then run merged sweeps that vary decode (`B`, `L`, `EP`) and prefill (`B`, `S`, `EP`) around that anchor. To change the anchor, pass different `batch_size`/`seq_len` or override the sweep lists. For routed-MoE models where `ep_size` is not provided by the model config, we default to ~4 routed experts per GPU (`EP=ceil(E/4)`) to reflect common expert sharding.

| Property | Value |
|---|---|
| Batch size (`B`) | `8` |
| Prefill sequence length (`S`) | `255` |
| Decode KV length (`L`) | `255` |
| Assumed parameter bytes (`W` dtype) | `1` (FP8 default) |
| Activation dtype bytes (`A`) | `1` |
| KV-cache dtype bytes (`A_kv`) | `1` |
| Tensor parallel size (`TP`) | `1` |
| Expert parallel size (`EP`) | `64` |
| Routed experts per GPU (`E/EP`) | `4` |
| Training FLOPs multiplier | `3.0` |
| Training bytes multiplier | `2.0` |
| Primary roofline target | `H200_SXM_FP8` |
| Primary peak compute (`P_peak`) | `1979.0 TFLOPs` |
| HBM bandwidth (`BW_hbm`) | `4800 GB/s` |
| Interconnect bandwidth (`BW_net`) | `900 GB/s` |
| HBM ridge (`OI_knee`) | `412.292 FLOP/byte` |
| Network ridge (`OI_net`) | `2198.889 FLOP/byte` |
| Decode batch sweep (`B`) | `[1, 2, 4, 8, 16, 32, 64, 128]` |
| Decode KV-length sweep (`L`) | `[255, 2048, 4096, 8192, 16384]` |
| Decode EP sweep (`EP`) | `[32, 64, 128]` |
| Prefill sequence sweep (`S`) | `[64, 128, 255, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 163840]` |
| Prefill batch sweep (`B`) | `[1, 8, 32]` |
| Prefill EP sweep (`EP`) | `[32, 64, 128]` |
| Requested roofline x-limits | `None` |
| Requested roofline y-limits | `None` |
| Rendered roofline x-limits | `(0.01, 100000000)` |
| Rendered roofline y-limits | `(0.016, 5175.0)` |
| Roofline label mode | `minimal` |

**Roofline Targets (FP8 Dense)**

| Chip | Peak TFLOPs | HBM GB/s | OI_knee (FLOP/byte) |
|---|---:|---:|---:|
| `H200_SXM_FP8` | 1979.0 | 4800 | 412.292 |
| `B200_SXM_FP8` | 4500.0 | 8000 | 562.500 |

### 6.3. Roofline Overview

Model points use the primary target assumptions above and are plotted at (`AI_hbm`, `TF_est`), where `TF_est = F_theory / T_est`. Chip curves show multi-chip roofline upper bounds (memory slope + compute ceiling) for comparison.

![](deepseek_model_report-1_roofline.png)

**Interpretation**

Efficient training/prefill intensities (`8171.58`, `5447.72`) should be read against `OI_knee=412.29` to classify regime, while decode (`AI_hbm=58.46`) indicates whether the serving path is still memory-limited under base assumptions. This figure should not be interpreted as parameter share implying runtime share.

### 6.4. Derivation Notes

We include this short note to keep the length variables unambiguous: prefill is parameterized by prompt length `S`, while decode is parameterized by KV-cache length `L`. This matters because different terms dominate: naive prefill attention can incur `O(S^2)` score/prob traffic, while decode attention is driven by `O(L)` KV reads and a small per-step compute core. We therefore plot rooflines using `AI_hbm` (HBM-only) as the x-axis and treat network effects separately via `bytes_net` and `T_net` inside `T_est=max(T_comp,T_hbm,T_net)`. The sweeps below vary one axis at a time (`B` for decode, `S` for prefill) so changes in `AI_hbm` and `TF_est` can be interpreted causally under the stated assumptions.

### 6.5. Why regimes differ across training/prefill/decode

Training and prefill are dominated by larger GEMMs with higher reuse, so they more often approach compute-side behavior under efficient kernels. Decode operates with smaller-M matmuls and explicit KV reads that scale with context length `L`, which keeps decode sensitive to memory traffic even when batch increases. The practical implication is that prefill optimization responds strongly to fusion and tensor-core utilization, while decode optimization is usually governed by byte traffic management and batching policy.

### 6.6. Execution Models (Naive vs Efficient)

This section compares execution modes through one consolidated matrix. The main claim is that mode shifts are primarily byte-path shifts: `F_theory` is stable while `bytes_hbm`, `AI_hbm`, and `T_est` move materially. Evidence comes from model-level KPIs and category-level AI deltas; limitations remain static-analysis assumptions around residency and fusion.

#### 6.6.1. Regime KPI Matrix (naive vs efficient)

We summarize each phase and execution mode with one row of modeled work, bytes, and time. `F_realizable` is the peak-equivalent FLOP cost used for compute time (`T_comp = F_realizable / P_peak`), while `AI_hbm` is the roofline x-coordinate (`F_theory / bytes_hbm`) and `TF_est` is derived later as `F_theory / T_est`. `Regime` is the max-time limiter (`argmax(T_comp,T_hbm,T_net)`), so it can differ from the ridge-side diagnostic (`AI_hbm` vs `OI_knee`). Tokens/s is computed per GPU with `tokens_per_step=B*S` for training/prefill and `tokens_per_step=B` for decode.

| Mode | Exec | F_realizable | bytes_hbm | AI_hbm | MFU_est | T_est (ms) | Tokens/s | Regime |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `training` | `naive` | 481.25 TF | 132.38 GB | 3.209e+03 | 9.477e-01 | 243.181 | 8388.82 | `compute-bound` |
| `training` | `efficient` | 481.25 TF | 51.98 GB | 8.172e+03 | 9.477e-01 | 243.181 | 8388.82 | `compute-bound` |
| `prefill` | `naive` | 160.42 TF | 66.19 GB | 2.139e+03 | 9.477e-01 | 81.060 | 25166.45 | `compute-bound` |
| `prefill` | `efficient` | 160.42 TF | 25.99 GB | 5.448e+03 | 9.477e-01 | 81.060 | 25166.45 | `compute-bound` |
| `decode` | `naive` | 3.33 TF | 25.69 GB | 2.161e+01 | 5.242e-02 | 5.747 | 1392.10 | `hbm-bound` |
| `decode` | `efficient` | 3.33 TF | 9.50 GB | 5.846e+01 | 1.418e-01 | 2.125 | 3765.36 | `hbm-bound` |

#### 6.6.2. Category Delta Overview (naive vs efficient)

We group module entries into coarse categories (`attention`, `experts`, `ffn`, `embedding`, `other`) to explain which operator families drive the deltas between `naive` and `efficient`. For each mode and category we recompute `AI_hbm` from aggregated `F_theory` and `bytes_hbm`, and we report `bytes_total` as HBM+network attribution. With `EP=1`, network bytes are zero by construction and `bytes_total` is dominated by HBM terms; when `EP>1`, routed-MoE dispatch/collect can make the network term material.

| Mode | Category | AI_hbm naive | AI_hbm eff | AI delta % | bytes_total naive | bytes_total eff |
|---|---|---:|---:|---:|---:|---:|
| `training` | `attention` | 3.064e+03 | 1.428e+04 | 365.9% | 44.83 GB | 9.62 GB |
| `training` | `embedding` | 2.330e-02 | 9.178e-02 | 293.9% | 1.75 GB | 455.82 MB |
| `training` | `experts` | 3.272e+03 | 6.544e+03 | 100.0% | 80.46 GB | 40.23 GB |
| `training` | `ffn` | 4.386e+03 | 1.367e+04 | 211.7% | 3.09 GB | 1015.24 MB |
| `training` | `network` | 0.000e+00 | 0.000e+00 | 0.0% | 12.44 GB | 12.44 GB |
| `training` | `other` | 4.706e+03 | 1.529e+04 | 224.9% | 2.24 GB | 707.33 MB |
| `prefill` | `attention` | 2.043e+03 | 9.518e+03 | 365.9% | 22.42 GB | 4.81 GB |
| `prefill` | `embedding` | 1.553e-02 | 6.119e-02 | 293.9% | 897.70 MB | 227.91 MB |
| `prefill` | `experts` | 2.181e+03 | 4.363e+03 | 100.0% | 40.23 GB | 20.12 GB |
| `prefill` | `ffn` | 2.924e+03 | 9.115e+03 | 211.7% | 1.55 GB | 507.62 MB |
| `prefill` | `network` | 0.000e+00 | 0.000e+00 | 0.0% | 12.44 GB | 12.44 GB |
| `prefill` | `other` | 3.138e+03 | 1.020e+04 | 224.9% | 1.12 GB | 353.67 MB |
| `decode` | `attention` | 1.671e+01 | 6.564e+01 | 292.9% | 10.75 GB | 2.74 GB |
| `decode` | `embedding` | 6.188e-05 | 2.475e-04 | 300.0% | 883.80 MB | 220.96 MB |
| `decode` | `experts` | 2.843e+01 | 5.686e+01 | 100.0% | 12.10 GB | 6.05 GB |
| `decode` | `ffn` | 1.598e+01 | 6.380e+01 | 299.4% | 1.11 GB | 284.38 MB |
| `decode` | `network` | 0.000e+00 | 0.000e+00 | 0.0% | 49.96 MB | 49.96 MB |
| `decode` | `other` | 1.598e+01 | 6.385e+01 | 299.5% | 884.79 MB | 221.46 MB |

#### 6.6.3. Cross-Mode Summary (naive vs efficient)

We provide a compact scanline across phases (training vs prefill vs decode) to make cross-mode comparisons easy. This is the fastest way to see whether an execution mode mainly changes bytes (`AI_hbm`) or whether it materially changes modeled time (`T_est`) and throughput (Tokens/s). In practice, prefill often benefits more from byte reductions (especially attention temporaries) than decode, which can remain constrained by streaming terms and small effective batch.

| Mode | AI_hbm naive | AI_hbm eff | T_est naive (ms) | T_est eff (ms) | Tokens/s naive | Tokens/s eff |
|---|---:|---:|---:|---:|---:|---:|
| `training` | 3.209e+03 | 8.172e+03 | 243.181 | 243.181 | 8388.82 | 8388.82 |
| `prefill` | 2.139e+03 | 5.448e+03 | 81.060 | 81.060 | 25166.45 | 25166.45 |
| `decode` | 2.161e+01 | 5.846e+01 | 5.747 | 2.125 | 1392.10 | 3765.36 |

#### 6.6.4. Decode Sweep (vary B, L, EP)

We treat decode as a three-axis workload: microbatch `B`, KV-cache length `L`, and expert parallelism `EP` (which sets routed experts per GPU `E/EP`). This matters because the dominant decode bytes are typically streaming terms (weights and KV reads), so `AI_hbm` can shift substantially with batching and with long-context KV traffic. We keep `TP` fixed (`TP=1`) and evaluate all points under the same static time model `T_est=max(T_comp,T_hbm,T_net)` so `Regime` is auditable as the max-time limiter.

| EP | E/EP | AI_hbm naive | AI_hbm eff | T_est naive (ms) | T_est eff (ms) | TF_est naive | TF_est eff | Regime naive | Regime eff |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 32 | 8 | 1.577e+01 | 3.896e+01 | 7.874 | 3.188 | 75.72 | 187.00 | `hbm-bound` | `hbm-bound` |
| 64 | 4 | 2.161e+01 | 5.846e+01 | 5.747 | 2.125 | 103.75 | 280.62 | `hbm-bound` | `hbm-bound` |
| 128 | 2 | 2.653e+01 | 7.800e+01 | 4.682 | 1.592 | 127.33 | 374.39 | `hbm-bound` | `hbm-bound` |

At the anchor slice (`EP=64`, `L=255`), `B_crit` at the HBM ridge (`OI_knee=412.29`) is: naive=`None`, efficient=`61.92846204395823`.

![](deepseek_model_report-1_decode_bl_roofline.png)

We plot `B×L` points in roofline space at the anchor `EP` to show how longer KV contexts shift the same model leftward in `AI_hbm`, while increasing batch moves points rightward by amortizing streamed weights and fixed overheads. The key question is whether practical serving batches can move decode near the ridge for the target `L`, or whether KV traffic leaves decode firmly HBM-limited.

##### Compute-Feasible Frontier (min B_g over EP×L)

We complement the roofline-space curves with a compute-feasibility frontier: under the simplifying per-GPU view (`DP=EP`, `TP=1`, no PP/SP), we search for the smallest microbatch `B_g` such that compute time is within `alpha` of the slower of HBM and network (`T_comp >= alpha*max(T_hbm,T_net)`). Empty cells indicate that KV/network asymptotes remain left of ridge (or that we did not find a crossing within the search grid), so decode cannot become compute-favorable at that (`EP`,`L`) under the model.

![](deepseek_model_report-1_decode_ep_l_minbg.png)

| EP | E/EP | L=255 | L=2048 | L=4096 | L=8192 | L=16384 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 8 | 1024 | 2048 |  |  |  |
| 64 | 4 | 512 | 2048 |  |  |  |
| 128 | 2 | 512 | 2048 |  |  |  |

**How These Numbers Are Calculated**

We compute each sweep point by fixing (`B`, `L`, `EP`), aggregating model-level FLOPs and bytes under fixed parallelism, and then evaluating the static time model and derived roofline diagnostics. The calculation chain is:

```text
F_theory(B,L,EP,m) = sum_i F_i(B,L,EP,m)
bytes_hbm(B,L,EP,m) = sum_i (bytes_weights_i + bytes_activations_i + bytes_kv_i + bytes_temporary_i)
bytes_net(B,L,EP,m) = sum_i bytes_net_i
AI_hbm = F_theory / bytes_hbm
T_comp = F_realizable / (P_peak * 1e12)
T_hbm = bytes_hbm / (BW_hbm * 1e9)
T_net = bytes_net / (BW_net * 1e9)
T_est = max(T_comp, T_hbm, T_net)
regime = argmax{T_comp, T_hbm, T_net}
TF_est = F_theory / T_est / 1e12
TF_roofline_hbm = min(P_peak, BW_hbm * AI_hbm / 1e12)
```

- Worked example (efficient, `EP=64`, `B=128`, fixed `L=255`):
  `F_theory=9.539e+12`, `bytes_hbm=1.238e+10`, so `AI_hbm=7.708e+02`.
  Ridge check: `AI_hbm >= OI_knee` -> `7.708e+02 >= 4.123e+02`.
  Roofline upper bound: `BW_hbm * AI_hbm / 1e12 = 3699.74 TFLOPs`; `TF_roofline_hbm = min(P_peak, unclipped) = min(1979.00, 3699.74) = 1979.00 TFLOPs`.
  Time path: `T_comp=5.086 ms`, `T_hbm=2.578 ms`, `T_net=0.931 ms`, `T_est=5.086 ms` -> regime `compute-bound`.
  Estimated throughput: `TF_est = F_theory / T_est / 1e12 = 1875.54 TFLOPs`.

#### 6.6.5. Prefill Sweep (vary B, S, EP)

We treat prefill as a three-axis workload: microbatch `B`, prompt length `S`, and `EP`. Prefill differs from decode because attention has prompt-side reuse and (in naive mode) explicit `O(S^2)` temporary traffic, so increasing `S` can either amortize weight streaming or amplify temporary bytes depending on the attention bytes model. We keep `TP` fixed (`TP=1`) and reuse the same time model as decode so the curves are directly comparable under the report’s assumptions.

![](deepseek_model_report-1_prefill_bs_roofline.png)

We plot an `S` sweep at multiple `B` values (fixed `EP`) to show how longer prompts shift `AI_hbm` and whether efficient attention reduces temporary-byte pressure enough to keep prefill compute-favorable at practical batch sizes.

To make the `EP` effect auditable, we report a small slice at the anchor batch (`B=8`) and selected prompt lengths.

| EP | E/EP | naive (S=255) | eff (S=255) | naive (S=1024) | eff (S=1024) | naive (S=4096) | eff (S=4096) |
|---:|---:|---|---|---|---|---|---|
| 32 | 8 | `AI=1.87e+03, TF=1875.5` | `AI=4.60e+03, TF=1875.5` | `AI=2.06e+03, TF=1872.3` | `AI=7.39e+03, TF=1872.3` | `AI=1.14e+03, TF=1861.9` | `AI=1.02e+04, TF=1861.9` |
| 64 | 4 | `AI=2.14e+03, TF=1875.5` | `AI=5.45e+03, TF=1875.5` | `AI=2.13e+03, TF=1872.3` | `AI=7.85e+03, TF=1872.3` | `AI=1.14e+03, TF=1861.9` | `AI=1.03e+04, TF=1861.9` |
| 128 | 2 | `AI=2.30e+03, TF=1875.5` | `AI=6.00e+03, TF=1875.5` | `AI=2.17e+03, TF=1872.3` | `AI=8.11e+03, TF=1872.3` | `AI=1.14e+03, TF=1861.9` | `AI=1.04e+04, TF=1861.9` |

#### 6.6.6. Mode Share Overview (naive vs efficient)

We visualize category shares to separate "where parameters live" from "what limits runtime" under this analytic model. Here, a "share" is computed from the modeled totals: FLOPs share is a fraction of `F_theory`, and bytes share is a fraction of `bytes_total` (HBM plus any network attribution), both aggregated per category. We report shares per GPU under the fixed parallelism setting (`TP=1`, `EP=64`), so the MoE expert set per GPU (`E/EP`) and dispatch attribution are consistent with the KPIs. This plot exists to prevent the common MoE fallacy: experts can dominate parameters while attention/KV or other streaming terms dominate runtime.

![](deepseek_model_report-1_naive_mode_category_stacks.png)

![](deepseek_model_report-1_efficient_mode_category_stacks.png)

#### 6.6.7. Category Roofline (Per Mode)

We aggregate entries by category and plot category-level points at (`AI_hbm`, `TF_est`) using the same time model as the mode-level KPIs. This view isolates which operator family is left of ridge (HBM-limited) versus right of ridge (compute-side) in each phase, and it helps explain why a phase-level regime label changes (or does not) when switching between `naive` and `efficient`.

**Training**

![](deepseek_model_report-1_training_category_roofline.png)

**Prefill**

![](deepseek_model_report-1_prefill_category_roofline.png)

**Decode**

![](deepseek_model_report-1_decode_category_roofline.png)

**Interpretation**

Category rooflines separate operator families by where their FLOPs and bytes come from. In decode, attention often sits at lower `AI_hbm` because KV reads grow with `L`, while GEMM-heavy categories (dense/experts) can sit closer to the compute side if weight streaming is sufficiently amortized. Importantly, parameter share does not imply runtime share; the bytes/FLOPs mix determines the regime.


### 6.7. Memory Feasibility

We include this section as a sanity check: the throughput regimes above only matter if the anchor workload is even plausible under the stated parallelism assumptions (`TP=1`, `EP=64`). We report a per-GPU resident-byte estimate for the same anchor (`B`, `S`, `L`) used elsewhere in the report, so we can see whether the model is parameter-dominated (common for large MoE) or KV/activation-dominated (common for long context) before we spend effort tuning kernels.

We deliberately keep the accounting coarse but consistent with `TP/EP`: we assume non-expert weights are sharded by `TP`, routed-expert weights are sharded by `TP×EP`, and KV/activations are per GPU. For inference we count `params + kv_cache`, with `kv_cache = B * L * sum(C_kv_per_layer) * A_kv`. For training we add gradients, optimizer+master weights, and a coarse saved-activation term. We do not model activation checkpoint schedules, allocator fragmentation, or overlap; interpret these totals as feasibility signals rather than allocator-accurate budgets.

| Item | Estimated Bytes |
|---|---:|
| Parameters (non-expert, per GPU) | 15.94 GB |
| Parameters (routed experts, per GPU) | 9.52 GB |
| Parameters (total, per GPU) | 25.46 GB |
| KV Cache (B=8, L=255) | 68.36 MB |
| Training Gradients | 25.46 GB |
| Training Optimizer+Master | 305.49 GB |
| Training Saved Activations (coarse) | 8.31 GB |
| Inference Total Resident | 25.52 GB |
| Training Total Resident (coarse) | 364.71 GB |
| HBM Budget | 141.00 GB |

Under these assumptions, inference fits the HBM budget: `yes` (assumed `param_bytes=1`, `kv_cache_bytes=1`), and training fits the HBM budget: `no` (coarse estimate).

### 6.8. Communication Envelope

We include this section to isolate when network can become a first-order limiter: routed-MoE dispatch/collect when experts are sharded across devices (`EP>1`). We report a simple envelope estimate (not a measured time) for activation bytes that must move to send token activations to experts and return expert outputs, which scales roughly with tokens, hidden size, and `top_k`. At the anchor setting (`EP=64`), we report both intra-device and inter-device attribution; when `EP=1` the inter-device portion is zero by construction, while larger `EP` values can make `T_net` material and motivate compression and overlap.

| Mode | Intra-device Dispatch | Inter-device Dispatch (est.) | Interconnect Time (ms, est.) |
|---|---:|---:|---:|
| `training` | 12.64 GB | 12.44 GB | 14.842 |
| `prefill` | 12.64 GB | 12.44 GB | 14.842 |
| `decode` | 50.75 MB | 49.96 MB | 0.058 |

## 7. Sensitivity Analysis

We run sensitivity sweeps because single-point KPIs can hide which knobs most strongly move decode performance. We rank knobs by effect size over the combinational grid to focus tuning effort on high-leverage controls. The output is a compact map from knob changes to `T_est` and regime changes. In the evidence below, we focus on which knobs most change `T_est` in decode, how regimes shift with `L`, KV dtype, and MoE routing (`top_k`), and whether naive vs efficient execution changes the ranking.

We run the `medium_full_grid` full combinational grid (total points: `432`) to quantify which configuration knobs most move decode `T_est` and regime labels under this static model.
The full sweep is saved as a CSV artifact: `deepseek_model_report-1_sensitivity.csv`.

| Exec | kv_dtype(B) | top_k | kv_rank_scale | hidden_scale | L | AI_hbm | T_est(ms) | MFU_est | Regime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `naive` | 1 | 2 | 0.50 | 0.75 | 2048 | 1.484e+01 | 3.382 | 3.599e-02 | `hbm-bound` |
| `naive` | 1 | 4 | 0.50 | 0.75 | 2048 | 1.754e+01 | 3.408 | 4.254e-02 | `hbm-bound` |
| `naive` | 1 | 8 | 0.50 | 0.75 | 2048 | 2.310e+01 | 3.416 | 5.604e-02 | `hbm-bound` |
| `naive` | 2 | 4 | 1.50 | 1.25 | 16384 | 3.151e+01 | 12.842 | 7.642e-02 | `hbm-bound` |
| `naive` | 2 | 8 | 1.50 | 1.25 | 16384 | 3.561e+01 | 12.856 | 8.637e-02 | `hbm-bound` |
| `efficient` | 1 | 2 | 0.50 | 0.75 | 2048 | 4.006e+01 | 1.253 | 9.717e-02 | `hbm-bound` |
| `efficient` | 1 | 4 | 0.50 | 0.75 | 2048 | 4.723e+01 | 1.266 | 1.146e-01 | `hbm-bound` |
| `efficient` | 1 | 8 | 0.50 | 0.75 | 2048 | 6.216e+01 | 1.270 | 1.508e-01 | `hbm-bound` |
| `efficient` | 1 | 4 | 1.50 | 1.25 | 16384 | 8.298e+01 | 7.627 | 1.287e-01 | `compute-bound` |
| `efficient` | 2 | 4 | 1.50 | 1.25 | 16384 | 6.407e+01 | 7.627 | 1.287e-01 | `compute-bound` |

As `L` increases, KV read bytes rise, so `AI_hbm` typically decreases and HBM-bound cases become more frequent; the sweep helps quantify how strongly this trend depends on KV dtype, routing `top_k`, and the execution-mode byte assumptions.

### 7.1. Knob Ranking (Decode)

We rank knobs using efficient-mode decode points; the score is the median `T_est(ms)` spread across knob values (larger means more leverage).
| Knob | Effect size | Median `T_est(ms)` by value |
|---|---:|---|
| hidden scale | 121.3% | `0.75:1.97, 1.0:2.94, 1.25:4.36` |
| decode KV length (L) | 90.0% | `16384:4.32, 2048:2.28, 4096:2.46, 8192:3.01` |
| KV rank scale | 45.2% | `0.5:2.66, 1.0:3.03, 1.5:3.86` |
| KV dtype bytes | 6.1% | `1:2.99, 2:3.17` |
| top-k experts | 0.9% | `2:3.07, 4:3.09, 8:3.10` |

How to use the CSV:
1. Filter rows to decode points for your target exec mode.
2. Plot `T_est_ms` vs `L` grouped by `kv_dtype_bytes` and `top_k`.
3. Track regime transitions to verify if a knob changes memory/compute behavior.

## 8. Architectural Limits

We synthesize the preceding sections into a simple systems view: training and prefill can move toward compute-favorable operation under efficient execution assumptions, while decode remains more sensitive to memory traffic and serving-batch constraints. In long-context decode, KV traffic growth with `L` drives the familiar `~1/L` intensity decline once KV bytes dominate. Finally, MoE dispatch is always present intra-device, but inter-device pressure becomes material only when `ep_size > 1`.

## 9. Appendix A: Full FLOP Derivations

- Detailed derivations are documented in `docs/model_info.md` (see the derivation appendix section).

## 10. Appendix B: Common Failure Modes / Debugging Checklist

- Full debugging checklist is documented in `docs/model_info.md` (see the checklist appendix section).