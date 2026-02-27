# DeepSeekMoE Auxiliary Losses (Expert vs Device Balance)

**Purpose**: Clarify DeepSeekMoE load-balance losses and map them to this repo's current
implementation.

**Audience**: Readers learning MoE router auxiliary objectives in the EP tutorial path.

**Prerequisites**: Familiarity with top-k routing and per-layer token/expert notation.

**Related Docs**:
- [TP + EP + DP Communication Guide](ep_tp_dp_communication.md)
- [TP + PP + EP + DP Communication Guide](pp_tp_ep_dp_communication.md)
- [TP + DP Backward Flow](tp_dp_communication.md)

This note explains the two load-balancing losses described in DeepSeekMoE
([arXiv:2401.06066](https://arxiv.org/abs/2401.06066)) and maps them to this repo's
current implementation.

## 1) Notation (per MoE layer)

- `T`: routed tokens in this layer
- `N`: routed expert count (shared experts are excluded)
- `K`: top-k routed experts per token
- `s[i, t]`: router probability mass for expert `i` on token `t`

From top-k routing:

- `count_i = sum_t 1(i in TopK(t))`
- `f_i = (N / (K*T)) * count_i` (realized routed-load proxy)
- `P_i = (1 / T) * sum_t s[i, t]` (importance proxy)

## 2) Expert-level balance loss (paper)

DeepSeekMoE defines:

`L_expbal = alpha_1 * sum_i(f_i * P_i)`

Intuition:

- `f_i` is non-differentiable top-k load information (acts like stop-grad weighting)
- `P_i` is differentiable router preference
- high-load + high-probability experts get penalized most

## 3) Device-level balance loss (paper)

When experts are EP-sharded, experts on one device can still become stragglers.
DeepSeekMoE adds a device-level term:

- Partition experts by device: `E_d`
- `f'_d = mean_{j in E_d}(f_j)`
- `P'_d = sum_{j in E_d}(P_j)`
- `L_devbal = alpha_2 * sum_d(f'_d * P'_d)`

This targets device-level skew directly.

## 4) What this repo currently implements

Current MoE router loss in this repo is expert-level only.

Code path:

- Router loss computation:
  [src/models/moe.py](../src/models/moe.py) in `TopKRouter._compute_aux_loss`
- Training objective composition:
  [examples/train_4d.py](../examples/train_4d.py) in `train_step`

Implemented form:

`aux_loss = N * sum_i(expert_fraction_i * expert_prob_i)`

where:

- `expert_fraction_i = count_i / (K*T)`
- `expert_prob_i = P_i`

So this is equivalent to paper-style expert-balance with fixed `alpha_1=1`
inside the router, then globally scaled by script-level `aux_loss_coef`:

`L_total = L_task + aux_loss_coef * sum_over_moe_layers(aux_loss_layer)`

## 5) Important current limitation

`L_devbal` (device-level balance) is not yet implemented in code.

That means balancing pressure is expert-level only. This is fine for learning
and small runs, but large EP runs may benefit from adding device-level loss.

In current canonical mapping, expert sharding is controlled by
`expert_model_parallel_size` directly. With TP+EP, each TP lane has its own
expert-parallel domain; expert gradients are synchronized on
`expert_data_parallel_group` (varying DP and TP for fixed expert shard).

## 6) Practical reading tip

When reading logs in `examples/train_4d.py`:

- `task` is language-model loss (`L_task`)
- `aux` is summed MoE expert-balance proxy
- `total` is `task + aux_loss_coef * aux`

## 7) Reference

- DeepSeekMoE paper: https://arxiv.org/abs/2401.06066
