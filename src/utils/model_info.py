"""
Model information dumping and analysis utilities.

Outputs a comprehensive Markdown report with:
- Architecture fingerprint
- Parameter and memory breakdown
- Weight statistics (optional)
- Static efficiency estimates for training/prefill/decode
- Roofline plots (H200 default)
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class LayerInfo:
    """Information about a single parameter tensor."""
    name: str
    shape: tuple
    num_params: int
    memory_mb: float
    dtype: torch.dtype
    requires_grad: bool
    mean: float
    std: float
    min: float
    max: float

    flops: float = 0.0
    arithmetic_intensity: float = 0.0


@dataclass
class ModuleSizeInfo:
    """Aggregated parameter stats for a module."""
    name: str
    num_params: int
    memory_mb: float
    percent_total: float


@dataclass
class ModulePatternInfo:
    """Aggregated parameter stats for repeated module patterns."""
    pattern: str
    instance_count: int
    params_per_instance: float
    total_params: int
    total_memory_mb: float
    percent_total: float
    example_module: str


@dataclass
class EfficiencyEntry:
    """
    Static efficiency estimate for one module and one mode.

    Note:
    - `flops_theory` is the symbolic FLOP count from shape-based formulas.
    - `flops_realizable` is a peak-equivalent compute cost used for `T_comp` estimation
      (see `_build_flop_breakdown`).
    """
    name: str
    kind: str
    flops: float
    flops_theory: float
    flops_tensorcore: float
    flops_realizable: float
    bytes_hbm: float
    bytes_net: float
    bytes_total: float
    bytes_weights: float
    bytes_activations: float
    bytes_kv: float
    bytes_temporary: float
    arithmetic_intensity_weights_only: float
    arithmetic_intensity_hbm: float
    arithmetic_intensity_total: float
    p_effective_tflops: float
    eta_tc: float
    regime: str
    roofline_tflops_hbm: float


@dataclass
class ModelInfo:
    """Comprehensive information about a model."""
    total_params: int
    trainable_params: int
    non_trainable_params: int
    total_memory_mb: float
    layers: List[LayerInfo]
    num_layers: int

    total_flops: float = 0.0
    architecture_type: str = ""
    bottlenecks: List[str] = field(default_factory=list)

    report_path: str = ""
    plot_paths: List[str] = field(default_factory=list)


@dataclass
class RooflineConfig:
    """Roofline hardware configuration."""
    name: str = "H200_SXM_FP8"
    peak_tflops: float = 1979.0
    mem_bw_gbps: float = 4800.0


def default_roofline_targets() -> List[RooflineConfig]:
    """
    Return default chip roofline targets (FP8 dense tensor throughput).

    Notes:
    - H200 uses 1979 TFLOPs FP8 dense and ~4.8 TB/s HBM bandwidth.
    - B200 uses 4500 TFLOPs FP8 dense and ~8.0 TB/s HBM bandwidth.
    """
    return [
        RooflineConfig(name="H200_SXM_FP8", peak_tflops=1979.0, mem_bw_gbps=4800.0),
        RooflineConfig(name="B200_SXM_FP8", peak_tflops=4500.0, mem_bw_gbps=8000.0),
    ]


@dataclass
class ExecutionModelConfig:
    """Execution assumptions for static FLOPs/bytes estimation."""

    name: str
    attention_bytes_model: str
    weight_residency_attn: float
    weight_residency_dense: float
    weight_residency_moe: float
    activation_fusion_factor: float
    elementwise_bytes_factor: float

    def validate(self) -> None:
        """Validate mode knobs and accepted enum values."""
        if self.attention_bytes_model not in {"naive", "flash"}:
            raise ValueError(
                "attention_bytes_model must be one of {'naive', 'flash'}"
            )
        factors = {
            "weight_residency_attn": self.weight_residency_attn,
            "weight_residency_dense": self.weight_residency_dense,
            "weight_residency_moe": self.weight_residency_moe,
            "activation_fusion_factor": self.activation_fusion_factor,
            "elementwise_bytes_factor": self.elementwise_bytes_factor,
        }
        for name, value in factors.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value}")


@dataclass
class TensorCoreModelConfig:
    """Tensor core efficiency model used for realizable FLOP estimates."""

    enabled: bool = True
    b_sat: int = 64
    eta_scalar: float = 0.35
    eligible_attention: float = 0.95
    eligible_linear: float = 0.98
    eligible_embedding: float = 0.70
    eligible_network: float = 0.0

    def validate(self) -> None:
        """Validate tensor-core model fields."""
        if self.b_sat <= 0:
            raise ValueError(f"b_sat must be > 0, got {self.b_sat}")
        if not (0.0 <= self.eta_scalar <= 1.0):
            raise ValueError(f"eta_scalar must be in [0, 1], got {self.eta_scalar}")
        for field_name in [
            "eligible_attention",
            "eligible_linear",
            "eligible_embedding",
            "eligible_network",
        ]:
            value = float(getattr(self, field_name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{field_name} must be in [0, 1], got {value}")


@dataclass
class FlopBreakdown:
    """FLOP decomposition for one module/mode entry."""

    theory: float
    tensorcore: float
    realizable: float
    p_effective_tflops: float
    eta_tc: float


@dataclass
class ByteBreakdown:
    """HBM byte decomposition for one module/mode entry."""

    weights: float
    activations: float
    kv: float
    temporary: float

    @property
    def hbm_total(self) -> float:
        """Total HBM bytes."""
        return self.weights + self.activations + self.kv + self.temporary


@dataclass
class SensitivityConfig:
    """Configuration for design-space sensitivity analysis."""

    name: str = "medium_full_grid"
    kv_dtype_bytes: Tuple[int, ...] = (1, 2)
    top_k_values: Tuple[int, ...] = (2, 4, 8)
    kv_rank_scales: Tuple[float, ...] = (0.5, 1.0, 1.5)
    hidden_scales: Tuple[float, ...] = (0.75, 1.0, 1.25)
    cache_lengths: Tuple[int, ...] = (2048, 4096, 8192, 16384)


@dataclass
class SensitivityPoint:
    """One sensitivity sweep point."""

    exec_model: str
    kv_dtype_bytes: int
    top_k: int
    kv_rank_scale: float
    hidden_scale: float
    cache_len: int
    ai_hbm: float
    ai_total: float
    t_est_ms: float
    mfu_est: float
    regime: str


@dataclass
class ModeKpiExtended:
    """
    Aggregated mode-level KPIs with extended FLOP and regime fields.

    `flops_realizable` is a peak-equivalent compute cost in FLOP units, so `t_comp` is computed as:
      `t_comp = flops_realizable / (P_peak * 1e12)`.
    """

    mode: str
    flops_theory: float
    flops_tensorcore: float
    flops_realizable: float
    bytes_weights: float
    bytes_activations: float
    bytes_kv: float
    bytes_temporary: float
    bytes_hbm: float
    bytes_net: float
    bytes_total: float
    ai_weights_only: float
    ai_hbm: float
    ai_total: float
    p_effective_tflops: float
    t_comp: float
    t_hbm: float
    t_net: float
    t_est: float
    roofline_tflops_hbm: float
    peak_pct: float
    mfu_est: float
    regime: str
    b_crit: Optional[float]


def default_execution_models() -> List[ExecutionModelConfig]:
    """Return conservative execution-mode knobs (not measured kernel claims)."""
    return [
        ExecutionModelConfig(
            name="naive",
            attention_bytes_model="naive",
            weight_residency_attn=1.0,
            weight_residency_dense=1.0,
            weight_residency_moe=1.0,
            activation_fusion_factor=1.0,
            elementwise_bytes_factor=1.0,
        ),
        ExecutionModelConfig(
            name="efficient",
            attention_bytes_model="flash",
            weight_residency_attn=4.0,
            weight_residency_dense=4.0,
            weight_residency_moe=2.0,
            activation_fusion_factor=0.5,
            elementwise_bytes_factor=0.7,
        ),
    ]


@dataclass
class SweepPoint:
    """Model-level sweep sample for one operating point."""

    x: int
    flops: float
    bytes_hbm: float
    bytes_net: float
    bytes_total: float
    ai_hbm: float
    ai_total: float
    roofline_tflops_hbm: float
    regime_hbm: str
    t_comp_ms: float
    t_hbm_ms: float
    t_net_ms: float
    t_est_ms: float
    flops_realizable: float = 0.0


@dataclass
class DecodeWorkloadPoint:
    """Decode operating point for workload sweeps varying (B, L, EP)."""

    batch: int
    cache_len: int
    ep_size: int
    flops: float
    flops_realizable: float
    bytes_hbm: float
    bytes_net: float
    bytes_total: float
    ai_hbm: float
    ai_total: float
    regime: str
    t_comp_ms: float
    t_hbm_ms: float
    t_net_ms: float
    t_est_ms: float


@dataclass
class PrefillWorkloadPoint:
    """Prefill operating point for workload sweeps varying (B, S, EP)."""

    batch: int
    seq_len: int
    ep_size: int
    flops: float
    flops_realizable: float
    bytes_hbm: float
    bytes_net: float
    bytes_total: float
    ai_hbm: float
    ai_total: float
    regime: str
    t_comp_ms: float
    t_hbm_ms: float
    t_net_ms: float
    t_est_ms: float


@dataclass
class DecodeEPExplorationConfig:
    """Configuration for decode compute-bound exploration under DP=EP, TP=1 assumptions."""
    ep_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    cache_lengths: List[int] = field(default_factory=lambda: [2048, 4096, 8192, 16384])
    alpha: float = 0.9  # "close to compute-bound" threshold: T_comp >= alpha * max(T_hbm, T_net)
    max_batch_per_gpu: int = 16384
    weight_residency_factor: float = 1.0  # 1.0 = stream weights; >1 amortizes weights (effective bytes /= factor)
    kv_element_bytes: Optional[int] = None  # if None, use kv_cache_bytes


@dataclass
class ParallelismAssumptions:
    """Per-report parallelism assumptions used to interpret MoE sharding and network attribution."""

    tp_size: int
    ep_size: int
    n_routed_experts: Optional[int]
    routed_experts_per_gpu: Optional[int]
    ep_from_config: bool


def _format_number(num: float) -> str:
    """Format large number with comma separators."""
    return f"{int(num):,}"


def _format_bytes(num_bytes: float) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def _format_flops(flops: float) -> str:
    """Format FLOPs with human readable suffix."""
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TF"
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} GF"
    if flops >= 1e6:
        return f"{flops / 1e6:.2f} MF"
    if flops >= 1e3:
        return f"{flops / 1e3:.2f} KF"
    return f"{flops:.2f} F"


def _safe_div(numer: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return numer / denom


def _tc_eligibility_for_kind(kind: str, tc_cfg: TensorCoreModelConfig) -> float:
    """Return tensor-core eligible FLOP ratio by module kind."""
    if kind == "attention":
        return tc_cfg.eligible_attention
    if kind in {"linear", "moe"}:
        return tc_cfg.eligible_linear
    if kind == "embedding":
        return tc_cfg.eligible_embedding
    return tc_cfg.eligible_network


def _eta_tc(batch_size: int, tc_cfg: TensorCoreModelConfig) -> float:
    """
    Simple tensor-core saturation model eta_tc(M)=min(1, M/M_sat).

    Historically this was written as eta_tc(B) for decode (where GEMM M ~= batch). In this report
    we treat the input as an "effective M dimension" proxy:
    - decode: M ~= B
    - dense prefill/training: M ~= B*S
    - MoE experts: M ~= tokens per active expert
    """
    if not tc_cfg.enabled:
        return 1.0
    return min(1.0, float(batch_size) / float(tc_cfg.b_sat))


def _build_flop_breakdown(
    flops_theory: float,
    kind: str,
    batch_size: int,
    roofline: RooflineConfig,
    tc_cfg: TensorCoreModelConfig,
    tc_m_dim: Optional[int] = None,
) -> FlopBreakdown:
    """
    Create theory/tensor-core/realizable FLOP decomposition.

    Note on `realizable`:
    We store a *peak-equivalent compute cost* in FLOP units that makes the compute-time estimate
    consistent when tensor-core utilization is < 1.

    In other words, `realizable` is not "achieved FLOPs". It is the amount of peak throughput work
    that would take the same time as the modeled execution.
    """
    # Tensor-core utilization should depend on the effective GEMM "M" dimension. For dense
    # prefill/training GEMMs, M typically scales with tokens (B*S), while decode often has M ~= B.
    # Callers can pass `tc_m_dim` to make this shape-aware; otherwise we fall back to `batch_size`
    # for backward compatibility.
    work_m = batch_size if tc_m_dim is None else int(tc_m_dim)
    eta = _eta_tc(work_m, tc_cfg)
    eligible_ratio = _tc_eligibility_for_kind(kind, tc_cfg)
    flops_tensorcore = flops_theory * eligible_ratio
    if tc_cfg.enabled:
        # Peak-equivalent compute cost:
        #   F_cost = F_tc / eta_tc + (F_theory - F_tc) / eta_scalar
        # so that T_comp = F_cost / P_peak.
        denom_tc = max(1e-12, float(eta))
        denom_scalar = max(1e-12, float(tc_cfg.eta_scalar))
        flops_realizable = (
            (flops_tensorcore / denom_tc) +
            ((flops_theory - flops_tensorcore) / denom_scalar)
        )
        p_effective_tflops = roofline.peak_tflops * _safe_div(flops_theory, flops_realizable)
    else:
        flops_realizable = flops_theory
        p_effective_tflops = roofline.peak_tflops
        eta = 1.0
    return FlopBreakdown(
        theory=flops_theory,
        tensorcore=flops_tensorcore,
        realizable=flops_realizable,
        p_effective_tflops=p_effective_tflops,
        eta_tc=eta,
    )


def _regime_label(
    flops: float,
    bytes_hbm: float,
    bytes_net: float,
    roofline: RooflineConfig,
    interconnect_bw_gbps: float,
) -> str:
    """Return limiting regime label from compute/HBM/network times."""
    t_comp, t_hbm, t_net, _ = _estimate_mode_time_seconds(
        total_flops=flops,
        total_hbm_bytes=bytes_hbm,
        total_net_bytes=bytes_net,
        roofline=roofline,
        interconnect_bw_gbps=interconnect_bw_gbps,
    )
    if t_comp >= t_hbm and t_comp >= t_net:
        return "compute-bound"
    if t_hbm >= t_comp and t_hbm >= t_net:
        return "hbm-bound"
    return "network-bound"


def _b_crit_from_sweep(points: List[SweepPoint], ai_knee: float) -> Optional[float]:
    """
    Estimate the first batch where AI_hbm crosses knee using linear interpolation.
    Returns None if no crossing occurs in the sampled range.
    """
    if not points:
        return None
    ordered = sorted(points, key=lambda point: point.x)
    if ordered[0].ai_hbm >= ai_knee:
        return float(ordered[0].x)
    for idx in range(len(ordered) - 1):
        left = ordered[idx]
        right = ordered[idx + 1]
        if left.ai_hbm < ai_knee <= right.ai_hbm:
            denom = right.ai_hbm - left.ai_hbm
            if denom <= 0:
                return float(right.x)
            ratio = (ai_knee - left.ai_hbm) / denom
            return float(left.x + ratio * (right.x - left.x))
    return None


def _default_ep_sweep_sizes(
    n_routed_experts: Optional[int],
    anchor_ep_size: int,
) -> List[int]:
    """Return a small EP sweep list around the anchor (E/8, E/4, E/2) when E is known."""
    anchor_ep_size = max(1, int(anchor_ep_size))
    if n_routed_experts is None or n_routed_experts <= 0:
        return [anchor_ep_size]

    E = int(n_routed_experts)
    candidates = [
        int(math.ceil(float(E) / 8.0)),
        int(math.ceil(float(E) / 4.0)),
        int(math.ceil(float(E) / 2.0)),
        anchor_ep_size,
    ]
    eps = sorted({max(1, min(E, int(ep))) for ep in candidates})
    if anchor_ep_size not in eps:
        eps.append(anchor_ep_size)
        eps = sorted(set(eps))
    return eps


def _default_decode_cache_lengths(
    anchor_seq_len: int,
    max_position_embeddings: Optional[int],
) -> List[int]:
    """Return decode KV-cache lengths `L` to evaluate, filtered to model max positions if known."""
    anchor_seq_len = max(1, int(anchor_seq_len))
    candidates = [anchor_seq_len, 2048, 4096, 8192, 16384]
    max_pos = int(max_position_embeddings) if max_position_embeddings is not None else None
    lengths: List[int] = []
    for value in candidates:
        if value <= 0:
            continue
        if max_pos is not None and value > max_pos:
            continue
        lengths.append(int(value))
    return sorted(set(lengths))


def _default_prefill_batch_sizes(anchor_batch_size: int) -> List[int]:
    """Return a small prefill batch sweep list used for roofline-space curves."""
    candidates = [1, int(anchor_batch_size), 32]
    return sorted({b for b in candidates if b > 0})


def _run_decode_workload_sweep(
    model: torch.nn.Module,
    execution_models: List[ExecutionModelConfig],
    batch_sizes: List[int],
    cache_lengths: List[int],
    ep_sizes: List[int],
    activation_bytes: int,
    kv_cache_bytes: int,
    param_bytes_assumed: int,
    roofline: RooflineConfig,
    interconnect_bw_gbps: float,
    training_flops_multiplier: float,
    training_bytes_multiplier: float,
    tc_cfg: TensorCoreModelConfig,
    module_weight_bytes_cache: Optional[Dict[int, float]],
    progress_cb=None,
) -> Dict[str, List[DecodeWorkloadPoint]]:
    """Return decode sweep points for the cartesian product of (B, L, EP) for each exec model."""
    points_by_exec: Dict[str, List[DecodeWorkloadPoint]] = {
        exec_model.name: [] for exec_model in execution_models
    }
    total = max(1, len(execution_models) * len(batch_sizes) * len(cache_lengths) * len(ep_sizes))
    done = 0
    for exec_model in execution_models:
        for ep_size in ep_sizes:
            for cache_len in cache_lengths:
                for batch in batch_sizes:
                    done += 1
                    if progress_cb is not None and done % 50 == 0:
                        progress_cb(f"Decode workload sweep progress: {done}/{total}")
                    eff = _estimate_efficiency(
                        model=model,
                        batch_size=batch,
                        seq_len=cache_len,
                        activation_bytes=activation_bytes,
                        kv_cache_bytes=kv_cache_bytes,
                        param_bytes_assumed=param_bytes_assumed,
                        roofline=roofline,
                        interconnect_bw_gbps=interconnect_bw_gbps,
                        training_flops_multiplier=training_flops_multiplier,
                        training_bytes_multiplier=training_bytes_multiplier,
                        exec_model=exec_model,
                        tc_cfg=tc_cfg,
                        module_weight_bytes_cache=module_weight_bytes_cache,
                        modes_to_estimate=("decode",),
                        ep_size_override=ep_size,
                    )
                    stats = _summarize_mode_entries(
                        mode="decode",
                        entries_for_mode=eff["decode"],
                        roofline=roofline,
                        interconnect_bw_gbps=interconnect_bw_gbps,
                    )
                    points_by_exec[exec_model.name].append(
                        DecodeWorkloadPoint(
                            batch=batch,
                            cache_len=cache_len,
                            ep_size=ep_size,
                            flops=stats.flops_theory,
                            flops_realizable=stats.flops_realizable,
                            bytes_hbm=stats.bytes_hbm,
                            bytes_net=stats.bytes_net,
                            bytes_total=stats.bytes_total,
                            ai_hbm=stats.ai_hbm,
                            ai_total=stats.ai_total,
                            regime=stats.regime,
                            t_comp_ms=stats.t_comp * 1000.0,
                            t_hbm_ms=stats.t_hbm * 1000.0,
                            t_net_ms=stats.t_net * 1000.0,
                            t_est_ms=stats.t_est * 1000.0,
                        )
                    )
    return points_by_exec


def _run_prefill_workload_sweep(
    model: torch.nn.Module,
    execution_models: List[ExecutionModelConfig],
    batch_sizes: List[int],
    seq_lengths: List[int],
    ep_sizes: List[int],
    activation_bytes: int,
    kv_cache_bytes: int,
    param_bytes_assumed: int,
    roofline: RooflineConfig,
    interconnect_bw_gbps: float,
    training_flops_multiplier: float,
    training_bytes_multiplier: float,
    tc_cfg: TensorCoreModelConfig,
    module_weight_bytes_cache: Optional[Dict[int, float]],
    progress_cb=None,
) -> Dict[str, List[PrefillWorkloadPoint]]:
    """Return prefill sweep points for the cartesian product of (B, S, EP) for each exec model."""
    points_by_exec: Dict[str, List[PrefillWorkloadPoint]] = {
        exec_model.name: [] for exec_model in execution_models
    }
    total = max(1, len(execution_models) * len(batch_sizes) * len(seq_lengths) * len(ep_sizes))
    done = 0
    for exec_model in execution_models:
        for ep_size in ep_sizes:
            for seq_len in seq_lengths:
                for batch in batch_sizes:
                    done += 1
                    if progress_cb is not None and done % 50 == 0:
                        progress_cb(f"Prefill workload sweep progress: {done}/{total}")
                    eff = _estimate_efficiency(
                        model=model,
                        batch_size=batch,
                        seq_len=seq_len,
                        activation_bytes=activation_bytes,
                        kv_cache_bytes=kv_cache_bytes,
                        param_bytes_assumed=param_bytes_assumed,
                        roofline=roofline,
                        interconnect_bw_gbps=interconnect_bw_gbps,
                        training_flops_multiplier=training_flops_multiplier,
                        training_bytes_multiplier=training_bytes_multiplier,
                        exec_model=exec_model,
                        tc_cfg=tc_cfg,
                        module_weight_bytes_cache=module_weight_bytes_cache,
                        modes_to_estimate=("prefill",),
                        ep_size_override=ep_size,
                    )
                    stats = _summarize_mode_entries(
                        mode="prefill",
                        entries_for_mode=eff["prefill"],
                        roofline=roofline,
                        interconnect_bw_gbps=interconnect_bw_gbps,
                    )
                    points_by_exec[exec_model.name].append(
                        PrefillWorkloadPoint(
                            batch=batch,
                            seq_len=seq_len,
                            ep_size=ep_size,
                            flops=stats.flops_theory,
                            flops_realizable=stats.flops_realizable,
                            bytes_hbm=stats.bytes_hbm,
                            bytes_net=stats.bytes_net,
                            bytes_total=stats.bytes_total,
                            ai_hbm=stats.ai_hbm,
                            ai_total=stats.ai_total,
                            regime=stats.regime,
                            t_comp_ms=stats.t_comp * 1000.0,
                            t_hbm_ms=stats.t_hbm * 1000.0,
                            t_net_ms=stats.t_net * 1000.0,
                            t_est_ms=stats.t_est * 1000.0,
                        )
                    )
    return points_by_exec


def _sensitivity_config_from_profile(profile: str) -> SensitivityConfig:
    """Create sensitivity config from profile name."""
    if profile != "medium_full_grid":
        raise ValueError(
            f"Unsupported sensitivity_profile: {profile}. "
            "Supported: {'medium_full_grid'}"
        )
    return SensitivityConfig(name=profile)


def _ensure_unique_path(path: str) -> str:
    """Add numeric suffix if path exists."""
    base, ext = os.path.splitext(path)
    if not os.path.exists(path):
        return path
    idx = 1
    while True:
        candidate = f"{base}-{idx}{ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def _number_markdown_headings(
    markdown_text: str,
    min_level: int = 2,
    max_level: int = 4,
) -> str:
    """
    Add hierarchical section numbers to markdown headings.

    Example:
      ## A
      ### B
      ### C
    becomes:
      ## 1. A
      ### 1.1. B
      ### 1.2. C
    """
    lines = markdown_text.splitlines()
    counters = [0] * 7
    in_code_block = False
    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        match = heading_re.match(line)
        if match is None:
            continue
        level = len(match.group(1))
        if level < min_level or level > max_level:
            continue

        counters[level] += 1
        for deeper in range(level + 1, 7):
            counters[deeper] = 0

        parts = [str(counters[lvl]) for lvl in range(min_level, level + 1) if counters[lvl] > 0]
        if not parts:
            continue
        prefix = ".".join(parts)
        heading_text = match.group(2)
        lines[idx] = f"{match.group(1)} {prefix}. {heading_text}"

    # Preserve trailing newline behavior from caller.
    return "\n".join(lines)


def _infer_architecture(model: torch.nn.Module) -> Dict[str, str]:
    """Infer architecture properties from the model object."""
    info = {
        "family": "Unknown",
        "attention_type": "None",
        "position_encoding": "Unknown",
        "moe": "None",
        "weight_tying": "No",
        "normalization": "Unknown",
        "activation": "Unknown",
    }

    if any(isinstance(m, torch.nn.TransformerEncoderLayer) for m in model.modules()):
        info["family"] = "Transformer"

    for name, module in model.named_modules():
        name_lower = name.lower()
        if "attention" in name_lower or "attn" in name_lower:
            info["family"] = "Transformer"
            if hasattr(module, "q_a_proj") and hasattr(module, "kv_a_proj"):
                info["attention_type"] = "MLA"
            if hasattr(module, "num_heads") and hasattr(module, "head_dim"):
                info["attention_type"] = "MHA"
            if hasattr(module, "qkv_proj"):
                info["attention_type"] = "Fused QKV"
        if "moe" in name_lower or "expert" in name_lower:
            info["moe"] = "Detected"
        if isinstance(module, torch.nn.RMSNorm) or module.__class__.__name__.lower() == "rmsnorm":
            info["normalization"] = "RMSNorm"
        if isinstance(module, torch.nn.LayerNorm):
            info["normalization"] = "LayerNorm"
        if isinstance(module, torch.nn.GELU):
            info["activation"] = "GELU"
        if isinstance(module, torch.nn.ReLU):
            info["activation"] = "ReLU"

    if hasattr(model, "position_embeddings"):
        info["position_encoding"] = "Learned Absolute"
    elif hasattr(model, "config") and hasattr(model.config, "rope_theta"):
        if hasattr(model.config, "rope_scaling") and getattr(model.config, "rope_scaling") is not None:
            info["position_encoding"] = "RoPE (scaled)"
        else:
            info["position_encoding"] = "RoPE"

    if hasattr(model, "lm_head") and hasattr(model, "token_embeddings"):
        tied_by_reference = model.lm_head.weight is model.token_embeddings.weight
        tied_by_storage = False
        if not tied_by_reference:
            lm_weight = model.lm_head.weight
            tok_weight = model.token_embeddings.weight
            if not getattr(lm_weight, "is_meta", False) and not getattr(tok_weight, "is_meta", False):
                tied_by_storage = lm_weight.data_ptr() == tok_weight.data_ptr()
        if tied_by_reference or tied_by_storage:
            info["weight_tying"] = "Yes"

    if hasattr(model, "config"):
        info["family"] = "Transformer"

    return info


def _infer_parallelism_assumptions(
    model: torch.nn.Module,
    target_routed_experts_per_gpu: int = 4,
) -> ParallelismAssumptions:
    """
    Infer TP/EP assumptions for report generation.

    Rationale:
    Many model configs do not declare `ep_size`. For large routed-MoE models, `EP=1` implies all
    experts live on one GPU, which is typically unrealistic. When `ep_size` is not explicitly
    provided by the model config, we infer EP by targeting a small routed-expert set per GPU.
    """
    cfg = getattr(model, "config", None)
    tp_size = 1
    if cfg is not None and hasattr(cfg, "tp_size"):
        try:
            tp_size = int(getattr(cfg, "tp_size") or 1)
        except Exception:
            tp_size = 1
    tp_size = max(1, int(tp_size))

    n_routed_experts: Optional[int] = None
    if cfg is not None:
        for key in ["n_routed_experts", "num_experts", "n_experts", "experts"]:
            if not hasattr(cfg, key):
                continue
            value = getattr(cfg, key)
            if isinstance(value, int) and value > 0:
                n_routed_experts = int(value)
                break

    ep_size = 1
    ep_from_config = False
    if cfg is not None and hasattr(cfg, "ep_size"):
        try:
            cfg_ep_size = int(getattr(cfg, "ep_size") or 0)
        except Exception:
            cfg_ep_size = 0
        if cfg_ep_size > 0:
            ep_size = cfg_ep_size
            ep_from_config = True

    if (
        not ep_from_config
        and n_routed_experts is not None
        and n_routed_experts > 0
    ):
        target = max(1, int(target_routed_experts_per_gpu))
        ep_size = int(math.ceil(float(n_routed_experts) / float(target)))

    ep_size = max(1, int(ep_size))
    if n_routed_experts is not None and n_routed_experts > 0:
        ep_size = min(ep_size, int(n_routed_experts))
        routed_experts_per_gpu = int(math.ceil(float(n_routed_experts) / float(ep_size)))
    else:
        routed_experts_per_gpu = None

    return ParallelismAssumptions(
        tp_size=tp_size,
        ep_size=ep_size,
        n_routed_experts=n_routed_experts,
        routed_experts_per_gpu=routed_experts_per_gpu,
        ep_from_config=ep_from_config,
    )


def _collect_layer_info(model: torch.nn.Module) -> List[LayerInfo]:
    layers = []
    for name, param in model.named_parameters():
        num_params = param.numel()
        memory_bytes = num_params * param.element_size()
        memory_mb = memory_bytes / (1024 ** 2)

        if getattr(param, "is_meta", False):
            # Meta tensors have no backing storage, so value statistics are unavailable.
            stats = {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
        else:
            with torch.no_grad():
                stats = {
                    "mean": param.data.mean().item(),
                    "std": param.data.std().item(),
                    "min": param.data.min().item(),
                    "max": param.data.max().item(),
                }

        layers.append(
            LayerInfo(
                name=name,
                shape=tuple(param.shape),
                num_params=num_params,
                memory_mb=memory_mb,
                dtype=param.dtype,
                requires_grad=param.requires_grad,
                mean=stats["mean"],
                std=stats["std"],
                min=stats["min"],
                max=stats["max"],
            )
        )

    return layers


def _aggregate_module_sizes(layers: List[LayerInfo]) -> List[ModuleSizeInfo]:
    totals: Dict[str, Tuple[int, float]] = {}
    total_params = sum(layer.num_params for layer in layers) or 1

    for layer in layers:
        if "." in layer.name:
            module_name = layer.name.rsplit(".", 1)[0]
        else:
            module_name = layer.name
        params, mem = totals.get(module_name, (0, 0.0))
        totals[module_name] = (params + layer.num_params, mem + layer.memory_mb)

    modules = []
    for name, (params, mem) in totals.items():
        modules.append(
            ModuleSizeInfo(
                name=name,
                num_params=params,
                memory_mb=mem,
                percent_total=100.0 * params / total_params,
            )
        )

    modules.sort(key=lambda m: m.num_params, reverse=True)
    return modules


def _module_pattern_name(module_name: str) -> str:
    """Collapse numeric indices so repeated layers are grouped into one pattern."""
    return re.sub(r"\.\d+(\.|$)", ".*\\1", module_name)


def _aggregate_module_patterns(modules: List[ModuleSizeInfo]) -> List[ModulePatternInfo]:
    """Aggregate module stats by repeated structural pattern."""
    total_params = sum(module.num_params for module in modules) or 1
    grouped: Dict[str, Dict[str, object]] = {}

    for module in modules:
        pattern = _module_pattern_name(module.name)
        if pattern not in grouped:
            grouped[pattern] = {
                "instance_count": 0,
                "total_params": 0,
                "total_memory_mb": 0.0,
                "example_module": module.name,
            }
        grouped[pattern]["instance_count"] = int(grouped[pattern]["instance_count"]) + 1
        grouped[pattern]["total_params"] = int(grouped[pattern]["total_params"]) + module.num_params
        grouped[pattern]["total_memory_mb"] = (
            float(grouped[pattern]["total_memory_mb"]) + module.memory_mb
        )

    patterns: List[ModulePatternInfo] = []
    for pattern, data in grouped.items():
        instance_count = int(data["instance_count"])
        total_pattern_params = int(data["total_params"])
        total_pattern_memory = float(data["total_memory_mb"])
        patterns.append(
            ModulePatternInfo(
                pattern=pattern,
                instance_count=instance_count,
                params_per_instance=total_pattern_params / max(1, instance_count),
                total_params=total_pattern_params,
                total_memory_mb=total_pattern_memory,
                percent_total=100.0 * total_pattern_params / total_params,
                example_module=str(data["example_module"]),
            )
        )

    patterns.sort(key=lambda p: p.total_params, reverse=True)
    return patterns


def _categorize_module(name: str) -> str:
    lower = name.lower()
    if "expert" in lower or "moe" in lower:
        return "experts"
    if "embedding" in lower:
        return "embedding"
    if (
        "qkv" in lower
        or "q_proj" in lower
        or "k_proj" in lower
        or "v_proj" in lower
        or "q_a_proj" in lower
        or "q_b_proj" in lower
        or "kv_a_proj" in lower
        or "kv_b_proj" in lower
    ):
        return "attn-qkv"
    if "out_proj" in lower or "o_proj" in lower:
        return "attn-out"
    if "attn" in lower or "attention" in lower:
        return "attention"
    if "mlp" in lower or "ffn" in lower or "fc" in lower:
        return "ffn"
    if "norm" in lower:
        return "norm"
    return "other"


def _categorize_efficiency_entry(entry: EfficiencyEntry) -> str:
    """
    Categorize efficiency entries for report aggregation/plots.

    We prefer `entry.kind` for network attribution, since network entries may not look like
    "network" in their names (for example `moe.dispatch.interconnect`).
    """
    if entry.kind == "network":
        return "network"
    return _categorize_module(entry.name)


def _aggregate_module_categories(modules: List[ModuleSizeInfo]) -> Dict[str, int]:
    totals: Dict[str, int] = {}
    for module in modules:
        category = _categorize_module(module.name)
        totals[category] = totals.get(category, 0) + module.num_params
    return totals


def _attention_module_names(model: torch.nn.Module) -> List[str]:
    names = []
    for name, module in model.named_modules():
        has_attention_proj = (
            hasattr(module, "qkv_proj")
            or (
                hasattr(module, "q_proj")
                and hasattr(module, "k_proj")
                and hasattr(module, "v_proj")
            )
            or (
                hasattr(module, "q_a_proj")
                and hasattr(module, "kv_a_proj")
                and hasattr(module, "kv_b_proj")
            )
        )
        if has_attention_proj and hasattr(module, "out_proj"):
            if hasattr(module, "num_heads"):
                names.append(name)
    return names


def _build_moe_active_fractions(
    model: torch.nn.Module,
    top_k_override: Optional[int] = None,
) -> Dict[str, float]:
    """
    Build per-module active fractions for routed-MoE execution.

    Routed experts process only a fraction of tokens on average:
      active_fraction ~= top_k / num_experts
    Shared experts and routers are always active.
    """
    fractions: Dict[str, float] = {}
    for parent_name, parent_module in model.named_modules():
        if not hasattr(parent_module, "num_experts") or not hasattr(parent_module, "top_k"):
            continue

        num_experts = max(1, int(getattr(parent_module, "num_experts")))
        if top_k_override is None:
            top_k = int(getattr(parent_module, "top_k"))
        else:
            top_k = int(top_k_override)
        top_k = max(1, min(top_k, num_experts))
        active_fraction = float(top_k) / float(num_experts)

        for child_name, _ in parent_module.named_modules():
            full_name = f"{parent_name}.{child_name}" if child_name else parent_name
            if ".experts." in full_name:
                fractions[full_name] = active_fraction
            elif ".shared_experts." in full_name:
                fractions[full_name] = 1.0
            elif ".router" in full_name:
                fractions[full_name] = 1.0

    return fractions


def _collect_kv_cache_elements_per_token(model: torch.nn.Module) -> Dict[str, int]:
    """Collect KV-cache elements per token for each attention module."""
    attention_names = _attention_module_names(model)
    result: Dict[str, int] = {}
    for name, module in model.named_modules():
        if name not in attention_names:
            continue
        hidden = getattr(module, "hidden_size", None)
        if hidden is None:
            hidden = getattr(module, "num_heads", 1) * getattr(module, "head_dim", 0)
        result[name] = _infer_kv_cache_elements_per_token(module, int(hidden))
    return result


def _estimate_activation_memory_bytes(
    model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    activation_bytes: int,
    activation_multiplier: float = 10.0,
) -> float:
    """
    Coarse activation-memory estimate for training.

    activation_multiplier captures saved intermediates for backward,
    residuals, and framework overhead.
    """
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    num_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if hidden_size is None:
        hidden_size = getattr(getattr(model, "config", None), "hidden_dim", 0)
    if num_layers is None:
        num_layers = getattr(getattr(model, "config", None), "num_layers", 0)
    hidden_size = int(hidden_size or 0)
    num_layers = int(num_layers or 0)
    if hidden_size <= 0 or num_layers <= 0:
        return 0.0
    return batch_size * seq_len * hidden_size * num_layers * activation_bytes * activation_multiplier


def _estimate_moe_dispatch_bytes(
    model: torch.nn.Module,
    mode: str,
    batch_size: int,
    seq_len: int,
    activation_bytes: int,
    top_k_override: Optional[int] = None,
    ep_size_override: Optional[int] = None,
    hidden_scale: float = 1.0,
) -> Tuple[float, float]:
    """
    Estimate MoE token dispatch/collect bytes.

    Returns:
      (intra_device_bytes, inter_device_bytes_estimate)
    """
    cfg = getattr(model, "config", None)
    if ep_size_override is None:
        ep_size = int(getattr(cfg, "ep_size", 1) or 1)
    else:
        ep_size = int(ep_size_override)
    ep_size = max(1, ep_size)

    if _is_deepseek_model(model) and cfg is not None:
        _, moe_layers = _deepseek_layer_counts(cfg)
        hidden_size = int(getattr(cfg, "hidden_size", 0) or 0)
        hidden_size = max(1, int(round(hidden_size * hidden_scale)))
        top_k = int(getattr(cfg, "num_experts_per_tok", 0) or 0)
        if top_k_override is not None:
            top_k = int(top_k_override)
        top_k = max(1, top_k)
        if mode in {"training", "prefill"}:
            tokens = batch_size * seq_len
        else:
            tokens = batch_size
        total_intra = 2.0 * float(tokens) * float(hidden_size) * float(top_k) * float(activation_bytes)
        total_intra *= float(moe_layers)
        if ep_size <= 1:
            return total_intra, 0.0
        inter_ratio = (ep_size - 1) / ep_size
        return total_intra, total_intra * inter_ratio

    total_intra = 0.0

    for _, module in model.named_modules():
        if not hasattr(module, "num_experts") or not hasattr(module, "top_k"):
            continue
        hidden_size = int(getattr(module, "hidden_size", 0) or 0)
        hidden_size = max(1, int(round(hidden_size * hidden_scale)))
        if top_k_override is None:
            top_k = int(getattr(module, "top_k", 0) or 0)
        else:
            top_k = int(top_k_override)
        if hidden_size <= 0 or top_k <= 0:
            continue

        if mode in {"training", "prefill"}:
            tokens = batch_size * seq_len
        else:
            tokens = batch_size

        # Each selected expert receives token hidden vector and sends output back.
        total_intra += 2.0 * tokens * hidden_size * top_k * activation_bytes

    if ep_size <= 1:
        return total_intra, 0.0

    inter_ratio = (ep_size - 1) / ep_size
    return total_intra, total_intra * inter_ratio


def _is_deepseek_model(model: torch.nn.Module) -> bool:
    """
    Heuristic detection for the local DeepSeek-style model implementation.

    We use duck typing to avoid importing DeepSeek classes here.
    """
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return False
    if not hasattr(model, "config"):
        return False
    try:
        first = blocks[0]
    except Exception:
        return False
    return all(hasattr(first, attr) for attr in ["attn", "ffn", "ffn_type"])


def _deepseek_layer_counts(config: object) -> Tuple[int, int]:
    """Return (dense_ffn_layers, moe_ffn_layers) from DeepSeek config knobs."""
    num_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    if num_layers <= 0:
        num_layers = int(getattr(config, "num_layers", 0) or 0)
    first_k_dense_replace = int(getattr(config, "first_k_dense_replace", 0) or 0)
    moe_layer_freq = int(getattr(config, "moe_layer_freq", 1) or 1)
    moe_layer_freq = max(1, moe_layer_freq)

    dense_layers = 0
    moe_layers = 0
    for layer_idx in range(num_layers):
        if layer_idx < first_k_dense_replace:
            dense_layers += 1
            continue
        if (layer_idx - first_k_dense_replace) % moe_layer_freq == 0:
            moe_layers += 1
        else:
            dense_layers += 1
    return dense_layers, moe_layers


def _estimate_efficiency_deepseek_fast(
    model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    activation_bytes: int,
    kv_cache_bytes: int,
    param_bytes_assumed: int,
    roofline: RooflineConfig,
    interconnect_bw_gbps: float,
    training_flops_multiplier: float,
    training_bytes_multiplier: float,
    exec_model: ExecutionModelConfig,
    tc_cfg: TensorCoreModelConfig,
    top_k_override: Optional[int],
    hidden_scale: float,
    kv_rank_scale: float,
    ep_size: int,
    requested_modes: set,
    module_weight_bytes_cache: Optional[Dict[int, float]],
) -> Dict[str, List[EfficiencyEntry]]:
    """
    DeepSeek-specific fast path for `_estimate_efficiency`.

    The DeepSeek model contains many repeated expert modules. Iterating over every expert for every
    sensitivity point is prohibitively slow and adds little modeling value, since shapes repeat.

    We instead:
    - pick representative MLA attention and FFN blocks
    - compute FLOPs/bytes for those blocks once
    - scale by layer counts and top-k routing
    """
    exec_model.validate()
    tc_cfg.validate()
    hidden_scale = max(0.1, float(hidden_scale))
    kv_rank_scale = max(0.1, float(kv_rank_scale))

    entries: Dict[str, List[EfficiencyEntry]] = {
        "training": [],
        "prefill": [],
        "decode": [],
    }
    need_prefill = ("prefill" in requested_modes) or ("training" in requested_modes)
    need_decode = "decode" in requested_modes

    cfg = getattr(model, "config", None)
    if cfg is None:
        return entries

    num_layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
    if num_layers <= 0:
        num_layers = int(getattr(cfg, "num_layers", 0) or 0)
    dense_layers, moe_layers = _deepseek_layer_counts(cfg)

    hidden = max(1, int(round(int(getattr(cfg, "hidden_size", 0) or 0) * hidden_scale)))
    num_heads = int(getattr(cfg, "num_attention_heads", 1) or 1)
    vocab_size = int(getattr(cfg, "vocab_size", 0) or 0)
    dense_intermediate = int(getattr(cfg, "intermediate_size", 0) or 0)
    moe_intermediate = int(getattr(cfg, "moe_intermediate_size", 0) or 0)

    dense_intermediate = max(1, int(round(dense_intermediate * hidden_scale)))
    moe_intermediate = max(1, int(round(moe_intermediate * hidden_scale)))

    n_routed_experts = int(getattr(cfg, "n_routed_experts", 0) or 0)
    n_shared_experts = int(getattr(cfg, "n_shared_experts", 0) or 0)
    top_k = int(getattr(cfg, "num_experts_per_tok", 0) or 0)
    if top_k_override is not None:
        top_k = int(top_k_override)
    top_k = max(1, min(max(1, n_routed_experts), top_k))

    ep_size = max(1, int(ep_size))
    if n_routed_experts > 0:
        ep_size = min(ep_size, n_routed_experts)

    # Representative modules.
    blocks = getattr(model, "blocks", [])
    if not blocks:
        return entries
    rep_attn = getattr(blocks[0], "attn", None)
    if rep_attn is None:
        return entries

    # Cache a few weight sizes (small number of modules).
    def module_weight_bytes(module: torch.nn.Module) -> float:
        module_id = id(module)
        if module_weight_bytes_cache is not None and module_id in module_weight_bytes_cache:
            return module_weight_bytes_cache[module_id]
        bytes_value = float(sum(p.numel() for p in module.parameters()) * param_bytes_assumed)
        if module_weight_bytes_cache is not None:
            module_weight_bytes_cache[module_id] = bytes_value
        return bytes_value

    mem_bw_bytes = roofline.mem_bw_gbps * 1e9

    def add_entry(
        mode: str,
        name: str,
        kind: str,
        flops: float,
        byte_breakdown: ByteBreakdown,
        bytes_net: float,
        tc_m_dim: Optional[int] = None,
    ) -> None:
        flop_breakdown = _build_flop_breakdown(
            flops_theory=flops,
            kind=kind,
            batch_size=batch_size,
            roofline=roofline,
            tc_cfg=tc_cfg,
            tc_m_dim=tc_m_dim,
        )
        bytes_hbm = byte_breakdown.hbm_total
        bytes_total = bytes_hbm + bytes_net
        intensity_weights_only = _safe_div(flop_breakdown.theory, byte_breakdown.weights)
        intensity_hbm = _safe_div(flop_breakdown.theory, bytes_hbm)
        intensity_total = _safe_div(flop_breakdown.theory, bytes_total)
        roofline_tflops_hbm = min(
            roofline.peak_tflops,
            (mem_bw_bytes * intensity_hbm) / 1e12,
        )
        regime = _regime_label(
            flops=flop_breakdown.realizable,
            bytes_hbm=bytes_hbm,
            bytes_net=bytes_net,
            roofline=roofline,
            interconnect_bw_gbps=interconnect_bw_gbps,
        )
        entries[mode].append(
            EfficiencyEntry(
                name=name,
                kind=kind,
                flops=flop_breakdown.theory,
                flops_theory=flop_breakdown.theory,
                flops_tensorcore=flop_breakdown.tensorcore,
                flops_realizable=flop_breakdown.realizable,
                bytes_hbm=bytes_hbm,
                bytes_net=bytes_net,
                bytes_total=bytes_total,
                bytes_weights=byte_breakdown.weights,
                bytes_activations=byte_breakdown.activations,
                bytes_kv=byte_breakdown.kv,
                bytes_temporary=byte_breakdown.temporary,
                arithmetic_intensity_weights_only=intensity_weights_only,
                arithmetic_intensity_hbm=intensity_hbm,
                arithmetic_intensity_total=intensity_total,
                p_effective_tflops=flop_breakdown.p_effective_tflops,
                eta_tc=flop_breakdown.eta_tc,
                regime=regime,
                roofline_tflops_hbm=roofline_tflops_hbm,
            )
        )

    def _scale_train(b: ByteBreakdown) -> ByteBreakdown:
        return ByteBreakdown(
            weights=b.weights * training_bytes_multiplier,
            activations=b.activations * training_bytes_multiplier,
            kv=b.kv * training_bytes_multiplier,
            temporary=b.temporary * training_bytes_multiplier,
        )

    # --- Attention (all layers) ---
    kv_elems_per_tok = _infer_kv_cache_elements_per_token(rep_attn, hidden)
    kv_elems_per_tok = max(1, int(round(kv_elems_per_tok * kv_rank_scale)))

    attn_weight_bytes = module_weight_bytes(rep_attn)
    attn_weight_bytes *= hidden_scale * max(hidden_scale, kv_rank_scale)
    attn_weight_bytes /= exec_model.weight_residency_attn

    scale_attn = hidden_scale * max(hidden_scale, kv_rank_scale)
    if _is_mla_attention(rep_attn):
        prefill_flops_layer = _compute_attention_flops_prefill_mla(batch_size, seq_len, rep_attn) * scale_attn
        decode_flops_layer = _compute_attention_flops_decode_mla(batch_size, seq_len, rep_attn) * scale_attn
    else:
        prefill_flops_layer = _compute_attention_flops_prefill(batch_size, seq_len, hidden, num_heads)
        decode_flops_layer = _compute_attention_flops_decode(batch_size, seq_len, hidden, num_heads)

    if exec_model.attention_bytes_model == "flash":
        prefill_bytes_layer = _compute_attention_byte_breakdown_prefill_flash(
            batch=batch_size,
            seq_len=seq_len,
            hidden=hidden,
            num_heads=num_heads,
            activation_bytes=activation_bytes,
            weight_bytes=attn_weight_bytes,
            kv_cache_elements_per_token=kv_elems_per_tok,
            kv_cache_bytes=kv_cache_bytes,
            activation_fusion_factor=exec_model.activation_fusion_factor,
            elementwise_bytes_factor=exec_model.elementwise_bytes_factor,
        )
        decode_bytes_layer = _compute_attention_byte_breakdown_decode_flash(
            batch=batch_size,
            cache_len=seq_len,
            hidden=hidden,
            num_heads=num_heads,
            activation_bytes=activation_bytes,
            weight_bytes=attn_weight_bytes,
            kv_cache_elements_per_token=kv_elems_per_tok,
            kv_cache_bytes=kv_cache_bytes,
            activation_fusion_factor=exec_model.activation_fusion_factor,
            elementwise_bytes_factor=exec_model.elementwise_bytes_factor,
        )
    else:
        prefill_bytes_layer = _compute_attention_byte_breakdown_prefill_naive(
            batch=batch_size,
            seq_len=seq_len,
            hidden=hidden,
            num_heads=num_heads,
            activation_bytes=activation_bytes,
            weight_bytes=attn_weight_bytes,
            kv_cache_elements_per_token=kv_elems_per_tok,
            kv_cache_bytes=kv_cache_bytes,
            activation_fusion_factor=exec_model.activation_fusion_factor,
            elementwise_bytes_factor=exec_model.elementwise_bytes_factor,
        )
        decode_bytes_layer = _compute_attention_byte_breakdown_decode_naive(
            batch=batch_size,
            cache_len=seq_len,
            hidden=hidden,
            num_heads=num_heads,
            activation_bytes=activation_bytes,
            weight_bytes=attn_weight_bytes,
            kv_cache_elements_per_token=kv_elems_per_tok,
            kv_cache_bytes=kv_cache_bytes,
            activation_fusion_factor=exec_model.activation_fusion_factor,
            elementwise_bytes_factor=exec_model.elementwise_bytes_factor,
        )

    if need_decode and num_layers > 0:
        add_entry(
            "decode",
            "blocks.*.attn",
            "attention",
            decode_flops_layer * num_layers,
            ByteBreakdown(
                weights=decode_bytes_layer.weights * num_layers,
                activations=decode_bytes_layer.activations * num_layers,
                kv=decode_bytes_layer.kv * num_layers,
                temporary=decode_bytes_layer.temporary * num_layers,
            ),
            0.0,
            tc_m_dim=batch_size,
        )

    if need_prefill and num_layers > 0:
        prefill_breakdown = ByteBreakdown(
            weights=prefill_bytes_layer.weights * num_layers,
            activations=prefill_bytes_layer.activations * num_layers,
            kv=prefill_bytes_layer.kv * num_layers,
            temporary=prefill_bytes_layer.temporary * num_layers,
        )
        add_entry(
            "prefill",
            "blocks.*.attn",
            "attention",
            prefill_flops_layer * num_layers,
            prefill_breakdown,
            0.0,
            tc_m_dim=batch_size * seq_len,
        )
        if "training" in requested_modes:
            add_entry(
                "training",
                "blocks.*.attn",
                "attention",
                prefill_flops_layer * num_layers * training_flops_multiplier,
                _scale_train(prefill_breakdown),
                0.0,
                tc_m_dim=batch_size * seq_len,
            )

    # --- Embedding + LM head (prefill; training derived from prefill) ---
    def add_embedding_prefill_and_train(tokens: int) -> None:
        if vocab_size <= 0:
            return
        weight = float(vocab_size * hidden * param_bytes_assumed) / exec_model.weight_residency_dense
        flops = float(tokens * hidden)
        b = ByteBreakdown(
            weights=weight,
            activations=float(tokens * hidden * activation_bytes * exec_model.activation_fusion_factor),
            kv=0.0,
            temporary=0.0,
        )
        add_entry("prefill", "token_embeddings", "embedding", flops, b, 0.0, tc_m_dim=tokens)
        if "training" in requested_modes:
            add_entry(
                "training",
                "token_embeddings",
                "embedding",
                flops * training_flops_multiplier,
                _scale_train(b),
                0.0,
                tc_m_dim=tokens,
            )

    def add_lm_head_prefill_and_train(tokens: int) -> None:
        if vocab_size <= 0:
            return
        weight = float(hidden * vocab_size * param_bytes_assumed) / exec_model.weight_residency_dense
        flops = 2.0 * float(tokens) * float(hidden) * float(vocab_size)
        act = float(tokens) * float(exec_model.activation_fusion_factor) * float(activation_bytes) * (
            float(hidden) + float(vocab_size)
        )
        b = ByteBreakdown(weights=weight, activations=act, kv=0.0, temporary=0.0)
        add_entry("prefill", "lm_head", "linear", flops, b, 0.0, tc_m_dim=tokens)
        if "training" in requested_modes:
            add_entry(
                "training",
                "lm_head",
                "linear",
                flops * training_flops_multiplier,
                _scale_train(b),
                0.0,
                tc_m_dim=tokens,
            )

    def add_embedding_decode(tokens: int) -> None:
        if vocab_size <= 0:
            return
        weight = float(vocab_size * hidden * param_bytes_assumed) / exec_model.weight_residency_dense
        flops = float(tokens * hidden)
        b = ByteBreakdown(
            weights=weight,
            activations=float(tokens * hidden * activation_bytes * exec_model.activation_fusion_factor),
            kv=0.0,
            temporary=0.0,
        )
        add_entry("decode", "token_embeddings", "embedding", flops, b, 0.0, tc_m_dim=tokens)

    def add_lm_head_decode(tokens: int) -> None:
        if vocab_size <= 0:
            return
        weight = float(hidden * vocab_size * param_bytes_assumed) / exec_model.weight_residency_dense
        flops = 2.0 * float(tokens) * float(hidden) * float(vocab_size)
        act = float(tokens) * float(exec_model.activation_fusion_factor) * float(activation_bytes) * (
            float(hidden) + float(vocab_size)
        )
        b = ByteBreakdown(weights=weight, activations=act, kv=0.0, temporary=0.0)
        add_entry("decode", "lm_head", "linear", flops, b, 0.0, tc_m_dim=tokens)

    if need_decode:
        add_embedding_decode(tokens=batch_size)
        add_lm_head_decode(tokens=batch_size)

    if need_prefill:
        tokens_prefill = batch_size * seq_len
        add_embedding_prefill_and_train(tokens=tokens_prefill)
        add_lm_head_prefill_and_train(tokens=tokens_prefill)

    # --- FFN: dense and MoE ---
    def ffn_prefill_breakdown(tokens: int, intermediate: int, experts_per_tok: int, residency: float) -> Tuple[float, ByteBreakdown]:
        # One SwiGLU-style MLP has three weight matrices:
        #   gate: H->d, up: H->d, down: d->H  => (2*H*d + d*H) = 3*H*d parameters.
        #
        # For routed MoE we model *compute/activation* cost as scaling with `experts_per_tok`,
        # but we keep `weights` as the bytes for **one** expert MLP so callers can scale it by
        # the expected number of *distinct* experts activated in the microbatch.
        weight = (3.0 * float(hidden) * float(intermediate) * float(param_bytes_assumed)) / max(1e-12, residency)
        flops = 6.0 * float(tokens) * float(hidden) * float(intermediate) * float(experts_per_tok)
        act = float(tokens) * float(experts_per_tok) * float(activation_bytes) * float(exec_model.activation_fusion_factor) * (
            3.0 * float(hidden) + 3.0 * float(intermediate)
        )
        return flops, ByteBreakdown(weights=weight, activations=act, kv=0.0, temporary=0.0)

    def add_dense_ffn(mode: str, tokens: int, layers: int) -> None:
        if layers <= 0:
            return
        flops, b = ffn_prefill_breakdown(tokens, dense_intermediate, experts_per_tok=1, residency=exec_model.weight_residency_dense)
        flops *= float(layers)
        b = ByteBreakdown(
            weights=b.weights * layers,
            activations=b.activations * layers,
            kv=0.0,
            temporary=0.0,
        )
        add_entry(mode, "blocks.*.ffn.dense", "linear", flops, b, 0.0, tc_m_dim=tokens)

    def add_moe_ffn(mode: str, tokens: int, layers: int) -> None:
        if layers <= 0 or n_routed_experts <= 0:
            return
        # Router: H -> E
        router_weight = (float(hidden) * float(n_routed_experts) * float(param_bytes_assumed)) / max(1e-12, exec_model.weight_residency_moe)
        router_flops = 2.0 * float(tokens) * float(hidden) * float(n_routed_experts)
        router_act = float(tokens) * float(activation_bytes) * float(exec_model.activation_fusion_factor) * (
            float(hidden) + float(n_routed_experts)
        )

        # Routed experts: weight traffic depends on how many *distinct* experts we touch in this
        # microbatch (not on `top_k` directly). Under uniform routing, with `n = tokens*top_k`
        # assignments into `E` experts, the expected active expert count is:
        #   E_active ~= E * (1 - exp(-n/E))
        #
        # This is per-rank: EP shards experts (E_per_rank = E/EP) but keeps per-rank tokens fixed.
        ep = max(1, int(ep_size))
        experts_per_rank = max(1, int(math.ceil(float(n_routed_experts) / float(ep))))
        assignments = float(tokens) * float(top_k)
        expected_assign_per_expert = assignments / float(experts_per_rank)
        p_active = 0.0 if expected_assign_per_expert <= 0 else 1.0 - math.exp(-expected_assign_per_expert)
        expected_unique_experts = float(experts_per_rank) * p_active
        tokens_per_active_expert = (
            0.0 if p_active <= 1e-12 else expected_assign_per_expert / p_active
        )

        routed_flops = 6.0 * float(tokens) * float(hidden) * float(moe_intermediate) * float(top_k)
        shared_flops = 6.0 * float(tokens) * float(hidden) * float(moe_intermediate) * float(max(0, n_shared_experts))
        expert_flops = routed_flops + shared_flops

        # One expert MLP's weights/activations; scale weights by expected distinct experts.
        experts_per_tok = top_k + max(0, n_shared_experts)
        _, expert_b = ffn_prefill_breakdown(
            tokens,
            moe_intermediate,
            experts_per_tok=experts_per_tok,
            residency=exec_model.weight_residency_moe,
        )
        routed_weight = expert_b.weights * expected_unique_experts
        shared_weight = expert_b.weights * float(max(0, n_shared_experts))

        weight = (router_weight + routed_weight + shared_weight) * float(layers)
        act = (router_act + expert_b.activations) * float(layers)
        flops = (router_flops + expert_flops) * float(layers)

        # Tensor-core utilization for MoE is governed by per-expert token counts, not tokens.
        # We approximate the "effective M" as a FLOP-weighted mixture of routed (per-active-expert)
        # and shared (dense-on-tokens) paths.
        total_expert_flops = max(1e-12, expert_flops)
        m_eff = (
            (routed_flops * tokens_per_active_expert) +
            (shared_flops * float(tokens))
        ) / total_expert_flops
        tc_m = max(1, int(round(m_eff)))
        add_entry(
            mode,
            "blocks.*.ffn.moe",
            "linear",
            flops,
            ByteBreakdown(weights=weight, activations=act, kv=0.0, temporary=0.0),
            0.0,
            tc_m_dim=tc_m,
        )

    if need_decode:
        add_dense_ffn("decode", tokens=batch_size, layers=dense_layers)
        add_moe_ffn("decode", tokens=batch_size, layers=moe_layers)

    if need_prefill:
        tokens_prefill = batch_size * seq_len
        add_dense_ffn("prefill", tokens=tokens_prefill, layers=dense_layers)
        add_moe_ffn("prefill", tokens=tokens_prefill, layers=moe_layers)
        if "training" in requested_modes:
            # Training is modeled as prefill * multipliers.
            flops_dense, b_dense = ffn_prefill_breakdown(tokens_prefill, dense_intermediate, 1, exec_model.weight_residency_dense)
            flops_dense *= float(dense_layers) * training_flops_multiplier
            b_dense = _scale_train(
                ByteBreakdown(
                    weights=b_dense.weights * dense_layers,
                    activations=b_dense.activations * dense_layers,
                    kv=0.0,
                    temporary=0.0,
                )
            )
            add_entry(
                "training",
                "blocks.*.ffn.dense",
                "linear",
                flops_dense,
                b_dense,
                0.0,
                tc_m_dim=tokens_prefill,
            )

            # MoE training.
            if moe_layers > 0 and n_routed_experts > 0:
                router_weight = (float(hidden) * float(n_routed_experts) * float(param_bytes_assumed)) / max(1e-12, exec_model.weight_residency_moe)
                router_flops = 2.0 * float(tokens_prefill) * float(hidden) * float(n_routed_experts)
                router_act = float(tokens_prefill) * float(activation_bytes) * float(exec_model.activation_fusion_factor) * (
                    float(hidden) + float(n_routed_experts)
                )
                experts_per_tok = top_k + max(0, n_shared_experts)
                # Expected distinct expert activation for this microbatch (uniform routing model).
                ep = max(1, int(ep_size))
                experts_per_rank = max(1, int(math.ceil(float(n_routed_experts) / float(ep))))
                assignments = float(tokens_prefill) * float(top_k)
                expected_assign_per_expert = assignments / float(experts_per_rank)
                p_active = 0.0 if expected_assign_per_expert <= 0 else 1.0 - math.exp(-expected_assign_per_expert)
                expected_unique_experts = float(experts_per_rank) * p_active
                tokens_per_active_expert = (
                    0.0 if p_active <= 1e-12 else expected_assign_per_expert / p_active
                )

                routed_flops = 6.0 * float(tokens_prefill) * float(hidden) * float(moe_intermediate) * float(top_k)
                shared_flops = 6.0 * float(tokens_prefill) * float(hidden) * float(moe_intermediate) * float(max(0, n_shared_experts))
                expert_flops = routed_flops + shared_flops

                _, expert_b = ffn_prefill_breakdown(
                    tokens_prefill,
                    moe_intermediate,
                    experts_per_tok,
                    exec_model.weight_residency_moe,
                )
                routed_weight = expert_b.weights * expected_unique_experts
                shared_weight = expert_b.weights * float(max(0, n_shared_experts))

                flops_moe = (router_flops + expert_flops) * float(moe_layers) * training_flops_multiplier
                b_moe = _scale_train(
                    ByteBreakdown(
                        weights=(router_weight + routed_weight + shared_weight) * float(moe_layers),
                        activations=(router_act + expert_b.activations) * float(moe_layers),
                        kv=0.0,
                        temporary=0.0,
                    )
                )
                total_expert_flops = max(1e-12, expert_flops)
                m_eff = (
                    (routed_flops * tokens_per_active_expert) +
                    (shared_flops * float(tokens_prefill))
                ) / total_expert_flops
                tc_m = max(1, int(round(m_eff)))
                add_entry(
                    "training",
                    "blocks.*.ffn.moe",
                    "linear",
                    flops_moe,
                    b_moe,
                    0.0,
                    tc_m_dim=tc_m,
                )

    # --- Network dispatch (only matters when EP>1) ---
    if ep_size > 1 and moe_layers > 0:
        inter_ratio = (ep_size - 1) / ep_size
        for mode in ["training", "prefill", "decode"]:
            if mode not in requested_modes:
                continue
            tokens = batch_size * seq_len if mode in {"training", "prefill"} else batch_size
            per_layer_intra = 2.0 * float(tokens) * float(hidden) * float(top_k) * float(activation_bytes)
            total_intra = per_layer_intra * float(moe_layers)
            inter_bytes = total_intra * float(inter_ratio)
            add_entry(
                mode=mode,
                name="moe.dispatch.interconnect",
                kind="network",
                flops=0.0,
                byte_breakdown=ByteBreakdown(0.0, 0.0, 0.0, 0.0),
                bytes_net=inter_bytes,
            )

    return entries


def _estimate_mode_time_seconds(
    total_flops: float,
    total_hbm_bytes: float,
    total_net_bytes: float,
    roofline: RooflineConfig,
    interconnect_bw_gbps: float,
) -> Tuple[float, float, float, float]:
    """Return (compute_time_s, hbm_time_s, net_time_s, estimated_time_s)."""
    compute_time = _safe_div(total_flops, roofline.peak_tflops * 1e12)
    hbm_time = _safe_div(total_hbm_bytes, roofline.mem_bw_gbps * 1e9)
    net_time = _safe_div(total_net_bytes, interconnect_bw_gbps * 1e9)
    return compute_time, hbm_time, net_time, max(compute_time, hbm_time, net_time)


def _rate_mode_confidence(mode: str) -> Tuple[str, str]:
    """Heuristic confidence level and rationale per mode."""
    if mode == "decode":
        return "Medium", "KV cache traffic modeled explicitly; kernel/runtime overlap still simplified."
    if mode == "prefill":
        return "Medium", "Core FLOPs/bytes modeled; fusion/cache residency effects not modeled."
    return "Low-Medium", "Training uses coarse backward multipliers and omits optimizer/comm overlap details."


def _compute_linear_flops(batch: int, seq_len: int, in_f: int, out_f: int) -> float:
    return 2.0 * batch * seq_len * in_f * out_f


def _compute_attention_flops_prefill(
    batch: int,
    seq_len: int,
    hidden: int,
    num_heads: int,
) -> float:
    head_dim = hidden // num_heads
    qkv = 6.0 * batch * seq_len * hidden * hidden
    scores = 2.0 * batch * seq_len * seq_len * hidden
    softmax = 1.0 * batch * num_heads * seq_len * seq_len
    attn = 2.0 * batch * seq_len * seq_len * hidden
    out = 2.0 * batch * seq_len * hidden * hidden
    return qkv + scores + softmax + attn + out


def _compute_attention_flops_decode(
    batch: int,
    cache_len: int,
    hidden: int,
    num_heads: int,
) -> float:
    head_dim = hidden // num_heads
    qkv = 6.0 * batch * 1 * hidden * hidden
    scores = 2.0 * batch * cache_len * hidden
    softmax = 1.0 * batch * num_heads * cache_len
    attn = 2.0 * batch * cache_len * hidden
    out = 2.0 * batch * 1 * hidden * hidden
    return qkv + scores + softmax + attn + out

def _is_mla_attention(module: torch.nn.Module) -> bool:
    """Heuristic detection for DeepSeek-style Multi-Head Latent Attention (MLA)."""
    return (
        hasattr(module, "q_a_proj")
        and hasattr(module, "q_b_proj")
        and hasattr(module, "kv_a_proj")
        and hasattr(module, "kv_b_proj")
        and hasattr(module, "qk_nope_head_dim")
        and hasattr(module, "qk_rope_head_dim")
        and hasattr(module, "v_head_dim")
        and hasattr(module, "num_heads")
        and hasattr(module, "hidden_size")
    )


def _mla_dims(module: torch.nn.Module) -> Tuple[int, int, int, int, int, int, int]:
    """Return (H, h, r_q, r_kv, d_nope, d_rope, d_v)."""
    H = int(getattr(module, "hidden_size"))
    h = int(getattr(module, "num_heads"))
    d_nope = int(getattr(module, "qk_nope_head_dim"))
    d_rope = int(getattr(module, "qk_rope_head_dim"))
    d_v = int(getattr(module, "v_head_dim"))
    # q_lora_rank inferred from q_a_proj output size
    r_q = int(getattr(getattr(module, "q_a_proj"), "out_features"))
    # kv_lora_rank inferred from kv_a_norm weight size if present; else kv_a_proj minus rope dim
    if hasattr(module, "kv_a_norm") and hasattr(getattr(module, "kv_a_norm"), "weight"):
        r_kv = int(module.kv_a_norm.weight.numel())
    else:
        r_kv = int(getattr(getattr(module, "kv_a_proj"), "out_features")) - d_rope
    r_kv = max(1, int(r_kv))
    return H, h, r_q, r_kv, d_nope, d_rope, d_v


def _compute_attention_flops_prefill_mla(batch: int, seq_len: int, module: torch.nn.Module) -> float:
    """MLA prefill FLOPs (forward) using module-provided dims."""
    H, h, r_q, r_kv, d_nope, d_rope, d_v = _mla_dims(module)
    d_q = d_nope + d_rope
    # Projections
    proj = (
        2.0 * batch * seq_len * H * r_q
        + 2.0 * batch * seq_len * r_q * (h * d_q)
        + 2.0 * batch * seq_len * H * (r_kv + d_rope)
        + 2.0 * batch * seq_len * r_kv * (h * (d_nope + d_v))
        + 2.0 * batch * seq_len * (h * d_v) * H
    )
    # Attention math
    scores = 2.0 * batch * h * seq_len * seq_len * d_q
    softmax = 1.0 * batch * h * seq_len * seq_len
    pv = 2.0 * batch * h * seq_len * seq_len * d_v
    return proj + scores + softmax + pv


def _compute_attention_flops_decode_mla(batch: int, cache_len: int, module: torch.nn.Module) -> float:
    """MLA decode FLOPs (forward) for one-step decode with KV length cache_len."""
    H, h, r_q, r_kv, d_nope, d_rope, d_v = _mla_dims(module)
    d_q = d_nope + d_rope
    # Projections (seq_len=1)
    proj = (
        2.0 * batch * 1 * H * r_q
        + 2.0 * batch * 1 * r_q * (h * d_q)
        + 2.0 * batch * 1 * H * (r_kv + d_rope)
        + 2.0 * batch * 1 * r_kv * (h * (d_nope + d_v))
        + 2.0 * batch * 1 * (h * d_v) * H
    )
    # Attention math (QK: 1 x cache_len)
    scores = 2.0 * batch * h * 1 * cache_len * d_q
    softmax = 1.0 * batch * h * cache_len
    pv = 2.0 * batch * h * 1 * cache_len * d_v
    return proj + scores + softmax + pv


def _compute_attention_byte_breakdown_prefill_naive(
    batch: int,
    seq_len: int,
    hidden: int,
    num_heads: int,
    activation_bytes: int,
    weight_bytes: float,
    kv_cache_elements_per_token: int,
    kv_cache_bytes: int,
    activation_fusion_factor: float,
    elementwise_bytes_factor: float,
) -> ByteBreakdown:
    """HBM byte decomposition for naive prefill attention."""
    attn_matrix = batch * num_heads * seq_len * seq_len
    core_activations = (
        batch * seq_len * hidden +
        3 * batch * seq_len * hidden +
        batch * seq_len * hidden
    ) * activation_fusion_factor * activation_bytes
    temporary = 2.0 * attn_matrix * elementwise_bytes_factor * activation_bytes
    kv_cache_write = batch * seq_len * kv_cache_elements_per_token * kv_cache_bytes
    return ByteBreakdown(
        weights=float(weight_bytes),
        activations=float(core_activations),
        kv=float(kv_cache_write),
        temporary=float(temporary),
    )


def _compute_attention_byte_breakdown_decode_naive(
    batch: int,
    cache_len: int,
    hidden: int,
    num_heads: int,
    activation_bytes: int,
    weight_bytes: float,
    kv_cache_elements_per_token: int,
    kv_cache_bytes: int,
    activation_fusion_factor: float,
    elementwise_bytes_factor: float,
) -> ByteBreakdown:
    """HBM byte decomposition for naive decode attention."""
    attn_vector = batch * num_heads * cache_len
    core_activations = (
        batch * hidden +
        3 * batch * hidden +
        batch * hidden +
        2 * batch * hidden
    ) * activation_fusion_factor * activation_bytes
    temporary = 2.0 * attn_vector * elementwise_bytes_factor * activation_bytes
    kv_read = batch * cache_len * kv_cache_elements_per_token * kv_cache_bytes
    kv_write = batch * kv_cache_elements_per_token * kv_cache_bytes
    return ByteBreakdown(
        weights=float(weight_bytes),
        activations=float(core_activations),
        kv=float(kv_read + kv_write),
        temporary=float(temporary),
    )


def _compute_attention_byte_breakdown_prefill_flash(
    batch: int,
    seq_len: int,
    hidden: int,
    num_heads: int,
    activation_bytes: int,
    weight_bytes: float,
    kv_cache_elements_per_token: int,
    kv_cache_bytes: int,
    activation_fusion_factor: float,
    elementwise_bytes_factor: float,
) -> ByteBreakdown:
    """HBM byte decomposition for flash-style prefill attention."""
    core_activations = (
        batch * seq_len * hidden +
        3 * batch * seq_len * hidden +
        batch * seq_len * hidden
    ) * activation_fusion_factor * activation_bytes
    temporary = batch * num_heads * seq_len * elementwise_bytes_factor * activation_bytes
    kv_cache_write = batch * seq_len * kv_cache_elements_per_token * kv_cache_bytes
    return ByteBreakdown(
        weights=float(weight_bytes),
        activations=float(core_activations),
        kv=float(kv_cache_write),
        temporary=float(temporary),
    )


def _compute_attention_byte_breakdown_decode_flash(
    batch: int,
    cache_len: int,
    hidden: int,
    num_heads: int,
    activation_bytes: int,
    weight_bytes: float,
    kv_cache_elements_per_token: int,
    kv_cache_bytes: int,
    activation_fusion_factor: float,
    elementwise_bytes_factor: float,
) -> ByteBreakdown:
    """HBM byte decomposition for flash-style decode attention."""
    core_activations = (
        batch * hidden +
        3 * batch * hidden +
        batch * hidden +
        2 * batch * hidden
    ) * activation_fusion_factor * activation_bytes
    temporary = batch * num_heads * elementwise_bytes_factor * activation_bytes
    kv_read = batch * cache_len * kv_cache_elements_per_token * kv_cache_bytes
    kv_write = batch * kv_cache_elements_per_token * kv_cache_bytes
    return ByteBreakdown(
        weights=float(weight_bytes),
        activations=float(core_activations),
        kv=float(kv_read + kv_write),
        temporary=float(temporary),
    )


def _compute_attention_bytes_prefill_naive(
    batch: int,
    seq_len: int,
    hidden: int,
    num_heads: int,
    activation_bytes: int,
    weight_bytes: int,
    kv_cache_elements_per_token: int,
    kv_cache_bytes: int,
    activation_fusion_factor: float,
    elementwise_bytes_factor: float,
) -> float:
    breakdown = _compute_attention_byte_breakdown_prefill_naive(
        batch=batch,
        seq_len=seq_len,
        hidden=hidden,
        num_heads=num_heads,
        activation_bytes=activation_bytes,
        weight_bytes=weight_bytes,
        kv_cache_elements_per_token=kv_cache_elements_per_token,
        kv_cache_bytes=kv_cache_bytes,
        activation_fusion_factor=activation_fusion_factor,
        elementwise_bytes_factor=elementwise_bytes_factor,
    )
    return breakdown.hbm_total


def _compute_attention_bytes_decode_naive(
    batch: int,
    cache_len: int,
    hidden: int,
    num_heads: int,
    activation_bytes: int,
    weight_bytes: int,
    kv_cache_elements_per_token: int,
    kv_cache_bytes: int,
    activation_fusion_factor: float,
    elementwise_bytes_factor: float,
) -> float:
    breakdown = _compute_attention_byte_breakdown_decode_naive(
        batch=batch,
        cache_len=cache_len,
        hidden=hidden,
        num_heads=num_heads,
        activation_bytes=activation_bytes,
        weight_bytes=weight_bytes,
        kv_cache_elements_per_token=kv_cache_elements_per_token,
        kv_cache_bytes=kv_cache_bytes,
        activation_fusion_factor=activation_fusion_factor,
        elementwise_bytes_factor=elementwise_bytes_factor,
    )
    return breakdown.hbm_total


def _compute_attention_bytes_prefill_flash(
    batch: int,
    seq_len: int,
    hidden: int,
    num_heads: int,
    activation_bytes: int,
    weight_bytes: int,
    kv_cache_elements_per_token: int,
    kv_cache_bytes: int,
    activation_fusion_factor: float,
    elementwise_bytes_factor: float,
) -> float:
    """
    Flash-style prefill byte model.

    This avoids storing [B, heads, S, S] score/probability tensors in HBM.
    KV-cache write remains explicit.
    """
    breakdown = _compute_attention_byte_breakdown_prefill_flash(
        batch=batch,
        seq_len=seq_len,
        hidden=hidden,
        num_heads=num_heads,
        activation_bytes=activation_bytes,
        weight_bytes=weight_bytes,
        kv_cache_elements_per_token=kv_cache_elements_per_token,
        kv_cache_bytes=kv_cache_bytes,
        activation_fusion_factor=activation_fusion_factor,
        elementwise_bytes_factor=elementwise_bytes_factor,
    )
    return breakdown.hbm_total


def _compute_attention_bytes_decode_flash(
    batch: int,
    cache_len: int,
    hidden: int,
    num_heads: int,
    activation_bytes: int,
    weight_bytes: int,
    kv_cache_elements_per_token: int,
    kv_cache_bytes: int,
    activation_fusion_factor: float,
    elementwise_bytes_factor: float,
) -> float:
    """
    Flash-style decode byte model.

    Per-step score/probability vectors are not materialized in HBM, but KV read/write
    remains explicit and usually dominates decode bytes.
    """
    breakdown = _compute_attention_byte_breakdown_decode_flash(
        batch=batch,
        cache_len=cache_len,
        hidden=hidden,
        num_heads=num_heads,
        activation_bytes=activation_bytes,
        weight_bytes=weight_bytes,
        kv_cache_elements_per_token=kv_cache_elements_per_token,
        kv_cache_bytes=kv_cache_bytes,
        activation_fusion_factor=activation_fusion_factor,
        elementwise_bytes_factor=elementwise_bytes_factor,
    )
    return breakdown.hbm_total


def _infer_kv_cache_elements_per_token(module: torch.nn.Module, hidden: int) -> int:
    """
    Estimate KV-cache elements per token for attention module.

    - Standard MHA/GQA approximation: K and V each hidden-sized -> 2 * hidden
    - DeepSeek MLA approximation: store compressed kv_latent + rope key:
      kv_lora_rank + qk_rope_head_dim
    """
    if hasattr(module, "kv_a_norm") and hasattr(module, "qk_rope_head_dim"):
        kv_latent_rank = int(module.kv_a_norm.weight.numel())
        qk_rope_head_dim = int(getattr(module, "qk_rope_head_dim"))
        return kv_latent_rank + qk_rope_head_dim
    return 2 * hidden


def _estimate_efficiency(
    model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    activation_bytes: int,
    kv_cache_bytes: int,
    param_bytes_assumed: int,
    roofline: RooflineConfig,
    interconnect_bw_gbps: float,
    training_flops_multiplier: float,
    training_bytes_multiplier: float,
    exec_model: ExecutionModelConfig,
    tc_cfg: TensorCoreModelConfig,
    top_k_override: Optional[int] = None,
    hidden_scale: float = 1.0,
    kv_rank_scale: float = 1.0,
    module_weight_bytes_cache: Optional[Dict[int, float]] = None,
    modes_to_estimate: Optional[Tuple[str, ...]] = None,
    attention_names_override: Optional[List[str]] = None,
    moe_active_fractions_override: Optional[Dict[str, float]] = None,
    ep_size_override: Optional[int] = None,
    tp_size_override: Optional[int] = None,
) -> Dict[str, List[EfficiencyEntry]]:
    exec_model.validate()
    tc_cfg.validate()
    hidden_scale = max(0.1, float(hidden_scale))
    kv_rank_scale = max(0.1, float(kv_rank_scale))
    if modes_to_estimate is None:
        modes_to_estimate = ("training", "prefill", "decode")
    requested_modes = set(modes_to_estimate)
    valid_modes = {"training", "prefill", "decode"}
    invalid_modes = sorted(requested_modes.difference(valid_modes))
    if invalid_modes:
        raise ValueError(
            f"Unsupported modes_to_estimate: {invalid_modes}; expected subset of {valid_modes}"
        )

    need_prefill = ("prefill" in requested_modes) or ("training" in requested_modes)
    need_decode = "decode" in requested_modes

    if _is_deepseek_model(model):
        cfg = getattr(model, "config", None)
        if ep_size_override is None:
            ep_size = int(getattr(cfg, "ep_size", 1) or 1)
        else:
            ep_size = int(ep_size_override)
        ep_size = max(1, ep_size)
        n_routed = int(getattr(cfg, "n_routed_experts", 0) or 0) if cfg is not None else 0
        if n_routed > 0:
            ep_size = min(ep_size, n_routed)
        return _estimate_efficiency_deepseek_fast(
            model=model,
            batch_size=batch_size,
            seq_len=seq_len,
            activation_bytes=activation_bytes,
            kv_cache_bytes=kv_cache_bytes,
            param_bytes_assumed=param_bytes_assumed,
            roofline=roofline,
            interconnect_bw_gbps=interconnect_bw_gbps,
            training_flops_multiplier=training_flops_multiplier,
            training_bytes_multiplier=training_bytes_multiplier,
            exec_model=exec_model,
            tc_cfg=tc_cfg,
            top_k_override=top_k_override,
            hidden_scale=hidden_scale,
            kv_rank_scale=kv_rank_scale,
            ep_size=ep_size,
            requested_modes=requested_modes,
            module_weight_bytes_cache=module_weight_bytes_cache,
        )

    if attention_names_override is None:
        attention_names = _attention_module_names(model)
    else:
        attention_names = list(attention_names_override)
    attention_name_set = set(attention_names)

    if moe_active_fractions_override is None:
        moe_active_fractions = _build_moe_active_fractions(
            model=model,
            top_k_override=top_k_override,
        )
    else:
        moe_active_fractions = dict(moe_active_fractions_override)

    entries: Dict[str, List[EfficiencyEntry]] = {
        "training": [],
        "prefill": [],
        "decode": [],
    }

    mem_bw_bytes = roofline.mem_bw_gbps * 1e9

    def module_weight_bytes(module: torch.nn.Module) -> float:
        module_id = id(module)
        if module_weight_bytes_cache is not None and module_id in module_weight_bytes_cache:
            return module_weight_bytes_cache[module_id]
        bytes_value = float(sum(p.numel() for p in module.parameters()) * param_bytes_assumed)
        if module_weight_bytes_cache is not None:
            module_weight_bytes_cache[module_id] = bytes_value
        return bytes_value

    def add_entry(
        mode: str,
        name: str,
        kind: str,
        flops: float,
        byte_breakdown: ByteBreakdown,
        bytes_net: float,
        tc_m_dim: Optional[int] = None,
    ) -> None:
        if tc_m_dim is None:
            # Default utilization proxy: dense prefill/training scales with tokens (B*S),
            # while decode has M ~= B.
            tc_m_dim = batch_size * seq_len if mode in {"training", "prefill"} else batch_size
        flop_breakdown = _build_flop_breakdown(
            flops_theory=flops,
            kind=kind,
            batch_size=batch_size,
            roofline=roofline,
            tc_cfg=tc_cfg,
            tc_m_dim=tc_m_dim,
        )
        bytes_hbm = byte_breakdown.hbm_total
        bytes_total = bytes_hbm + bytes_net
        intensity_weights_only = _safe_div(flop_breakdown.theory, byte_breakdown.weights)
        intensity_hbm = _safe_div(flop_breakdown.theory, bytes_hbm)
        intensity_total = _safe_div(flop_breakdown.theory, bytes_total)
        roofline_tflops_hbm = min(
            roofline.peak_tflops,
            (mem_bw_bytes * intensity_hbm) / 1e12,
        )
        regime = _regime_label(
            flops=flop_breakdown.realizable,
            bytes_hbm=bytes_hbm,
            bytes_net=bytes_net,
            roofline=roofline,
            interconnect_bw_gbps=interconnect_bw_gbps,
        )
        entries[mode].append(
            EfficiencyEntry(
                name=name,
                kind=kind,
                flops=flop_breakdown.theory,
                flops_theory=flop_breakdown.theory,
                flops_tensorcore=flop_breakdown.tensorcore,
                flops_realizable=flop_breakdown.realizable,
                bytes_hbm=bytes_hbm,
                bytes_net=bytes_net,
                bytes_total=bytes_total,
                bytes_weights=byte_breakdown.weights,
                bytes_activations=byte_breakdown.activations,
                bytes_kv=byte_breakdown.kv,
                bytes_temporary=byte_breakdown.temporary,
                arithmetic_intensity_weights_only=intensity_weights_only,
                arithmetic_intensity_hbm=intensity_hbm,
                arithmetic_intensity_total=intensity_total,
                p_effective_tflops=flop_breakdown.p_effective_tflops,
                eta_tc=flop_breakdown.eta_tc,
                regime=regime,
                roofline_tflops_hbm=roofline_tflops_hbm,
            )
        )

    for name, module in model.named_modules():
        if name == "":
            continue

        if name in attention_name_set:
            hidden = getattr(module, "hidden_size", None)
            if hidden is None:
                hidden = getattr(module, "num_heads", 1) * getattr(module, "head_dim", 0)
            hidden = max(1, int(round(int(hidden) * hidden_scale)))
            num_heads = int(getattr(module, "num_heads", 1))
            kv_cache_elements_per_token = _infer_kv_cache_elements_per_token(module, hidden)
            kv_cache_elements_per_token = max(
                1,
                int(round(kv_cache_elements_per_token * kv_rank_scale)),
            )

            raw_weight_bytes = module_weight_bytes(module)
            raw_weight_bytes *= hidden_scale * max(hidden_scale, kv_rank_scale)
            effective_weight_bytes = raw_weight_bytes / exec_model.weight_residency_attn

            prefill_flops = 0.0
            prefill_bytes = ByteBreakdown(0.0, 0.0, 0.0, 0.0)
            if need_prefill:
                if _is_mla_attention(module):
                    prefill_flops = _compute_attention_flops_prefill_mla(
                        batch_size,
                        seq_len,
                        module,
                    )
                    prefill_flops *= hidden_scale * max(hidden_scale, kv_rank_scale)
                else:
                    prefill_flops = _compute_attention_flops_prefill(
                        batch_size,
                        seq_len,
                        hidden,
                        num_heads,
                    )
                if exec_model.attention_bytes_model == "flash":
                    prefill_bytes = _compute_attention_byte_breakdown_prefill_flash(
                        batch=batch_size,
                        seq_len=seq_len,
                        hidden=hidden,
                        num_heads=num_heads,
                        activation_bytes=activation_bytes,
                        weight_bytes=effective_weight_bytes,
                        kv_cache_elements_per_token=kv_cache_elements_per_token,
                        kv_cache_bytes=kv_cache_bytes,
                        activation_fusion_factor=exec_model.activation_fusion_factor,
                        elementwise_bytes_factor=exec_model.elementwise_bytes_factor,
                    )
                else:
                    prefill_bytes = _compute_attention_byte_breakdown_prefill_naive(
                        batch=batch_size,
                        seq_len=seq_len,
                        hidden=hidden,
                        num_heads=num_heads,
                        activation_bytes=activation_bytes,
                        weight_bytes=effective_weight_bytes,
                        kv_cache_elements_per_token=kv_cache_elements_per_token,
                        kv_cache_bytes=kv_cache_bytes,
                        activation_fusion_factor=exec_model.activation_fusion_factor,
                        elementwise_bytes_factor=exec_model.elementwise_bytes_factor,
                    )

            if need_decode:
                if _is_mla_attention(module):
                    decode_flops = _compute_attention_flops_decode_mla(
                        batch_size,
                        seq_len,
                        module,
                    )
                    decode_flops *= hidden_scale * max(hidden_scale, kv_rank_scale)
                else:
                    decode_flops = _compute_attention_flops_decode(
                        batch_size,
                        seq_len,
                        hidden,
                        num_heads,
                    )
                if exec_model.attention_bytes_model == "flash":
                    decode_bytes = _compute_attention_byte_breakdown_decode_flash(
                        batch=batch_size,
                        cache_len=seq_len,
                        hidden=hidden,
                        num_heads=num_heads,
                        activation_bytes=activation_bytes,
                        weight_bytes=effective_weight_bytes,
                        kv_cache_elements_per_token=kv_cache_elements_per_token,
                        kv_cache_bytes=kv_cache_bytes,
                        activation_fusion_factor=exec_model.activation_fusion_factor,
                        elementwise_bytes_factor=exec_model.elementwise_bytes_factor,
                    )
                else:
                    decode_bytes = _compute_attention_byte_breakdown_decode_naive(
                        batch=batch_size,
                        cache_len=seq_len,
                        hidden=hidden,
                        num_heads=num_heads,
                        activation_bytes=activation_bytes,
                        weight_bytes=effective_weight_bytes,
                        kv_cache_elements_per_token=kv_cache_elements_per_token,
                        kv_cache_bytes=kv_cache_bytes,
                        activation_fusion_factor=exec_model.activation_fusion_factor,
                        elementwise_bytes_factor=exec_model.elementwise_bytes_factor,
                    )
                add_entry("decode", name, "attention", decode_flops, decode_bytes, 0.0)

            if "prefill" in requested_modes:
                add_entry("prefill", name, "attention", prefill_flops, prefill_bytes, 0.0)
            if "training" in requested_modes:
                add_entry(
                    "training",
                    name,
                    "attention",
                    prefill_flops * training_flops_multiplier,
                    ByteBreakdown(
                        weights=prefill_bytes.weights * training_bytes_multiplier,
                        activations=prefill_bytes.activations * training_bytes_multiplier,
                        kv=prefill_bytes.kv * training_bytes_multiplier,
                        temporary=prefill_bytes.temporary * training_bytes_multiplier,
                    ),
                    0.0,
                )
            continue

        skip = False
        for attn_name in attention_name_set:
            if name.startswith(f"{attn_name}."):
                skip = True
                break
        if skip:
            continue

        is_linear_like = (
            isinstance(module, torch.nn.Linear)
            or (
                hasattr(module, "in_features")
                and hasattr(module, "out_features")
                and hasattr(module, "weight")
            )
        )
        if is_linear_like:
            in_features = int(getattr(module, "in_features"))
            out_features = int(getattr(module, "out_features"))
            in_features_scaled = max(1, int(round(in_features * hidden_scale)))
            out_features_scaled = max(1, int(round(out_features * hidden_scale)))
            active_fraction = float(moe_active_fractions.get(name, 1.0))
            is_routed_expert = active_fraction < 0.999
            residency = exec_model.weight_residency_dense
            if (
                active_fraction < 0.999
                or ".experts." in name
                or ".shared_experts." in name
                or ".router" in name
            ):
                residency = exec_model.weight_residency_moe
            raw_weight_bytes = module_weight_bytes(module)
            denom = max(1.0, float(in_features * out_features))
            raw_weight_bytes *= (in_features_scaled * out_features_scaled) / denom
            effective_weight_bytes = raw_weight_bytes / residency
            prefill_flops = 0.0
            prefill_bytes = ByteBreakdown(0.0, 0.0, 0.0, 0.0)
            if need_prefill:
                raw_prefill_flops = _compute_linear_flops(
                    batch_size,
                    seq_len,
                    in_features_scaled,
                    out_features_scaled,
                )
                raw_prefill_activation_bytes = (
                    activation_bytes
                    * exec_model.activation_fusion_factor
                    * (
                        batch_size * seq_len * in_features_scaled
                        + batch_size * seq_len * out_features_scaled
                    )
                )
                # For routed experts, weights are read only if the expert is selected at least once
                # in the microbatch. Under a uniform routing model, tokens-per-expert is Poisson
                # with mean lambda = tokens * (top_k/num_experts). We approximate:
                #   P(expert active) ~= 1 - exp(-lambda)
                # and use that to scale expert weight traffic (while FLOPs scale with expected
                # tokens assigned = lambda).
                tokens_prefill = float(batch_size * seq_len)
                lambda_prefill = tokens_prefill * active_fraction
                p_active_prefill = (
                    0.0 if (not is_routed_expert or lambda_prefill <= 0.0)
                    else 1.0 - math.exp(-lambda_prefill)
                )
                weight_scale_prefill = 1.0 if not is_routed_expert else p_active_prefill
                tc_m_prefill = (
                    max(1, int(round(tokens_prefill)))
                    if not is_routed_expert
                    else max(1, int(round(lambda_prefill / max(1e-12, p_active_prefill))))
                )
                prefill_flops = raw_prefill_flops * active_fraction
                prefill_bytes = ByteBreakdown(
                    weights=effective_weight_bytes * weight_scale_prefill,
                    activations=raw_prefill_activation_bytes * active_fraction,
                    kv=0.0,
                    temporary=0.0,
                )
            if need_decode:
                raw_decode_flops = _compute_linear_flops(
                    batch_size,
                    1,
                    in_features_scaled,
                    out_features_scaled,
                )
                raw_decode_activation_bytes = (
                    activation_bytes
                    * exec_model.activation_fusion_factor
                    * (batch_size * in_features_scaled + batch_size * out_features_scaled)
                )
                tokens_decode = float(batch_size)
                lambda_decode = tokens_decode * active_fraction
                p_active_decode = (
                    0.0 if (not is_routed_expert or lambda_decode <= 0.0)
                    else 1.0 - math.exp(-lambda_decode)
                )
                weight_scale_decode = 1.0 if not is_routed_expert else p_active_decode
                tc_m_decode = (
                    max(1, int(round(tokens_decode)))
                    if not is_routed_expert
                    else max(1, int(round(lambda_decode / max(1e-12, p_active_decode))))
                )
                decode_flops = raw_decode_flops * active_fraction
                decode_bytes = ByteBreakdown(
                    weights=effective_weight_bytes * weight_scale_decode,
                    activations=raw_decode_activation_bytes * active_fraction,
                    kv=0.0,
                    temporary=0.0,
                )
                add_entry(
                    "decode",
                    name,
                    "linear",
                    decode_flops,
                    decode_bytes,
                    0.0,
                    tc_m_dim=tc_m_decode,
                )
            if "prefill" in requested_modes:
                add_entry(
                    "prefill",
                    name,
                    "linear",
                    prefill_flops,
                    prefill_bytes,
                    0.0,
                    tc_m_dim=tc_m_prefill if need_prefill else None,
                )
            if "training" in requested_modes:
                add_entry(
                    "training",
                    name,
                    "linear",
                    prefill_flops * training_flops_multiplier,
                    ByteBreakdown(
                        weights=prefill_bytes.weights * training_bytes_multiplier,
                        activations=prefill_bytes.activations * training_bytes_multiplier,
                        kv=0.0,
                        temporary=0.0,
                    ),
                    0.0,
                    tc_m_dim=tc_m_prefill if need_prefill else None,
                )
            continue

        is_embedding_like = (
            isinstance(module, torch.nn.Embedding)
            or (
                hasattr(module, "embedding_dim")
                and hasattr(module, "num_embeddings")
                and hasattr(module, "weight")
            )
        )
        if is_embedding_like:
            embedding_dim = int(getattr(module, "embedding_dim"))
            embedding_dim = max(1, int(round(embedding_dim * hidden_scale)))
            raw_weight_bytes = module_weight_bytes(module)
            raw_weight_bytes *= hidden_scale
            effective_weight_bytes = raw_weight_bytes / exec_model.weight_residency_dense
            prefill_flops = 0.0
            prefill_bytes = ByteBreakdown(0.0, 0.0, 0.0, 0.0)
            if need_prefill:
                prefill_flops = batch_size * seq_len * embedding_dim
                prefill_bytes = ByteBreakdown(
                    weights=effective_weight_bytes,
                    activations=activation_bytes
                    * (batch_size * seq_len * embedding_dim)
                    * exec_model.activation_fusion_factor,
                    kv=0.0,
                    temporary=0.0,
                )
            if need_decode:
                decode_flops = batch_size * embedding_dim
                decode_bytes = ByteBreakdown(
                    weights=effective_weight_bytes,
                    activations=activation_bytes
                    * (batch_size * embedding_dim)
                    * exec_model.activation_fusion_factor,
                    kv=0.0,
                    temporary=0.0,
                )
                add_entry("decode", name, "embedding", decode_flops, decode_bytes, 0.0)
            if "prefill" in requested_modes:
                add_entry("prefill", name, "embedding", prefill_flops, prefill_bytes, 0.0)
            if "training" in requested_modes:
                add_entry(
                    "training",
                    name,
                    "embedding",
                    prefill_flops * training_flops_multiplier,
                    ByteBreakdown(
                        weights=prefill_bytes.weights * training_bytes_multiplier,
                        activations=prefill_bytes.activations * training_bytes_multiplier,
                        kv=0.0,
                        temporary=0.0,
                    ),
                    0.0,
                )

    # Add explicit network dispatch/collect attribution as a separate entry.
    for mode in ["training", "prefill", "decode"]:
        if mode not in requested_modes:
            continue
        _, inter_dispatch_bytes = _estimate_moe_dispatch_bytes(
            model=model,
            mode=mode,
            batch_size=batch_size,
            seq_len=seq_len,
            activation_bytes=activation_bytes,
            top_k_override=top_k_override,
            ep_size_override=ep_size_override,
            hidden_scale=hidden_scale,
        )
        if inter_dispatch_bytes > 0.0:
            add_entry(
                mode=mode,
                name="moe.dispatch.interconnect",
                kind="network",
                flops=0.0,
                byte_breakdown=ByteBreakdown(
                    weights=0.0,
                    activations=0.0,
                    kv=0.0,
                    temporary=0.0,
                ),
                bytes_net=inter_dispatch_bytes,
            )

    return entries


def _summarize_mode_entries(
    mode: str,
    entries_for_mode: List[EfficiencyEntry],
    roofline: RooflineConfig,
    interconnect_bw_gbps: float,
) -> ModeKpiExtended:
    """Aggregate mode entries into extended KPI structure."""
    mem_bw_bytes = roofline.mem_bw_gbps * 1e9
    total_flops = sum(entry.flops_theory for entry in entries_for_mode)
    total_flops_tc = sum(entry.flops_tensorcore for entry in entries_for_mode)
    # `flops_realizable` stores a peak-equivalent compute cost (see `_build_flop_breakdown`).
    total_flops_cost = sum(entry.flops_realizable for entry in entries_for_mode)
    total_hbm_bytes = sum(entry.bytes_hbm for entry in entries_for_mode)
    total_net_bytes = sum(entry.bytes_net for entry in entries_for_mode)
    total_bytes = total_hbm_bytes + total_net_bytes
    total_w = sum(entry.bytes_weights for entry in entries_for_mode)
    total_a = sum(entry.bytes_activations for entry in entries_for_mode)
    total_kv = sum(entry.bytes_kv for entry in entries_for_mode)
    total_tmp = sum(entry.bytes_temporary for entry in entries_for_mode)
    ai_w = _safe_div(total_flops, total_w)
    ai_hbm = _safe_div(total_flops, total_hbm_bytes)
    ai_total = _safe_div(total_flops, total_bytes)
    tflops_hbm = min(roofline.peak_tflops, (mem_bw_bytes * ai_hbm) / 1e12)

    # Compute time is derived from the peak-equivalent cost:
    #   T_comp = F_cost / P_peak.
    t_comp = _safe_div(total_flops_cost, roofline.peak_tflops * 1e12)
    p_effective_tflops = roofline.peak_tflops * _safe_div(total_flops, total_flops_cost)
    t_hbm = _safe_div(total_hbm_bytes, mem_bw_bytes)
    t_net = _safe_div(total_net_bytes, interconnect_bw_gbps * 1e9)
    t_est = max(t_comp, t_hbm, t_net)
    tf_est = _safe_div(total_flops, t_est * 1e12)
    peak_pct = _safe_div(tf_est, roofline.peak_tflops) * 100.0
    mfu_est = _safe_div(total_flops, roofline.peak_tflops * 1e12 * t_est)
    regime = _regime_label(
        flops=total_flops_cost,
        bytes_hbm=total_hbm_bytes,
        bytes_net=total_net_bytes,
        roofline=roofline,
        interconnect_bw_gbps=interconnect_bw_gbps,
    )
    return ModeKpiExtended(
        mode=mode,
        flops_theory=total_flops,
        flops_tensorcore=total_flops_tc,
        flops_realizable=total_flops_cost,
        bytes_weights=total_w,
        bytes_activations=total_a,
        bytes_kv=total_kv,
        bytes_temporary=total_tmp,
        bytes_hbm=total_hbm_bytes,
        bytes_net=total_net_bytes,
        bytes_total=total_bytes,
        ai_weights_only=ai_w,
        ai_hbm=ai_hbm,
        ai_total=ai_total,
        p_effective_tflops=p_effective_tflops,
        t_comp=t_comp,
        t_hbm=t_hbm,
        t_net=t_net,
        t_est=t_est,
        roofline_tflops_hbm=tflops_hbm,
        peak_pct=peak_pct,
        mfu_est=mfu_est,
        regime=regime,
        b_crit=None,
    )


def _build_roofline_piecewise_curve(
    chip: RooflineConfig,
    min_intensity: float,
    max_intensity: float,
    count_per_segment: int = 64,
) -> Tuple[List[float], List[float], List[float], List[float], float]:
    """
    Build piecewise roofline points for one chip.

    Returns:
        mem_x, mem_y, comp_x, comp_y, knee
    """
    min_intensity = max(min_intensity, 1e-12)
    max_intensity = max(max_intensity, min_intensity * 1.01)
    mem_bw = chip.mem_bw_gbps * 1e9
    peak = chip.peak_tflops
    knee = _safe_div(peak * 1e12, max(mem_bw, 1e-12))

    mem_end = min(max_intensity, knee)
    if mem_end <= min_intensity:
        mem_x = [min_intensity]
    else:
        mem_x = _logspace_points(min_intensity, mem_end, count=max(2, count_per_segment))
    mem_y = [(mem_bw * x) / 1e12 for x in mem_x]

    comp_start = max(min_intensity, knee)
    if comp_start >= max_intensity:
        comp_x = [max_intensity]
    else:
        comp_x = _logspace_points(comp_start, max_intensity, count=max(2, count_per_segment))
    comp_y = [peak for _ in comp_x]
    return mem_x, mem_y, comp_x, comp_y, knee


def _draw_roofline_chip_lines(
    ax,
    rooflines: List[RooflineConfig],
    min_intensity: float,
    max_intensity: float,
) -> None:
    """Draw piecewise roofline (memory slope + compute ceiling) for each chip."""
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]
    min_intensity = max(min_intensity, 1e-12)
    max_intensity = max(max_intensity, min_intensity * 1.01)
    for idx, chip in enumerate(rooflines):
        color = palette[idx % len(palette)]
        mem_x, mem_y, comp_x, comp_y, knee = _build_roofline_piecewise_curve(
            chip=chip,
            min_intensity=min_intensity,
            max_intensity=max_intensity,
            count_per_segment=64,
        )
        roofline_label = (
            f"{chip.name} roofline "
            f"(P={chip.peak_tflops:.0f} TF, BW={chip.mem_bw_gbps:.0f} GB/s, "
            f"knee={knee:.1f})"
        )
        ax.loglog(mem_x, mem_y, color=color, linewidth=1.8)
        ax.loglog(comp_x, comp_y, color=color, linewidth=1.8, label=roofline_label)

        if min_intensity <= knee <= max_intensity:
            ax.axvline(knee, color=color, linestyle=":", linewidth=0.9, alpha=0.7)
            ax.scatter([knee], [chip.peak_tflops], color=color, s=28, zorder=5, alpha=0.9)
            ax.annotate(
                f"{chip.name} knee={knee:.1f}",
                (knee, chip.peak_tflops),
                textcoords="offset points",
                xytext=(5, 6),
                fontsize=7,
                color=color,
            )


def _logspace_points(start: float, end: float, count: int = 64) -> List[float]:
    """Return log-spaced points between positive start/end values (inclusive)."""
    start = max(start, 1e-12)
    end = max(end, start)
    if count <= 2 or start == end:
        return [start, end]
    log_s = math.log10(start)
    log_e = math.log10(end)
    step = (log_e - log_s) / float(count - 1)
    return [10 ** (log_s + idx * step) for idx in range(count)]


def _resolve_roofline_axis_limits(
    roofline_targets: List[RooflineConfig],
    point_intensities: List[float],
    point_tflops: List[float],
    requested_x_limits: Optional[Tuple[float, float]],
    requested_y_limits: Optional[Tuple[float, float]],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute robust roofline axis limits and guarantee knee visibility.

    If explicit x-limits hide all knees, widen to include knees and warn.
    """
    import warnings

    knees = [
        _safe_div(chip.peak_tflops * 1e12, chip.mem_bw_gbps * 1e9)
        for chip in roofline_targets
        if chip.mem_bw_gbps > 0
    ]
    positive_points_x = [value for value in point_intensities if value > 0]
    positive_points_y = [value for value in point_tflops if value > 0]

    ref_x = knees + positive_points_x if (knees or positive_points_x) else [1.0]
    x_min_auto = 10 ** math.floor(math.log10(min(ref_x)) - 2)
    x_max_auto = 10 ** math.ceil(math.log10(max(ref_x)) + 2)
    x_min_auto = max(x_min_auto, 1e-4)
    x_max_auto = max(x_max_auto, x_min_auto * 10.0)

    if requested_x_limits is not None:
        x_min = max(min(requested_x_limits), 1e-12)
        x_max = max(max(requested_x_limits), x_min * 1.01)
    else:
        x_min, x_max = x_min_auto, x_max_auto

    if knees and any((knee < x_min) or (knee > x_max) for knee in knees):
        warnings.warn(
            "Provided roofline_x_limits hide one or more chip knees; widening range.",
            RuntimeWarning,
        )
        x_min = min(x_min, min(knees) / 10.0)
        x_max = max(x_max, max(knees) * 10.0)
        x_min = max(x_min, 1e-12)

    mem_floor = min((chip.mem_bw_gbps * 1e9 * x_min) / 1e12 for chip in roofline_targets)
    y_min_auto = max(1e-2, min(mem_floor, min(positive_points_y) if positive_points_y else mem_floor) / 3.0)
    y_max_auto = 1.15 * max(
        max(chip.peak_tflops for chip in roofline_targets),
        max(positive_points_y) if positive_points_y else 1.0,
    )
    y_max_auto = max(y_max_auto, y_min_auto * 10.0)

    if requested_y_limits is not None:
        y_min = max(min(requested_y_limits), 1e-6)
        y_max = max(max(requested_y_limits), y_min * 1.01)
    else:
        y_min, y_max = y_min_auto, y_max_auto

    return (x_min, x_max), (y_min, y_max)


def _linspace(start: float, end: float, count: int) -> List[float]:
    """Return evenly spaced values between start and end (inclusive)."""
    if count <= 0:
        return []
    if count == 1:
        return [start]
    step = (end - start) / float(count - 1)
    return [start + i * step for i in range(count)]


def _annotate_points_side_panel(
    ax,
    points: List[Tuple[float, float, str]],
    color: str,
    panel_x: float,
    y_top: float = 0.95,
    y_bottom: float = 0.08,
    fontsize: int = 7,
) -> None:
    """
    Place point labels in a right-side panel to avoid label overlap on crowded rooflines.

    Args:
        points: List of tuples (x_data, y_data, label_text).
        color: Label and connector color.
        panel_x: X location in axes-fraction coordinates (can be >1.0).
    """
    if not points:
        return
    ordered_points = sorted(points, key=lambda item: item[1], reverse=True)
    y_slots = _linspace(y_top, y_bottom, len(ordered_points))
    for (x_val, y_val, label_text), y_slot in zip(ordered_points, y_slots):
        ax.annotate(
            label_text,
            xy=(x_val, y_val),
            xycoords="data",
            xytext=(panel_x, y_slot),
            textcoords="axes fraction",
            ha="left",
            va="center",
            fontsize=fontsize,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.15",
                "fc": "white",
                "ec": color,
                "alpha": 0.8,
            },
            arrowprops={"arrowstyle": "-", "color": color, "alpha": 0.55, "lw": 0.8},
            annotation_clip=False,
        )


def _plot_roofline_summary(
    efficiencies_by_exec: Dict[str, Dict[str, List[EfficiencyEntry]]],
    primary_roofline: RooflineConfig,
    roofline_targets: List[RooflineConfig],
    output_path: str,
    batch_size: int,
    seq_len: int,
    activation_bytes: int,
    kv_cache_bytes: int,
    roofline_x_limits: Optional[Tuple[float, float]],
    roofline_y_limits: Optional[Tuple[float, float]],
    roofline_label_mode: str,
    interconnect_bw_gbps: float,
) -> Optional[str]:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use("Agg")
    except ImportError:
        return None

    mode_points: List[Tuple[str, str, float, float, float, float]] = []
    for exec_name, efficiency in efficiencies_by_exec.items():
        for mode in ["training", "prefill", "decode"]:
            mode_entries = efficiency.get(mode, [])
            mode_stats = _summarize_mode_entries(
                mode=mode,
                entries_for_mode=mode_entries,
                roofline=primary_roofline,
                interconnect_bw_gbps=interconnect_bw_gbps,
            )
            mode_ai_hbm = mode_stats.ai_hbm
            mode_ai_total = mode_stats.ai_total
            if mode_ai_hbm <= 0.0 or mode_stats.t_est <= 0.0:
                continue
            mode_tflops = mode_stats.flops_theory / (mode_stats.t_est * 1e12)
            mode_points.append(
                (
                    exec_name,
                    mode,
                    mode_ai_hbm,
                    mode_ai_total,
                    mode_tflops,
                    mode_stats.peak_pct,
                )
            )

    if not mode_points:
        return None

    point_intensities = [point[2] for point in mode_points]
    point_tflops = [point[4] for point in mode_points]
    (min_intensity, max_intensity), (min_tflops, max_tflops) = _resolve_roofline_axis_limits(
        roofline_targets=roofline_targets,
        point_intensities=point_intensities,
        point_tflops=point_tflops,
        requested_x_limits=roofline_x_limits,
        requested_y_limits=roofline_y_limits,
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    _draw_roofline_chip_lines(ax, roofline_targets, min_intensity, max_intensity)

    mode_markers = {"training": "o", "prefill": "s", "decode": "^"}
    exec_colors = {"naive": "#1f77b4", "efficient": "#ff7f0e"}
    for exec_name, mode, ai_hbm, _, tflops, peak_pct in mode_points:
        marker = mode_markers.get(mode, "o")
        color = exec_colors.get(exec_name, "#555555")
        point_label = (
            f"{mode} ({exec_name}) "
            f"AI={ai_hbm:.2f}, TF={tflops:.1f}, Peak={peak_pct:.1f}%"
        )
        ax.scatter(
            ai_hbm,
            tflops,
            s=64,
            alpha=0.9,
            marker=marker,
            color=color,
            edgecolors="#333333",
            linewidths=0.8,
            label=point_label,
        )
        if roofline_label_mode == "full":
            ax.annotate(
                f"{mode}/{exec_name}\nAI={ai_hbm:.2e}",
                (ai_hbm, tflops),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )

    ax.set_xlabel("Arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("Estimated TFLOPs (TF_est)")
    ax.set_title(
        f"Mode-Level Roofline Summary (primary={primary_roofline.name})\n"
        f"B={batch_size}, S=L={seq_len}, A={activation_bytes}B, A_kv={kv_cache_bytes}B"
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_xlim(min_intensity, max_intensity)
    ax.set_ylim(min_tflops, max_tflops)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def _plot_decode_batch_roofline(
    points_by_exec: Dict[str, List[SweepPoint]],
    primary_roofline: RooflineConfig,
    roofline_targets: List[RooflineConfig],
    output_path: str,
    seq_len: int,
    activation_bytes: int,
    kv_cache_bytes: int,
    roofline_x_limits: Optional[Tuple[float, float]],
    roofline_y_limits: Optional[Tuple[float, float]],
    roofline_label_mode: str,
) -> Optional[str]:
    """
    Plot decode roofline points across different batch sizes.

    Args:
        points: Sweep points for decode where x=batch size.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use("Agg")
    except ImportError:
        return None

    all_points = [point for points in points_by_exec.values() for point in points if point.ai_hbm > 0.0]
    if not all_points:
        return None

    def _point_tflops(point: SweepPoint) -> float:
        if point.t_est_ms <= 0.0:
            return point.roofline_tflops_hbm
        return point.flops / ((point.t_est_ms / 1000.0) * 1e12)

    point_intensities = [point.ai_hbm for point in all_points]
    point_tflops = [_point_tflops(point) for point in all_points]
    (min_intensity, max_intensity), (min_tflops, max_tflops) = _resolve_roofline_axis_limits(
        roofline_targets=roofline_targets,
        point_intensities=point_intensities,
        point_tflops=point_tflops,
        requested_x_limits=roofline_x_limits,
        requested_y_limits=roofline_y_limits,
    )
    primary_mem_bw = primary_roofline.mem_bw_gbps * 1e9
    primary_peak = primary_roofline.peak_tflops
    primary_knee = (primary_peak * 1e12) / primary_mem_bw

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    _draw_roofline_chip_lines(ax, roofline_targets, min_intensity, max_intensity)

    exec_colors = {"naive": "#1f77b4", "efficient": "#ff7f0e"}
    for exec_name, points in points_by_exec.items():
        series = [p for p in points if p.ai_hbm > 0.0]
        if not series:
            continue
        xs = [p.ai_hbm for p in series]
        ys = [_point_tflops(point) for point in series]
        color = exec_colors.get(exec_name, "#7f7f7f")
        series_label = (
            f"{exec_name} "
            f"(AI {min(xs):.2f}->{max(xs):.2f}, TF max {max(ys):.1f})"
        )
        ax.plot(
            xs,
            ys,
            linestyle="--",
            linewidth=1.3,
            color=color,
            alpha=0.85,
            label=series_label,
        )
        for point in series:
            ax.scatter(point.ai_hbm, _point_tflops(point), s=36, alpha=0.85, color=color)
            if roofline_label_mode == "full":
                ax.annotate(
                    f"B={point.x}",
                    (point.ai_hbm, _point_tflops(point)),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                )
        b_crit = _b_crit_from_sweep(series, primary_knee)
        if b_crit is not None:
            ai_at_bcrit = primary_knee
            y_at_bcrit = min(primary_peak, (primary_mem_bw * ai_at_bcrit) / 1e12)
            ax.scatter(
                [ai_at_bcrit],
                [y_at_bcrit],
                marker="x",
                s=80,
                color=color,
                linewidths=1.2,
            )
            ax.annotate(
                f"{exec_name} B_crit~{b_crit:.1f}",
                (ai_at_bcrit, y_at_bcrit),
                textcoords="offset points",
                xytext=(6, -12),
                fontsize=7,
                color=color,
            )

    ax.set_xlabel("Decode arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("Estimated TFLOPs (TF_est)")
    ax.set_title(
        f"Decode Roofline Sweep (primary={primary_roofline.name})\n"
        f"vary B, fixed S=L={seq_len}, A={activation_bytes}B, A_kv={kv_cache_bytes}B"
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_xlim(min_intensity, max_intensity)
    ax.set_ylim(min_tflops, max_tflops)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def _plot_decode_bl_roofline(
    points_by_exec: Dict[str, List[DecodeWorkloadPoint]],
    primary_roofline: RooflineConfig,
    roofline_targets: List[RooflineConfig],
    output_path: str,
    anchor_ep_size: int,
    cache_lengths: List[int],
    batch_sizes: List[int],
    activation_bytes: int,
    kv_cache_bytes: int,
    routed_experts_per_gpu: Optional[int],
    roofline_x_limits: Optional[Tuple[float, float]],
    roofline_y_limits: Optional[Tuple[float, float]],
    roofline_label_mode: str,
) -> Optional[str]:
    """Plot decode sweep in roofline space for the cartesian product of (B, L) at fixed EP."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use("Agg")
    except ImportError:
        return None

    def tf_est(point: DecodeWorkloadPoint) -> float:
        if point.t_est_ms <= 0.0:
            return 0.0
        return point.flops / ((point.t_est_ms / 1000.0) * 1e12)

    cache_lengths = sorted({int(value) for value in cache_lengths if int(value) > 0})
    batch_sizes = sorted({int(value) for value in batch_sizes if int(value) > 0})
    anchor_ep_size = max(1, int(anchor_ep_size))

    selected: Dict[str, List[DecodeWorkloadPoint]] = {}
    for exec_name in ["naive", "efficient"]:
        selected[exec_name] = [
            point
            for point in points_by_exec.get(exec_name, [])
            if point.ep_size == anchor_ep_size
            and point.cache_len in cache_lengths
            and point.batch in batch_sizes
            and point.ai_hbm > 0.0
            and point.t_est_ms > 0.0
        ]
    if not selected["naive"] and not selected["efficient"]:
        return None

    all_points = selected["naive"] + selected["efficient"]
    point_intensities = [point.ai_hbm for point in all_points]
    point_tflops = [tf_est(point) for point in all_points]
    (min_intensity, max_intensity), (min_tflops, max_tflops) = _resolve_roofline_axis_limits(
        roofline_targets=roofline_targets,
        point_intensities=point_intensities,
        point_tflops=point_tflops,
        requested_x_limits=roofline_x_limits,
        requested_y_limits=roofline_y_limits,
    )

    try:
        import numpy as np
    except ImportError:
        np = None

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), sharex=True, sharey=True)
    exec_order = ["naive", "efficient"]
    for ax, exec_name in zip(axes, exec_order):
        _draw_roofline_chip_lines(ax, roofline_targets, min_intensity, max_intensity)
        points = selected.get(exec_name, [])
        if not points:
            ax.set_title(f"{exec_name} (no points)")
            continue

        grouped: Dict[int, List[DecodeWorkloadPoint]] = {}
        for point in points:
            grouped.setdefault(point.cache_len, []).append(point)

        palette = (
            plt.cm.viridis(np.linspace(0.15, 0.95, len(cache_lengths)))
            if np is not None
            else None
        )
        for idx, cache_len in enumerate(cache_lengths):
            series = grouped.get(cache_len, [])
            if not series:
                continue
            series_sorted = sorted(series, key=lambda p: p.batch)
            xs = [p.ai_hbm for p in series_sorted]
            ys = [tf_est(p) for p in series_sorted]
            color = palette[idx] if palette is not None else None
            ax.plot(
                xs,
                ys,
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
                color=color,
                label=f"L={cache_len}",
            )
            ax.scatter(xs, ys, s=26, alpha=0.9, color=color)
            if roofline_label_mode == "full":
                for point in series_sorted:
                    ax.annotate(
                        f"B={point.batch}",
                        (point.ai_hbm, tf_est(point)),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=7,
                    )

        experts_label = (
            f"E/EP={routed_experts_per_gpu}" if routed_experts_per_gpu is not None else "E/EP=?"
        )
        ax.set_title(f"{exec_name} (EP={anchor_ep_size}, {experts_label})")
        ax.set_xlim(min_intensity, max_intensity)
        ax.set_ylim(min_tflops, max_tflops)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend(loc="lower right", fontsize=8)

    axes[0].set_ylabel("Estimated TFLOPs (TF_est)")
    for ax in axes:
        ax.set_xlabel("Arithmetic intensity (FLOPs / byte)")
    fig.suptitle(
        f"Decode Roofline Sweep (primary={primary_roofline.name})\n"
        f"vary B and L, fixed EP={anchor_ep_size}, A={activation_bytes}B, A_kv={kv_cache_bytes}B",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def _plot_decode_ep_l_minbg_heatmap(
    exploration_metrics: Dict[Tuple[int, int], Dict[str, float]],
    ep_sizes: List[int],
    cache_lengths: List[int],
    primary_roofline: RooflineConfig,
    output_path: str,
    alpha: float,
) -> Optional[str]:
    """Plot a heatmap of compute-feasible `min B_g` over (EP, L)."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import colors
        matplotlib.use("Agg")
    except ImportError:
        return None

    try:
        import numpy as np
    except ImportError:
        return None

    ep_sizes = sorted({int(value) for value in ep_sizes if int(value) > 0})
    cache_lengths = sorted({int(value) for value in cache_lengths if int(value) > 0})
    if not ep_sizes or not cache_lengths:
        return None

    grid = np.full((len(ep_sizes), len(cache_lengths)), np.nan, dtype=float)
    for i, ep in enumerate(ep_sizes):
        for j, cache_len in enumerate(cache_lengths):
            metrics = exploration_metrics.get((ep, cache_len))
            if metrics is None:
                continue
            min_bg = metrics.get("min_bg")
            if min_bg is None or (isinstance(min_bg, float) and math.isnan(min_bg)):
                continue
            grid[i, j] = float(min_bg)

    positive = grid[np.isfinite(grid) & (grid > 0)]
    if positive.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#e6e6e6")
    norm = colors.LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
    im = ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(list(range(len(cache_lengths))))
    ax.set_xticklabels([str(value) for value in cache_lengths], rotation=0)
    ax.set_xlabel("KV length L")
    ax.set_yticks(list(range(len(ep_sizes))))
    ax.set_yticklabels([str(value) for value in ep_sizes])
    ax.set_ylabel("Expert parallel size EP")
    ax.set_title(
        f"Decode Compute-Feasible Frontier: min B_g (primary={primary_roofline.name}, alpha={alpha:.2f})"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("min B_g (log scale)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def _plot_prefill_sequence_roofline(
    points_by_exec: Dict[str, List[SweepPoint]],
    primary_roofline: RooflineConfig,
    roofline_targets: List[RooflineConfig],
    output_path: str,
    batch_size: int,
    activation_bytes: int,
    kv_cache_bytes: int,
    roofline_x_limits: Optional[Tuple[float, float]],
    roofline_y_limits: Optional[Tuple[float, float]],
    roofline_label_mode: str,
) -> Optional[str]:
    """
    Plot prefill roofline points across different sequence lengths.

    Args:
        points: Sweep points for prefill where x=sequence length.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use("Agg")
    except ImportError:
        return None

    all_points = [point for points in points_by_exec.values() for point in points if point.ai_hbm > 0.0]
    if not all_points:
        return None

    def _point_tflops(point: SweepPoint) -> float:
        if point.t_est_ms <= 0.0:
            return point.roofline_tflops_hbm
        return point.flops / ((point.t_est_ms / 1000.0) * 1e12)

    point_intensities = [point.ai_hbm for point in all_points]
    point_tflops = [_point_tflops(point) for point in all_points]
    (min_intensity, max_intensity), (min_tflops, max_tflops) = _resolve_roofline_axis_limits(
        roofline_targets=roofline_targets,
        point_intensities=point_intensities,
        point_tflops=point_tflops,
        requested_x_limits=roofline_x_limits,
        requested_y_limits=roofline_y_limits,
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    _draw_roofline_chip_lines(ax, roofline_targets, min_intensity, max_intensity)

    exec_colors = {"naive": "#1f77b4", "efficient": "#ff7f0e"}
    for exec_name, points in points_by_exec.items():
        series = [p for p in points if p.ai_hbm > 0.0]
        if not series:
            continue
        xs = [p.ai_hbm for p in series]
        ys = [_point_tflops(point) for point in series]
        color = exec_colors.get(exec_name, "#7f7f7f")
        series_label = (
            f"{exec_name} "
            f"(AI {min(xs):.2f}->{max(xs):.2f}, TF max {max(ys):.1f})"
        )
        ax.plot(
            xs,
            ys,
            linestyle="--",
            linewidth=1.3,
            color=color,
            alpha=0.85,
            label=series_label,
        )
        for point in series:
            ax.scatter(point.ai_hbm, _point_tflops(point), s=36, alpha=0.85, color=color)
            if roofline_label_mode == "full":
                ax.annotate(
                    f"S={point.x}",
                    (point.ai_hbm, _point_tflops(point)),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                )

    ax.set_xlabel("Prefill arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("Estimated TFLOPs (TF_est)")
    ax.set_title(
        f"Prefill Roofline Sweep (primary={primary_roofline.name})\n"
        f"vary S, fixed B={batch_size}, A={activation_bytes}B, A_kv={kv_cache_bytes}B"
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_xlim(min_intensity, max_intensity)
    ax.set_ylim(min_tflops, max_tflops)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def _plot_prefill_bs_roofline(
    points_by_exec: Dict[str, List[PrefillWorkloadPoint]],
    primary_roofline: RooflineConfig,
    roofline_targets: List[RooflineConfig],
    output_path: str,
    anchor_ep_size: int,
    batch_sizes: List[int],
    activation_bytes: int,
    kv_cache_bytes: int,
    routed_experts_per_gpu: Optional[int],
    roofline_x_limits: Optional[Tuple[float, float]],
    roofline_y_limits: Optional[Tuple[float, float]],
    roofline_label_mode: str,
) -> Optional[str]:
    """Plot prefill sweep in roofline space for an S sweep at multiple B with fixed EP."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use("Agg")
    except ImportError:
        return None

    def tf_est(point: PrefillWorkloadPoint) -> float:
        if point.t_est_ms <= 0.0:
            return 0.0
        return point.flops / ((point.t_est_ms / 1000.0) * 1e12)

    anchor_ep_size = max(1, int(anchor_ep_size))
    batch_sizes = sorted({int(value) for value in batch_sizes if int(value) > 0})
    selected: Dict[str, List[PrefillWorkloadPoint]] = {}
    for exec_name in ["naive", "efficient"]:
        selected[exec_name] = [
            point
            for point in points_by_exec.get(exec_name, [])
            if point.ep_size == anchor_ep_size
            and point.batch in batch_sizes
            and point.ai_hbm > 0.0
            and point.t_est_ms > 0.0
        ]
    if not selected["naive"] and not selected["efficient"]:
        return None

    all_points = selected["naive"] + selected["efficient"]
    point_intensities = [point.ai_hbm for point in all_points]
    point_tflops = [tf_est(point) for point in all_points]
    (min_intensity, max_intensity), (min_tflops, max_tflops) = _resolve_roofline_axis_limits(
        roofline_targets=roofline_targets,
        point_intensities=point_intensities,
        point_tflops=point_tflops,
        requested_x_limits=roofline_x_limits,
        requested_y_limits=roofline_y_limits,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), sharex=True, sharey=True)
    exec_order = ["naive", "efficient"]
    for ax, exec_name in zip(axes, exec_order):
        _draw_roofline_chip_lines(ax, roofline_targets, min_intensity, max_intensity)
        points = selected.get(exec_name, [])
        grouped: Dict[int, List[PrefillWorkloadPoint]] = {}
        for point in points:
            grouped.setdefault(point.batch, []).append(point)

        for batch in batch_sizes:
            series = grouped.get(batch, [])
            if not series:
                continue
            series_sorted = sorted(series, key=lambda p: p.seq_len)
            xs = [p.ai_hbm for p in series_sorted]
            ys = [tf_est(p) for p in series_sorted]
            ax.plot(
                xs,
                ys,
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
                label=f"B={batch}",
            )
            ax.scatter(xs, ys, s=26, alpha=0.9)
            if roofline_label_mode == "full":
                for point in series_sorted:
                    ax.annotate(
                        f"S={point.seq_len}",
                        (point.ai_hbm, tf_est(point)),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=7,
                    )

        experts_label = (
            f"E/EP={routed_experts_per_gpu}" if routed_experts_per_gpu is not None else "E/EP=?"
        )
        ax.set_title(f"{exec_name} (EP={anchor_ep_size}, {experts_label})")
        ax.set_xlim(min_intensity, max_intensity)
        ax.set_ylim(min_tflops, max_tflops)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend(loc="lower right", fontsize=8)

    axes[0].set_ylabel("Estimated TFLOPs (TF_est)")
    for ax in axes:
        ax.set_xlabel("Arithmetic intensity (FLOPs / byte)")
    fig.suptitle(
        f"Prefill Roofline Sweep (primary={primary_roofline.name})\n"
        f"vary B and S, fixed EP={anchor_ep_size}, A={activation_bytes}B, A_kv={kv_cache_bytes}B",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def _plot_category_roofline(
    efficiencies_by_exec: Dict[str, Dict[str, List[EfficiencyEntry]]],
    primary_roofline: RooflineConfig,
    roofline_targets: List[RooflineConfig],
    output_path: str,
    mode: str,
    interconnect_bw_gbps: float,
    roofline_x_limits: Optional[Tuple[float, float]],
    roofline_y_limits: Optional[Tuple[float, float]],
    roofline_label_mode: str,
) -> Optional[str]:
    """Plot category-level roofline points (attention/moe/dense/embedding/network)."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use("Agg")
    except ImportError:
        return None

    points: List[Tuple[str, str, float, float]] = []
    for exec_name, mode_map in efficiencies_by_exec.items():
        entries = mode_map.get(mode, [])
        grouped: Dict[str, List[EfficiencyEntry]] = {}
        for entry in entries:
            category = _categorize_efficiency_entry(entry)
            grouped.setdefault(category, []).append(entry)

        for category, cat_entries in grouped.items():
            flops_theory = sum(entry.flops_theory for entry in cat_entries)
            flops_cost = sum(entry.flops_realizable for entry in cat_entries)
            bytes_hbm = sum(entry.bytes_hbm for entry in cat_entries)
            bytes_net = sum(entry.bytes_net for entry in cat_entries)
            ai_hbm = _safe_div(flops_theory, bytes_hbm)
            if ai_hbm <= 0.0 or flops_theory <= 0.0:
                continue
            t_comp = _safe_div(flops_cost, primary_roofline.peak_tflops * 1e12)
            t_hbm = _safe_div(bytes_hbm, primary_roofline.mem_bw_gbps * 1e9)
            t_net = _safe_div(bytes_net, interconnect_bw_gbps * 1e9)
            t_est = max(t_comp, t_hbm, t_net)
            tf_est = _safe_div(flops_theory, max(1e-12, t_est) * 1e12)
            points.append((exec_name, category, ai_hbm, tf_est))

    if not points:
        return None

    point_intensities = [point[2] for point in points]
    point_tflops = [point[3] for point in points]
    (min_intensity, max_intensity), (min_tflops, max_tflops) = _resolve_roofline_axis_limits(
        roofline_targets=roofline_targets,
        point_intensities=point_intensities,
        point_tflops=point_tflops,
        requested_x_limits=roofline_x_limits,
        requested_y_limits=roofline_y_limits,
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    _draw_roofline_chip_lines(ax, roofline_targets, min_intensity, max_intensity)

    category_markers = {
        "attention": "o",
        "experts": "s",
        "ffn": "^",
        "embedding": "D",
        "norm": "v",
        "network": "x",
        "other": "P",
    }
    exec_colors = {"naive": "#1f77b4", "efficient": "#ff7f0e"}
    for exec_name, category, ai_hbm, tflops in points:
        color = exec_colors.get(exec_name, "#555555")
        marker = category_markers.get(category, "o")
        point_label = f"{category} ({exec_name}) AI={ai_hbm:.2f}, TF={tflops:.1f}"
        ax.scatter(
            ai_hbm,
            tflops,
            color=color,
            marker=marker,
            s=62,
            alpha=0.9,
            edgecolors="#333333",
            linewidths=0.8,
            label=point_label,
        )
        if roofline_label_mode == "full":
            ax.annotate(
                f"{category}/{exec_name}",
                (ai_hbm, tflops),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )

    ax.set_xlabel("Category arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("Estimated TFLOPs (TF_est)")
    categories_in_plot = sorted({point[1] for point in points})
    ax.set_title(
        f"Category Roofline ({mode}, primary={primary_roofline.name})\n"
        f"Categories: {', '.join(categories_in_plot)}"
    )
    ax.set_xlim(min_intensity, max_intensity)
    ax.set_ylim(min_tflops, max_tflops)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def _plot_mode_category_stacks(
    mode_category_shares: Dict[str, Dict[str, Tuple[float, float]]],
    output_path: str,
) -> Optional[str]:
    """
    Plot stacked category shares across train/prefill/decode.

    For each mode/category tuple:
      value = (flops_share_pct, bytes_share_pct)
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use("Agg")
    except ImportError:
        return None

    modes = ["training", "prefill", "decode"]
    categories = sorted(
        {
            category
            for mode in modes
            for category in mode_category_shares.get(mode, {}).keys()
        }
    )
    if not categories:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
    titles = ["FLOPs Share by Category", "Bytes Share by Category"]
    value_idx = [0, 1]

    for ax, title, idx in zip(axes, titles, value_idx):
        bottoms = [0.0, 0.0, 0.0]
        x = list(range(len(modes)))
        for category in categories:
            vals = []
            for mode in modes:
                shares = mode_category_shares.get(mode, {}).get(category, (0.0, 0.0))
                vals.append(shares[idx])
            ax.bar(x, vals, bottom=bottoms, label=category)
            bottoms = [b + v for b, v in zip(bottoms, vals)]

        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in modes])
        ax.set_ylim(0, 100)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_ylabel("Percent")

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def _default_prefill_seq_lengths(model: torch.nn.Module, base_seq_len: int) -> List[int]:
    """Build a useful prefill sequence-length sweep list."""
    model_max_seq = base_seq_len
    if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings"):
        try:
            model_max_seq = int(model.config.max_position_embeddings)
        except Exception:
            model_max_seq = base_seq_len

    candidates = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    sweep = [seq for seq in candidates if seq <= model_max_seq]
    if model_max_seq > 32768:
        sweep.append(model_max_seq)
    if base_seq_len > 0 and base_seq_len <= model_max_seq:
        # Always include the anchor length used for the report's base prefill KPIs.
        sweep.append(base_seq_len)
    if not sweep:
        sweep = [max(1, base_seq_len)]

    return sorted(set(sweep))


def _plot_param_pie(
    category_params: Dict[str, int],
    output_path: str,
) -> Optional[str]:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use("Agg")
    except ImportError:
        return None

    if not category_params:
        return None

    labels = []
    values = []
    for name, value in sorted(category_params.items(), key=lambda item: item[1], reverse=True):
        labels.append(name)
        values.append(value)

    if sum(values) == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, _, _ = ax.pie(
        values,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("Parameter Distribution by Category")
    ax.legend(
        wedges,
        labels,
        title="Categories",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def _plot_weight_histogram(weights: torch.Tensor, output_path: str) -> Optional[str]:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use("Agg")
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(weights.cpu().numpy(), bins=120, color="#2ca02c", alpha=0.75)
    ax.set_title("Aggregate Weight Distribution")
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def _collect_weights_sample(
    layers: List[LayerInfo],
    model: torch.nn.Module,
    max_samples: int
) -> torch.Tensor:
    samples = []
    remaining = max_samples

    params = {name: p for name, p in model.named_parameters()}
    for layer in layers:
        param = params.get(layer.name)
        if param is None:
            continue
        if getattr(param, "is_meta", False):
            continue
        flat = param.detach().flatten()
        if flat.numel() <= remaining:
            samples.append(flat)
            remaining -= flat.numel()
        else:
            idx = torch.randperm(flat.numel())[:remaining]
            samples.append(flat[idx])
            remaining = 0
        if remaining <= 0:
            break

    if not samples:
        return torch.tensor([])

    return torch.cat(samples).float()



def _collect_decode_ep_exploration(
    model: torch.nn.Module,
    roofline: RooflineConfig,
    activation_bytes: int,
    kv_cache_bytes: int,
    interconnect_bw_gbps: float,
    param_bytes_assumed: int,
    cfg: DecodeEPExplorationConfig,
) -> Tuple[str, Dict[Tuple[int, int], Dict[str, float]]]:
    """
    Explore, under the simplifying assumption DP=EP, TP=1, no PP/SP:
      - per-GPU microbatch B_g required to make decode "close to compute-bound"
      - detect when it is impossible due to KV/HBM ridge or network ridge.

    Returns:
      - Markdown section string
      - a dict keyed by (EP, L) with metrics (for potential downstream plotting).
    """
    peak = roofline.peak_tflops * 1e12
    bw_hbm = roofline.mem_bw_gbps * 1e9
    bw_net = float(interconnect_bw_gbps) * 1e9
    oi_hbm_ridge = _safe_div(peak, bw_hbm)
    oi_net_ridge = _safe_div(peak, bw_net) if bw_net > 0 else float('inf')

    def params_bytes(module: torch.nn.Module) -> int:
        return int(sum(p.numel() for p in module.parameters()) * param_bytes_assumed)

    # Prefer block-level representative modules for DeepSeek-style models.
    # This avoids scanning all submodules/parameters when layer shapes are repeated.
    n_attn_layers = 0
    n_moe_layers = 0
    n_dense_layers = 0
    representative_mla: Optional[torch.nn.Module] = None
    representative_moe: Optional[torch.nn.Module] = None
    attn_weight_bytes_total = 0.0

    blocks = list(getattr(model, "blocks", [])) if hasattr(model, "blocks") else []
    if blocks:
        n_attn_layers = len(blocks)

        for block in blocks:
            attn = getattr(block, "attn", None)
            if attn is not None and _is_mla_attention(attn):
                representative_mla = attn
                break

        for block in blocks:
            ffn = getattr(block, "ffn", None)
            if hasattr(ffn, "num_experts") and hasattr(ffn, "top_k") and hasattr(ffn, "experts"):
                if representative_moe is None:
                    representative_moe = ffn
                n_moe_layers += 1

        n_dense_layers = max(0, n_attn_layers - n_moe_layers)
        if representative_mla is not None:
            attn_weight_bytes_total = float(n_attn_layers * params_bytes(representative_mla))
    else:
        # Generic fallback for non-DeepSeek models.
        attention_names = set(_attention_module_names(model))
        attn_modules = [m for name, m in model.named_modules() if name in attention_names]
        mla_modules = [m for m in attn_modules if _is_mla_attention(m)]
        moe_modules = [
            m
            for _, m in model.named_modules()
            if hasattr(m, "num_experts") and hasattr(m, "top_k") and hasattr(m, "experts")
        ]
        if attn_modules:
            n_attn_layers = len(attn_modules)
        if moe_modules:
            n_moe_layers = len(moe_modules)
        n_dense_layers = max(0, n_attn_layers - n_moe_layers)
        if mla_modules:
            representative_mla = mla_modules[0]
            attn_weight_bytes_total = float(n_attn_layers * params_bytes(representative_mla))
        if moe_modules:
            representative_moe = moe_modules[0]

    # If missing key modules, return a short note.
    if representative_mla is None or representative_moe is None or n_attn_layers <= 0 or n_moe_layers <= 0:
        md = (
            "### Decode compute-bound exploration (DP=EP, TP=1)\n\n"
            "We include this subsection to answer a feasibility question for decode: can decode "
            "ever become compute-limited under the static model assumptions, and if so what "
            "per-GPU microbatch would be required as a function of `EP` and `L`. This analysis is "
            "only meaningful when we can identify both MLA attention (for KV sizing) and routed "
            "MoE (for expert scaling).\n\n"
            "- Skipped: could not confidently detect both MLA attention and routed MoE modules.\n"
        )
        return md, {}

    # Use representative modules and layer counts.
    H, h, r_q, r_kv, d_nope, d_rope, d_v = _mla_dims(representative_mla)
    d_q = d_nope + d_rope
    kv_elems_per_tok = r_kv + d_rope

    # MoE dims (representative)
    moe0 = representative_moe
    E = max(1, int(getattr(moe0, "num_experts")))
    k = max(1, int(getattr(moe0, "top_k")))

    # Infer expert intermediate size from the first expert (assume SwiGLU-like).
    d_moe = None
    if hasattr(moe0.experts[0], "gate_proj") and hasattr(moe0.experts[0].gate_proj, "out_features"):
        d_moe = int(moe0.experts[0].gate_proj.out_features)
    if d_moe is None:
        # Fallback to a common DeepSeek-V3 setting; keep explicit in report.
        d_moe = 2048

    # Per MoE layer, estimate bytes for router/shared and one expert; then scale with EP & active experts.
    # Router and shared experts are always present (if exist).
    router_bytes_per_layer = 0
    if hasattr(moe0, "router"):
        router_bytes_per_layer = params_bytes(moe0.router)

    shared_bytes_per_layer = 0
    if hasattr(moe0, "shared_experts"):
        shared_bytes_per_layer = sum(params_bytes(se) for se in getattr(moe0, "shared_experts"))

    # One expert MLP bytes (assume identical)
    expert_bytes = params_bytes(moe0.experts[0])

    # FLOPs per token (per layer)
    # Attention: decode (seq_len=1) has O(L) term + constant projections.
    def attn_flops_per_token(L: int) -> float:
        proj = (
            2.0 * H * r_q
            + 2.0 * r_q * (h * d_q)
            + 2.0 * H * (r_kv + d_rope)
            + 2.0 * r_kv * (h * (d_nope + d_v))
            + 2.0 * (h * d_v) * H
        )
        qk = 2.0 * h * L * d_q
        pv = 2.0 * h * L * d_v
        softmax = 1.0 * h * L
        return proj + qk + pv + softmax

    # MoE: per token expert-call FLOPs (SwiGLU-like) ~= 6 * H * d_moe
    expert_flops = 6.0 * H * d_moe
    router_flops = 2.0 * H * E  # H -> E

    def moe_flops_per_token_per_gpu(EP: int) -> float:
        EP = max(1, int(EP))
        return router_flops + (float(k) / float(EP)) * expert_flops

    # Bytes per token (dominant terms)
    def kv_bytes_per_token(L: int) -> float:
        # Read KV for cache_len and write current token KV. MLA stores kv_latent + rope key per position.
        read_b = L * kv_elems_per_tok * kv_cache_bytes
        write_b = kv_elems_per_tok * kv_cache_bytes
        return float(read_b + write_b)

    # Network bytes per token (MoE dispatch): send+recv hidden states to k experts.
    def net_bytes_per_token() -> float:
        return float(2 * k * H * activation_bytes)

    # Routing-aware active experts model (step-level): fraction of experts activated globally this step.
    # global_tokens = B_g * EP; expected activated experts ~ E*(1-exp(-global_tokens*k/E))
    def active_expert_count_per_gpu(Bg: int, EP: int) -> float:
        EP = max(1, int(EP))
        Bg = max(1, int(Bg))
        global_calls = float(Bg * EP * k)
        p_active = 1.0 - math.exp(-_safe_div(global_calls, float(E)))
        # local experts per GPU ~ E/EP
        return p_active * (float(E) / float(EP))

    # Solve Bg for each (EP, L).
    metrics: Dict[Tuple[int, int], Dict[str, float]] = {}

    def time_components(Bg: int, EP: int, L: int) -> Tuple[float, float, float, float]:
        # FLOPs per step per GPU
        flops_tok = n_attn_layers * attn_flops_per_token(L) + n_moe_layers * moe_flops_per_token_per_gpu(EP)
        F = float(Bg) * flops_tok

        # HBM bytes per step per GPU
        # - weights: attention weights always; MoE weights depend on activated local experts
        w_factor = max(1e-6, float(cfg.weight_residency_factor))
        attn_w = attn_weight_bytes_total / w_factor

        local_active_experts = active_expert_count_per_gpu(Bg, EP)
        moe_w = n_moe_layers * (router_bytes_per_layer + shared_bytes_per_layer + local_active_experts * expert_bytes) / w_factor

        kv_b = float(Bg) * n_attn_layers * kv_bytes_per_token(L)

        M_hbm = float(attn_w + moe_w + kv_b)

        # Net bytes per step per GPU
        M_net = float(Bg) * net_bytes_per_token()

        T_comp = _safe_div(F, peak)
        T_hbm = _safe_div(M_hbm, bw_hbm)
        T_net = _safe_div(M_net, bw_net) if bw_net > 0 else 0.0
        return T_comp, T_hbm, T_net, F

    def oi_infty_hbm(EP: int, L: int) -> float:
        # Bg->inf removes weight terms; KV dominates HBM traffic.
        flops_tok = n_attn_layers * attn_flops_per_token(L) + n_moe_layers * moe_flops_per_token_per_gpu(EP)
        bytes_tok = n_attn_layers * kv_bytes_per_token(L)
        return _safe_div(flops_tok, bytes_tok)

    def oi_net(EP: int, L: int) -> float:
        flops_tok = n_attn_layers * attn_flops_per_token(L) + n_moe_layers * moe_flops_per_token_per_gpu(EP)
        return _safe_div(flops_tok, net_bytes_per_token())

    rows: List[str] = []
    rows.append("### Decode compute-bound exploration (DP=EP, TP=1, no PP/SP)\n")
    rows.append(
        "We answer a feasibility question for decode: for a given expert-parallel size `EP` and "
        "KV-cache length `L`, can decode ever become compute-limited under the same static model "
        "used elsewhere in this report, and if so what per-GPU microbatch `B_g` would be required. "
        "We define `B_g` as the microbatch processed by one GPU under the simplifying assumption "
        "`DP=EP` with `TP=1`, so increasing `EP` reduces the routed expert set per GPU to "
        "`ceil(E/EP)` but can introduce network dispatch. We compute ridge intensities from the "
        f"hardware target (`OI_hbm={oi_hbm_ridge:.1f}`, `OI_net={oi_net_ridge:.1f}`, "
        f"BW_net={interconnect_bw_gbps:.0f} GB/s) and then search for the smallest `B_g` such that "
        f"`T_comp >= alpha*max(T_hbm,T_net)` with `alpha={cfg.alpha:.2f}`.\n"
    )
    rows.append("\n")
    rows.append("Assumptions:\n")
    rows.append("- `DP=EP`, `TP=1`, no PP/SP; all quantities are interpreted per GPU.\n")
    rows.append(f"- Routed experts per GPU: `ceil(E/EP)` with `E={E}` and `top_k={k}`.\n")
    rows.append(
        "- We reuse the same time model terms (`T_comp`, `T_hbm`, `T_net`) and treat a setting as "
        "compute-favorable when `T_comp` is within `alpha` of the slower of HBM and network.\n"
    )
    rows.append("\n")
    rows.append(
        "How to read the table: `OI_inf_hbm(L)` is the asymptotic HBM arithmetic intensity as "
        "`B_g→∞` where weight terms amortize and KV reads dominate; if this asymptote is below the "
        "HBM ridge, no batch can make decode compute-limited for that `L`. `OI_net` is a similar "
        "diagnostic against the network ridge. `min B_g` is the first batch in our search grid "
        f"(up to `{cfg.max_batch_per_gpu}`) that satisfies the compute-favorable criterion; an "
        "empty cell indicates a KV/network asymptote or a missed crossing within the search bound.\n"
    )
    rows.append("\n")

    # Table header
    header = "| EP | L (KV len) | OI_inf_hbm (F/B) | OI_net (F/B) | min B_g (alpha={:.2f}) | limiter @ min B_g |\n|---:|---:|---:|---:|---:|:--|\n".format(cfg.alpha)
    rows.append(header)

    for EP in cfg.ep_sizes:
        for L in cfg.cache_lengths:
            oi_inf = oi_infty_hbm(EP, L)
            oi_n = oi_net(EP, L)
            status = ""
            min_bg = None
            limiter = ""

            # Asymptotic checks
            if oi_inf < oi_hbm_ridge * cfg.alpha:
                status = "HBM(KV)-bound for any B"
                min_bg = math.nan
                limiter = "HBM (KV)"
            elif oi_n < oi_net_ridge * cfg.alpha:
                status = "NET-bound for any B"
                min_bg = math.nan
                limiter = "NET"
            else:
                # brute-force search Bg
                for Bg in [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384]:
                    if Bg > cfg.max_batch_per_gpu:
                        break
                    T_comp, T_hbm, T_net, F = time_components(Bg, EP, L)
                    bound = max(T_hbm, T_net)
                    if bound <= 0:
                        continue
                    if T_comp >= cfg.alpha * bound:
                        min_bg = Bg
                        if T_hbm >= T_net:
                            limiter = "HBM"
                        else:
                            limiter = "NET"
                        break
                if min_bg is None:
                    min_bg = math.nan
                    limiter = "HBM/NET"
                    status = "Not found within search grid"

            rows.append(f"| {EP} | {L} | {oi_inf:.1f} | {oi_n:.1f} | {'' if math.isnan(min_bg) else int(min_bg)} | {limiter if limiter else status} |\n")
            metrics[(EP, L)] = {
                "oi_inf_hbm": float(oi_inf),
                "oi_net": float(oi_n),
                "min_bg": float(min_bg) if not math.isnan(min_bg) else float('nan'),
            }

    rows.append("\n**Model-side drivers (inferred):**\n")
    rows.append(
        f"- MLA KV elements per token per layer: `r_kv + d_rope = {r_kv} + {d_rope} = {kv_elems_per_tok}`.\n"
    )
    rows.append(
        f"- MoE: routed experts `E={E}`, `top_k={k}`, per-expert MLP dim `d_moe={d_moe}`; expert "
        f"compute scales like `~6*H*d_moe` with `H={H}` per token.\n"
    )
    rows.append(
        "As `L` grows, KV bytes scale linearly, so `OI_inf_hbm(L)` falls roughly like `~1/L` once "
        "KV dominates. In that regime, batching cannot move decode to the compute side because the "
        "asymptotic intensity stays left of the ridge.\n"
    )

    return "".join(rows), metrics


def _run_sensitivity_analysis(
    model: torch.nn.Module,
    execution_models: List[ExecutionModelConfig],
    sensitivity_cfg: SensitivityConfig,
    batch_size: int,
    seq_len: int,
    ep_size_override: Optional[int],
    activation_bytes: int,
    kv_cache_bytes: int,
    param_bytes_assumed: int,
    roofline: RooflineConfig,
    interconnect_bw_gbps: float,
    training_flops_multiplier: float,
    training_bytes_multiplier: float,
    tc_cfg: TensorCoreModelConfig,
    module_weight_bytes_cache: Optional[Dict[int, float]],
    progress_cb=None,
) -> Dict[str, List[SensitivityPoint]]:
    """
    Run combinational sensitivity sweep and return per-exec-mode points.

    Sweep dimensions:
      kv_dtype_bytes x top_k x kv_rank_scale x hidden_scale x cache_len
    """
    points_by_exec: Dict[str, List[SensitivityPoint]] = {
        model_cfg.name: [] for model_cfg in execution_models
    }
    is_deepseek = _is_deepseek_model(model)
    attention_names = _attention_module_names(model) if not is_deepseek else []
    moe_fraction_cache: Dict[int, Dict[str, float]] = {}
    start_time = time.time()
    grid = []
    for kv_dtype in sensitivity_cfg.kv_dtype_bytes:
        for top_k in sensitivity_cfg.top_k_values:
            for kv_rank_scale in sensitivity_cfg.kv_rank_scales:
                for hidden_scale in sensitivity_cfg.hidden_scales:
                    for cache_len in sensitivity_cfg.cache_lengths:
                        grid.append((kv_dtype, top_k, kv_rank_scale, hidden_scale, cache_len))

    total = len(grid) * len(execution_models)
    done = 0
    for exec_model in execution_models:
        for kv_dtype, top_k, kv_rank_scale, hidden_scale, cache_len in grid:
            done += 1
            if progress_cb is not None and done % 20 == 0:
                elapsed = max(1e-9, time.time() - start_time)
                points_per_second = done / elapsed
                eta_seconds = (total - done) / max(1e-9, points_per_second)
                progress_cb(
                    f"Sensitivity sweep progress: {done}/{total} "
                    f"({points_per_second:.2f} pts/s, ETA {eta_seconds:.1f}s)"
                )
            if top_k not in moe_fraction_cache:
                if not is_deepseek:
                    moe_fraction_cache[top_k] = _build_moe_active_fractions(
                        model=model,
                        top_k_override=top_k,
                    )
            eff = _estimate_efficiency(
                model=model,
                batch_size=batch_size,
                seq_len=cache_len,
                activation_bytes=activation_bytes,
                kv_cache_bytes=kv_dtype,
                param_bytes_assumed=param_bytes_assumed,
                roofline=roofline,
                interconnect_bw_gbps=interconnect_bw_gbps,
                training_flops_multiplier=training_flops_multiplier,
                training_bytes_multiplier=training_bytes_multiplier,
                exec_model=exec_model,
                tc_cfg=tc_cfg,
                top_k_override=top_k,
                hidden_scale=hidden_scale,
                kv_rank_scale=kv_rank_scale,
                module_weight_bytes_cache=module_weight_bytes_cache,
                modes_to_estimate=("decode",),
                attention_names_override=attention_names if attention_names else None,
                moe_active_fractions_override=moe_fraction_cache.get(top_k),
                ep_size_override=ep_size_override,
            )
            mode_kpi = _summarize_mode_entries(
                mode="decode",
                entries_for_mode=eff["decode"],
                roofline=roofline,
                interconnect_bw_gbps=interconnect_bw_gbps,
            )
            points_by_exec[exec_model.name].append(
                SensitivityPoint(
                    exec_model=exec_model.name,
                    kv_dtype_bytes=kv_dtype,
                    top_k=top_k,
                    kv_rank_scale=kv_rank_scale,
                    hidden_scale=hidden_scale,
                    cache_len=cache_len,
                    ai_hbm=mode_kpi.ai_hbm,
                    ai_total=mode_kpi.ai_total,
                    t_est_ms=mode_kpi.t_est * 1000.0,
                    mfu_est=mode_kpi.mfu_est,
                    regime=mode_kpi.regime,
                )
            )
    return points_by_exec


def _write_sensitivity_csv(
    points_by_exec: Dict[str, List[SensitivityPoint]],
    output_path: str,
) -> Optional[str]:
    """Write sensitivity points to CSV and return path."""
    import csv

    rows: List[SensitivityPoint] = []
    for points in points_by_exec.values():
        rows.extend(points)
    if not rows:
        return None
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "exec_model",
            "kv_dtype_bytes",
            "top_k",
            "kv_rank_scale",
            "hidden_scale",
            "cache_len",
            "ai_hbm",
            "ai_total",
            "t_est_ms",
            "mfu_est",
            "regime",
        ])
        for row in rows:
            writer.writerow([
                row.exec_model,
                row.kv_dtype_bytes,
                row.top_k,
                row.kv_rank_scale,
                row.hidden_scale,
                row.cache_len,
                row.ai_hbm,
                row.ai_total,
                row.t_est_ms,
                row.mfu_est,
                row.regime,
            ])
    return output_path


def _append_section_preamble(
    report_lines: List[str],
    why_lines: List[str],
    look_for_lines: List[str],
) -> None:
    """
    Append a compact paper-style paragraph introducing a section.

    We deliberately prefer prose over checklists: the reader should understand *why* this section
    exists before seeing dense tables.
    """

    def _ensure_sentence(text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        text = text.rstrip(".")
        return f"{text}."

    sentences: List[str] = []
    if why_lines:
        sentences.extend(
            sentence
            for sentence in (_ensure_sentence(line) for line in why_lines)
            if sentence
        )
    else:
        sentences.append(
            "We use this section to set up the reasoning that connects model structure to the "
            "efficiency estimates shown later."
        )

    if look_for_lines:
        cues = [cue.strip().rstrip(".") for cue in look_for_lines if cue.strip()]
        if cues:
            if len(cues) == 1:
                cue_text = cues[0]
            elif len(cues) == 2:
                cue_text = f"{cues[0]} and {cues[1]}"
            else:
                cue_text = ", ".join(cues[:-1]) + f", and {cues[-1]}"
            sentences.append(f"In the evidence below, we focus on {cue_text}.")

    report_lines.append(" ".join(sentences).strip())
    report_lines.append("")


def _compute_mode_byte_shares(mode_kpi: ModeKpiExtended) -> Dict[str, float]:
    """Return byte-share fractions for a mode KPI entry."""
    denom = max(1e-12, mode_kpi.bytes_hbm)
    return {
        "weights": mode_kpi.bytes_weights / denom,
        "activations": mode_kpi.bytes_activations / denom,
        "kv": mode_kpi.bytes_kv / denom,
        "temporary": mode_kpi.bytes_temporary / denom,
    }


def _render_how_to_read_section() -> List[str]:
    """Render the top-level reading guide section."""
    lines: List[str] = []
    lines.append("## How to Read This Report")
    lines.append("")
    lines.append("### Goal")
    lines.append("")
    lines.append(
        "We want a simple outcome: given a model and a chip target, predict what limits throughput "
        "or latency in training, prefill, and decode, and then identify which knobs actually move "
        "those limits. We proceed from architecture to math/bytes to roofline to regimes to "
        "sensitivity, so each later table has an explicit definition and a reason to exist."
    )
    lines.append("")
    lines.append("### Workflow (3 Steps)")
    lines.append("")
    lines.append(
        "First, we classify the limiting regime by comparing arithmetic intensity to the chip "
        "ridge (`OI_knee = P_peak / BW_hbm`) and by inspecting the modeled times "
        "`T_comp`, `T_hbm`, and `T_net`. Second, we locate the dominant byte term using the byte "
        "decomposition (weights, activations, KV, temporary). Third, we map the regime to an "
        "optimization family: compute-bound suggests utilization/fusion; HBM(weight)-bound suggests "
        "weight residency/compression; HBM(KV)-bound suggests KV format/dtype/layout; and "
        "network-bound suggests topology/compression/overlap."
    )
    lines.append("")
    lines.append("### Common Gotchas")
    lines.append("")
    lines.append(
        "`F_theory` is an algorithmic count; it is not a performance claim. We explicitly separate "
        "`F_theory` from `F_realizable` (a peak-equivalent compute cost under a utilization model) "
        "because tiny-batch decode and thin GEMMs can leave tensor cores under-saturated. Decode "
        "also has a distinct length variable (`L`, KV length), and long-context decode can trend "
        "toward KV-driven `~1/L` intensity decline. Finally, MoE dispatch becomes a first-order "
        "concern primarily when expert parallelism (`EP`) is greater than 1."
    )
    lines.append("")
    return lines


def _term_catalog() -> List[Tuple[str, str, str, str]]:
    """Canonical glossary entries: (term, definition, units, where-used)."""
    return [
        ("F_theory", "Symbolic FLOPs from model equations before hardware mapping.", "FLOPs", "Math, KPIs"),
        ("F_tensorcore", "Tensor-core-eligible subset of FLOPs.", "FLOPs", "Math, KPIs"),
        (
            "F_realizable",
            "Peak-equivalent compute cost FLOPs after utilization model (`eta_tc`) and scalar fallback.",
            "FLOPs",
            "Math, KPIs, Roofline",
        ),
        (
            "eta_tc(B)",
            "Tensor-core utilization factor as a function of effective GEMM M dimension (decode proxy: batch).",
            "ratio",
            "Math",
        ),
        ("P_peak", "Chip peak compute throughput.", "TFLOPs", "Roofline"),
        (
            "P_effective",
            "Effective compute ceiling implied by the utilization model "
            "(`P_peak * F_theory / F_realizable`).",
            "TFLOPs",
            "KPIs",
        ),
        (
            "WRF_attn/dense/moe",
            "Weight Residency Factor(s) (WRF). We model effective streamed weight bytes as "
            "`W_eff = W / WRF`, with separate factors for attention/dense/MoE families.",
            "ratio",
            "Byte model, KPIs",
        ),
        (
            "activation_fusion_factor",
            "Scales activation/intermediate bytes to represent fewer HBM trips under fused kernels.",
            "ratio",
            "Byte model",
        ),
        (
            "elementwise_bytes_factor",
            "Scales elementwise-heavy terms (for example softmax/norm) to represent fusion and "
            "reduced temporaries.",
            "ratio",
            "Byte model",
        ),
        ("bytes_weights", "Streamed weight bytes.", "bytes", "Byte model, KPIs"),
        ("bytes_activations", "Activation input/output bytes.", "bytes", "Byte model, KPIs"),
        ("bytes_kv", "KV-cache read/write bytes.", "bytes", "Byte model, KPIs"),
        ("bytes_temporary", "Temporary/intermediate buffer bytes.", "bytes", "Byte model, KPIs"),
        ("bytes_hbm", "Total HBM bytes = weights + activations + KV + temporary.", "bytes", "Byte model"),
        ("bytes_net", "Interconnect bytes (for example MoE dispatch).", "bytes", "Byte model, KPIs"),
        ("AI_hbm", "Arithmetic intensity using HBM bytes (`FLOPs / bytes_hbm`).", "FLOPs/byte", "Roofline, KPIs"),
        ("AI_total", "Arithmetic intensity using HBM+network bytes.", "FLOPs/byte", "KPIs"),
        ("OI_knee", "Roofline ridge point (`P_peak / BW_hbm`).", "FLOPs/byte", "Roofline"),
        ("T_comp", "Compute time estimate.", "seconds", "KPIs"),
        ("T_hbm", "HBM transfer time estimate.", "seconds", "KPIs"),
        ("T_net", "Network transfer time estimate.", "seconds", "KPIs"),
        ("T_est", "Estimated step time (`max(T_comp, T_hbm, T_net)`).", "seconds", "KPIs"),
        (
            "TF_est",
            "Estimated throughput from the time model (`F_theory / T_est`).",
            "TFLOPs",
            "Roofline plots, sweeps",
        ),
    ]


def _render_key_terms_section() -> List[str]:
    """Render top glossary table for frequently used technical terms."""
    lines: List[str] = []
    lines.append("## Key Terms and Units")
    lines.append("")
    lines.append(
        "We use a small chain of definitions throughout the report. We start from `F_theory` "
        "(symbolic FLOPs derived from operator shapes, e.g. GEMM `2*M*K*N`) and a byte decomposition "
        "(`bytes_weights`, `bytes_activations`, `bytes_kv`, `bytes_temporary`). We then form "
        "`AI_hbm = F_theory / bytes_hbm` and convert it into chip ceilings via roofline "
        "(`OI_knee = P_peak / BW_hbm`). Separately, we estimate time by combining a compute cost "
        "`F_realizable` (peak-equivalent compute cost under utilization assumptions) with "
        "HBM/network transfer times and taking `T_est = max(T_comp, T_hbm, T_net)`. Finally, we "
        "report estimated throughput as `TF_est = F_theory / T_est` and plot points at "
        "(`AI_hbm`, `TF_est`)."
    )
    lines.append("")
    lines.append("| Term | Definition | Units | Where Used |")
    lines.append("|---|---|---|---|")
    for term, definition, units, where_used in _term_catalog():
        lines.append(f"| `{term}` | {definition} | `{units}` | {where_used} |")
    lines.append("")
    lines.append(
        "- Term chain: `F_theory -> F_tensorcore -> F_realizable -> AI -> "
        "roofline/time limits`."
    )
    lines.append("")
    return lines


def _render_architecture_diagrams(model: torch.nn.Module) -> List[str]:
    """
    Render model architecture diagrams as Mermaid blocks.

    We keep these diagrams intentionally schematic (paper-style block diagrams), not a full
    module-level graph, so they remain readable for very large MoE models.
    """
    lines: List[str] = []
    lines.append("### Architecture Diagrams")
    lines.append("")

    if _is_deepseek_model(model):
        cfg = getattr(model, "config", None)
        num_layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
        dense_layers, moe_layers = _deepseek_layer_counts(cfg)
        hidden = int(getattr(cfg, "hidden_size", 0) or 0)
        vocab = int(getattr(cfg, "vocab_size", 0) or 0)
        top_k = int(getattr(cfg, "num_experts_per_tok", 0) or 0)
        num_experts = int(getattr(cfg, "n_routed_experts", 0) or 0)
        n_shared_experts = int(getattr(cfg, "n_shared_experts", 0) or 0)
        ffn_dense = int(getattr(cfg, "intermediate_size", 0) or 0)
        ffn_moe = int(getattr(cfg, "moe_intermediate_size", 0) or 0)
        q_rank = int(getattr(cfg, "q_lora_rank", 0) or 0)
        kv_rank = int(getattr(cfg, "kv_lora_rank", 0) or 0)
        d_nope = int(getattr(cfg, "qk_nope_head_dim", 0) or 0)
        d_rope = int(getattr(cfg, "qk_rope_head_dim", 0) or 0)
        d_v = int(getattr(cfg, "v_head_dim", 0) or 0)
        num_heads = int(getattr(cfg, "num_attention_heads", 0) or 0)

        # DeepSeek-style decoder stack: dense stage, then MoE stage.
        emb_label = f"Token embedding<br/>V={vocab}, H={hidden}"
        dense_stage_label = (
            f"Dense decoder blocks x {dense_layers}<br/>"
            "Block = RMSNorm -> MLA -> + -> RMSNorm -> Dense FFN -> +"
        )
        moe_stage_label = (
            f"MoE decoder blocks x {moe_layers}<br/>"
            "Block = RMSNorm -> MLA -> + -> RMSNorm -> Routed MoE -> +"
        )
        lines.append("```mermaid")
        lines.append("flowchart TB")
        lines.append('    ids["Input token ids<br/>[B,S]"] --> emb["' + emb_label + '"]')
        if dense_layers > 0:
            lines.append('    emb --> dense["' + dense_stage_label + '"]')
            prev = "dense"
        else:
            prev = "emb"
        if moe_layers > 0:
            lines.append('    ' + prev + ' --> moe["' + moe_stage_label + '"]')
            prev = "moe"
        lines.append('    ' + prev + ' --> norm["Final RMSNorm"]')
        lines.append('    norm --> head["LM head / logits<br/>[B,S,V]"]')
        lines.append("```")
        lines.append("")

        # One decoder block with MLA attention and dense/MoE FFN.
        c_kv_text = f"{kv_rank}+{d_rope}" if kv_rank and d_rope else "C_kv"
        attn_label = (
            "MLA attention (per-layer)<br/>"
            f"Q: H->{q_rank}->h*({d_nope}+{d_rope})<br/>"
            f"KV cache: [B,L,({c_kv_text})]<br/>"
            f"Out: h*{d_v}->H"
        )
        ffn_label = (
            "FFN (per-layer)<br/>"
            f"Dense: H->{ffn_dense}->H<br/>"
            f"MoE: E={num_experts}, top-k={top_k}, shared={n_shared_experts}, d_moe={ffn_moe}"
        )
        lines.append("```mermaid")
        lines.append("flowchart TB")
        lines.append('    x0["x_l<br/>[B,S,H]"] --> n1["RMSNorm"]')
        lines.append('    n1 --> attn["' + attn_label + '"]')
        lines.append('    x0 --> add1["Residual add"]')
        lines.append("    attn --> add1")
        lines.append('    add1 --> n2["RMSNorm"]')
        lines.append('    n2 --> ffn["' + ffn_label + '"]')
        lines.append('    add1 --> add2["Residual add"]')
        lines.append("    ffn --> add2")
        lines.append('    add2 --> x1["x_{l+1}<br/>[B,S,H]"]')
        lines.append('    kv["KV cache (decode)<br/>[B,L,(' + c_kv_text + ')]"] --> attn')
        lines.append('    attn --> kvw["Append KV_t"]')
        lines.append("```")
        lines.append("")

        # Routed MoE dataflow (schematic).
        topk_label = f"Top-k select (k={top_k})"
        lines.append("```mermaid")
        lines.append("flowchart TB")
        lines.append('    h["Token hidden<br/>[B*S,H]"] --> r["Router logits<br/>[B*S,E]"]')
        lines.append('    r --> tk["' + topk_label + '"]')
        lines.append('    tk --> ex["Selected experts (k MLPs)<br/>H->d_moe->H"]')
        lines.append('    ex --> mix["Weighted combine"]')
        lines.append('    mix --> out["FFN out<br/>[B*S,H]"]')
        if n_shared_experts > 0:
            lines.append('    h --> sh["Shared expert(s) (n=' + str(n_shared_experts) + ')"]')
            lines.append('    sh --> out')
        lines.append("```")
        lines.append("")
        lines.append(
            "We keep these diagrams schematic: an end-to-end decoder stack, a representative "
            "decoder block (residual + MLA + FFN), and the routed MoE dataflow. Dense vs MoE FFN "
            "placement is controlled by `first_k_dense_replace` and `moe_layer_freq`."
        )
        return lines

    # Generic decoder-only transformer schematic.
    lines.append("```mermaid")
    lines.append("flowchart TB")
    lines.append("    ids[\"Input token ids<br/>[B,S]\"] --> emb[\"Token embedding\"]")
    lines.append("    emb --> stack[\"Decoder blocks x N\"]")
    lines.append("    stack --> norm[\"Final norm\"]")
    lines.append("    norm --> head[\"LM head / logits<br/>[B,S,V]\"]")
    lines.append("```")
    lines.append("")
    lines.append("```mermaid")
    lines.append("flowchart TB")
    lines.append("    x0[\"x_l<br/>[B,S,H]\"] --> n1[\"Norm\"]")
    lines.append("    n1 --> attn[\"Self-attention\"]")
    lines.append("    x0 --> add1[\"Residual add\"]")
    lines.append("    attn --> add1")
    lines.append("    add1 --> n2[\"Norm\"]")
    lines.append("    n2 --> mlp[\"FFN / MLP\"]")
    lines.append("    add1 --> add2[\"Residual add\"]")
    lines.append("    mlp --> add2")
    lines.append("    add2 --> x1[\"x_{l+1}<br/>[B,S,H]\"]")
    lines.append("```")
    return lines


_SECTION_TERM_EXPLANATIONS: Dict[str, str] = {
    "F_theory": "symbolic FLOPs from formulas",
    "F_tensorcore": "tensor-core-eligible FLOPs",
    "F_realizable": "peak-equivalent compute cost after utilization model",
    "AI_hbm": "FLOPs divided by HBM bytes",
    "AI_total": "FLOPs divided by HBM+network bytes",
    "OI_knee": "ridge intensity `P_peak / BW_hbm`",
    "bytes_hbm": "HBM byte total",
    "T_est": "estimated step time from compute/memory/network times",
}
_MAJOR_SECTION_CRITICAL_TERMS: Tuple[str, ...] = (
    "F_theory",
    "F_tensorcore",
    "F_realizable",
    "AI_hbm",
    "AI_total",
    "OI_knee",
    "bytes_hbm",
    "T_est",
)


def _term_ref(
    section_term_tracker: Dict[str, set],
    section_name: str,
    term: str,
) -> str:
    """Return first-use expanded term text within a section."""
    seen = section_term_tracker.setdefault(section_name, set())
    if term not in seen:
        seen.add(term)
        detail = _SECTION_TERM_EXPLANATIONS.get(term)
        if detail is not None:
            return f"`{term}` ({detail})"
    return f"`{term}`"


def _append_section_term_primer(
    report_lines: List[str],
    section_term_tracker: Dict[str, set],
    section_name: str,
) -> None:
    """Legacy hook retained for compatibility; primer lines are intentionally suppressed."""
    # Top glossary and in-section term usage already provide definitions.
    _ = report_lines, section_term_tracker, section_name


def _validate_report_term_hygiene(markdown_text: str) -> None:
    """Validate glossary presence and ordering of critical term introductions."""
    glossary_title = "## Key Terms and Units"
    glossary_idx = markdown_text.find(glossary_title)
    if glossary_idx < 0:
        raise ValueError("Report quality check failed: missing 'Key Terms and Units' section.")

    next_section_idx = markdown_text.find("\n## ", glossary_idx + len(glossary_title))
    glossary_block = (
        markdown_text[glossary_idx:] if next_section_idx < 0 else markdown_text[glossary_idx:next_section_idx]
    )
    required_terms = [
        "F_theory",
        "F_tensorcore",
        "F_realizable",
        "eta_tc(B)",
        "P_peak",
        "P_effective",
        "bytes_weights",
        "bytes_activations",
        "bytes_kv",
        "bytes_temporary",
        "bytes_hbm",
        "bytes_net",
        "AI_hbm",
        "AI_total",
        "OI_knee",
        "T_comp",
        "T_hbm",
        "T_net",
        "T_est",
    ]
    missing_in_glossary = [term for term in required_terms if term not in glossary_block]
    if missing_in_glossary:
        raise ValueError(
            "Report quality check failed: glossary missing terms: "
            f"{', '.join(missing_in_glossary)}"
        )

    chain_required = "F_theory -> F_tensorcore -> F_realizable -> AI -> roofline/time limits"
    if chain_required not in markdown_text:
        raise ValueError("Report quality check failed: missing top-level term chain line.")

    kpi_markers = [
        "Regime KPI Matrix (naive vs efficient)",
        "Cross-Mode Summary (naive vs efficient)",
        "Decode Sweep (vary B, L, EP)",
    ]
    marker_positions = [markdown_text.find(marker) for marker in kpi_markers if marker in markdown_text]
    if marker_positions:
        first_kpi_idx = min(marker_positions)
        critical_terms = ["F_theory", "F_tensorcore", "F_realizable", "AI_hbm", "AI_total", "OI_knee"]
        late_terms = []
        for term in critical_terms:
            idx = markdown_text.find(term)
            if idx < 0 or idx > first_kpi_idx:
                late_terms.append(term)
        if late_terms:
            raise ValueError(
                "Report quality check failed: critical terms missing/late before first KPI: "
                f"{', '.join(late_terms)}"
            )


def _render_executive_summary(
    mode_stats_by_exec: Dict[str, Dict[str, ModeKpiExtended]],
    decode_sweep_by_exec: Dict[str, List[SweepPoint]],
    ai_knee: float,
    category_totals: Dict[str, int],
    total_params: int,
) -> List[str]:
    """Render top-level executive summary with dynamic conclusions/actions."""
    lines: List[str] = []
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("### 3 Conclusions")
    lines.append("")

    eff_training = mode_stats_by_exec["efficient"]["training"]
    eff_prefill = mode_stats_by_exec["efficient"]["prefill"]
    eff_decode = mode_stats_by_exec["efficient"]["decode"]
    decode_bcrit = _b_crit_from_sweep(decode_sweep_by_exec["efficient"], ai_knee)
    experts_pct = 100.0 * _safe_div(category_totals.get("experts", 0), max(1, total_params))

    lines.append(
        f"1. In efficient mode, training is `{eff_training.regime}` and prefill is "
        f"`{eff_prefill.regime}` (`AI_hbm`: {eff_training.ai_hbm:.2f}, "
        f"{eff_prefill.ai_hbm:.2f})."
    )
    if decode_bcrit is None:
        lines.append(
            f"2. Decode remains `{eff_decode.regime}` across sampled batches; no `B_crit` "
            f"crossing was found for `OI_knee={ai_knee:.1f}`."
        )
    else:
        lines.append(
            f"2. Decode in efficient mode is `{eff_decode.regime}` at base config, with "
            f"`B_crit≈{decode_bcrit:.1f}` to approach the HBM ridge (`OI_knee={ai_knee:.1f}`)."
        )
    lines.append(
        f"3. Experts hold ~{experts_pct:.1f}% of parameters, but parameter share does not "
        "equal runtime cost share."
    )
    lines.append("")
    lines.append("### 3 Next Optimizations")
    lines.append("")

    decode_share = _compute_mode_byte_shares(eff_decode)
    if decode_share["kv"] > 0.30:
        first_opt = (
            "1. Prioritize KV traffic reduction: lower KV dtype, compact KV layout, and "
            "cache-locality-aware decode kernels."
        )
    elif decode_share["weights"] > 0.70:
        first_opt = (
            "1. Prioritize weight-stream reduction: raise effective residency (WRF), "
            "improve expert weight staging, and reduce streaming bytes."
        )
    else:
        first_opt = (
            "1. Prioritize mixed optimization: both weight and KV bytes are material; "
            "tune residency and KV format together."
        )
    lines.append(first_opt)
    lines.append(
        "2. Keep improving compute path: increase tensor-core utilization and fuse "
        "bandwidth-heavy elementwise steps."
    )
    lines.append(
        "3. For serving, optimize batching policy around `B_crit` and latency constraints "
        "instead of targeting peak TFLOPs alone."
    )
    lines.append("")
    return lines


def _render_roofline_point_walkthrough(
    mode_stats_by_exec: Dict[str, Dict[str, ModeKpiExtended]],
    ai_knee: float,
    primary_roofline: RooflineConfig,
) -> List[str]:
    """Render short dynamic roofline walkthrough examples."""
    lines: List[str] = []
    eff_decode = mode_stats_by_exec["efficient"]["decode"]
    eff_prefill = mode_stats_by_exec["efficient"]["prefill"]
    decode_bound = "memory-bound" if eff_decode.ai_hbm < ai_knee else "compute-bound"
    prefill_bound = "memory-bound" if eff_prefill.ai_hbm < ai_knee else "compute-bound"

    lines.append("### Reading a Roofline Point")
    lines.append("")
    lines.append(
        "A roofline point is positioned by its arithmetic intensity and its estimated throughput. "
        "The roofline bound is the minimum of the compute ceiling (`P_peak`) and the memory "
        "ceiling (`BW_hbm * AI_hbm`). Points left of the ridge (`AI_hbm < OI_knee`) are bounded by "
        "HBM, while points right of the ridge are bounded by compute."
    )
    lines.append(
        "This ridge-side classification is a roofline diagnostic; the `Regime` labels in KPI "
        "tables come from the time model (`T_est=max(T_comp,T_hbm,T_net)`)."
    )
    lines.append("")
    lines.append(
        f"For this run (efficient mode, primary target `{primary_roofline.name}`), decode has "
        f"`AI_hbm={eff_decode.ai_hbm:.2f}` compared to `OI_knee={ai_knee:.2f}`, so it is "
        f"`{decode_bound}` under the HBM roofline model."
    )
    lines.append(
        f"Prefill has `AI_hbm={eff_prefill.ai_hbm:.2f}` compared to `OI_knee={ai_knee:.2f}`, so it "
        f"is `{prefill_bound}` under the same assumptions."
    )
    lines.append("")
    return lines


def _render_debugging_checklist_appendix() -> List[str]:
    """Render Appendix B checklist for suspicious results."""
    lines: List[str] = []
    lines.append("## Appendix B: Common Failure Modes / Debugging Checklist")
    lines.append("")
    lines.append(
        "- Verify `T_comp` uses `F_realizable` as peak-equivalent compute cost "
        "(`T_comp = F_realizable / P_peak`)."
    )
    lines.append("- Verify WRF is applied consistently in prefill and decode paths.")
    lines.append("- Verify KV bytes are counted per layer and multiplied by layer count.")
    lines.append("- Verify temporary buffer bytes are not double-counted as activations.")
    lines.append("- Verify dtype byte assumptions are consistent across weights/acts/KV.")
    return lines


def _median(values: List[float]) -> float:
    """Compute median for a numeric list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return 0.5 * float(ordered[mid - 1] + ordered[mid])


def _rank_sensitivity_knobs(
    points: List[SensitivityPoint],
) -> List[Tuple[str, float, Dict[str, float]]]:
    """
    Rank sensitivity knobs by median `t_est_ms` spread.

    Returns tuples: (knob_name, effect_size, medians_by_value_str).
    """
    knob_fields = [
        ("kv_dtype_bytes", "KV dtype bytes"),
        ("top_k", "top-k experts"),
        ("kv_rank_scale", "KV rank scale"),
        ("hidden_scale", "hidden scale"),
        ("cache_len", "decode KV length (L)"),
    ]
    rankings: List[Tuple[str, float, Dict[str, float]]] = []
    for attr_name, display_name in knob_fields:
        groups: Dict[str, List[float]] = {}
        for point in points:
            value = getattr(point, attr_name)
            key = str(value)
            if key not in groups:
                groups[key] = []
            groups[key].append(point.t_est_ms)
        if not groups:
            continue
        medians = {key: _median(vals) for key, vals in groups.items()}
        min_med = min(medians.values())
        max_med = max(medians.values())
        effect = _safe_div(max_med, max(1e-12, min_med)) - 1.0
        rankings.append((display_name, effect, medians))
    rankings.sort(key=lambda item: item[1], reverse=True)
    return rankings


def _render_appendix_derivations(model_config) -> List[str]:
    """Render Appendix A derivation lines for markdown output."""
    hidden_size_cfg = int(getattr(model_config, "hidden_size", 0) or 0)
    num_heads_cfg = int(getattr(model_config, "num_attention_heads", 0) or 0)
    q_lora_rank = int(getattr(model_config, "q_lora_rank", 0) or 0)
    kv_lora_rank = int(getattr(model_config, "kv_lora_rank", 0) or 0)
    qk_rope_head_dim = int(getattr(model_config, "qk_rope_head_dim", 0) or 0)
    moe_dim = int(getattr(model_config, "moe_intermediate_size", 0) or 0)
    top_k_cfg = int(getattr(model_config, "num_experts_per_tok", 0) or 0)

    lines: List[str] = []
    lines.append("## Appendix A: Full FLOP Derivations")
    lines.append("")
    lines.append("### Dense Linear Layer")
    lines.append("")
    lines.append("- Input shape: `[B,S,In]`, weight shape: `[In,Out]`")
    lines.append("- `F_linear = 2 * B * S * In * Out`")
    lines.append("")
    lines.append("### MLA Attention FLOPs (Prefill)")
    lines.append("")
    lines.append("- `F_Q = 2 * B * S * H * r_q`")
    lines.append("- `F_K = 2 * B * S * H * r_kv`")
    lines.append("- `F_attn_score = 2 * B * h * S^2 * d_eff`")
    lines.append("- `d_eff = d_nope + d_rope`")
    if hidden_size_cfg > 0 and num_heads_cfg > 0:
        lines.append(
            f"- For this config: `H={hidden_size_cfg}`, `h={num_heads_cfg}`, "
            f"`r_q={q_lora_rank}`, `r_kv={kv_lora_rank}`, `d_rope={qk_rope_head_dim}`."
        )
    lines.append("")
    lines.append("### MoE FLOPs Per Token")
    lines.append("")
    lines.append("- One expert MLP: `F_expert = 6 * H * d_moe`")
    lines.append("- Routed total: `F_MoE = B * S * top_k * 6 * H * d_moe`")
    if moe_dim > 0:
        lines.append(f"- For this config: `d_moe={moe_dim}`, `top_k={top_k_cfg}`.")
    return lines


def _render_appendix_derivations_pointer() -> List[str]:
    """Render concise Appendix A pointer for the report body."""
    lines: List[str] = []
    lines.append("## Appendix A: Full FLOP Derivations")
    lines.append("")
    lines.append(
        "- Detailed derivations are documented in `docs/model_info_appendix.md` "
        "(see the derivation appendix section)."
    )
    return lines


def _render_debugging_checklist_pointer() -> List[str]:
    """Render concise Appendix B pointer for the report body."""
    lines: List[str] = []
    lines.append("## Appendix B: Common Failure Modes / Debugging Checklist")
    lines.append("")
    lines.append(
        "- Full debugging checklist is documented in `docs/model_info_appendix.md` "
        "(see the checklist appendix section)."
    )
    return lines


def dump_model_info(
    model: torch.nn.Module,
    logger=None,
    report_path: str = "outputs/model_reports/model_report.md",
    plot_distributions: bool = True,
    plot_roofline: bool = True,
    batch_size: int = 1,
    seq_len: Optional[int] = None,
    activation_bytes: int = 1,
    kv_cache_bytes: Optional[int] = None,
    param_bytes_assumed: int = 1,
    optimizer_state_bytes: int = 8,
    master_weight_bytes: int = 4,
    hbm_capacity_gb: float = 141.0,
    interconnect_bw_gbps: float = 900.0,
    roofline: Optional[RooflineConfig] = None,
    roofline_targets: Optional[List[RooflineConfig]] = None,
    training_flops_multiplier: float = 3.0,
    training_bytes_multiplier: float = 2.0,
    max_weight_samples: int = 1_000_000,
    roofline_plot_top_n: int = 20,
    decode_batch_sizes: Optional[List[int]] = None,
    prefill_seq_lengths: Optional[List[int]] = None,
    include_architecture_diagrams: bool = True,
    include_appendix_derivations: bool = True,
    tensor_core_b_sat: int = 64,
    enable_tensor_core_model: bool = True,
    sensitivity_enable: bool = True,
    sensitivity_profile: str = "medium_full_grid",
    roofline_x_limits: Optional[Tuple[float, float]] = None,
    roofline_y_limits: Optional[Tuple[float, float]] = None,
    roofline_label_mode: str = "minimal",
) -> ModelInfo:
    """
    Dump comprehensive information about a model to a Markdown report.

    Scientific-report extensions:
    - Tensor-core efficiency model (`tensor_core_b_sat`, `enable_tensor_core_model`)
    - Full-grid sensitivity analysis (`sensitivity_enable`, `sensitivity_profile`)
    - Roofline rendering controls (`roofline_x_limits`, `roofline_y_limits`, `roofline_label_mode`)
    - Architecture diagrams and derivation appendix toggles

    Returns a ModelInfo object with metadata and file paths.
    """
    def _progress(message: str) -> None:
        tagged = f"[dump_model_info] {message}"
        if logger is not None:
            try:
                logger.info(tagged)
                return
            except Exception:
                pass
        print(tagged)

    _progress("Starting model report generation.")

    if roofline_targets is None:
        roofline_targets = default_roofline_targets()
    if not roofline_targets:
        roofline_targets = default_roofline_targets()
    primary_roofline = roofline if roofline is not None else roofline_targets[0]
    if decode_batch_sizes is None:
        decode_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    if kv_cache_bytes is None:
        kv_cache_bytes = activation_bytes
    param_bytes_assumed = max(1, int(param_bytes_assumed))
    kv_cache_bytes = max(1, int(kv_cache_bytes))
    activation_bytes = max(1, int(activation_bytes))
    optimizer_state_bytes = max(0, int(optimizer_state_bytes))
    master_weight_bytes = max(0, int(master_weight_bytes))
    if roofline_label_mode not in {"minimal", "full"}:
        raise ValueError(
            "roofline_label_mode must be one of {'minimal', 'full'}"
        )
    tc_cfg = TensorCoreModelConfig(
        enabled=bool(enable_tensor_core_model),
        b_sat=max(1, int(tensor_core_b_sat)),
    )
    tc_cfg.validate()
    sensitivity_cfg = _sensitivity_config_from_profile(sensitivity_profile)

    if seq_len is None:
        if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings"):
            seq_len = int(model.config.max_position_embeddings)
        else:
            seq_len = 2048
    if prefill_seq_lengths is None:
        prefill_seq_lengths = _default_prefill_seq_lengths(model, seq_len)

    if report_path is None or report_path.strip() == "":
        report_path = "outputs/model_reports/model_report.md"
    requested_report_path = report_path
    report_path = _ensure_unique_path(requested_report_path)
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    if report_path != requested_report_path:
        os.makedirs(os.path.dirname(requested_report_path) or ".", exist_ok=True)

    _progress("Collecting parameter tensors and statistics...")
    layers = _collect_layer_info(model)
    _progress(f"Collected {len(layers)} parameter tensors.")
    has_meta_params = any(getattr(param, "is_meta", False) for param in model.parameters())
    total_params = sum(layer.num_params for layer in layers)
    trainable_params = sum(layer.num_params for layer in layers if layer.requires_grad)
    non_trainable_params = total_params - trainable_params
    total_memory_mb = sum(layer.memory_mb for layer in layers)

    model_info = ModelInfo(
        total_params=total_params,
        trainable_params=trainable_params,
        non_trainable_params=non_trainable_params,
        total_memory_mb=total_memory_mb,
        layers=layers,
        num_layers=len(layers),
        report_path=report_path,
    )

    architecture = _infer_architecture(model)
    module_sizes = _aggregate_module_sizes(layers)
    module_patterns = _aggregate_module_patterns(module_sizes)
    _progress("Inferred architecture and module size breakdown.")

    parallelism = _infer_parallelism_assumptions(
        model=model,
        target_routed_experts_per_gpu=4,
    )
    tp_size_assumed = parallelism.tp_size
    ep_size_assumed = parallelism.ep_size

    def summarize_mode(mode: str, entries_for_mode: List[EfficiencyEntry]) -> ModeKpiExtended:
        return _summarize_mode_entries(
            mode=mode,
            entries_for_mode=entries_for_mode,
            roofline=primary_roofline,
            interconnect_bw_gbps=interconnect_bw_gbps,
        )

    def compute_mode_category_shares(
        efficiency_values: Dict[str, List[EfficiencyEntry]],
    ) -> Dict[str, Dict[str, Tuple[float, float]]]:
        shares: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for mode in ["training", "prefill", "decode"]:
            mode_entries = efficiency_values.get(mode, [])
            total_flops = sum(entry.flops_theory for entry in mode_entries)
            total_bytes = sum(entry.bytes_total for entry in mode_entries)
            grouped: Dict[str, Dict[str, float]] = {}
            for entry in mode_entries:
                category = _categorize_efficiency_entry(entry)
                if category not in grouped:
                    grouped[category] = {"flops": 0.0, "bytes": 0.0}
                grouped[category]["flops"] += entry.flops_theory
                grouped[category]["bytes"] += entry.bytes_total

            shares[mode] = {}
            for category, values in grouped.items():
                flops_share = _safe_div(values["flops"], total_flops) * 100.0
                bytes_share = _safe_div(values["bytes"], total_bytes) * 100.0
                shares[mode][category] = (flops_share, bytes_share)
        return shares

    mem_bw_bytes = primary_roofline.mem_bw_gbps * 1e9
    net_bw_bytes = interconnect_bw_gbps * 1e9
    ai_knee = _safe_div(primary_roofline.peak_tflops * 1e12, mem_bw_bytes)
    net_knee = _safe_div(primary_roofline.peak_tflops * 1e12, net_bw_bytes)
    execution_models = default_execution_models()
    module_weight_bytes_cache: Dict[int, float] = {}
    efficiency_by_exec: Dict[str, Dict[str, List[EfficiencyEntry]]] = {}
    decode_workload_points_by_exec: Dict[str, List[DecodeWorkloadPoint]] = {}
    prefill_workload_points_by_exec: Dict[str, List[PrefillWorkloadPoint]] = {}
    mode_category_shares_by_exec: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {}
    _progress(
        "Estimating static efficiency for execution modes: "
        + ", ".join(mode.name for mode in execution_models)
    )
    for execution_model in execution_models:
        execution_model.validate()
        _progress(f"Estimating base mode metrics (exec={execution_model.name})...")
        efficiency = _estimate_efficiency(
            model=model,
            batch_size=batch_size,
            seq_len=seq_len,
            activation_bytes=activation_bytes,
            kv_cache_bytes=kv_cache_bytes,
            param_bytes_assumed=param_bytes_assumed,
            roofline=primary_roofline,
            interconnect_bw_gbps=interconnect_bw_gbps,
            training_flops_multiplier=training_flops_multiplier,
            training_bytes_multiplier=training_bytes_multiplier,
            exec_model=execution_model,
            tc_cfg=tc_cfg,
            module_weight_bytes_cache=module_weight_bytes_cache,
            ep_size_override=ep_size_assumed,
            tp_size_override=tp_size_assumed,
        )
        efficiency_by_exec[execution_model.name] = efficiency
        mode_category_shares_by_exec[execution_model.name] = compute_mode_category_shares(efficiency)
    _progress("Execution-mode efficiency estimates complete.")

    max_position_embeddings: Optional[int] = None
    if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings"):
        try:
            max_position_embeddings = int(getattr(model.config, "max_position_embeddings"))
        except Exception:
            max_position_embeddings = None
        if max_position_embeddings is not None and max_position_embeddings <= 0:
            max_position_embeddings = None

    decode_cache_lengths = _default_decode_cache_lengths(seq_len, max_position_embeddings)
    ep_sweep_sizes = _default_ep_sweep_sizes(parallelism.n_routed_experts, ep_size_assumed)
    prefill_batch_sizes = _default_prefill_batch_sizes(batch_size)

    _progress(
        "Running merged decode workload sweep "
        f"(B={len(decode_batch_sizes)}, L={len(decode_cache_lengths)}, EP={len(ep_sweep_sizes)})..."
    )
    decode_workload_points_by_exec = _run_decode_workload_sweep(
        model=model,
        execution_models=execution_models,
        batch_sizes=decode_batch_sizes,
        cache_lengths=decode_cache_lengths,
        ep_sizes=ep_sweep_sizes,
        activation_bytes=activation_bytes,
        kv_cache_bytes=kv_cache_bytes,
        param_bytes_assumed=param_bytes_assumed,
        roofline=primary_roofline,
        interconnect_bw_gbps=interconnect_bw_gbps,
        training_flops_multiplier=training_flops_multiplier,
        training_bytes_multiplier=training_bytes_multiplier,
        tc_cfg=tc_cfg,
        module_weight_bytes_cache=module_weight_bytes_cache,
        progress_cb=_progress,
    )

    _progress(
        "Running merged prefill workload sweep "
        f"(B={len(prefill_batch_sizes)}, S={len(prefill_seq_lengths)}, EP={len(ep_sweep_sizes)})..."
    )
    prefill_workload_points_by_exec = _run_prefill_workload_sweep(
        model=model,
        execution_models=execution_models,
        batch_sizes=prefill_batch_sizes,
        seq_lengths=prefill_seq_lengths,
        ep_sizes=ep_sweep_sizes,
        activation_bytes=activation_bytes,
        kv_cache_bytes=kv_cache_bytes,
        param_bytes_assumed=param_bytes_assumed,
        roofline=primary_roofline,
        interconnect_bw_gbps=interconnect_bw_gbps,
        training_flops_multiplier=training_flops_multiplier,
        training_bytes_multiplier=training_bytes_multiplier,
        tc_cfg=tc_cfg,
        module_weight_bytes_cache=module_weight_bytes_cache,
        progress_cb=_progress,
    )

    decode_frontier_md = ""
    decode_frontier_metrics: Dict[Tuple[int, int], Dict[str, float]] = {}
    decode_frontier_cfg = DecodeEPExplorationConfig(
        ep_sizes=ep_sweep_sizes,
        cache_lengths=decode_cache_lengths,
        alpha=0.9,
        max_batch_per_gpu=16384,
        weight_residency_factor=1.0,
        kv_element_bytes=kv_cache_bytes,
    )
    try:
        decode_frontier_md, decode_frontier_metrics = _collect_decode_ep_exploration(
            model=model,
            roofline=primary_roofline,
            activation_bytes=activation_bytes,
            kv_cache_bytes=kv_cache_bytes,
            interconnect_bw_gbps=interconnect_bw_gbps,
            param_bytes_assumed=param_bytes_assumed,
            cfg=decode_frontier_cfg,
        )
    except Exception:
        # Keep report generation robust; the decode sweep subsection handles empty metrics.
        decode_frontier_md = ""
        decode_frontier_metrics = {}

    # Preserve a 1D decode batch slice at the anchor (L=seq_len, EP=ep_size_assumed) for
    # executive-summary `B_crit` and a worked example, while the report body uses the merged
    # 3-axis sweep presentation.
    decode_sweep_by_exec: Dict[str, List[SweepPoint]] = {}
    for exec_name, points in decode_workload_points_by_exec.items():
        slice_points = [
            point
            for point in points
            if point.ep_size == ep_size_assumed
            and point.cache_len == seq_len
            and point.batch in decode_batch_sizes
            and point.ai_hbm > 0.0
            and point.t_est_ms > 0.0
        ]
        sweep_points: List[SweepPoint] = []
        for point in slice_points:
            roofline_tflops_hbm = min(
                primary_roofline.peak_tflops,
                primary_roofline.mem_bw_gbps * point.ai_hbm / 1e3,
            )
            sweep_points.append(
                SweepPoint(
                    x=point.batch,
                    flops=point.flops,
                    bytes_hbm=point.bytes_hbm,
                    bytes_net=point.bytes_net,
                    bytes_total=point.bytes_total,
                    ai_hbm=point.ai_hbm,
                    ai_total=point.ai_total,
                    roofline_tflops_hbm=roofline_tflops_hbm,
                    regime_hbm=point.regime,
                    t_comp_ms=point.t_comp_ms,
                    t_hbm_ms=point.t_hbm_ms,
                    t_net_ms=point.t_net_ms,
                    t_est_ms=point.t_est_ms,
                    flops_realizable=point.flops_realizable,
                )
            )
        decode_sweep_by_exec[exec_name] = sorted(sweep_points, key=lambda p: p.x)

    roofline_points_ai: List[float] = []
    roofline_points_tf: List[float] = []
    for exec_name, efficiency in efficiency_by_exec.items():
        for mode in ["training", "prefill", "decode"]:
            mode_kpi = summarize_mode(mode, efficiency.get(mode, []))
            if mode_kpi.ai_hbm > 0.0 and mode_kpi.t_est > 0.0:
                roofline_points_ai.append(mode_kpi.ai_hbm)
                roofline_points_tf.append(mode_kpi.flops_theory / (mode_kpi.t_est * 1e12))
    for exec_name, points in decode_workload_points_by_exec.items():
        for point in points:
            if point.ep_size != ep_size_assumed:
                continue
            if point.ai_hbm > 0.0 and point.t_est_ms > 0.0:
                roofline_points_ai.append(point.ai_hbm)
                roofline_points_tf.append(point.flops / ((point.t_est_ms / 1000.0) * 1e12))
    for exec_name, points in prefill_workload_points_by_exec.items():
        for point in points:
            if point.ep_size != ep_size_assumed:
                continue
            if point.ai_hbm > 0.0 and point.t_est_ms > 0.0:
                roofline_points_ai.append(point.ai_hbm)
                roofline_points_tf.append(point.flops / ((point.t_est_ms / 1000.0) * 1e12))
    resolved_roofline_x_limits, resolved_roofline_y_limits = _resolve_roofline_axis_limits(
        roofline_targets=roofline_targets,
        point_intensities=roofline_points_ai,
        point_tflops=roofline_points_tf,
        requested_x_limits=roofline_x_limits,
        requested_y_limits=roofline_y_limits,
    )
    plot_paths: List[str] = []
    pie_plot_path: Optional[str] = None
    roofline_summary_path: Optional[str] = None
    decode_bl_roofline_path: Optional[str] = None
    decode_frontier_heatmap_path: Optional[str] = None
    prefill_bs_roofline_path: Optional[str] = None
    category_roofline_paths: Dict[str, str] = {}
    mode_stack_paths: Dict[str, str] = {}
    report_dir = os.path.dirname(report_path) or "."
    base_name = os.path.splitext(os.path.basename(report_path))[0]
    sensitivity_points_by_exec: Dict[str, List[SensitivityPoint]] = {}
    sensitivity_csv_path: Optional[str] = None
    if sensitivity_enable:
        _progress(
            "Running sensitivity analysis "
            f"(profile={sensitivity_cfg.name}, full combinational grid)..."
        )
        sensitivity_points_by_exec = _run_sensitivity_analysis(
            model=model,
            execution_models=execution_models,
            sensitivity_cfg=sensitivity_cfg,
            batch_size=batch_size,
            seq_len=seq_len,
            ep_size_override=ep_size_assumed,
            activation_bytes=activation_bytes,
            kv_cache_bytes=kv_cache_bytes,
            param_bytes_assumed=param_bytes_assumed,
            roofline=primary_roofline,
            interconnect_bw_gbps=interconnect_bw_gbps,
            training_flops_multiplier=training_flops_multiplier,
            training_bytes_multiplier=training_bytes_multiplier,
            tc_cfg=tc_cfg,
            module_weight_bytes_cache=module_weight_bytes_cache,
            progress_cb=_progress,
        )
        sensitivity_csv = os.path.join(report_dir, f"{base_name}_sensitivity.csv")
        sensitivity_csv_path = _write_sensitivity_csv(
            points_by_exec=sensitivity_points_by_exec,
            output_path=sensitivity_csv,
        )
        _progress("Sensitivity analysis complete.")

    if plot_roofline:
        _progress("Generating roofline plots...")
        plot_name = f"{base_name}_roofline.png"
        plot_path = os.path.join(report_dir, plot_name)
        out_path = _plot_roofline_summary(
            efficiency_by_exec,
            primary_roofline,
            roofline_targets,
            output_path=plot_path,
            batch_size=batch_size,
            seq_len=seq_len,
            activation_bytes=activation_bytes,
            kv_cache_bytes=kv_cache_bytes,
            roofline_x_limits=resolved_roofline_x_limits,
            roofline_y_limits=resolved_roofline_y_limits,
            roofline_label_mode=roofline_label_mode,
            interconnect_bw_gbps=interconnect_bw_gbps,
        )
        if out_path:
            roofline_summary_path = out_path
            plot_paths.append(out_path)

        decode_bl_plot_name = f"{base_name}_decode_bl_roofline.png"
        decode_bl_plot_path = os.path.join(report_dir, decode_bl_plot_name)
        decode_bl_out = _plot_decode_bl_roofline(
            points_by_exec=decode_workload_points_by_exec,
            primary_roofline=primary_roofline,
            roofline_targets=roofline_targets,
            output_path=decode_bl_plot_path,
            anchor_ep_size=ep_size_assumed,
            cache_lengths=decode_cache_lengths,
            batch_sizes=decode_batch_sizes,
            activation_bytes=activation_bytes,
            kv_cache_bytes=kv_cache_bytes,
            routed_experts_per_gpu=parallelism.routed_experts_per_gpu,
            roofline_x_limits=resolved_roofline_x_limits,
            roofline_y_limits=resolved_roofline_y_limits,
            roofline_label_mode=roofline_label_mode,
        )
        if decode_bl_out:
            decode_bl_roofline_path = decode_bl_out
            plot_paths.append(decode_bl_out)

        decode_frontier_plot_name = f"{base_name}_decode_ep_l_minbg.png"
        decode_frontier_plot_path = os.path.join(report_dir, decode_frontier_plot_name)
        decode_frontier_out = _plot_decode_ep_l_minbg_heatmap(
            exploration_metrics=decode_frontier_metrics,
            ep_sizes=ep_sweep_sizes,
            cache_lengths=decode_cache_lengths,
            primary_roofline=primary_roofline,
            output_path=decode_frontier_plot_path,
            alpha=decode_frontier_cfg.alpha,
        )
        if decode_frontier_out:
            decode_frontier_heatmap_path = decode_frontier_out
            plot_paths.append(decode_frontier_out)

        prefill_bs_plot_name = f"{base_name}_prefill_bs_roofline.png"
        prefill_bs_plot_path = os.path.join(report_dir, prefill_bs_plot_name)
        prefill_bs_out = _plot_prefill_bs_roofline(
            points_by_exec=prefill_workload_points_by_exec,
            primary_roofline=primary_roofline,
            roofline_targets=roofline_targets,
            output_path=prefill_bs_plot_path,
            anchor_ep_size=ep_size_assumed,
            batch_sizes=prefill_batch_sizes,
            activation_bytes=activation_bytes,
            kv_cache_bytes=kv_cache_bytes,
            routed_experts_per_gpu=parallelism.routed_experts_per_gpu,
            roofline_x_limits=resolved_roofline_x_limits,
            roofline_y_limits=resolved_roofline_y_limits,
            roofline_label_mode=roofline_label_mode,
        )
        if prefill_bs_out:
            prefill_bs_roofline_path = prefill_bs_out
            plot_paths.append(prefill_bs_out)

        for mode in ["training", "prefill", "decode"]:
            category_plot_name = f"{base_name}_{mode}_category_roofline.png"
            category_plot_path = os.path.join(report_dir, category_plot_name)
            category_out = _plot_category_roofline(
                efficiencies_by_exec=efficiency_by_exec,
                primary_roofline=primary_roofline,
                roofline_targets=roofline_targets,
                output_path=category_plot_path,
                mode=mode,
                interconnect_bw_gbps=interconnect_bw_gbps,
                roofline_x_limits=resolved_roofline_x_limits,
                roofline_y_limits=resolved_roofline_y_limits,
                roofline_label_mode=roofline_label_mode,
            )
            if category_out is not None:
                category_roofline_paths[mode] = category_out
                plot_paths.append(category_out)

        for execution_model in execution_models:
            mode_name = execution_model.name
            stack_plot_path = os.path.join(report_dir, f"{base_name}_{mode_name}_mode_category_stacks.png")
            stack_out = _plot_mode_category_stacks(
                mode_category_shares_by_exec[mode_name],
                stack_plot_path,
            )
            if stack_out:
                mode_stack_paths[mode_name] = stack_out
                plot_paths.append(stack_out)

    category_totals = _aggregate_module_categories(module_sizes)
    _progress("Generating parameter distribution plot...")
    pie_path = os.path.join(report_dir, f"{base_name}_module_pie.png")
    pie_out = _plot_param_pie(category_totals, pie_path)
    if pie_out:
        pie_plot_path = pie_out
        plot_paths.append(pie_out)

    if has_meta_params and plot_distributions:
        # Weight values are unavailable for meta tensors; skip histogram generation.
        plot_distributions = False
        if logger is not None:
            try:
                logger.info("Meta parameters detected: skipping weight distribution histogram.")
            except Exception:
                pass

    if plot_distributions:
        _progress("Sampling weights for histogram...")
        weights_sample = _collect_weights_sample(layers, model, max_weight_samples)
        if weights_sample.numel() > 0:
            _progress("Generating weight distribution histogram...")
            hist_path = os.path.join(report_dir, f"{base_name}_weights_hist.png")
            out_path = _plot_weight_histogram(weights_sample, hist_path)
            if out_path:
                plot_paths.append(out_path)

    model_info.plot_paths = plot_paths
    mode_stats_by_exec: Dict[str, Dict[str, ModeKpiExtended]] = {}
    tokens_per_second_by_exec: Dict[str, Dict[str, float]] = {}
    for execution_model in execution_models:
        exec_name = execution_model.name
        efficiency = efficiency_by_exec[exec_name]
        mode_stats_by_exec[exec_name] = {}
        tokens_per_second_by_exec[exec_name] = {}
        for mode in ["training", "prefill", "decode"]:
            mode_entries = efficiency.get(mode, [])
            stats = summarize_mode(mode, mode_entries)
            tokens_per_step = batch_size * seq_len if mode in {"training", "prefill"} else batch_size
            tokens_per_second = _safe_div(tokens_per_step, stats.t_est)
            tokens_per_second_by_exec[exec_name][mode] = tokens_per_second
            mode_stats_by_exec[exec_name][mode] = stats

    _progress("Assembling markdown report content...")
    _progress("Rendering header and model fingerprint sections...")
    report_lines: List[str] = []
    report_lines.append("# Model Report")
    report_lines.append("")
    report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    section_term_tracker: Dict[str, set] = {}
    report_lines.extend(_render_how_to_read_section())
    report_lines.extend(_render_key_terms_section())
    report_lines.extend(
        _render_executive_summary(
            mode_stats_by_exec=mode_stats_by_exec,
            decode_sweep_by_exec=decode_sweep_by_exec,
            ai_knee=ai_knee,
            category_totals=category_totals,
            total_params=total_params,
        )
    )

    report_lines.append("## Architecture Overview")
    report_lines.append("")
    arch_section = "Architecture Overview"
    _append_section_preamble(
        report_lines,
        why_lines=[
            "We begin with architecture because it determines where FLOPs and bytes come from "
            "before any kernel or system tuning.",
            "We focus on MLA and MoE because they reshape attention KV traffic, parameter "
            "concentration, and (optionally) dispatch behavior.",
            "These choices largely determine whether training, prefill, and decode are compute-, "
            "HBM-, or network-limited on a given chip.",
        ],
        look_for_lines=[
            "how parameters concentrate in experts (drives weight-residency priorities)",
            "how MLA sets KV elements per token (drives decode KV bandwidth)",
            "how the dense-vs-MoE layer mix shifts where bytes and FLOPs concentrate",
        ],
    )
    _append_section_term_primer(
        report_lines=report_lines,
        section_term_tracker=section_term_tracker,
        section_name=arch_section,
    )
    report_lines.append(
        "> **Callout:** Parameter distribution != runtime cost distribution."
    )
    report_lines.append("")
    report_lines.append("### Model Fingerprint")
    report_lines.append("")
    report_lines.append("| Property | Value |")
    report_lines.append("|---|---|")
    report_lines.append(f"| Family | `{architecture['family']}` |")
    report_lines.append(f"| Attention | `{architecture['attention_type']}` |")
    report_lines.append(f"| Position Encoding | `{architecture['position_encoding']}` |")
    report_lines.append(f"| Normalization | `{architecture['normalization']}` |")
    report_lines.append(f"| Activation | `{architecture['activation']}` |")
    report_lines.append(f"| MoE | `{architecture['moe']}` |")
    report_lines.append(f"| Weight Tying | `{architecture['weight_tying']}` |")

    if hasattr(model, "config"):
        cfg = model.config
        config_keys = [
            "hidden_size",
            "num_hidden_layers",
            "num_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "moe_intermediate_size",
            "n_routed_experts",
            "n_shared_experts",
            "num_experts_per_tok",
            "first_k_dense_replace",
            "moe_layer_freq",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "scoring_func",
            "n_group",
            "topk_group",
            "vocab_size",
            "max_position_embeddings",
            "dropout",
        ]
        config_rows: List[Tuple[str, object]] = []
        for key in config_keys:
            if hasattr(cfg, key):
                value = getattr(cfg, key)
                if value is not None:
                    config_rows.append((key, value))

        if config_rows:
            report_lines.append("")
            report_lines.append("**Config**")
            report_lines.append("")
            report_lines.append("| Property | Value |")
            report_lines.append("|---|---|")
            for key, value in config_rows:
                report_lines.append(f"| `{key}` | `{value}` |")

    if include_architecture_diagrams:
        report_lines.append("")
        report_lines.extend(_render_architecture_diagrams(model))

    report_lines.append("")
    report_lines.append("### Parameter & Memory Summary")
    report_lines.append("")
    assumed_param_memory_mb = (total_params * param_bytes_assumed) / (1024 ** 2)
    report_lines.append(
        "We summarize parameter counts and memory in two complementary views: the instantiated "
        "tensor-dtype footprint (from parameter tensors) and the assumed parameter-byte setting "
        "used consistently by the analytic byte model."
    )
    report_lines.append("")
    report_lines.append("| Metric | Value |")
    report_lines.append("|---|---:|")
    report_lines.append(f"| Total parameters | `{_format_number(total_params)}` |")
    report_lines.append(f"| Trainable parameters | `{_format_number(trainable_params)}` |")
    report_lines.append(f"| Non-trainable parameters | `{_format_number(non_trainable_params)}` |")
    report_lines.append(f"| Parameter memory (tensor dtypes) | `{total_memory_mb:.2f} MB` |")
    report_lines.append(
        f"| Parameter memory (assumed, {param_bytes_assumed}-byte params) | "
        f"`{assumed_param_memory_mb:.2f} MB` |"
    )
    if has_meta_params:
        report_lines.append("")
        report_lines.append(
            "Note: parameters are meta-initialized, so memory is estimated from tensor shapes and dtypes."
        )
    if pie_plot_path is not None:
        report_lines.append("")
        report_lines.append("**Parameter Category Breakdown**")
        report_lines.append("")
        report_lines.append(f"![]({os.path.basename(pie_plot_path)})")

    report_lines.append("")
    report_lines.append("### Module Size Breakdown")
    report_lines.append("")
    report_lines.append(
        "Pattern-based view groups repeated layer indices (for example, "
        "`blocks.*.attn.out_proj`) to avoid repetitive rows."
    )
    report_lines.append("")
    report_lines.append(
        "| Module Pattern | Instances | Params / Instance | Total Params | % Total | "
        "Total Memory (MB) | Example Module |"
    )
    report_lines.append("|---|---:|---:|---:|---:|---:|---|")
    for pattern in module_patterns[:20]:
        report_lines.append(
            f"| `{pattern.pattern}` | {pattern.instance_count} | "
            f"{_format_number(pattern.params_per_instance)} | "
            f"{_format_number(pattern.total_params)} | {pattern.percent_total:.2f}% | "
            f"{pattern.total_memory_mb:.2f} | `{pattern.example_module}` |"
        )
    if len(module_patterns) > 20:
        report_lines.append(f"... {len(module_patterns) - 20} more patterns")

    report_lines.append("")
    report_lines.append("### Weight Statistics (Top 20 Layers)")
    report_lines.append("")
    if has_meta_params:
        report_lines.append(
            "- Weight value statistics are unavailable for meta-initialized parameters "
            "(no backing storage)."
        )
    else:
        report_lines.append("| Layer | Mean | Std | Min | Max |")
        report_lines.append("|---|---:|---:|---:|---:|")
        for layer in layers[:20]:
            report_lines.append(
                f"| `{layer.name}` | {layer.mean:.6f} | {layer.std:.6f} | "
                f"{layer.min:.6f} | {layer.max:.6f} |"
            )
        if len(layers) > 20:
            report_lines.append(f"... {len(layers) - 20} more layers")

    _progress("Rendering static efficiency section...")
    report_lines.append("")
    report_lines.append("## Analytical Model (FLOPs, Bytes, Time)")
    report_lines.append("")
    analysis_section = "Analytical Model"
    _append_section_preamble(
        report_lines,
        why_lines=[
            "We define an analytical model that turns architecture into symbolic FLOPs, explicit "
            "bytes, and a static time estimate.",
            "We separate algorithmic work (`F_theory`) from a utilization-aware compute cost "
            "(`F_realizable`) and combine compute, HBM, and network times into `T_est` so later "
            "TFLOPs are not misread as measured performance.",
            "All roofline points and regime tables in the report are derived from this chain, so "
            "this section is the contract and audit trail for the numbers that follow.",
        ],
        look_for_lines=[
            "the full definition chain "
            "`F_theory -> F_tensorcore -> F_realizable -> (AI_hbm, T_est) -> TF_est`",
            "execution assumptions (WRF, fusion, flash attention) that change bytes without "
            "changing algorithms",
            "byte accounting and dominance tests that drive optimization conclusions",
        ],
    )
    _append_section_term_primer(
        report_lines=report_lines,
        section_term_tracker=section_term_tracker,
        section_name=analysis_section,
    )
    report_lines.append("### Modeling intent and scope")
    report_lines.append("")
    report_lines.append(
        "The model is static: it does not ingest runtime profiler timelines, scheduler queues, or "
        "overlap traces. Its role is to rank bottlenecks and optimization priorities, not to claim "
        "measured peak attainment. The core mapping is "
        f"{_term_ref(section_term_tracker, analysis_section, 'F_theory')} -> "
        f"{_term_ref(section_term_tracker, analysis_section, 'F_tensorcore')} -> "
        f"{_term_ref(section_term_tracker, analysis_section, 'F_realizable')}, combined with "
        f"{_term_ref(section_term_tracker, analysis_section, 'bytes_hbm')} decomposition and "
        f"{_term_ref(section_term_tracker, analysis_section, 'AI_hbm')} / "
        f"{_term_ref(section_term_tracker, analysis_section, 'AI_total')} to determine roofline and "
        f"timing outcomes. We use "
        f"{_term_ref(section_term_tracker, analysis_section, 'AI_hbm')} relative to "
        f"{_term_ref(section_term_tracker, analysis_section, 'OI_knee')} to describe which side "
        "of the roofline ridge a point lies on, but the `Regime` labels in KPI tables come from "
        f"the time model ({_term_ref(section_term_tracker, analysis_section, 'T_est')} as "
        "`max(T_comp, T_hbm, T_net)`)."
    )
    report_lines.append("")
    report_lines.append(
        "**Sanity checks.** As batch `B` grows, GEMM FLOPs scale with `B` while some weight bytes "
        "are amortized, so arithmetic intensity typically rises. In decode, KV read bytes scale "
        "with `L`; once KV dominates, intensity tends toward a `~1/L` decline."
    )
    report_lines.append("")
    report_lines.append("### Notation")
    report_lines.append("")
    report_lines.append("| Symbol | Meaning |")
    report_lines.append("|---|---|")
    report_lines.append("| `B` | microbatch size per GPU |")
    report_lines.append("| `S` | prefill sequence length |")
    report_lines.append("| `L` | decode KV-cache length |")
    report_lines.append("| `H` | hidden size |")
    report_lines.append("| `h` | attention head count |")
    report_lines.append("| `In`, `Out` | linear input/output channel size |")
    report_lines.append("| `D` | embedding width |")
    report_lines.append("| `W` | raw module weight bytes |")
    report_lines.append("| `W_eff = W / WRF` | effective streamed weight bytes |")
    report_lines.append("| `A` | activation bytes/element |")
    report_lines.append("| `A_kv` | KV-cache bytes/element |")
    report_lines.append("| `C_kv` | KV-cache elements per token |")
    report_lines.append("| `F_theory` | mathematical FLOPs from symbolic formulas |")
    report_lines.append("| `F_tensorcore` | tensor-core-eligible FLOPs subset |")
    report_lines.append(
        "| `F_realizable` | peak-equivalent compute cost after utilization/scalar-efficiency model |"
    )
    report_lines.append(
        "| `eta_tc(B)=min(1,M_eff/B_sat)` | tensor-core saturation factor (proxy: decode uses `M_eff≈B`; dense prefill/training uses `M_eff≈B*S`) |"
    )
    report_lines.append(
        "| `P_effective = P_peak * F_theory / F_realizable` | "
        "effective compute ceiling implied by utilization model |"
    )
    report_lines.append("| `AI_hbm = FLOPs / bytes_hbm` | HBM arithmetic intensity |")
    report_lines.append("| `AI_total = FLOPs / (bytes_hbm + bytes_net)` | end-to-end intensity |")
    report_lines.append("| `T_comp = F_realizable / P_peak` | compute time estimate |")
    report_lines.append("| `T_hbm = bytes_hbm / BW_hbm` | HBM time estimate |")
    report_lines.append("| `T_net = bytes_net / BW_net` | network time estimate |")
    report_lines.append("| `T_est = max(T_comp, T_hbm, T_net)` | step-time upper bound |")
    report_lines.append("")
    report_lines.append("### Execution / Kernel Assumptions")
    report_lines.append("")
    report_lines.append(
        "These are conservative knobs for sensitivity analysis, not claims about measured kernel reuse."
    )
    report_lines.append("")
    report_lines.append(
        "| exec_model | attention bytes | WRF attn/dense/moe | act fusion | elementwise |"
    )
    report_lines.append("|---|---|---|---:|---:|")
    for execution_model in execution_models:
        report_lines.append(
            f"| `{execution_model.name}` | `{execution_model.attention_bytes_model}` | "
            f"`{execution_model.weight_residency_attn:.2f}/"
            f"{execution_model.weight_residency_dense:.2f}/"
            f"{execution_model.weight_residency_moe:.2f}` | "
            f"{execution_model.activation_fusion_factor:.2f} | "
            f"{execution_model.elementwise_bytes_factor:.2f} |"
        )
    report_lines.append("")
    report_lines.append("#### Knob Semantics and Rationale")
    report_lines.append("")
    report_lines.append(
        "We introduce a small set of execution-mode knobs to express how kernel families change "
        "HBM traffic without changing the underlying algorithm. `WRF` (Weight Residency Factor) "
        "models effective streamed weights as `W_eff = W / WRF`, and we use separate factors for "
        "attention/dense/MoE because reuse differs by module family. `act fusion` scales "
        "activation/intermediate bytes to represent fewer HBM trips under fused kernels, while "
        "`elementwise` scales elementwise-heavy terms (softmax/norm/masking temporaries). Finally, "
        "`attention bytes` selects the attention byte path: `naive` materializes score/prob "
        "matrices in HBM, while `flash` removes most `SxS` temporary traffic while still counting "
        "explicit KV reads and writes."
    )
    report_lines.append("")
    report_lines.append(
        "These are conservative knobs for sensitivity analysis, not claims about measured kernel "
        "reuse. We treat `naive` as a pessimistic baseline (`WRF=1`, no fusion) and `efficient` as "
        "a conservative approximation of common optimizations (partial residency + fusion). If you "
        "have measured counters in your serving/training stack, tune these factors to match "
        "observed bytes and regenerate the report."
    )
    report_lines.append("")
    report_lines.append("### Per-Module Formulas")
    report_lines.append("")
    report_lines.append(
        "We use GEMM as the primitive building block: multiplying `[M,K] @ [K,N]` costs "
        "`2*M*K*N` FLOPs. For any module, we combine its FLOP model with its byte model to form "
        "`AI_hbm`, and we call a point compute-favorable when `AI_hbm >= OI_knee` "
        "(with `OI_knee = P_peak / BW_hbm`)."
    )
    report_lines.append("")
    report_lines.append(
        "| Module | Shape Explanation | Sample Torch | FLOPs | Bytes (HBM, naive) | "
        "Native AI | Efficient AI | Note |"
    )
    report_lines.append("|---|---|---|---|---|---|---|---|")
    report_lines.append(
        "| Linear prefill | `X:[B,S,In] @ W:[In,Out] -> Y:[B,S,Out]` | "
        "`y = x @ w` | "
        "`2*B*S*In*Out` | `W + A*(B*S*In + B*S*Out)` | "
        "`F / Bytes_naive` | `F / (W/WRF_dense + A*act_fusion*(B*S*In + B*S*Out))` |"
        " `Usually modest vs native unless WRF or fusion is high` |"
    )
    report_lines.append(
        "| Linear decode | `X:[B,In] @ W:[In,Out] -> Y:[B,Out]` | "
        "`y = x @ w` | "
        "`2*B*In*Out` | `W + A*(B*In + B*Out)` | "
        "`F / Bytes_naive` | `F / (W/WRF_dense + A*act_fusion*(B*In + B*Out))` |"
        " `Typically similar; weight reuse is limited in decode` |"
    )
    report_lines.append(
        "| Embedding prefill | `ids:[B,S] -> out:[B,S,D]` from `table:[V,D]` | "
        "`out = table[ids]` | "
        "`B*S*D` | `W + A*(B*S*D)` | "
        "`F / Bytes_naive` | `F / (W/WRF_dense + A*act_fusion*(B*S*D))` |"
        " `Usually close; dominated by table read` |"
    )
    report_lines.append(
        "| Embedding decode | `ids:[B] -> out:[B,D]` from `table:[V,D]` | "
        "`out = table[ids]` | "
        "`B*D` | `W + A*(B*D)` | "
        "`F / Bytes_naive` | `F / (W/WRF_dense + A*act_fusion*(B*D))` |"
        " `Close to native; little activation reuse` |"
    )
    report_lines.append(
        "| Attention prefill | "
        "`Q,K,V:[B,h,S,d]`, score `[B,h,S,S]`, context `[B,h,S,d]` | "
        "`p=softmax(q@kT); y=p@v` | "
        "`8*B*S*H^2 + 4*B*S^2*H + B*h*S^2` | "
        "`W + A*(B*S*H + 3*B*S*H + 2*B*h*S^2 + B*S*H) + A_kv*(B*S*C_kv)` | "
        "`F / Bytes_naive` | "
        "`F / (W/WRF_attn + A*(core_terms) + A_kv*(B*S*C_kv))` (flash removes SxS HBM materialization) |"
        " `Often much higher vs native when S is large` |"
    )
    report_lines.append(
        "| Attention decode | "
        "`Q:[B,h,1,d]`, cache `K,V:[B,h,L,d]`, score `[B,h,1,L]` | "
        "`p=softmax(q@kT); y=p@v` | "
        "`8*B*H^2 + 4*B*L*H + B*h*L` | "
        "`W + A*(B*H + 3*B*H + 2*B*h*L + B*H + 2*B*H) + A_kv*(B*L*C_kv + B*C_kv)` | "
        "`F / Bytes_naive` | "
        "`F / (W/WRF_attn + A*(core_terms) + A_kv*(B*L*C_kv + B*C_kv))` |"
        " `Usually limited by KV reads; efficient ~= native` |"
    )
    report_lines.append("")
    report_lines.append("Footnotes:")
    report_lines.append(
        "- `C_kv` mapping: standard MHA/GQA uses K/V cache in head space; "
        "DeepSeek-MLA uses `kv_lora_rank + qk_rope_head_dim`."
    )
    report_lines.append(
        "- Attention formulas above are generic dense attention references; "
        "module-level MLA estimates use detected dims (`r_q`, `r_kv`, "
        "`d_nope`, `d_rope`, `d_v`) in `_compute_attention_flops_*_mla`."
    )
    report_lines.append(
        "- Training maps from prefill: `F_train = F_prefill * training_flops_multiplier`, "
        "`bytes_train = bytes_prefill * training_bytes_multiplier`."
    )
    report_lines.append("")
    report_lines.append("### Tensor Core Mapping")
    report_lines.append("")
    report_lines.append(
        "Tensor cores deliver their advertised throughput only when GEMM shapes provide enough "
        "parallel work and pack well into hardware tiles. In decode, GEMMs often have a small "
        "effective M dimension (roughly the microbatch `B`), while dense prefill/training GEMMs "
        "have `M≈B*S` and MoE expert GEMMs have `M≈(tokens per active expert)`. We model "
        "tensor-core saturation with `eta_tc(B)=min(1, M_eff/B_sat)` (the symbol uses `B` for "
        "historical decode intuition; the implementation uses `M_eff`). We then define a "
        "peak-equivalent compute cost as "
        "`F_realizable = F_tensorcore/eta_tc + (F_theory-F_tensorcore)/eta_scalar`, so that "
        "`T_comp = F_realizable / P_peak` is consistent without double-counting utilization."
    )
    report_lines.append("")
    report_lines.append("### Byte Accounting")
    report_lines.append("")
    report_lines.append(
        "`bytes_hbm = bytes_weights + bytes_activations + bytes_kv + bytes_temporary`"
    )
    report_lines.append(
        "For routed MoE, `bytes_weights` is not scaled linearly by `top_k`: we approximate expert "
        "weight traffic by the expected number of *distinct* experts activated in the microbatch "
        "(uniform routing baseline), because weight reads occur per active expert."
    )
    report_lines.append("")
    report_lines.append("| Byte Term | Meaning |")
    report_lines.append("|---|---|")
    report_lines.append("| `bytes_weights` | Streamed weight traffic after residency factor (WRF) |")
    report_lines.append("| `bytes_activations` | Input/output activation movement |")
    report_lines.append("| `bytes_kv` | KV-cache read/write traffic (`L` for decode, `S` write for prefill) |")
    report_lines.append("| `bytes_temporary` | Score/prob/intermediate temporary buffers |")
    report_lines.append("")
    report_lines.append("### Byte Dominance Test")
    report_lines.append("")
    report_lines.append("`share(x) = bytes_x / bytes_hbm`")
    report_lines.append("")
    report_lines.append(
        "Decision rules: if `share(weights) > 70%`, we prioritize weight residency/compression/"
        "paging; if `share(kv) > 30%` and grows with `L`, we prioritize KV dtype/layout/cache; and "
        "if `share(temporary)` is large only in naive attention, we prioritize flash/fused attention."
    )
    report_lines.append("")
    report_lines.append(
        "| Mode | weights naive | kv naive | temp naive | weights eff | kv eff | temp eff |"
    )
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for mode in ["training", "prefill", "decode"]:
        naive_share = _compute_mode_byte_shares(mode_stats_by_exec["naive"][mode])
        eff_share = _compute_mode_byte_shares(mode_stats_by_exec["efficient"][mode])
        report_lines.append(
            f"| `{mode}` | {naive_share['weights'] * 100.0:.1f}% | "
            f"{naive_share['kv'] * 100.0:.1f}% | {naive_share['temporary'] * 100.0:.1f}% | "
            f"{eff_share['weights'] * 100.0:.1f}% | {eff_share['kv'] * 100.0:.1f}% | "
            f"{eff_share['temporary'] * 100.0:.1f}% |"
        )
    prefill_naive_share = _compute_mode_byte_shares(mode_stats_by_exec["naive"]["prefill"])
    prefill_eff_share = _compute_mode_byte_shares(mode_stats_by_exec["efficient"]["prefill"])
    decode_eff_share = _compute_mode_byte_shares(mode_stats_by_exec["efficient"]["decode"])
    report_lines.append("")
    report_lines.append(
        "In this run, prefill temporary-byte share drops from "
        f"`{prefill_naive_share['temporary'] * 100.0:.1f}%` (naive) to "
        f"`{prefill_eff_share['temporary'] * 100.0:.1f}%` (efficient), which is the intended "
        "effect of switching from score/prob materialization to flash-style attention. Decode in "
        "efficient mode has a byte mix of "
        f"weights `{decode_eff_share['weights'] * 100.0:.1f}%`, KV "
        f"`{decode_eff_share['kv'] * 100.0:.1f}%`, and temporary "
        f"`{decode_eff_share['temporary'] * 100.0:.1f}%`. If decode appears weight-dominant here, "
        "that reflects residency assumptions (WRF/paging strategy) and can shift under continuous "
        "batching; decode is often KV-bound in literature, and this static model can show either "
        "KV or weight-stream dominance depending on those assumptions."
    )
    report_lines.append("")
    report_lines.append("### Mode Byte Decomposition (naive vs efficient)")
    report_lines.append("")
    report_lines.append(
        "| Mode | bytes_weights naive | bytes_activations naive | bytes_kv naive | "
        "bytes_temporary naive | bytes_hbm naive | bytes_weights eff | "
        "bytes_activations eff | bytes_kv eff | bytes_temporary eff | bytes_hbm eff |"
    )
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mode in ["training", "prefill", "decode"]:
        naive_stats = mode_stats_by_exec["naive"][mode]
        eff_stats = mode_stats_by_exec["efficient"][mode]
        report_lines.append(
            f"| `{mode}` | {_format_bytes(naive_stats.bytes_weights)} | "
            f"{_format_bytes(naive_stats.bytes_activations)} | "
            f"{_format_bytes(naive_stats.bytes_kv)} | "
            f"{_format_bytes(naive_stats.bytes_temporary)} | "
            f"{_format_bytes(naive_stats.bytes_hbm)} | "
            f"{_format_bytes(eff_stats.bytes_weights)} | "
            f"{_format_bytes(eff_stats.bytes_activations)} | "
            f"{_format_bytes(eff_stats.bytes_kv)} | "
            f"{_format_bytes(eff_stats.bytes_temporary)} | "
            f"{_format_bytes(eff_stats.bytes_hbm)} |"
        )

    report_lines.append("")
    report_lines.append("## Roofline Analysis")
    report_lines.append("")
    roofline_section = "Roofline Analysis"
    _append_section_preamble(
        report_lines,
        why_lines=[
            "We combine roofline ceilings with the static time model to explain regimes across "
            "training, prefill, and decode.",
            "We plot points at (`AI_hbm`, `TF_est`) and use `T_est=max(T_comp, T_hbm, T_net)` to "
            "identify the limiting resource without claiming measured performance.",
            "Sweeps over batch and prompt/KV length then show which knobs can move decode toward "
            "the ridge and which cannot under the model assumptions.",
        ],
        look_for_lines=[
            "how the anchor workload (`B`, `S`, `L`) sets the base point for reported KPIs",
            "whether phases are left/right of ridge and which time term dominates",
            "how batch and sequence/KV-length sweeps shift `AI_hbm` and `TF_est`",
        ],
    )
    _append_section_term_primer(
        report_lines=report_lines,
        section_term_tracker=section_term_tracker,
        section_name=roofline_section,
    )
    report_lines.extend(
        _render_roofline_point_walkthrough(
            mode_stats_by_exec=mode_stats_by_exec,
            ai_knee=ai_knee,
            primary_roofline=primary_roofline,
        )
    )
    report_lines.append("### Configuration & Assumptions")
    report_lines.append("")
    config_note = (
        "We evaluate roofline points at an anchor workload defined by the caller-provided "
        "`batch_size` and `seq_len` arguments to `dump_model_info(...)`. In "
        "`examples/train_deepseek.py`, this corresponds to the training microbatch "
        "(`config.training.batch_size`) and the dataset/training context length "
        "(`config.data.max_seq_length - 1`). We set the base point with prefill length `S=seq_len` "
        "and decode KV length `L=seq_len`, and then run merged sweeps that vary decode (`B`, `L`, "
        "`EP`) and prefill (`B`, `S`, `EP`) around that anchor. To change the anchor, pass "
        "different `batch_size`/`seq_len` or override the sweep lists."
    )
    if parallelism.n_routed_experts is not None and not parallelism.ep_from_config:
        config_note += (
            " For routed-MoE models where `ep_size` is not provided by the model config, we "
            "default to ~4 routed experts per GPU (`EP=ceil(E/4)`) to reflect common expert "
            "sharding."
        )
    report_lines.append(config_note)
    report_lines.append("")
    tp_size = tp_size_assumed
    ep_size = ep_size_assumed
    routed_experts_per_gpu = parallelism.routed_experts_per_gpu

    report_lines.append("| Property | Value |")
    report_lines.append("|---|---|")
    report_lines.append(f"| Batch size (`B`) | `{batch_size}` |")
    report_lines.append(f"| Prefill sequence length (`S`) | `{seq_len}` |")
    report_lines.append(f"| Decode KV length (`L`) | `{seq_len}` |")
    report_lines.append(
        f"| Assumed parameter bytes (`W` dtype) | `{param_bytes_assumed}` (FP8 default) |"
    )
    report_lines.append(f"| Activation dtype bytes (`A`) | `{activation_bytes}` |")
    report_lines.append(f"| KV-cache dtype bytes (`A_kv`) | `{kv_cache_bytes}` |")
    report_lines.append(f"| Tensor parallel size (`TP`) | `{tp_size}` |")
    report_lines.append(f"| Expert parallel size (`EP`) | `{ep_size}` |")
    if routed_experts_per_gpu is not None:
        report_lines.append(f"| Routed experts per GPU (`E/EP`) | `{routed_experts_per_gpu}` |")
    report_lines.append(f"| Training FLOPs multiplier | `{training_flops_multiplier}` |")
    report_lines.append(f"| Training bytes multiplier | `{training_bytes_multiplier}` |")
    report_lines.append(f"| Primary roofline target | `{primary_roofline.name}` |")
    report_lines.append(
        f"| Primary peak compute (`P_peak`) | `{primary_roofline.peak_tflops:.1f} TFLOPs` |"
    )
    report_lines.append(
        f"| HBM bandwidth (`BW_hbm`) | `{primary_roofline.mem_bw_gbps:.0f} GB/s` |"
    )
    report_lines.append(f"| Interconnect bandwidth (`BW_net`) | `{interconnect_bw_gbps:.0f} GB/s` |")
    report_lines.append(f"| HBM ridge (`OI_knee`) | `{ai_knee:.3f} FLOP/byte` |")
    report_lines.append(f"| Network ridge (`OI_net`) | `{net_knee:.3f} FLOP/byte` |")
    report_lines.append(f"| Decode batch sweep (`B`) | `{decode_batch_sizes}` |")
    report_lines.append(f"| Decode KV-length sweep (`L`) | `{decode_cache_lengths}` |")
    report_lines.append(f"| Decode EP sweep (`EP`) | `{ep_sweep_sizes}` |")
    report_lines.append(f"| Prefill sequence sweep (`S`) | `{prefill_seq_lengths}` |")
    report_lines.append(f"| Prefill batch sweep (`B`) | `{prefill_batch_sizes}` |")
    report_lines.append(f"| Prefill EP sweep (`EP`) | `{ep_sweep_sizes}` |")
    report_lines.append(f"| Requested roofline x-limits | `{roofline_x_limits}` |")
    report_lines.append(f"| Requested roofline y-limits | `{roofline_y_limits}` |")
    report_lines.append(
        f"| Rendered roofline x-limits | `{resolved_roofline_x_limits}` |"
    )
    report_lines.append(
        f"| Rendered roofline y-limits | `{resolved_roofline_y_limits}` |"
    )
    report_lines.append(f"| Roofline label mode | `{roofline_label_mode}` |")
    report_lines.append("")
    report_lines.append("**Roofline Targets (FP8 Dense)**")
    report_lines.append("")
    report_lines.append("| Chip | Peak TFLOPs | HBM GB/s | OI_knee (FLOP/byte) |")
    report_lines.append("|---|---:|---:|---:|")
    for chip in roofline_targets:
        chip_knee = _safe_div(chip.peak_tflops * 1e12, chip.mem_bw_gbps * 1e9)
        report_lines.append(
            f"| `{chip.name}` | {chip.peak_tflops:.1f} | {chip.mem_bw_gbps:.0f} | {chip_knee:.3f} |"
        )
    report_lines.append("")
    report_lines.append("### Roofline Overview")
    report_lines.append("")
    report_lines.append(
        "Model points use the primary target assumptions above and are plotted at "
        "(`AI_hbm`, `TF_est`), where `TF_est = F_theory / T_est`. Chip curves show multi-chip "
        "roofline upper bounds (memory slope + compute ceiling) for comparison."
    )
    report_lines.append("")
    if roofline_summary_path is not None:
        report_lines.append(f"![]({os.path.basename(roofline_summary_path)})")
    else:
        report_lines.append("- Roofline plot unavailable (matplotlib may be missing).")
    report_lines.append("")
    report_lines.append("**Interpretation**")
    report_lines.append("")
    report_lines.append(
        f"Efficient training/prefill intensities "
        f"(`{mode_stats_by_exec['efficient']['training'].ai_hbm:.2f}`, "
        f"`{mode_stats_by_exec['efficient']['prefill'].ai_hbm:.2f}`) should be read against "
        f"`OI_knee={ai_knee:.2f}` to classify regime, while decode "
        f"(`AI_hbm={mode_stats_by_exec['efficient']['decode'].ai_hbm:.2f}`) indicates whether "
        "the serving path is still memory-limited under base assumptions. This figure should not "
        "be interpreted as parameter share implying runtime share."
    )
    report_lines.append("")
    report_lines.append("### Derivation Notes")
    report_lines.append("")
    report_lines.append(
        "We include this short note to keep the length variables unambiguous: prefill is "
        "parameterized by prompt length `S`, while decode is parameterized by KV-cache length "
        "`L`. This matters because different terms dominate: naive prefill attention can incur "
        "`O(S^2)` score/prob traffic, while decode attention is driven by `O(L)` KV reads and a "
        "small per-step compute core. We therefore plot rooflines using `AI_hbm` (HBM-only) as the "
        "x-axis and treat network effects separately via `bytes_net` and `T_net` inside "
        "`T_est=max(T_comp,T_hbm,T_net)`. The sweeps below vary one axis at a time (`B` for decode, "
        "`S` for prefill) so changes in `AI_hbm` and `TF_est` can be interpreted causally under the "
        "stated assumptions."
    )
    report_lines.append("")
    report_lines.append("### Why regimes differ across training/prefill/decode")
    report_lines.append("")
    report_lines.append(
        "Training and prefill are dominated by larger GEMMs with higher reuse, so they more often "
        "approach compute-side behavior under efficient kernels. Decode operates with smaller-M "
        "matmuls and explicit KV reads that scale with context length `L`, which keeps decode "
        "sensitive to memory traffic even when batch increases. The practical implication is that "
        "prefill optimization responds strongly to fusion and tensor-core utilization, while decode "
        "optimization is usually governed by byte traffic management and batching policy."
    )
    report_lines.append("")
    report_lines.append("### Execution Models (Naive vs Efficient)")
    report_lines.append("")
    report_lines.append(
        "This section compares execution modes through one consolidated matrix. The main claim "
        "is that mode shifts are primarily byte-path shifts: `F_theory` is stable while "
        "`bytes_hbm`, `AI_hbm`, and `T_est` move materially. Evidence comes from model-level KPIs "
        "and category-level AI deltas; limitations remain static-analysis assumptions around "
        "residency and fusion."
    )
    report_lines.append("")
    report_lines.append("#### Regime KPI Matrix (naive vs efficient)")
    report_lines.append("")
    report_lines.append(
        "We summarize each phase and execution mode with one row of modeled work, bytes, and time. "
        "`F_realizable` is the peak-equivalent FLOP cost used for compute time "
        "(`T_comp = F_realizable / P_peak`), while `AI_hbm` is the roofline x-coordinate "
        "(`F_theory / bytes_hbm`) and `TF_est` is derived later as `F_theory / T_est`. `Regime` is "
        "the max-time limiter (`argmax(T_comp,T_hbm,T_net)`), so it can differ from the ridge-side "
        "diagnostic (`AI_hbm` vs `OI_knee`). Tokens/s is computed per GPU with "
        "`tokens_per_step=B*S` for training/prefill and `tokens_per_step=B` for decode."
    )
    report_lines.append("")
    report_lines.append(
        "| Mode | Exec | F_realizable | bytes_hbm | AI_hbm | MFU_est | T_est (ms) | Tokens/s | Regime |"
    )
    report_lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for mode in ["training", "prefill", "decode"]:
        for exec_name in ["naive", "efficient"]:
            stats = mode_stats_by_exec[exec_name][mode]
            tokens_s = tokens_per_second_by_exec[exec_name][mode]
            report_lines.append(
                f"| `{mode}` | `{exec_name}` | {_format_flops(stats.flops_realizable)} | "
                f"{_format_bytes(stats.bytes_hbm)} | {stats.ai_hbm:.3e} | {stats.mfu_est:.3e} | "
                f"{stats.t_est * 1000.0:.3f} | {tokens_s:.2f} | `{stats.regime}` |"
            )

    report_lines.append("")
    report_lines.append("#### Category Delta Overview (naive vs efficient)")
    report_lines.append("")
    report_lines.append(
        "We group module entries into coarse categories (`attention`, `experts`, `ffn`, "
        "`embedding`, `other`) to explain which operator families drive the deltas between "
        "`naive` and `efficient`. For each mode and category we recompute `AI_hbm` from aggregated "
        "`F_theory` and `bytes_hbm`, and we report `bytes_total` as HBM+network attribution. With "
        "`EP=1`, network bytes are zero by construction and `bytes_total` is dominated by HBM "
        "terms; when `EP>1`, routed-MoE dispatch/collect can make the network term material."
    )
    report_lines.append("")
    report_lines.append(
        "| Mode | Category | AI_hbm naive | AI_hbm eff | AI delta % | "
        "bytes_total naive | bytes_total eff |"
    )
    report_lines.append("|---|---|---:|---:|---:|---:|---:|")
    for mode in ["training", "prefill", "decode"]:
        categories = sorted(
            {
                _categorize_efficiency_entry(entry)
                for exec_name in ["naive", "efficient"]
                for entry in efficiency_by_exec[exec_name].get(mode, [])
            }
        )
        for category in categories:
            vals = {}
            for exec_name in ["naive", "efficient"]:
                grouped = {"flops": 0.0, "bytes_hbm": 0.0, "bytes_total": 0.0}
                for entry in efficiency_by_exec[exec_name].get(mode, []):
                    if _categorize_efficiency_entry(entry) != category:
                        continue
                    grouped["flops"] += entry.flops_theory
                    grouped["bytes_hbm"] += entry.bytes_hbm
                    grouped["bytes_total"] += entry.bytes_total
                vals[exec_name] = grouped
            ai_naive = _safe_div(vals["naive"]["flops"], vals["naive"]["bytes_hbm"])
            ai_eff = _safe_div(vals["efficient"]["flops"], vals["efficient"]["bytes_hbm"])
            ai_delta_pct = (_safe_div(ai_eff - ai_naive, max(1e-12, ai_naive))) * 100.0
            report_lines.append(
                f"| `{mode}` | `{category}` | {ai_naive:.3e} | {ai_eff:.3e} | "
                f"{ai_delta_pct:.1f}% | {_format_bytes(vals['naive']['bytes_total'])} | "
                f"{_format_bytes(vals['efficient']['bytes_total'])} |"
            )

    report_lines.append("")
    report_lines.append("#### Cross-Mode Summary (naive vs efficient)")
    report_lines.append("")
    report_lines.append(
        "We provide a compact scanline across phases (training vs prefill vs decode) to make "
        "cross-mode comparisons easy. This is the fastest way to see whether an execution mode "
        "mainly changes bytes (`AI_hbm`) or whether it materially changes modeled time (`T_est`) "
        "and throughput (Tokens/s). In practice, prefill often benefits more from byte reductions "
        "(especially attention temporaries) than decode, which can remain constrained by streaming "
        "terms and small effective batch."
    )
    report_lines.append("")
    report_lines.append(
        "| Mode | AI_hbm naive | AI_hbm eff | T_est naive (ms) | T_est eff (ms) | Tokens/s naive | Tokens/s eff |"
    )
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for mode in ["training", "prefill", "decode"]:
        naive_stats = mode_stats_by_exec["naive"][mode]
        eff_stats = mode_stats_by_exec["efficient"][mode]
        report_lines.append(
            f"| `{mode}` | {naive_stats.ai_hbm:.3e} | "
            f"{eff_stats.ai_hbm:.3e} | "
            f"{naive_stats.t_est * 1000.0:.3f} | "
            f"{eff_stats.t_est * 1000.0:.3f} | "
            f"{tokens_per_second_by_exec['naive'][mode]:.2f} | "
            f"{tokens_per_second_by_exec['efficient'][mode]:.2f} |"
        )

    report_lines.append("")
    report_lines.append("#### Decode Sweep (vary B, L, EP)")
    report_lines.append("")
    report_lines.append(
        "We treat decode as a three-axis workload: microbatch `B`, KV-cache length `L`, and expert "
        "parallelism `EP` (which sets routed experts per GPU `E/EP`). This matters because the "
        "dominant decode bytes are typically streaming terms (weights and KV reads), so `AI_hbm` "
        "can shift substantially with batching and with long-context KV traffic. We keep `TP` fixed "
        f"(`TP={tp_size}`) and evaluate all points under the same static time model "
        "`T_est=max(T_comp,T_hbm,T_net)` so `Regime` is auditable as the max-time limiter."
    )
    report_lines.append("")

    decode_index_by_exec: Dict[str, Dict[Tuple[int, int, int], DecodeWorkloadPoint]] = {}
    for exec_name, points in decode_workload_points_by_exec.items():
        decode_index_by_exec[exec_name] = {
            (point.batch, point.cache_len, point.ep_size): point for point in points
        }

    report_lines.append(
        "| EP | E/EP | AI_hbm naive | AI_hbm eff | T_est naive (ms) | T_est eff (ms) | "
        "TF_est naive | TF_est eff | Regime naive | Regime eff |"
    )
    report_lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for ep_value in ep_sweep_sizes:
        experts_per_gpu = ""
        if parallelism.n_routed_experts is not None:
            experts_per_gpu = str(int(math.ceil(parallelism.n_routed_experts / max(1, ep_value))))
        naive_point = decode_index_by_exec.get("naive", {}).get((batch_size, seq_len, ep_value))
        eff_point = decode_index_by_exec.get("efficient", {}).get((batch_size, seq_len, ep_value))
        if naive_point is None or eff_point is None:
            continue
        tf_est_naive = naive_point.flops / ((max(1e-12, naive_point.t_est_ms) / 1000.0) * 1e12)
        tf_est_eff = eff_point.flops / ((max(1e-12, eff_point.t_est_ms) / 1000.0) * 1e12)
        report_lines.append(
            f"| {ep_value} | {experts_per_gpu} | {naive_point.ai_hbm:.3e} | {eff_point.ai_hbm:.3e} | "
            f"{naive_point.t_est_ms:.3f} | {eff_point.t_est_ms:.3f} | "
            f"{tf_est_naive:.2f} | {tf_est_eff:.2f} | "
            f"`{naive_point.regime}` | `{eff_point.regime}` |"
        )

    naive_bcrit = _b_crit_from_sweep(decode_sweep_by_exec.get("naive", []), ai_knee)
    eff_bcrit = _b_crit_from_sweep(decode_sweep_by_exec.get("efficient", []), ai_knee)
    report_lines.append("")
    report_lines.append(
        f"At the anchor slice (`EP={ep_size}`, `L={seq_len}`), `B_crit` at the HBM ridge "
        f"(`OI_knee={ai_knee:.2f}`) is: naive=`{naive_bcrit}`, efficient=`{eff_bcrit}`."
    )

    if decode_bl_roofline_path is not None:
        report_lines.append("")
        report_lines.append(f"![]({os.path.basename(decode_bl_roofline_path)})")
        report_lines.append("")
        report_lines.append(
            "We plot `B×L` points in roofline space at the anchor `EP` to show how longer KV "
            "contexts shift the same model leftward in `AI_hbm`, while increasing batch moves "
            "points rightward by amortizing streamed weights and fixed overheads. The key question "
            "is whether practical serving batches can move decode near the ridge for the target "
            "`L`, or whether KV traffic leaves decode firmly HBM-limited."
        )

    report_lines.append("")
    report_lines.append("##### Compute-Feasible Frontier (min B_g over EP×L)")
    report_lines.append("")
    report_lines.append(
        "We complement the roofline-space curves with a compute-feasibility frontier: under the "
        "simplifying per-GPU view (`DP=EP`, `TP=1`, no PP/SP), we search for the smallest microbatch "
        "`B_g` such that compute time is within `alpha` of the slower of HBM and network "
        "(`T_comp >= alpha*max(T_hbm,T_net)`). Empty cells indicate that KV/network asymptotes "
        "remain left of ridge (or that we did not find a crossing within the search grid), so "
        "decode cannot become compute-favorable at that (`EP`,`L`) under the model."
    )
    if decode_frontier_heatmap_path is not None:
        report_lines.append("")
        report_lines.append(f"![]({os.path.basename(decode_frontier_heatmap_path)})")
    if decode_frontier_metrics:
        report_lines.append("")
        col_titles = " | ".join(f"L={value}" for value in decode_cache_lengths)
        report_lines.append(f"| EP | E/EP | {col_titles} |")
        report_lines.append("|---:|---:|" + "|".join(["---:"] * len(decode_cache_lengths)) + "|")
        for ep_value in ep_sweep_sizes:
            experts_per_gpu = ""
            if parallelism.n_routed_experts is not None:
                experts_per_gpu = str(int(math.ceil(parallelism.n_routed_experts / max(1, ep_value))))
            row = [str(ep_value), experts_per_gpu]
            for cache_len in decode_cache_lengths:
                metrics = decode_frontier_metrics.get((ep_value, cache_len))
                min_bg = None if metrics is None else metrics.get("min_bg")
                if min_bg is None or (isinstance(min_bg, float) and math.isnan(min_bg)):
                    row.append("")
                else:
                    row.append(str(int(min_bg)))
            report_lines.append("| " + " | ".join(row) + " |")

    report_lines.append("")
    report_lines.append("**How These Numbers Are Calculated**")
    report_lines.append("")
    report_lines.append(
        "We compute each sweep point by fixing (`B`, `L`, `EP`), aggregating model-level FLOPs "
        "and bytes under fixed parallelism, and then evaluating the static time model and derived "
        "roofline diagnostics. The calculation chain is:"
    )
    report_lines.append("")
    report_lines.append("```text")
    report_lines.append("F_theory(B,L,EP,m) = sum_i F_i(B,L,EP,m)")
    report_lines.append(
        "bytes_hbm(B,L,EP,m) = sum_i (bytes_weights_i + bytes_activations_i + "
        "bytes_kv_i + bytes_temporary_i)"
    )
    report_lines.append("bytes_net(B,L,EP,m) = sum_i bytes_net_i")
    report_lines.append("AI_hbm = F_theory / bytes_hbm")
    report_lines.append("T_comp = F_realizable / (P_peak * 1e12)")
    report_lines.append("T_hbm = bytes_hbm / (BW_hbm * 1e9)")
    report_lines.append("T_net = bytes_net / (BW_net * 1e9)")
    report_lines.append("T_est = max(T_comp, T_hbm, T_net)")
    report_lines.append("regime = argmax{T_comp, T_hbm, T_net}")
    report_lines.append("TF_est = F_theory / T_est / 1e12")
    report_lines.append("TF_roofline_hbm = min(P_peak, BW_hbm * AI_hbm / 1e12)")
    report_lines.append("```")
    report_lines.append("")
    example_batch = 128 if 128 in decode_batch_sizes else max(decode_batch_sizes)
    example_key = (example_batch, seq_len, ep_size)
    eff_example = decode_index_by_exec.get("efficient", {}).get(example_key)
    if eff_example is not None:
        unclipped_tf = (primary_roofline.mem_bw_gbps * 1e9 * eff_example.ai_hbm) / 1e12
        clipped_tf = min(primary_roofline.peak_tflops, unclipped_tf)
        ai_side = ">=" if eff_example.ai_hbm >= ai_knee else "<"
        tf_est = eff_example.flops / ((max(1e-12, eff_example.t_est_ms) / 1000.0) * 1e12)
        report_lines.append(
            f"- Worked example (efficient, `EP={ep_size}`, `B={example_batch}`, fixed `L={seq_len}`):"
        )
        report_lines.append(
            f"  `F_theory={eff_example.flops:.3e}`, `bytes_hbm={eff_example.bytes_hbm:.3e}`, so "
            f"`AI_hbm={eff_example.ai_hbm:.3e}`."
        )
        report_lines.append(
            f"  Ridge check: `AI_hbm {ai_side} OI_knee` -> `{eff_example.ai_hbm:.3e} {ai_side} "
            f"{ai_knee:.3e}`."
        )
        report_lines.append(
            f"  Roofline upper bound: `BW_hbm * AI_hbm / 1e12 = {unclipped_tf:.2f} TFLOPs`; "
            f"`TF_roofline_hbm = min(P_peak, unclipped) = min({primary_roofline.peak_tflops:.2f}, "
            f"{unclipped_tf:.2f}) = {clipped_tf:.2f} TFLOPs`."
        )
        report_lines.append(
            f"  Time path: `T_comp={eff_example.t_comp_ms:.3f} ms`, "
            f"`T_hbm={eff_example.t_hbm_ms:.3f} ms`, `T_net={eff_example.t_net_ms:.3f} ms`, "
            f"`T_est={eff_example.t_est_ms:.3f} ms` -> regime `{eff_example.regime}`."
        )
        report_lines.append(
            f"  Estimated throughput: `TF_est = F_theory / T_est / 1e12 = {tf_est:.2f} TFLOPs`."
        )

    report_lines.append("")
    report_lines.append("#### Prefill Sweep (vary B, S, EP)")
    report_lines.append("")
    report_lines.append(
        "We treat prefill as a three-axis workload: microbatch `B`, prompt length `S`, and `EP`. "
        "Prefill differs from decode because attention has prompt-side reuse and (in naive mode) "
        "explicit `O(S^2)` temporary traffic, so increasing `S` can either amortize weight streaming "
        "or amplify temporary bytes depending on the attention bytes model. We keep `TP` fixed "
        f"(`TP={tp_size}`) and reuse the same time model as decode so the curves are directly "
        "comparable under the report’s assumptions."
    )
    if prefill_bs_roofline_path is not None:
        report_lines.append("")
        report_lines.append(f"![]({os.path.basename(prefill_bs_roofline_path)})")
        report_lines.append("")
        report_lines.append(
            "We plot an `S` sweep at multiple `B` values (fixed `EP`) to show how longer prompts "
            "shift `AI_hbm` and whether efficient attention reduces temporary-byte pressure enough "
            "to keep prefill compute-favorable at practical batch sizes."
        )

    prefill_index_by_exec: Dict[str, Dict[Tuple[int, int, int], PrefillWorkloadPoint]] = {}
    for exec_name, points in prefill_workload_points_by_exec.items():
        prefill_index_by_exec[exec_name] = {
            (point.batch, point.seq_len, point.ep_size): point for point in points
        }

    slice_seq_lengths: List[int] = []
    for candidate in [seq_len, 1024, 4096]:
        if candidate <= 0:
            continue
        if max_position_embeddings is not None and candidate > max_position_embeddings:
            continue
        if candidate not in prefill_seq_lengths:
            continue
        slice_seq_lengths.append(int(candidate))
    slice_seq_lengths = sorted(set(slice_seq_lengths))

    if slice_seq_lengths:
        report_lines.append("")
        report_lines.append(
            "To make the `EP` effect auditable, we report a small slice at the anchor batch "
            f"(`B={batch_size}`) and selected prompt lengths."
        )
        report_lines.append("")
        header_parts = ["EP", "E/EP"]
        for s_value in slice_seq_lengths:
            header_parts.append(f"naive (S={s_value})")
            header_parts.append(f"eff (S={s_value})")
        report_lines.append("| " + " | ".join(header_parts) + " |")
        report_lines.append("|" + "|".join(["---:"] * 2 + ["---"] * (len(header_parts) - 2)) + "|")
        for ep_value in ep_sweep_sizes:
            experts_per_gpu = ""
            if parallelism.n_routed_experts is not None:
                experts_per_gpu = str(int(math.ceil(parallelism.n_routed_experts / max(1, ep_value))))
            row_cells = [str(ep_value), experts_per_gpu]
            for s_value in slice_seq_lengths:
                naive_point = prefill_index_by_exec.get("naive", {}).get((batch_size, s_value, ep_value))
                eff_point = prefill_index_by_exec.get("efficient", {}).get((batch_size, s_value, ep_value))
                if naive_point is None or eff_point is None:
                    row_cells.extend(["", ""])
                    continue
                tf_est_naive = naive_point.flops / ((max(1e-12, naive_point.t_est_ms) / 1000.0) * 1e12)
                tf_est_eff = eff_point.flops / ((max(1e-12, eff_point.t_est_ms) / 1000.0) * 1e12)
                row_cells.append(f"`AI={naive_point.ai_hbm:.2e}, TF={tf_est_naive:.1f}`")
                row_cells.append(f"`AI={eff_point.ai_hbm:.2e}, TF={tf_est_eff:.1f}`")
            report_lines.append("| " + " | ".join(row_cells) + " |")

    report_lines.append("")
    report_lines.append("#### Mode Share Overview (naive vs efficient)")
    report_lines.append("")
    report_lines.append(
        "We visualize category shares to separate \"where parameters live\" from \"what limits "
        "runtime\" under this analytic model. Here, a \"share\" is computed from the modeled "
        "totals: FLOPs share is a fraction of `F_theory`, and bytes share is a fraction of "
        "`bytes_total` (HBM plus any network attribution), both aggregated per category. We report "
        f"shares per GPU under the fixed parallelism setting (`TP={tp_size}`, `EP={ep_size}`), so "
        "the MoE expert set per GPU (`E/EP`) and dispatch attribution are consistent with the KPIs. "
        "This plot exists to prevent the common MoE fallacy: experts can dominate parameters while "
        "attention/KV or other streaming terms dominate runtime."
    )
    report_lines.append("")
    for exec_name in ["naive", "efficient"]:
        if exec_name in mode_stack_paths:
            report_lines.append(f"![]({os.path.basename(mode_stack_paths[exec_name])})")
            report_lines.append("")
    if category_roofline_paths:
        report_lines.append("#### Category Roofline (Per Mode)")
        report_lines.append("")
        report_lines.append(
            "We aggregate entries by category and plot category-level points at "
            "(`AI_hbm`, `TF_est`) using the same time model as the mode-level KPIs. This view "
            "isolates which operator family is left of ridge (HBM-limited) versus right of ridge "
            "(compute-side) in each phase, and it helps explain why a phase-level regime label "
            "changes (or does not) when switching between `naive` and `efficient`."
        )
        report_lines.append("")
        for mode in ["training", "prefill", "decode"]:
            if mode not in category_roofline_paths:
                continue
            report_lines.append(f"**{mode.capitalize()}**")
            report_lines.append("")
            report_lines.append(f"![]({os.path.basename(category_roofline_paths[mode])})")
            report_lines.append("")
        report_lines.append("**Interpretation**")
        report_lines.append("")
        report_lines.append(
            "Category rooflines separate operator families by where their FLOPs and bytes come "
            "from. In decode, attention often sits at lower `AI_hbm` because KV reads grow with "
            "`L`, while GEMM-heavy categories (dense/experts) can sit closer to the compute side "
            "if weight streaming is sufficiently amortized. Importantly, parameter share does not "
            "imply runtime share; the bytes/FLOPs mix determines the regime."
        )
        report_lines.append("")

    kv_cache_map = _collect_kv_cache_elements_per_token(model)
    total_kv_elements_per_token = sum(kv_cache_map.values())
    kv_cache_resident_bytes = (
        batch_size * seq_len * total_kv_elements_per_token * kv_cache_bytes
    )
    expert_param_count = sum(layer.num_params for layer in layers if ".experts." in layer.name)
    non_expert_param_count = max(0, total_params - expert_param_count)
    tp_shard = max(1, int(tp_size))
    ep_shard = max(1, int(ep_size))

    expert_param_bytes_global = expert_param_count * param_bytes_assumed
    non_expert_param_bytes_global = non_expert_param_count * param_bytes_assumed
    non_expert_param_bytes_per_gpu = non_expert_param_bytes_global / tp_shard
    expert_param_bytes_per_gpu = expert_param_bytes_global / (tp_shard * ep_shard)
    parameter_resident_bytes = non_expert_param_bytes_per_gpu + expert_param_bytes_per_gpu

    training_grad_bytes = (
        (non_expert_param_count * param_bytes_assumed) / tp_shard
        + (expert_param_count * param_bytes_assumed) / (tp_shard * ep_shard)
    )
    training_optimizer_bytes = (
        (non_expert_param_count * (optimizer_state_bytes + master_weight_bytes)) / tp_shard
        + (expert_param_count * (optimizer_state_bytes + master_weight_bytes)) / (tp_shard * ep_shard)
    )
    training_activation_bytes = _estimate_activation_memory_bytes(
        model=model,
        batch_size=batch_size,
        seq_len=seq_len,
        activation_bytes=activation_bytes,
    )
    training_total_resident_bytes = (
        parameter_resident_bytes
        + training_grad_bytes
        + training_optimizer_bytes
        + training_activation_bytes
    )
    inference_total_resident_bytes = parameter_resident_bytes + kv_cache_resident_bytes
    hbm_capacity_bytes = hbm_capacity_gb * (1024 ** 3)

    _progress("Rendering memory feasibility section...")
    report_lines.append("")
    report_lines.append("### Memory Feasibility")
    report_lines.append("")
    report_lines.append(
        "We include this section as a sanity check: the throughput regimes above only matter if "
        "the anchor workload is even plausible under the stated parallelism assumptions "
        f"(`TP={tp_size}`, `EP={ep_size}`). We report a per-GPU resident-byte estimate for the "
        "same anchor (`B`, `S`, `L`) used elsewhere in the report, so we can see whether the model "
        "is parameter-dominated (common for large MoE) or KV/activation-dominated (common for long "
        "context) before we spend effort tuning kernels."
    )
    report_lines.append("")
    report_lines.append(
        "We deliberately keep the accounting coarse but consistent with `TP/EP`: we assume "
        "non-expert weights are sharded by `TP`, routed-expert weights are sharded by `TP×EP`, and "
        "KV/activations are per GPU. For inference we count `params + kv_cache`, with "
        "`kv_cache = B * L * sum(C_kv_per_layer) * A_kv`. For training we add gradients, "
        "optimizer+master weights, and a coarse saved-activation term. We do not model activation "
        "checkpoint schedules, allocator fragmentation, or overlap; interpret these totals as "
        "feasibility signals rather than allocator-accurate budgets."
    )
    report_lines.append("")
    report_lines.append("| Item | Estimated Bytes |")
    report_lines.append("|---|---:|")
    report_lines.append(
        f"| Parameters (non-expert, per GPU) | {_format_bytes(non_expert_param_bytes_per_gpu)} |"
    )
    report_lines.append(
        f"| Parameters (routed experts, per GPU) | {_format_bytes(expert_param_bytes_per_gpu)} |"
    )
    report_lines.append(f"| Parameters (total, per GPU) | {_format_bytes(parameter_resident_bytes)} |")
    report_lines.append(f"| KV Cache (B={batch_size}, L={seq_len}) | {_format_bytes(kv_cache_resident_bytes)} |")
    report_lines.append(f"| Training Gradients | {_format_bytes(training_grad_bytes)} |")
    report_lines.append(f"| Training Optimizer+Master | {_format_bytes(training_optimizer_bytes)} |")
    report_lines.append(f"| Training Saved Activations (coarse) | {_format_bytes(training_activation_bytes)} |")
    report_lines.append(f"| Inference Total Resident | {_format_bytes(inference_total_resident_bytes)} |")
    report_lines.append(f"| Training Total Resident (coarse) | {_format_bytes(training_total_resident_bytes)} |")
    report_lines.append(f"| HBM Budget | {_format_bytes(hbm_capacity_bytes)} |")
    report_lines.append("")
    inference_feasible = "yes" if inference_total_resident_bytes <= hbm_capacity_bytes else "no"
    training_feasible = "yes" if training_total_resident_bytes <= hbm_capacity_bytes else "no"
    report_lines.append(
        f"Under these assumptions, inference fits the HBM budget: `{inference_feasible}` "
        f"(assumed `param_bytes={param_bytes_assumed}`, `kv_cache_bytes={kv_cache_bytes}`), and "
        f"training fits the HBM budget: `{training_feasible}` (coarse estimate)."
    )

    _progress("Rendering communication envelope section...")
    report_lines.append("")
    report_lines.append("### Communication Envelope")
    report_lines.append("")
    report_lines.append(
        "We include this section to isolate when network can become a first-order limiter: "
        "routed-MoE dispatch/collect when experts are sharded across devices (`EP>1`). We report a "
        "simple envelope estimate (not a measured time) for activation bytes that must move to "
        "send token activations to experts and return expert outputs, which scales roughly with "
        "tokens, hidden size, and `top_k`. At the anchor setting (`EP="
        f"{ep_size}`), we report both intra-device and inter-device attribution; when `EP=1` the "
        "inter-device portion is zero by construction, while larger `EP` values can make `T_net` "
        "material and motivate compression and overlap."
    )
    report_lines.append("")
    report_lines.append("| Mode | Intra-device Dispatch | Inter-device Dispatch (est.) | Interconnect Time (ms, est.) |")
    report_lines.append("|---|---:|---:|---:|")
    for mode in ["training", "prefill", "decode"]:
        intra_b, inter_b = _estimate_moe_dispatch_bytes(
            model=model,
            mode=mode,
            batch_size=batch_size,
            seq_len=seq_len,
            activation_bytes=activation_bytes,
            ep_size_override=ep_size,
        )
        inter_ms = _safe_div(inter_b, interconnect_bw_gbps * 1e9) * 1000.0
        report_lines.append(
            f"| `{mode}` | {_format_bytes(intra_b)} | {_format_bytes(inter_b)} | {inter_ms:.3f} |"
        )

    report_lines.append("")
    report_lines.append("## Sensitivity Analysis")
    report_lines.append("")
    sensitivity_section = "Sensitivity Analysis"
    _append_section_preamble(
        report_lines,
        why_lines=[
            "We run sensitivity sweeps because single-point KPIs can hide which knobs most "
            "strongly move decode performance.",
            "We rank knobs by effect size over the combinational grid to focus tuning effort on "
            "high-leverage controls.",
            "The output is a compact map from knob changes to `T_est` and regime changes.",
        ],
        look_for_lines=[
            "which knobs most change `T_est` in decode",
            "how regimes shift with `L`, KV dtype, and MoE routing (`top_k`)",
            "whether naive vs efficient execution changes the ranking",
        ],
    )
    _append_section_term_primer(
        report_lines=report_lines,
        section_term_tracker=section_term_tracker,
        section_name=sensitivity_section,
    )
    if sensitivity_points_by_exec:
        total_points = sum(len(points) for points in sensitivity_points_by_exec.values())
        report_lines.append(
            f"We run the `{sensitivity_cfg.name}` full combinational grid "
            f"(total points: `{total_points}`) to quantify which configuration knobs most move "
            "decode `T_est` and regime labels under this static model."
        )
        if sensitivity_csv_path is not None:
            report_lines.append(
                f"The full sweep is saved as a CSV artifact: `{os.path.basename(sensitivity_csv_path)}`."
            )
        report_lines.append("")
        report_lines.append(
            "| Exec | kv_dtype(B) | top_k | kv_rank_scale | hidden_scale | L | "
            "AI_hbm | T_est(ms) | MFU_est | Regime |"
        )
        report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for exec_name in ["naive", "efficient"]:
            points = sensitivity_points_by_exec.get(exec_name, [])
            if not points:
                continue
            points_sorted = sorted(points, key=lambda point: point.t_est_ms)
            selected = points_sorted[:3] + points_sorted[-2:]
            for point in selected:
                report_lines.append(
                    f"| `{exec_name}` | {point.kv_dtype_bytes} | {point.top_k} | "
                    f"{point.kv_rank_scale:.2f} | {point.hidden_scale:.2f} | "
                    f"{point.cache_len} | {point.ai_hbm:.3e} | {point.t_est_ms:.3f} | "
                    f"{point.mfu_est:.3e} | `{point.regime}` |"
                )
        report_lines.append("")
        report_lines.append(
            "As `L` increases, KV read bytes rise, so `AI_hbm` typically decreases and HBM-bound "
            "cases become more frequent; the sweep helps quantify how strongly this trend depends "
            "on KV dtype, routing `top_k`, and the execution-mode byte assumptions."
        )
        report_lines.append("")
        report_lines.append("### Knob Ranking (Decode)")
        report_lines.append("")
        report_lines.append(
            "We rank knobs using efficient-mode decode points; the score is the median `T_est(ms)` "
            "spread across knob values (larger means more leverage)."
        )
        ranking = _rank_sensitivity_knobs(sensitivity_points_by_exec.get("efficient", []))
        report_lines.append("| Knob | Effect size | Median `T_est(ms)` by value |")
        report_lines.append("|---|---:|---|")
        for knob_name, effect_size, medians in ranking:
            median_text = ", ".join(
                f"{value}:{medians[value]:.2f}" for value in sorted(medians.keys())
            )
            report_lines.append(
                f"| {knob_name} | {effect_size * 100.0:.1f}% | `{median_text}` |"
            )
        report_lines.append("")
        report_lines.append("How to use the CSV:")
        report_lines.append("1. Filter rows to decode points for your target exec mode.")
        report_lines.append("2. Plot `T_est_ms` vs `L` grouped by `kv_dtype_bytes` and `top_k`.")
        report_lines.append(
            "3. Track regime transitions to verify if a knob changes memory/compute behavior."
        )
    else:
        report_lines.append("- Sensitivity analysis disabled.")

    report_lines.append("")
    report_lines.append("## Architectural Limits")
    report_lines.append("")
    report_lines.append(
        "We synthesize the preceding sections into a simple systems view: training and prefill can "
        "move toward compute-favorable operation under efficient execution assumptions, while "
        "decode remains more sensitive to memory traffic and serving-batch constraints. In "
        "long-context decode, KV traffic growth with `L` drives the familiar `~1/L` intensity "
        "decline once KV bytes dominate. Finally, MoE dispatch is always present intra-device, but "
        "inter-device pressure becomes material only when `ep_size > 1`."
    )

    if include_appendix_derivations:
        report_lines.append("")
        report_lines.extend(_render_appendix_derivations_pointer())
    report_lines.append("")
    report_lines.extend(_render_debugging_checklist_pointer())

    extra_plots = []
    core_plot_paths = set()
    if pie_plot_path is not None:
        core_plot_paths.add(pie_plot_path)
    if roofline_summary_path is not None:
        core_plot_paths.add(roofline_summary_path)
    if decode_bl_roofline_path is not None:
        core_plot_paths.add(decode_bl_roofline_path)
    if decode_frontier_heatmap_path is not None:
        core_plot_paths.add(decode_frontier_heatmap_path)
    if prefill_bs_roofline_path is not None:
        core_plot_paths.add(prefill_bs_roofline_path)
    core_plot_paths.update(mode_stack_paths.values())
    core_plot_paths.update(category_roofline_paths.values())
    for path in plot_paths:
        if path in core_plot_paths:
            continue
        extra_plots.append(path)

    if extra_plots:
        report_lines.append("")
        report_lines.append("### Additional Plots")
        report_lines.append("")
        for path in extra_plots:
            report_lines.append(f"![]({os.path.basename(path)})")
            report_lines.append("")

    _progress("Finalizing markdown serialization...")
    report_content = "\n".join(report_lines) + "\n"
    _validate_report_term_hygiene(report_content)
    report_content = _number_markdown_headings(report_content)
    _progress(f"Writing report file to {report_path} ...")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    if report_path != requested_report_path:
        _progress(f"Updating latest report alias at {requested_report_path} ...")
        with open(requested_report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

    _progress("Model report generation finished.")
    if logger is not None:
        try:
            logger.info(f"Model report saved to {report_path}")
            if report_path != requested_report_path:
                logger.info(f"Latest report updated at {requested_report_path}")
        except Exception:
            pass

    return model_info


def dump_mode_info(*args, **kwargs) -> ModelInfo:
    """Backward-compatible alias for dump_model_info."""
    return dump_model_info(*args, **kwargs)
