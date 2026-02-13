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
class EfficiencyEntry:
    """Static efficiency estimate for one module and one mode."""
    name: str
    kind: str
    flops: float
    bytes_moved: float
    arithmetic_intensity: float
    roofline_tflops: float


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
    name: str = "H200_SXM"
    peak_tflops: float = 989.0
    mem_bw_gbps: float = 4800.0


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
            if hasattr(module, "num_heads") and hasattr(module, "head_dim"):
                info["attention_type"] = "MHA"
            if hasattr(module, "qkv_proj"):
                info["attention_type"] = "Fused QKV"
        if "moe" in name_lower or "expert" in name_lower:
            info["moe"] = "Detected"
        if isinstance(module, torch.nn.RMSNorm):
            info["normalization"] = "RMSNorm"
        if isinstance(module, torch.nn.LayerNorm):
            info["normalization"] = "LayerNorm"
        if isinstance(module, torch.nn.GELU):
            info["activation"] = "GELU"
        if isinstance(module, torch.nn.ReLU):
            info["activation"] = "ReLU"

    if hasattr(model, "position_embeddings"):
        info["position_encoding"] = "Learned Absolute"

    if hasattr(model, "lm_head") and hasattr(model, "token_embeddings"):
        if model.lm_head.weight.data_ptr() == model.token_embeddings.weight.data_ptr():
            info["weight_tying"] = "Yes"

    if hasattr(model, "config"):
        info["family"] = "Transformer"

    return info


def _collect_layer_info(model: torch.nn.Module) -> List[LayerInfo]:
    layers = []
    for name, param in model.named_parameters():
        num_params = param.numel()
        memory_bytes = num_params * param.element_size()
        memory_mb = memory_bytes / (1024 ** 2)

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


def _categorize_module(name: str) -> str:
    lower = name.lower()
    if "expert" in lower or "moe" in lower:
        return "experts"
    if "embedding" in lower:
        return "embedding"
    if "qkv" in lower or "q_proj" in lower or "k_proj" in lower or "v_proj" in lower:
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


def _aggregate_module_categories(modules: List[ModuleSizeInfo]) -> Dict[str, int]:
    totals: Dict[str, int] = {}
    for module in modules:
        category = _categorize_module(module.name)
        totals[category] = totals.get(category, 0) + module.num_params
    return totals


def _attention_module_names(model: torch.nn.Module) -> List[str]:
    names = []
    for name, module in model.named_modules():
        if hasattr(module, "qkv_proj") and hasattr(module, "out_proj"):
            if hasattr(module, "num_heads") and hasattr(module, "head_dim"):
                names.append(name)
    return names


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


def _compute_attention_bytes_prefill(
    batch: int,
    seq_len: int,
    hidden: int,
    num_heads: int,
    activation_bytes: int,
    weight_bytes: int,
) -> float:
    attn_matrix = batch * num_heads * seq_len * seq_len
    activations = (
        batch * seq_len * hidden +
        3 * batch * seq_len * hidden +
        2 * attn_matrix +
        batch * seq_len * hidden
    )
    return weight_bytes + activations * activation_bytes


def _compute_attention_bytes_decode(
    batch: int,
    cache_len: int,
    hidden: int,
    num_heads: int,
    activation_bytes: int,
    weight_bytes: int,
) -> float:
    attn_vector = batch * num_heads * cache_len
    kv_cache = 2 * batch * cache_len * hidden
    activations = (
        batch * 1 * hidden +
        3 * batch * 1 * hidden +
        kv_cache +
        2 * attn_vector +
        batch * 1 * hidden +
        2 * batch * 1 * hidden
    )
    return weight_bytes + activations * activation_bytes


def _estimate_efficiency(
    model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    activation_bytes: int,
    roofline: RooflineConfig,
    training_flops_multiplier: float,
    training_bytes_multiplier: float,
) -> Dict[str, List[EfficiencyEntry]]:
    attention_names = _attention_module_names(model)

    entries: Dict[str, List[EfficiencyEntry]] = {
        "training": [],
        "prefill": [],
        "decode": [],
    }

    mem_bw_bytes = roofline.mem_bw_gbps * 1e9

    def add_entry(mode: str, name: str, kind: str, flops: float, bytes_moved: float) -> None:
        intensity = _safe_div(flops, bytes_moved)
        roofline_tflops = min(roofline.peak_tflops, (mem_bw_bytes * intensity) / 1e12)
        entries[mode].append(
            EfficiencyEntry(
                name=name,
                kind=kind,
                flops=flops,
                bytes_moved=bytes_moved,
                arithmetic_intensity=intensity,
                roofline_tflops=roofline_tflops,
            )
        )

    for name, module in model.named_modules():
        if name == "":
            continue

        if name in attention_names:
            hidden = getattr(module, "hidden_size", None)
            if hidden is None:
                hidden = getattr(module, "num_heads", 1) * getattr(module, "head_dim", 0)
            num_heads = getattr(module, "num_heads", 1)

            weight_bytes = sum(p.numel() * p.element_size() for p in module.parameters())

            prefill_flops = _compute_attention_flops_prefill(batch_size, seq_len, hidden, num_heads)
            prefill_bytes = _compute_attention_bytes_prefill(
                batch_size,
                seq_len,
                hidden,
                num_heads,
                activation_bytes,
                weight_bytes,
            )

            decode_flops = _compute_attention_flops_decode(batch_size, seq_len, hidden, num_heads)
            decode_bytes = _compute_attention_bytes_decode(
                batch_size,
                seq_len,
                hidden,
                num_heads,
                activation_bytes,
                weight_bytes,
            )

            add_entry("prefill", name, "attention", prefill_flops, prefill_bytes)
            add_entry("decode", name, "attention", decode_flops, decode_bytes)
            add_entry(
                "training",
                name,
                "attention",
                prefill_flops * training_flops_multiplier,
                prefill_bytes * training_bytes_multiplier,
            )
            continue

        skip = False
        for attn_name in attention_names:
            if name.startswith(f"{attn_name}."):
                skip = True
                break
        if skip:
            continue

        if isinstance(module, torch.nn.Linear):
            weight_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
            prefill_flops = _compute_linear_flops(
                batch_size,
                seq_len,
                module.in_features,
                module.out_features
            )
            prefill_bytes = weight_bytes + activation_bytes * (
                batch_size * seq_len * module.in_features +
                batch_size * seq_len * module.out_features
            )
            decode_flops = _compute_linear_flops(
                batch_size,
                1,
                module.in_features,
                module.out_features
            )
            decode_bytes = weight_bytes + activation_bytes * (
                batch_size * module.in_features +
                batch_size * module.out_features
            )

            add_entry("prefill", name, "linear", prefill_flops, prefill_bytes)
            add_entry("decode", name, "linear", decode_flops, decode_bytes)
            add_entry(
                "training",
                name,
                "linear",
                prefill_flops * training_flops_multiplier,
                prefill_bytes * training_bytes_multiplier,
            )
            continue

        if isinstance(module, torch.nn.Embedding):
            weight_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
            prefill_flops = batch_size * seq_len * module.embedding_dim
            prefill_bytes = weight_bytes + activation_bytes * (
                batch_size * seq_len * module.embedding_dim
            )
            decode_flops = batch_size * 1 * module.embedding_dim
            decode_bytes = weight_bytes + activation_bytes * (batch_size * module.embedding_dim)

            add_entry("prefill", name, "embedding", prefill_flops, prefill_bytes)
            add_entry("decode", name, "embedding", decode_flops, decode_bytes)
            add_entry(
                "training",
                name,
                "embedding",
                prefill_flops * training_flops_multiplier,
                prefill_bytes * training_bytes_multiplier,
            )

    return entries


def _plot_roofline_summary(
    efficiency: Dict[str, List[EfficiencyEntry]],
    roofline: RooflineConfig,
    output_path: str,
    top_n: int = 20,
) -> Optional[str]:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use("Agg")
    except ImportError:
        return None

    entries = {
        mode: sorted(items, key=lambda e: e.flops, reverse=True)[:top_n]
        for mode, items in efficiency.items()
    }

    intensities = [
        e.arithmetic_intensity
        for mode_entries in entries.values()
        for e in mode_entries
        if e.arithmetic_intensity > 0
    ]
    if not intensities:
        return None

    min_intensity = max(min(intensities) * 0.1, 1e-4)
    max_intensity = max(intensities) * 10

    mem_bw = roofline.mem_bw_gbps * 1e9
    peak = roofline.peak_tflops
    knee = (peak * 1e12) / mem_bw

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    mem_end = min(knee, max_intensity)
    if mem_end > min_intensity:
        mem_x = [min_intensity, mem_end]
        mem_y = [(mem_bw * x) / 1e12 for x in mem_x]
        ax.loglog(mem_x, mem_y, label="Memory bound", color="#1f77b4")

    if max_intensity > knee:
        ax.hlines(peak, knee, max_intensity, colors="#ff7f0e", label="Compute bound")

    mode_markers = {
        "training": "o",
        "prefill": "s",
        "decode": "^",
    }
    kind_colors = {
        "attention": "#d62728",
        "linear": "#2ca02c",
        "embedding": "#9467bd",
        "other": "#7f7f7f",
    }

    for mode, mode_entries in entries.items():
        marker = mode_markers.get(mode, "o")
        for entry in mode_entries:
            color = kind_colors.get(entry.kind, kind_colors["other"])
            ax.scatter(
                entry.arithmetic_intensity,
                entry.roofline_tflops,
                s=30,
                alpha=0.75,
                marker=marker,
                color=color
            )

    ax.set_xlabel("Arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("Attainable TFLOPs")
    ax.set_title(f"Roofline Summary ({roofline.name})")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    legend_items = [
        ("training", mode_markers["training"], "black"),
        ("prefill", mode_markers["prefill"], "black"),
        ("decode", mode_markers["decode"], "black"),
        ("attention", "o", kind_colors["attention"]),
        ("linear", "o", kind_colors["linear"]),
        ("embedding", "o", kind_colors["embedding"]),
    ]
    handles = []
    labels = []
    for label, marker, color in legend_items:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor=color,
                markersize=7,
            )
        )
        labels.append(label)
    ax.legend(handles, labels, fontsize=8, loc="lower right", ncol=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def _plot_decode_batch_roofline(
    points: List[Tuple[int, float, float]],
    roofline: RooflineConfig,
    output_path: str,
) -> Optional[str]:
    """
    Plot decode roofline points across different batch sizes.

    Args:
        points: List of (batch_size, arithmetic_intensity, roofline_tflops)
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use("Agg")
    except ImportError:
        return None

    valid_points = [p for p in points if p[1] > 0.0]
    if not valid_points:
        return None

    intensities = [p[1] for p in valid_points]
    min_intensity = max(min(intensities) * 0.1, 1e-4)
    max_intensity = max(intensities) * 10

    mem_bw = roofline.mem_bw_gbps * 1e9
    peak = roofline.peak_tflops
    knee = (peak * 1e12) / mem_bw

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    mem_end = min(knee, max_intensity)
    if mem_end > min_intensity:
        mem_x = [min_intensity, mem_end]
        mem_y = [(mem_bw * x) / 1e12 for x in mem_x]
        ax.loglog(mem_x, mem_y, label="Memory bound", color="#1f77b4")

    if max_intensity > knee:
        ax.hlines(peak, knee, max_intensity, colors="#ff7f0e", label="Compute bound")

    for batch, intensity, tflops in valid_points:
        ax.scatter(intensity, tflops, s=36, alpha=0.85, color="#d62728")
        ax.annotate(
            f"B={batch}",
            (intensity, tflops),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    ax.set_xlabel("Decode arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("Attainable TFLOPs")
    ax.set_title(f"Decode Roofline Sweep ({roofline.name})")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


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

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Parameter Distribution by Category")
    fig.tight_layout()
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


def dump_model_info(
    model: torch.nn.Module,
    logger=None,
    report_path: str = "outputs/model_reports/model_report.md",
    plot_distributions: bool = True,
    plot_roofline: bool = True,
    batch_size: int = 1,
    seq_len: Optional[int] = None,
    activation_bytes: int = 2,
    roofline: Optional[RooflineConfig] = None,
    training_flops_multiplier: float = 3.0,
    training_bytes_multiplier: float = 2.0,
    max_weight_samples: int = 1_000_000,
    roofline_plot_top_n: int = 20,
    decode_batch_sizes: Optional[List[int]] = None,
) -> ModelInfo:
    """
    Dump comprehensive information about a model to a Markdown report.

    Returns a ModelInfo object with metadata and file paths.
    """
    if roofline is None:
        roofline = RooflineConfig()
    if decode_batch_sizes is None:
        decode_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    if seq_len is None:
        if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings"):
            seq_len = int(model.config.max_position_embeddings)
        else:
            seq_len = 2048

    if report_path is None or report_path.strip() == "":
        report_path = "outputs/model_reports/model_report.md"
    report_path = _ensure_unique_path(report_path)
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)

    layers = _collect_layer_info(model)
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

    efficiency = _estimate_efficiency(
        model=model,
        batch_size=batch_size,
        seq_len=seq_len,
        activation_bytes=activation_bytes,
        roofline=roofline,
        training_flops_multiplier=training_flops_multiplier,
        training_bytes_multiplier=training_bytes_multiplier,
    )
    mem_bw_bytes = roofline.mem_bw_gbps * 1e9
    ai_knee = _safe_div(roofline.peak_tflops * 1e12, mem_bw_bytes)

    decode_sweep_points: List[Tuple[int, float, float]] = []
    decode_sweep_detail: List[Tuple[int, float, float, float, float, str]] = []
    for sweep_batch in decode_batch_sizes:
        sweep = _estimate_efficiency(
            model=model,
            batch_size=sweep_batch,
            seq_len=seq_len,
            activation_bytes=activation_bytes,
            roofline=roofline,
            training_flops_multiplier=training_flops_multiplier,
            training_bytes_multiplier=training_bytes_multiplier,
        )
        decode_total_flops = sum(entry.flops for entry in sweep["decode"])
        decode_total_bytes = sum(entry.bytes_moved for entry in sweep["decode"])
        decode_ai = _safe_div(decode_total_flops, decode_total_bytes)
        decode_tflops = min(
            roofline.peak_tflops,
            (mem_bw_bytes * decode_ai) / 1e12
        )
        regime = "compute-bound" if decode_ai >= ai_knee else "memory-bound"
        decode_sweep_points.append((sweep_batch, decode_ai, decode_tflops))
        decode_sweep_detail.append(
            (
                sweep_batch,
                decode_total_flops,
                decode_total_bytes,
                decode_ai,
                decode_tflops,
                regime,
            )
        )

    plot_paths: List[str] = []
    pie_plot_path: Optional[str] = None
    report_dir = os.path.dirname(report_path) or "."
    base_name = os.path.splitext(os.path.basename(report_path))[0]

    if plot_roofline:
        plot_name = f"{base_name}_roofline.png"
        plot_path = os.path.join(report_dir, plot_name)
        out_path = _plot_roofline_summary(
            efficiency,
            roofline,
            output_path=plot_path,
            top_n=roofline_plot_top_n,
        )
        if out_path:
            plot_paths.append(out_path)

        decode_plot_name = f"{base_name}_decode_batch_roofline.png"
        decode_plot_path = os.path.join(report_dir, decode_plot_name)
        decode_out = _plot_decode_batch_roofline(
            decode_sweep_points,
            roofline,
            output_path=decode_plot_path,
        )
        if decode_out:
            plot_paths.append(decode_out)

    category_totals = _aggregate_module_categories(module_sizes)
    pie_path = os.path.join(report_dir, f"{base_name}_module_pie.png")
    pie_out = _plot_param_pie(category_totals, pie_path)
    if pie_out:
        pie_plot_path = pie_out
        plot_paths.append(pie_out)

    if plot_distributions:
        weights_sample = _collect_weights_sample(layers, model, max_weight_samples)
        if weights_sample.numel() > 0:
            hist_path = os.path.join(report_dir, f"{base_name}_weights_hist.png")
            out_path = _plot_weight_histogram(weights_sample, hist_path)
            if out_path:
                plot_paths.append(out_path)

    model_info.plot_paths = plot_paths

    report_lines: List[str] = []
    report_lines.append("# Model Report")
    report_lines.append("")
    report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    report_lines.append("## Model Fingerprint")
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
        report_lines.append("")
        report_lines.append("**Config**")
        report_lines.append("")
        for key in [
            "hidden_size",
            "num_layers",
            "num_attention_heads",
            "intermediate_size",
            "vocab_size",
            "max_position_embeddings",
            "dropout",
        ]:
            if hasattr(cfg, key):
                report_lines.append(f"- `{key}`: `{getattr(cfg, key)}`")

    report_lines.append("")
    report_lines.append("## Parameter & Memory Summary")
    report_lines.append("")
    report_lines.append(f"- Total parameters: `{_format_number(total_params)}`")
    report_lines.append(f"- Trainable parameters: `{_format_number(trainable_params)}`")
    report_lines.append(f"- Non-trainable parameters: `{_format_number(non_trainable_params)}`")
    report_lines.append(f"- Total parameter memory: `{total_memory_mb:.2f} MB`")
    if pie_plot_path is not None:
        report_lines.append("")
        report_lines.append("**Parameter Category Breakdown**")
        report_lines.append("")
        report_lines.append(f"![]({os.path.basename(pie_plot_path)})")

    report_lines.append("")
    report_lines.append("## Module Size Breakdown (Top 20)")
    report_lines.append("")
    report_lines.append("| Module | Params | % Total | Memory (MB) |")
    report_lines.append("|---|---:|---:|---:|")
    for module in module_sizes[:20]:
        report_lines.append(
            f"| `{module.name}` | {_format_number(module.num_params)} | "
            f"{module.percent_total:.2f}% | {module.memory_mb:.2f} |"
        )

    report_lines.append("")
    report_lines.append("## Weight Statistics (Top 20 Layers)")
    report_lines.append("")
    report_lines.append("| Layer | Mean | Std | Min | Max |")
    report_lines.append("|---|---:|---:|---:|---:|")
    for layer in layers[:20]:
        report_lines.append(
            f"| `{layer.name}` | {layer.mean:.6f} | {layer.std:.6f} | "
            f"{layer.min:.6f} | {layer.max:.6f} |"
        )
    if len(layers) > 20:
        report_lines.append(f"... {len(layers) - 20} more layers")

    report_lines.append("")
    report_lines.append("## Static Efficiency Estimates")
    report_lines.append("")
    report_lines.append("**Assumptions**")
    report_lines.append("")
    report_lines.append(f"- Batch size: `{batch_size}`")
    report_lines.append(f"- Sequence length: `{seq_len}`")
    report_lines.append(f"- Activation dtype bytes: `{activation_bytes}`")
    report_lines.append(f"- Training FLOPs multiplier: `{training_flops_multiplier}`")
    report_lines.append(f"- Training bytes multiplier: `{training_bytes_multiplier}`")
    report_lines.append(f"- Roofline target: `{roofline.name}`")
    report_lines.append(
        f"- Roofline peak: `{roofline.peak_tflops:.1f} TFLOPs`, "
        f"memory BW: `{roofline.mem_bw_gbps:.0f} GB/s`"
    )
    report_lines.append("")
    report_lines.append("**Arithmetic Intensity Definitions**")
    report_lines.append("")
    report_lines.append("**Notation**")
    report_lines.append("")
    report_lines.append("- `B`: batch size")
    report_lines.append("- `S`: prefill sequence length")
    report_lines.append("- `L`: decode KV-cache length (set to `S` in this report)")
    report_lines.append("- `H`: hidden size")
    report_lines.append("- `In`, `Out`: linear input/output width")
    report_lines.append("- `D`: embedding dimension")
    report_lines.append("- `heads`: number of attention heads")
    report_lines.append("- `W`: module weight bytes (`sum(param_numel * dtype_bytes)`)")
    report_lines.append("- `A`: activation bytes per element (`activation_bytes`)")
    report_lines.append("- `AI`: arithmetic intensity `FLOPs / Bytes`")
    report_lines.append("- Roofline bound: `min(peak_tflops, mem_bw_bytes * AI / 1e12)`")
    report_lines.append("")
    report_lines.append("**How Bytes Are Counted**")
    report_lines.append("")
    report_lines.append(
        "- Static first-order estimate: `Bytes = W + A * N_act`, where `N_act` "
        "is counted per operator formula below."
    )
    report_lines.append("- Activation bytes include major input/output/intermediate tensors.")
    report_lines.append(
        "- Runtime effects (cache hierarchy hits, kernel fusion, overlap) "
        "are not modeled."
    )
    report_lines.append("")
    report_lines.append("**Per-Module Formulas**")
    report_lines.append("")
    report_lines.append("- Linear prefill FLOPs: `2 * B * S * In * Out`")
    report_lines.append("- Linear prefill Bytes: `W + A * (B * S * In + B * S * Out)`")
    report_lines.append("- Linear decode FLOPs: `2 * B * In * Out`")
    report_lines.append("- Linear decode Bytes: `W + A * (B * In + B * Out)`")
    report_lines.append("- Embedding prefill FLOPs: `B * S * D`")
    report_lines.append("- Embedding prefill Bytes: `W + A * (B * S * D)`")
    report_lines.append("- Embedding decode FLOPs: `B * D`")
    report_lines.append("- Embedding decode Bytes: `W + A * (B * D)`")
    report_lines.append(
        "- Attention prefill FLOPs: `6*B*S*H^2 + 2*B*S^2*H + B*heads*S^2 + "
        "2*B*S^2*H + 2*B*S*H^2`"
    )
    report_lines.append(
        "- Attention prefill Bytes: `W + A * (B*S*H + 3*B*S*H + "
        "2*B*heads*S^2 + B*S*H)`"
    )
    report_lines.append(
        "- Attention decode FLOPs: `6*B*H^2 + 2*B*L*H + B*heads*L + "
        "2*B*L*H + 2*B*H^2`"
    )
    report_lines.append(
        "- Attention decode Bytes: `W + A * (B*H + 3*B*H + 2*B*L*H + "
        "2*B*heads*L + B*H + 2*B*H)`"
    )
    report_lines.append("")
    report_lines.append("**Mode Mapping**")
    report_lines.append("")
    report_lines.append("- Prefill mode uses prefill formulas directly.")
    report_lines.append("- Decode mode uses decode formulas directly.")
    report_lines.append("- Training mode scales prefill estimates:")
    report_lines.append("- Training FLOPs: `prefill_flops * training_flops_multiplier`")
    report_lines.append("- Training Bytes: `prefill_bytes * training_bytes_multiplier`")

    def append_mode_analysis(mode: str, top_n: int = 12) -> None:
        mode_entries = efficiency[mode]
        total_mode_flops = sum(entry.flops for entry in mode_entries)
        total_mode_bytes = sum(entry.bytes_moved for entry in mode_entries)
        mode_ai = _safe_div(total_mode_flops, total_mode_bytes)
        mode_tflops = min(roofline.peak_tflops, (mem_bw_bytes * mode_ai) / 1e12)
        mode_regime = "compute-bound" if mode_ai >= ai_knee else "memory-bound"
        peak_pct = _safe_div(mode_tflops, roofline.peak_tflops) * 100.0
        knee_ratio = _safe_div(mode_ai, ai_knee)

        report_lines.append("")
        report_lines.append(f"### {mode.capitalize()} - Model KPIs")
        report_lines.append("")
        report_lines.append("| Metric | Value |")
        report_lines.append("|---|---:|")
        report_lines.append(f"| Total FLOPs | {_format_flops(total_mode_flops)} |")
        report_lines.append(f"| Total Bytes | {_format_bytes(total_mode_bytes)} |")
        report_lines.append(f"| Model AI (FLOPs/Byte) | {mode_ai:.3e} |")
        report_lines.append(f"| Roofline-bound Throughput (TFLOPs) | {mode_tflops:.2f} |")
        report_lines.append(f"| Bound Regime | `{mode_regime}` |")
        report_lines.append(f"| % of Peak Throughput | {peak_pct:.2f}% |")
        report_lines.append(f"| AI / Knee AI | {knee_ratio:.3f} |")

        category_totals: Dict[str, Dict[str, float]] = {}
        for entry in mode_entries:
            category = _categorize_module(entry.name)
            if category not in category_totals:
                category_totals[category] = {"flops": 0.0, "bytes": 0.0}
            category_totals[category]["flops"] += entry.flops
            category_totals[category]["bytes"] += entry.bytes_moved

        report_lines.append("")
        report_lines.append(f"### {mode.capitalize()} - Category Breakdown")
        report_lines.append("")
        report_lines.append(
            "| Category | FLOPs | % FLOPs | Bytes | % Bytes | AI (FLOPs/Byte) |"
        )
        report_lines.append("|---|---:|---:|---:|---:|---:|")
        for category, data in sorted(
            category_totals.items(),
            key=lambda item: item[1]["flops"],
            reverse=True
        ):
            category_flops = data["flops"]
            category_bytes = data["bytes"]
            category_ai = _safe_div(category_flops, category_bytes)
            flops_share = _safe_div(category_flops, total_mode_flops) * 100.0
            bytes_share = _safe_div(category_bytes, total_mode_bytes) * 100.0
            report_lines.append(
                f"| `{category}` | {_format_flops(category_flops)} | "
                f"{flops_share:.2f}% | {_format_bytes(category_bytes)} | "
                f"{bytes_share:.2f}% | {category_ai:.3e} |"
            )

        report_lines.append("")
        report_lines.append(f"### {mode.capitalize()} - Top Modules")
        report_lines.append("")
        report_lines.append(
            "| Module | Type | Category | FLOPs | % FLOPs | Bytes | % Bytes | "
            "AI | Cum. FLOPs % |"
        )
        report_lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        cumulative_flops_pct = 0.0
        for entry in sorted(
            mode_entries,
            key=lambda item: item.flops,
            reverse=True
        )[:top_n]:
            flops_share = _safe_div(entry.flops, total_mode_flops) * 100.0
            bytes_share = _safe_div(entry.bytes_moved, total_mode_bytes) * 100.0
            cumulative_flops_pct += flops_share
            report_lines.append(
                f"| `{entry.name}` | `{entry.kind}` | "
                f"`{_categorize_module(entry.name)}` | {_format_flops(entry.flops)} | "
                f"{flops_share:.2f}% | {_format_bytes(entry.bytes_moved)} | "
                f"{bytes_share:.2f}% | {entry.arithmetic_intensity:.3e} | "
                f"{cumulative_flops_pct:.2f}% |"
            )

    append_mode_analysis("training")
    append_mode_analysis("prefill")
    append_mode_analysis("decode")

    first_compute_batch = None
    for batch, _, _, ai_value, _, _ in decode_sweep_detail:
        if ai_value >= ai_knee:
            first_compute_batch = batch
            break

    report_lines.append("")
    report_lines.append("### Decode Batch Sweep - Model-Level")
    report_lines.append("")
    report_lines.append(
        "| Batch | Decode FLOPs | Decode Bytes | Decode AI | Roofline TFLOPs | Regime |"
    )
    report_lines.append("|---:|---:|---:|---:|---:|---|")
    for batch, total_flops, total_bytes, ai_value, tflops_value, regime in decode_sweep_detail:
        report_lines.append(
            f"| {batch} | {_format_flops(total_flops)} | {_format_bytes(total_bytes)} | "
            f"{ai_value:.3e} | {tflops_value:.2f} | `{regime}` |"
        )
    report_lines.append("")
    if first_compute_batch is None:
        report_lines.append(
            f"- First compute-bound batch: `not reached` "
            f"(knee AI = `{ai_knee:.3e}` FLOPs/Byte)."
        )
    else:
        report_lines.append(
            f"- First compute-bound batch: `B={first_compute_batch}` "
            f"(knee AI = `{ai_knee:.3e}` FLOPs/Byte)."
        )

    if plot_paths:
        report_lines.append("")
        report_lines.append("## Plots")
        report_lines.append("")
        report_lines.append("**Plot Settings**")
        report_lines.append("")
        report_lines.append(f"- Batch size: `{batch_size}`")
        report_lines.append(f"- Sequence length: `{seq_len}`")
        report_lines.append(f"- Decode batch sweep: `{decode_batch_sizes}`")
        report_lines.append(f"- Roofline target: `{roofline.name}`")
        report_lines.append(
            f"- Roofline peak: `{roofline.peak_tflops:.1f} TFLOPs`, "
            f"memory BW: `{roofline.mem_bw_gbps:.0f} GB/s`"
        )
        report_lines.append("")
        for path in plot_paths:
            if pie_plot_path is not None and path == pie_plot_path:
                continue
            report_lines.append(f"![]({os.path.basename(path)})")
            report_lines.append("")

    report_content = "\n".join(report_lines) + "\n"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    if logger is not None:
        try:
            logger.info(f"Model report saved to {report_path}")
        except Exception:
            pass

    return model_info


def dump_mode_info(*args, **kwargs) -> ModelInfo:
    """Backward-compatible alias for dump_model_info."""
    return dump_model_info(*args, **kwargs)
