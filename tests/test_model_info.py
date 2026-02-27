"""
Test model information dumping functionality.
"""

import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import List
from unittest import mock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.models.deepseek import DeepSeekModel, DeepSeekModelConfig
from src.utils.model_info import (
    ExecutionModelConfig,
    LayerInfo,
    ModelInfo,
    RooflineConfig,
    SensitivityConfig,
    SweepPoint,
    TensorCoreModelConfig,
    _build_flop_breakdown,
    _build_roofline_piecewise_curve,
    _eta_tc,
    _estimate_efficiency,
    _plot_decode_batch_roofline,
    _render_appendix_derivations,
    _resolve_roofline_axis_limits,
    _summarize_mode_entries,
    _run_sensitivity_analysis,
    _sensitivity_config_from_profile,
    default_roofline_targets,
    dump_model_info,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)
        self.embedding = nn.Embedding(100, 10)

    def forward(self, x):
        return self.linear2(self.linear1(x))


def test_layer_info_creation():
    """Test LayerInfo dataclass creation."""
    print("Testing LayerInfo creation...")

    layer = LayerInfo(
        name="test.layer",
        shape=(10, 20),
        num_params=200,
        memory_mb=0.0016,
        dtype=torch.float32,
        requires_grad=True,
        mean=0.0,
        std=0.1,
        min=-0.5,
        max=0.5
    )

    assert layer.name == "test.layer"
    assert layer.num_params == 200
    assert layer.requires_grad == True
    assert layer.mean == 0.0

    print("  ✓ LayerInfo creation")


def test_model_info_creation():
    """Test ModelInfo dataclass creation."""
    print("Testing ModelInfo creation...")

    layers = [
        LayerInfo("layer1", (10, 20), 200, 0.001, torch.float32, True, 0.0, 0.1, -0.5, 0.5),
        LayerInfo("layer2", (20, 5), 100, 0.0005, torch.float32, True, 0.0, 0.1, -0.5, 0.5),
    ]

    model_info = ModelInfo(
        total_params=300,
        trainable_params=300,
        non_trainable_params=0,
        total_memory_mb=0.002,
        layers=layers,
        num_layers=2
    )

    assert model_info.total_params == 300
    assert len(model_info.layers) == 2
    assert model_info.num_layers == 2

    print("  ✓ ModelInfo creation")


def test_dump_model_info_basic():
    """Test basic model info dumping."""
    print("Testing dump_model_info (basic)...")

    # Create simple model
    model = SimpleModel()

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "model_report.md")

        # Dump model info
        model_info = dump_model_info(
            model,
            logger=None,
            report_path=report_path,
            plot_distributions=False,
            plot_roofline=False,
            sensitivity_enable=False,
        )

        # Verify ModelInfo returned
        assert isinstance(model_info, ModelInfo)
        assert model_info.total_params > 0
        assert model_info.trainable_params > 0
        assert len(model_info.layers) > 0

        # Verify report contains expected sections
        assert os.path.exists(model_info.report_path)
        with open(model_info.report_path, "r", encoding="utf-8") as f:
            report_str = f.read()

        assert "# Model Report" in report_str
        assert "Architecture Overview" in report_str
        assert "How to Read This Report" in report_str
        assert "Key Terms and Units" in report_str
        assert "Executive Summary" in report_str
        assert "3 Conclusions" in report_str
        assert "3 Next Optimizations" in report_str
        assert "Model Fingerprint" in report_str
        assert "Parameter & Memory Summary" in report_str
        assert "Module Size Breakdown" in report_str
        assert "Weight Statistics" in report_str
        assert "Analytical Model" in report_str
        assert "Roofline Analysis" in report_str
        assert "Sensitivity Analysis" in report_str
        assert "Architectural Limits" in report_str
        assert "Appendix A: Full FLOP Derivations" in report_str
        assert "Appendix B: Common Failure Modes / Debugging Checklist" in report_str
        assert "**Claim.**" not in report_str
        assert "Modeling intent and scope" in report_str
        assert "~1/L" in report_str
        assert "Byte Dominance Test" in report_str
        assert "share(x) = bytes_x / bytes_hbm" in report_str
        assert "Regime KPI Matrix (naive vs efficient)" in report_str
        assert "Category Delta Overview (naive vs efficient)" in report_str
        assert "Training - Model KPIs (naive vs efficient)" not in report_str
        assert "Training - Category Breakdown (naive vs efficient)" not in report_str
        assert "Knob Semantics and Rationale" in report_str
        assert "Weight Residency Factor" in report_str
        assert "W_eff = W / WRF" in report_str
        assert "act fusion" in report_str
        assert "elementwise" in report_str
        assert "How These Numbers Are Calculated" in report_str
        assert "fixing (`B`, `L`, `EP`)" in report_str
        assert "AI_hbm = F_theory / bytes_hbm" in report_str
        assert "TF_est = F_theory / T_est / 1e12" in report_str
        assert "TF_roofline_hbm = min(P_peak, BW_hbm * AI_hbm / 1e12)" in report_str
        assert "T_est = max(T_comp, T_hbm, T_net)" in report_str
        assert "Worked example (efficient, `EP=" in report_str
        assert "`B=128`" in report_str
        assert "Reading a Roofline Point" in report_str
        assert "**Interpretation**" in report_str
        assert (
            ("Knob Ranking (Decode)" in report_str)
            or ("Sensitivity analysis disabled." in report_str)
        )
        mandatory_terms = [
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
        for term in mandatory_terms:
            assert term in report_str
        assert "F_theory -> F_tensorcore -> F_realizable -> AI -> roofline/time limits" in report_str
        assert "Section term primer" not in report_str
        order = [
            "How to Read This Report",
            "Key Terms and Units",
            "Executive Summary",
            "Architecture Overview",
            "Analytical Model",
            "Roofline Analysis",
            "Sensitivity Analysis",
            "Architectural Limits",
            "Appendix A: Full FLOP Derivations",
        ]
        positions = [report_str.index(section) for section in order]
        assert positions == sorted(positions)

        # Narrative guardrail: major sections should start with prose, not a checklist.
        report_lines = report_str.splitlines()

        def _assert_prose_after_heading(title: str) -> None:
            heading_idx = None
            for idx, line in enumerate(report_lines):
                if line.startswith("## ") and title in line:
                    heading_idx = idx
                    break
            assert heading_idx is not None, f"Missing heading for section: {title}"
            for line in report_lines[heading_idx + 1: heading_idx + 20]:
                if not line.strip():
                    continue
                assert line.startswith("We "), (
                    f"Section '{title}' should start with a narrative paragraph, got: {line!r}"
                )
                return
            raise AssertionError(f"Section '{title}' missing prose paragraph after heading.")

        for section_title in [
            "Architecture Overview",
            "Analytical Model",
            "Roofline Analysis",
            "Sensitivity Analysis",
        ]:
            _assert_prose_after_heading(section_title)

        def _assert_prose_after_subheading(title: str) -> None:
            heading_idx = None
            for idx, line in enumerate(report_lines):
                if line.startswith("### ") and title in line:
                    heading_idx = idx
                    break
            assert heading_idx is not None, f"Missing subheading: {title}"
            for line in report_lines[heading_idx + 1: heading_idx + 30]:
                if not line.strip():
                    continue
                assert line.startswith("We "), (
                    f"Subheading '{title}' should start with a narrative paragraph, got: {line!r}"
                )
                return
            raise AssertionError(f"Subheading '{title}' missing prose paragraph after heading.")

        for subheading_title in [
            "Derivation Notes",
            "Memory Feasibility",
            "Communication Envelope",
        ]:
            _assert_prose_after_subheading(subheading_title)

        def _assert_prose_after_subsubheading(title: str) -> None:
            heading_idx = None
            for idx, line in enumerate(report_lines):
                if line.startswith("#### ") and title in line:
                    heading_idx = idx
                    break
            assert heading_idx is not None, f"Missing subsubheading: {title}"
            for line in report_lines[heading_idx + 1: heading_idx + 30]:
                if not line.strip():
                    continue
                assert line.startswith("We "), (
                    f"Subsubheading '{title}' should start with a narrative paragraph, got: {line!r}"
                )
                return
            raise AssertionError(f"Subsubheading '{title}' missing prose paragraph after heading.")

        for subsubheading_title in [
            "Decode Sweep (vary B, L, EP)",
            "Prefill Sweep (vary B, S, EP)",
        ]:
            _assert_prose_after_subsubheading(subsubheading_title)

        first_kpi_index = report_str.find("Regime KPI Matrix (naive vs efficient)")
        if first_kpi_index == -1:
            first_kpi_index = report_str.find("Cross-Mode Summary (naive vs efficient)")
        glossary_index = report_str.find("Key Terms and Units")
        assert glossary_index != -1 and first_kpi_index != -1
        assert glossary_index < first_kpi_index
        assert (
            ("### 1) Notation" in report_str)
            or ("**Notation**" in report_str)
            or ("### Notation" in report_str)
            or ("Notation" in report_str)
        )
        assert (
            ("### 2) Formula Reference" in report_str)
            or ("**Per-Module Formulas**" in report_str)
            or ("**Per-Module Formulas (Shape-Aware)**" in report_str)
            or ("**Per-Module Formulas (Shape Table)**" in report_str)
            or ("Per-Module Formulas" in report_str)
        )
        assert (
            ("### 3) Roofline Setup" in report_str)
            or ("**Plot Configuration**" in report_str)
            or ("Plot Configuration" in report_str)
            or ("Configuration & Assumptions" in report_str)
            or ("Roofline Overview" in report_str)
        )
        assert (
            ("| Module Family | Prefill FLOPs |" in report_str)
            or ("- Linear prefill FLOPs:" in report_str)
            or ("| Module | Tensor Shapes | FLOPs Derivation | Bytes Model |" in report_str)
            or ("| Module | Shape Explanation | FLOPs | Bytes (Native) | Native AI | Efficient AI |" in report_str)
            or ("| Module | Shape Explanation | FLOPs | Bytes (HBM, naive) | Native AI | Efficient AI |" in report_str)
            or (
                "| Module | Shape Explanation | Sample Torch | FLOPs | Bytes (HBM, naive) | "
                "Native AI | Efficient AI | Note |" in report_str
            )
        )
        assert ("Training" in report_str) or ("Training - Model KPIs" in report_str)
        assert ("Prefill" in report_str) or ("Prefill - Model KPIs" in report_str)
        assert ("Decode" in report_str) or ("Decode - Model KPIs" in report_str)
        assert ("Comparison Across Modes" in report_str) or ("Cross-Mode Summary" in report_str)
        assert "Decode Sweep (vary B, L, EP)" in report_str
        assert "Prefill Sweep (vary B, S, EP)" in report_str
        assert "KV-cache dtype bytes" in report_str
        assert "`C_kv` mapping" in report_str
        assert "MFU_est" in report_str
        assert "F_theory" in report_str
        assert (
            "Detailed derivations are documented in "
            "`docs/model_info_appendix.md`"
        ) in report_str
        assert (
            "Full debugging checklist is documented in "
            "`docs/model_info_appendix.md`"
        ) in report_str

        docs_text = Path("docs/model_info_appendix.md").read_text(encoding="utf-8")
        assert "## Appendix A: Detailed Derivations" in docs_text
        assert "F_linear = 2 * B * S * In * Out" in docs_text
        assert "## Appendix B: Full Debugging Checklist" in docs_text
        assert "Verify WRF is applied consistently in prefill and decode paths." in docs_text

    print("  ✓ Basic model info dump")


def test_parameter_counts():
    """Test accurate parameter counting."""
    print("Testing parameter counts...")

    model = SimpleModel()
    logger = type('MockLogger', (), {
        'info': lambda self, msg: None,
        'warning': lambda self, msg: None,
        'error': lambda self, msg: None,
    })()

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "model_report.md")
        model_info = dump_model_info(
            model,
            logger,
            report_path=report_path,
            plot_distributions=False,
            plot_roofline=False,
            sensitivity_enable=False,
        )

        # Count parameters manually
        expected_params = sum(p.numel() for p in model.parameters())

        assert model_info.total_params == expected_params, \
            f"Expected {expected_params} params, got {model_info.total_params}"

    print("  ✓ Parameter counts accurate")


def test_memory_calculation():
    """Test memory usage calculation."""
    print("Testing memory calculation...")

    model = SimpleModel()
    logger = type('MockLogger', (), {
        'info': lambda self, msg: None,
        'warning': lambda self, msg: None,
        'error': lambda self, msg: None,
    })()

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "model_report.md")
        model_info = dump_model_info(
            model,
            logger,
            report_path=report_path,
            plot_distributions=False,
            plot_roofline=False,
            sensitivity_enable=False,
        )

        # Calculate memory manually
        expected_memory_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024 ** 2)

        assert abs(model_info.total_memory_mb - expected_memory_mb) < 0.01, \
            f"Expected {expected_memory_mb} MB, got {model_info.total_memory_mb}"

    print("  ✓ Memory calculation accurate")


def test_trainable_vs_non_trainable():
    """Test trainable parameter detection."""
    print("Testing trainable vs non-trainable...")

    # Create model with frozen layer
    model = SimpleModel()
    model.linear1.weight.requires_grad = False
    model.linear1.bias.requires_grad = False

    logger = type('MockLogger', (), {
        'info': lambda self, msg: None,
        'warning': lambda self, msg: None,
        'error': lambda self, msg: None,
    })()

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "model_report.md")
        model_info = dump_model_info(
            model,
            logger,
            report_path=report_path,
            plot_distributions=False,
            plot_roofline=False,
            sensitivity_enable=False,
        )

        # Verify counts - count numel (number of elements), not just parameter objects
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)

        print(f"    Trainable: {trainable_count}, Non-trainable: {non_trainable_count}")
        print(
            "    Model info trainable: "
            f"{model_info.trainable_params}, "
            f"non-trainable: {model_info.non_trainable_params}"
        )

        assert model_info.trainable_params == trainable_count, \
            f"Expected {trainable_count} trainable, got {model_info.trainable_params}"
        assert model_info.non_trainable_params == non_trainable_count, \
            f"Expected {non_trainable_count} non-trainable, got {model_info.non_trainable_params}"
        assert model_info.non_trainable_params > 0

    print("  ✓ Trainable parameter detection")


def test_weight_statistics():
    """Test weight statistics computation."""
    print("Testing weight statistics...")

    model = nn.Linear(10, 20)
    # Set known weights
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(0.5)

    logger = type('MockLogger', (), {
        'info': lambda self, msg: None,
        'warning': lambda self, msg: None,
        'error': lambda self, msg: None,
    })()

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "model_report.md")
        model_info = dump_model_info(
            model,
            logger,
            report_path=report_path,
            plot_distributions=False,
            plot_roofline=False,
            sensitivity_enable=False,
        )

        # Find weight layer in results
        weight_layer = None
        for layer in model_info.layers:
            if "weight" in layer.name:
                weight_layer = layer
                break

        assert weight_layer is not None
        assert abs(weight_layer.mean - 1.0) < 0.01, \
            f"Expected mean ~1.0, got {weight_layer.mean}"
        assert abs(weight_layer.std) < 0.01, \
            f"Expected std ~0.0, got {weight_layer.std}"

    print("  ✓ Weight statistics computation")


def test_distribution_plot_generation():
    """Test distribution plot generation (if matplotlib available)."""
    print("Testing distribution plot generation...")

    model = SimpleModel()

    logger = type('MockLogger', (), {
        'info': lambda self, msg: None,
        'warning': lambda self, msg: None,
        'error': lambda self, msg: None,
    })()

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "model_report.md")
        model_info = dump_model_info(
            model,
            logger,
            report_path=report_path,
            plot_distributions=True,
            plot_roofline=False,
            sensitivity_enable=False,
        )

        if model_info.plot_paths:
            plot_path = model_info.plot_paths[0]
            assert os.path.exists(plot_path)
            assert os.path.getsize(plot_path) > 0
            print("  ✓ Distribution plot generation")
        else:
            print("  ⚠ Distribution plot skipped (matplotlib not available)")


def test_empty_model():
    """Test handling of model with no parameters."""
    print("Testing empty model...")

    class EmptyModel(nn.Module):
        def __init__(self):
            super().__init__()

    model = EmptyModel()
    logger = type('MockLogger', (), {
        'info': lambda self, msg: None,
        'warning': lambda self, msg: None,
        'error': lambda self, msg: None,
    })()

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "model_report.md")
        model_info = dump_model_info(
            model,
            logger,
            report_path=report_path,
            plot_distributions=False,
            plot_roofline=False,
            sensitivity_enable=False,
        )

        assert model_info.total_params == 0
        assert model_info.num_layers == 0

    print("  ✓ Empty model handling")


def test_meta_model_info_dump():
    """Test model info dump with meta-initialized parameters."""
    print("Testing meta-initialized model info dump...")

    with torch.device("meta"):
        model = SimpleModel()

    logger = type('MockLogger', (), {
        'info': lambda self, msg: None,
        'warning': lambda self, msg: None,
        'error': lambda self, msg: None,
    })()

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "model_report.md")
        model_info = dump_model_info(
            model,
            logger,
            report_path=report_path,
            plot_distributions=True,
            plot_roofline=False,
            sensitivity_enable=False,
        )

        assert model_info.total_params > 0
        assert os.path.exists(model_info.report_path)

        with open(model_info.report_path, "r", encoding="utf-8") as f:
            report_str = f.read()

        assert "Weight value statistics are unavailable for meta-initialized parameters" in report_str

    print("  ✓ Meta model info dump")


def test_derivation_renderer():
    """Test derivation appendix renderer output."""
    print("Testing derivation renderer...")

    cfg = type(
        "Cfg",
        (),
        {
            "hidden_size": 7168,
            "num_attention_heads": 128,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "moe_intermediate_size": 2048,
            "num_experts_per_tok": 8,
        },
    )()
    lines = _render_appendix_derivations(cfg)
    text = "\n".join(lines)

    assert "F_linear = 2 * B * S * In * Out" in text
    assert "F_attn_score = 2 * B * h * S^2 * d_eff" in text
    assert "F_MoE = B * S * top_k * 6 * H * d_moe" in text

    print("  ✓ Derivation renderer")


def test_tensor_core_eta_model():
    """Test tensor-core efficiency model properties."""
    print("Testing tensor-core eta model...")

    tc_cfg = TensorCoreModelConfig(enabled=True, b_sat=64)
    assert _eta_tc(1, tc_cfg) > 0
    assert _eta_tc(1, tc_cfg) < _eta_tc(32, tc_cfg)
    assert _eta_tc(64, tc_cfg) <= 1.0
    assert _eta_tc(256, tc_cfg) == 1.0

    print("  ✓ Tensor-core eta model")


def test_peak_equivalent_flop_cost_model():
    """Test peak-equivalent compute cost definition (F_realizable) is coherent."""
    print("Testing peak-equivalent FLOP cost model...")

    roofline = RooflineConfig(name="test_chip", peak_tflops=1000.0, mem_bw_gbps=5000.0)
    tc_cfg = TensorCoreModelConfig(enabled=True, b_sat=64, eta_scalar=0.35)

    flops_theory = 1e12
    small_b = _build_flop_breakdown(
        flops_theory=flops_theory,
        kind="linear",
        batch_size=1,
        roofline=roofline,
        tc_cfg=tc_cfg,
    )
    sat_b = _build_flop_breakdown(
        flops_theory=flops_theory,
        kind="linear",
        batch_size=64,
        roofline=roofline,
        tc_cfg=tc_cfg,
    )

    assert small_b.realizable >= small_b.theory
    assert sat_b.realizable >= sat_b.theory
    assert small_b.realizable > sat_b.realizable

    assert 0.0 < small_b.p_effective_tflops <= roofline.peak_tflops
    assert 0.0 < sat_b.p_effective_tflops <= roofline.peak_tflops
    assert small_b.p_effective_tflops < sat_b.p_effective_tflops

    print("  ✓ Peak-equivalent FLOP cost model")


def test_regime_label_matches_time_model():
    """Test regime is derived from the same time model used for T_est."""
    print("Testing regime label matches time model...")

    model = SimpleModel()
    eff = _estimate_efficiency(
        model=model,
        batch_size=1,
        seq_len=8,
        activation_bytes=1,
        kv_cache_bytes=1,
        param_bytes_assumed=1,
        roofline=RooflineConfig(),
        interconnect_bw_gbps=900.0,
        training_flops_multiplier=3.0,
        training_bytes_multiplier=2.0,
        exec_model=ExecutionModelConfig(
            name="naive",
            attention_bytes_model="naive",
            weight_residency_attn=1.0,
            weight_residency_dense=1.0,
            weight_residency_moe=1.0,
            activation_fusion_factor=1.0,
            elementwise_bytes_factor=1.0,
        ),
        tc_cfg=TensorCoreModelConfig(enabled=True, b_sat=64),
    )
    kpi = _summarize_mode_entries(
        mode="prefill",
        entries_for_mode=eff["prefill"],
        roofline=RooflineConfig(),
        interconnect_bw_gbps=900.0,
    )
    times = {"compute-bound": kpi.t_comp, "hbm-bound": kpi.t_hbm, "network-bound": kpi.t_net}
    expected = max(times, key=times.get)
    assert kpi.regime == expected

    print("  ✓ Regime label matches time model")


def test_byte_decomposition_consistency():
    """Test byte decomposition sums to bytes_hbm."""
    print("Testing byte decomposition consistency...")

    model = SimpleModel()
    efficiency = _estimate_efficiency(
        model=model,
        batch_size=2,
        seq_len=8,
        activation_bytes=1,
        kv_cache_bytes=1,
        param_bytes_assumed=1,
        roofline=RooflineConfig(),
        interconnect_bw_gbps=900.0,
        training_flops_multiplier=3.0,
        training_bytes_multiplier=2.0,
        exec_model=ExecutionModelConfig(
            name="naive",
            attention_bytes_model="naive",
            weight_residency_attn=1.0,
            weight_residency_dense=1.0,
            weight_residency_moe=1.0,
            activation_fusion_factor=1.0,
            elementwise_bytes_factor=1.0,
        ),
        tc_cfg=TensorCoreModelConfig(),
    )
    for mode in ["training", "prefill", "decode"]:
        for entry in efficiency[mode]:
            expected = (
                entry.bytes_weights
                + entry.bytes_activations
                + entry.bytes_kv
                + entry.bytes_temporary
            )
            assert abs(expected - entry.bytes_hbm) < 1e-6

    print("  ✓ Byte decomposition consistency")


def test_estimate_efficiency_decode_only_mode():
    """Test decode-only estimation path preserves decode outputs."""
    print("Testing decode-only efficiency mode...")

    model = SimpleModel()
    common_kwargs = dict(
        model=model,
        batch_size=2,
        seq_len=8,
        activation_bytes=1,
        kv_cache_bytes=1,
        param_bytes_assumed=1,
        roofline=RooflineConfig(),
        interconnect_bw_gbps=900.0,
        training_flops_multiplier=3.0,
        training_bytes_multiplier=2.0,
        exec_model=ExecutionModelConfig(
            name="naive",
            attention_bytes_model="naive",
            weight_residency_attn=1.0,
            weight_residency_dense=1.0,
            weight_residency_moe=1.0,
            activation_fusion_factor=1.0,
            elementwise_bytes_factor=1.0,
        ),
        tc_cfg=TensorCoreModelConfig(),
    )
    full = _estimate_efficiency(**common_kwargs)
    decode_only = _estimate_efficiency(
        **common_kwargs,
        modes_to_estimate=("decode",),
    )

    assert len(decode_only["training"]) == 0
    assert len(decode_only["prefill"]) == 0
    assert len(decode_only["decode"]) > 0

    full_decode_flops = sum(entry.flops_theory for entry in full["decode"])
    decode_only_flops = sum(entry.flops_theory for entry in decode_only["decode"])
    full_decode_bytes = sum(entry.bytes_hbm for entry in full["decode"])
    decode_only_bytes = sum(entry.bytes_hbm for entry in decode_only["decode"])
    assert abs(full_decode_flops - decode_only_flops) < 1e-6
    assert abs(full_decode_bytes - decode_only_bytes) < 1e-6

    print("  ✓ Decode-only efficiency mode")


def test_sensitivity_grid_cardinality():
    """Test sensitivity grid count for medium full grid."""
    print("Testing sensitivity grid cardinality...")

    model = SimpleModel()
    cfg = _sensitivity_config_from_profile("medium_full_grid")
    execution_models = [
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
    points = _run_sensitivity_analysis(
        model=model,
        execution_models=execution_models,
        sensitivity_cfg=cfg,
        batch_size=2,
        seq_len=8,
        ep_size_override=None,
        activation_bytes=1,
        kv_cache_bytes=1,
        param_bytes_assumed=1,
        roofline=RooflineConfig(),
        interconnect_bw_gbps=900.0,
        training_flops_multiplier=3.0,
        training_bytes_multiplier=2.0,
        tc_cfg=TensorCoreModelConfig(),
        module_weight_bytes_cache={},
        progress_cb=None,
    )
    expected = (
        len(cfg.kv_dtype_bytes)
        * len(cfg.top_k_values)
        * len(cfg.kv_rank_scales)
        * len(cfg.hidden_scales)
        * len(cfg.cache_lengths)
    )
    assert len(points["naive"]) == expected
    assert len(points["efficient"]) == expected

    print("  ✓ Sensitivity grid cardinality")


def test_sensitivity_deepseek_does_not_scan_named_modules():
    """Test DeepSeek sensitivity path does not iterate over all submodules."""
    print("Testing DeepSeek sensitivity avoids named_modules scans...")

    cfg = DeepSeekModelConfig(
        param_dtype=torch.float32,
        param_device=None,
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=64,
        moe_intermediate_size=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        n_group=1,
        topk_group=1,
        dropout=0.0,
    )
    model = DeepSeekModel(cfg)

    def _boom(*args, **kwargs):  # noqa: ANN001
        raise RuntimeError("named_modules should not be called for DeepSeek fast path")

    model.named_modules = _boom  # type: ignore[assignment]

    sensitivity_cfg = SensitivityConfig(
        name="unit_tiny",
        kv_dtype_bytes=(1,),
        top_k_values=(2,),
        kv_rank_scales=(1.0,),
        hidden_scales=(1.0,),
        cache_lengths=(64,),
    )
    execution_models = [
        ExecutionModelConfig(
            name="naive",
            attention_bytes_model="naive",
            weight_residency_attn=1.0,
            weight_residency_dense=1.0,
            weight_residency_moe=1.0,
            activation_fusion_factor=1.0,
            elementwise_bytes_factor=1.0,
        )
    ]
    points = _run_sensitivity_analysis(
        model=model,
        execution_models=execution_models,
        sensitivity_cfg=sensitivity_cfg,
        batch_size=2,
        seq_len=8,
        ep_size_override=1,
        activation_bytes=1,
        kv_cache_bytes=1,
        param_bytes_assumed=1,
        roofline=RooflineConfig(),
        interconnect_bw_gbps=900.0,
        training_flops_multiplier=3.0,
        training_bytes_multiplier=2.0,
        tc_cfg=TensorCoreModelConfig(enabled=True, b_sat=64),
        module_weight_bytes_cache={},
        progress_cb=None,
    )
    assert len(points["naive"]) == 1

    print("  ✓ DeepSeek sensitivity avoids named_modules scans")


def test_ep_inference_defaults_to_four_experts_per_gpu():
    """Ensure report defaults to ~4 routed experts per GPU when ep_size is not provided."""
    print("Testing EP inference defaults to ~4 experts per GPU...")

    cfg = DeepSeekModelConfig(
        param_dtype=torch.float32,
        param_device=None,
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=64,
        moe_intermediate_size=16,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        n_group=1,
        topk_group=1,
        dropout=0.0,
        max_position_embeddings=128,
    )
    model = DeepSeekModel(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "model_report.md")
        model_info = dump_model_info(
            model=model,
            logger=None,
            report_path=report_path,
            plot_distributions=False,
            plot_roofline=False,
            sensitivity_enable=False,
            include_architecture_diagrams=False,
            include_appendix_derivations=False,
        )
        report_text = Path(model_info.report_path).read_text(encoding="utf-8")
        assert "| Expert parallel size (`EP`) | `2` |" in report_text
        assert "| Routed experts per GPU (`E/EP`) | `4` |" in report_text

    print("  ✓ EP inference matches 4 experts/GPU default")


def test_roofline_piecewise_curve_correctness():
    """Test piecewise roofline curve construction for one chip."""
    print("Testing roofline piecewise curve correctness...")

    chip = RooflineConfig(name="test_chip", peak_tflops=1000.0, mem_bw_gbps=5000.0)
    mem_x, mem_y, comp_x, comp_y, knee = _build_roofline_piecewise_curve(
        chip=chip,
        min_intensity=1e-1,
        max_intensity=1e5,
        count_per_segment=16,
    )

    expected_knee = (chip.peak_tflops * 1e12) / (chip.mem_bw_gbps * 1e9)
    assert abs(knee - expected_knee) < 1e-9
    assert all(x <= knee * 1.000001 for x in mem_x)
    assert all(x >= knee * 0.999999 for x in comp_x)

    mem_bw_bytes = chip.mem_bw_gbps * 1e9
    for x_val, y_val in zip(mem_x, mem_y):
        expected_y = (mem_bw_bytes * x_val) / 1e12
        assert abs(expected_y - y_val) / max(1e-9, expected_y) < 1e-9

    assert all(abs(y_val - chip.peak_tflops) < 1e-9 for y_val in comp_y)

    print("  ✓ Roofline piecewise curve correctness")


def test_roofline_axis_limits_include_chip_knees():
    """Test auto-axis handling includes all chip knees."""
    print("Testing roofline axis limits include chip knees...")

    chips = default_roofline_targets()
    knees = [
        (chip.peak_tflops * 1e12) / (chip.mem_bw_gbps * 1e9)
        for chip in chips
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        (x_limits, y_limits) = _resolve_roofline_axis_limits(
            roofline_targets=chips,
            point_intensities=[10.0, 1000.0],
            point_tflops=[20.0, 2000.0],
            requested_x_limits=(1e4, 1e5),
            requested_y_limits=None,
        )
    x_min, x_max = x_limits
    y_min, y_max = y_limits

    assert any("hide one or more chip knees" in str(w.message) for w in caught)
    assert x_min < min(knees)
    assert x_max > max(knees)
    assert y_max > max(chip.peak_tflops for chip in chips)
    assert y_min > 0.0

    print("  ✓ Roofline axis limits include chip knees")


def test_decode_roofline_minimal_label_mode():
    """Test minimal roofline mode avoids dense per-point labels."""
    print("Testing decode roofline minimal label behavior...")

    try:
        import matplotlib  # noqa: F401
        from matplotlib.axes import Axes
    except ImportError:
        print("  ⚠ matplotlib unavailable, skipping label-mode test")
        return

    points_by_exec = {
        "naive": [
            SweepPoint(
                x=1,
                flops=1.0e12,
                bytes_hbm=1.0e11,
                bytes_net=0.0,
                bytes_total=1.0e11,
                ai_hbm=10.0,
                ai_total=10.0,
                roofline_tflops_hbm=50.0,
                regime_hbm="hbm-bound",
                t_comp_ms=1.0,
                t_hbm_ms=2.0,
                t_net_ms=0.0,
                t_est_ms=2.0,
            ),
            SweepPoint(
                x=128,
                flops=8.0e12,
                bytes_hbm=2.0e10,
                bytes_net=0.0,
                bytes_total=2.0e10,
                ai_hbm=400.0,
                ai_total=400.0,
                roofline_tflops_hbm=1500.0,
                regime_hbm="compute-bound",
                t_comp_ms=3.0,
                t_hbm_ms=1.2,
                t_net_ms=0.0,
                t_est_ms=3.0,
            ),
        ],
        "efficient": [
            SweepPoint(
                x=1,
                flops=1.0e12,
                bytes_hbm=8.0e10,
                bytes_net=0.0,
                bytes_total=8.0e10,
                ai_hbm=12.5,
                ai_total=12.5,
                roofline_tflops_hbm=60.0,
                regime_hbm="hbm-bound",
                t_comp_ms=1.0,
                t_hbm_ms=1.6,
                t_net_ms=0.0,
                t_est_ms=1.6,
            ),
            SweepPoint(
                x=128,
                flops=8.0e12,
                bytes_hbm=1.5e10,
                bytes_net=0.0,
                bytes_total=1.5e10,
                ai_hbm=533.3,
                ai_total=533.3,
                roofline_tflops_hbm=1700.0,
                regime_hbm="compute-bound",
                t_comp_ms=3.0,
                t_hbm_ms=0.9,
                t_net_ms=0.0,
                t_est_ms=3.0,
            ),
        ],
    }

    def _collect_annotation_texts(label_mode: str) -> List[str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, f"decode_{label_mode}.png")
            texts: List[str] = []
            original_annotate = Axes.annotate

            def _wrapped_annotate(self, text, *args, **kwargs):
                texts.append(str(text))
                return original_annotate(self, text, *args, **kwargs)

            with mock.patch.object(Axes, "annotate", _wrapped_annotate):
                _plot_decode_batch_roofline(
                    points_by_exec=points_by_exec,
                    primary_roofline=default_roofline_targets()[0],
                    roofline_targets=default_roofline_targets(),
                    output_path=output_path,
                    seq_len=255,
                    activation_bytes=1,
                    kv_cache_bytes=1,
                    roofline_x_limits=None,
                    roofline_y_limits=None,
                    roofline_label_mode=label_mode,
                )
        return texts

    minimal_texts = _collect_annotation_texts("minimal")
    full_texts = _collect_annotation_texts("full")

    assert not any(text.startswith("B=") for text in minimal_texts)
    assert any(text.startswith("B=") for text in full_texts)

    print("  ✓ Decode roofline minimal label behavior")


def run_all_tests():
    """Run all model info tests."""
    print("\n" + "=" * 70)
    print("Testing Model Info Dumping")
    print("=" * 70)
    print()

    test_layer_info_creation()
    test_model_info_creation()
    test_dump_model_info_basic()
    test_parameter_counts()
    test_memory_calculation()
    test_trainable_vs_non_trainable()
    test_weight_statistics()
    test_distribution_plot_generation()
    test_empty_model()
    test_meta_model_info_dump()
    test_derivation_renderer()
    test_tensor_core_eta_model()
    test_byte_decomposition_consistency()
    test_estimate_efficiency_decode_only_mode()
    test_sensitivity_grid_cardinality()
    test_roofline_piecewise_curve_correctness()
    test_roofline_axis_limits_include_chip_knees()
    test_decode_roofline_minimal_label_mode()

    print()
    print("=" * 70)
    print("ALL MODEL INFO TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
