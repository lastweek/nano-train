"""
Test model information dumping functionality.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.utils.model_info import dump_model_info, ModelInfo, LayerInfo


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
            plot_roofline=False
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
        assert "## Model Fingerprint" in report_str
        assert "## Parameter & Memory Summary" in report_str
        assert "## Module Size Breakdown" in report_str
        assert "## Weight Statistics" in report_str
        assert "## Static Efficiency Estimates" in report_str

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
            plot_roofline=False
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
            plot_roofline=False
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
            plot_roofline=False
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
            plot_roofline=False
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
            plot_roofline=False
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
            plot_roofline=False
        )

        assert model_info.total_params == 0
        assert model_info.num_layers == 0

    print("  ✓ Empty model handling")


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

    print()
    print("=" * 70)
    print("ALL MODEL INFO TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
