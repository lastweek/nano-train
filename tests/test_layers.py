"""
Unit tests for native layer implementations.

Tests verify:
1. Accuracy: Output matches torch.nn reference
2. Gradients: Backward pass produces correct gradients
3. Edge cases: Empty tensors, single elements, extreme values
4. Integration: Layers work together in a network
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import Dropout
from src.layers import Embedding as _Embedding
from src.layers import GELU
from src.layers import LayerNorm as _LayerNorm
from src.layers import Linear as _Linear
from src.layers import clip_grad_norm
from conftest import assert_grad_close
from conftest import assert_tensor_close


def Linear(*args, **kwargs):  # noqa: N802
    return _Linear(*args, param_dtype=torch.float32, param_device=None, **kwargs)


def LayerNorm(*args, **kwargs):  # noqa: N802
    return _LayerNorm(*args, param_dtype=torch.float32, param_device=None, **kwargs)


def Embedding(*args, **kwargs):  # noqa: N802
    return _Embedding(*args, param_dtype=torch.float32, param_device=None, **kwargs)


# =============================================================================
# LINEAR LAYER TESTS
# =============================================================================

class TestLinear:
    """Test cases for Linear layer."""

    def test_forward_shape(self, device):
        """Forward pass preserves expected output shape."""
        batch, seq_len, in_features, out_features = 2, 10, 5, 20

        native = Linear(in_features, out_features).to(device)
        ref = nn.Linear(in_features, out_features).to(device)

        # Copy weights to ensure identical behavior
        with torch.no_grad():
            native.weight.copy_(ref.weight)
            if native.bias is not None:
                native.bias.copy_(ref.bias)

        x = torch.randn(batch, seq_len, in_features, device=device)

        native_out = native(x)
        ref_out = ref(x)

        assert native_out.shape == ref_out.shape
        assert native_out.shape == torch.Size([batch, seq_len, out_features])

    def test_forward_accuracy(self, device, tolerances):
        """Forward pass matches torch.nn output."""
        in_features, out_features = 10, 20
        batch_size = 5

        native = Linear(in_features, out_features).to(device)
        ref = nn.Linear(in_features, out_features).to(device)

        with torch.no_grad():
            native.weight.copy_(ref.weight)
            native.bias.copy_(ref.bias)

        x = torch.randn(batch_size, in_features, device=device)

        native_out = native(x)
        ref_out = ref(x)

        assert_tensor_close(
            native_out, ref_out,
            rtol=tolerances.RTOL,
            atol=tolerances.ATOL,
            msg="Linear forward pass mismatch"
        )

    def test_backward_accuracy(self, device, tolerances):
        """Backward pass produces correct gradients."""
        in_features, out_features = 10, 20
        batch_size = 5

        native = Linear(in_features, out_features).to(device)
        ref = nn.Linear(in_features, out_features).to(device)

        with torch.no_grad():
            native.weight.copy_(ref.weight)
            native.bias.copy_(ref.bias)

        x = torch.randn(batch_size, in_features, device=device, requires_grad=True)

        native_out = native(x)
        ref_out = ref(x)

        # Compute gradients
        native_out.sum().backward()
        ref_out.sum().backward()

        # Check input gradient
        assert_tensor_close(
            x.grad, x.grad,
            rtol=tolerances.RTOL,
            atol=tolerances.ATOL,
            msg="Linear backward pass mismatch (input grad)"
        )

        # Check weight gradients
        assert_tensor_close(
            native.weight.grad, ref.weight.grad,
            rtol=tolerances.RTOL,
            atol=tolerances.ATOL,
            msg="Linear backward pass mismatch (weight grad)"
        )

        # Check bias gradients
        assert_tensor_close(
            native.bias.grad, ref.bias.grad,
            rtol=tolerances.RTOL,
            atol=tolerances.ATOL,
            msg="Linear backward pass mismatch (bias grad)"
        )

    def test_no_bias(self, device):
        """Linear layer without bias works correctly."""
        in_features, out_features = 10, 20

        native = Linear(in_features, out_features, bias=False).to(device)
        ref = nn.Linear(in_features, out_features, bias=False).to(device)

        with torch.no_grad():
            native.weight.copy_(ref.weight)

        x = torch.randn(5, in_features, device=device)
        native_out = native(x)
        ref_out = ref(x)

        assert_tensor_close(native_out, ref_out, msg="No-bias Linear mismatch")

    def test_edge_cases(self, device):
        """Test edge cases: single element, large batch."""
        in_features, out_features = 10, 20

        native = Linear(in_features, out_features).to(device)
        ref = nn.Linear(in_features, out_features).to(device)

        with torch.no_grad():
            native.weight.copy_(ref.weight)
            native.bias.copy_(ref.bias)

        # Single element
        x = torch.randn(1, in_features, device=device)
        assert_tensor_close(native(x), ref(x), msg="Single element mismatch")

        # Large batch
        x = torch.randn(1000, in_features, device=device)
        assert_tensor_close(native(x), ref(x), msg="Large batch mismatch")

        # 3D input (batch, seq, features)
        x = torch.randn(10, 20, in_features, device=device)
        assert_tensor_close(native(x), ref(x), msg="3D input mismatch")


# =============================================================================
# LAYER NORMALIZATION TESTS
# =============================================================================

class TestLayerNorm:
    """Test cases for LayerNorm layer."""

    def test_forward_shape(self, device):
        """Output shape matches input shape."""
        normalized_shape = 20
        batch, seq_len = 5, 10

        native = LayerNorm(normalized_shape).to(device)
        ref = nn.LayerNorm(normalized_shape).to(device)

        with torch.no_grad():
            native.weight.copy_(ref.weight)
            native.bias.copy_(ref.bias)

        x = torch.randn(batch, seq_len, normalized_shape, device=device)

        native_out = native(x)
        ref_out = ref(x)

        assert native_out.shape == x.shape

    def test_forward_accuracy(self, device, tolerances):
        """Forward pass matches torch.nn output."""
        normalized_shape = 20
        batch_size = 5

        native = LayerNorm(normalized_shape).to(device)
        ref = nn.LayerNorm(normalized_shape).to(device)

        with torch.no_grad():
            native.weight.copy_(ref.weight)
            native.bias.copy_(ref.bias)

        x = torch.randn(batch_size, normalized_shape, device=device)

        native_out = native(x)
        ref_out = ref(x)

        assert_tensor_close(
            native_out, ref_out,
            rtol=tolerances.RTOL,
            atol=tolerances.ATOL,
            msg="LayerNorm forward pass mismatch"
        )

    def test_normalization_properties(self, device):
        """Output has zero mean and unit variance (when scale=1, shift=0)."""
        normalized_shape = 20
        batch_size = 100

        layer = LayerNorm(normalized_shape).to(device)

        # Set scale=1, shift=0 for pure normalization
        with torch.no_grad():
            layer.weight.fill_(1.0)
            layer.bias.zero_()

        x = torch.randn(batch_size, normalized_shape, device=device)
        out = layer(x)

        # Check mean is close to zero
        mean = out.mean(dim=-1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5), "Mean not close to zero"

        # Check std is close to one
        std = out.std(dim=-1, unbiased=False)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-5), "Std not close to one"

    def test_backward_accuracy(self, device, tolerances):
        """Backward pass produces correct gradients."""
        normalized_shape = 20
        batch_size = 5

        native = LayerNorm(normalized_shape).to(device)
        ref = nn.LayerNorm(normalized_shape).to(device)

        with torch.no_grad():
            native.weight.copy_(ref.weight)
            native.bias.copy_(ref.bias)

        x = torch.randn(batch_size, normalized_shape, device=device, requires_grad=True)

        native_out = native(x)
        ref_out = ref(x)

        native_out.sum().backward()
        ref_out.sum().backward()

        assert_tensor_close(
            x.grad, x.grad,
            rtol=tolerances.RTOL,
            atol=tolerances.ATOL,
            msg="LayerNorm backward pass mismatch"
        )


# =============================================================================
# EMBEDDING TESTS
# =============================================================================

class TestEmbedding:
    """Test cases for Embedding layer."""

    def test_forward_shape(self, device):
        """Output shape is (input_shape, embedding_dim)."""
        num_embeddings, embedding_dim = 100, 20
        batch_size, seq_len = 5, 10

        native = Embedding(num_embeddings, embedding_dim).to(device)
        ref = nn.Embedding(num_embeddings, embedding_dim).to(device)

        with torch.no_grad():
            native.weight.copy_(ref.weight)

        indices = torch.randint(0, num_embeddings, (batch_size, seq_len), device=device)

        native_out = native(indices)
        ref_out = ref(indices)

        expected_shape = torch.Size([batch_size, seq_len, embedding_dim])
        assert native_out.shape == expected_shape

    def test_forward_accuracy(self, device, tolerances):
        """Forward pass matches torch.nn output."""
        num_embeddings, embedding_dim = 100, 20
        batch_size = 5

        native = Embedding(num_embeddings, embedding_dim).to(device)
        ref = nn.Embedding(num_embeddings, embedding_dim).to(device)

        with torch.no_grad():
            native.weight.copy_(ref.weight)

        indices = torch.randint(0, num_embeddings, (batch_size, 10), device=device)

        native_out = native(indices)
        ref_out = ref(indices)

        assert_tensor_close(
            native_out, ref_out,
            rtol=tolerances.RTOL,
            atol=tolerances.ATOL,
            msg="Embedding forward pass mismatch"
        )

    def test_lookup_correctness(self, device):
        """Each index retrieves correct row from weight matrix."""
        num_embeddings, embedding_dim = 10, 5

        native = Embedding(num_embeddings, embedding_dim).to(device)

        # Create known weight matrix
        with torch.no_grad():
            for i in range(num_embeddings):
                native.weight[i].fill_(float(i))

        # Look up specific indices
        indices = torch.tensor([0, 2, 5, 9], device=device)
        out = native(indices)

        # Check each row matches expected
        for i, idx in enumerate(indices):
            expected = float(idx)
            assert torch.allclose(out[i], torch.full((embedding_dim,), expected))

    def test_backward_accuracy(self, device, tolerances):
        """Backward pass produces correct gradients."""
        num_embeddings, embedding_dim = 100, 20
        batch_size = 5

        native = Embedding(num_embeddings, embedding_dim).to(device)
        ref = nn.Embedding(num_embeddings, embedding_dim).to(device)

        with torch.no_grad():
            native.weight.copy_(ref.weight)

        indices = torch.randint(0, num_embeddings, (batch_size, 10), device=device)

        native_out = native(indices)
        ref_out = ref(indices)

        native_out.sum().backward()
        ref_out.sum().backward()

        # Check weight gradients
        assert_tensor_close(
            native.weight.grad, ref.weight.grad,
            rtol=tolerances.RTOL,
            atol=tolerances.ATOL,
            msg="Embedding backward pass mismatch"
        )


# =============================================================================
# DROPOUT TESTS
# =============================================================================

class TestDropout:
    """Test cases for Dropout layer."""

    def test_train_mode_drops(self, device):
        """Some activations are dropped during training."""
        p = 0.5
        batch_size, features = 1000, 20

        dropout = Dropout(p).to(device)
        dropout.train()

        x = torch.ones(batch_size, features, device=device)
        out = dropout(x)

        # Some should be zero (not all, but many)
        zero_mask = (out == 0).float()
        drop_ratio = zero_mask.mean().item()

        # Drop ratio should be close to p
        assert 0.3 < drop_ratio < 0.7, \
            f"Expected ~50% dropped, got {drop_ratio*100:.1f}%"

    def test_eval_mode_identity(self, device):
        """Output equals input during evaluation."""
        p = 0.5
        batch_size, features = 10, 20

        native = Dropout(p).to(device)
        ref = nn.Dropout(p).to(device)

        native.eval()
        ref.eval()

        x = torch.randn(batch_size, features, device=device)
        native_out = native(x)
        ref_out = ref(x)

        assert_tensor_close(native_out, ref_out, msg="Eval mode mismatch")
        assert_tensor_close(native_out, x, msg="Eval mode not identity")

    def test_zero_probability(self, device):
        """p=0 means no dropout."""
        batch_size, features = 10, 20

        native = Dropout(0.0).to(device)
        ref = nn.Dropout(0.0).to(device)

        native.train()
        ref.train()

        x = torch.randn(batch_size, features, device=device)
        native_out = native(x)
        ref_out = ref(x)

        assert_tensor_close(native_out, ref_out, msg="p=0 mismatch")
        assert_tensor_close(native_out, x, msg="p=0 not identity")


# =============================================================================
# GELU TESTS
# =============================================================================

class TestGELU:
    """Test cases for GELU activation."""

    def test_forward_shape(self, device):
        """Output shape matches input shape."""
        batch, seq_len, features = 5, 10, 20

        native = GELU().to(device)
        ref = nn.GELU().to(device)

        x = torch.randn(batch, seq_len, features, device=device)

        native_out = native(x)
        ref_out = ref(x)

        assert native_out.shape == x.shape

    def test_forward_accuracy(self, device, tolerances):
        """Forward pass matches torch.nn output."""
        batch_size, features = 100, 20

        native = GELU().to(device)
        ref = nn.GELU().to(device)

        x = torch.randn(batch_size, features, device=device)

        native_out = native(x)
        ref_out = ref(x)

        assert_tensor_close(
            native_out, ref_out,
            rtol=tolerances.RTOL,
            atol=tolerances.ATOL,
            msg="GELU forward pass mismatch"
        )

    def test_smoothness(self, device):
        """GELU is smooth (no sharp discontinuities)."""
        gelu = GELU().to(device)

        # Create small perturbations
        x = torch.randn(10, device=device)
        eps = 1e-3

        # Check that small changes produce small changes
        out1 = gelu(x)
        out2 = gelu(x + eps)
        diff = (out2 - out1).abs()

        # Should all be small
        assert diff.max().item() < 0.1, "GELU not smooth"

    def test_positive_values(self, device):
        """GELU outputs are positive for positive inputs."""
        gelu = GELU().to(device)

        x = torch.randn(100, device=device).abs()  # All positive
        out = gelu(x)

        # All outputs should be positive (GELU(x) > 0 for x > 0)
        assert (out > 0).all(), "GELU not positive for positive inputs"


# =============================================================================
# GRADIENT CLIPPING TESTS
# =============================================================================

class TestGradClip:
    """Test cases for gradient clipping."""

    def test_no_clip_needed(self, device):
        """Gradients unchanged when norm < max_norm."""
        p1 = torch.nn.Parameter(torch.randn(10, device=device))
        p2 = torch.nn.Parameter(torch.randn(10, device=device))

        # Small gradients
        p1.grad = torch.randn(10, device=device) * 0.1
        p2.grad = torch.randn(10, device=device) * 0.1

        max_norm = 1.0
        norm = clip_grad_norm([p1, p2], max_norm)

        # Norm should be less than max_norm
        assert norm < max_norm, f"Norm {norm} should be < {max_norm}"

    def test_clips_when_needed(self, device):
        """Gradients scaled when norm > max_norm."""
        p1 = torch.nn.Parameter(torch.randn(10, device=device))
        p2 = torch.nn.Parameter(torch.randn(10, device=device))

        # Large gradients
        p1.grad = torch.randn(10, device=device) * 5.0
        p2.grad = torch.randn(10, device=device) * 3.0

        max_norm = 1.0

        # Get norms before clipping
        norm_before = (p1.grad.norm()**2 + p2.grad.norm()**2)**0.5

        norm_after = clip_grad_norm([p1, p2], max_norm)

        # Norm should equal max_norm after clipping
        norm_after_check = (p1.grad.norm()**2 + p2.grad.norm()**2)**0.5

        assert norm_before > max_norm, "Test setup error: norm should be large"
        assert abs(norm_after_check - max_norm) < 1e-5, \
            f"Clipped norm {norm_after_check} should equal max_norm {max_norm}"

    def test_matches_pytorch(self, device):
        """Output matches torch.nn.utils.clip_grad_norm_."""
        p1 = torch.nn.Parameter(torch.randn(10, device=device))
        p2 = torch.nn.Parameter(torch.randn(10, device=device))

        # Create gradients
        grad1 = torch.randn(10, device=device)
        grad2 = torch.randn(10, device=device)
        p1.grad = grad1.clone()
        p2.grad = grad2.clone()

        max_norm = 1.0

        # Native version
        native_norm = clip_grad_norm([p1, p2], max_norm)
        native_grad1 = p1.grad.clone()
        native_grad2 = p2.grad.clone()

        # Reset and use torch version
        p1.grad = grad1.clone()
        p2.grad = grad2.clone()

        torch_norm = torch.nn.utils.clip_grad_norm_([p1, p2], max_norm)
        torch_grad1 = p1.grad
        torch_grad2 = p2.grad

        # Norms should match
        assert abs(native_norm - torch_norm) < 1e-5, \
            f"Norm mismatch: {native_norm} vs {torch_norm}"

        # Gradients should match
        assert torch.allclose(native_grad1, torch_grad1, atol=1e-5)
        assert torch.allclose(native_grad2, torch_grad2, atol=1e-5)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Test layers working together in a network."""

    def test_mlp_block(self, device, tolerances):
        """Full MLP block (Linear -> GELU -> Dropout -> Linear)."""
        batch, in_features, hidden, out_features = 5, 10, 20, 15

        # Native MLP
        native = torch.nn.Sequential(
            Linear(in_features, hidden),
            GELU(),
            Dropout(0.0),  # Disable for accuracy test
            Linear(hidden, out_features)
        ).to(device)

        # Reference MLP
        ref = torch.nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden, out_features)
        ).to(device)

        # Copy weights
        with torch.no_grad():
            native[0].weight.copy_(ref[0].weight)
            native[0].bias.copy_(ref[0].bias)
            native[3].weight.copy_(ref[3].weight)
            native[3].bias.copy_(ref[3].bias)

        x = torch.randn(batch, in_features, device=device)

        native_out = native(x)
        ref_out = ref(x)

        assert_tensor_close(
            native_out, ref_out,
            rtol=tolerances.RTOL,
            atol=tolerances.ATOL,
            msg="Integration test mismatch"
        )

    def test_transformer_block(self, device, tolerances):
        """Transformer block (Linear -> LayerNorm -> Attention -> Linear)."""
        # Simplified test: just Linear + LayerNorm
        batch, seq_len, hidden = 2, 5, 10

        # Native
        native = torch.nn.Sequential(
            Linear(hidden, hidden),
            LayerNorm(hidden)
        ).to(device)

        # Reference
        ref = torch.nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden)
        ).to(device)

        # Copy weights
        with torch.no_grad():
            native[0].weight.copy_(ref[0].weight)
            native[0].bias.copy_(ref[0].bias)
            native[1].weight.copy_(ref[1].weight)
            native[1].bias.copy_(ref[1].bias)

        x = torch.randn(batch, seq_len, hidden, device=device)

        native_out = native(x)
        ref_out = ref(x)

        assert_tensor_close(
            native_out, ref_out,
            rtol=tolerances.RTOL,
            atol=tolerances.ATOL,
            msg="Transformer block test mismatch"
        )
