"""Trainability checks for persistent low-bit parameter paths."""

from __future__ import annotations

import torch

from src.layers import Linear
from src.runtime.contracts import ModulePrecisionAssignment



def _assignment() -> ModulePrecisionAssignment:
    return ModulePrecisionAssignment(
        module_name="linear",
        module_type="Linear",
        compute_lowbit_mode=None,
        persistent_lowbit_mode="fp4",
        persistent_scale_granularity="per_channel",
        fp4_persistent_format="nf4",
    )



def test_persistent_lowbit_weights_train_through_master_param() -> None:
    torch.manual_seed(7)
    layer = Linear(
        4,
        1,
        param_dtype=torch.float32,
        param_device=None,
    )
    layer.set_precision_assignment(_assignment())

    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    x = torch.randn(8, 4)
    target = torch.randn(8, 1)

    initial_weight = layer.weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        pred = layer(x)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        layer.refresh_persistent_lowbit_params()

    assert not torch.allclose(layer.weight.detach(), initial_weight)
