"""Math parity checks for ZeRO optimizer against reference AdamW."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.distributed.zero import DataParallelShardingStrategy
from src.distributed.zero import DistributedOptimizerConfig
from src.distributed.zero import MegatronZeroOptimizer


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(6, 8)
        self.linear2 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(torch.relu(self.linear1(x)))


def _run_compare(strategy: DataParallelShardingStrategy) -> None:
    torch.manual_seed(7)
    model_ref = _TinyModel()
    model_zero = _TinyModel()
    model_zero.load_state_dict(model_ref.state_dict())

    ref_opt = torch.optim.AdamW(
        model_ref.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    zero_opt = MegatronZeroOptimizer(
        model=model_zero,
        config=DistributedOptimizerConfig(
            use_distributed_optimizer=True,
            data_parallel_sharding_strategy=strategy,
            learning_rate=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        ),
        data_parallel_group=None,
        expert_data_parallel_group=None,
        expert_param_ids=set(),
    )

    for _ in range(6):
        x = torch.randn(5, 6)
        y = torch.randn(5, 4)

        ref_opt.zero_grad(set_to_none=True)
        zero_opt.zero_grad(set_to_none=True)

        loss_ref = torch.nn.functional.mse_loss(model_ref(x), y)
        loss_zero = torch.nn.functional.mse_loss(model_zero(x), y)
        loss_ref.backward()
        loss_zero.backward()

        ref_opt.step()
        zero_opt.step_with_ready_grads()

    for p_ref, p_zero in zip(model_ref.parameters(), model_zero.parameters(), strict=True):
        assert torch.allclose(p_ref, p_zero, atol=1e-6, rtol=1e-5)


def test_zero1_matches_adamw_single_rank() -> None:
    """ZeRO-1 path should match AdamW on single-rank runs."""
    _run_compare("optim")


def test_zero2_matches_adamw_single_rank() -> None:
    """ZeRO-2 path should match AdamW on single-rank runs."""
    _run_compare("optim_grads")
