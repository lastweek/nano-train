"""ZeRO optimizer dtype-split behavior tests."""

from __future__ import annotations

import torch

from src.distributed.zero import DistributedOptimizerConfig
from src.distributed.zero import MegatronZeroOptimizer


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(4, dtype=torch.float32))


def test_zero_optimizer_allocates_state_with_configured_dtypes() -> None:
    model = _TinyModel()
    cfg = DistributedOptimizerConfig(
        use_distributed_optimizer=True,
        data_parallel_sharding_strategy="optim",
        main_params_dtype="bf16",
        main_grads_dtype="fp16",
        exp_avg_dtype="bf16",
        exp_avg_sq_dtype="fp32",
    )
    optimizer = MegatronZeroOptimizer(
        model=model,
        config=cfg,
        data_parallel_group=None,
        expert_data_parallel_group=None,
        expert_param_ids=set(),
    )

    state = optimizer.state[model.w]
    assert torch.is_tensor(state["master_param"])
    assert torch.is_tensor(state["exp_avg"])
    assert torch.is_tensor(state["exp_avg_sq"])
    assert state["master_param"].dtype == torch.bfloat16
    assert state["exp_avg"].dtype == torch.bfloat16
    assert state["exp_avg_sq"].dtype == torch.float32

    model.w.grad = torch.ones_like(model.w)
    optimizer.step_with_ready_grads()
    assert model.w.dtype == torch.float32

    payload = optimizer.get_parameter_state_dp_zero(include_tensors=True)
    assert payload["optimizer_state_dtypes"]["main_params_dtype"] == "bf16"
    assert payload["optimizer_state_dtypes"]["main_grads_dtype"] == "fp16"

    manifest = optimizer.build_checkpoint_manifest(rank=0, world_size=1)
    assert manifest["optimizer_state_dtypes"]["exp_avg_dtype"] == "bf16"


def test_zero_optimizer_loads_parameter_state_into_existing_state_dtypes() -> None:
    model = _TinyModel()
    cfg = DistributedOptimizerConfig(
        use_distributed_optimizer=True,
        data_parallel_sharding_strategy="optim",
        main_params_dtype="fp16",
        main_grads_dtype="fp16",
        exp_avg_dtype="fp16",
        exp_avg_sq_dtype="fp16",
    )
    optimizer = MegatronZeroOptimizer(
        model=model,
        config=cfg,
        data_parallel_group=None,
        expert_data_parallel_group=None,
        expert_param_ids=set(),
    )

    payload = optimizer.get_parameter_state_dp_zero(include_tensors=True)
    for record in payload["parameters"].values():
        record["master_param"] = record["master_param"].double()
        record["exp_avg"] = record["exp_avg"].double()
        record["exp_avg_sq"] = record["exp_avg_sq"].double()

    optimizer.load_parameter_state_from_dp_zero(payload, strict=True)
    state = optimizer.state[model.w]
    assert state["master_param"].dtype == torch.float16
    assert state["exp_avg"].dtype == torch.float16
    assert state["exp_avg_sq"].dtype == torch.float16
