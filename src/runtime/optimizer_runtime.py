"""Shared optimizer-runtime helpers."""

from __future__ import annotations

from typing import Callable

import torch

from src.runtime.context import RuntimeContext
from src.runtime.contracts import OptimizerState
from src.runtime.contracts import PrecisionDType
from src.runtime.mixed_precision import dtype_alias_to_torch


class PrecisionAdamW:
    """AdamW optimizer with explicit dtype controls for main params/grads/moments."""

    def __init__(
        self,
        params,
        *,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float,
        main_params_dtype: PrecisionDType,
        main_grads_dtype: PrecisionDType,
        exp_avg_dtype: PrecisionDType,
        exp_avg_sq_dtype: PrecisionDType,
    ) -> None:
        self._params = [param for param in params if param.requires_grad]
        self.param_groups = [
            {
                "params": self._params,
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "betas": tuple(betas),
                "eps": float(eps),
            }
        ]

        self._main_params_dtype = dtype_alias_to_torch(main_params_dtype)
        self._main_grads_dtype = dtype_alias_to_torch(main_grads_dtype)
        self._exp_avg_dtype = dtype_alias_to_torch(exp_avg_dtype)
        self._exp_avg_sq_dtype = dtype_alias_to_torch(exp_avg_sq_dtype)
        self.state: dict[torch.nn.Parameter, dict[str, torch.Tensor | int]] = {}
        self._global_step = 0

        for param in self._params:
            main_param = param.detach().to(dtype=self._main_params_dtype).clone()
            self.state[param] = {
                "step": 0,
                "main_param": main_param,
                "exp_avg": torch.zeros_like(main_param, dtype=self._exp_avg_dtype),
                "exp_avg_sq": torch.zeros_like(main_param, dtype=self._exp_avg_sq_dtype),
            }

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients on tracked parameters."""
        for param in self._params:
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                param.grad.zero_()

    @torch.no_grad()
    def step(self) -> bool:
        """Apply one AdamW step using dtype-configured optimizer state tensors."""
        group = self.param_groups[0]
        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])

        self._global_step += 1

        for param in self._params:
            grad = param.grad
            if grad is None:
                continue

            state = self.state[param]
            step = int(state["step"]) + 1
            state["step"] = step

            main_param = state["main_param"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            assert torch.is_tensor(main_param)
            assert torch.is_tensor(exp_avg)
            assert torch.is_tensor(exp_avg_sq)

            grad_for_update = grad.detach().to(dtype=self._main_grads_dtype).float()
            exp_avg_f = exp_avg.float()
            exp_avg_sq_f = exp_avg_sq.float()
            main_param_f = main_param.float()

            exp_avg_f.mul_(beta1).add_(grad_for_update, alpha=1.0 - beta1)
            exp_avg_sq_f.mul_(beta2).addcmul_(grad_for_update, grad_for_update, value=1.0 - beta2)

            bias_correction1 = 1.0 - float(beta1) ** step
            bias_correction2 = 1.0 - float(beta2) ** step
            denom = exp_avg_sq_f.sqrt().div_(bias_correction2**0.5).add_(eps)
            step_size = lr / bias_correction1

            main_param_f.mul_(1.0 - lr * weight_decay)
            main_param_f.addcdiv_(exp_avg_f, denom, value=-step_size)

            main_param.copy_(main_param_f.to(dtype=self._main_params_dtype))
            exp_avg.copy_(exp_avg_f.to(dtype=self._exp_avg_dtype))
            exp_avg_sq.copy_(exp_avg_sq_f.to(dtype=self._exp_avg_sq_dtype))
            param.data.copy_(main_param_f.to(dtype=param.dtype))

        return True


def zero_grad_optimizer(state: OptimizerState) -> None:
    """Zero gradients on the wrapped optimizer with `set_to_none=True`."""
    zero_grad_fn = getattr(state.optimizer, "zero_grad", None)
    if not callable(zero_grad_fn):
        raise TypeError("optimizer in OptimizerState does not implement zero_grad")
    zero_grad_fn(set_to_none=True)


def step_with_sync_policy(
    *,
    model: torch.nn.Module,
    state: OptimizerState,
    ctx: RuntimeContext,
    synchronize_gradients_fn: Callable[..., None],
) -> None:
    """
    Apply ZeRO-aware optimizer step policy.

    - If distributed optimizer is enabled, call `step_with_ready_grads()`.
    - Otherwise run gradient synchronization then call regular `step()`.
    """
    args = ctx.run_config.args
    if args.use_distributed_optimizer:
        step_with_ready_grads = getattr(state.optimizer, "step_with_ready_grads", None)
        if not callable(step_with_ready_grads):
            raise TypeError(
                "Expected optimizer with step_with_ready_grads when "
                "use_distributed_optimizer=True"
            )
        step_with_ready_grads()
        return

    synchronize_gradients_fn(
        model=model,
        shard_info=state.shard_info,
        data_parallel_size=ctx.parallel.data_parallel_size,
        expert_data_parallel_size=ctx.parallel.expert_data_parallel_size,
        data_parallel_group=ctx.parallel.data_parallel_group,
        expert_data_parallel_group=ctx.parallel.expert_data_parallel_group,
    )
    step_fn = getattr(state.optimizer, "step", None)
    if not callable(step_fn):
        raise TypeError("optimizer in OptimizerState does not implement step")
    step_fn()
