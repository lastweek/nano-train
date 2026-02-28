"""
Megatron-style ZeRO-1/2 distributed optimizer primitives for nano-train.

Learning primer for this file:
- Training state piles: parameters (P), gradients (G), optimizer states (O).
- `optim` (ZeRO-1): shard O, keep P replicated, reduce grads with all-reduce.
- `optim_grads` (ZeRO-2): shard O + G, keep P replicated, reduce grads with
  reduce-scatter (or all-reduce fallback), then all-gather updated P shards.
- Dense parameters communicate on DP groups, expert parameters on EDP groups.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Set

import torch
import torch.distributed as dist
import torch.nn as nn


LOGGER = logging.getLogger(__name__)
ZERO_CHECKPOINT_FORMAT_VERSION = 2


DataParallelShardingStrategy = Literal["no_shard", "optim", "optim_grads"]
OptimizerStateDType = Literal["fp32", "bf16", "fp16"]


@dataclass
class DistributedOptimizerConfig:
    """Configuration for Megatron-style distributed optimizer behavior."""

    use_distributed_optimizer: bool
    data_parallel_sharding_strategy: DataParallelShardingStrategy
    num_distributed_optimizer_instances: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    use_reduce_scatter: bool = True
    main_params_dtype: OptimizerStateDType = "fp32"
    main_grads_dtype: OptimizerStateDType = "fp32"
    exp_avg_dtype: OptimizerStateDType = "fp32"
    exp_avg_sq_dtype: OptimizerStateDType = "fp32"
    precision_recipe_name: str = "default"
    fp8_rounding: str = "nearest"
    fp8_activation_quant_granularity: str = "tensor"
    fp8_weight_quant_granularity: str = "tensor"
    fp8_comm_quant_enabled: bool = False
    fp8_comm_quant_granularity: str = "tensor"
    debug: bool = False
    debug_max_steps: int = 1
    debug_max_params: int = 8


@dataclass
class _ShardMeta:
    """Per-parameter shard metadata for one communication domain."""

    name: str
    numel: int
    shape: tuple[int, ...]
    dtype: str
    is_expert_param: bool
    group_size: int
    group_rank: int
    shard_start: int
    shard_end: int
    chunk_size: int


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _world_size(group) -> int:
    if group is None or not _is_distributed():
        return 1
    return int(dist.get_world_size(group=group))


def _group_rank(group) -> int:
    if group is None or not _is_distributed():
        return 0
    return int(dist.get_rank(group=group))


def _dist_rank() -> int:
    if not _is_distributed():
        return 0
    return int(dist.get_rank())


def _ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def _resolve_dtype_alias(dtype_alias: OptimizerStateDType) -> torch.dtype:
    if dtype_alias == "fp32":
        return torch.float32
    if dtype_alias == "bf16":
        return torch.bfloat16
    if dtype_alias == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported optimizer dtype alias: {dtype_alias}")


def _compute_shard(numel: int, rank: int, size: int) -> tuple[int, int, int]:
    # Split flattened tensor into ceil-div chunks so every rank has deterministic
    # ownership bounds even when `numel` is not divisible by group size.
    chunk_size = _ceil_div(numel, size)
    start = min(rank * chunk_size, numel)
    end = min(start + chunk_size, numel)
    return start, end, chunk_size


class MegatronZeroOptimizer:
    """
    Minimal Megatron-style ZeRO optimizer for AdamW semantics.

    Supported sharding strategies:
    - `optim`: ZeRO-1 (optimizer state sharded, grads full-reduced)
    - `optim_grads`: ZeRO-2 (optimizer state + grads sharded)
    """

    def __init__(
        self,
        model: nn.Module,
        config: DistributedOptimizerConfig,
        data_parallel_group,
        expert_data_parallel_group,
        expert_param_ids: Set[int],
    ) -> None:
        if not config.use_distributed_optimizer:
            raise ValueError("MegatronZeroOptimizer requires use_distributed_optimizer=True")
        if config.data_parallel_sharding_strategy not in ("optim", "optim_grads"):
            raise ValueError(
                "MegatronZeroOptimizer supports only sharded strategies: "
                "'optim' or 'optim_grads'"
            )
        if config.num_distributed_optimizer_instances != 1:
            raise ValueError("v1 supports only num_distributed_optimizer_instances=1")
        if config.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if config.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if config.debug_max_steps < 1:
            raise ValueError("debug_max_steps must be >= 1")
        if config.debug_max_params < 1:
            raise ValueError("debug_max_params must be >= 1")
        for dtype_name in (
            config.main_params_dtype,
            config.main_grads_dtype,
            config.exp_avg_dtype,
            config.exp_avg_sq_dtype,
        ):
            _resolve_dtype_alias(dtype_name)

        self.model = model
        self.config = config
        self.data_parallel_group = data_parallel_group
        self.expert_data_parallel_group = expert_data_parallel_group
        self.expert_param_ids = set(expert_param_ids)
        self._reduce_scatter_fallback = False
        self._main_params_dtype = _resolve_dtype_alias(config.main_params_dtype)
        self._main_grads_dtype = _resolve_dtype_alias(config.main_grads_dtype)
        self._exp_avg_dtype = _resolve_dtype_alias(config.exp_avg_dtype)
        self._exp_avg_sq_dtype = _resolve_dtype_alias(config.exp_avg_sq_dtype)

        self._named_params: list[tuple[str, nn.Parameter]] = [
            (name, param) for name, param in model.named_parameters() if param.requires_grad
        ]
        self._params = [param for _, param in self._named_params]
        self._meta_by_param: Dict[int, _ShardMeta] = {}
        self._meta_by_name: Dict[str, _ShardMeta] = {}

        self.param_groups = [
            {
                "params": self._params,
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
                "betas": config.betas,
                "eps": config.eps,
            }
        ]
        self.state: Dict[nn.Parameter, dict[str, torch.Tensor | int]] = {}
        self._global_step = 0

        self._initialize_states()
        self._log_debug_initial_state()

    def _log_debug_initial_state(self) -> None:
        """Emit one-time ZeRO sharding summary for learning/debugging."""
        if not self.config.debug:
            return

        dense_params = 0
        expert_params = 0
        dense_local_shard_numel = 0
        expert_local_shard_numel = 0
        dense_total_numel = 0
        expert_total_numel = 0

        for _, param in self._named_params:
            meta = self._meta_by_param[id(param)]
            local_shard_numel = max(0, meta.shard_end - meta.shard_start)
            if meta.is_expert_param:
                expert_params += 1
                expert_local_shard_numel += local_shard_numel
                expert_total_numel += meta.numel
            else:
                dense_params += 1
                dense_local_shard_numel += local_shard_numel
                dense_total_numel += meta.numel

        LOGGER.info(
            "[ZeRO Debug][rank=%d] init strategy=%s total_params=%d dense_params=%d "
            "expert_params=%d dense_local_shard_numel=%d/%d expert_local_shard_numel=%d/%d",
            _dist_rank(),
            self._strategy(),
            len(self._named_params),
            dense_params,
            expert_params,
            dense_local_shard_numel,
            dense_total_numel,
            expert_local_shard_numel,
            expert_total_numel,
        )

        for param_idx, (name, param) in enumerate(
            self._named_params[: self.config.debug_max_params]
        ):
            meta = self._meta_by_param[id(param)]
            domain = "EDP(expert)" if meta.is_expert_param else "DP(dense)"
            LOGGER.info(
                "[ZeRO Debug][rank=%d] param[%d] name=%s domain=%s numel=%d "
                "shard=[%d:%d) group_size=%d group_rank=%d",
                _dist_rank(),
                param_idx,
                name,
                domain,
                meta.numel,
                meta.shard_start,
                meta.shard_end,
                meta.group_size,
                meta.group_rank,
            )

    def _initialize_states(self) -> None:
        """Create local optimizer shard state for each parameter."""
        for name, param in self._named_params:
            is_expert = id(param) in self.expert_param_ids
            # Dense parameters reduce/update on DP; expert parameters use EDP.
            group = self.expert_data_parallel_group if is_expert else self.data_parallel_group
            group_size = _world_size(group)
            group_rank = _group_rank(group)
            numel = int(param.numel())
            # Ownership map: each rank owns only [shard_start:shard_end] for O.
            shard_start, shard_end, chunk_size = _compute_shard(numel, group_rank, group_size)
            shard_numel = max(0, shard_end - shard_start)

            full_flat = param.detach().to(dtype=self._main_params_dtype).view(-1)
            # ZeRO keeps full model params for compute but only local optimizer
            # shards (`master_param`, `exp_avg`, `exp_avg_sq`) for updates.
            if shard_numel > 0:
                master_param = full_flat[shard_start:shard_end].clone()
            else:
                master_param = full_flat.new_empty((0,))
            exp_avg = torch.zeros_like(master_param, dtype=self._exp_avg_dtype)
            exp_avg_sq = torch.zeros_like(master_param, dtype=self._exp_avg_sq_dtype)

            meta = _ShardMeta(
                name=name,
                numel=numel,
                shape=tuple(param.shape),
                dtype=str(param.dtype),
                is_expert_param=is_expert,
                group_size=group_size,
                group_rank=group_rank,
                shard_start=shard_start,
                shard_end=shard_end,
                chunk_size=chunk_size,
            )
            self._meta_by_param[id(param)] = meta
            self._meta_by_name[name] = meta
            self.state[param] = {
                "step": 0,
                "master_param": master_param,
                "exp_avg": exp_avg,
                "exp_avg_sq": exp_avg_sq,
            }

    def _strategy(self) -> DataParallelShardingStrategy:
        return self.config.data_parallel_sharding_strategy

    def _current_hparams(self) -> tuple[float, float, float, float]:
        group0 = self.param_groups[0]
        lr = float(group0["lr"])
        betas = group0["betas"]
        beta1 = float(betas[0])
        beta2 = float(betas[1])
        eps = float(group0["eps"])
        return lr, beta1, beta2, eps

    def _reduce_scatter_or_allreduce_slice(
        self,
        grad_flat: torch.Tensor,
        meta: _ShardMeta,
        group,
    ) -> tuple[torch.Tensor, bool, bool]:
        """
        Produce local averaged grad shard for ZeRO-2.

        Preferred path is reduce-scatter; fallback is all-reduce + local slicing.
        """
        if meta.group_size <= 1:
            return grad_flat[meta.shard_start:meta.shard_end], False, False

        # Correctness fallback used when reduce-scatter is unavailable or failed:
        # all-reduce full grads then slice local ownership range.
        if (
            not self.config.use_reduce_scatter
            or self._reduce_scatter_fallback
            or not hasattr(dist, "reduce_scatter_tensor")
        ):
            dist.all_reduce(grad_flat, op=dist.ReduceOp.SUM, group=group)
            grad_flat.div_(meta.group_size)
            return grad_flat[meta.shard_start:meta.shard_end].clone(), False, True

        # reduce_scatter_tensor expects evenly-sized shards, so pad to
        # chunk_size * group_size before the collective.
        padded_numel = meta.chunk_size * meta.group_size
        if grad_flat.numel() < padded_numel:
            pad = torch.zeros(
                padded_numel - grad_flat.numel(),
                dtype=grad_flat.dtype,
                device=grad_flat.device,
            )
            grad_input = torch.cat([grad_flat, pad], dim=0)
        else:
            grad_input = grad_flat

        out = torch.empty(meta.chunk_size, dtype=grad_flat.dtype, device=grad_flat.device)
        try:
            dist.reduce_scatter_tensor(out, grad_input, op=dist.ReduceOp.SUM, group=group)
            out.div_(meta.group_size)
            return out[: max(0, meta.shard_end - meta.shard_start)].clone(), True, False
        except (RuntimeError, TypeError, AttributeError):
            # If backend/runtime rejects reduce-scatter once, switch to fallback
            # for the rest of the run to keep behavior deterministic.
            self._reduce_scatter_fallback = True
            dist.all_reduce(grad_flat, op=dist.ReduceOp.SUM, group=group)
            grad_flat.div_(meta.group_size)
            return grad_flat[meta.shard_start:meta.shard_end].clone(), False, True

    def _all_gather_param(self, local_shard: torch.Tensor, meta: _ShardMeta, group) -> torch.Tensor:
        """Gather full flattened parameter tensor from local shards."""
        if meta.group_size <= 1:
            return local_shard

        # After sharded local optimizer updates, re-replicate full parameters so
        # the next forward/backward can run with standard model tensors.
        send = torch.zeros(meta.chunk_size, dtype=local_shard.dtype, device=local_shard.device)
        if local_shard.numel() > 0:
            send[: local_shard.numel()].copy_(local_shard)
        gather_list = [torch.empty_like(send) for _ in range(meta.group_size)]
        dist.all_gather(gather_list, send, group=group)
        full = torch.cat(gather_list, dim=0)[: meta.numel]
        return full

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients on wrapped model parameters."""
        for param in self._params:
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                param.grad.zero_()

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Run one ZeRO-1/2 optimizer step on local shards and synchronize updated params."""
        # Contract: backward has already populated `param.grad` on model params.
        self._global_step += 1
        strategy = self._strategy()
        lr, beta1, beta2, eps = self._current_hparams()
        weight_decay = float(self.param_groups[0]["weight_decay"])
        all_reduce_calls = 0
        reduce_scatter_calls = 0
        all_gather_calls = 0
        zero_filled_grad_count = 0
        fallback_all_reduce_calls = 0
        params_with_local_updates = 0

        for _, param in self._named_params:
            # Stage 1: resolve per-parameter metadata/state and ownership.
            meta = self._meta_by_param[id(param)]
            state = self.state[param]
            master_param = state["master_param"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            step = int(state["step"]) + 1
            state["step"] = step

            # Stage 2: choose communication domain (DP for dense, EDP for expert).
            group = (
                self.expert_data_parallel_group
                if meta.is_expert_param
                else self.data_parallel_group
            )
            grad = param.grad
            # Keep collective ordering identical across ranks by materializing a
            # zero grad tensor when this rank has no local grad for this param.
            if grad is None and meta.group_size > 1:
                grad = torch.zeros_like(param)
                zero_filled_grad_count += 1
            if grad is None:
                continue

            grad_flat = grad.detach().to(dtype=self._main_grads_dtype).contiguous().view(-1)
            if strategy == "optim":
                # ZeRO-1: all-reduce averaged full grads, then keep local shard.
                if meta.group_size > 1:
                    dist.all_reduce(grad_flat, op=dist.ReduceOp.SUM, group=group)
                    grad_flat.div_(meta.group_size)
                    all_reduce_calls += 1
                grad_shard = grad_flat[meta.shard_start:meta.shard_end].clone()
            else:
                # ZeRO-2: reduce-scatter averaged grads directly to local shard.
                grad_shard, used_reduce_scatter, used_all_reduce_fallback = (
                    self._reduce_scatter_or_allreduce_slice(grad_flat, meta, group)
                )
                if used_reduce_scatter:
                    reduce_scatter_calls += 1
                if used_all_reduce_fallback:
                    all_reduce_calls += 1
                    fallback_all_reduce_calls += 1

            # Stage 3: local AdamW update on owned FP32 shard state only.
            if master_param.numel() > 0:
                params_with_local_updates += 1
                grad_shard_f = grad_shard.float()
                exp_avg_f = exp_avg.float()
                exp_avg_sq_f = exp_avg_sq.float()
                master_param_f = master_param.float()

                exp_avg_f.mul_(beta1).add_(grad_shard_f, alpha=1.0 - beta1)
                exp_avg_sq_f.mul_(beta2).addcmul_(
                    grad_shard_f,
                    grad_shard_f,
                    value=1.0 - beta2,
                )
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denom = exp_avg_sq_f.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                master_param_f.mul_(1.0 - lr * weight_decay)
                master_param_f.addcdiv_(exp_avg_f, denom, value=-step_size)

                master_param.copy_(master_param_f.to(dtype=self._main_params_dtype))
                exp_avg.copy_(exp_avg_f.to(dtype=self._exp_avg_dtype))
                exp_avg_sq.copy_(exp_avg_sq_f.to(dtype=self._exp_avg_sq_dtype))

            # Stage 4: all-gather updated shards so model parameters are
            # replicated for the next forward pass.
            full_flat = self._all_gather_param(master_param, meta, group)
            if meta.group_size > 1:
                all_gather_calls += 1
            param.data.copy_(full_flat.view(param.shape).to(dtype=param.dtype))

        if self.config.debug and self._global_step <= self.config.debug_max_steps:
            LOGGER.info(
                "[ZeRO Debug][rank=%d] step=%d strategy=%s all_reduce_calls=%d "
                "reduce_scatter_calls=%d all_gather_calls=%d zero_filled_grads=%d "
                "fallback_all_reduce_calls=%d local_param_updates=%d",
                _dist_rank(),
                self._global_step,
                strategy,
                all_reduce_calls,
                reduce_scatter_calls,
                all_gather_calls,
                zero_filled_grad_count,
                fallback_all_reduce_calls,
                params_with_local_updates,
            )

        return True

    def step(self) -> bool:
        """Alias to align with torch optimizer call sites."""
        return self.step_with_ready_grads()

    def _param_signature_items(self) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        for name, param in self._named_params:
            meta = self._meta_by_name[name]
            items.append(
                {
                    "name": name,
                    "shape": list(param.shape),
                    "dtype": str(param.dtype),
                    "is_expert_param": meta.is_expert_param,
                }
            )
        return items

    def parameter_signature_hash(self) -> str:
        """Return a stable hash over parameter names/shapes/domain tags."""
        payload = json.dumps(self._param_signature_items(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def state_dict(self) -> dict:
        """Return non-parameter optimizer metadata state."""
        # This payload is replicated metadata (hyperparameters, signatures), not
        # sharded tensor state. Per-rank tensor shards are saved separately.
        return {
            "format_version": ZERO_CHECKPOINT_FORMAT_VERSION,
            "global_step": self._global_step,
            "config": asdict(self.config),
            "precision_recipe": {
                "name": self.config.precision_recipe_name,
                "fp8_rounding": self.config.fp8_rounding,
                "fp8_activation_quant_granularity": self.config.fp8_activation_quant_granularity,
                "fp8_weight_quant_granularity": self.config.fp8_weight_quant_granularity,
                "fp8_comm_quant_enabled": self.config.fp8_comm_quant_enabled,
                "fp8_comm_quant_granularity": self.config.fp8_comm_quant_granularity,
            },
            "param_groups": [
                {
                    "lr": float(group["lr"]),
                    "weight_decay": float(group["weight_decay"]),
                    "betas": tuple(group["betas"]),
                    "eps": float(group["eps"]),
                }
                for group in self.param_groups
            ],
            "parameter_signature_hash": self.parameter_signature_hash(),
            "parameter_signature_items": self._param_signature_items(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Load non-parameter optimizer metadata state."""
        format_version = int(state.get("format_version", -1))
        if format_version != ZERO_CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                "Unsupported ZeRO nonparam checkpoint format version: "
                f"{format_version}. Expected {ZERO_CHECKPOINT_FORMAT_VERSION}."
            )

        expected_recipe = {
            "name": self.config.precision_recipe_name,
            "fp8_rounding": self.config.fp8_rounding,
            "fp8_activation_quant_granularity": self.config.fp8_activation_quant_granularity,
            "fp8_weight_quant_granularity": self.config.fp8_weight_quant_granularity,
            "fp8_comm_quant_enabled": self.config.fp8_comm_quant_enabled,
            "fp8_comm_quant_granularity": self.config.fp8_comm_quant_granularity,
        }
        loaded_recipe = state.get("precision_recipe")
        if loaded_recipe is not None and loaded_recipe != expected_recipe:
            raise ValueError(
                "ZeRO precision recipe metadata mismatch between checkpoint and optimizer config"
            )

        global_step = int(state.get("global_step", 0))
        self._global_step = global_step

        loaded_groups = state.get("param_groups", [])
        if loaded_groups:
            for current, loaded in zip(self.param_groups, loaded_groups):
                current["lr"] = float(loaded.get("lr", current["lr"]))
                current["weight_decay"] = float(
                    loaded.get("weight_decay", current["weight_decay"])
                )
                current["betas"] = tuple(loaded.get("betas", current["betas"]))
                current["eps"] = float(loaded.get("eps", current["eps"]))

    def get_parameter_state_dp_zero(self, include_tensors: bool = True) -> dict:
        """Return local shard parameter state for distributed checkpointing."""
        # This is the sharded parameter-dependent optimizer state payload.
        payload: dict[str, object] = {
            "format_version": ZERO_CHECKPOINT_FORMAT_VERSION,
            "global_step": self._global_step,
            "strategy": self._strategy(),
            "parameter_signature_hash": self.parameter_signature_hash(),
            "precision_recipe": {
                "name": self.config.precision_recipe_name,
                "fp8_rounding": self.config.fp8_rounding,
                "fp8_activation_quant_granularity": self.config.fp8_activation_quant_granularity,
                "fp8_weight_quant_granularity": self.config.fp8_weight_quant_granularity,
                "fp8_comm_quant_enabled": self.config.fp8_comm_quant_enabled,
                "fp8_comm_quant_granularity": self.config.fp8_comm_quant_granularity,
            },
            "optimizer_state_dtypes": {
                "main_params_dtype": self.config.main_params_dtype,
                "main_grads_dtype": self.config.main_grads_dtype,
                "exp_avg_dtype": self.config.exp_avg_dtype,
                "exp_avg_sq_dtype": self.config.exp_avg_sq_dtype,
            },
            "parameters": {},
        }
        parameters = payload["parameters"]
        assert isinstance(parameters, dict)

        for name, param in self._named_params:
            meta = self._meta_by_name[name]
            state = self.state[param]
            record: dict[str, object] = {
                "name": name,
                "shape": list(meta.shape),
                "dtype": meta.dtype,
                "numel": meta.numel,
                "is_expert_param": meta.is_expert_param,
                "group_size": meta.group_size,
                "group_rank": meta.group_rank,
                "shard_start": meta.shard_start,
                "shard_end": meta.shard_end,
                "chunk_size": meta.chunk_size,
                "step": int(state["step"]),
            }
            if include_tensors:
                record["master_param"] = state["master_param"].detach().cpu()
                record["exp_avg"] = state["exp_avg"].detach().cpu()
                record["exp_avg_sq"] = state["exp_avg_sq"].detach().cpu()
            parameters[name] = record

        return payload

    def load_parameter_state_from_dp_zero(self, state_dict: dict, strict: bool = True) -> None:
        """Load local shard tensors from distributed checkpoint payload."""
        format_version = int(state_dict.get("format_version", -1))
        if format_version != ZERO_CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                "Unsupported ZeRO shard checkpoint format version: "
                f"{format_version}. Expected {ZERO_CHECKPOINT_FORMAT_VERSION}."
            )

        expected_recipe = {
            "name": self.config.precision_recipe_name,
            "fp8_rounding": self.config.fp8_rounding,
            "fp8_activation_quant_granularity": self.config.fp8_activation_quant_granularity,
            "fp8_weight_quant_granularity": self.config.fp8_weight_quant_granularity,
            "fp8_comm_quant_enabled": self.config.fp8_comm_quant_enabled,
            "fp8_comm_quant_granularity": self.config.fp8_comm_quant_granularity,
        }
        loaded_recipe = state_dict.get("precision_recipe")
        if loaded_recipe is not None and loaded_recipe != expected_recipe:
            raise ValueError(
                "ZeRO shard precision recipe metadata mismatch between checkpoint and optimizer config"
            )

        expected_hash = self.parameter_signature_hash()
        found_hash = state_dict.get("parameter_signature_hash")
        if found_hash != expected_hash:
            raise ValueError(
                "Optimizer parameter signature mismatch. "
                f"expected={expected_hash}, found={found_hash}"
            )

        parameters = state_dict.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError("Invalid parameter state payload: missing 'parameters' dict")

        missing = []
        for name, param in self._named_params:
            record = parameters.get(name)
            if record is None:
                missing.append(name)
                continue

            state = self.state[param]
            shard_numel = state["master_param"].numel()
            for key in ("master_param", "exp_avg", "exp_avg_sq"):
                tensor = record.get(key)
                if not torch.is_tensor(tensor):
                    raise ValueError(f"Missing tensor '{key}' for parameter '{name}'")
                target = state[key]
                assert torch.is_tensor(target)
                tensor = tensor.to(device=target.device, dtype=target.dtype)
                if tensor.numel() != shard_numel:
                    raise ValueError(
                        f"Shard shape mismatch for {name}:{key} "
                        f"(expected {shard_numel}, got {tensor.numel()})"
                    )
                target.copy_(tensor.view_as(target))
            state["step"] = int(record.get("step", state["step"]))

        if strict and missing:
            raise ValueError(f"Missing parameter shards in optimizer state: {missing[:5]}")

    def save_parameter_state(self, checkpoint_dir: str, rank: int, world_size: int) -> None:
        """Persist local shard tensors for this rank."""
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
        os.makedirs(checkpoint_dir, exist_ok=True)
        # One rank writes one shard file. Manifest/nonparam metadata are written
        # by the caller alongside these per-rank shard payloads.
        path = os.path.join(checkpoint_dir, f"optimizer_shard_rank{rank}.pt")
        payload = self.get_parameter_state_dp_zero(include_tensors=True)
        payload["rank"] = rank
        payload["world_size"] = world_size
        torch.save(payload, path)

    def load_parameter_state(self, checkpoint_dir: str, rank: int, world_size: int) -> None:
        """Load local shard tensors for this rank from checkpoint dir."""
        path = os.path.join(checkpoint_dir, f"optimizer_shard_rank{rank}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing optimizer shard file: {path}")
        payload = torch.load(path, map_location="cpu")
        shard_world_size = int(payload.get("world_size", world_size))
        if shard_world_size != world_size:
            raise ValueError(
                "Checkpoint world_size mismatch for optimizer shard: "
                f"{shard_world_size} vs {world_size}"
            )
        self.load_parameter_state_from_dp_zero(payload, strict=True)

    def build_checkpoint_manifest(self, rank: int, world_size: int) -> dict[str, object]:
        """Build JSON-serializable manifest for ZeRO checkpoint files."""
        # Manifest ties together replicated metadata file and rank-local shard
        # file names with a parameter signature compatibility hash.
        return {
            "format_version": ZERO_CHECKPOINT_FORMAT_VERSION,
            "optimizer_type": "MegatronZeroOptimizer",
            "rank": rank,
            "world_size": world_size,
            "strategy": self._strategy(),
            "num_distributed_optimizer_instances": self.config.num_distributed_optimizer_instances,
            "parameter_signature_hash": self.parameter_signature_hash(),
            "precision_recipe": {
                "name": self.config.precision_recipe_name,
                "fp8_rounding": self.config.fp8_rounding,
                "fp8_activation_quant_granularity": self.config.fp8_activation_quant_granularity,
                "fp8_weight_quant_granularity": self.config.fp8_weight_quant_granularity,
                "fp8_comm_quant_enabled": self.config.fp8_comm_quant_enabled,
                "fp8_comm_quant_granularity": self.config.fp8_comm_quant_granularity,
            },
            "optimizer_state_dtypes": {
                "main_params_dtype": self.config.main_params_dtype,
                "main_grads_dtype": self.config.main_grads_dtype,
                "exp_avg_dtype": self.config.exp_avg_dtype,
                "exp_avg_sq_dtype": self.config.exp_avg_sq_dtype,
            },
            "files": {
                "nonparam": "optimizer_nonparam.pt",
                "shard": f"optimizer_shard_rank{rank}.pt",
            },
        }


def is_zero_optimizer(optimizer: object) -> bool:
    """Return True when optimizer is ZeRO optimizer-like in this repo."""
    required = (
        "step_with_ready_grads",
        "save_parameter_state",
        "load_parameter_state",
    )
    return all(hasattr(optimizer, attr) for attr in required)
