"""Optimizer-owned low-bit master parameter storage and binding utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from typing import TYPE_CHECKING

import torch.nn as nn

from src.layers import ColumnParallelLinear
from src.layers import RowParallelLinear

if TYPE_CHECKING:
    from src.runtime.contracts import PrecisionConfig


@dataclass(frozen=True)
class MasterBindingMetadata:
    """Metadata describing one optimizer-owned master parameter binding."""

    module_path: str
    parameter_name: str
    shard_domain: str
    shape: tuple[int, ...]


class LowBitMasterStore(nn.Module):
    """Module-attached registry of optimizer-owned master parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.masters = nn.ParameterDict()
        self.metadata: dict[str, MasterBindingMetadata] = {}

    def add_master(
        self,
        *,
        key: str,
        param: nn.Parameter,
        metadata: MasterBindingMetadata,
    ) -> nn.Parameter:
        """Register a master parameter and record binding metadata."""
        self.masters[key] = param
        self.metadata[key] = metadata
        return self.masters[key]


def _infer_shard_domain(module: nn.Module, module_path: str) -> str:
    """Infer gradient synchronization domain for one module weight."""
    if isinstance(module, (ColumnParallelLinear, RowParallelLinear)) and module.tp_size > 1:
        return "tensor_model_parallel"
    if ".experts." in module_path:
        return "expert_model_parallel"
    return "data_parallel"


def materialize_optimizer_owned_masters(
    model: nn.Module,
    *,
    precision_config: "PrecisionConfig",
) -> Optional[LowBitMasterStore]:
    """
    Materialize optimizer-owned low-bit masters and bind them to selected modules.

    The binding is currently applied to low-bit-capable modules selected for persistent
    low-bit parameter storage.
    """
    if precision_config.lowbit_master_ownership != "optimizer":
        return None

    store = LowBitMasterStore()
    binding_count = 0

    for _, module in model.named_modules():
        assignment = getattr(module, "module_precision_assignment", None)
        bind_fn = getattr(module, "bind_optimizer_master_weight", None)
        weight = getattr(module, "weight", None)

        if assignment is None or not callable(bind_fn):
            continue
        if getattr(assignment, "persistent_lowbit_mode", "off") == "off":
            continue
        if not isinstance(weight, nn.Parameter):
            continue

        module_path = str(getattr(module, "module_path", getattr(assignment, "module_name", "")))
        if not module_path:
            continue

        key = f"{module_path.replace('.', '__')}__weight"
        cloned = nn.Parameter(weight.detach().clone())
        shard_domain = _infer_shard_domain(module, module_path)
        bound = store.add_master(
            key=key,
            param=cloned,
            metadata=MasterBindingMetadata(
                module_path=module_path,
                parameter_name="weight",
                shard_domain=shard_domain,
                shape=tuple(weight.shape),
            ),
        )
        bind_fn(bound)
        binding_count += 1

    if binding_count == 0:
        return None

    model.add_module("_lowbit_master_store", store)
    return store


def get_lowbit_master_store(model: nn.Module) -> Optional[LowBitMasterStore]:
    """Return attached optimizer-owned master store when present."""
    store = getattr(model, "_lowbit_master_store", None)
    if isinstance(store, LowBitMasterStore):
        return store
    return None
