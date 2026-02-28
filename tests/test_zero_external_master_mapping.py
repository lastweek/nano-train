"""Shard-domain mapping coverage for optimizer-owned external master params."""

from __future__ import annotations

import torch.nn as nn

from src.runtime.master_store import LowBitMasterStore
from src.runtime.master_store import MasterBindingMetadata
from src.runtime.sync import collect_param_shard_info


class _StoreOnlyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)


def test_collect_param_shard_info_includes_external_master_store_domains() -> None:
    model = _StoreOnlyModel()
    store = LowBitMasterStore()
    tp_master = store.add_master(
        key="tp_weight",
        param=nn.Parameter(model.linear.weight.detach().clone()),
        metadata=MasterBindingMetadata(
            module_path="blocks.0.mlp.fc1",
            parameter_name="weight",
            shard_domain="tensor_model_parallel",
            shape=tuple(model.linear.weight.shape),
        ),
    )
    expert_master = store.add_master(
        key="expert_weight",
        param=nn.Parameter(model.linear.weight.detach().clone()),
        metadata=MasterBindingMetadata(
            module_path="moe.experts.0.fc1",
            parameter_name="weight",
            shard_domain="expert_model_parallel",
            shape=tuple(model.linear.weight.shape),
        ),
    )
    model.add_module("_lowbit_master_store", store)

    shard_info = collect_param_shard_info(model, tensor_model_parallel_group=None)
    assert id(tp_master) in shard_info.tensor_model_parallel_sharded_param_ids
    assert id(expert_master) in shard_info.expert_model_parallel_sharded_param_ids
