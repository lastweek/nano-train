#!/usr/bin/env python3
"""
Canonical TP+PP+EP+DP tutorial script with runtime-core orchestration.

This script owns CLI parsing, compatibility wrappers, and runtime components.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import sys
from typing import Optional

import torch
import torch.distributed as dist  # Exposed for tests that monkeypatch this symbol.
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler

# Add parent directory to path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import create_dataloader
from src.distributed.topology import ModelParallelTopology
from src.distributed.topology import initialize_model_parallel
from src.distributed.zero import DistributedOptimizerConfig
from src.distributed.zero import MegatronZeroOptimizer
from src.logging import get_logger
from src.logging import setup_logging
from src.models.deepseek import DeepSeekModel
from src.models.deepseek import DeepSeekModelConfig
from src.models.deepseek import DeepSeekParallelContext
from src.models.moe import ExpertParallelMoE
from src.models.moe import LocalRoutedMoE
from src.runtime.checkpoint import NoOpCheckpointManager
from src.runtime.context import RunConfig
from src.runtime.context import RuntimeContext
from src.runtime.contracts import OptimizerState
from src.runtime.contracts import ModulePrecisionResolver
from src.runtime.contracts import PrecisionConfig
from src.runtime.contracts import ResumeState
from src.runtime.contracts import RuntimeBootstrap
from src.runtime.contracts import RuntimeComponents
from src.runtime.contracts import ScheduleStrategy
from src.runtime.contracts import StepContext
from src.runtime.contracts import StepOutput
from src.runtime.contracts import TrainDataBundle
from src.runtime.engine import RuntimeEngine
from src.runtime.mixed_precision import MixedPrecisionController
from src.runtime.mixed_precision import build_module_precision_resolver
from src.runtime.mixed_precision import dtype_alias_to_torch
from src.runtime.mixed_precision import finalize_module_precision_resolver
from src.runtime.mixed_precision import refresh_persistent_lowbit_params
from src.runtime.master_store import materialize_optimizer_owned_masters
from src.runtime.optimizer_runtime import step_with_sync_policy
from src.runtime.optimizer_runtime import PrecisionAdamW
from src.runtime.optimizer_runtime import zero_grad_optimizer
from src.runtime.pipeline import train_step_pipeline as runtime_train_step_pipeline
from src.runtime.precision_args import add_mixed_precision_args
from src.runtime.precision_args import normalize_and_resolve_precision
from src.runtime import sync as runtime_sync
from src.runtime import validation as runtime_validation
from src.runtime.schedules.non_pipeline import NonPipelineSchedule
from src.runtime.schedules.pipeline_1f1b import Pipeline1F1BSchedule
from src.runtime.schedules.selector import DefaultScheduleSelector


logger = get_logger("ep")

# Backward-compatible exports expected by tests and existing call sites.
ParamShardInfo = runtime_sync.ParamShardInfo


def parse_pp_layer_splits(raw_splits: Optional[str]) -> Optional[tuple[int, ...]]:
    """Parse optional comma-separated PP layer boundaries into a tuple."""
    return runtime_validation.parse_pp_layer_splits(raw_splits)


def validate_args(
    args: argparse.Namespace,
    world_size: int,
    pp_layer_splits: Optional[tuple[int, ...]],
) -> None:
    """Validate args that depend on global world size and topology choices."""
    runtime_validation.validate_args(args, world_size, pp_layer_splits)


def collect_param_shard_info(
    model: nn.Module,
    tensor_model_parallel_group: object,
) -> ParamShardInfo:
    """Compatibility wrapper for runtime sync shard classification."""
    return runtime_sync.collect_param_shard_info(model, tensor_model_parallel_group)


def synchronize_initial_parameters(
    model: nn.Module,
    shard_info: ParamShardInfo,
    parallel: ModelParallelTopology,
) -> None:
    """Compatibility wrapper for runtime initial parameter synchronization."""
    runtime_sync.synchronize_initial_parameters(
        model=model,
        shard_info=shard_info,
        parallel=parallel,
    )


def synchronize_gradients(
    model: nn.Module,
    shard_info: ParamShardInfo,
    data_parallel_size: int,
    expert_data_parallel_size: int,
    data_parallel_group: object,
    expert_data_parallel_group: object,
) -> None:
    """Compatibility wrapper for runtime gradient synchronization."""
    runtime_sync.synchronize_gradients(
        model=model,
        shard_info=shard_info,
        data_parallel_size=data_parallel_size,
        expert_data_parallel_size=expert_data_parallel_size,
        data_parallel_group=data_parallel_group,
        expert_data_parallel_group=expert_data_parallel_group,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for TP+PP+EP+DP tutorial training."""
    parser = argparse.ArgumentParser(description="Canonical TP+PP+EP+DP tutorial script")
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        required=True,
        help="Tensor model parallel size",
    )
    parser.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        required=True,
        help="Pipeline model parallel size",
    )
    parser.add_argument(
        "--expert-model-parallel-size",
        type=int,
        required=True,
        help="Expert model parallel size",
    )
    parser.add_argument(
        "--context-parallel-size",
        type=int,
        default=1,
        help="Context parallel size (currently must be 1 in this tutorial)",
    )
    parser.add_argument(
        "--expert-tensor-parallel-size",
        type=int,
        default=1,
        help="Expert tensor parallel size (tutorial currently supports only 1)",
    )
    parser.add_argument(
        "--num-layers-per-virtual-pipeline-stage",
        type=int,
        default=None,
        help="Reserved virtual PP setting (not used in this tutorial)",
    )
    parser.add_argument(
        "--use-distributed-optimizer",
        action="store_true",
        help="Enable Megatron-style ZeRO optimizer path",
    )
    parser.add_argument(
        "--data-parallel-sharding-strategy",
        type=str,
        default="no_shard",
        help="Megatron-style DP sharding strategy: no_shard | optim | optim_grads",
    )
    parser.add_argument(
        "--num-distributed-optimizer-instances",
        type=int,
        default=1,
        help="Number of distributed optimizer instances (currently supports only 1)",
    )
    parser.add_argument(
        "--zero-debug",
        action="store_true",
        help="Enable verbose ZeRO debug logging in distributed optimizer",
    )
    parser.add_argument(
        "--zero-debug-max-steps",
        type=int,
        default=1,
        help="Number of early optimizer steps to emit ZeRO debug counters",
    )
    parser.add_argument(
        "--zero-debug-max-params",
        type=int,
        default=8,
        help="Number of parameter shard mappings to print at ZeRO init",
    )
    parser.add_argument(
        "--num_microbatches",
        type=int,
        default=1,
        help="Pipeline microbatch count for 1F1B schedule",
    )
    parser.add_argument(
        "--pp_layer_splits",
        type=str,
        default=None,
        help="Optional PP layer boundaries, e.g. '0,2,5'",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--intermediate_size", type=int, default=256)
    parser.add_argument("--moe_intermediate_size", type=int, default=128)
    parser.add_argument("--q_lora_rank", type=int, default=64)
    parser.add_argument("--kv_lora_rank", type=int, default=32)
    parser.add_argument("--qk_nope_head_dim", type=int, default=8)
    parser.add_argument("--qk_rope_head_dim", type=int, default=8)
    parser.add_argument("--v_head_dim", type=int, default=8)
    parser.add_argument("--max_position_embeddings", type=int, default=256)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--n_group", type=int, default=4)
    parser.add_argument("--topk_group", type=int, default=2)
    parser.add_argument("--n_shared_experts", type=int, default=1)
    parser.add_argument("--first_k_dense_replace", type=int, default=1)
    parser.add_argument("--moe_layer_freq", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--capacity_factor", type=float, default=1.0)
    parser.add_argument("--aux_loss_coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    add_mixed_precision_args(parser)
    return parser.parse_args()


class DummyTokenDataset(Dataset):
    """Deterministic token dataset for causal LM communication demos."""

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(num_samples, seq_len),
            generator=generator,
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[index]}


def build_tiny_deepseek_config(
    args: argparse.Namespace,
    *,
    param_dtype: torch.dtype,
    param_device: torch.device | None,
    precision_resolver: ModulePrecisionResolver,
    module_compute_dtype_overrides: dict[str, str] | None = None,
) -> DeepSeekModelConfig:
    """Build a small DeepSeek config for TP/PP/EP/DP learning runs."""
    return DeepSeekModelConfig(
        param_dtype=param_dtype,
        param_device=param_device,
        precision_resolver=precision_resolver,
        module_compute_dtype_overrides=(
            {} if module_compute_dtype_overrides is None else module_compute_dtype_overrides
        ),
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=max(args.max_position_embeddings, args.seq_len),
        q_lora_rank=args.q_lora_rank,
        kv_lora_rank=args.kv_lora_rank,
        qk_nope_head_dim=args.qk_nope_head_dim,
        qk_rope_head_dim=args.qk_rope_head_dim,
        v_head_dim=args.v_head_dim,
        attention_dropout=args.dropout,
        dropout=args.dropout,
        moe_intermediate_size=args.moe_intermediate_size,
        n_routed_experts=args.num_experts,
        n_shared_experts=args.n_shared_experts,
        num_experts_per_tok=args.top_k,
        first_k_dense_replace=args.first_k_dense_replace,
        moe_layer_freq=args.moe_layer_freq,
        scoring_func="sigmoid",
        n_group=args.n_group,
        topk_group=args.topk_group,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
    )


def create_data_loader(
    dataset: Dataset,
    batch_size: int,
    dp_size: int,
    dp_rank: int,
    seed: int,
) -> TrainDataBundle:
    """Create a dataloader with DP-aware sharding."""
    sampler: Optional[DistributedSampler] = None
    if dp_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dp_size,
            rank=dp_rank,
            shuffle=True,
            seed=seed,
        )

    loader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        sampler=sampler,
    )
    return TrainDataBundle(loader=loader, sampler=sampler)


def gather_moe_metrics(model: nn.Module, device: torch.device) -> tuple[torch.Tensor, float]:
    """Collect auxiliary MoE loss and dropped fraction from all MoE modules."""
    aux_terms: list[torch.Tensor] = []
    drop_terms: list[float] = []

    for module in model.modules():
        if isinstance(module, (ExpertParallelMoE, LocalRoutedMoE)):
            aux_terms.append(module.last_aux_loss)
            drop_terms.append(module.last_dropped_fraction)

    if aux_terms:
        aux_loss = torch.stack([term.reshape(()) for term in aux_terms]).sum()
    else:
        aux_loss = torch.zeros((), device=device)

    avg_drop = float(sum(drop_terms) / len(drop_terms)) if drop_terms else 0.0
    return aux_loss, avg_drop


def _build_model_for_context(ctx: RuntimeContext) -> DeepSeekModel:
    """Build DeepSeek model for the current distributed context."""
    args = ctx.run_config.args
    parallel = ctx.parallel
    seed_offset = (
        parallel.pipeline_model_parallel_rank * 1_000_003
        + parallel.tensor_model_parallel_rank * 1009
        + parallel.expert_model_parallel_rank
    )
    torch.manual_seed(args.seed + seed_offset)
    if parallel.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + seed_offset)

    precision_config = ctx.run_config.precision_config
    if precision_config is None:
        raise RuntimeError("precision_config must be resolved in Train4PBootstrap")
    param_dtype = dtype_alias_to_torch(precision_config.params_dtype)

    model_config = build_tiny_deepseek_config(
        args,
        param_dtype=param_dtype,
        param_device=parallel.device,
        precision_resolver=build_module_precision_resolver(precision_config),
        module_compute_dtype_overrides={
            # Example (exact module path): force this norm to fp16 while
            # linears still follow low-bit policy.
            # "blocks.0.attn.q_a_norm": "fp16",
        },
    )
    parallel_context = DeepSeekParallelContext(
        tensor_model_parallel_rank=parallel.tensor_model_parallel_rank,
        tensor_model_parallel_size=parallel.tensor_model_parallel_size,
        tensor_model_parallel_group=parallel.tensor_model_parallel_group,
        expert_model_parallel_rank=parallel.expert_model_parallel_rank,
        expert_model_parallel_size=parallel.expert_model_parallel_size,
        expert_model_parallel_group=parallel.expert_model_parallel_group,
        pipeline_model_parallel_rank=parallel.pipeline_model_parallel_rank,
        pipeline_model_parallel_size=parallel.pipeline_model_parallel_size,
        pipeline_model_parallel_group=parallel.pipeline_model_parallel_group,
        context_parallel_rank=parallel.context_parallel_rank,
        context_parallel_size=parallel.context_parallel_size,
        context_parallel_group=parallel.context_parallel_group,
        pp_layer_splits=ctx.run_config.pp_layer_splits,
        capacity_factor=args.capacity_factor,
        expert_tensor_parallel_size=args.expert_tensor_parallel_size,
        sequence_parallel=True,
    )
    model = DeepSeekModel(model_config, parallel_context=parallel_context)
    precision_summary = finalize_module_precision_resolver(model_config.precision_resolver)
    master_store = materialize_optimizer_owned_masters(
        model,
        precision_config=precision_config,
    )
    if parallel.rank == 0:
        logger.info(
            "Per-module low-bit policy: compute_modules=%d persistent_modules=%d "
            "high_precision_exceptions=%d",
            precision_summary.compute_lowbit_module_count,
            precision_summary.persistent_lowbit_module_count,
            precision_summary.high_precision_exception_module_count,
        )
        logger.info(
            "Low-bit master ownership: mode=%s bound_modules=%d",
            precision_config.lowbit_master_ownership,
            0 if master_store is None else len(master_store.metadata),
        )
    refresh_persistent_lowbit_params(model)

    return model


def _build_optimizer(
    *,
    model: DeepSeekModel,
    shard_info: ParamShardInfo,
    ctx: RuntimeContext,
    precision_config: PrecisionConfig,
) -> object:
    """Build ZeRO or AdamW optimizer based on CLI flags."""
    args = ctx.run_config.args
    parallel = ctx.parallel
    if args.use_distributed_optimizer:
        recipe = precision_config.deepseek_v3_recipe
        return MegatronZeroOptimizer(
            model=model,
            config=DistributedOptimizerConfig(
                use_distributed_optimizer=True,
                data_parallel_sharding_strategy=args.data_parallel_sharding_strategy,
                num_distributed_optimizer_instances=args.num_distributed_optimizer_instances,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8,
                use_reduce_scatter=True,
                main_params_dtype=precision_config.main_params_dtype,
                main_grads_dtype=precision_config.main_grads_dtype,
                exp_avg_dtype=precision_config.exp_avg_dtype,
                exp_avg_sq_dtype=precision_config.exp_avg_sq_dtype,
                precision_recipe_name=precision_config.precision_recipe_name,
                fp8_rounding="nearest" if recipe is None else recipe.rounding_mode,
                fp8_activation_quant_granularity=(
                    "tensor" if recipe is None else recipe.activation_quant_granularity
                ),
                fp8_weight_quant_granularity=(
                    "tensor" if recipe is None else recipe.weight_quant_granularity
                ),
                fp8_comm_quant_enabled=False if recipe is None else recipe.comm_quant_enabled,
                fp8_comm_quant_granularity=(
                    "tensor" if recipe is None else recipe.comm_quant_granularity
                ),
                debug=args.zero_debug,
                debug_max_steps=args.zero_debug_max_steps,
                debug_max_params=args.zero_debug_max_params,
            ),
            data_parallel_group=parallel.data_parallel_group,
            expert_data_parallel_group=parallel.expert_data_parallel_group,
            expert_param_ids=shard_info.expert_model_parallel_sharded_param_ids,
        )

    return PrecisionAdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        main_params_dtype=precision_config.main_params_dtype,
        main_grads_dtype=precision_config.main_grads_dtype,
        exp_avg_dtype=precision_config.exp_avg_dtype,
        exp_avg_sq_dtype=precision_config.exp_avg_sq_dtype,
    )


def _log_training_start(
    ctx: RuntimeContext,
    model: DeepSeekModel,
    precision_config: PrecisionConfig,
) -> None:
    """Log rank-0 training configuration summary."""
    args = ctx.run_config.args
    parallel = ctx.parallel
    if parallel.rank != 0:
        return

    local_params = sum(parameter.numel() for parameter in model.parameters())
    logger.info("Local params: %d", local_params)
    logger.info(
        "DeepSeek tiny config: layers=%d hidden=%d experts=%d top_k=%d seq_len=%d",
        args.num_layers,
        args.hidden_size,
        args.num_experts,
        args.top_k,
        args.seq_len,
    )
    logger.info(
        "Model context: tensor=%d/%d pipeline=%d/%d expert=%d/%d data=%d/%d context=%d/%d",
        parallel.tensor_model_parallel_rank,
        parallel.tensor_model_parallel_size,
        parallel.pipeline_model_parallel_rank,
        parallel.pipeline_model_parallel_size,
        parallel.expert_model_parallel_rank,
        parallel.expert_model_parallel_size,
        parallel.data_parallel_rank,
        parallel.data_parallel_size,
        parallel.context_parallel_rank,
        parallel.context_parallel_size,
    )
    logger.info(
        "Distributed optimizer: enabled=%s strategy=%s dist_opt_instances=%d "
        "zero_debug=%s zero_debug_max_steps=%d zero_debug_max_params=%d",
        args.use_distributed_optimizer,
        args.data_parallel_sharding_strategy,
        args.num_distributed_optimizer_instances,
        args.zero_debug,
        args.zero_debug_max_steps,
        args.zero_debug_max_params,
    )
    logger.info(
        "Precision: mode=%s recipe=%s activation_dtype=%s params_dtype=%s main_params_dtype=%s "
        "main_grads_dtype=%s exp_avg_dtype=%s exp_avg_sq_dtype=%s fp8_backend=%s fp4_backend=%s "
        "fp8_param=%s fp4_param=%s persistent_scale=%s",
        precision_config.mode,
        precision_config.precision_recipe_name,
        precision_config.activation_dtype,
        precision_config.params_dtype,
        precision_config.main_params_dtype,
        precision_config.main_grads_dtype,
        precision_config.exp_avg_dtype,
        precision_config.exp_avg_sq_dtype,
        precision_config.fp8_backend,
        precision_config.fp4_backend,
        precision_config.fp8_param,
        precision_config.fp4_param,
        precision_config.persistent_scale_granularity,
    )
    if precision_config.deepseek_v3_recipe is not None:
        recipe = precision_config.deepseek_v3_recipe
        logger.info(
            "DeepSeek-V3 recipe: act_granularity=%s weight_granularity=%s rounding=%s "
            "comm_quant=%s comm_granularity=%s",
            recipe.activation_quant_granularity,
            recipe.weight_quant_granularity,
            recipe.rounding_mode,
            recipe.comm_quant_enabled,
            recipe.comm_quant_granularity,
        )
    logger.info("Starting training: epochs=%d, max_steps=%d", args.epochs, args.max_steps)


def _infer_mode(parallel: ModelParallelTopology) -> str:
    """Infer high-level runtime mode from topology."""
    if parallel.world_size == 1:
        return "single"
    if parallel.pipeline_model_parallel_size > 1:
        return "tp_pp_ep_dp"
    if parallel.tensor_model_parallel_size == 1 and parallel.expert_model_parallel_size > 1:
        return "ep_only"
    return "tp_ep_dp"


@dataclass
class Train4PBootstrap(RuntimeBootstrap):
    """Build runtime context for train_4d CLI arguments."""

    def build_context(self, args: argparse.Namespace) -> RuntimeContext:
        """Validate args, initialize topology, and construct runtime context."""
        pp_layer_splits = parse_pp_layer_splits(getattr(args, "pp_layer_splits", None))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        validate_args(args, world_size, pp_layer_splits)

        parallel = initialize_model_parallel(
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            expert_model_parallel_size=args.expert_model_parallel_size,
            context_parallel_size=args.context_parallel_size,
        )
        precision_config = normalize_and_resolve_precision(args, parallel.device)
        return RuntimeContext(
            parallel=parallel,
            mode=_infer_mode(parallel),
            run_config=RunConfig(
                args=args,
                pp_layer_splits=pp_layer_splits,
                precision_config=precision_config,
            ),
        )


@dataclass
class Train4PModelProvider:
    """Model provider for train_4d DeepSeek tutorial stack."""

    def build_model(self, ctx: RuntimeContext) -> torch.nn.Module:
        """Build the train_4d model for the current rank context."""
        return _build_model_for_context(ctx)


@dataclass
class Train4PDataProvider:
    """Data provider for train_4d deterministic token batches."""

    def build_train_data(self, ctx: RuntimeContext) -> TrainDataBundle:
        """Build deterministic token data for the current DP shard."""
        args = ctx.run_config.args
        parallel = ctx.parallel
        dataset = DummyTokenDataset(
            num_samples=args.num_samples,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            seed=args.seed,
        )
        return create_data_loader(
            dataset=dataset,
            batch_size=args.batch_size,
            dp_size=parallel.data_parallel_size,
            dp_rank=parallel.data_parallel_rank,
            seed=args.seed,
        )


@dataclass
class Train4POptimizerRuntime:
    """Optimizer runtime for train_4d ZeRO and DP/EDP sync policies."""

    def initialize(self, model: torch.nn.Module, ctx: RuntimeContext) -> OptimizerState:
        """Initialize shard metadata, parameter sync, and optimizer state."""
        if not isinstance(model, DeepSeekModel):
            raise TypeError("Train4POptimizerRuntime requires a DeepSeekModel")
        precision_config = ctx.run_config.precision_config
        if precision_config is None:
            raise RuntimeError("precision_config must be resolved in Train4PBootstrap")

        shard_info = collect_param_shard_info(
            model=model,
            tensor_model_parallel_group=ctx.parallel.tensor_model_parallel_group,
        )
        synchronize_initial_parameters(
            model=model,
            shard_info=shard_info,
            parallel=ctx.parallel,
        )
        optimizer = _build_optimizer(
            model=model,
            shard_info=shard_info,
            ctx=ctx,
            precision_config=precision_config,
        )
        precision_controller = MixedPrecisionController(
            precision_config,
            device=ctx.parallel.device,
        )
        _log_training_start(ctx, model, precision_config)
        return OptimizerState(
            optimizer=optimizer,
            shard_info=shard_info,
            extra_state={"precision_controller": precision_controller},
        )

    def zero_grad(self, state: OptimizerState) -> None:
        """Zero optimizer gradients with runtime helper behavior."""
        zero_grad_optimizer(state)

    def step(
        self,
        *,
        model: torch.nn.Module,
        state: OptimizerState,
        ctx: RuntimeContext,
    ) -> None:
        """Apply ZeRO-aware step policy and required gradient synchronization."""
        step_with_sync_policy(
            model=model,
            state=state,
            ctx=ctx,
            synchronize_gradients_fn=synchronize_gradients,
        )


@dataclass
class Train4PNonPipelineSchedule:
    """Schedule implementation for non-pipeline train_4d steps."""

    optimizer_runtime: Train4POptimizerRuntime

    def run_step(self, step_ctx: StepContext) -> StepOutput:
        """Execute one non-pipeline language-model train step."""
        if step_ctx.batch is None:
            raise RuntimeError("Non-pipeline schedule requires a batch")

        args = step_ctx.runtime_context.run_config.args
        device = step_ctx.runtime_context.parallel.device
        local_input_ids = step_ctx.batch["input_ids"].to(device)
        precision_controller = step_ctx.optimizer_state.extra_state.get("precision_controller")
        if precision_controller is not None and not isinstance(
            precision_controller, MixedPrecisionController
        ):
            raise TypeError("precision_controller must be MixedPrecisionController when provided")

        self.optimizer_runtime.zero_grad(step_ctx.optimizer_state)

        if precision_controller is None:
            logits = step_ctx.model(local_input_ids)
            if local_input_ids.size(0) == 0 or local_input_ids.size(1) < 2:
                task_loss = logits.sum() * 0.0
            else:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = local_input_ids[:, 1:].contiguous()
                task_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                )
            moe_aux_loss, drop_fraction = gather_moe_metrics(step_ctx.model, device=device)
            total_loss = task_loss + (args.aux_loss_coef * moe_aux_loss)
            total_loss.backward()
            should_step = True
        else:
            with precision_controller.autocast_context():
                logits = step_ctx.model(local_input_ids)
                if local_input_ids.size(0) == 0 or local_input_ids.size(1) < 2:
                    task_loss = logits.sum() * 0.0
                else:
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = local_input_ids[:, 1:].contiguous()
                    task_loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.reshape(-1),
                    )
                moe_aux_loss, drop_fraction = gather_moe_metrics(step_ctx.model, device=device)
                total_loss = task_loss + (args.aux_loss_coef * moe_aux_loss)
            precision_controller.backward(total_loss)
            should_step = precision_controller.prepare_optimizer_step(step_ctx.model)

        if should_step:
            self.optimizer_runtime.step(
                model=step_ctx.model,
                state=step_ctx.optimizer_state,
                ctx=step_ctx.runtime_context,
            )
            refresh_persistent_lowbit_params(step_ctx.model)

        if precision_controller is not None:
            precision_controller.update_after_step(step_applied=should_step)
            if (
                step_ctx.runtime_context.parallel.rank == 0
                and args.log_every > 0
                and step_ctx.train_state.global_step % args.log_every == 0
                and precision_controller.uses_loss_scaling
            ):
                logger.info(
                    "precision step=%d mode=%s loss_scale=%.4f skipped_steps=%d",
                    step_ctx.train_state.global_step,
                    precision_controller.config.mode,
                    precision_controller.runtime_state.loss_scale,
                    precision_controller.runtime_state.skipped_steps,
                )

        return StepOutput(
            task_loss=float(task_loss.item()),
            aux_loss=float(moe_aux_loss.item()),
            total_loss=float(total_loss.item()),
            drop_fraction=drop_fraction,
            counters={"objective_count": 1, "drop_count": 1},
        )


@dataclass
class Train4PPipelineSchedule:
    """Schedule implementation for train_4d non-interleaved 1F1B pipeline."""

    optimizer_runtime: Train4POptimizerRuntime

    def run_step(self, step_ctx: StepContext) -> StepOutput:
        """Execute one non-interleaved 1F1B pipeline train step."""
        args = step_ctx.runtime_context.run_config.args
        parallel = step_ctx.runtime_context.parallel
        precision_config = step_ctx.runtime_context.run_config.precision_config
        model = step_ctx.model
        if not isinstance(model, DeepSeekModel):
            raise TypeError("Train4PPipelineSchedule requires a DeepSeekModel")
        precision_controller = step_ctx.optimizer_state.extra_state.get("precision_controller")
        if precision_controller is not None and not isinstance(
            precision_controller, MixedPrecisionController
        ):
            raise TypeError("precision_controller must be MixedPrecisionController when provided")

        def _apply_optimizer_step(**kwargs) -> None:
            del kwargs
            self.optimizer_runtime.step(
                model=step_ctx.model,
                state=step_ctx.optimizer_state,
                ctx=step_ctx.runtime_context,
            )

        task_sum, aux_sum, drop_sum, objective_count, drop_count = runtime_train_step_pipeline(
            model=model,
            optimizer=step_ctx.optimizer_state.optimizer,
            use_distributed_optimizer=args.use_distributed_optimizer,
            batch=step_ctx.batch,
            parallel=parallel,
            num_microbatches=args.num_microbatches,
            expected_local_batch=args.batch_size,
            seq_len=args.seq_len,
            aux_loss_coef=args.aux_loss_coef,
            shard_info=step_ctx.optimizer_state.shard_info,
            gather_moe_metrics_fn=gather_moe_metrics,
            apply_optimizer_step_fn=_apply_optimizer_step,
            sync_plugin=None,
            zero_grad_fn=lambda _: self.optimizer_runtime.zero_grad(step_ctx.optimizer_state),
            refresh_persistent_params_fn=refresh_persistent_lowbit_params,
            precision_controller=precision_controller,
            precision_config=precision_config,
        )
        if (
            precision_controller is not None
            and parallel.rank == 0
            and args.log_every > 0
            and step_ctx.train_state.global_step % args.log_every == 0
            and precision_controller.uses_loss_scaling
        ):
            logger.info(
                "precision step=%d mode=%s loss_scale=%.4f skipped_steps=%d",
                step_ctx.train_state.global_step,
                precision_controller.config.mode,
                precision_controller.runtime_state.loss_scale,
                precision_controller.runtime_state.skipped_steps,
            )
        total_sum = task_sum + (args.aux_loss_coef * aux_sum)
        return StepOutput(
            task_loss=task_sum,
            aux_loss=aux_sum,
            total_loss=total_sum,
            drop_fraction=drop_sum,
            counters={"objective_count": objective_count, "drop_count": drop_count},
        )


@dataclass
class Train4PScheduleSelector:
    """Select train_4d schedule based on pipeline model-parallel size."""

    optimizer_runtime: Train4POptimizerRuntime

    def __post_init__(self) -> None:
        self._selector = DefaultScheduleSelector(
            non_pipeline_schedule=NonPipelineSchedule(
                step_fn=Train4PNonPipelineSchedule(self.optimizer_runtime).run_step
            ),
            pipeline_schedule=Pipeline1F1BSchedule(
                step_fn=Train4PPipelineSchedule(self.optimizer_runtime).run_step
            ),
        )

    def select(self, ctx: RuntimeContext) -> ScheduleStrategy:
        """Select non-pipeline or pipeline schedule from runtime topology."""
        return self._selector.select(ctx)


class Train4PCheckpointManager(NoOpCheckpointManager):
    """Checkpoint manager for train_4d runtime path (no-op save/load parity)."""

    def load(
        self,
        *,
        model: torch.nn.Module,
        optimizer_state: OptimizerState,
        ctx: RuntimeContext,
    ) -> ResumeState:
        """Return no-op resume state for tutorial save/load parity."""
        del model, optimizer_state, ctx
        return ResumeState()


def build_train_4d_components() -> RuntimeComponents:
    """Build runtime component bundle for train_4d tutorial."""
    optimizer_runtime = Train4POptimizerRuntime()
    return RuntimeComponents(
        bootstrap=Train4PBootstrap(),
        model_provider=Train4PModelProvider(),
        data_provider=Train4PDataProvider(),
        optimizer_runtime=optimizer_runtime,
        schedule_selector=Train4PScheduleSelector(optimizer_runtime=optimizer_runtime),
        checkpoint_manager=Train4PCheckpointManager(),
    )


def setup_process_logging(rank: int) -> None:
    """Configure process-local logging using shared logging utilities."""
    level = "INFO" if rank == 0 else "WARNING"
    setup_logging(log_level=level, use_colors=False)


def main() -> None:
    """Run canonical TP+PP+EP+DP tutorial training with runtime orchestration."""
    args = parse_args()

    rank_pre = int(os.environ.get("RANK", "0"))
    setup_process_logging(rank_pre)

    components = build_train_4d_components()
    engine = RuntimeEngine()
    engine.run(components=components, args=args)


if __name__ == "__main__":
    main()
