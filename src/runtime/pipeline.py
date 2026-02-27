"""Pipeline schedule helpers for runtime orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import TYPE_CHECKING
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from src.distributed.topology import ModelParallelTopology
from src.models.deepseek import DeepSeekModel
from src.runtime.contracts import PrecisionConfig
from src.runtime.mixed_precision import dtype_alias_to_torch
from src.runtime.sync import ParamShardInfo

if TYPE_CHECKING:
    from src.runtime.mixed_precision import MixedPrecisionController


def activation_tag(microbatch_idx: int) -> int:
    """Tag id used for activation p2p traffic."""
    return 10_000 + microbatch_idx


def grad_tag(microbatch_idx: int) -> int:
    """Tag id used for activation-gradient p2p traffic."""
    return 20_000 + microbatch_idx


def label_tag(microbatch_idx: int) -> int:
    """Tag id used for label p2p traffic."""
    return 30_000 + microbatch_idx


@dataclass
class PipelinePeers:
    """Resolved PP peer ranks for this rank's pipeline chain."""

    prev_rank: Optional[int]
    next_rank: Optional[int]
    first_stage_rank: int
    last_stage_rank: int


@dataclass
class PipelineStepState:
    """Mutable state containers used across 1F1B forward/backward phases."""

    stage_inputs: list[Optional[torch.Tensor]] = field(default_factory=list)
    stage_outputs: list[torch.Tensor] = field(default_factory=list)
    stage_aux_losses: list[torch.Tensor] = field(default_factory=list)
    stage_labels: list[Optional[torch.Tensor]] = field(default_factory=list)
    label_send_reqs: list[dist.Work] = field(default_factory=list)
    activation_send_reqs: list[dist.Work] = field(default_factory=list)
    activation_send_buffers: list[torch.Tensor] = field(default_factory=list)
    grad_send_reqs: list[dist.Work] = field(default_factory=list)
    grad_send_buffers: list[torch.Tensor] = field(default_factory=list)
    task_loss_sum: float = 0.0
    aux_loss_sum: float = 0.0
    drop_sum: float = 0.0
    drop_count: int = 0


def resolve_pipeline_peers(parallel: ModelParallelTopology) -> PipelinePeers:
    """Resolve stage-neighbor ranks and stage endpoints for the local PP chain."""
    prev_pp_rank = parallel.pipeline_model_parallel_rank - 1
    next_pp_rank = parallel.pipeline_model_parallel_rank + 1

    prev_rank = None
    if prev_pp_rank >= 0:
        prev_rank = parallel.rank_from_coords(
            data_parallel_rank=parallel.data_parallel_rank,
            pipeline_model_parallel_rank=prev_pp_rank,
            tensor_model_parallel_rank=parallel.tensor_model_parallel_rank,
            expert_model_parallel_rank=parallel.expert_model_parallel_rank,
            context_parallel_rank=parallel.context_parallel_rank,
        )

    next_rank = None
    if next_pp_rank < parallel.pipeline_model_parallel_size:
        next_rank = parallel.rank_from_coords(
            data_parallel_rank=parallel.data_parallel_rank,
            pipeline_model_parallel_rank=next_pp_rank,
            tensor_model_parallel_rank=parallel.tensor_model_parallel_rank,
            expert_model_parallel_rank=parallel.expert_model_parallel_rank,
            context_parallel_rank=parallel.context_parallel_rank,
        )

    first_stage_rank = parallel.rank_from_coords(
        data_parallel_rank=parallel.data_parallel_rank,
        pipeline_model_parallel_rank=0,
        tensor_model_parallel_rank=parallel.tensor_model_parallel_rank,
        expert_model_parallel_rank=parallel.expert_model_parallel_rank,
        context_parallel_rank=parallel.context_parallel_rank,
    )
    last_stage_rank = parallel.rank_from_coords(
        data_parallel_rank=parallel.data_parallel_rank,
        pipeline_model_parallel_rank=parallel.pipeline_model_parallel_size - 1,
        tensor_model_parallel_rank=parallel.tensor_model_parallel_rank,
        expert_model_parallel_rank=parallel.expert_model_parallel_rank,
        context_parallel_rank=parallel.context_parallel_rank,
    )
    return PipelinePeers(
        prev_rank=prev_rank,
        next_rank=next_rank,
        first_stage_rank=first_stage_rank,
        last_stage_rank=last_stage_rank,
    )


def prepare_microbatches(
    batch: Optional[dict[str, torch.Tensor]],
    model: DeepSeekModel,
    expected_local_batch: int,
    num_microbatches: int,
    device: torch.device,
) -> list[Optional[torch.Tensor]]:
    """Prepare local microbatch list for first stage; placeholders elsewhere."""
    if not model.is_first_pp_stage:
        return [None for _ in range(num_microbatches)]

    if batch is None:
        raise ValueError("batch is required on first PP stage")

    local_input_ids = batch["input_ids"].to(device)
    if local_input_ids.size(0) != expected_local_batch:
        raise ValueError(
            "Observed local batch size differs from expected fixed pipeline shape. "
            "Adjust num_samples/batch_size to keep full batches."
        )
    return list(torch.chunk(local_input_ids, num_microbatches, dim=0))


def pipeline_forward_microbatch(
    microbatch_idx: int,
    model: DeepSeekModel,
    microbatches: list[Optional[torch.Tensor]],
    peers: PipelinePeers,
    state: PipelineStepState,
    microbatch_batch_size: int,
    seq_len: int,
    hidden_size: int,
    activation_dtype: torch.dtype,
    device: torch.device,
    gather_moe_metrics_fn: Callable[[nn.Module, torch.device], tuple[torch.Tensor, float]],
    precision_controller: Optional["MixedPrecisionController"] = None,
) -> None:
    """Run one forward microbatch and stage outputs/metadata for later backward."""
    label_tensor: Optional[torch.Tensor] = None
    stage_input: Optional[torch.Tensor] = None

    if model.is_first_pp_stage:
        local_input_ids = microbatches[microbatch_idx]
        if local_input_ids is None:
            raise RuntimeError("Missing first-stage microbatch")

        if not model.is_last_pp_stage:
            state.label_send_reqs.append(
                dist.isend(
                    tensor=local_input_ids.contiguous(),
                    dst=peers.last_stage_rank,
                    tag=label_tag(microbatch_idx),
                )
            )

        if precision_controller is None:
            stage_output = model.forward_stage(
                input_ids=local_input_ids,
                hidden_states=None,
                attention_mask=None,
            )
        else:
            with precision_controller.autocast_context():
                stage_output = model.forward_stage(
                    input_ids=local_input_ids,
                    hidden_states=None,
                    attention_mask=None,
                )
        if model.is_last_pp_stage:
            label_tensor = local_input_ids
    else:
        stage_input = torch.empty(
            (microbatch_batch_size, seq_len, hidden_size),
            device=device,
            dtype=activation_dtype,
            requires_grad=True,
        )
        if peers.prev_rank is None:
            raise RuntimeError("Missing previous PP rank")
        dist.recv(stage_input, src=peers.prev_rank, tag=activation_tag(microbatch_idx))

        if precision_controller is None:
            stage_output = model.forward_stage(
                input_ids=None,
                hidden_states=stage_input,
                attention_mask=None,
            )
        else:
            with precision_controller.autocast_context():
                stage_output = model.forward_stage(
                    input_ids=None,
                    hidden_states=stage_input,
                    attention_mask=None,
                )

        if model.is_last_pp_stage:
            label_tensor = torch.empty(
                (microbatch_batch_size, seq_len),
                dtype=torch.long,
                device=device,
            )
            dist.recv(label_tensor, src=peers.first_stage_rank, tag=label_tag(microbatch_idx))

    local_aux_loss, local_drop = gather_moe_metrics_fn(model, device=device)
    state.aux_loss_sum += float(local_aux_loss.detach().item())
    state.drop_sum += float(local_drop)
    state.drop_count += 1

    if not model.is_last_pp_stage:
        if peers.next_rank is None:
            raise RuntimeError("Missing next PP rank")
        activation_payload = stage_output.contiguous()
        state.activation_send_buffers.append(activation_payload)
        state.activation_send_reqs.append(
            dist.isend(
                tensor=activation_payload,
                dst=peers.next_rank,
                tag=activation_tag(microbatch_idx),
            )
        )

    state.stage_inputs.append(stage_input)
    state.stage_outputs.append(stage_output)
    state.stage_aux_losses.append(local_aux_loss)
    state.stage_labels.append(label_tensor)


def pipeline_backward_microbatch(
    microbatch_idx: int,
    model: DeepSeekModel,
    peers: PipelinePeers,
    state: PipelineStepState,
    num_microbatches: int,
    aux_loss_coef: float,
    microbatch_batch_size: int,
    seq_len: int,
    hidden_size: int,
    activation_dtype: torch.dtype,
    device: torch.device,
    precision_controller: Optional["MixedPrecisionController"] = None,
) -> None:
    """Run one backward microbatch and send dX to previous stage when needed."""
    stage_input = state.stage_inputs.pop(0)
    stage_output = state.stage_outputs.pop(0)
    aux_loss = state.stage_aux_losses.pop(0)
    labels = state.stage_labels.pop(0)

    if model.is_last_pp_stage:
        if labels is None:
            raise RuntimeError("Missing labels on last PP stage")

        if labels.size(0) == 0 or labels.size(1) < 2:
            task_loss = stage_output.sum() * 0.0
        else:
            shift_logits = stage_output[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            task_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )

        state.task_loss_sum += float(task_loss.detach().item())
        scale = 1.0 / float(num_microbatches)
        total = (task_loss * scale) + (aux_loss_coef * scale * aux_loss)
        if precision_controller is None:
            total.backward()
        else:
            precision_controller.backward(total)

        if not model.is_first_pp_stage:
            if stage_input is None or stage_input.grad is None:
                raise RuntimeError("Expected stage input gradient on last PP stage")
            if peers.prev_rank is None:
                raise RuntimeError("Missing previous PP rank")
            grad_payload = stage_input.grad.contiguous()
            state.grad_send_buffers.append(grad_payload)
            state.grad_send_reqs.append(
                dist.isend(
                    tensor=grad_payload,
                    dst=peers.prev_rank,
                    tag=grad_tag(microbatch_idx),
                )
            )
        return

    grad_output = torch.empty(
        (microbatch_batch_size, seq_len, hidden_size),
        device=device,
        dtype=activation_dtype,
    )
    if peers.next_rank is None:
        raise RuntimeError("Missing next PP rank")
    dist.recv(grad_output, src=peers.next_rank, tag=grad_tag(microbatch_idx))

    if aux_loss.requires_grad:
        aux_loss_scale = 1.0
        if precision_controller is not None and precision_controller.uses_loss_scaling:
            aux_loss_scale = float(precision_controller.runtime_state.loss_scale)
        aux_scale = torch.tensor(
            (aux_loss_coef / float(num_microbatches)) * aux_loss_scale,
            dtype=aux_loss.dtype,
            device=device,
        )
        torch.autograd.backward([stage_output, aux_loss], [grad_output, aux_scale])
    else:
        torch.autograd.backward(stage_output, grad_output)

    if not model.is_first_pp_stage:
        if stage_input is None or stage_input.grad is None:
            raise RuntimeError("Expected stage input gradient on middle PP stage")
        if peers.prev_rank is None:
            raise RuntimeError("Missing previous PP rank")
        grad_payload = stage_input.grad.contiguous()
        state.grad_send_buffers.append(grad_payload)
        state.grad_send_reqs.append(
            dist.isend(
                tensor=grad_payload,
                dst=peers.prev_rank,
                tag=grad_tag(microbatch_idx),
            )
        )


def execute_1f1b_schedule(
    parallel: ModelParallelTopology,
    num_microbatches: int,
    run_forward: Callable[[int], None],
    run_backward: Callable[[int], None],
) -> None:
    """Execute warmup/steady/cooldown phases for non-interleaved 1F1B."""
    num_warmup = min(
        parallel.pipeline_model_parallel_size - parallel.pipeline_model_parallel_rank - 1,
        num_microbatches,
    )
    num_remaining = num_microbatches - num_warmup

    for microbatch_idx in range(num_warmup):
        run_forward(microbatch_idx)

    for steady_idx in range(num_remaining):
        forward_idx = steady_idx + num_warmup
        run_forward(forward_idx)
        run_backward(steady_idx)

    for cooldown_idx in range(num_warmup):
        backward_idx = num_remaining + cooldown_idx
        run_backward(backward_idx)


def finalize_pipeline_sends(state: PipelineStepState) -> None:
    """Wait on async send requests and clear send payload references."""
    for req in state.activation_send_reqs:
        req.wait()
    for req in state.grad_send_reqs:
        req.wait()
    for req in state.label_send_reqs:
        req.wait()

    state.activation_send_buffers.clear()
    state.grad_send_buffers.clear()


def train_step_pipeline(
    model: DeepSeekModel,
    optimizer,
    use_distributed_optimizer: bool,
    batch: Optional[dict[str, torch.Tensor]],
    parallel: ModelParallelTopology,
    num_microbatches: int,
    expected_local_batch: int,
    seq_len: int,
    aux_loss_coef: float,
    shard_info: ParamShardInfo,
    gather_moe_metrics_fn: Callable[[nn.Module, torch.device], tuple[torch.Tensor, float]],
    apply_optimizer_step_fn: Callable[..., None],
    sync_plugin,
    zero_grad_fn: Optional[Callable[[object], None]] = None,
    refresh_persistent_params_fn: Optional[Callable[[nn.Module], None]] = None,
    precision_controller: Optional["MixedPrecisionController"] = None,
    precision_config: Optional[PrecisionConfig] = None,
) -> tuple[float, float, float, int, int]:
    """One TP+PP+EP+DP training step using non-interleaved 1F1B schedule."""
    if parallel.pipeline_model_parallel_size <= 1:
        raise ValueError("train_step_pipeline requires pipeline_model_parallel_size > 1")

    device = parallel.device
    if precision_config is None:
        activation_dtype = next(model.parameters()).dtype
    else:
        activation_dtype = dtype_alias_to_torch(precision_config.activation_dtype)
    hidden_size = model.config.hidden_size
    microbatch_batch_size = expected_local_batch // num_microbatches
    peers = resolve_pipeline_peers(parallel)
    state = PipelineStepState()
    if zero_grad_fn is None:
        optimizer.zero_grad(set_to_none=True)
    else:
        zero_grad_fn(optimizer)

    microbatches = prepare_microbatches(
        batch=batch,
        model=model,
        expected_local_batch=expected_local_batch,
        num_microbatches=num_microbatches,
        device=device,
    )

    execute_1f1b_schedule(
        parallel=parallel,
        num_microbatches=num_microbatches,
        run_forward=lambda microbatch_idx: pipeline_forward_microbatch(
            microbatch_idx=microbatch_idx,
            model=model,
            microbatches=microbatches,
            peers=peers,
            state=state,
            microbatch_batch_size=microbatch_batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            activation_dtype=activation_dtype,
            device=device,
            gather_moe_metrics_fn=gather_moe_metrics_fn,
            precision_controller=precision_controller,
        ),
        run_backward=lambda microbatch_idx: pipeline_backward_microbatch(
            microbatch_idx=microbatch_idx,
            model=model,
            peers=peers,
            state=state,
            num_microbatches=num_microbatches,
            aux_loss_coef=aux_loss_coef,
            microbatch_batch_size=microbatch_batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            activation_dtype=activation_dtype,
            device=device,
            precision_controller=precision_controller,
        ),
    )
    finalize_pipeline_sends(state)

    should_step = True
    if precision_controller is not None:
        should_step = precision_controller.prepare_optimizer_step(model)

    if should_step:
        apply_optimizer_step_fn(
            model=model,
            optimizer=optimizer,
            use_distributed_optimizer=use_distributed_optimizer,
            shard_info=shard_info,
            data_parallel_size=parallel.data_parallel_size,
            expert_data_parallel_size=parallel.expert_data_parallel_size,
            data_parallel_group=parallel.data_parallel_group,
            expert_data_parallel_group=parallel.expert_data_parallel_group,
            sync_plugin=sync_plugin,
        )
        if refresh_persistent_params_fn is not None:
            refresh_persistent_params_fn(model)

    if precision_controller is not None:
        precision_controller.update_after_step(step_applied=should_step)

    objective_count = num_microbatches if model.is_last_pp_stage else 0
    return (
        state.task_loss_sum,
        state.aux_loss_sum,
        state.drop_sum,
        objective_count,
        state.drop_count,
    )
