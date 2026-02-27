"""Pipeline mixed-precision helper tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from src.runtime.contracts import PrecisionConfig
from src.runtime.mixed_precision import MixedPrecisionController
from src.runtime.pipeline import PipelinePeers
from src.runtime.pipeline import PipelineStepState
from src.runtime.pipeline import pipeline_backward_microbatch
from src.runtime.pipeline import train_step_pipeline
from src.runtime.sync import ParamShardInfo


class _TinyPipelineModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0]))
        self.is_last_pp_stage = False
        self.is_first_pp_stage = False
        self.config = SimpleNamespace(hidden_size=8)


class _LastStageModel:
    is_last_pp_stage = True
    is_first_pp_stage = True


def test_train_step_pipeline_uses_precision_activation_dtype(monkeypatch) -> None:
    model = _TinyPipelineModel()
    parallel = SimpleNamespace(
        pipeline_model_parallel_size=2,
        pipeline_model_parallel_rank=0,
        data_parallel_rank=0,
        tensor_model_parallel_rank=0,
        expert_model_parallel_rank=0,
        context_parallel_rank=0,
        data_parallel_size=1,
        expert_data_parallel_size=1,
        data_parallel_group=None,
        expert_data_parallel_group=None,
        device=torch.device("cpu"),
        rank_from_coords=lambda **kwargs: 0,
    )

    seen: dict[str, object] = {"activation_dtype": None}

    def _fake_prepare_microbatches(**kwargs):
        del kwargs
        return [None]

    def _fake_execute_1f1b_schedule(*, parallel, num_microbatches, run_forward, run_backward):
        del parallel, num_microbatches
        run_forward(0)
        run_backward(0)

    def _fake_forward_microbatch(**kwargs):
        seen["activation_dtype"] = kwargs["activation_dtype"]

    def _fake_backward_microbatch(**kwargs):
        del kwargs

    def _fake_finalize_sends(state):
        del state

    called = {"step": 0}

    def _apply_optimizer_step_fn(**kwargs):
        del kwargs
        called["step"] += 1

    monkeypatch.setattr("src.runtime.pipeline.prepare_microbatches", _fake_prepare_microbatches)
    monkeypatch.setattr("src.runtime.pipeline.execute_1f1b_schedule", _fake_execute_1f1b_schedule)
    monkeypatch.setattr("src.runtime.pipeline.pipeline_forward_microbatch", _fake_forward_microbatch)
    monkeypatch.setattr("src.runtime.pipeline.pipeline_backward_microbatch", _fake_backward_microbatch)
    monkeypatch.setattr("src.runtime.pipeline.finalize_pipeline_sends", _fake_finalize_sends)

    precision_config = PrecisionConfig(mode="fp8", activation_dtype="bf16", fp8_backend="emulated")

    train_step_pipeline(
        model=model,
        optimizer=object(),
        use_distributed_optimizer=False,
        batch=None,
        parallel=parallel,
        num_microbatches=1,
        expected_local_batch=2,
        seq_len=4,
        aux_loss_coef=0.01,
        shard_info=ParamShardInfo(set(), set()),
        gather_moe_metrics_fn=lambda m, d: (torch.zeros(()), 0.0),
        apply_optimizer_step_fn=_apply_optimizer_step_fn,
        sync_plugin=None,
        zero_grad_fn=lambda _: None,
        precision_controller=None,
        precision_config=precision_config,
    )

    assert seen["activation_dtype"] == torch.bfloat16
    assert called["step"] == 1


def test_pipeline_backward_scales_last_stage_loss_with_controller() -> None:
    labels = torch.tensor([[1, 2, 3]], dtype=torch.long)
    no_scale_output = torch.randn(1, 3, 5, requires_grad=True)

    state_no_scale = PipelineStepState(
        stage_inputs=[None],
        stage_outputs=[no_scale_output],
        stage_aux_losses=[torch.zeros((), requires_grad=True)],
        stage_labels=[labels],
    )

    peers = PipelinePeers(prev_rank=None, next_rank=None, first_stage_rank=0, last_stage_rank=0)

    pipeline_backward_microbatch(
        microbatch_idx=0,
        model=_LastStageModel(),
        peers=peers,
        state=state_no_scale,
        num_microbatches=1,
        aux_loss_coef=0.01,
        microbatch_batch_size=1,
        seq_len=3,
        hidden_size=5,
        activation_dtype=torch.float32,
        device=torch.device("cpu"),
        precision_controller=None,
    )
    assert no_scale_output.grad is not None
    grad_norm_no_scale = float(no_scale_output.grad.norm().item())

    scaled_output = no_scale_output.clone().detach().requires_grad_(True)
    state_scaled = PipelineStepState(
        stage_inputs=[None],
        stage_outputs=[scaled_output],
        stage_aux_losses=[torch.zeros((), requires_grad=True)],
        stage_labels=[labels],
    )

    cfg = PrecisionConfig(
        mode="fp4",
        activation_dtype="fp32",
        fp4_backend="emulated",
        loss_scale_init=8.0,
        loss_scale_growth_factor=2.0,
        loss_scale_backoff_factor=0.5,
        loss_scale_growth_interval=100,
        loss_scale_min=1.0,
        loss_scale_max=1024.0,
    )
    controller = MixedPrecisionController(cfg, device=torch.device("cpu"))

    pipeline_backward_microbatch(
        microbatch_idx=0,
        model=_LastStageModel(),
        peers=peers,
        state=state_scaled,
        num_microbatches=1,
        aux_loss_coef=0.01,
        microbatch_batch_size=1,
        seq_len=3,
        hidden_size=5,
        activation_dtype=torch.float32,
        device=torch.device("cpu"),
        precision_controller=controller,
    )

    assert scaled_output.grad is not None
    grad_norm_scaled = float(scaled_output.grad.norm().item())
    assert grad_norm_scaled > grad_norm_no_scale * 4.0
