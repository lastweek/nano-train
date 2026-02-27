"""Runtime mixed-precision policy, scaling, and backend orchestration."""

from __future__ import annotations

from contextlib import contextmanager
from contextlib import nullcontext
from dataclasses import replace
import fnmatch
import re
from typing import Iterable
from typing import Iterator
from typing import Optional

import torch
import torch.distributed as dist

from src.runtime.contracts import FP4PersistentFormat
from src.runtime.contracts import LowBitComputeMode
from src.runtime.contracts import ModelPrecisionPlan
from src.runtime.contracts import ModulePatternType
from src.runtime.contracts import ModulePrecisionAssignment
from src.runtime.contracts import ModulePrecisionPolicy
from src.runtime.contracts import PersistentLowBitMode
from src.runtime.contracts import PersistentScaleGranularity
from src.runtime.contracts import PrecisionConfig
from src.runtime.contracts import PrecisionDType
from src.runtime.contracts import PrecisionMode
from src.runtime.contracts import PrecisionRuntimeState
from src.runtime.te_backend import ActiveLowBitContext
from src.runtime.te_backend import LowBitBackend
from src.runtime.te_backend import build_lowbit_backend_for_mode
from src.runtime.te_backend import clear_active_lowbit_context
from src.runtime.te_backend import set_active_lowbit_context


_DTYPE_ALIAS_TO_TORCH: dict[PrecisionDType, torch.dtype] = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def dtype_alias_to_torch(dtype_alias: PrecisionDType) -> torch.dtype:
    """Map precision dtype alias to torch dtype."""
    try:
        return _DTYPE_ALIAS_TO_TORCH[dtype_alias]
    except KeyError as exc:
        raise ValueError(f"Unsupported precision dtype alias: {dtype_alias}") from exc


def _default_mode_for_device(device: torch.device) -> PrecisionMode:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp32"


def _resolve_mode_from_flags(args) -> Optional[PrecisionMode]:
    mode_flags = {
        "bf16": bool(getattr(args, "bf16", False)),
        "fp16": bool(getattr(args, "fp16", False)),
        "fp8": bool(getattr(args, "fp8", False)),
        "fp4": bool(getattr(args, "fp4", False)),
    }
    selected = [name for name, enabled in mode_flags.items() if enabled]
    if len(selected) > 1:
        raise ValueError("At most one of --bf16/--fp16/--fp8/--fp4 may be set")
    if not selected:
        return None
    return selected[0]  # type: ignore[return-value]


def _default_activation_dtype(mode: PrecisionMode, device: torch.device) -> PrecisionDType:
    if mode == "bf16":
        return "bf16"
    if mode == "fp16":
        return "fp16"
    if mode in ("fp8", "fp4"):
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return "bf16"
        if device.type == "cuda":
            return "fp16"
        return "fp32"
    return "fp32"


def _default_params_dtype(mode: PrecisionMode, device: torch.device) -> PrecisionDType:
    if mode == "bf16":
        return "bf16"
    if mode == "fp16":
        return "fp16"
    if mode in ("fp8", "fp4"):
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp16" if device.type == "cuda" else "fp32"
    return "fp32"


def _iter_patterns(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()

    if isinstance(value, Iterable):
        patterns = []
        for item in value:
            text = str(item).strip()
            if text:
                patterns.append(text)
        return tuple(patterns)

    text = str(value).strip()
    return (text,) if text else ()


def _normalize_pattern_type(raw: object) -> ModulePatternType:
    value = str(raw or "regex")
    if value not in ("regex", "glob"):
        raise ValueError("--module-pattern-type must be regex or glob")
    return value  # type: ignore[return-value]


def _normalize_compute_mode(raw: object) -> Optional[LowBitComputeMode]:
    if raw is None:
        return None
    value = str(raw)
    if value in ("", "none", "off"):
        return None
    if value not in ("fp8", "fp4"):
        raise ValueError("--compute-lowbit-mode must be fp8 or fp4")
    return value  # type: ignore[return-value]


def _normalize_persistent_mode(raw: object) -> PersistentLowBitMode:
    if raw is None:
        return "off"
    value = str(raw)
    if value in ("", "none"):
        return "off"
    if value not in ("off", "fp8", "fp4"):
        raise ValueError("--persistent-lowbit-mode must be off, fp8, or fp4")
    return value  # type: ignore[return-value]


def _normalize_scale_granularity(raw: object) -> PersistentScaleGranularity:
    value = str(raw or "per_channel")
    if value not in ("per_tensor", "per_channel"):
        raise ValueError("--persistent-scale-granularity must be per_tensor or per_channel")
    return value  # type: ignore[return-value]


def _normalize_fp4_format(raw: object) -> FP4PersistentFormat:
    value = str(raw or "nf4")
    if value != "nf4":
        raise ValueError("--fp4-param-format currently supports only nf4")
    return "nf4"


def _collect_required_lowbit_modes(config: PrecisionConfig) -> set[LowBitComputeMode]:
    modes: set[LowBitComputeMode] = set()
    if config.mode in ("fp8", "fp4"):
        modes.add(config.mode)

    policy = config.module_precision_policy
    if policy is not None and policy.compute_lowbit_mode is not None:
        modes.add(policy.compute_lowbit_mode)

    return modes


def resolve_module_precision_policy(
    args,
    precision_config: PrecisionConfig,
) -> ModulePrecisionPolicy:
    """Resolve per-module low-bit precision policy from CLI flags."""
    pattern_type = _normalize_pattern_type(getattr(args, "module_pattern_type", "regex"))

    explicit_compute_mode = _normalize_compute_mode(getattr(args, "compute_lowbit_mode", None))
    if explicit_compute_mode is None and precision_config.mode in ("fp8", "fp4"):
        compute_mode: Optional[LowBitComputeMode] = precision_config.mode
    else:
        compute_mode = explicit_compute_mode

    fp8_param = bool(getattr(args, "fp8_param", False))
    fp4_param = bool(getattr(args, "fp4_param", False))
    if fp8_param and fp4_param:
        raise ValueError("At most one of --fp8-param/--fp4-param may be set")

    persistent_mode_flag = _normalize_persistent_mode(
        getattr(args, "persistent_lowbit_mode", None)
    )
    if persistent_mode_flag == "off":
        if fp8_param:
            persistent_mode: PersistentLowBitMode = "fp8"
        elif fp4_param:
            persistent_mode = "fp4"
        else:
            persistent_mode = "off"
    else:
        if fp8_param and persistent_mode_flag != "fp8":
            raise ValueError("--fp8-param conflicts with --persistent-lowbit-mode")
        if fp4_param and persistent_mode_flag != "fp4":
            raise ValueError("--fp4-param conflicts with --persistent-lowbit-mode")
        persistent_mode = persistent_mode_flag

    policy = ModulePrecisionPolicy(
        pattern_type=pattern_type,
        compute_lowbit_mode=compute_mode,
        compute_lowbit_include=_iter_patterns(getattr(args, "compute_lowbit_include", None)),
        compute_lowbit_exclude=_iter_patterns(getattr(args, "compute_lowbit_exclude", None)),
        persistent_lowbit_mode=persistent_mode,
        persistent_lowbit_include=_iter_patterns(getattr(args, "persistent_lowbit_include", None)),
        persistent_lowbit_exclude=_iter_patterns(getattr(args, "persistent_lowbit_exclude", None)),
        persistent_scale_granularity=_normalize_scale_granularity(
            getattr(args, "persistent_scale_granularity", "per_channel")
        ),
        fp4_persistent_format=_normalize_fp4_format(
            getattr(args, "fp4_param_format", "nf4")
        ),
    )

    return policy


def _match_pattern(
    name: str,
    pattern: str,
    *,
    pattern_type: ModulePatternType,
) -> bool:
    if pattern_type == "regex":
        return re.search(pattern, name) is not None
    return fnmatch.fnmatch(name, pattern)


def _resolve_matches(
    module_names: Iterable[str],
    *,
    default_all_when_no_include: bool,
    include_patterns: tuple[str, ...],
    exclude_patterns: tuple[str, ...],
    pattern_type: ModulePatternType,
    label: str,
) -> set[str]:
    names = list(module_names)
    if include_patterns:
        matches = {
            name
            for name in names
            if any(
                _match_pattern(name, pattern, pattern_type=pattern_type)
                for pattern in include_patterns
            )
        }
        if not matches:
            raise ValueError(f"{label} include patterns matched zero modules")
    elif default_all_when_no_include:
        matches = set(names)
    else:
        matches = set()

    excluded = {
        name
        for name in matches
        if any(_match_pattern(name, pattern, pattern_type=pattern_type) for pattern in exclude_patterns)
    }
    return matches - excluded


def _is_lowbit_capable(module: torch.nn.Module) -> bool:
    return callable(getattr(module, "set_precision_assignment", None)) and callable(
        getattr(module, "refresh_persistent_lowbit_params", None)
    )


def build_model_precision_plan(
    model: torch.nn.Module,
    policy: ModulePrecisionPolicy,
) -> ModelPrecisionPlan:
    """Resolve concrete per-module assignments for one model instance.

    Low-bit compute is assignment-driven: layers do not infer compute mode from a global
    runtime default.
    """
    named_modules = dict(model.named_modules())
    module_names = tuple(name for name in named_modules if name)
    lowbit_capable_names = {
        name for name in module_names if _is_lowbit_capable(named_modules[name])
    }

    if policy.compute_lowbit_mode is not None and not policy.compute_lowbit_include:
        compute_matches = set(lowbit_capable_names)
        compute_matches = {
            name
            for name in compute_matches
            if not any(
                _match_pattern(name, pattern, pattern_type=policy.pattern_type)
                for pattern in policy.compute_lowbit_exclude
            )
        }
    else:
        compute_matches = _resolve_matches(
            module_names,
            default_all_when_no_include=False,
            include_patterns=policy.compute_lowbit_include,
            exclude_patterns=policy.compute_lowbit_exclude,
            pattern_type=policy.pattern_type,
            label="compute_lowbit",
        )

    if policy.persistent_lowbit_mode != "off" and not policy.persistent_lowbit_include:
        persistent_matches = set(lowbit_capable_names)
        persistent_matches = {
            name
            for name in persistent_matches
            if not any(
                _match_pattern(name, pattern, pattern_type=policy.pattern_type)
                for pattern in policy.persistent_lowbit_exclude
            )
        }
    else:
        persistent_matches = _resolve_matches(
            module_names,
            default_all_when_no_include=False,
            include_patterns=policy.persistent_lowbit_include,
            exclude_patterns=policy.persistent_lowbit_exclude,
            pattern_type=policy.pattern_type,
            label="persistent_lowbit",
        )

    unsupported_compute = sorted(
        name
        for name in compute_matches
        if not _is_lowbit_capable(named_modules[name])
    )
    if unsupported_compute and policy.compute_lowbit_mode is not None:
        sample = ", ".join(unsupported_compute[:5])
        raise ValueError(
            "compute_lowbit patterns matched modules without low-bit support: "
            f"{sample}"
        )

    unsupported_persistent = sorted(
        name
        for name in persistent_matches
        if not _is_lowbit_capable(named_modules[name])
    )
    if unsupported_persistent and policy.persistent_lowbit_mode != "off":
        sample = ", ".join(unsupported_persistent[:5])
        raise ValueError(
            "persistent_lowbit patterns matched modules without low-bit support: "
            f"{sample}"
        )

    assignments: dict[str, ModulePrecisionAssignment] = {}
    compute_count = 0
    persistent_count = 0

    for name, module in named_modules.items():
        if not name or not _is_lowbit_capable(module):
            continue

        compute_mode = policy.compute_lowbit_mode if name in compute_matches else None
        persistent_mode = policy.persistent_lowbit_mode if name in persistent_matches else "off"

        if compute_mode is not None:
            compute_count += 1
        if persistent_mode != "off":
            persistent_count += 1

        assignments[name] = ModulePrecisionAssignment(
            module_name=name,
            module_type=module.__class__.__name__,
            compute_lowbit_mode=compute_mode,
            persistent_lowbit_mode=persistent_mode,
            persistent_scale_granularity=policy.persistent_scale_granularity,
            fp4_persistent_format=policy.fp4_persistent_format,
        )

    return ModelPrecisionPlan(
        assignments=assignments,
        compute_lowbit_module_count=compute_count,
        persistent_lowbit_module_count=persistent_count,
    )


def apply_model_precision_plan(
    model: torch.nn.Module,
    plan: ModelPrecisionPlan,
) -> None:
    """Apply a model precision plan to module-local precision-capable layers.

    Low-bit execution requires explicit assignments on individual modules. This function installs
    those assignments before training.
    """
    for module_name, module in model.named_modules():
        assignment = plan.assignments.get(module_name)
        if assignment is None:
            continue

        setter = getattr(module, "set_precision_assignment", None)
        if not callable(setter):
            raise ValueError(
                "Model precision plan includes unsupported module "
                f"'{module_name}' ({module.__class__.__name__})"
            )
        setter(assignment)

    refresh_persistent_lowbit_params(model)


def refresh_persistent_lowbit_params(model: torch.nn.Module) -> int:
    """Refresh persistent low-bit params for modules that expose the hook."""
    refreshed = 0
    for module in model.modules():
        refresh_fn = getattr(module, "refresh_persistent_lowbit_params", None)
        if callable(refresh_fn):
            refresh_fn()
            refreshed += 1
    return refreshed


def ensure_lowbit_compute_assignments(
    config: PrecisionConfig,
    plan: ModelPrecisionPlan,
    *,
    script_name: str,
) -> None:
    """Fail fast when low-bit mode is active but no modules were assigned for compute."""
    if config.mode not in ("fp8", "fp4"):
        return
    if plan.compute_lowbit_module_count > 0:
        return
    raise RuntimeError(
        f"{script_name}: low-bit mode '{config.mode}' is active but zero modules were assigned "
        "for low-bit compute. Check --compute-lowbit-include/--compute-lowbit-exclude or verify "
        "the model uses low-bit-capable layers, then rerun build_model_precision_plan(...) and "
        "apply_model_precision_plan(...)."
    )


def resolve_precision_config(args, device: torch.device) -> PrecisionConfig:
    """Resolve run precision config from Megatron-style precision flags."""
    explicit_mode = _resolve_mode_from_flags(args)
    mode = explicit_mode or _default_mode_for_device(device)

    if mode == "bf16":
        if device.type != "cuda":
            raise ValueError("--bf16 requires CUDA")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("--bf16 requires torch.cuda.is_bf16_supported()")
    if mode == "fp16" and device.type != "cuda":
        raise ValueError("--fp16 requires CUDA")

    fp8_backend = str(getattr(args, "fp8_backend", "transformer_engine"))
    fp8_format = str(getattr(args, "fp8_format", "e4m3"))
    fp8_amax_history_len = int(getattr(args, "fp8_amax_history_len", 16))
    fp8_amax_compute_algo = str(getattr(args, "fp8_amax_compute_algo", "most_recent"))
    fp4_backend = str(getattr(args, "fp4_backend", "emulated"))

    if mode == "fp8" and fp8_backend not in ("transformer_engine", "emulated"):
        raise ValueError("--fp8-backend must be transformer_engine or emulated")
    if mode == "fp8" and fp8_format not in ("e4m3", "hybrid"):
        raise ValueError("--fp8-format must be e4m3 or hybrid")
    if mode == "fp8" and fp8_amax_compute_algo not in ("most_recent", "max"):
        raise ValueError("--fp8-amax-compute-algo must be most_recent or max")
    if mode == "fp8" and fp8_amax_history_len < 1:
        raise ValueError("--fp8-amax-history-len must be >= 1")

    if mode == "fp4" and fp4_backend != "emulated":
        raise ValueError("--fp4-backend currently supports only emulated")

    params_dtype_raw = getattr(args, "params_dtype", None)
    main_params_dtype_raw = getattr(args, "main_params_dtype", None)
    main_grads_dtype_raw = getattr(args, "main_grads_dtype", None)
    exp_avg_dtype_raw = getattr(args, "exp_avg_dtype", None)
    exp_avg_sq_dtype_raw = getattr(args, "exp_avg_sq_dtype", None)

    params_dtype = (
        _default_params_dtype(mode, device) if params_dtype_raw is None else str(params_dtype_raw)
    )
    main_params_dtype = "fp32" if main_params_dtype_raw is None else str(main_params_dtype_raw)
    main_grads_dtype = "fp32" if main_grads_dtype_raw is None else str(main_grads_dtype_raw)
    exp_avg_dtype = "fp32" if exp_avg_dtype_raw is None else str(exp_avg_dtype_raw)
    exp_avg_sq_dtype = "fp32" if exp_avg_sq_dtype_raw is None else str(exp_avg_sq_dtype_raw)

    valid_dtype_values = {"fp32", "bf16", "fp16"}
    for key, value in {
        "params_dtype": params_dtype,
        "main_params_dtype": main_params_dtype,
        "main_grads_dtype": main_grads_dtype,
        "exp_avg_dtype": exp_avg_dtype,
        "exp_avg_sq_dtype": exp_avg_sq_dtype,
    }.items():
        if value not in valid_dtype_values:
            raise ValueError(f"--{key.replace('_', '-')} must be fp32, bf16, or fp16")

    activation_dtype = _default_activation_dtype(mode, device)

    loss_scale_growth_interval = int(getattr(args, "loss_scale_growth_interval", 2000))
    if loss_scale_growth_interval < 1:
        raise ValueError("--loss-scale-growth-interval must be >= 1")

    config = PrecisionConfig(
        mode=mode,
        params_dtype=params_dtype,  # type: ignore[arg-type]
        main_params_dtype=main_params_dtype,  # type: ignore[arg-type]
        main_grads_dtype=main_grads_dtype,  # type: ignore[arg-type]
        exp_avg_dtype=exp_avg_dtype,  # type: ignore[arg-type]
        exp_avg_sq_dtype=exp_avg_sq_dtype,  # type: ignore[arg-type]
        activation_dtype=activation_dtype,
        fp8_backend=fp8_backend,  # type: ignore[arg-type]
        fp8_format=fp8_format,  # type: ignore[arg-type]
        fp8_amax_history_len=fp8_amax_history_len,
        fp8_amax_compute_algo=fp8_amax_compute_algo,  # type: ignore[arg-type]
        fp4_backend=fp4_backend,  # type: ignore[arg-type]
        fp8_param=bool(getattr(args, "fp8_param", False)),
        fp4_param=bool(getattr(args, "fp4_param", False)),
        fp4_param_format=_normalize_fp4_format(getattr(args, "fp4_param_format", "nf4")),
        persistent_scale_granularity=_normalize_scale_granularity(
            getattr(args, "persistent_scale_granularity", "per_channel")
        ),
        loss_scale_init=float(getattr(args, "loss_scale_init", 65536.0)),
        loss_scale_growth_factor=float(getattr(args, "loss_scale_growth_factor", 2.0)),
        loss_scale_backoff_factor=float(getattr(args, "loss_scale_backoff_factor", 0.5)),
        loss_scale_growth_interval=loss_scale_growth_interval,
        loss_scale_min=float(getattr(args, "loss_scale_min", 1.0)),
        loss_scale_max=float(getattr(args, "loss_scale_max", 16777216.0)),
    )

    if config.loss_scale_init <= 0:
        raise ValueError("--loss-scale-init must be > 0")
    if config.loss_scale_growth_factor <= 1.0:
        raise ValueError("--loss-scale-growth-factor must be > 1")
    if not (0.0 < config.loss_scale_backoff_factor < 1.0):
        raise ValueError("--loss-scale-backoff-factor must be in (0, 1)")
    if config.loss_scale_min <= 0.0:
        raise ValueError("--loss-scale-min must be > 0")
    if config.loss_scale_max < config.loss_scale_min:
        raise ValueError("--loss-scale-max must be >= --loss-scale-min")

    policy = resolve_module_precision_policy(args, config)
    config.module_precision_policy = policy

    for required_mode in _collect_required_lowbit_modes(config):
        _ = build_lowbit_backend_for_mode(config, mode=required_mode)

    return config


class MixedPrecisionController:
    """Apply autocast, low-bit backend context, and dynamic loss scaling policies."""

    def __init__(
        self,
        config: PrecisionConfig,
        *,
        device: torch.device,
    ) -> None:
        self.config = replace(config)
        self.device = device
        self.lowbit_backends: dict[str, LowBitBackend] = {}
        for mode in sorted(_collect_required_lowbit_modes(config)):
            backend = build_lowbit_backend_for_mode(config, mode=mode)
            if backend is not None:
                self.lowbit_backends[mode] = backend
        self.runtime_state = PrecisionRuntimeState(loss_scale=float(config.loss_scale_init))

    @property
    def uses_loss_scaling(self) -> bool:
        return self.config.mode in ("fp16", "fp8", "fp4")

    def activation_torch_dtype(self) -> torch.dtype:
        """Return activation compute dtype as torch dtype."""
        return dtype_alias_to_torch(self.config.activation_dtype)

    @contextmanager
    def autocast_context(self) -> Iterator[None]:
        """Context manager for per-step autocast and low-bit backend activation.

        Layer-level module assignments still control whether low-bit kernels are used.
        """
        lowbit_context = None
        if self.lowbit_backends:
            lowbit_context = ActiveLowBitContext(
                backend_by_mode=dict(self.lowbit_backends),
                default_mode=self.config.mode,
                config=self.config,
            )
            set_active_lowbit_context(lowbit_context)

        try:
            with self._autocast_context_impl():
                yield
        finally:
            if lowbit_context is not None:
                clear_active_lowbit_context()

    def _autocast_context_impl(self):
        dtype = self.activation_torch_dtype()
        if dtype == torch.float32:
            return nullcontext()

        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=dtype)

        if self.device.type == "cpu" and dtype == torch.bfloat16:
            return torch.autocast(device_type="cpu", dtype=dtype)

        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        """Execute backward pass with optional dynamic loss scaling."""
        if self.uses_loss_scaling:
            scale = float(self.runtime_state.loss_scale)
            (loss * scale).backward()
            return
        loss.backward()

    def _unscale_grads_(self, model: torch.nn.Module) -> None:
        if not self.uses_loss_scaling:
            return

        scale = float(self.runtime_state.loss_scale)
        if scale == 1.0:
            return

        inv_scale = 1.0 / scale
        for param in model.parameters():
            grad = param.grad
            if grad is None:
                continue
            grad.mul_(inv_scale)

    def _local_grads_finite(self, model: torch.nn.Module) -> bool:
        for param in model.parameters():
            grad = param.grad
            if grad is None:
                continue
            if not torch.isfinite(grad).all():
                return False
        return True

    def prepare_optimizer_step(self, model: torch.nn.Module) -> bool:
        """Unscale grads and return whether optimizer step should be applied."""
        self._unscale_grads_(model)

        is_finite_local = 1 if self._local_grads_finite(model) else 0
        finite_tensor = torch.tensor(
            [is_finite_local],
            dtype=torch.int32,
            device=self.device,
        )
        if dist.is_initialized():
            dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)

        should_step = bool(finite_tensor.item() == 1)
        if not should_step:
            self.runtime_state.found_inf_steps += 1
            self.runtime_state.skipped_steps += 1
        return should_step

    def update_after_step(self, *, step_applied: bool) -> None:
        """Update dynamic loss scale state after each attempted optimizer step."""
        if not self.uses_loss_scaling:
            return

        if not step_applied:
            new_scale = max(
                float(self.config.loss_scale_min),
                float(self.runtime_state.loss_scale) * float(self.config.loss_scale_backoff_factor),
            )
            self.runtime_state.loss_scale = float(new_scale)
            self.runtime_state.growth_tracker = 0
            return

        self.runtime_state.growth_tracker += 1
        if self.runtime_state.growth_tracker < int(self.config.loss_scale_growth_interval):
            return

        self.runtime_state.growth_tracker = 0
        self.runtime_state.loss_scale = float(
            min(
                float(self.config.loss_scale_max),
                float(self.runtime_state.loss_scale) * float(self.config.loss_scale_growth_factor),
            )
        )
