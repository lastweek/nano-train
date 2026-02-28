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

from src.runtime.contracts import DeepSeekV3PrecisionRecipe
from src.runtime.contracts import FP4PersistentFormat
from src.runtime.contracts import LowBitKernelSpec
from src.runtime.contracts import LowBitCapableModuleType
from src.runtime.contracts import LowBitComputeMode
from src.runtime.contracts import MasterOwnershipMode
from src.runtime.contracts import ModulePatternType
from src.runtime.contracts import ModulePrecisionAssignment
from src.runtime.contracts import ModulePrecisionInitState
from src.runtime.contracts import ModulePrecisionPolicy
from src.runtime.contracts import ModulePrecisionResolver
from src.runtime.contracts import ModulePrecisionSummary
from src.runtime.contracts import PersistentLowBitMode
from src.runtime.contracts import PersistentScaleGranularity
from src.runtime.contracts import PrecisionConfig
from src.runtime.contracts import PrecisionDType
from src.runtime.contracts import PrecisionMode
from src.runtime.contracts import PrecisionRecipeName
from src.runtime.contracts import PrecisionRuntimeState
from src.runtime.contracts import QuantGranularity
from src.runtime.contracts import RoundingMode
from src.runtime.te_backend import LowBitBackend
from src.runtime.te_backend import build_lowbit_backend_for_mode


_DTYPE_ALIAS_TO_TORCH: dict[PrecisionDType, torch.dtype] = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


_LOWBIT_CAPABLE_TYPES: set[str] = {
    "linear",
    "column_parallel_linear",
    "row_parallel_linear",
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


def _parse_module_compute_dtype_rules(raw_rules: object) -> tuple[tuple[str, PrecisionDType], ...]:
    rules = _iter_patterns(raw_rules)
    parsed: list[tuple[str, PrecisionDType]] = []
    valid_dtypes = {"fp32", "bf16", "fp16"}
    for rule in rules:
        if "=" not in rule:
            raise ValueError(
                "--module-compute-dtype-rule must use '<pattern>=<fp32|bf16|fp16>' format"
            )
        pattern, dtype_raw = rule.split("=", 1)
        pattern = pattern.strip()
        dtype_raw = dtype_raw.strip()
        if not pattern:
            raise ValueError("--module-compute-dtype-rule pattern cannot be empty")
        if dtype_raw not in valid_dtypes:
            raise ValueError(
                "--module-compute-dtype-rule dtype must be one of fp32, bf16, fp16"
            )
        parsed.append((pattern, dtype_raw))  # type: ignore[arg-type]
    return tuple(parsed)


def _normalize_precision_recipe_name(raw: object) -> PrecisionRecipeName:
    value = str(raw or "default")
    if value not in ("default", "deepseek_v3"):
        raise ValueError("--precision-recipe must be default or deepseek_v3")
    return value  # type: ignore[return-value]


def _normalize_quant_granularity(
    raw: object,
    *,
    flag_name: str,
) -> QuantGranularity:
    value = str(raw)
    if value not in ("tensor", "channel", "tile_1x128", "block_128x128"):
        raise ValueError(
            f"{flag_name} must be one of tensor, channel, tile_1x128, block_128x128"
        )
    return value  # type: ignore[return-value]


def _normalize_rounding_mode(raw: object) -> RoundingMode:
    value = str(raw)
    if value not in ("nearest", "stochastic"):
        raise ValueError("--fp8-rounding must be nearest or stochastic")
    return value  # type: ignore[return-value]


def _default_deepseek_v3_recipe() -> DeepSeekV3PrecisionRecipe:
    """Return DeepSeek-V3 default precision recipe with conservative exceptions."""
    return DeepSeekV3PrecisionRecipe(
        activation_quant_granularity="tile_1x128",
        weight_quant_granularity="block_128x128",
        rounding_mode="stochastic",
        comm_quant_enabled=True,
        comm_quant_granularity="block_128x128",
        high_precision_module_patterns=(
            r"(^|\\.)token_embeddings$",
            r"(^|\\.)position_embeddings$",
            r"(^|\\.)lm_head$",
            r"(^|\\.)final_norm$",
            r"(^|\\.).*norm$",
            r"(^|\\.).*router(\\.|$)",
        ),
        high_precision_grad_patterns=(
            r"(^|\\.)lm_head(\\.|$)",
            r"(^|\\.).*router(\\.|$)",
        ),
    )


def _build_deepseek_v3_recipe_from_args(args) -> DeepSeekV3PrecisionRecipe:
    default = _default_deepseek_v3_recipe()

    activation_raw = getattr(args, "fp8_activation_granularity", None)
    weight_raw = getattr(args, "fp8_weight_granularity", None)
    rounding_raw = getattr(args, "fp8_rounding", None)
    comm_enabled_raw = getattr(args, "fp8_comm_quant", None)
    comm_granularity_raw = getattr(args, "fp8_comm_granularity", None)

    activation_granularity = default.activation_quant_granularity
    if activation_raw is not None:
        activation_granularity = _normalize_quant_granularity(
            activation_raw,
            flag_name="--fp8-activation-granularity",
        )

    weight_granularity = default.weight_quant_granularity
    if weight_raw is not None:
        weight_granularity = _normalize_quant_granularity(
            weight_raw,
            flag_name="--fp8-weight-granularity",
        )

    rounding_mode = default.rounding_mode
    if rounding_raw is not None:
        rounding_mode = _normalize_rounding_mode(rounding_raw)

    comm_quant_enabled = default.comm_quant_enabled
    if comm_enabled_raw is not None:
        comm_quant_enabled = bool(comm_enabled_raw)

    comm_quant_granularity = default.comm_quant_granularity
    if comm_granularity_raw is not None:
        comm_quant_granularity = _normalize_quant_granularity(
            comm_granularity_raw,
            flag_name="--fp8-comm-granularity",
        )

    return DeepSeekV3PrecisionRecipe(
        activation_quant_granularity=activation_granularity,
        weight_quant_granularity=weight_granularity,
        rounding_mode=rounding_mode,
        comm_quant_enabled=comm_quant_enabled,
        comm_quant_granularity=comm_quant_granularity,
        high_precision_module_patterns=default.high_precision_module_patterns,
        high_precision_grad_patterns=default.high_precision_grad_patterns,
    )


def _match_recipe_pattern(name: str, pattern: str) -> bool:
    return re.search(pattern, name) is not None


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
        module_compute_dtype_rules=_iter_patterns(
            getattr(args, "module_compute_dtype_rule", None)
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


class _ModulePrecisionResolverImpl(ModulePrecisionResolver):
    """Resolve constructor-time per-module precision states from one run policy."""

    def __init__(self, config: PrecisionConfig) -> None:
        self._config = replace(config)
        policy = config.module_precision_policy
        if policy is None:
            policy = ModulePrecisionPolicy()
        self._policy = policy
        self._module_compute_dtype_rules = _parse_module_compute_dtype_rules(
            self._policy.module_compute_dtype_rules
        )

        self._seen_paths: set[str] = set()
        self._compute_include_hits: set[str] = set()
        self._persistent_include_hits: set[str] = set()
        self._high_precision_hits: set[str] = set()
        self._module_compute_dtype_rule_hits: set[tuple[str, PrecisionDType]] = set()

        self._parameterized_count = 0
        self._lowbit_capable_count = 0
        self._compute_lowbit_count = 0
        self._persistent_lowbit_count = 0
        self._high_precision_exception_count = 0

    def _resolve_compute_dtype_override(
        self,
        module_path: str,
    ) -> Optional[PrecisionDType]:
        override: Optional[PrecisionDType] = None
        for pattern, dtype_alias in self._module_compute_dtype_rules:
            if _match_pattern(module_path, pattern, pattern_type=self._policy.pattern_type):
                override = dtype_alias
                self._module_compute_dtype_rule_hits.add((pattern, dtype_alias))
        return override

    def _select(
        self,
        *,
        module_path: str,
        include_patterns: tuple[str, ...],
        exclude_patterns: tuple[str, ...],
        default_all_when_no_include: bool,
    ) -> tuple[bool, bool]:
        include_matched = False
        if include_patterns:
            include_matched = any(
                _match_pattern(module_path, pattern, pattern_type=self._policy.pattern_type)
                for pattern in include_patterns
            )
            selected = include_matched
        else:
            selected = default_all_when_no_include

        if selected and any(
            _match_pattern(module_path, pattern, pattern_type=self._policy.pattern_type)
            for pattern in exclude_patterns
        ):
            selected = False
        return selected, include_matched

    def resolve_module_init_state(
        self,
        *,
        module_path: str,
        module_type: str,
        lowbit_capable_type: Optional[LowBitCapableModuleType],
        kernel_spec: Optional[LowBitKernelSpec] = None,
    ) -> ModulePrecisionInitState:
        if not module_path:
            raise ValueError("module_path must be non-empty for precision resolution")
        if module_path in self._seen_paths:
            raise ValueError(f"Duplicate module_path for precision resolution: {module_path}")

        self._seen_paths.add(module_path)
        self._parameterized_count += 1

        lowbit_capable = lowbit_capable_type in _LOWBIT_CAPABLE_TYPES
        if lowbit_capable:
            self._lowbit_capable_count += 1

        compute_selected, compute_include_hit = self._select(
            module_path=module_path,
            include_patterns=self._policy.compute_lowbit_include,
            exclude_patterns=self._policy.compute_lowbit_exclude,
            default_all_when_no_include=(
                self._policy.compute_lowbit_mode is not None and lowbit_capable
            ),
        )
        if compute_include_hit:
            self._compute_include_hits.add(module_path)

        persistent_selected, persistent_include_hit = self._select(
            module_path=module_path,
            include_patterns=self._policy.persistent_lowbit_include,
            exclude_patterns=self._policy.persistent_lowbit_exclude,
            default_all_when_no_include=(
                self._policy.persistent_lowbit_mode != "off" and lowbit_capable
            ),
        )
        if persistent_include_hit:
            self._persistent_include_hits.add(module_path)

        if compute_selected and not lowbit_capable and self._policy.compute_lowbit_mode is not None:
            raise ValueError(
                "compute_lowbit patterns matched modules without low-bit support: "
                f"{module_path} ({module_type})"
            )

        if persistent_selected and not lowbit_capable and self._policy.persistent_lowbit_mode != "off":
            raise ValueError(
                "persistent_lowbit patterns matched modules without low-bit support: "
                f"{module_path} ({module_type})"
            )

        compute_mode = None
        if lowbit_capable and compute_selected:
            compute_mode = self._policy.compute_lowbit_mode

        persistent_mode: PersistentLowBitMode = "off"
        if lowbit_capable and persistent_selected:
            persistent_mode = self._policy.persistent_lowbit_mode

        recipe = self._config.deepseek_v3_recipe
        high_precision_selected = False
        if recipe is not None and recipe.high_precision_module_patterns:
            high_precision_selected = any(
                _match_recipe_pattern(module_path, pattern)
                for pattern in recipe.high_precision_module_patterns
            )

        if high_precision_selected:
            self._high_precision_hits.add(module_path)
            self._high_precision_exception_count += 1
            compute_mode = None
            persistent_mode = "off"

        compute_dtype_override = self._resolve_compute_dtype_override(module_path)
        if compute_mode is not None and compute_dtype_override is not None:
            raise ValueError(
                "module matched both low-bit compute and compute-dtype override: "
                f"{module_path}. Adjust --compute-lowbit-* or --module-compute-dtype-rule."
            )

        if compute_mode is not None:
            self._compute_lowbit_count += 1
        if persistent_mode != "off":
            self._persistent_lowbit_count += 1

        assignment = ModulePrecisionAssignment(
            module_name=module_path,
            module_type=module_type,
            compute_lowbit_mode=compute_mode,
            persistent_lowbit_mode=persistent_mode,
            persistent_scale_granularity=self._policy.persistent_scale_granularity,
            fp4_persistent_format=self._policy.fp4_persistent_format,
            compute_dtype_override=compute_dtype_override,
        )

        backend: Optional[LowBitBackend] = None
        if compute_mode is not None:
            backend_kernel_spec = kernel_spec
            if backend_kernel_spec is not None and recipe is not None:
                backend_kernel_spec = replace(
                    backend_kernel_spec,
                    activation_quant_granularity=recipe.activation_quant_granularity,
                    weight_quant_granularity=recipe.weight_quant_granularity,
                    rounding_mode=recipe.rounding_mode,
                )
            backend = build_lowbit_backend_for_mode(
                self._config,
                mode=compute_mode,
                kernel_spec=backend_kernel_spec,
            )
            if backend is None:
                raise RuntimeError(
                    f"No backend available for compute mode {compute_mode} at module {module_path}"
                )

        master_ownership_mode: MasterOwnershipMode = self._config.lowbit_master_ownership

        return ModulePrecisionInitState(
            assignment=assignment,
            lowbit_backend=backend,
            lowbit_capable_type=lowbit_capable_type,
            master_ownership_mode=master_ownership_mode,
        )

    def finalize(self) -> ModulePrecisionSummary:
        if self._policy.compute_lowbit_include and not self._compute_include_hits:
            raise ValueError("compute_lowbit include patterns matched zero modules")
        if self._policy.persistent_lowbit_include and not self._persistent_include_hits:
            raise ValueError("persistent_lowbit include patterns matched zero modules")
        unmatched_dtype_rules = [
            f"{pattern}={dtype_alias}"
            for pattern, dtype_alias in self._module_compute_dtype_rules
            if (pattern, dtype_alias) not in self._module_compute_dtype_rule_hits
        ]
        if unmatched_dtype_rules:
            raise ValueError(
                "module compute dtype rules matched zero modules: "
                f"{unmatched_dtype_rules[:3]}"
            )

        if self._config.mode in ("fp8", "fp4") and self._compute_lowbit_count == 0:
            raise RuntimeError(
                f"Low-bit mode '{self._config.mode}' is active but zero modules were assigned "
                "for low-bit compute. Check --compute-lowbit-include/--compute-lowbit-exclude "
                "or verify the model uses low-bit-capable layers."
            )

        recipe = self._config.deepseek_v3_recipe
        if (
            recipe is not None
            and recipe.high_precision_module_patterns
            and not self._high_precision_hits
        ):
            raise ValueError(
                "DeepSeek-V3 high_precision_module_patterns matched zero modules"
            )

        return ModulePrecisionSummary(
            parameterized_module_count=self._parameterized_count,
            lowbit_capable_module_count=self._lowbit_capable_count,
            compute_lowbit_module_count=self._compute_lowbit_count,
            persistent_lowbit_module_count=self._persistent_lowbit_count,
            high_precision_exception_module_count=self._high_precision_exception_count,
        )

    def deepseek_v3_recipe(self) -> Optional[DeepSeekV3PrecisionRecipe]:
        """Return recipe configured for this run when DeepSeek-V3 preset is active."""
        return self._config.deepseek_v3_recipe


def build_module_precision_resolver(config: PrecisionConfig) -> ModulePrecisionResolver:
    """Build constructor-time precision resolver for one runtime precision config."""
    return _ModulePrecisionResolverImpl(config)


def finalize_module_precision_resolver(
    resolver: ModulePrecisionResolver,
) -> ModulePrecisionSummary:
    """Validate resolver coverage and return assignment summary."""
    return resolver.finalize()


def refresh_persistent_lowbit_params(model: torch.nn.Module) -> int:
    """Refresh persistent low-bit params for modules that expose the hook."""
    refreshed = 0
    for module in model.modules():
        refresh_fn = getattr(module, "refresh_persistent_lowbit_params", None)
        if callable(refresh_fn):
            refresh_fn()
            refreshed += 1
    return refreshed


def resolve_precision_config(args, device: torch.device) -> PrecisionConfig:
    """Resolve run precision config from Megatron-style precision flags."""
    precision_recipe_name = _normalize_precision_recipe_name(
        getattr(args, "precision_recipe", "default")
    )
    explicit_mode = _resolve_mode_from_flags(args)
    if precision_recipe_name == "deepseek_v3":
        if explicit_mode is None:
            mode: PrecisionMode = "fp8"
        elif explicit_mode != "fp8":
            raise ValueError("--precision-recipe deepseek_v3 requires --fp8 (or no explicit mode)")
        else:
            mode = explicit_mode
    else:
        mode = explicit_mode or _default_mode_for_device(device)

    deepseek_v3_recipe: Optional[DeepSeekV3PrecisionRecipe] = None
    if precision_recipe_name == "deepseek_v3":
        deepseek_v3_recipe = _build_deepseek_v3_recipe_from_args(args)

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
    lowbit_master_ownership = str(getattr(args, "lowbit_master_ownership", "optimizer"))

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
    if lowbit_master_ownership not in ("module", "optimizer"):
        raise ValueError("--lowbit-master-ownership must be module or optimizer")

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
        lowbit_master_ownership=lowbit_master_ownership,  # type: ignore[arg-type]
        precision_recipe_name=precision_recipe_name,
        deepseek_v3_recipe=deepseek_v3_recipe,
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
        _ = build_lowbit_backend_for_mode(config, mode=required_mode, kernel_spec=None)

    return config


class MixedPrecisionController:
    """Apply autocast and dynamic loss-scaling policies."""

    def __init__(
        self,
        config: PrecisionConfig,
        *,
        device: torch.device,
    ) -> None:
        self.config = replace(config)
        self.device = device
        self.runtime_state = PrecisionRuntimeState(loss_scale=float(config.loss_scale_init))

    @property
    def uses_loss_scaling(self) -> bool:
        return self.config.mode in ("fp16", "fp8", "fp4")

    def activation_torch_dtype(self) -> torch.dtype:
        """Return activation compute dtype as torch dtype."""
        return dtype_alias_to_torch(self.config.activation_dtype)

    @contextmanager
    def autocast_context(self) -> Iterator[None]:
        """Context manager for per-step autocast policy."""
        with self._autocast_context_impl():
            yield

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
