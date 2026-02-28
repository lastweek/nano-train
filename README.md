# Nano-Train

Learning-first distributed training repo focused on Megatron-style parallelism:
Tensor (TP), Pipeline (PP), Expert (EP), Data (DP), and ZeRO-1/2.

## Current Status

- Main tutorial entrypoint: `examples/train_4d.py` (TP/PP/EP/DP + ZeRO-1/2).
- Focused TP/DP learning path: `examples/train_tp.py`.
- ZeRO implementation: `src/distributed/zero.py` (`optim`, `optim_grads`).
- DeepSeek-style model stack:
  - `src/models/deepseek.py`
  - `src/models/moe.py`
- Current tutorial constraints in `examples/train_4d.py`:
  - `tensor-model-parallel-size > 1` with `expert-model-parallel-size > 1` is disallowed.
  - `expert-tensor-parallel-size == 1`.
  - `context-parallel-size == 1`.
  - ZeRO-3 (`optim_grads_params`) is out of scope.

## Quick Start

### Install

```bash
git clone https://github.com/lastweek/nano-train.git
cd nano-train
pip install -r requirements.txt
```

### Single-Rank Smoke

```bash
python3 examples/train_4d.py \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --max_steps 2
```

### 4P + ZeRO-2 Smoke

```bash
python3 examples/launch.py --world-size 4 --backend gloo \
  --script examples/train_4d.py --script-args \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 2 \
  --expert-model-parallel-size 2 \
  --num_microbatches 2 \
  --use-distributed-optimizer \
  --data-parallel-sharding-strategy optim_grads \
  --max_steps 1
```

### Mixed Precision Smoke (FP8 Emulated)

```bash
python3 examples/train_4d.py \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --fp8 \
  --fp8-backend emulated \
  --lowbit-master-ownership optimizer \
  --params-dtype bf16 \
  --main-params-dtype fp32 \
  --main-grads-dtype fp32 \
  --exp-avg-dtype fp32 \
  --exp-avg-sq-dtype fp32 \
  --max_steps 1
```

Notes:
- Low-bit execution is strict per-module and bound at module construction time.
- ZeRO checkpoint optimizer payloads now use format v2 only (old formats are unsupported).

### DeepSeek-V3 Recipe Smoke (FP8 Emulated)

```bash
python3 examples/train_4d.py \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --precision-recipe deepseek_v3 \
  --fp8 \
  --fp8-backend emulated \
  --max_steps 1
```

Recipe notes:
- `--precision-recipe deepseek_v3` defaults to FP8 with:
  - activation quant granularity `tile_1x128`
  - weight quant granularity `block_128x128`
  - rounding mode `stochastic`
  - MoE dispatch/combine payload comm-quant enabled
- You can override each recipe field with `--fp8-*` flags.
- For `DeepSeekModel`, prefer exact-path config overrides through
  `DeepSeekModelConfig.module_compute_dtype_overrides`.
- Generic CLI fallback is still available for quick experiments or non-DeepSeek scripts:
  `--module-compute-dtype-rule '<module_pattern>=<fp32|bf16|fp16>'`.

DeepSeek config-first example:

```python
model_config = build_tiny_deepseek_config(
    args,
    param_dtype=param_dtype,
    param_device=parallel.device,
    precision_resolver=build_module_precision_resolver(precision_config),
    module_compute_dtype_overrides={
        "blocks.0.attn.q_a_norm": "fp16",
    },
)
```

See [DeepSeek Precision Configuration](docs/deepseek_precision_configuration.md) for
exact module-path control, precedence, and fallback behavior.

## Key Entrypoints

| Path | Purpose |
|---|---|
| `examples/train_4d.py` | Canonical TP/PP/EP/DP tutorial script with optional ZeRO-1/2 |
| `examples/train_tp.py` | TP-only and TP+DP tutorial path |
| `examples/launch.py` | Multi-process launcher for local distributed runs |
| `src/distributed/topology.py` | Parallel group/rank topology setup |
| `src/distributed/zero.py` | Megatron-style ZeRO-1/2 optimizer implementation |
| `src/runtime/engine.py` | Runtime orchestration engine (mode dispatch + loops) |
| `src/runtime/contracts.py` | Runtime component contracts (`providers`, `schedule`, `optimizer`, `checkpoint`) |
| `src/trainer.py` | Shared trainer loop and checkpoint integration hooks |

## Runtime Core

`examples/train_4d.py`, `examples/train_tp.py`, `examples/train_ddp.py`, and `examples/train_mvp.py`
now use a thin-script pattern:

- Script owns CLI and component wiring.
- `src/runtime/engine.py` owns orchestration flow.
- Model/data/optimizer/schedule/checkpoint behavior is provided through components.
  Example-specific component implementations live in each script under `examples/`.
- See [Runtime Core Design](docs/runtime_core_design.md) for architecture, API,
  responsibilities, and script usage patterns.

## Architecture Diagram

See `docs/runtime_core_design.md` for full API and extension details.

```mermaid
flowchart TD
    subgraph E["Entry Scripts (`examples/`)"]
        E1["train_4d.py"]
        E2["train_tp.py"]
        E3["train_ddp.py"]
        E4["train_mvp.py"]
        E5["Parse args + assemble RuntimeComponents"]
    end

    subgraph C["Component Bundle (`RuntimeComponents`)"]
        C0["RuntimeComponents"]
        C1["RuntimeBootstrap"]
        C2["ModelProvider"]
        C3["DataProvider"]
        C4["OptimizerRuntime"]
        C5["ScheduleSelector"]
        C6["ScheduleStrategy"]
        C7["CheckpointManager"]
    end

    subgraph R["Runtime Engine (`src/runtime/engine.py`)"]
        R0["RuntimeEngine.run(...)"]
        R1["build_context"]
        R2["build model/data"]
        R3["init optimizer state"]
        R4["select schedule"]
        R5["load checkpoint"]
        R6["step loop"]
        R7["checkpoint hooks"]
        R8["finalize + destroy process group"]
    end

    subgraph H["Runtime Helpers (`src/runtime/*`)"]
        H1["sync.py"]
        H2["pipeline.py"]
        H3["optimizer_runtime.py"]
        H4["checkpoint.py"]
    end

    subgraph V["Verification & Docs"]
        V1["tests/* runtime + script wiring"]
        V2["docs/runtime_core_design.md"]
    end

    E1 --> E5
    E2 --> E5
    E3 --> E5
    E4 --> E5
    E5 --> C0
    C0 --> R0

    C0 --> C1
    C0 --> C2
    C0 --> C3
    C0 --> C4
    C0 --> C5
    C5 --> C6
    C0 --> C7

    C1 --> R1
    C2 --> R2
    C3 --> R2
    C4 --> R3
    C5 --> R4
    C7 --> R5

    R0 --> R1
    R1 --> R2
    R2 --> R3
    R3 --> R4
    R4 --> R5
    R5 --> R6
    R6 --> C6
    C6 --> R6
    R6 --> R7
    R7 --> R8

    C4 -. uses .-> H1
    C6 -. uses .-> H2
    C4 -. uses .-> H3
    C7 -. uses .-> H4
    R0 -. uses .-> H2
    R0 -. uses .-> H4

    V1 -. validates .-> R0
    V1 -. validates .-> E1
    V1 -. validates .-> E2
    V1 -. validates .-> E3
    V1 -. validates .-> E4
    V2 -. documents .-> R0
    V2 -. documents .-> C0
```

## Learning Guides

- [Docs Index](docs/README.md)
- [TP + DP Communication](docs/tp_dp_communication.md)
- [TP + EP + DP Communication](docs/ep_tp_dp_communication.md)
- [TP + PP + EP + DP Communication](docs/pp_tp_ep_dp_communication.md)
- [DeepSeekMoE Auxiliary Losses](docs/deepseek_moe_aux_losses.md)
- [Megatron ZeRO-1/2 Design](docs/megatron_zero1_zero2_design.md)
- [ZeRO-1/2 Intuitive Summary](docs/zero1_zero2_intuitive_summary.md)
- [ZeRO-1/2 Quickstart](docs/zero1_zero2_quickstart.md)

## Progress Tracker

### Completed Milestones

| Date | Commit | Milestone | Major Files Changed |
|---|---|---|---|
| 2026-02-09 | `5cfeb63` | Initial repo bootstrap | `README.md`, `src/*`, `examples/*`, `tests/*` |
| 2026-02-13 | `8044208` | MVP stack refactor + model efficiency reporting | `src/trainer.py`, `src/utils/model_info.py`, `docs/model_info.md` |
| 2026-02-19 | `9c12e7e` | Monitoring stability/perf metrics | `src/trainer.py`, `src/config.py`, `src/monitoring.py`, `docs/training_monitoring_metrics_reference.md` |
| 2026-02-24 | `5206984` | Canonical TP + DP tutorial pipeline | `examples/train_tp.py`, `src/layers.py`, `docs/tp_dp_communication.md` |
| 2026-02-25 | `64b9df3` | EP tutorial path (TP + EP + DP) | `examples/train_4d.py`, `src/models/moe.py`, `src/models/deepseek.py`, `docs/ep_tp_dp_communication.md` |
| 2026-02-25 | `5855268` | Docs IA/readability overhaul | `docs/README.md`, `docs/*.md`, `README.md`, `src/utils/model_info.py` |
| 2026-02-26 | `69188d8` | 4P entrypoint rename + ZeRO-1/2 integration and debug visibility | `examples/train_4d.py`, `src/distributed/zero.py`, `src/trainer.py`, `docs/zero1_zero2_*.md`, `tests/test_zero_*.py` |

### Planned Next Milestones

| Status | Milestone | Expected Focus Files |
|---|---|---|
| In Progress | Canonical TP+EP mapping (remove TP+EP guard, avoid expert replication) | `examples/train_4d.py`, `src/distributed/topology.py`, `src/models/deepseek.py`, `docs/ep_tp_dp_communication.md`, `docs/pp_tp_ep_dp_communication.md` |
| Planned | EP robustness hardening (EDP sync/diagnostics + checks) | `examples/train_4d.py`, `src/models/moe.py`, `tests/test_train_4d_script_logic.py` |
| Planned | DeepSeek parallel context cleanup and simplification | `src/models/deepseek.py`, `tests/test_deepseek_model.py` |
| Planned | Device-level MoE aux loss (`L_devbal`) support | `src/models/moe.py`, `examples/train_4d.py`, `docs/deepseek_moe_aux_losses.md` |
| Planned | Checkpoint resume path for ZeRO sharded optimizer in trainer | `src/trainer.py`, `src/distributed/zero.py`, `docs/zero1_zero2_quickstart.md` |

## Repository Layout

```text
docs/         learning and implementation guides
examples/     runnable training/tutorial scripts
src/          core training/model/distributed modules
tests/        unit and distributed smoke tests
scripts/      local helper scripts
```

## Development Checks

```bash
pytest -q
ruff check .
ruff format --check .
mypy src/
```

## License

MIT License.
