# Nano-Train

Learning-first distributed training repo focused on Megatron-style parallelism:
Tensor (TP), Pipeline (PP), Expert (EP), Data (DP), and ZeRO-1/2.

## Current Status

- Main tutorial entrypoint: `examples/train_4p.py` (TP/PP/EP/DP + ZeRO-1/2).
- Focused TP/DP learning path: `examples/tp.py`.
- ZeRO implementation: `src/distributed/zero.py` (`optim`, `optim_grads`).
- DeepSeek-style model stack:
  - `src/models/deepseek.py`
  - `src/models/moe.py`
- Current tutorial constraints in `examples/train_4p.py`:
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
python3 examples/train_4p.py \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --max_steps 2
```

### 4P + ZeRO-2 Smoke

```bash
python3 examples/launch.py --world-size 4 --backend gloo \
  --script examples/train_4p.py --script-args \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 2 \
  --expert-model-parallel-size 2 \
  --num_microbatches 2 \
  --use-distributed-optimizer \
  --data-parallel-sharding-strategy optim_grads \
  --max_steps 1
```

## Key Entrypoints

| Path | Purpose |
|---|---|
| `examples/train_4p.py` | Canonical TP/PP/EP/DP tutorial script with optional ZeRO-1/2 |
| `examples/tp.py` | TP-only and TP+DP tutorial path |
| `examples/launch.py` | Multi-process launcher for local distributed runs |
| `src/distributed/topology.py` | Parallel group/rank topology setup |
| `src/distributed/zero.py` | Megatron-style ZeRO-1/2 optimizer implementation |
| `src/trainer.py` | Shared trainer loop and checkpoint integration hooks |

## Architecture Diagram

```mermaid
flowchart TB
    subgraph E["Entry Points (`examples/`)"]
        E1["train_4p.py\nCanonical TP/PP/EP/DP + ZeRO tutorial"]
        E2["tp.py\nTP-only and TP+DP tutorial"]
        E3["launch.py\nMulti-process launcher"]
        E4["mvp.py / deepseek.py / ddp.py\nFocused demos"]
    end

    subgraph O["Orchestration (`src/`)"]
        O1["trainer.py\nTraining loop + checkpoint hooks"]
        O2["config.py\nTyped runtime config"]
        O3["dataset.py\nDataset + dataloader helpers"]
        O4["logging.py\nStructured logging"]
        O5["optimizer.py + scheduler.py\nOptimization policies"]
    end

    subgraph D["Distributed Runtime (`src/distributed/`)"]
        D1["device.py\nDevice/backend selection"]
        D2["topology.py\nTP/PP/EP/DP process groups"]
        D3["zero.py\nMegatron-style ZeRO-1/2"]
    end

    subgraph M["Model Stack (`src/models/` + `src/layers.py`)"]
        M1["deepseek.py\nDeepSeek model + parallel context"]
        M2["moe.py\nRouter + Local/EP MoE"]
        M3["transformer.py\nBaseline transformer model"]
        M4["attention.py + mlp.py\nBlock components"]
        M5["layers.py\nParallel linear + seq-parallel primitives"]
        M6["losses.py\nTraining loss modules"]
    end

    subgraph X["Observability & Utility"]
        X1["monitoring.py\nTraining diagnostics"]
        X2["utils/model_info.py\nModel size/compute reporting"]
    end

    subgraph V["Verification & Knowledge"]
        V1["tests/\nUnit + distributed smoke"]
        V2["docs/\nCommunication + ZeRO guides"]
    end

    E3 --> E1
    E3 --> E2
    E3 --> E4

    E1 --> D1
    E1 --> D2
    E1 --> D3
    E1 --> O3
    E1 --> O4
    E1 --> M1
    E1 --> M2
    E1 --> M5

    E2 --> D1
    E2 --> O3
    E2 --> O4
    E2 --> M5

    E4 --> O1
    O1 --> O2
    O1 --> O3
    O1 --> O4
    O1 --> O5
    O1 --> M6
    O1 --> X1

    D3 --> D2

    M1 --> M2
    M1 --> M5
    M3 --> M4
    M3 --> M5
    M4 --> M5
    M2 --> M5

    X2 --> M1
    X2 --> M3

    V1 -. validates .-> E1
    V1 -. validates .-> E2
    V1 -. validates .-> D3
    V1 -. validates .-> M1
    V2 -. documents .-> E1
    V2 -. documents .-> D2
    V2 -. documents .-> D3
    V2 -. documents .-> M2
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
| 2026-02-19 | `9c12e7e` | Monitoring v2 stability/perf metrics | `src/trainer.py`, `src/config.py`, `src/monitoring.py`, `docs/training_monitoring_metrics_reference.md` |
| 2026-02-24 | `5206984` | Canonical TP + DP tutorial pipeline | `examples/tp.py`, `src/layers.py`, `docs/tp_dp_communication.md` |
| 2026-02-25 | `64b9df3` | EP tutorial path (TP + EP + DP) | `examples/train_4p.py`, `src/models/moe.py`, `src/models/deepseek.py`, `docs/ep_tp_dp_communication.md` |
| 2026-02-25 | `5855268` | Docs IA/readability overhaul | `docs/README.md`, `docs/*.md`, `README.md`, `src/utils/model_info.py` |
| 2026-02-26 | `69188d8` | 4P entrypoint rename + ZeRO-1/2 integration and debug visibility | `examples/train_4p.py`, `src/distributed/zero.py`, `src/trainer.py`, `docs/zero1_zero2_*.md`, `tests/test_zero_*.py` |

### Planned Next Milestones

| Status | Milestone | Expected Focus Files |
|---|---|---|
| In Progress | Canonical TP+EP mapping (remove TP+EP guard, avoid expert replication) | `examples/train_4p.py`, `src/distributed/topology.py`, `src/models/deepseek.py`, `docs/ep_tp_dp_communication.md`, `docs/pp_tp_ep_dp_communication.md` |
| Planned | EP robustness hardening (EDP sync/diagnostics + checks) | `examples/train_4p.py`, `src/models/moe.py`, `tests/test_train_4p_script_logic.py` |
| Planned | DeepSeek parallel context cleanup and simplification | `src/models/deepseek.py`, `tests/test_deepseek_model.py` |
| Planned | Device-level MoE aux loss (`L_devbal`) support | `src/models/moe.py`, `examples/train_4p.py`, `docs/deepseek_moe_aux_losses.md` |
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
