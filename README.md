# Nano-Train: A Distributed LLM Training Framework

A production-grade distributed LLM training framework from scratch, targeting
models like DeepSeek-R1 (671B MoE parameters). The framework will implement
state-of-the-art parallelism strategies (tensor, pipeline, data, sequence, and
expert parallelism), memory optimizations, and training infrastructure similar
to Megatron-LM but with a cleaner, more modular architecture.

## 🎯 Goal

Train SOTA LLMs (7B to 671B parameters) with support for:
- Mixture-of-Experts (MoE) architectures like DeepSeek-R1
- 3D parallelism: Tensor, Pipeline, Data, Sequence, Expert
- Scale to 1000+ GPUs
- Efficient memory usage for training large models

## 🚀 Quick Start (Google Colab)

**New:** Train with GPU in Google Colab!

1. Open [notebooks/train_in_colab.ipynb](notebooks/train_in_colab.ipynb) in Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → T4 GPU)
3. Run all cells to automatically train

**Local Development:**
- Run [check_gpu.py](check_gpu.py) to verify your local GPU setup
- Use [scripts/sync_and_run.sh](scripts/sync_and_run.sh) to automate workflow

## 📁 Project Structure

```
nano_train/
├── core/                    # Core infrastructure
│   ├── config.py           # Configuration system
│   ├── distributed.py      # Distributed initialization
│   └── logging.py          # Logging utilities
├── models/                 # Model architectures
│   ├── transformer.py      # Transformer blocks
│   ├── attention.py        # Attention mechanisms (MHA, GQA, MQA)
│   ├── mlp.py              # MLP/MoE layers
│   └── embedding.py        # Embeddings & RoPE
├── parallelism/            # Parallelism strategies
│   ├── tensor_parallel.py  # Tensor parallelism
│   ├── pipeline_parallel.py # Pipeline parallelism
│   ├── data_parallel.py    # DDP/FSDP wrappers
│   ├── sequence_parallel.py # Sequence parallelism
│   └── expert_parallel.py  # Expert parallelism (MoE)
├── memory/                 # Memory optimization
│   ├── checkpointing.py    # Gradient/activation checkpointing
│   ├── offload.py          # CPU offloading
│   └── metrics.py          # Memory tracking
├── communication/          # Communication primitives
│   ├── collectives.py      # All-reduce, all-gather wrappers
│   └── overlap.py          # Computation-communication overlap
├── training/               # Training infrastructure
│   ├── optimizer.py        # Optimizers (AdamW, etc.)
│   ├── scheduler.py        # LR schedulers
│   ├── checkpoint.py       # Checkpoint saving/loading
│   └── trainer.py          # Main training loop
├── data/                   # Data loading
│   ├── dataset.py          # Dataset classes
│   ├── loader.py           # DataLoader wrappers
│   └── preprocessing.py    # Tokenization & preprocessing
├── kernels/                # Custom CUDA kernels
│   ├── flash_attention.py  # Flash Attention interface
│   ├── rotary.py           # RoPE kernels
│   └── moe_routing.py      # MoE routing kernels
└── utils/                  # Utilities
    ├── metrics.py          # Training metrics
    └── timers.py           # Performance timing
examples/                   # Example training scripts
configs/                    # Configuration files
tests/                      # Test suite
notebooks/                   # Google Colab notebooks
scripts/                    # Automation scripts
```

## 🏃️ Installation

```bash
git clone https://github.com/lastweek/nano-train.git
cd nano-train

# For local development with GPU:
python check_gpu.py  # Verify GPU setup

# For training:
pip install -r requirements.txt
python examples/train_mvp.py

# View logs (TensorBoard):
./scripts/start_tensorboard.sh
```

## 📊 Current Status

### ✅ Phase 0 Complete (Weeks 1-3)
**MVP Training Cycle Working**
- [x] Configuration system (dataclass-based)
- [x] Basic transformer block (MHA, MLP)
- [x] Training loop with optimizer & scheduler
- [x] Simple data loader
- [x] Character-level vocab for MVP

**Training Results (125M model):**
- Steps completed: 1000/1000
- Training time: 33 minutes 24 seconds
- Final loss: 0.0000 (decreased from ~3.5)
- Loss decrease: ✅ Model is learning
- Checkpointing: ✅ Working

### 🔄 In Progress
- [ ] Phase 1 (Weeks 4-6): Production-ready foundation (OmegaConf + Hydra, distributed training)
- [ ] Phase 2 (Weeks 7-10): Flash Attention, gradient checkpointing, BF16
- [ ] Phase 3 (Weeks 11-14): Tensor Parallelism
- [ ] Phase 4 (Weeks 15-16): Data Parallelism
- [ ] Phase 5 (Weeks 17-18): Attention enhancements (RoPE, GQA)
- [ ] Phase 6 (Weeks 19-20): Pipeline Parallelism
- [ ] Phase 7 (Weeks 21-24): Mixture-of-Experts (MoE)
- [ ] Phase 8 (Weeks 25-26): Sequence Parallelism
- [ ] Phase 9 (Weeks 27-30): Production features (checkpointing, monitoring)
- [ ] Phase 10 (Weeks 31-34): Advanced optimization (fused ops, CPU offload)
- [ ] Phase 11 (Weeks 35-36): Production hardening (testing, docs)

## 📌 Progress Tracker

This section tracks major repo improvements in chronological order.
Use this as the source of truth for "what changed when".

### Completed Milestones

| Date | Commit | Milestone | Major Files Changed |
|---|---|---|---|
| 2026-02-09 | `5cfeb63` | Initial repo bootstrap | `README.md`, `src/*`, `examples/*`, `tests/*` |
| 2026-02-13 | `8044208` | MVP stack refactor + model efficiency reporting | `src/trainer.py`, `src/utils/model_info.py`, `docs/model_info.md` |
| 2026-02-19 | `9c12e7e` | Monitoring v2 stability/perf metrics | `src/trainer.py`, `src/config.py`, `src/monitoring.py`, `docs/training_monitoring_metrics_reference.md` |
| 2026-02-24 | `5206984` | Canonical TP + DP tutorial pipeline | `examples/tp.py`, `src/layers.py`, `docs/tp_dp_communication.md` |
| 2026-02-25 | `64b9df3` | EP tutorial path (TP + EP + DP) | `examples/ep.py`, `src/models/moe.py`, `src/models/deepseek.py`, `docs/ep_tp_dp_communication.md` |
| 2026-02-25 | `5855268` | Docs IA/readability overhaul | `docs/README.md`, `docs/*.md`, `README.md`, `src/utils/model_info.py` |

### Planned Next Milestones

| Status | Milestone | Expected Focus Files |
|---|---|---|
| In Progress | 4D TP+PP+EP+DP tutorial path | `examples/ep.py`, `src/distributed/topology.py`, `src/models/deepseek.py`, `docs/pp_tp_ep_dp_communication.md` |
| Planned | EP robustness hardening (EDP sync/diagnostics + checks) | `examples/ep.py`, `src/models/moe.py`, `tests/test_ep_script_logic.py` |
| Planned | DeepSeek parallel context cleanup and simplification | `src/models/deepseek.py`, `tests/test_deepseek_model.py` |
| Planned | TP/EP learning script consistency pass | `examples/tp.py`, `examples/ep.py`, `docs/ep_tp_dp_communication.md` |
| Planned | Device-level MoE aux loss (`L_devbal`) support | `src/models/moe.py`, `examples/ep.py`, `docs/deepseek_moe_aux_losses.md` |

## 🗺️ Roadmap

1. **Milestone 1:** Train 1B parameter model (current target: 125M ✅)
2. **Milestone 2:** Add Flash Attention for 10x speedup
3. **Milestone 3:** Implement tensor parallelism (TP=8)
4. **Milestone 4:** Train 7B dense model
5. **Milestone 5:** Implement MoE for DeepSeek-R1 style models
6. **Final Goal:** Train 671B MoE model at scale

## 🔧 Development Workflow

### Local Development
```bash
# 1. Make changes locally
# 2. Push to GitHub
./scripts/sync_and_run.sh
```

### Google Colab Training
```bash
# 1. Open notebooks/train_in_colab.ipynb in Colab
# 2. Enable GPU runtime
# 3. Run all cells
```

The [sync_and_run.sh](scripts/sync_and_run.sh) script automates:
1. Detects local git changes
2. Commits and pushes to GitHub
3. Generates a Colab notebook script
4. The Colab script pulls latest code and starts training

## 📈 Architecture Decisions

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Framework | PyTorch | Industry standard, best distributed support |
| Attention | Flash Attention | Proven performance, widely adopted |
| Parallelism | Support all types (TP, PP, DP, SP, EP) | Maximum flexibility |
| Precision | BF16 primary | Better numerical stability than FP16 |
| Configuration | OmegaConf + Hydra | Flexible, hierarchical configs |
| Checkpointing | Sharded for training, full for inference | Balance storage and compatibility |

## 📘 Learning Guides

- [Docs Index](docs/README.md) - organized navigation across guides, operations, and reference docs.
- [TP + DP Backward Flow](docs/tp_dp_communication.md) - communication domains, collectives,
  and gradient flow in 2D parallel training.
- [TP + EP + DP Communication](docs/ep_tp_dp_communication.md) - expert dispatch/return
  all-to-all flow, gradient synchronization domains, and expert TP=1 rationale.
- [TP + PP + EP + DP Communication](docs/pp_tp_ep_dp_communication.md) - 4D topology,
  non-interleaved 1F1B pipeline schedule, stage-to-stage communication, and label transfer.
- [DeepSeekMoE Aux Losses](docs/deepseek_moe_aux_losses.md) - expert/device load-balance
  objectives from DeepSeekMoE, plus mapping to this repo's current implementation.

## 📚 References

- [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - 3D parallelism reference
- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [DeepSeek-R1 GitHub](https://github.com/deepseek-ai/DeepSeek-R1) - MoE architecture reference

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

Built with inspiration from:
- NVIDIA Megatron-LM team
- DeepSeek-AI team
- PyTorch team
- And the broader open-source ML community
