# MCTS Hardware Benchmark Suite

Comprehensive benchmarking suite for comparing Monte Carlo Tree Search (MCTS) performance across different hardware platforms and evaluation methods.

Designed for fair hardware comparison against the **ReasonCore MCTS accelerator** with support for both traditional random rollout and neural network evaluation approaches.

---

## 🎯 Overview

This benchmark suite provides **4 primary implementations**:

| Platform | Method | Language | Purpose |
|----------|--------|----------|---------|
| **CPU Traditional** | Random rollout | C++ | Baseline for traditional MCTS |
| **CPU NN** | Neural network | C++ | Algorithmic improvement baseline |
| **GPU Traditional** | Random rollout | Python/Numba | GPU baseline for traditional MCTS |
| **GPU NN** | Neural network | CUDA C++ | GPU accelerated NN-MCTS |

This enables **fair comparisons**:
- **Algorithmic**: Random vs NN on same hardware (isolate algorithm impact)
- **Hardware**: CPU vs GPU vs ASIC with same algorithm (isolate hardware impact)

---

## ⚡ Quick Start

**See [QUICKSTART.md](QUICKSTART.md) for copy-paste commands to run each benchmark.**

### Build All Benchmarks
```bash
make all
```

### Run Examples
```bash
# CPU Traditional (random rollout)
./benchmark_cpu_traditional --board-size 9 --iterations 5000

# CPU Neural Network
./benchmark_cpu_nn --board-size 9 --iterations 1000

# GPU Neural Network (requires CUDA)
./benchmark_gpu_nn --board-size 9 --iterations 1000

# GPU Traditional (Python)
cd benchmarks/benchmark_gpu_traditional && python benchmark.py --board-size 9 --iterations 5000
```

---

## 📊 Features

### Evaluation Methods
- ✅ **Traditional MCTS**: Random simulation rollouts (baseline approach)
- ✅ **Neural Network MCTS**: Learned position evaluation (AlphaZero-style)
- ✅ **Fair Comparison**: Same algorithm across all platforms

### Hardware Platforms
- ✅ **CPU**: Optimized C++17 with -O3 compilation
- ✅ **GPU**: CUDA C++ (NN) and Python/Numba (traditional)
- ✅ **Dual-Mode GPU** (traditional only):
  - **Fair Comparison Mode**: Single-tree, sequential playouts
  - **Capability Mode**: Multi-tree, parallel playouts (max throughput)

### Energy Measurement
- ✅ **Intel CPU**: RAPL hardware counters (±1-5% accuracy)
- ✅ **AMD CPU**: TDP-based estimation with RAPL when available
- ✅ **NVIDIA GPU**: TDP-based estimation (NVML integration pending)

### Output Format
- ✅ **Standardized CSV**: Per-phase timing breakdown
- ✅ **Detailed Metrics**: Energy, throughput, tree statistics
- ✅ **Publication-Ready**: Direct use in analysis and papers

---

## 📁 Directory Structure

```
mcts-hardware-benchmark/
├── QUICKSTART.md                       # Quick command reference
├── README.md                           # This file
├── Makefile                            # Build all benchmarks
│
├── benchmarks/                         # All 4 primary benchmarks
│   ├── benchmark_cpu_traditional.cpp   # CPU random rollout (C++)
│   ├── benchmark_cpu_nn.cpp            # CPU NN evaluation (C++)
│   ├── benchmark_gpu_nn.cu             # GPU NN evaluation (CUDA)
│   └── benchmark_gpu_traditional/      # GPU random rollout (Python)
│       ├── benchmark.py                # Main entry point
│       ├── mctsnc.py                   # Core MCTS + Numba CUDA
│       └── README.md                   # GPU-specific docs
│
├── include/                            # Headers
│   └── nn_inference.h                  # NN inference library (CPU/GPU)
│
├── weights/                            # Pre-trained neural networks
│   ├── 2x2/, 3x3/, 5x5/, 9x9/, 13x13/, 19x19/
│   │   ├── weights1.bin                # Layer 1 (input → hidden)
│   │   └── weights2.bin                # Layer 2 (hidden → output)
│   └── convert_weights_to_binary.py    # Converts .pkl to binary
│
├── results/                            # Benchmark outputs (CSV)
│
├── analysis/                           # Post-processing
│   ├── tables/                         # Generated comparison tables
│   └── utils/                          # Analysis scripts
│
├── docs/                               # Documentation
│   ├── QUICKSTART_GPU_SERVER.md        # GPU cluster setup guide
│   └── FINAL_RESULTS_SUMMARY.md        # Result analysis
│
└── archive/                            # Archived/deprecated code
```

---

## 🔧 Requirements

### CPU Benchmarks
- **Compiler**: g++ 7.0+ with C++17 support
- **OS**: Linux (for RAPL energy monitoring)
- **Dependencies**: None (plain C++ implementation)

### GPU NN Benchmark (CUDA)
- **GPU**: NVIDIA GPU with CUDA Compute Capability 7.0+
- **CUDA**: 11.0 or later
- **Libraries**: cuBLAS
- **Compiler**: nvcc

### GPU Traditional Benchmark (Python)
- **Python**: 3.8+
- **GPU**: NVIDIA GPU
- **Libraries**: Numba, NumPy, CUDA toolkit
- Install: `cd benchmarks/benchmark_gpu_traditional && pip install -r requirements.txt`

---

## 🧪 Neural Network Architectures

Pre-trained 2-layer feedforward networks for board size `N×N`:

| Board | Architecture | Accuracy | Input Size | File Size |
|-------|--------------|----------|------------|-----------|
| 2×2 | 8→16→3 | 46.3% | 8 | 520 B + 200 B |
| 3×3 | 18→24→3 | 83.6% | 18 | 1.7 KB + 296 B |
| 5×5 | 50→32→3 | 81.7% | 50 | 6.4 KB + 392 B |
| 9×9 | 162→96→3 | 82.0% | 162 | 62 KB + 1.2 KB |
| 13×13 | 338→128→3 | 83.9% | 338 | 173 KB + 1.5 KB |
| 19×19 | 722→192→3 | 90.2% | 722 | 555 KB + 2.3 KB |

**Training**: Supervised learning on self-play games with outcome labels (white wins, draw, black wins).

**Format**: Binary files with 8-byte header (rows, cols) + float32 weights in row-major order.

---

## 📈 Typical Workflow

### 1. Fair Algorithm Comparison (Same Hardware)
```bash
# CPU: Random vs NN
./benchmark_cpu_traditional --all-sizes --iterations 5000
./benchmark_cpu_nn --all-sizes --iterations 1000

# Compare results in results/*.csv
```

### 2. Fair Hardware Comparison (Same Algorithm)
```bash
# NN-MCTS: CPU vs GPU
./benchmark_cpu_nn --all-sizes --iterations 1000
./benchmark_gpu_nn --all-sizes --iterations 1000
```

### 3. Full Benchmark Suite
```bash
# Traditional baselines
./benchmark_cpu_traditional --all-sizes --iterations 5000
cd benchmarks/benchmark_gpu_traditional && python benchmark.py --all-sizes

# NN-based implementations
cd ../.. && ./benchmark_cpu_nn --all-sizes --iterations 1000
./benchmark_gpu_nn --all-sizes --iterations 1000
```

---

## 📊 Output Format

All benchmarks produce CSV files in `results/` with standardized columns:

**Core Metrics**:
- `timestamp`, `hostname`, `processor`, `board_size`, `iterations`
- `total_time_s`, `iterations_per_sec`
- `energy_j`, `energy_per_iter_uj`
- `tree_size`

**Phase Breakdown**:
- `selection_time_s`, `expansion_time_s`, `rollout_time_s`, `backpropagation_time_s`
- `selection_percent`, `expansion_percent`, `rollout_percent`, `backpropagation_percent`

**Metadata**:
- `energy_method` (RAPL, TDP, NVML)
- `implementation` (C++, CUDA, Python)
- `rollout_method` (random_simulation, neural_network)

---

## 🎓 Academic Use

This benchmark suite was developed for the paper:

> **In-Memory Acceleration of Monte Carlo Tree Search**
> ISCA 2026 Submission

**Citation** (preprint):
```bibtex
@inproceedings{mcts2026,
  title={In-Memory Acceleration of Monte Carlo Tree Search},
  author={[Authors]},
  booktitle={International Symposium on Computer Architecture (ISCA)},
  year={2026}
}
```

**Key Results**:
- CPU-NN vs CPU-Random: 1.72× throughput improvement (same hardware, NN algorithm)
- CPU-NN vs ASIC-NN: 81× energy efficiency (same algorithm, custom hardware)
- Total improvement: 95× energy (1.17× algorithm × 81× hardware)

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional hardware platforms (ARM, RISC-V)
- Alternative MCTS variants (PUCT, RAVE)
- Improved energy monitoring (NVML integration)
- Additional board games (Chess, Hex)

---

## 📝 License

MIT License - See LICENSE file for details.

---

## 🔗 Related Work

- **AlphaGo/AlphaZero**: Neural network guided MCTS (Silver et al., Nature 2016)
- **GPU MCTS**: Parallel MCTS on GPUs (Rocki & Suda, 2011)
- **In-Memory Computing**: Analog computation for ML (Ielmini & Wong, Nature Electronics 2018)

---

## 📚 Additional Documentation

- **Quick Start**: [QUICKSTART.md](QUICKSTART.md) - Copy-paste commands
- **GPU Setup**: [docs/QUICKSTART_GPU_SERVER.md](docs/QUICKSTART_GPU_SERVER.md) - HPC cluster configuration
- **Results**: [docs/FINAL_RESULTS_SUMMARY.md](docs/FINAL_RESULTS_SUMMARY.md) - Detailed analysis

---

## ⚠️ Known Issues

- **GPU NN benchmark**: Requires GPU node with CUDA (not compiled on CPU-only systems)
- **RAPL permissions**: May require `sudo` or capability adjustments for energy monitoring
- **19×19 board**: Long execution times (minutes to hours depending on iteration count)

---

## 💡 Tips

1. **Iteration counts**: NN-MCTS typically needs 5-10× fewer iterations than random rollout
2. **Energy accuracy**: Intel RAPL (±1-5%) > AMD RAPL (±10-20%) > TDP estimation (±30-50%)
3. **GPU utilization**: NN benchmarks show better GPU utilization than random rollout
4. **Board complexity**: Tree size and execution time scale exponentially with board size

---

**Questions?** Open an issue or see [QUICKSTART.md](QUICKSTART.md) for common commands.
