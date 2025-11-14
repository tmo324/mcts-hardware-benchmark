# MCTS Benchmark Quick Start Guide

This directory contains **4 primary benchmarks** for Monte Carlo Tree Search (MCTS) evaluation:

1. **CPU Traditional** - Random rollout on CPU (C++)
2. **CPU Neural Network** - NN evaluation on CPU (C++)
3. **GPU Traditional** - Random rollout on GPU (Python/Numba)
4. **GPU Neural Network** - NN evaluation on GPU (CUDA C++)

---

## 🚀 Quick Commands

### Build All Benchmarks
```bash
make all
```

### CPU Traditional (Random Rollout)
```bash
# Build
make cpu-trad

# Run single board size
./benchmark_cpu_traditional --board-size 9 --iterations 1000

# Run all board sizes (2,3,5,9,13,19)
./benchmark_cpu_traditional --all-sizes --iterations 5000
```

### CPU Neural Network
```bash
# Build
make cpu-nn

# Run single board size (specify iterations manually)
./benchmark_cpu_nn --board-size 9 --iterations 5000

# Run all board sizes (uses standard iteration counts automatically)
# 2×2: 200, 3×3: 500, 5×5: 1000, 9×9: 5000, 13×13: 7500, 19×19: 10000
./benchmark_cpu_nn --all-sizes
```

### GPU Neural Network (CUDA)
```bash
# Build (requires CUDA/cuBLAS)
make gpu-nn

# Run single board size (specify iterations manually)
./benchmark_gpu_nn --board-size 9 --iterations 5000

# Run all board sizes (uses standard iteration counts automatically)
# 2×2: 200, 3×3: 500, 5×5: 1000, 9×9: 5000, 13×13: 7500, 19×19: 10000
./benchmark_gpu_nn --all-sizes
```

### GPU Traditional (Python/Numba)
```bash
cd benchmarks/benchmark_gpu_traditional

# Run single board size
python benchmark.py --board-size 9 --iterations 1000 --approach fair

# See all options
python benchmark.py --help
```

---

## 📊 Results

All benchmarks write CSV results to `results/` directory with standardized format:
- Timestamp, hostname, processor info
- Energy measurements (RAPL for CPU, TDP-based for GPU)
- Per-phase timing breakdown (Selection, Expansion, Rollout, Backpropagation)
- Tree statistics

---

## 🧪 Quick Test

Test all benchmarks with small board (2×2, 100 iterations):

```bash
# CPU Traditional
./benchmark_cpu_traditional --board-size 2 --iterations 100

# CPU NN
./benchmark_cpu_nn --board-size 2 --iterations 100

# GPU NN (if CUDA available)
./benchmark_gpu_nn --board-size 2 --iterations 100

# GPU Traditional
cd benchmarks/benchmark_gpu_traditional && python benchmark.py --board-size 2 --iterations 100
```

---

## 📦 Directory Structure

```
mcts-hardware-benchmark/
├── benchmarks/                         # All 4 primary benchmarks
│   ├── benchmark_cpu_traditional.cpp   # CPU random rollout
│   ├── benchmark_cpu_nn.cpp            # CPU NN evaluation
│   ├── benchmark_gpu_nn.cu             # GPU NN evaluation (CUDA)
│   └── benchmark_gpu_traditional/      # GPU random rollout (Python)
│
├── include/                            # Headers
│   └── nn_inference.h                  # NN inference library
│
├── weights/                            # Pre-trained NN weights
│   ├── 2x2/, 3x3/, 5x5/, 9x9/, 13x13/, 19x19/
│   └── convert_weights_to_binary.py
│
├── results/                            # Benchmark CSV outputs
├── analysis/                           # Post-processing scripts
├── docs/                               # Additional documentation
└── Makefile                            # Build system
```

---

## 🔧 Requirements

### CPU Benchmarks
- C++17 compiler (g++ 7.0+)
- No external dependencies (plain C++ for matrix ops)

### GPU NN Benchmark
- NVIDIA GPU with CUDA 11.0+
- cuBLAS library
- `nvcc` compiler

### GPU Traditional Benchmark
- Python 3.8+
- Numba, NumPy
- See `benchmarks/benchmark_gpu_traditional/requirements.txt`

---

## 📈 Full Benchmark Protocol (5 Trials per Board Size)

All benchmarks automatically run **5 trials** per board size with board-specific iteration counts:
- **2×2**: 200 iterations
- **3×3**: 500 iterations
- **5×5**: 1,000 iterations
- **9×9**: 5,000 iterations
- **13×13**: 7,500 iterations
- **19×19**: 10,000 iterations

### AMD Ryzen Threadripper (CPU)
```bash
# Traditional MCTS (Random Rollout)
make cpu-trad
./benchmark_cpu_traditional --all-sizes
# Output: results/traditional/mcts_benchmark_cpu_amd_*_TIMESTAMP.csv

# NN-MCTS
make cpu-nn
./benchmark_cpu_nn --all-sizes
# Output: results/nn/cpu_nn_mcts_*_amd_*_TIMESTAMP.csv
```

### Intel Xeon (CPU)
```bash
# Traditional MCTS (Random Rollout)
make cpu-trad
./benchmark_cpu_traditional --all-sizes
# Output: results/traditional/mcts_benchmark_cpu_intel_*_TIMESTAMP.csv

# NN-MCTS
make cpu-nn
./benchmark_cpu_nn --all-sizes
# Output: results/nn/cpu_nn_mcts_*_intel_*_TIMESTAMP.csv
```

### NVIDIA H100 - Fair Comparison Mode (GPU)
Single-tree, sequential playouts (matches CPU algorithm):
```bash
# Traditional MCTS (Random Rollout)
cd benchmarks/benchmark_gpu_traditional
python benchmark.py --all-sizes --approach fair
# Output: results/traditional/mcts_benchmark_gpu_fair_*_TIMESTAMP.csv

# NN-MCTS
cd ../..
make gpu-nn
./benchmark_gpu_nn --all-sizes
# Output: results/nn/gpu_nn_mcts_*_TIMESTAMP.csv
```

### NVIDIA H100 - Maximum Capability Mode (GPU)
Multi-tree (8 trees), parallel playouts (128), demonstrates max throughput:
```bash
# Traditional MCTS only (8× more work than fair mode)
cd benchmarks/benchmark_gpu_traditional
python benchmark.py --all-sizes --approach capability
# Output: results/traditional/mcts_benchmark_gpu_capability_*_TIMESTAMP.csv

# Note: NN-MCTS uses same implementation for both modes (already optimized)
```

### Complete Comparison Workflow
```bash
# 1. Run on CPU node (AMD or Intel)
make cpu-trad cpu-nn
./benchmark_cpu_traditional --all-sizes
./benchmark_cpu_nn --all-sizes

# 2. Run on GPU node
cd benchmarks/benchmark_gpu_traditional
python benchmark.py --all-sizes --approach fair
python benchmark.py --all-sizes --approach capability
cd ../..
make gpu-nn
./benchmark_gpu_nn --all-sizes

# 3. Collect results
ls -lh results/traditional/*.csv
ls -lh results/nn/*.csv
```

---

## 💡 Tips

1. **Iteration counts**: NN-based MCTS typically needs fewer iterations than random rollout for same decision quality
2. **Energy monitoring**: CPU benchmarks use Intel RAPL (root access may be needed). GPU uses TDP-based estimation
3. **Board sizes**: Smaller boards (2×2, 3×3) complete quickly; 19×19 takes significantly longer
4. **Results location**: Check `results/*.csv` for detailed benchmark data

---

## 📚 More Information

- Full documentation: `README.md`
- GPU server setup: `docs/QUICKSTART_GPU_SERVER.md`
- Result analysis: `docs/FINAL_RESULTS_SUMMARY.md`
- Paper reference: See main README
