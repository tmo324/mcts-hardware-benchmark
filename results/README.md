# Results Directory

This directory contains benchmark CSV outputs organized by evaluation method.

## Structure

```
results/
├── README.md           # This file
├── traditional/        # Random rollout MCTS results
│   ├── CPU benchmarks
│   └── GPU benchmarks
└── nn/                 # Neural network MCTS results
    ├── CPU benchmarks
    └── GPU benchmarks
```

## Organization

### `traditional/` - Random Rollout MCTS

Traditional MCTS with random simulation rollouts (baseline approach).

**Naming convention**: `mcts_benchmark_{platform}_{processor}_{timestamp}.csv`

Examples:
- `mcts_benchmark_cpu_amd_ryzen_threadripper_pro_5945wx_12_cores_20251112_182746.csv`
- `mcts_benchmark_gpu_fair_smc4_20251111_192646.csv`

### `nn/` - Neural Network MCTS

Modern MCTS with learned position evaluation (AlphaZero-style).

**Naming convention**: `{platform}_nn_mcts_{hostname}_{processor}.csv`

Examples:
- `cpu_nn_mcts_tr220-dt01.egr.duke.edu_amd_ryzen_threadripper_pro_5945wx_12_cores.csv`
- `gpu_nn_mcts_smc4_h100_80gb.csv` (when available)

## CSV Format

All result files follow a standardized format with the following key columns:

### Core Metrics
- `timestamp`: When the benchmark was run
- `hostname`: System identifier
- `processor`: CPU/GPU model
- `board_size`: Go board dimensions (2, 3, 5, 9, 13, or 19)
- `iterations`: Number of MCTS iterations performed
- `total_time_s`: Total execution time in seconds
- `iterations_per_sec`: Throughput
- `energy_j`: Total energy consumed in joules
- `energy_per_iter_uj`: Energy per iteration in microjoules
- `tree_size`: Number of nodes in final MCTS tree

### Phase Breakdown
- `selection_time_s`, `expansion_time_s`, `rollout_time_s`, `backpropagation_time_s`
- `selection_percent`, `expansion_percent`, `rollout_percent`, `backpropagation_percent`

### Metadata
- `energy_method`: How energy was measured (RAPL, TDP, NVML)
- `implementation`: Programming language/framework (C++, CUDA, Python)
- `rollout_method`: Evaluation approach (random_simulation, neural_network)

## Comparing Results

### Fair Algorithmic Comparison (Same Hardware)
Compare traditional vs NN on the same processor:

```bash
# CPU: Random vs NN
results/traditional/mcts_benchmark_cpu_amd_*.csv
results/nn/cpu_nn_mcts_*amd*.csv
```

### Fair Hardware Comparison (Same Algorithm)
Compare CPU vs GPU with the same evaluation method:

```bash
# Traditional MCTS: CPU vs GPU
results/traditional/mcts_benchmark_cpu_*.csv
results/traditional/mcts_benchmark_gpu_*.csv

# NN-MCTS: CPU vs GPU
results/nn/cpu_nn_mcts_*.csv
results/nn/gpu_nn_mcts_*.csv
```

## Analysis Scripts

Post-processing and visualization scripts are available in `analysis/`:

```bash
# Generate comparison tables
python analysis/utils/generate_tables.py

# Compare with SST simulation results
python analysis/utils/compare_to_sst.py
```

## Typical Result Patterns

### Traditional MCTS
- **Rollout dominates**: 60-95% of execution time in simulation phase
- **Higher iteration counts needed**: 5,000-10,000 iterations for convergence
- **Energy scales with time**: Direct correlation between time and energy

### Neural Network MCTS
- **Rollout still significant**: 40-90% depending on board size
- **Fewer iterations needed**: 1,000-2,000 iterations sufficient
- **Better parallelization**: GPU shows larger speedups for NN evaluation

### Board Size Scaling
- **2×2**: Fast (milliseconds), good for testing
- **3×3, 5×5**: Moderate (seconds), good for development
- **9×9**: Standard benchmark (tens of seconds)
- **13×13, 19×19**: Large (minutes to hours), production workloads

## Energy Measurement Accuracy

| Method | Accuracy | Platforms | Notes |
|--------|----------|-----------|-------|
| **Intel RAPL** | ±1-5% | Intel CPUs | Hardware counters, most accurate |
| **AMD RAPL** | ±10-20% | AMD CPUs | Available but less reliable |
| **TDP Estimation** | ±30-50% | All platforms | Fallback when RAPL unavailable |
| **NVML** | ±5-15% | NVIDIA GPUs | Direct GPU power measurement (pending) |

Check the `energy_method` column to understand accuracy for each result.

## Adding New Results

Benchmarks automatically append to existing CSV files or create new ones:

```bash
# CPU Traditional
./benchmark_cpu_traditional --all-sizes --iterations 5000
# → results/traditional/mcts_benchmark_cpu_*.csv

# CPU NN
./benchmark_cpu_nn --all-sizes --iterations 1000
# → results/nn/cpu_nn_mcts_*.csv

# GPU NN
./benchmark_gpu_nn --all-sizes --iterations 1000
# → results/nn/gpu_nn_mcts_*.csv

# GPU Traditional
cd benchmarks/benchmark_gpu_traditional
python benchmark.py --all-sizes --approach fair
# → results/traditional/mcts_benchmark_gpu_*.csv
```

## Archiving Old Results

To archive results from a previous round of benchmarking:

```bash
# Create dated archive
mkdir -p results/archive/2024-11
mv results/traditional/* results/archive/2024-11/traditional/
mv results/nn/* results/archive/2024-11/nn/

# Or compress
tar -czf results/archive/results_2024-11.tar.gz results/traditional/ results/nn/
```

## File Size Guidelines

- **Per result**: ~5-20 KB (depends on number of board sizes and trials)
- **Full benchmark suite**: ~50-100 KB total
- **Recommended cleanup**: Archive results older than 6 months

## Questions?

See main [README.md](../README.md) or [QUICKSTART.md](../QUICKSTART.md) for benchmark usage.
