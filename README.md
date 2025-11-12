# MCTS Hardware Benchmark

Standalone benchmarking tool for comparing Monte Carlo Tree Search (MCTS) performance across different hardware platforms.

Designed to provide fair comparisons against the **ReasonCore MCTS accelerator** by running the same algorithm with identical configurations.

## Features

- ✅ **CPU & GPU Support**: Optimized C++ (CPU) and Numba CUDA (GPU) implementations
- ✅ **High-Performance CPU Baseline**: C++17 with -O3 optimization for fair hardware comparison
- ✅ **Dual-Mode GPU Benchmarking**:
  - **Fair Comparison Mode**: Single-tree, sequential playouts (matches CPU algorithm exactly)
  - **Capability Mode**: Multi-tree, parallel playouts (demonstrates maximum GPU throughput)
- ✅ **Cross-Platform Power Measurement**: Intelligent detection of best available method
  - Linux Intel CPU: RAPL (accurate hardware counters, ±1-5%)
  - Linux AMD CPU: TDP-based estimation
  - NVIDIA GPU: NVML power readings
- ✅ **Publication-Ready Output**: CSV format with detailed per-trial statistics
- ✅ **Board Size Scaling**: 2×2 to 19×19 Go boards with configurable iterations

## Quick Start

### 1. Installation

```bash
git clone https://github.com/tmo324/mcts-hardware-benchmark.git
cd mcts-hardware-benchmark
```

**For CPU benchmarks**:
```bash
# Compile C++ benchmark (requires g++ with C++17 support)
make
```

**For GPU benchmarks**:
```bash
# Install Python dependencies
pip install -r requirements.txt
```

### 2. Run Benchmarks

#### CPU Benchmark (C++)

**Quick test (single board size)**:
```bash
./mcts_cpu_benchmark --board-size 9 --iterations 5000
```

**Complete benchmark (all board sizes)**:
```bash
./mcts_cpu_benchmark --all-sizes --iterations 5000 --trials 5
# Or use convenience script:
./run_all_benchmarks.sh
```

**Output**: `results/cpu_cpp_benchmark_YYYYMMDD_HHMMSS.csv`

**Options**:
- `--board-size N`: Run single board size (2-19)
- `--all-sizes`: Run all board sizes (2,3,5,9,13,19)
- `--iterations N`: MCTS iterations per trial (default: 1000)
- `--trials N`: Number of trials per board size (default: 5)
- `--tdp W`: CPU TDP in watts (default: 55.3W for Xeon 8462Y+)
- `--help`: Show help message

#### GPU Benchmark - Fair Comparison Mode

For fair hardware comparison (matches CPU algorithm):

```bash
# On HPC systems, load CUDA module first
module load cuda  # Or: export PATH=/usr/local/cuda/bin:$PATH

python main_gpu_fair.py
```

**Configuration**: n_trees=1, n_playouts=1, variant='ocp_thrifty'
**Outputs**: `results/mcts_benchmark_gpu_fair_<hostname>_<timestamp>.csv`

This mode performs the **same computational work** as CPU, allowing true hardware comparison.

#### GPU Benchmark - Maximum Capability Mode

To demonstrate maximum GPU throughput:

```bash
python main_gpu_capability.py
```

**Configuration**: n_trees=8, n_playouts=128, variant='acp_prodigal'
**Outputs**: `results/mcts_benchmark_gpu_capability_<hostname>_<timestamp>.csv`

This mode performs **8× more work** than CPU/GPU-fair mode (not a fair comparison).

### 3. Generate Publication Tables

After running benchmarks, generate publication-ready tables:

```bash
python utils/generate_tables.py --auto-find --board-size 9x9 --output-dir tables
```

This creates:
- `tables/phase_breakdown.[md|tex|csv]` - Phase-by-phase comparison for one board size
- `tables/scalability_analysis.[md|tex|csv]` - Performance across all board sizes

See [Table Generation](#table-generation) for more options.

## Board Size Configurations

The benchmark uses **Medium play strength** settings matching the ReasonCore SST simulations.

**Multiple trials** are run for each board size to compute statistical confidence:

| Board | Iterations | Trials | Exploration C | Rollout Depth | Expected Latency (ReasonCore) |
|-------|------------|--------|---------------|---------------|-------------------------------|
| 2×2   | 200        | 5      | 1.414         | 10            | 20 ms                         |
| 3×3   | 500        | 5      | 1.414         | 15            | 50 ms                         |
| 5×5   | 1,000      | 5      | 1.414         | 25            | 100 ms                        |
| 9×9   | 5,000      | 3      | 1.414         | 40            | 625 ms                        |
| 13×13 | 7,500      | 3      | 1.414         | 60            | 1,350 ms                      |
| 19×19 | 10,000     | 3      | 1.414         | 80            | 1,680 ms                      |

Results include **mean ± std** for all metrics (latency, power, energy).

## Output Format

### CPU Benchmark (C++) CSV Output

The CSV output contains **one row per trial** with the following columns:

- `board_size`: Board dimensions (2-19)
- `trial`: Trial number (1-N)
- `iterations`: MCTS iterations performed
- `elapsed_ms`: Total execution time (milliseconds)
- `throughput_iter_s`: Iterations per second
- `latency_ms_iter`: Milliseconds per iteration
- `energy_j`: Total energy consumed (joules)
- `energy_per_iter_uj`: Energy per iteration (microjoules)
- `power_w`: Average power (watts)
- `method`: Energy measurement method (RAPL or TDP)

Example row:
```
board_size,trial,iterations,elapsed_ms,throughput_iter_s,latency_ms_iter,energy_j,energy_per_iter_uj,power_w,method
9,1,5000,14.85,336590.45,0.00297,1.644,328.86,110.69,RAPL
```

### GPU Benchmark (Python) CSV Output

The GPU CSV output includes additional GPU-specific metrics. See Python implementation documentation for details.

## Table Generation

The `utils/generate_tables.py` script generates publication-ready tables from benchmark CSV files.

### Basic Usage

```bash
# Auto-find latest benchmark files
python utils/generate_tables.py --auto-find

# Manually specify files
python utils/generate_tables.py \
  --cpu-file results/mcts_benchmark_hostname_20251111_120000.csv \
  --gpu-fair-file results/mcts_benchmark_gpu_fair_hostname_20251111_120100.csv \
  --gpu-cap-file results/mcts_benchmark_gpu_capability_hostname_20251111_120200.csv
```

### Options

- `--auto-find`: Automatically find latest CPU, GPU-fair, and GPU-capability CSV files
- `--cpu-file`: Path to CPU benchmark CSV
- `--gpu-fair-file`: Path to GPU fair mode CSV
- `--gpu-cap-file`: Path to GPU capability mode CSV (optional)
- `--board-size`: Board size for phase breakdown table (default: 9x9)
- `--output-dir`: Output directory for tables (default: tables/)

### Generated Tables

#### 1. Phase Breakdown Table

Shows latency and energy for each MCTS phase (Selection, Expansion, Rollout, Backpropagation) for a single board size.

Example (`tables/phase_breakdown.md`):

```markdown
| Phase            | Latency (ms)      |                   | Energy (mJ)      |                  |
|                  | **CPU**           | **GPU**           | **CPU**          | **GPU**          |
|------------------|-------------------|-------------------|------------------|------------------|
| Selection        | 125.3 ± 2.1       | 98.7 ± 1.5        | 450.2 ± 8.3      | 320.1 ± 5.2      |
| Expansion        | 48.2 ± 1.2        | 35.6 ± 0.9        | 173.5 ± 4.1      | 115.8 ± 2.7      |
| Rollout          | 312.5 ± 5.3       | 245.2 ± 3.8       | 1124.8 ± 18.2    | 798.5 ± 12.1     |
| Backpropagation  | 22.1 ± 0.8        | 18.3 ± 0.5        | 79.6 ± 2.8       | 59.7 ± 1.6       |
|------------------|-------------------|-------------------|------------------|------------------|
| Total            | 508.1 ± 7.2       | 397.8 ± 5.1       | 1828.1 ± 25.3    | 1294.1 ± 16.8    |
```

#### 2. Scalability Analysis Table

Shows performance metrics across all board sizes:
- **Latency (ms)**: Total MCTS search time
- **Energy (mJ)**: Total energy consumed
- **Energy per Move (mJ/iteration)**: Energy efficiency metric
- **Throughput per Watt (moves/s/W)**: Performance density metric

All values shown as `mean ± std` across trials.

### Output Formats

- **Markdown** (`.md`): For README files, GitHub, documentation
- **LaTeX** (`.tex`): For research papers, publications
- **CSV** (`.csv`): For further analysis, plotting

## Project Structure

```
mcts-hardware-benchmark/
├── mcts_cpu_benchmark.cpp         # CPU benchmark (C++, single-file implementation)
├── Makefile                       # Build configuration
├── run_all_benchmarks.sh          # Convenience script for all board sizes
│
├── main_gpu_fair.py               # GPU fair comparison mode (1T×1P)
├── main_gpu_capability.py         # GPU max capability mode (8T×128P)
│
├── archive_python/                # Archived Python CPU implementation
│   ├── main_cpu.py
│   ├── main_cpu_psutil.py
│   └── mcts_cpu/
│
├── mcts_gpu/                      # GPU implementation (mcts_numba_cuda)
│   ├── __init__.py
│   ├── benchmark.py               # GPU benchmark wrapper
│   ├── mctsnc.py                  # Core MCTS-NC engine
│   ├── mctsnc_game_mechanics.py   # CUDA device functions
│   ├── utils.py                   # Helper functions
│   └── README.md                  # GPU documentation
│
├── utils/                         # Analysis utilities
│   ├── __init__.py
│   ├── generate_tables.py         # Table generator (for GPU results)
│   └── compare_to_sst.py          # SST comparison script
│
├── results/                       # Benchmark CSV outputs
│   ├── cpu_cpp_benchmark_*.csv    # C++ CPU results
│   └── mcts_benchmark_gpu_*.csv   # GPU results
│
├── requirements.txt               # Python dependencies (GPU only)
└── README.md                      # This file
```

## Power Measurement Methods

The benchmark automatically selects the best available method:

### RAPL (Intel CPUs on Linux)
- **Accuracy**: ±2-5% (hardware counters)
- **Requirements**: Linux with Intel CPU
- **Access**: Requires read permissions to `/sys/class/powercap/intel-rapl/`

### nvidia-smi (NVIDIA GPUs)
- **Accuracy**: ±5-10% (GPU firmware reporting)
- **Requirements**: NVIDIA GPU with `nvidia-smi` installed
- **Notes**: Measures GPU power only (excludes CPU)

### psutil Estimation (AMD CPUs, Fallback)
- **Accuracy**: ±20-30% (estimation based on CPU usage and TDP)
- **Requirements**: `psutil` library
- **Method**: `Power = Idle + (TDP - Idle) × CPU%`

## Example Usage

### Complete Workflow for Research Paper

```bash
# 1. Clone repository
git clone https://github.com/tmo324/mcts-hardware-benchmark.git
cd mcts-hardware-benchmark

# 2. Run CPU benchmark (C++)
make                              # Compile
./mcts_cpu_benchmark --all-sizes  # Run all board sizes

# 3. Run GPU benchmarks (on GPU machine)
pip install -r requirements.txt   # Install Python dependencies
module load cuda                  # HPC systems
python main_gpu_fair.py           # Fair comparison
python main_gpu_capability.py     # Maximum throughput

# 4. Analyze results
cat results/cpu_cpp_benchmark_*.csv      # View CPU results
cat results/mcts_benchmark_gpu_*.csv     # View GPU results

# 5. Compare to hardware accelerator
python utils/compare_to_sst.py --cpu-file results/cpu_cpp_benchmark_*.csv
```

### Multi-Machine Comparison

**Machine 1 (Intel Xeon CPU)**:
```bash
git clone https://github.com/tmo324/mcts-hardware-benchmark.git
cd mcts-hardware-benchmark
make                                                # Compile C++ benchmark
./mcts_cpu_benchmark --all-sizes --trials 5         # Run all board sizes
git add results/ && git commit -m "Results: Intel Xeon Platinum 8462Y+"
git push
```

**Machine 2 (NVIDIA H100 GPU)**:
```bash
git pull                          # Get latest code
pip install -r requirements.txt   # Install Python dependencies
module load cuda                  # HPC systems
python main_gpu_fair.py           # Fair comparison mode
python main_gpu_capability.py     # Maximum capability mode
git add results/ && git commit -m "Results: NVIDIA H100 GPU"
git push
```

**Local Analysis**:
```bash
git pull  # Get all results from both machines
# Compare results manually or write custom analysis script
cat results/cpu_cpp_benchmark_*.csv
cat results/mcts_benchmark_gpu_*.csv
```

## C++ Implementation Details

The CPU benchmark is implemented in optimized C++17 with the following characteristics:

**Compilation**:
- Compiler: g++ with `-std=c++17 -O3 -march=native -ffast-math`
- Single-threaded execution (fair comparison)
- No external dependencies (standard library only)

**Algorithm**:
- Four-phase MCTS: Selection (UCB1), Expansion, Rollout, Backpropagation
- Hash-based tree storage using `std::unordered_map`
- Simplified Go rules (no capture detection, winner by stone count)

**Energy Measurement**:
- **Intel RAPL** (preferred): Reads `/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj`
  - Accuracy: ±1-5% (hardware counters)
  - Measures entire CPU package energy
- **TDP Estimation** (fallback): `Energy = Power × Time` where Power = specified TDP
  - Accuracy: ±20-30% (approximation)
  - Use when RAPL is unavailable (AMD CPUs, no permissions)

**Performance**:
- Expected throughput: 15-20× faster than Python 3.11
- Example (9×9 board): ~330,000 iterations/second on Xeon 8462Y+

## Comparing to Hardware Accelerators

Use the C++ baseline for fair comparison against custom hardware:

**Example comparison (9×9 board)**:
- **CPU (Xeon 8462Y+)**: 336k iter/s, 328 µJ/iter, 110W
- **Custom Accelerator**: Compare your energy/latency measurements
- **Energy efficiency gain**: `CPU_energy / Accelerator_energy`
- **Throughput ratio**: `Accelerator_throughput / CPU_throughput`

## Troubleshooting

### C++ Compilation Errors

**Error: g++ not found**
```bash
# Ubuntu/Debian
sudo apt install build-essential

# CentOS/RHEL
sudo yum install gcc-c++
```

**Error: C++17 not supported**
Requires g++ 7.0+ or clang++ 5.0+. Update your compiler:
```bash
gcc --version  # Check version
# If < 7.0, update compiler or use module system on HPC
```

### Permission Denied (RAPL on Linux)

```bash
sudo chmod -R a+r /sys/class/powercap/intel-rapl/
```

**Alternative**: Use TDP estimation with `--tdp` flag:
```bash
./mcts_cpu_benchmark --all-sizes --tdp 55.3
```

### GPU Not Detected

Check `nvidia-smi` is installed:
```bash
nvidia-smi
```

### Energy Measurements Show Zero

This happens when:
1. Benchmark runs too fast (< 1ms)
2. RAPL counter resolution insufficient

**Solution**: Run more iterations:
```bash
./mcts_cpu_benchmark --board-size 9 --iterations 10000
```

## Contributing

Contributions welcome! Areas for improvement:
- Support for Apple Silicon (M1/M2) power measurement
- Integration with external power meters
- Advanced MCTS features (neural network rollouts)
- Additional board sizes

## License

MIT License - Free to use and modify

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{mcts-hardware-benchmark,
  title={MCTS Hardware Benchmark: Cross-Platform Monte Carlo Tree Search Performance},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/mcts-hardware-benchmark}
}
```

## Contact

Questions? Issues? Open a GitHub issue or contact [your-email@example.com]
