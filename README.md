# MCTS Hardware Benchmark

Standalone benchmarking tool for comparing Monte Carlo Tree Search (MCTS) performance across different hardware platforms.

Designed to provide fair comparisons against the **ReasonCore MCTS accelerator** by running the same algorithm with identical configurations.

## Features

- ✅ **CPU & GPU Support**: Python (CPU) and Numba CUDA (GPU) implementations
- ✅ **Dual-Mode GPU Benchmarking**:
  - **Fair Comparison Mode**: Single-tree, sequential playouts (matches CPU algorithm exactly)
  - **Capability Mode**: Multi-tree, parallel playouts (demonstrates maximum GPU throughput)
- ✅ **4-Phase Instrumentation**: Measures Selection, Expansion, Simulation, and Backpropagation separately
- ✅ **Cross-Platform Power Measurement**: Intelligent detection of best available method
  - Linux Intel CPU: RAPL (accurate hardware counters)
  - Linux AMD CPU: TDP-based estimation using `psutil`
  - NVIDIA GPU: NVML power readings
  - Fallback: CPU usage estimation
- ✅ **SST-Matched Configuration**: Uses same iteration counts as ReasonCore SST simulations (Medium play strength)
- ✅ **Publication-Ready Tables**: Auto-generate Markdown/LaTeX tables for research papers
- ✅ **CSV Output**: Easy to analyze and compare across machines

## Quick Start

### 1. Installation

```bash
git clone https://github.com/yourusername/mcts-hardware-benchmark.git
cd mcts-hardware-benchmark
pip install -r requirements.txt
```

### 2. Run Benchmarks

#### CPU Benchmark (Python)

```bash
python main_cpu.py
```

Outputs: `results/mcts_benchmark_<hostname>_<timestamp>.csv`

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

The CSV output contains **one row per trial** (not aggregated):

**System Info**:
- `timestamp`, `hostname`, `processor`, `cpu_count`, `power_method`

**Per-Trial Results**:
- `board_size`, `num_positions`, `iterations`, `trial_num`
- **GPU-specific** (only in GPU CSVs): `n_trees`, `n_playouts`, `variant`
- `total_latency_ms`: Total MCTS search time
- `total_power_mw`: Average power during search
- `total_energy_uj`: Total energy consumed
- `tree_size`: Final MCTS tree size

**Per-Phase Breakdown** (Selection, Expansion, Simulation, Backpropagation):
- `{phase}_latency_ms`: Time spent in phase
- `{phase}_power_mw`: Estimated power for phase
- `{phase}_energy_uj`: Estimated energy for phase
- `{phase}_percent`: Percentage of total time

**Note**: Multiple trials per board size are stored as separate rows. Use `utils/generate_tables.py` to aggregate statistics.

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
├── main_cpu.py                    # Entry point: CPU benchmark
├── main_gpu_fair.py               # Entry point: GPU fair comparison (1T×1P)
├── main_gpu_capability.py         # Entry point: GPU max capability (8T×128P)
│
├── mcts_cpu/                      # CPU implementation
│   ├── __init__.py
│   ├── benchmark.py               # Benchmark orchestration
│   ├── mcts_core.py               # 4-phase MCTS algorithm
│   └── power_monitor.py           # Power monitoring (RAPL/psutil/NVML)
│
├── mcts_gpu/                      # GPU implementation (mcts_numba_cuda)
│   ├── __init__.py
│   ├── benchmark.py               # GPU benchmark wrapper
│   ├── mctsnc.py                  # Core MCTS-NC engine
│   ├── mctsnc_game_mechanics.py   # CUDA device functions (Simplified Go)
│   ├── utils.py                   # Helper functions
│   └── README.md                  # GPU documentation
│
├── utils/                         # Analysis utilities
│   ├── __init__.py
│   ├── generate_tables.py         # Publication table generator
│   └── compare_to_sst.py          # SST comparison script
│
├── results/                       # Benchmark CSV outputs
│   └── *.csv
│
├── requirements.txt               # Python dependencies
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
git clone https://github.com/yourusername/mcts-hardware-benchmark.git
cd mcts-hardware-benchmark
pip install -r requirements.txt

# 2. Run CPU benchmark
python main_cpu.py

# 3. Run GPU benchmarks (on GPU machine)
module load cuda  # HPC systems
python main_gpu_fair.py          # Fair comparison
python main_gpu_capability.py     # Maximum throughput

# 4. Generate publication tables
python utils/generate_tables.py --auto-find --board-size 9x9

# 5. Check tables
cat tables/phase_breakdown.md
cat tables/scalability_analysis.md

# 6. Use LaTeX tables in paper
cat tables/phase_breakdown.tex
cat tables/scalability_analysis.tex
```

### Multi-Machine Comparison

**Machine 1 (Intel Xeon CPU)**:
```bash
git clone https://github.com/yourusername/mcts-hardware-benchmark.git
cd mcts-hardware-benchmark
pip install -r requirements.txt
python main_cpu.py
git add results/ && git commit -m "Results: Intel Xeon Platinum 8280"
git push
```

**Machine 2 (NVIDIA A100 GPU)**:
```bash
git pull  # Get latest code
module load cuda
python main_gpu_fair.py           # Fair comparison
python main_gpu_capability.py      # Max GPU throughput
git add results/ && git commit -m "Results: NVIDIA A100 GPU (fair + capability)"
git push
```

**Local Analysis**:
```bash
git pull  # Get all results
python utils/generate_tables.py --auto-find
# Now you have comparison tables in tables/
```

## Comparing to ReasonCore

To compare your results against ReasonCore SST simulations:

1. Run this benchmark on your CPU/GPU
2. Compare `total_energy_uj` values to ReasonCore's simulated energy
3. Calculate speedup: `ReasonCore_latency / Your_latency`
4. Calculate energy efficiency: `Your_energy / ReasonCore_energy`

Example for 5×5 board:
- **ReasonCore (SST)**: 1.4 µJ, 100 ms (simulated)
- **Your CPU**: 3,300 µJ, 150 ms (measured)
- **Energy ratio**: 2,357× more energy
- **Speedup**: 0.67× slower

## Troubleshooting

### Permission Denied (RAPL on Linux)

```bash
sudo chmod -R a+r /sys/class/powercap/intel-rapl/
```

### GPU Not Detected

Check `nvidia-smi` is installed:
```bash
nvidia-smi
```

### Low Accuracy (psutil estimation)

This is expected for AMD CPUs. For better accuracy:
- Use external power meter
- Run on Intel CPU with RAPL
- Manually calibrate TDP in `power_monitor.py`

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
