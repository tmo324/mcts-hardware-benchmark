# MCTS Hardware Benchmark

Standalone benchmarking tool for comparing Monte Carlo Tree Search (MCTS) performance across different hardware platforms.

Designed to provide fair comparisons against the **ReasonCore MCTS accelerator** by running the same algorithm with identical configurations.

## Features

- ✅ **4-Phase Instrumentation**: Measures Selection, Expansion, Simulation, and Backpropagation separately
- ✅ **Cross-Platform Power Measurement**: Intelligent detection of best available method
  - Linux Intel CPU: RAPL (accurate hardware counters)
  - Linux AMD CPU: TDP-based estimation using `psutil`
  - NVIDIA GPU: `nvidia-smi` power readings
  - Fallback: CPU usage estimation
- ✅ **SST-Matched Configuration**: Uses same iteration counts as ReasonCore SST simulations (Medium play strength)
- ✅ **Self-Contained**: Single Python script, minimal dependencies
- ✅ **CSV Output**: Easy to analyze and compare across machines

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/mcts-hardware-benchmark.git
cd mcts-hardware-benchmark
pip install -r requirements.txt
```

### 2. Run Benchmark

```bash
python benchmark_mcts.py
```

This will:
- Auto-detect your CPU/GPU
- Run benchmarks for all board sizes (2×2, 3×3, 5×5, 9×9, 13×13, 19×19)
- Generate CSV output in `results/` directory

### 3. Push Results

```bash
git add results/
git commit -m "Add benchmark results for [YOUR_MACHINE_NAME]"
git push
```

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

The CSV output contains:

**System Info**:
- `timestamp`, `hostname`, `processor`, `cpu_count`, `power_method`

**Per-Board Results**:
- `board_size`, `num_positions`, `iterations`, `num_trials`
- `total_latency_ms`, `total_latency_std_ms`: Latency (mean ± std)
- `total_power_mw`, `total_power_std_mw`: Power (mean ± std)
- `total_energy_uj`, `total_energy_std_uj`: Energy (mean ± std)

**Per-Phase Breakdown** (Selection, Expansion, Simulation, Backpropagation):
- `{phase}_latency_ms`, `{phase}_latency_std_ms`: Time spent in phase (mean ± std)
- `{phase}_power_mw`: Estimated power for phase
- `{phase}_energy_uj`: Estimated energy for phase
- `{phase}_percent`: Percentage of total time

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

### Run on Multiple Machines

**Machine 1 (AMD Threadripper)**:
```bash
git clone https://github.com/yourusername/mcts-hardware-benchmark.git
cd mcts-hardware-benchmark
pip install -r requirements.txt
python benchmark_mcts.py
git add results/ && git commit -m "Results: AMD Threadripper PRO 5945WX"
git push
```

**Machine 2 (Intel Xeon)**:
```bash
git pull  # Get latest code
python benchmark_mcts.py
git add results/ && git commit -m "Results: Intel Xeon Platinum 8280"
git push
```

**Machine 3 (NVIDIA A100)**:
```bash
git pull
python benchmark_mcts.py
git add results/ && git commit -m "Results: NVIDIA A100 GPU"
git push
```

### Analyze All Results

```bash
git pull  # Get all results
python analyze_results.py  # Compare all machines
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
