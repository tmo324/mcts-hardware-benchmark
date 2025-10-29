#!/usr/bin/env python3
"""
MCTS Hardware Benchmark
=======================

Standalone benchmarking tool for comparing MCTS performance across different hardware.
Designed to match ReasonCore SST simulation configurations.

Usage:
    python benchmark_mcts.py

Output:
    CSV file with latency and power breakdown by MCTS phase

Author: MCTS Hardware Benchmark Project
"""

import os
import sys
import csv
import time
import platform
import socket
from datetime import datetime
from typing import Dict, List

# Import our modules
from mcts_core import run_benchmark
from power_monitor import PowerMonitor

# Medium play strength configuration (matching SST)
BENCHMARK_CONFIG = {
    2: {'iterations': 200, 'exploration': 1.414, 'rollout_depth': 10},
    3: {'iterations': 500, 'exploration': 1.414, 'rollout_depth': 15},
    5: {'iterations': 1000, 'exploration': 1.414, 'rollout_depth': 25},
    9: {'iterations': 5000, 'exploration': 1.414, 'rollout_depth': 40},
    13: {'iterations': 7500, 'exploration': 1.414, 'rollout_depth': 60},
    19: {'iterations': 10000, 'exploration': 1.414, 'rollout_depth': 80},
}


def get_system_info() -> Dict:
    """Get system information"""
    info = {
        'hostname': socket.gethostname(),
        'platform': platform.system(),
        'processor': 'Unknown',
        'cpu_count': 1,
        'memory_gb': 0
    }

    try:
        import psutil
        info['cpu_count'] = psutil.cpu_count(logical=False) or 1
        info['memory_gb'] = round(psutil.virtual_memory().total / (1024**3), 1)

        # Get CPU model
        if platform.system() == 'Linux':
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if 'model name' in line:
                        info['processor'] = line.split(':')[1].strip()
                        break
        else:
            info['processor'] = platform.processor()
    except:
        pass

    return info


def run_single_benchmark(board_size: int, power_monitor: PowerMonitor) -> Dict:
    """
    Run benchmark for a single board size

    Returns:
        dict with timing and power data
    """
    config = BENCHMARK_CONFIG[board_size]

    print(f"\n  Running {board_size}×{board_size} board ({config['iterations']} iterations)...")

    # Start power measurement
    power_start = power_monitor.start_measurement()
    time_start = time.perf_counter()

    # Run MCTS benchmark
    result = run_benchmark(
        board_size=board_size,
        iterations=config['iterations'],
        exploration_constant=config['exploration'],
        rollout_depth=config['rollout_depth']
    )

    # Stop power measurement
    elapsed_time = time.perf_counter() - time_start
    power_result = power_monitor.stop_measurement(power_start, elapsed_time)

    # Combine results
    output = {
        'board_size': f'{board_size}x{board_size}',
        'num_positions': board_size ** 2,
        'iterations': config['iterations'],
        'total_latency_ms': result['total_time_ms'],
        'total_power_mw': power_result.get('power_mw', 0),
        'total_energy_uj': power_result.get('energy_uj', 0),
        'power_method': power_result.get('method', 'unknown'),
        'tree_size': result['tree_size']
    }

    # Add per-phase data
    for phase in ['selection', 'expansion', 'simulation', 'backpropagation']:
        phase_time_ms = result['phase_times_ms'][phase]
        phase_percent = result['phase_percentages'][phase]

        # Estimate phase power (proportional to time)
        phase_power = output['total_power_mw'] * (phase_percent / 100)
        phase_energy = output['total_energy_uj'] * (phase_percent / 100)

        output[f'{phase}_latency_ms'] = phase_time_ms
        output[f'{phase}_power_mw'] = phase_power
        output[f'{phase}_energy_uj'] = phase_energy
        output[f'{phase}_percent'] = phase_percent

    print(f"    Total: {output['total_latency_ms']:.1f} ms, "
          f"{output['total_power_mw']:.1f} mW, "
          f"{output['total_energy_uj']:.1f} µJ")

    return output


def save_results(results: List[Dict], output_file: str, system_info: Dict):
    """Save results to CSV file"""

    # Prepare CSV header
    fieldnames = [
        'timestamp', 'hostname', 'processor', 'cpu_count', 'power_method',
        'board_size', 'num_positions', 'iterations',
        'total_latency_ms', 'total_power_mw', 'total_energy_uj', 'tree_size',
        'selection_latency_ms', 'selection_power_mw', 'selection_energy_uj', 'selection_percent',
        'expansion_latency_ms', 'expansion_power_mw', 'expansion_energy_uj', 'expansion_percent',
        'simulation_latency_ms', 'simulation_power_mw', 'simulation_energy_uj', 'simulation_percent',
        'backpropagation_latency_ms', 'backpropagation_power_mw', 'backpropagation_energy_uj', 'backpropagation_percent'
    ]

    # Add timestamp and system info to each result
    timestamp = datetime.now().isoformat()
    for result in results:
        result['timestamp'] = timestamp
        result['hostname'] = system_info['hostname']
        result['processor'] = system_info['processor']
        result['cpu_count'] = system_info['cpu_count']

    # Write CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Results saved to: {output_file}")


def main():
    """Main benchmark function"""

    print("=" * 70)
    print("MCTS Hardware Benchmark")
    print("Matching ReasonCore SST Medium Play Strength Configuration")
    print("=" * 70)

    # Get system info
    system_info = get_system_info()
    print(f"\nSystem Information:")
    print(f"  Hostname: {system_info['hostname']}")
    print(f"  Processor: {system_info['processor']}")
    print(f"  CPU Cores: {system_info['cpu_count']}")
    print(f"  Memory: {system_info['memory_gb']} GB")

    # Initialize power monitor
    print(f"\nInitializing power monitor...")
    power_monitor = PowerMonitor()

    # Run benchmarks for all board sizes
    print(f"\nRunning benchmarks...")
    print(f"Board sizes: {list(BENCHMARK_CONFIG.keys())}")

    results = []
    for board_size in sorted(BENCHMARK_CONFIG.keys()):
        try:
            result = run_single_benchmark(board_size, power_monitor)
            results.append(result)
        except Exception as e:
            print(f"  ❌ Error on {board_size}×{board_size}: {e}")

    # Save results
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename with hostname and timestamp
    hostname = system_info['hostname'].split('.')[0]  # Remove domain
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'mcts_benchmark_{hostname}_{timestamp}.csv')

    save_results(results, output_file, system_info)

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Board':<10} {'Latency':<15} {'Power':<15} {'Energy':<15}")
    print("-" * 70)
    for result in results:
        print(f"{result['board_size']:<10} "
              f"{result['total_latency_ms']:>10.1f} ms   "
              f"{result['total_power_mw']:>10.1f} mW   "
              f"{result['total_energy_uj']:>10.1f} µJ")

    print("\n" + "=" * 70)
    print("✅ Benchmark complete!")
    print(f"📊 Results: {output_file}")
    print("\nNext steps:")
    print("  1. Commit and push results: git add results/ && git commit -m 'Add benchmark results'")
    print("  2. Run on other machines (CPU/GPU) and push their results")
    print("  3. Analyze all results together")
    print("=" * 70)


if __name__ == "__main__":
    main()
