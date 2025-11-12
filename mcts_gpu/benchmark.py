#!/usr/bin/env python3
"""
GPU MCTS Benchmark using mcts_numba_cuda
=========================================

Wrapper around MCTSNC for hardware benchmarking with:
- Simplified Go game implementation
- Power monitoring (NVML for GPU)
- CSV output matching CPU benchmark format
- Multi-trial statistics

Author: MCTS Hardware Benchmark Project
"""

import os
import sys
import csv
import time
import numpy as np
import socket
import platform
from datetime import datetime
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts_gpu.mctsnc import MCTSNC
from mcts_cpu.power_monitor import PowerMonitor

# Medium play strength configuration (matching CPU benchmarks)
BENCHMARK_CONFIG = {
    2: {'iterations': 200, 'exploration': 1.414, 'rollout_depth': 10, 'trials': 5},
    3: {'iterations': 500, 'exploration': 1.414, 'rollout_depth': 15, 'trials': 5},
    5: {'iterations': 1000, 'exploration': 1.414, 'rollout_depth': 25, 'trials': 5},
    9: {'iterations': 5000, 'exploration': 1.414, 'rollout_depth': 40, 'trials': 5},
    13: {'iterations': 7500, 'exploration': 1.414, 'rollout_depth': 60, 'trials': 5},
    19: {'iterations': 10000, 'exploration': 1.414, 'rollout_depth': 80, 'trials': 5},
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

        # Try to get GPU info
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info['processor'] = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            pynvml.nvmlShutdown()
        except:
            info['processor'] = 'NVIDIA GPU (name unavailable)'
    except:
        pass

    return info


def run_single_trial_gpu(board_size: int, config: Dict, ai: MCTSNC, power_monitor: PowerMonitor) -> Dict:
    """Run a single GPU MCTS trial"""

    # Initialize board state
    board = np.zeros((board_size, board_size), dtype=np.int8)
    extra_info = np.zeros(16, dtype=np.int8)  # extra_info[0] = consecutive passes
    turn = -1  # Start with black

    # Start power measurement
    power_start = power_monitor.start_measurement()
    time_start = time.perf_counter()

    # Run MCTS search
    try:
        best_action = ai.run(board, extra_info, turn)
        # Get performance and actions info separately
        performance_info = ai._make_performance_info()
        actions_info = ai._make_actions_info_thrifty() if 'thrifty' in ai.variant else ai._make_actions_info_prodigal()
    except Exception as e:
        print(f"    Error during GPU MCTS: {e}")
        raise

    # Stop power measurement
    elapsed_time = time.perf_counter() - time_start
    power_result = power_monitor.stop_measurement(power_start, elapsed_time)

    return {
        'performance': performance_info,
        'power': power_result,
        'time': elapsed_time
    }


def run_single_benchmark_gpu(board_size: int, power_monitor: PowerMonitor,
                            n_trees: int = 8, n_playouts: int = 128,
                            variant: str = 'acp_prodigal') -> List[Dict]:
    """
    Run GPU benchmark for a single board size with multiple trials

    Args:
        board_size: Size of the Go board (e.g., 9 for 9x9)
        power_monitor: PowerMonitor instance for measuring energy
        n_trees: Number of parallel MCTS trees (1 for fair CPU comparison, 8 for capability)
        n_playouts: Number of parallel playouts per node (1 for fair, 128 for capability)
        variant: MCTSNC variant ('ocp_thrifty' for fair, 'acp_prodigal' for capability)

    Returns:
        list of dicts, one per trial with timing and power data
    """
    config = BENCHMARK_CONFIG[board_size]
    num_trials = config.get('trials', 3)

    print(f"\n  Running {board_size}×{board_size} board ({config['iterations']} iterations, {num_trials} trials)...")
    print(f"    GPU Config: n_trees={n_trees}, n_playouts={n_playouts}, variant={variant}")

    # Initialize MCTSNC for this board size
    # Note: Pass is action m*n, so max_actions = m*n + 1
    try:
        ai = MCTSNC(
            state_board_shape=(board_size, board_size),
            state_extra_info_memory=16,  # Just need 1 byte for pass counter
            state_max_actions=board_size * board_size + 1,  # positions + pass
            search_steps_limit=config['iterations'],
            search_time_limit=float('inf'),  # Use step limit, not time
            n_trees=n_trees,  # Parameterized: 1 for fair comparison, 8 for capability
            n_playouts=n_playouts,  # Parameterized: 1 for fair comparison, 128 for capability
            variant=variant,  # Parameterized: ocp_thrifty for fair, acp_prodigal for capability
            device_memory=2.0,  # 2 GB GPU memory
            ucb_c=config['exploration'],
            seed=42,
            verbose_debug=False,
            verbose_info=False
        )

        # Initialize GPU arrays
        ai.init_device_side_arrays()

    except Exception as e:
        print(f"    Error initializing MCTSNC: {e}")
        raise

    # Run multiple trials
    trial_outputs = []
    trials = []

    for trial_num in range(num_trials):
        print(f"    Trial {trial_num + 1}/{num_trials}...", end=' ', flush=True)

        try:
            trial_result = run_single_trial_gpu(board_size, config, ai, power_monitor)
            trials.append(trial_result)

            total_time_ms = trial_result['performance']['times_[ms]']['total']
            print(f"{total_time_ms:.1f} ms")

            # Extract phase timings from performance_info
            perf = trial_result['performance']
            times_ms = perf['times_[ms]']

            # Calculate phase percentages
            total_time = times_ms['total']
            selection_pct = (times_ms['mean_select'] * perf['steps'] / total_time) * 100 if total_time > 0 else 0
            expansion_pct = (times_ms['mean_expand'] * perf['steps'] / total_time) * 100 if total_time > 0 else 0
            playout_pct = (times_ms['mean_playout'] * perf['steps'] / total_time) * 100 if total_time > 0 else 0
            backup_pct = (times_ms['mean_backup'] * perf['steps'] / total_time) * 100 if total_time > 0 else 0

            # Create output for this individual trial
            output = {
                'board_size': f'{board_size}x{board_size}',
                'num_positions': board_size ** 2,
                'iterations': config['iterations'],
                'trial_num': trial_num + 1,
                'n_trees': n_trees,
                'n_playouts': n_playouts,
                'variant': variant,
                'total_latency_ms': total_time_ms,
                'total_power_mw': trial_result['power'].get('power_mw', 0),
                'total_energy_uj': trial_result['power'].get('energy_uj', 0),
                'power_method': trial_result['power'].get('method', 'unknown'),
                'tree_size': perf['trees']['mean_size']
            }

            # Add per-phase data
            selection_time = times_ms['mean_select'] * perf['steps']
            expansion_time = times_ms['mean_expand'] * perf['steps']
            playout_time = times_ms['mean_playout'] * perf['steps']
            backup_time = times_ms['mean_backup'] * perf['steps']

            phases = {
                'selection': (selection_time, selection_pct),
                'expansion': (expansion_time, expansion_pct),
                'simulation': (playout_time, playout_pct),  # Note: "playout" maps to "simulation"
                'backpropagation': (backup_time, backup_pct)  # Note: "backup" maps to "backpropagation"
            }

            for phase, (phase_time_ms, phase_percent) in phases.items():
                # Estimate phase power (proportional to time)
                phase_power = output['total_power_mw'] * (phase_percent / 100)
                phase_energy = output['total_energy_uj'] * (phase_percent / 100)

                output[f'{phase}_latency_ms'] = phase_time_ms
                output[f'{phase}_power_mw'] = phase_power
                output[f'{phase}_energy_uj'] = phase_energy
                output[f'{phase}_percent'] = phase_percent

            trial_outputs.append(output)

        except Exception as e:
            print(f"Error: {e}")
            raise

    # Calculate and print statistics (for display only)
    import statistics

    total_latencies = [t['performance']['times_[ms]']['total'] for t in trials]
    total_powers = [t['power'].get('power_mw', 0) for t in trials]
    total_energies = [t['power'].get('energy_uj', 0) for t in trials]

    avg_latency = statistics.mean(total_latencies)
    std_latency = statistics.stdev(total_latencies) if len(total_latencies) > 1 else 0
    avg_power = statistics.mean(total_powers)
    std_power = statistics.stdev(total_powers) if len(total_powers) > 1 else 0
    avg_energy = statistics.mean(total_energies)
    std_energy = statistics.stdev(total_energies) if len(total_energies) > 1 else 0

    print(f"    Average: {avg_latency:.1f} ± {std_latency:.1f} ms, "
          f"{avg_power:.1f} ± {std_power:.1f} mW, "
          f"{avg_energy:.1f} ± {std_energy:.1f} µJ")

    return trial_outputs


def save_results(results: List[Dict], output_file: str, system_info: Dict):
    """Save results to CSV file (same format as CPU benchmark with GPU-specific columns)"""

    # Prepare CSV header (added n_trees, n_playouts, variant)
    fieldnames = [
        'timestamp', 'hostname', 'processor', 'cpu_count', 'power_method',
        'board_size', 'num_positions', 'iterations', 'trial_num',
        'n_trees', 'n_playouts', 'variant',
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


def main(n_trees: int = 8, n_playouts: int = 128, variant: str = 'acp_prodigal', mode_name: str = 'gpu'):
    """
    Main GPU benchmark function

    Args:
        n_trees: Number of parallel MCTS trees (1 for fair comparison, 8+ for capability)
        n_playouts: Number of parallel playouts per node (1 for fair, 128+ for capability)
        variant: MCTSNC variant ('ocp_thrifty' for fair, 'acp_prodigal' for capability)
        mode_name: Mode name for filename (e.g., 'fair', 'capability', 'gpu')
    """

    print("=" * 70)
    print("MCTS GPU Benchmark (mcts_numba_cuda + Simplified Go)")
    print("Matching ReasonCore SST Medium Play Strength Configuration")
    print(f"Mode: {mode_name} (n_trees={n_trees}, n_playouts={n_playouts}, variant={variant})")
    print("=" * 70)

    # Get system info
    system_info = get_system_info()
    print(f"\nSystem Information:")
    print(f"  Hostname: {system_info['hostname']}")
    print(f"  GPU: {system_info['processor']}")
    print(f"  CPU Cores: {system_info['cpu_count']}")
    print(f"  Memory: {system_info['memory_gb']} GB")

    # Initialize power monitor (will use NVML for GPU)
    print(f"\nInitializing power monitor...")
    power_monitor = PowerMonitor()

    # Run benchmarks for all board sizes
    print(f"\nRunning GPU benchmarks...")
    print(f"Board sizes: {list(BENCHMARK_CONFIG.keys())}")

    results = []
    for board_size in sorted(BENCHMARK_CONFIG.keys()):
        try:
            board_results = run_single_benchmark_gpu(board_size, power_monitor,
                                                    n_trees=n_trees,
                                                    n_playouts=n_playouts,
                                                    variant=variant)
            results.extend(board_results)  # Flatten: add all trials to results
        except Exception as e:
            print(f"  ❌ Error on {board_size}×{board_size}: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("\n❌ No results to save!")
        return

    # Save results
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename with mode name, hostname, and timestamp
    hostname = system_info['hostname'].split('.')[0]  # Remove domain
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'mcts_benchmark_gpu_{mode_name}_{hostname}_{timestamp}.csv')

    save_results(results, output_file, system_info)

    # Print summary (grouped by board size, showing averages)
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Board':<10} {'Trials':<8} {'Avg Latency':<15} {'Avg Power':<15} {'Avg Energy':<15}")
    print("-" * 70)

    import statistics
    # Group results by board size
    board_groups = {}
    for result in results:
        board = result['board_size']
        if board not in board_groups:
            board_groups[board] = []
        board_groups[board].append(result)

    for board in sorted(board_groups.keys(), key=lambda x: int(x.split('x')[0])):
        trials = board_groups[board]
        avg_lat = statistics.mean([t['total_latency_ms'] for t in trials])
        avg_pow = statistics.mean([t['total_power_mw'] for t in trials])
        avg_eng = statistics.mean([t['total_energy_uj'] for t in trials])
        print(f"{board:<10} {len(trials):<8} "
              f"{avg_lat:>10.1f} ms   "
              f"{avg_pow:>10.1f} mW   "
              f"{avg_eng:>10.1f} µJ")

    print("\n" + "=" * 70)
    print("✅ GPU Benchmark complete!")
    print(f"📊 Results: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
