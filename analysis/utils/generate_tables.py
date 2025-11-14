#!/usr/bin/env python3
"""
Generate Publication-Ready Tables from MCTS Benchmark Results
==============================================================

This script reads CSV files from the results/ directory and generates
publication-ready tables for research papers:

1. Phase Breakdown Table: Shows latency and energy for each MCTS phase
2. Scalability Analysis Table: Shows performance across different board sizes

Outputs:
    - Markdown format (for README, reports)
    - LaTeX format (for papers)
    - CSV format (for further analysis)

Usage:
    python utils/generate_tables.py [--board-size 9] [--output-dir tables]

Author: MCTS Hardware Benchmark Project
"""

import os
import sys
import csv
import argparse
import statistics
from typing import Dict, List, Tuple
from pathlib import Path


def load_csv_file(filepath: str) -> List[Dict]:
    """Load and parse a CSV benchmark file"""
    results = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            numeric_fields = [
                'num_positions', 'iterations', 'trial_num', 'n_trees', 'n_playouts',
                'total_latency_ms', 'total_power_mw', 'total_energy_uj', 'tree_size',
                'selection_latency_ms', 'selection_power_mw', 'selection_energy_uj', 'selection_percent',
                'expansion_latency_ms', 'expansion_power_mw', 'expansion_energy_uj', 'expansion_percent',
                'simulation_latency_ms', 'simulation_power_mw', 'simulation_energy_uj', 'simulation_percent',
                'backpropagation_latency_ms', 'backpropagation_power_mw', 'backpropagation_energy_uj', 'backpropagation_percent'
            ]
            for field in numeric_fields:
                if field in row and row[field]:
                    row[field] = float(row[field])
            results.append(row)
    return results


def aggregate_by_board_size(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Group results by board size"""
    grouped = {}
    for result in results:
        board = result['board_size']
        if board not in grouped:
            grouped[board] = []
        grouped[board].append(result)
    return grouped


def calculate_stats(values: List[float]) -> Tuple[float, float]:
    """Calculate mean and standard deviation"""
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def format_value(mean: float, std: float, units: str = "", decimals: int = 1) -> str:
    """Format value as 'mean ± std units'"""
    if std == 0:
        return f"{mean:.{decimals}f}{units}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}{units}"


def generate_phase_breakdown_table(cpu_data: List[Dict], gpu_fair_data: List[Dict],
                                   board_size: str = '9x9') -> Dict[str, str]:
    """
    Generate phase breakdown comparison table

    Returns dict with keys: 'markdown', 'latex', 'csv'
    """

    # Filter by board size
    cpu_trials = [t for t in cpu_data if t['board_size'] == board_size]
    gpu_trials = [t for t in gpu_fair_data if t['board_size'] == board_size]

    if not cpu_trials or not gpu_trials:
        return {
            'markdown': f"Error: No data for board size {board_size}",
            'latex': "",
            'csv': ""
        }

    phases = ['selection', 'expansion', 'simulation', 'backpropagation']
    phase_names = ['Selection', 'Expansion', 'Rollout', 'Backpropagation']

    # Calculate statistics for each phase
    table_data = []
    for phase, display_name in zip(phases, phase_names):
        cpu_lat = [t[f'{phase}_latency_ms'] for t in cpu_trials]
        gpu_lat = [t[f'{phase}_latency_ms'] for t in gpu_trials]
        cpu_eng = [t[f'{phase}_energy_uj'] / 1000 for t in cpu_trials]  # Convert to mJ
        gpu_eng = [t[f'{phase}_energy_uj'] / 1000 for t in gpu_trials]  # Convert to mJ

        table_data.append({
            'phase': display_name,
            'cpu_lat_mean': statistics.mean(cpu_lat),
            'cpu_lat_std': statistics.stdev(cpu_lat) if len(cpu_lat) > 1 else 0,
            'gpu_lat_mean': statistics.mean(gpu_lat),
            'gpu_lat_std': statistics.stdev(gpu_lat) if len(gpu_lat) > 1 else 0,
            'cpu_eng_mean': statistics.mean(cpu_eng),
            'cpu_eng_std': statistics.stdev(cpu_eng) if len(cpu_eng) > 1 else 0,
            'gpu_eng_mean': statistics.mean(gpu_eng),
            'gpu_eng_std': statistics.stdev(gpu_eng) if len(gpu_eng) > 1 else 0,
        })

    # Add totals
    cpu_total_lat = [t['total_latency_ms'] for t in cpu_trials]
    gpu_total_lat = [t['total_latency_ms'] for t in gpu_trials]
    cpu_total_eng = [t['total_energy_uj'] / 1000 for t in cpu_trials]
    gpu_total_eng = [t['total_energy_uj'] / 1000 for t in gpu_trials]

    table_data.append({
        'phase': 'Total',
        'cpu_lat_mean': statistics.mean(cpu_total_lat),
        'cpu_lat_std': statistics.stdev(cpu_total_lat) if len(cpu_total_lat) > 1 else 0,
        'gpu_lat_mean': statistics.mean(gpu_total_lat),
        'gpu_lat_std': statistics.stdev(gpu_total_lat) if len(gpu_total_lat) > 1 else 0,
        'cpu_eng_mean': statistics.mean(cpu_total_eng),
        'cpu_eng_std': statistics.stdev(cpu_total_eng) if len(cpu_total_eng) > 1 else 0,
        'gpu_eng_mean': statistics.mean(gpu_total_eng),
        'gpu_eng_std': statistics.stdev(gpu_total_eng) if len(gpu_total_eng) > 1 else 0,
    })

    # Get iterations from config
    iterations = cpu_trials[0]['iterations']

    # Markdown format
    md = f"# Phase Breakdown Table - {board_size} Board ({iterations:.0f} iterations)\n\n"
    md += "| Phase            | Latency (ms)      |                   | Energy (mJ)      |                  |\n"
    md += "|------------------|-------------------|-------------------|------------------|------------------|\n"
    md += "|                  | **CPU**           | **GPU**           | **CPU**          | **GPU**          |\n"
    md += "|------------------|-------------------|-------------------|------------------|------------------|\n"

    for row in table_data:
        phase = row['phase']
        cpu_lat = format_value(row['cpu_lat_mean'], row['cpu_lat_std'], "", 2)
        gpu_lat = format_value(row['gpu_lat_mean'], row['gpu_lat_std'], "", 2)
        cpu_eng = format_value(row['cpu_eng_mean'], row['cpu_eng_std'], "", 2)
        gpu_eng = format_value(row['gpu_eng_mean'], row['gpu_eng_std'], "", 2)

        separator = "|------------------|-------------------|-------------------|------------------|------------------|" if phase == 'Total' else ""
        if phase == 'Total':
            md += separator + "\n"
        md += f"| {phase:<16} | {cpu_lat:>17} | {gpu_lat:>17} | {cpu_eng:>16} | {gpu_eng:>16} |\n"

    # LaTeX format
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{Phase Breakdown - {board_size} Board ({iterations:.0f} iterations)}}\n"
    latex += "\\begin{tabular}{l|cc|cc}\n"
    latex += "\\hline\n"
    latex += " & \\multicolumn{2}{c|}{Latency (ms)} & \\multicolumn{2}{c}{Energy (mJ)} \\\\\n"
    latex += "Phase & CPU & GPU & CPU & GPU \\\\\n"
    latex += "\\hline\n"

    for row in table_data:
        if row['phase'] == 'Total':
            latex += "\\hline\n"
        phase = row['phase']
        cpu_lat = f"{row['cpu_lat_mean']:.2f} $\\pm$ {row['cpu_lat_std']:.2f}" if row['cpu_lat_std'] > 0 else f"{row['cpu_lat_mean']:.2f}"
        gpu_lat = f"{row['gpu_lat_mean']:.2f} $\\pm$ {row['gpu_lat_std']:.2f}" if row['gpu_lat_std'] > 0 else f"{row['gpu_lat_mean']:.2f}"
        cpu_eng = f"{row['cpu_eng_mean']:.2f} $\\pm$ {row['cpu_eng_std']:.2f}" if row['cpu_eng_std'] > 0 else f"{row['cpu_eng_mean']:.2f}"
        gpu_eng = f"{row['gpu_eng_mean']:.2f} $\\pm$ {row['gpu_eng_std']:.2f}" if row['gpu_eng_std'] > 0 else f"{row['gpu_eng_mean']:.2f}"

        latex += f"{phase} & {cpu_lat} & {gpu_lat} & {cpu_eng} & {gpu_eng} \\\\\n"

    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    # CSV format
    csv_content = "phase,cpu_latency_mean,cpu_latency_std,gpu_latency_mean,gpu_latency_std,cpu_energy_mean,cpu_energy_std,gpu_energy_mean,gpu_energy_std\n"
    for row in table_data:
        csv_content += f"{row['phase']},{row['cpu_lat_mean']:.3f},{row['cpu_lat_std']:.3f},"
        csv_content += f"{row['gpu_lat_mean']:.3f},{row['gpu_lat_std']:.3f},"
        csv_content += f"{row['cpu_eng_mean']:.3f},{row['cpu_eng_std']:.3f},"
        csv_content += f"{row['gpu_eng_mean']:.3f},{row['gpu_eng_std']:.3f}\n"

    return {
        'markdown': md,
        'latex': latex,
        'csv': csv_content
    }


def generate_scalability_table(cpu_data: List[Dict], gpu_fair_data: List[Dict],
                               gpu_cap_data: List[Dict] = None) -> Dict[str, str]:
    """
    Generate scalability analysis table across all board sizes

    Returns dict with keys: 'markdown', 'latex', 'csv'
    """

    # Group by board size
    cpu_grouped = aggregate_by_board_size(cpu_data)
    gpu_fair_grouped = aggregate_by_board_size(gpu_fair_data)
    gpu_cap_grouped = aggregate_by_board_size(gpu_cap_data) if gpu_cap_data else {}

    board_sizes = sorted(cpu_grouped.keys(), key=lambda x: int(x.split('x')[0]))

    table_data = []
    for board_size in board_sizes:
        cpu_trials = cpu_grouped.get(board_size, [])
        gpu_fair_trials = gpu_fair_grouped.get(board_size, [])
        gpu_cap_trials = gpu_cap_grouped.get(board_size, [])

        if not cpu_trials or not gpu_fair_trials:
            continue

        iterations = cpu_trials[0]['iterations']

        # Calculate statistics
        cpu_lat = [t['total_latency_ms'] for t in cpu_trials]
        cpu_eng = [t['total_energy_uj'] / 1000 for t in cpu_trials]  # Convert to mJ
        gpu_fair_lat = [t['total_latency_ms'] for t in gpu_fair_trials]
        gpu_fair_eng = [t['total_energy_uj'] / 1000 for t in gpu_fair_trials]

        # Calculate derived metrics
        # Energy per move (mJ/iteration)
        cpu_eng_per_move = [e / iterations for e in cpu_eng]
        gpu_fair_eng_per_move = [e / iterations for e in gpu_fair_eng]

        # Throughput per Watt (iterations/s/W)
        # Throughput = iterations / (latency_ms / 1000) = iterations * 1000 / latency_ms
        # Power = energy_mJ / latency_ms = energy_mJ / latency_ms
        # Throughput/W = (iterations * 1000 / latency_ms) / (energy_mJ / latency_ms)
        #              = iterations * 1000 / energy_mJ
        cpu_throughput_per_w = [(iterations * 1000) / e for e in cpu_eng]
        gpu_fair_throughput_per_w = [(iterations * 1000) / e for e in gpu_fair_eng]

        row = {
            'board_size': board_size,
            'iterations': iterations,
            'cpu_lat_mean': statistics.mean(cpu_lat),
            'cpu_lat_std': statistics.stdev(cpu_lat) if len(cpu_lat) > 1 else 0,
            'gpu_fair_lat_mean': statistics.mean(gpu_fair_lat),
            'gpu_fair_lat_std': statistics.stdev(gpu_fair_lat) if len(gpu_fair_lat) > 1 else 0,
            'cpu_eng_mean': statistics.mean(cpu_eng),
            'cpu_eng_std': statistics.stdev(cpu_eng) if len(cpu_eng) > 1 else 0,
            'gpu_fair_eng_mean': statistics.mean(gpu_fair_eng),
            'gpu_fair_eng_std': statistics.stdev(gpu_fair_eng) if len(gpu_fair_eng) > 1 else 0,
            'cpu_eng_per_move_mean': statistics.mean(cpu_eng_per_move),
            'cpu_eng_per_move_std': statistics.stdev(cpu_eng_per_move) if len(cpu_eng_per_move) > 1 else 0,
            'gpu_fair_eng_per_move_mean': statistics.mean(gpu_fair_eng_per_move),
            'gpu_fair_eng_per_move_std': statistics.stdev(gpu_fair_eng_per_move) if len(gpu_fair_eng_per_move) > 1 else 0,
            'cpu_throughput_per_w_mean': statistics.mean(cpu_throughput_per_w),
            'cpu_throughput_per_w_std': statistics.stdev(cpu_throughput_per_w) if len(cpu_throughput_per_w) > 1 else 0,
            'gpu_fair_throughput_per_w_mean': statistics.mean(gpu_fair_throughput_per_w),
            'gpu_fair_throughput_per_w_std': statistics.stdev(gpu_fair_throughput_per_w) if len(gpu_fair_throughput_per_w) > 1 else 0,
        }

        # Add GPU capability data if available
        if gpu_cap_trials:
            gpu_cap_lat = [t['total_latency_ms'] for t in gpu_cap_trials]
            gpu_cap_eng = [t['total_energy_uj'] / 1000 for t in gpu_cap_trials]
            gpu_cap_eng_per_move = [e / iterations for e in gpu_cap_eng]
            gpu_cap_throughput_per_w = [(iterations * 1000) / e for e in gpu_cap_eng]

            row.update({
                'gpu_cap_lat_mean': statistics.mean(gpu_cap_lat),
                'gpu_cap_lat_std': statistics.stdev(gpu_cap_lat) if len(gpu_cap_lat) > 1 else 0,
                'gpu_cap_eng_mean': statistics.mean(gpu_cap_eng),
                'gpu_cap_eng_std': statistics.stdev(gpu_cap_eng) if len(gpu_cap_eng) > 1 else 0,
                'gpu_cap_eng_per_move_mean': statistics.mean(gpu_cap_eng_per_move),
                'gpu_cap_eng_per_move_std': statistics.stdev(gpu_cap_eng_per_move) if len(gpu_cap_eng_per_move) > 1 else 0,
                'gpu_cap_throughput_per_w_mean': statistics.mean(gpu_cap_throughput_per_w),
                'gpu_cap_throughput_per_w_std': statistics.stdev(gpu_cap_throughput_per_w) if len(gpu_cap_throughput_per_w) > 1 else 0,
            })

        table_data.append(row)

    # Markdown format
    has_gpu_cap = any('gpu_cap_lat_mean' in row for row in table_data)

    md = "# Scalability Analysis Table\n\n"
    md += "## Latency (ms)\n\n"
    if has_gpu_cap:
        md += "| Board Size | Iterations | CPU              | GPU (Fair)       | GPU (Capability) |\n"
        md += "|------------|------------|------------------|------------------|------------------|\n"
    else:
        md += "| Board Size | Iterations | CPU              | GPU (Fair)       |\n"
        md += "|------------|------------|------------------|------------------|\n"

    for row in table_data:
        cpu_lat = format_value(row['cpu_lat_mean'], row['cpu_lat_std'], "", 1)
        gpu_fair_lat = format_value(row['gpu_fair_lat_mean'], row['gpu_fair_lat_std'], "", 1)

        if has_gpu_cap and 'gpu_cap_lat_mean' in row:
            gpu_cap_lat = format_value(row['gpu_cap_lat_mean'], row['gpu_cap_lat_std'], "", 1)
            md += f"| {row['board_size']:<10} | {row['iterations']:<10.0f} | {cpu_lat:>16} | {gpu_fair_lat:>16} | {gpu_cap_lat:>16} |\n"
        else:
            md += f"| {row['board_size']:<10} | {row['iterations']:<10.0f} | {cpu_lat:>16} | {gpu_fair_lat:>16} |\n"

    md += "\n## Energy (mJ)\n\n"
    if has_gpu_cap:
        md += "| Board Size | Iterations | CPU              | GPU (Fair)       | GPU (Capability) |\n"
        md += "|------------|------------|------------------|------------------|------------------|\n"
    else:
        md += "| Board Size | Iterations | CPU              | GPU (Fair)       |\n"
        md += "|------------|------------|------------------|------------------|\n"

    for row in table_data:
        cpu_eng = format_value(row['cpu_eng_mean'], row['cpu_eng_std'], "", 1)
        gpu_fair_eng = format_value(row['gpu_fair_eng_mean'], row['gpu_fair_eng_std'], "", 1)

        if has_gpu_cap and 'gpu_cap_eng_mean' in row:
            gpu_cap_eng = format_value(row['gpu_cap_eng_mean'], row['gpu_cap_eng_std'], "", 1)
            md += f"| {row['board_size']:<10} | {row['iterations']:<10.0f} | {cpu_eng:>16} | {gpu_fair_eng:>16} | {gpu_cap_eng:>16} |\n"
        else:
            md += f"| {row['board_size']:<10} | {row['iterations']:<10.0f} | {cpu_eng:>16} | {gpu_fair_eng:>16} |\n"

    md += "\n## Energy per Move (mJ/iteration)\n\n"
    if has_gpu_cap:
        md += "| Board Size | CPU              | GPU (Fair)       | GPU (Capability) |\n"
        md += "|------------|------------------|------------------|------------------|\n"
    else:
        md += "| Board Size | CPU              | GPU (Fair)       |\n"
        md += "|------------|------------------|------------------|\n"

    for row in table_data:
        cpu_epm = format_value(row['cpu_eng_per_move_mean'], row['cpu_eng_per_move_std'], "", 4)
        gpu_fair_epm = format_value(row['gpu_fair_eng_per_move_mean'], row['gpu_fair_eng_per_move_std'], "", 4)

        if has_gpu_cap and 'gpu_cap_eng_per_move_mean' in row:
            gpu_cap_epm = format_value(row['gpu_cap_eng_per_move_mean'], row['gpu_cap_eng_per_move_std'], "", 4)
            md += f"| {row['board_size']:<10} | {cpu_epm:>16} | {gpu_fair_epm:>16} | {gpu_cap_epm:>16} |\n"
        else:
            md += f"| {row['board_size']:<10} | {cpu_epm:>16} | {gpu_fair_epm:>16} |\n"

    md += "\n## Throughput per Watt (moves/s/W)\n\n"
    if has_gpu_cap:
        md += "| Board Size | CPU              | GPU (Fair)       | GPU (Capability) |\n"
        md += "|------------|------------------|------------------|------------------|\n"
    else:
        md += "| Board Size | CPU              | GPU (Fair)       |\n"
        md += "|------------|------------------|------------------|\n"

    for row in table_data:
        cpu_tpw = format_value(row['cpu_throughput_per_w_mean'], row['cpu_throughput_per_w_std'], "", 1)
        gpu_fair_tpw = format_value(row['gpu_fair_throughput_per_w_mean'], row['gpu_fair_throughput_per_w_std'], "", 1)

        if has_gpu_cap and 'gpu_cap_throughput_per_w_mean' in row:
            gpu_cap_tpw = format_value(row['gpu_cap_throughput_per_w_mean'], row['gpu_cap_throughput_per_w_std'], "", 1)
            md += f"| {row['board_size']:<10} | {cpu_tpw:>16} | {gpu_fair_tpw:>16} | {gpu_cap_tpw:>16} |\n"
        else:
            md += f"| {row['board_size']:<10} | {cpu_tpw:>16} | {gpu_fair_tpw:>16} |\n"

    # LaTeX format
    latex = "\\begin{table}[h]\n\\centering\n"
    latex += "\\caption{Scalability Analysis}\n"

    if has_gpu_cap:
        latex += "\\begin{tabular}{l|r|rrr|rrr}\n\\hline\n"
        latex += " & & \\multicolumn{3}{c|}{Latency (ms)} & \\multicolumn{3}{c}{Energy (mJ)} \\\\\n"
        latex += "Board & Iter. & CPU & GPU-F & GPU-C & CPU & GPU-F & GPU-C \\\\\n"
    else:
        latex += "\\begin{tabular}{l|r|rr|rr}\n\\hline\n"
        latex += " & & \\multicolumn{2}{c|}{Latency (ms)} & \\multicolumn{2}{c}{Energy (mJ)} \\\\\n"
        latex += "Board & Iter. & CPU & GPU-F & CPU & GPU-F \\\\\n"

    latex += "\\hline\n"

    for row in table_data:
        board = row['board_size']
        iterations = f"{row['iterations']:.0f}"
        cpu_lat = f"{row['cpu_lat_mean']:.1f}"
        gpu_fair_lat = f"{row['gpu_fair_lat_mean']:.1f}"
        cpu_eng = f"{row['cpu_eng_mean']:.1f}"
        gpu_fair_eng = f"{row['gpu_fair_eng_mean']:.1f}"

        if has_gpu_cap and 'gpu_cap_lat_mean' in row:
            gpu_cap_lat = f"{row['gpu_cap_lat_mean']:.1f}"
            gpu_cap_eng = f"{row['gpu_cap_eng_mean']:.1f}"
            latex += f"{board} & {iterations} & {cpu_lat} & {gpu_fair_lat} & {gpu_cap_lat} & {cpu_eng} & {gpu_fair_eng} & {gpu_cap_eng} \\\\\n"
        else:
            latex += f"{board} & {iterations} & {cpu_lat} & {gpu_fair_lat} & {cpu_eng} & {gpu_fair_eng} \\\\\n"

    latex += "\\hline\n\\end{tabular}\n\\end{table}\n"

    # CSV format
    csv_header = "board_size,iterations,cpu_lat_mean,cpu_lat_std,gpu_fair_lat_mean,gpu_fair_lat_std,"
    csv_header += "cpu_eng_mean,cpu_eng_std,gpu_fair_eng_mean,gpu_fair_eng_std,"
    csv_header += "cpu_eng_per_move_mean,cpu_eng_per_move_std,gpu_fair_eng_per_move_mean,gpu_fair_eng_per_move_std,"
    csv_header += "cpu_throughput_per_w_mean,cpu_throughput_per_w_std,gpu_fair_throughput_per_w_mean,gpu_fair_throughput_per_w_std"

    if has_gpu_cap:
        csv_header += ",gpu_cap_lat_mean,gpu_cap_lat_std,gpu_cap_eng_mean,gpu_cap_eng_std,"
        csv_header += "gpu_cap_eng_per_move_mean,gpu_cap_eng_per_move_std,gpu_cap_throughput_per_w_mean,gpu_cap_throughput_per_w_std"

    csv_header += "\n"

    csv_content = csv_header
    for row in table_data:
        line = f"{row['board_size']},{row['iterations']:.0f},"
        line += f"{row['cpu_lat_mean']:.3f},{row['cpu_lat_std']:.3f},"
        line += f"{row['gpu_fair_lat_mean']:.3f},{row['gpu_fair_lat_std']:.3f},"
        line += f"{row['cpu_eng_mean']:.3f},{row['cpu_eng_std']:.3f},"
        line += f"{row['gpu_fair_eng_mean']:.3f},{row['gpu_fair_eng_std']:.3f},"
        line += f"{row['cpu_eng_per_move_mean']:.6f},{row['cpu_eng_per_move_std']:.6f},"
        line += f"{row['gpu_fair_eng_per_move_mean']:.6f},{row['gpu_fair_eng_per_move_std']:.6f},"
        line += f"{row['cpu_throughput_per_w_mean']:.3f},{row['cpu_throughput_per_w_std']:.3f},"
        line += f"{row['gpu_fair_throughput_per_w_mean']:.3f},{row['gpu_fair_throughput_per_w_std']:.3f}"

        if has_gpu_cap and 'gpu_cap_lat_mean' in row:
            line += f",{row['gpu_cap_lat_mean']:.3f},{row['gpu_cap_lat_std']:.3f},"
            line += f"{row['gpu_cap_eng_mean']:.3f},{row['gpu_cap_eng_std']:.3f},"
            line += f"{row['gpu_cap_eng_per_move_mean']:.6f},{row['gpu_cap_eng_per_move_std']:.6f},"
            line += f"{row['gpu_cap_throughput_per_w_mean']:.3f},{row['gpu_cap_throughput_per_w_std']:.3f}"

        csv_content += line + "\n"

    return {
        'markdown': md,
        'latex': latex,
        'csv': csv_content
    }


def main():
    parser = argparse.ArgumentParser(description='Generate publication tables from MCTS benchmark results')
    parser.add_argument('--cpu-file', type=str, help='Path to CPU benchmark CSV file')
    parser.add_argument('--gpu-fair-file', type=str, help='Path to GPU fair mode CSV file')
    parser.add_argument('--gpu-cap-file', type=str, help='Path to GPU capability mode CSV file (optional)')
    parser.add_argument('--board-size', type=str, default='9x9', help='Board size for phase breakdown table (default: 9x9)')
    parser.add_argument('--output-dir', type=str, default='tables', help='Output directory (default: tables)')
    parser.add_argument('--auto-find', action='store_true', help='Automatically find latest CSV files in results/')

    args = parser.parse_args()

    # Auto-find CSV files if requested
    if args.auto_find:
        results_dir = Path('results')
        if not results_dir.exists():
            print("❌ Error: results/ directory not found!")
            sys.exit(1)

        # Find files
        cpu_files = sorted(results_dir.glob('mcts_benchmark_*[!gpu]*.csv'), key=os.path.getmtime, reverse=True)
        gpu_fair_files = sorted(results_dir.glob('mcts_benchmark_gpu_fair_*.csv'), key=os.path.getmtime, reverse=True)
        gpu_cap_files = sorted(results_dir.glob('mcts_benchmark_gpu_capability_*.csv'), key=os.path.getmtime, reverse=True)

        if not cpu_files:
            print("❌ Error: No CPU benchmark files found in results/")
            sys.exit(1)
        if not gpu_fair_files:
            print("❌ Error: No GPU fair mode benchmark files found in results/")
            sys.exit(1)

        args.cpu_file = str(cpu_files[0])
        args.gpu_fair_file = str(gpu_fair_files[0])
        args.gpu_cap_file = str(gpu_cap_files[0]) if gpu_cap_files else None

        print(f"📂 Auto-detected files:")
        print(f"   CPU:      {args.cpu_file}")
        print(f"   GPU Fair: {args.gpu_fair_file}")
        if args.gpu_cap_file:
            print(f"   GPU Cap:  {args.gpu_cap_file}")

    # Validate required files
    if not args.cpu_file or not args.gpu_fair_file:
        print("❌ Error: --cpu-file and --gpu-fair-file are required (or use --auto-find)")
        parser.print_help()
        sys.exit(1)

    # Load data
    print(f"\n📖 Loading data...")
    try:
        cpu_data = load_csv_file(args.cpu_file)
        gpu_fair_data = load_csv_file(args.gpu_fair_file)
        gpu_cap_data = load_csv_file(args.gpu_cap_file) if args.gpu_cap_file else None

        print(f"   CPU: {len(cpu_data)} trials")
        print(f"   GPU Fair: {len(gpu_fair_data)} trials")
        if gpu_cap_data:
            print(f"   GPU Capability: {len(gpu_cap_data)} trials")
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Generate phase breakdown table
    print(f"\n📊 Generating phase breakdown table (board size: {args.board_size})...")
    phase_tables = generate_phase_breakdown_table(cpu_data, gpu_fair_data, args.board_size)

    # Save phase breakdown tables
    with open(output_dir / 'phase_breakdown.md', 'w') as f:
        f.write(phase_tables['markdown'])
    with open(output_dir / 'phase_breakdown.tex', 'w') as f:
        f.write(phase_tables['latex'])
    with open(output_dir / 'phase_breakdown.csv', 'w') as f:
        f.write(phase_tables['csv'])

    print(f"   ✅ Phase breakdown saved to {output_dir}/phase_breakdown.[md|tex|csv]")

    # Generate scalability table
    print(f"\n📊 Generating scalability analysis table...")
    scalability_tables = generate_scalability_table(cpu_data, gpu_fair_data, gpu_cap_data)

    # Save scalability tables
    with open(output_dir / 'scalability_analysis.md', 'w') as f:
        f.write(scalability_tables['markdown'])
    with open(output_dir / 'scalability_analysis.tex', 'w') as f:
        f.write(scalability_tables['latex'])
    with open(output_dir / 'scalability_analysis.csv', 'w') as f:
        f.write(scalability_tables['csv'])

    print(f"   ✅ Scalability analysis saved to {output_dir}/scalability_analysis.[md|tex|csv]")

    # Print preview
    print("\n" + "=" * 70)
    print("PREVIEW - Phase Breakdown Table")
    print("=" * 70)
    print(phase_tables['markdown'])

    print("\n" + "=" * 70)
    print("PREVIEW - Scalability Analysis Table")
    print("=" * 70)
    print(scalability_tables['markdown'])

    print("\n✅ Table generation complete!")
    print(f"📁 Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
