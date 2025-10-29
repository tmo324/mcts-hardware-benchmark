#!/usr/bin/env python3
"""
Quick comparison of benchmark results to ReasonCore SST simulations
"""

import csv

# ReasonCore SST data (from scalability_energy.csv)
SST_DATA = {
    '2x2': {'energy_uj': 0.185, 'latency_ms': 0.00005},  # Estimated from iterations/sec
    '3x3': {'energy_uj': 0.290, 'latency_ms': 0.00005},
    '5x5': {'energy_uj': 1.405, 'latency_ms': 0.000113},
    '9x9': {'energy_uj': 24.836, 'latency_ms': 0.000444},
    '13x13': {'energy_uj': 152.621, 'latency_ms': 0.000886},
    '19x19': {'energy_uj': 1286.528, 'latency_ms': 0.001959}
}

# Read benchmark results
with open('results/mcts_benchmark_tr220-dt01_20251029_154111.csv') as f:
    reader = csv.DictReader(f)
    benchmark_data = list(reader)

print("=" * 90)
print("THREADRIPPER vs REASONCORE SST COMPARISON")
print("=" * 90)
print(f"\n{'Board':<10} {'Threadripper':<30} {'ReasonCore SST':<30} {'Ratio (CPU/ASIC)':<20}")
print("-" * 90)

for row in benchmark_data:
    board = row['board_size']

    # Threadripper results
    thr_energy = float(row['total_energy_uj'])
    thr_latency = float(row['total_latency_ms'])

    # SST results
    sst_energy = SST_DATA[board]['energy_uj']
    sst_latency = SST_DATA[board]['latency_ms']

    # Calculate ratios
    energy_ratio = thr_energy / sst_energy
    latency_ratio = thr_latency / sst_latency

    print(f"{board:<10} {thr_energy:>12.1f} µJ, {thr_latency:>8.1f} ms   "
          f"{sst_energy:>12.3f} µJ, {sst_latency:>8.5f} ms   "
          f"{energy_ratio:>8.0f}× energy, {latency_ratio:>8.0f}× slower")

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print("The AMD Threadripper PRO 5945WX consumes ~47,000× to 2,800,000× more energy")
print("than the simulated ReasonCore ASIC for the same MCTS workload.")
print("\nThis demonstrates the dramatic efficiency advantage of custom silicon")
print("for specialized algorithms like MCTS.")
