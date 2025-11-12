#!/usr/bin/env python3
"""
MCTS CPU Benchmark - Force psutil Power Monitoring

This version forces psutil power estimation instead of nvidia-smi,
useful on GPU servers where nvidia-smi returns 0 for CPU benchmarks.

Psutil provides power estimation based on:
- CPU utilization percentage
- Estimated TDP for CPU model (270W for Xeon Platinum 8462Y+)
- Formula: Power = Idle + (TDP - Idle) × CPU%
- Accuracy: ±20-30%

Usage:
    python3 main_cpu_psutil.py
"""

import os
import sys

# Set environment variable to force psutil method
os.environ['FORCE_PSUTIL_POWER'] = '1'

# Import after setting environment variable
from mcts_cpu.benchmark import main

if __name__ == "__main__":
    print("=" * 70)
    print("MCTS CPU Benchmark - Using psutil Power Estimation")
    print("=" * 70)
    print("\nForcing psutil power monitoring method")
    print("This provides estimated power based on CPU usage and TDP.")
    print("Estimated TDP for Xeon Platinum: ~270W")
    print("Accuracy: ±20-30%")
    print("=" * 70)

    main()
