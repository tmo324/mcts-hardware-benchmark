#!/usr/bin/env python3
"""
MCTS GPU Benchmark - Fair Comparison Mode
==========================================

Run Monte Carlo Tree Search benchmarks on GPU hardware with single-tree mode
for fair comparison with CPU implementation.

This mode uses:
    - n_trees = 1 (single tree, like CPU)
    - n_playouts = 1 (sequential playouts, like CPU)
    - variant = 'ocp_thrifty' (one child playout, minimal overhead)

This configuration performs the SAME computational work as the CPU version,
allowing for a true hardware-to-hardware performance comparison.

Requirements:
    - CUDA Toolkit installed
    - NVIDIA GPU with CUDA support
    - numba package (pip install numba)

Usage:
    python main_gpu_fair.py

Output:
    CSV file in results/ directory: mcts_benchmark_gpu_fair_<hostname>_<timestamp>.csv

Author: MCTS Hardware Benchmark Project
"""

import sys
import os

# Check for numba before running
try:
    import numba
    import numba.cuda
    if not numba.cuda.is_available():
        print("❌ Error: CUDA is not available!")
        print("   Make sure:")
        print("   1. NVIDIA GPU drivers are installed")
        print("   2. CUDA Toolkit is installed")
        print("   3. GPU is CUDA-capable")
        sys.exit(1)
except ImportError:
    print("❌ Error: numba not installed!")
    print("   Install with: pip install numba")
    print("   Also requires CUDA Toolkit to be installed")
    sys.exit(1)

# Import the main benchmark function
from mcts_gpu.benchmark import main

if __name__ == "__main__":
    # Run in fair comparison mode: single tree, sequential playouts
    main(n_trees=1, n_playouts=1, variant='ocp_thrifty', mode_name='fair')
