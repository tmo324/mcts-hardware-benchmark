#!/usr/bin/env python3
"""
MCTS GPU Benchmark - Maximum Capability Mode
=============================================

Run Monte Carlo Tree Search benchmarks on GPU hardware with full parallelization
to demonstrate maximum GPU performance capabilities.

This mode uses:
    - n_trees = 8 (8 parallel MCTS trees)
    - n_playouts = 128 (128 parallel rollouts per node)
    - variant = 'acp_prodigal' (all children playouts, maximum parallelism)

This configuration leverages GPU's massive parallelism and performs significantly
MORE computational work than the CPU version. This demonstrates GPU's throughput
capability but is NOT a fair comparison to CPU.

For fair hardware comparison, use main_gpu_fair.py instead.

Requirements:
    - CUDA Toolkit installed
    - NVIDIA GPU with CUDA support (at least 2GB memory)
    - numba package (pip install numba)

Usage:
    python main_gpu_capability.py

Output:
    CSV file in results/ directory: mcts_benchmark_gpu_capability_<hostname>_<timestamp>.csv

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
    # Run in maximum capability mode: 8 trees, 128 parallel playouts
    main(n_trees=8, n_playouts=128, variant='acp_prodigal', mode_name='capability')
