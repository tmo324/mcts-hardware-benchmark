#!/usr/bin/env python3
"""
MCTS GPU Benchmark - Main Entry Point
======================================

Run Monte Carlo Tree Search benchmarks on GPU hardware using mcts_numba_cuda.

Requirements:
    - CUDA Toolkit installed
    - NVIDIA GPU with CUDA support
    - numba package (pip install numba)

Usage:
    python main_gpu.py

Output:
    CSV file in results/ directory with per-phase timing and power measurements

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
    main()
