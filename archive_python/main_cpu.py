#!/usr/bin/env python3
"""
MCTS CPU Benchmark - Main Entry Point
======================================

Run Monte Carlo Tree Search benchmarks on CPU hardware.

Usage:
    python main_cpu.py

Output:
    CSV file in results/ directory with per-phase timing and power measurements

Author: MCTS Hardware Benchmark Project
"""

from mcts_cpu.benchmark import main

if __name__ == "__main__":
    main()
