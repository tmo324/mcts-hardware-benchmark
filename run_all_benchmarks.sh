#!/bin/bash
# Run comprehensive MCTS CPU benchmark across all board sizes
# Usage: ./run_all_benchmarks.sh [iterations] [trials]

set -e  # Exit on error

# Default parameters (matching old Python defaults)
ITERATIONS=${1:-5000}
TRIALS=${2:-5}

echo "=========================================="
echo "MCTS CPU Benchmark (C++ Implementation)"
echo "=========================================="
echo "Iterations per trial: $ITERATIONS"
echo "Trials per board size: $TRIALS"
echo ""

# Compile if needed
if [ ! -f mcts_cpu_benchmark ]; then
    echo "Compiling benchmark..."
    make
    echo ""
fi

# Run benchmark with all board sizes
echo "Starting benchmark..."
./mcts_cpu_benchmark --all-sizes --iterations $ITERATIONS --trials $TRIALS

echo ""
echo "✓ Benchmark complete!"
echo "Results saved to: results/cpu_cpp_benchmark_*.csv"
