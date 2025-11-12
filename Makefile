# Makefile for MCTS CPU Benchmark (C++ Implementation)
# Optimized for Intel Xeon Platinum 8462Y+ (Sapphire Rapids)

CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -ffast-math -Wall -Wextra
TARGET = mcts_cpu_benchmark
SOURCE = mcts_cpu_benchmark.cpp

# Default target
all: $(TARGET)

# Compile the benchmark
$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE)
	@echo "✓ Compiled successfully: ./$(TARGET)"
	@echo "  Run with: ./$(TARGET) --help"

# Clean build artifacts
clean:
	rm -f $(TARGET)
	@echo "✓ Cleaned build artifacts"

# Run quick test (2x2 board, 1000 iterations)
test: $(TARGET)
	@echo "Running quick test (2x2 board, 1000 iterations)..."
	./$(TARGET) --board-size 2 --iterations 1000 --trials 1

# Run all board sizes (equivalent to old run_all_benchmarks.sh)
benchmark: $(TARGET)
	@echo "Running comprehensive benchmark (all board sizes)..."
	./$(TARGET) --all-sizes --iterations 5000 --trials 5

# Help target
help:
	@echo "MCTS CPU Benchmark - Make targets:"
	@echo "  make          - Build the benchmark (default)"
	@echo "  make test     - Quick test (2x2, 1000 iterations)"
	@echo "  make benchmark - Run full benchmark suite"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make help     - Show this help message"

.PHONY: all clean test benchmark help
