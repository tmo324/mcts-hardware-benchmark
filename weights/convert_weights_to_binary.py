#!/usr/bin/env python3
"""
Convert trained neural network weights from pickle format to binary format for C++.

This script converts the .pkl weight files from the crossbar training directory
into binary files that can be easily loaded by C++/CUDA code.

Binary format:
[4 bytes: rows (int32)]
[4 bytes: cols (int32)]
[rows×cols×4 bytes: weights (float32) in row-major order]

Usage:
    python3 convert_weights_to_binary.py
"""

import pickle
import numpy as np
import struct
import os
from pathlib import Path

# Paths
SOURCE_DIR = "/home/tm431/HPE_Duke/IMC_MCTS_main/crossbar_training/results"
TARGET_DIR = "/home/tm431/HPE_Duke/mcts-hardware-benchmark/weights"

def convert_pkl_to_bin(board_size):
    """
    Convert pickle weights to binary format for a specific board size.

    Args:
        board_size: Size of the Go board (2, 3, 5, 9, 13, or 19)
    """
    # Source pickle file
    pkl_path = os.path.join(SOURCE_DIR, f"weights_{board_size}x{board_size}", "best_improved_model.pkl")

    if not os.path.exists(pkl_path):
        print(f"⚠️  Warning: {pkl_path} not found, skipping...")
        return False

    # Load pickle file
    print(f"Loading {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Convert to numpy arrays if they're lists
    weights1 = np.array(data['weights1'], dtype=np.float32)
    weights2 = np.array(data['weights2'], dtype=np.float32)

    # Print info
    print(f"  weights1 shape: {weights1.shape} ({weights1.dtype})")
    print(f"  weights2 shape: {weights2.shape} ({weights2.dtype})")
    print(f"  accuracy: {data.get('accuracy', 'N/A')}")

    # Target binary files
    target_subdir = os.path.join(TARGET_DIR, f"{board_size}x{board_size}")
    bin_path1 = os.path.join(target_subdir, "weights1.bin")
    bin_path2 = os.path.join(target_subdir, "weights2.bin")

    # Save weights1 as binary
    with open(bin_path1, 'wb') as f:
        # Write dimensions: rows, cols (4 bytes each, int32)
        f.write(struct.pack('ii', weights1.shape[0], weights1.shape[1]))
        # Write data as float32 in row-major order
        weights1.astype(np.float32).tofile(f)

    # Save weights2 as binary
    with open(bin_path2, 'wb') as f:
        f.write(struct.pack('ii', weights2.shape[0], weights2.shape[1]))
        weights2.astype(np.float32).tofile(f)

    # Verify file sizes
    size1 = os.path.getsize(bin_path1)
    size2 = os.path.getsize(bin_path2)
    expected1 = 8 + weights1.size * 4  # 8 bytes header + data
    expected2 = 8 + weights2.size * 4

    print(f"✅ Saved {bin_path1} ({size1} bytes, expected {expected1})")
    print(f"✅ Saved {bin_path2} ({size2} bytes, expected {expected2})")

    return True

def main():
    """Convert all board sizes."""
    print("="*60)
    print("Converting Neural Network Weights to Binary Format")
    print("="*60)
    print()

    board_sizes = [2, 3, 5, 9, 13, 19]
    success_count = 0

    for size in board_sizes:
        print(f"\n[{size}×{size} Board]")
        if convert_pkl_to_bin(size):
            success_count += 1
        print()

    print("="*60)
    print(f"Conversion Complete: {success_count}/{len(board_sizes)} successful")
    print("="*60)

    # Print summary
    print("\nGenerated files:")
    for size in board_sizes:
        subdir = os.path.join(TARGET_DIR, f"{size}x{size}")
        if os.path.exists(os.path.join(subdir, "weights1.bin")):
            print(f"  {size}x{size}/weights1.bin")
            print(f"  {size}x{size}/weights2.bin")

if __name__ == "__main__":
    main()
