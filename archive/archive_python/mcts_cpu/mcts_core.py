#!/usr/bin/env python3
"""
4-Phase Instrumented MCTS Implementation
=========================================

Simple, clean MCTS matching ReasonCore algorithm with per-phase timing.
Designed for fair hardware comparison benchmarking.

Author: MCTS Hardware Benchmark Project
"""

import random
import math
import time
from typing import List, Tuple, Optional, Dict

class GoState:
    """Simplified Go game state"""

    def __init__(self, board_size: int):
        self.board_size = board_size
        self.board = [[0 for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = 1  # 1=black, 2=white
        self.passes = 0
        self.game_over = False

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get list of valid moves"""
        moves = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r][c] == 0:  # Empty position
                    moves.append((r, c))
        return moves + [None]  # Include pass

    def make_move(self, move: Optional[Tuple[int, int]]) -> 'GoState':
        """Create new state after move"""
        new_state = GoState(self.board_size)
        new_state.board = [row[:] for row in self.board]
        new_state.current_player = self.current_player
        new_state.passes = self.passes

        if move is None:  # Pass
            new_state.passes += 1
            if new_state.passes >= 2:
                new_state.game_over = True
        else:
            r, c = move
            new_state.board[r][c] = new_state.current_player
            new_state.passes = 0
            if len(new_state.get_valid_moves()) == 1:  # Only pass left
                new_state.game_over = True

        new_state.current_player = 3 - new_state.current_player
        return new_state

    def get_winner(self) -> float:
        """Simple scoring: count stones"""
        black = sum(row.count(1) for row in self.board)
        white = sum(row.count(2) for row in self.board)
        return 1.0 if black > white else (0.5 if black == white else 0.0)


class MCTSNode:
    """MCTS tree node"""

    def __init__(self, state: GoState, parent: Optional['MCTSNode'] = None,
                 move: Optional[Tuple[int, int]] = None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children: Dict[Tuple[int, int], 'MCTSNode'] = {}
        self.visits = 0
        self.wins = 0.0
        self.untried_moves = state.get_valid_moves()

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        return self.state.game_over

    def uct_value(self, exploration_constant: float = 1.414) -> float:
        """Upper Confidence Bound for Trees"""
        if self.visits == 0:
            return float('inf')
        exploit = self.wins / self.visits
        explore = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore


class InstrumentedMCTS:
    """
    MCTS with 4-phase instrumentation matching ReasonCore architecture

    Phases:
      1. Selection: Tree policy (UCT)
      2. Expansion: Add new node
      3. Simulation: Random rollout
      4. Backpropagation: Update statistics
    """

    def __init__(self, board_size: int, iterations: int,
                 exploration_constant: float = 1.414,
                 rollout_depth: int = 100):
        self.board_size = board_size
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.rollout_depth = rollout_depth

        # Phase timing accumulators
        self.phase_times = {
            'selection': 0.0,
            'expansion': 0.0,
            'simulation': 0.0,
            'backpropagation': 0.0
        }
        self.phase_counts = {
            'selection': 0,
            'expansion': 0,
            'simulation': 0,
            'backpropagation': 0
        }

    def search(self, root_state: GoState) -> Dict:
        """
        Run MCTS search and return statistics

        Returns:
            dict with best_move and phase timing breakdowns
        """
        root = MCTSNode(root_state)

        # Reset phase timers
        for phase in self.phase_times:
            self.phase_times[phase] = 0.0
            self.phase_counts[phase] = 0

        start_time = time.perf_counter()

        # Run MCTS iterations
        for _ in range(self.iterations):
            self._run_iteration(root)

        total_time = time.perf_counter() - start_time

        # Find best move (most visits)
        best_move = None
        max_visits = -1
        for move, child in root.children.items():
            if child.visits > max_visits:
                max_visits = child.visits
                best_move = move

        return {
            'best_move': best_move,
            'total_time_ms': total_time * 1000,
            'iterations': self.iterations,
            'phase_times_ms': {k: v * 1000 for k, v in self.phase_times.items()},
            'phase_counts': self.phase_counts.copy(),
            'tree_size': self._count_nodes(root)
        }

    def _run_iteration(self, root: MCTSNode):
        """Single MCTS iteration with 4 phases instrumented"""

        # PHASE 1: SELECTION
        t_start = time.perf_counter()
        node = self._select(root)
        self.phase_times['selection'] += time.perf_counter() - t_start
        self.phase_counts['selection'] += 1

        # PHASE 2: EXPANSION
        t_start = time.perf_counter()
        if not node.is_terminal() and not node.is_fully_expanded():
            node = self._expand(node)
        self.phase_times['expansion'] += time.perf_counter() - t_start
        self.phase_counts['expansion'] += 1

        # PHASE 3: SIMULATION (Rollout)
        t_start = time.perf_counter()
        result = self._simulate(node.state)
        self.phase_times['simulation'] += time.perf_counter() - t_start
        self.phase_counts['simulation'] += 1

        # PHASE 4: BACKPROPAGATION
        t_start = time.perf_counter()
        self._backpropagate(node, result)
        self.phase_times['backpropagation'] += time.perf_counter() - t_start
        self.phase_counts['backpropagation'] += 1

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selection phase: traverse tree using UCT"""
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            else:
                # Select child with highest UCT value
                node = max(node.children.values(),
                          key=lambda n: n.uct_value(self.exploration_constant))
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expansion phase: add new child node"""
        move = random.choice(node.untried_moves)
        node.untried_moves.remove(move)
        new_state = node.state.make_move(move)
        child = MCTSNode(new_state, parent=node, move=move)
        node.children[move] = child
        return child

    def _simulate(self, state: GoState) -> float:
        """Simulation phase: random rollout"""
        current_state = GoState(state.board_size)
        current_state.board = [row[:] for row in state.board]
        current_state.current_player = state.current_player
        current_state.passes = state.passes
        current_state.game_over = state.game_over

        depth = 0
        while not current_state.game_over and depth < self.rollout_depth:
            moves = current_state.get_valid_moves()
            if not moves:
                break
            move = random.choice(moves)
            current_state = current_state.make_move(move)
            depth += 1

        return current_state.get_winner()

    def _backpropagate(self, node: MCTSNode, result: float):
        """Backpropagation phase: update node statistics"""
        while node is not None:
            node.visits += 1
            # Flip result for opponent
            node.wins += result
            result = 1.0 - result
            node = node.parent

    def _count_nodes(self, node: MCTSNode) -> int:
        """Count total nodes in tree"""
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count


def run_benchmark(board_size: int, iterations: int,
                  exploration_constant: float = 1.414,
                  rollout_depth: int = 100) -> Dict:
    """
    Run MCTS benchmark for a given configuration

    Args:
        board_size: Board dimension (e.g., 5 for 5x5)
        iterations: Number of MCTS iterations
        exploration_constant: UCT exploration parameter
        rollout_depth: Maximum rollout depth

    Returns:
        Dictionary with timing and phase breakdown
    """
    # Create initial state
    state = GoState(board_size)

    # Run MCTS
    mcts = InstrumentedMCTS(board_size, iterations, exploration_constant, rollout_depth)
    results = mcts.search(state)

    # Calculate phase percentages
    total = sum(results['phase_times_ms'].values())
    results['phase_percentages'] = {
        k: (v / total * 100) if total > 0 else 0
        for k, v in results['phase_times_ms'].items()
    }

    return results


if __name__ == "__main__":
    # Quick test
    print("MCTS Core - Quick Test")
    print("=" * 60)

    result = run_benchmark(board_size=5, iterations=100)

    print(f"\nBoard: 5x5, Iterations: 100")
    print(f"Total time: {result['total_time_ms']:.2f} ms")
    print(f"Tree size: {result['tree_size']} nodes")
    print(f"\nPhase Breakdown:")
    for phase, time_ms in result['phase_times_ms'].items():
        pct = result['phase_percentages'][phase]
        print(f"  {phase:15s}: {time_ms:8.2f} ms ({pct:5.1f}%)")
