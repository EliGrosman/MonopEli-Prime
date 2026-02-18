"""MCTS (Monte Carlo Tree Search) package for Monopoly AI.

Provides MCTS search with optional learned value network for action selection.
"""

from __future__ import annotations

from .data import ReplayBuffer, TrainingExample, compute_outcomes, generate_training_data
from .features import extract_features, get_feature_size
from .network import ValueNetwork
from .search import MCTSConfig, MCTSNode, MCTSSearch, clone_game_state

__all__ = [
    "MCTSConfig",
    "MCTSNode",
    "MCTSSearch",
    "ReplayBuffer",
    "TrainingExample",
    "ValueNetwork",
    "clone_game_state",
    "compute_outcomes",
    "extract_features",
    "generate_training_data",
    "get_feature_size",
]
