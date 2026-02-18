"""MCTS (Monte Carlo Tree Search) package for Monopoly AI.

Provides MCTS search with optional learned value network for action selection.

Note: The network and training submodules require PyTorch. They are imported
lazily so the rest of the package works without torch installed.
"""

from __future__ import annotations

from .data import ReplayBuffer, TrainingExample, compute_outcomes, generate_training_data
from .eval import MCTSEvalResult, evaluate_mcts_agent, play_evaluation_game
from .features import extract_features, get_feature_size
from .search import MCTSConfig, MCTSNode, MCTSSearch, clone_game_state

__all__ = [
    "MCTSConfig",
    "MCTSEvalResult",
    "MCTSNode",
    "MCTSSearch",
    "ReplayBuffer",
    "TrainingExample",
    "clone_game_state",
    "compute_outcomes",
    "evaluate_mcts_agent",
    "extract_features",
    "generate_training_data",
    "get_feature_size",
    "play_evaluation_game",
]

# Torch-dependent submodules — only export when torch is available
try:
    from .network import ValueNetwork
    from .training import (
        SelfPlayConfig,
        TrainingConfig,
        TrainingStats,
        load_checkpoint,
        save_checkpoint,
        self_play_loop,
        train_value_network,
    )

    __all__ += [
        "SelfPlayConfig",
        "TrainingConfig",
        "TrainingStats",
        "ValueNetwork",
        "load_checkpoint",
        "save_checkpoint",
        "self_play_loop",
        "train_value_network",
    ]
except ImportError:
    pass
