"""Training infrastructure for Monopoly RL agents."""

from .curriculum import (
    CurriculumConfig,
    CurriculumStage,
    CurriculumTrainer,
    StageResult,
)
from .evaluate import (
    EvaluationResult,
    EvaluationSummary,
    evaluate_against_opponent,
    evaluate_agent,
    play_game,
    quick_evaluate,
)
from .self_play import (
    SelfPlayConfig,
    SelfPlayEnv,
    SelfPlayOpponent,
    SelfPlayStats,
    SelfPlayTrainer,
    train_self_play,
)
from .train import (
    EnvironmentConfig,
    TrainingConfig,
    create_model,
    create_vectorized_env,
    load_model,
    make_env,
    train_agent,
)

__all__ = [
    # Training
    "TrainingConfig",
    "EnvironmentConfig",
    "train_agent",
    "load_model",
    "create_vectorized_env",
    "create_model",
    "make_env",
    # Evaluation
    "EvaluationResult",
    "EvaluationSummary",
    "evaluate_agent",
    "evaluate_against_opponent",
    "quick_evaluate",
    "play_game",
    # Curriculum
    "CurriculumStage",
    "CurriculumConfig",
    "CurriculumTrainer",
    "StageResult",
    # Self-Play
    "SelfPlayConfig",
    "SelfPlayStats",
    "SelfPlayTrainer",
    "SelfPlayEnv",
    "SelfPlayOpponent",
    "train_self_play",
]
