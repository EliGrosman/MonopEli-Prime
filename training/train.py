"""Main training module for Monopoly RL agents.

This module provides training functionality using Stable-Baselines3's MaskablePPO
with action masking support for the Monopoly environment.

Usage:
    from training import train_agent

    model = train_agent(
        total_timesteps=1_000_000,
        opponent_type="random",
        save_path="models/ppo_monopoly",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class TrainingConfig:
    """Configuration for training.

    Attributes:
        total_timesteps: Total training steps
        learning_rate: Learning rate for optimizer
        n_steps: Steps per update
        batch_size: Minibatch size
        n_epochs: Epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm for clipping
        num_envs: Number of parallel environments
        seed: Random seed
    """
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_envs: int = 8
    seed: int | None = 42


@dataclass
class EnvironmentConfig:
    """Configuration for environment.

    Attributes:
        num_players: Number of players
        opponent_type: Type of opponent ("random", "rule_based", "aggressive", "conservative")
        max_turns: Max turns before truncation
        reward_type: Reward type ("sparse" or "dense")
    """
    num_players: int = 2
    opponent_type: str = "random"
    max_turns: int = 1000
    reward_type: str = "sparse"


def make_env(
    env_config: EnvironmentConfig,
    seed: int | None = None,
) -> Callable[[], Any]:
    """Create environment factory for vectorized environments.

    Args:
        env_config: Environment configuration
        seed: Random seed

    Returns:
        Factory function that creates environments
    """
    def _init() -> Any:
        from monopoly_gym import SingleAgentMonopolyEnv

        env = SingleAgentMonopolyEnv(
            num_players=env_config.num_players,
            opponent_type=env_config.opponent_type,
            max_turns=env_config.max_turns,
            reward_type=env_config.reward_type,
            seed=seed,
        )
        return env

    return _init


def create_vectorized_env(
    env_config: EnvironmentConfig,
    num_envs: int = 8,
    seed: int | None = None,
) -> Any:
    """Create vectorized environment for parallel training.

    Args:
        env_config: Environment configuration
        num_envs: Number of parallel environments
        seed: Base random seed

    Returns:
        Vectorized environment
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    import numpy as np

    env_fns = []
    streams = np.random.SeedSequence(seed).spawn(num_envs)
    for i in range(num_envs):
        env_seed = int(streams[i].generate_state(1)[0])
        env_fns.append(make_env(env_config, env_seed))

    # Use SubprocVecEnv for true parallelism, DummyVecEnv for debugging
    if num_envs > 1:
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)


def create_model(
    env: Any,
    config: TrainingConfig,
    tensorboard_log: str | None = None,
) -> Any:
    """Create MaskablePPO model.

    Args:
        env: Vectorized environment
        config: Training configuration
        tensorboard_log: Path for TensorBoard logs

    Returns:
        MaskablePPO model
    """
    from sb3_contrib import MaskablePPO

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        verbose=1,
        tensorboard_log=tensorboard_log,
        seed=config.seed,
    )

    return model


def train_agent(
    training_config: TrainingConfig | None = None,
    env_config: EnvironmentConfig | None = None,
    save_path: str | Path = "models/ppo_monopoly",
    tensorboard_log: str | Path | None = "logs/tensorboard",
    checkpoint_freq: int = 100_000,
    eval_freq: int = 50_000,
    eval_episodes: int = 20,
) -> Any:
    """Train a Monopoly agent using MaskablePPO.

    Args:
        training_config: Training hyperparameters
        env_config: Environment settings
        save_path: Path to save final model
        tensorboard_log: Path for TensorBoard logs
        checkpoint_freq: Save checkpoint every N steps
        eval_freq: Evaluate every N steps
        eval_episodes: Number of evaluation episodes

    Returns:
        Trained model
    """
    from stable_baselines3.common.callbacks import (
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )

    if training_config is None:
        training_config = TrainingConfig()
    if env_config is None:
        env_config = EnvironmentConfig()

    # Create save directory
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Create TensorBoard directory
    if tensorboard_log is not None:
        tensorboard_log = str(Path(tensorboard_log))
        Path(tensorboard_log).mkdir(parents=True, exist_ok=True)

    # Create vectorized training environment
    train_env = create_vectorized_env(
        env_config,
        num_envs=training_config.num_envs,
        seed=training_config.seed,
    )

    # Create evaluation environment (single env)
    eval_env = create_vectorized_env(
        env_config,
        num_envs=1,
        seed=training_config.seed + 1000 if training_config.seed else None,
    )

    # Create model
    model = create_model(train_env, training_config, tensorboard_log)

    # Set up callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // training_config.num_envs,
        save_path=str(save_path.parent / "checkpoints"),
        name_prefix="monopoly_ppo",
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path.parent / "best_model"),
        log_path=str(save_path.parent / "eval_logs"),
        eval_freq=eval_freq // training_config.num_envs,
        n_eval_episodes=eval_episodes,
        deterministic=True,
    )
    callbacks.append(eval_callback)

    # Train
    model.learn(
        total_timesteps=training_config.total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=True,
    )

    # Save final model
    model.save(str(save_path))

    # Cleanup
    train_env.close()
    eval_env.close()

    return model


def load_model(model_path: str | Path) -> Any:
    """Load a trained model.

    Args:
        model_path: Path to saved model

    Returns:
        Loaded model
    """
    from sb3_contrib import MaskablePPO

    return MaskablePPO.load(str(model_path))
