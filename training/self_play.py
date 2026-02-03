"""Self-play training for Monopoly RL agents.

This module implements self-play training where an agent learns by playing
against past versions of itself, creating a curriculum of increasingly
skilled opponents.

Usage:
    from training.self_play import SelfPlayTrainer, SelfPlayConfig

    config = SelfPlayConfig(
        total_timesteps=1_000_000,
        checkpoint_freq=50_000,
        past_version_prob=0.5,
    )
    trainer = SelfPlayTrainer(config)
    model = trainer.train()
"""

from __future__ import annotations

import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class SelfPlayConfig:
    """Configuration for self-play training.

    Attributes:
        total_timesteps: Total training steps
        num_envs: Number of parallel environments
        checkpoint_freq: Save checkpoint every N steps
        past_version_prob: Probability of playing against past version (vs current)
        max_past_versions: Maximum number of past versions to keep
        eval_freq: Evaluate against past versions every N steps
        eval_episodes: Number of evaluation episodes
        learning_rate: Learning rate for optimizer
        n_steps: Steps per update
        batch_size: Minibatch size
        seed: Random seed
        save_dir: Directory to save models and checkpoints
    """

    total_timesteps: int = 1_000_000
    num_envs: int = 8
    checkpoint_freq: int = 50_000
    past_version_prob: float = 0.5
    max_past_versions: int = 10
    eval_freq: int = 25_000
    eval_episodes: int = 20
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 128
    seed: int | None = 42
    save_dir: str = "models/self_play"


@dataclass
class SelfPlayStats:
    """Statistics from self-play training.

    Attributes:
        total_steps: Total steps trained
        num_checkpoints: Number of checkpoints saved
        win_rates: Win rates against past versions over time
        training_time: Total training time in seconds
    """

    total_steps: int = 0
    num_checkpoints: int = 0
    win_rates: list[dict[str, float]] = field(default_factory=list)
    training_time: float = 0.0


class SelfPlayOpponent:
    """Wrapper for using a trained model as an opponent.

    This class wraps a trained MaskablePPO model to act as an opponent
    in the SingleAgentMonopolyEnv.
    """

    def __init__(self, model: Any, player_id: int, deterministic: bool = True):
        """Initialize self-play opponent.

        Args:
            model: Trained MaskablePPO model
            player_id: Player ID this opponent controls
            deterministic: Whether to use deterministic actions
        """
        self.model = model
        self.player_id = player_id
        self.deterministic = deterministic
        self.name = "SelfPlay"

    def choose_action(
        self,
        observation: dict[str, Any] | None,
        action_mask: "NDArray[np.bool_]",
        game: Any,
    ) -> int:
        """Select action using the trained model.

        Args:
            observation: Dict observation (may be None for optimization)
            action_mask: Binary mask of valid actions
            game: Game instance

        Returns:
            Action index
        """
        # If observation is None, we need to create one from the game
        if observation is None:
            from monopoly_gym.observation import ObservationEncoder, flatten_observation

            encoder = ObservationEncoder(len(game.players))
            obs_dict = encoder.encode(game, self.player_id)
            obs = flatten_observation(obs_dict)
        else:
            from monopoly_gym.observation import flatten_observation

            if isinstance(observation, dict):
                obs = flatten_observation(observation)
            else:
                obs = observation

        action, _ = self.model.predict(
            obs, action_masks=action_mask, deterministic=self.deterministic
        )
        return int(action)

    def reset(self) -> None:
        """Reset opponent state."""
        pass

    def notify_result(self, *args: Any, **kwargs: Any) -> None:
        """Receive feedback (unused for self-play opponent)."""
        pass


class SelfPlayEnv(gym.Env[NDArray[np.float32], int]):
    """Environment wrapper that uses past model versions as opponents.

    This environment randomly selects opponents from a pool of past
    model checkpoints, enabling self-play training.

    Inherits from gymnasium.Env for SB3 compatibility.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        checkpoint_dir: str | Path,
        past_version_prob: float = 0.5,
        num_players: int = 2,
        max_turns: int = 500,
        seed: int | None = None,
    ):
        """Initialize self-play environment.

        Args:
            checkpoint_dir: Directory containing model checkpoints
            past_version_prob: Probability of using past version vs random
            num_players: Number of players
            max_turns: Maximum turns per game
            seed: Random seed
        """
        super().__init__()

        self.checkpoint_dir = Path(checkpoint_dir)
        self.past_version_prob = past_version_prob
        self.num_players = num_players
        self.max_turns = max_turns
        self._seed = seed
        self.rng = random.Random(seed)

        self._checkpoints: list[Path] = []
        self._loaded_models: dict[str, Any] = {}
        self._current_opponent: Any = None

        # Create underlying environment
        from monopoly_gym import SingleAgentMonopolyEnv

        self._env = SingleAgentMonopolyEnv(
            num_players=num_players,
            opponent_type="random",  # Default, will be overridden
            max_turns=max_turns,
            reward_type="dense",
            seed=seed,
        )

        # Copy spaces from underlying env
        self.observation_space: spaces.Space[Any] = self._env.observation_space
        self.action_space: spaces.Space[int] = self._env.action_space
        self.render_mode = self._env.render_mode

    def refresh_checkpoints(self) -> None:
        """Refresh the list of available checkpoints."""
        if self.checkpoint_dir.exists():
            self._checkpoints = sorted(
                self.checkpoint_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime
            )

    def _select_opponent(self) -> Any:
        """Select an opponent (past version or random).

        Returns:
            Opponent agent
        """
        # Refresh checkpoints
        self.refresh_checkpoints()

        # Decide whether to use past version
        if self._checkpoints and self.rng.random() < self.past_version_prob:
            # Select random checkpoint (weighted toward recent)
            weights = [i + 1 for i in range(len(self._checkpoints))]
            checkpoint = self.rng.choices(self._checkpoints, weights=weights, k=1)[0]

            # Load model (with caching)
            checkpoint_str = str(checkpoint)
            if checkpoint_str not in self._loaded_models:
                from sb3_contrib import MaskablePPO

                self._loaded_models[checkpoint_str] = MaskablePPO.load(checkpoint_str)

            model = self._loaded_models[checkpoint_str]
            return SelfPlayOpponent(model, player_id=1)

        # Fall back to random agent
        from agents import RandomAgent

        return RandomAgent(player_id=1, seed=self._seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset environment with new opponent.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        # Select new opponent
        self._current_opponent = self._select_opponent()

        # Update environment's opponent
        self._env._opponents = {"player_1": self._current_opponent}

        return self._env.reset(seed=seed, options=options)

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        return self._env.step(action)

    def action_masks(self) -> NDArray[np.bool_]:
        """Get action mask for MaskablePPO compatibility."""
        return self._env.action_masks()

    def render(self) -> Any:
        """Render the environment."""
        return self._env.render()

    def close(self) -> None:
        """Clean up resources."""
        self._env.close()
        self._loaded_models.clear()


class SelfPlayTrainer:
    """Trainer for self-play learning.

    This trainer implements a self-play loop where the agent trains
    against past versions of itself, gradually improving through
    competition with increasingly skilled opponents.
    """

    def __init__(self, config: SelfPlayConfig | None = None):
        """Initialize self-play trainer.

        Args:
            config: Training configuration
        """
        self.config = config or SelfPlayConfig()
        self.stats = SelfPlayStats()

        # Create directories
        self.save_dir = Path(self.config.save_dir)
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _create_env(self) -> Any:
        """Create self-play environment.

        Returns:
            Vectorized self-play environment
        """
        from stable_baselines3.common.vec_env import DummyVecEnv

        def make_env(seed: int) -> Any:
            def _init() -> SelfPlayEnv:
                return SelfPlayEnv(
                    checkpoint_dir=self.checkpoint_dir,
                    past_version_prob=self.config.past_version_prob,
                    num_players=2,
                    max_turns=500,
                    seed=seed,
                )

            return _init

        env_fns = [
            make_env(self.config.seed + i if self.config.seed else i)
            for i in range(self.config.num_envs)
        ]

        return DummyVecEnv(env_fns)

    def _save_checkpoint(self, model: Any, step: int) -> Path:
        """Save a model checkpoint.

        Args:
            model: Model to save
            step: Current training step

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step}.zip"
        model.save(str(checkpoint_path))
        self.stats.num_checkpoints += 1

        # Prune old checkpoints if needed
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.zip"),
            key=lambda p: p.stat().st_mtime,
        )
        while len(checkpoints) > self.config.max_past_versions:
            oldest = checkpoints.pop(0)
            oldest.unlink()

        return checkpoint_path

    def _evaluate_against_past(self, model: Any) -> dict[str, float]:
        """Evaluate current model against past versions.

        Args:
            model: Current model to evaluate

        Returns:
            Dictionary with win rates
        """
        from monopoly_gym import SingleAgentMonopolyEnv

        results = {"vs_random": 0.0, "vs_past": 0.0}

        # Evaluate vs random
        random_wins = 0
        for i in range(self.config.eval_episodes):
            env = SingleAgentMonopolyEnv(
                num_players=2, opponent_type="random", max_turns=500, seed=9000 + i
            )
            obs, info = env.reset()
            done = False
            final_reward: SupportsFloat = 0.0
            while not done:
                mask = info.get("action_mask")
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                obs, final_reward, term, trunc, info = env.step(int(action))
                done = term or trunc
            if float(final_reward) > 0:
                random_wins += 1
            env.close()
        results["vs_random"] = random_wins / self.config.eval_episodes

        # Evaluate vs past versions (if any)
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.zip"))
        if checkpoints:
            from sb3_contrib import MaskablePPO

            past_wins = 0
            past_model = MaskablePPO.load(str(checkpoints[-1]))
            opponent = SelfPlayOpponent(past_model, player_id=1)

            for i in range(self.config.eval_episodes):
                env = SingleAgentMonopolyEnv(
                    num_players=2, opponent_type="random", max_turns=500, seed=9500 + i
                )
                env._opponents = {"player_1": opponent}
                obs, info = env.reset()
                done = False
                past_reward: SupportsFloat = 0.0
                while not done:
                    mask = info.get("action_mask")
                    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                    obs, past_reward, term, trunc, info = env.step(int(action))
                    done = term or trunc
                if float(past_reward) > 0:
                    past_wins += 1
                env.close()
            results["vs_past"] = past_wins / self.config.eval_episodes

        return results

    def train(self, verbose: bool = True) -> Any:
        """Run self-play training.

        Args:
            verbose: Whether to print progress

        Returns:
            Trained model
        """
        from sb3_contrib import MaskablePPO

        start_time = time.time()

        if verbose:
            print("=== Self-Play Training ===")
            print(f"Total steps: {self.config.total_timesteps:,}")
            print(f"Past version probability: {self.config.past_version_prob}")
            print(f"Checkpoint frequency: {self.config.checkpoint_freq:,}")
            print()

        # Create environment
        env = self._create_env()

        # Create model
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            verbose=0,
            seed=self.config.seed,
        )

        # Save initial checkpoint (so we have something to play against)
        self._save_checkpoint(model, 0)

        # Training loop with periodic checkpoints and evaluation
        steps_trained = 0
        while steps_trained < self.config.total_timesteps:
            # Train for checkpoint_freq steps
            steps_to_train = min(
                self.config.checkpoint_freq,
                self.config.total_timesteps - steps_trained,
            )

            model.learn(
                total_timesteps=steps_to_train,
                reset_num_timesteps=False,
                progress_bar=verbose,
            )
            steps_trained += steps_to_train

            # Save checkpoint
            self._save_checkpoint(model, steps_trained)

            # Refresh checkpoints in environments
            for i in range(self.config.num_envs):
                env.envs[i].refresh_checkpoints()

            # Evaluate
            if steps_trained % self.config.eval_freq == 0 or steps_trained >= self.config.total_timesteps:
                results = self._evaluate_against_past(model)
                self.stats.win_rates.append({"step": steps_trained, **results})

                if verbose:
                    print(f"\nStep {steps_trained:,}:")
                    print(f"  vs random: {results['vs_random']:.1%}")
                    if "vs_past" in results:
                        print(f"  vs past:   {results['vs_past']:.1%}")

        # Cleanup
        env.close()

        # Save final model
        final_path = self.save_dir / "final_model"
        model.save(str(final_path))

        self.stats.total_steps = steps_trained
        self.stats.training_time = time.time() - start_time

        if verbose:
            print(f"\n=== Training Complete ===")
            print(f"Total steps: {self.stats.total_steps:,}")
            print(f"Checkpoints: {self.stats.num_checkpoints}")
            print(f"Time: {self.stats.training_time:.0f}s")
            print(f"Final model: {final_path}")

        return model


def train_self_play(
    total_timesteps: int = 500_000,
    checkpoint_freq: int = 50_000,
    past_version_prob: float = 0.5,
    save_dir: str = "models/self_play",
    verbose: bool = True,
) -> Any:
    """Convenience function for self-play training.

    Args:
        total_timesteps: Total training steps
        checkpoint_freq: Save checkpoint every N steps
        past_version_prob: Probability of playing past version
        save_dir: Directory to save models
        verbose: Whether to print progress

    Returns:
        Trained model
    """
    config = SelfPlayConfig(
        total_timesteps=total_timesteps,
        checkpoint_freq=checkpoint_freq,
        past_version_prob=past_version_prob,
        save_dir=save_dir,
    )
    trainer = SelfPlayTrainer(config)
    return trainer.train(verbose=verbose)
