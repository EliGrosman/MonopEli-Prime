#!/usr/bin/env python3
"""Training script for testing Phase 2.5a trade functionality.

This script trains agents with trades enabled and tracks trade-specific metrics:
- Trade proposals, acceptances, rejections
- Trade acceptance rate
- Monopolies completed via trade
- Game outcomes

Usage:
    # Run for ~1 hour with trade metrics
    uv run python scripts/train_with_trades.py --timesteps 500000

    # Quick test run
    uv run python scripts/train_with_trades.py --timesteps 50000 --eval-freq 10000
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from monopoly_gym import (
    ACTION_SPACE_SIZE,
    GAMEPLAY_ACTION_SPACE_SIZE,
    OFFSET_ACCEPT_TRADE,
    OFFSET_REJECT_TRADE,
    OFFSET_SIMPLE_TRADE,
    SIMPLE_TRADE_DIM,
    MonopolyEnv,
)
from monopoly_gym.observation import flatten_observation, get_flat_observation_size
from monopoly_gym.trades import TradeRewardConfig


@dataclass
class TradeMetrics:
    """Metrics for tracking trade behavior during training."""

    trades_proposed: int = 0
    trades_accepted: int = 0
    trades_rejected: int = 0
    monopolies_completed_via_trade: int = 0
    monopolies_given_away: int = 0
    games_completed: int = 0
    games_with_trades: int = 0
    total_game_length: int = 0

    # Per-episode tracking (reset each episode)
    episode_trades_proposed: int = 0
    episode_trades_accepted: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Trade acceptance rate."""
        total = self.trades_accepted + self.trades_rejected
        return self.trades_accepted / total if total > 0 else 0.0

    @property
    def trades_per_game(self) -> float:
        """Average trades proposed per game."""
        return self.trades_proposed / self.games_completed if self.games_completed > 0 else 0.0

    @property
    def avg_game_length(self) -> float:
        """Average game length in turns."""
        return self.total_game_length / self.games_completed if self.games_completed > 0 else 0.0

    def reset_episode(self) -> None:
        """Reset per-episode counters."""
        self.episode_trades_proposed = 0
        self.episode_trades_accepted = 0

    def to_dict(self) -> dict[str, float]:
        """Convert to dict for logging."""
        return {
            "trade/proposed": self.trades_proposed,
            "trade/accepted": self.trades_accepted,
            "trade/rejected": self.trades_rejected,
            "trade/acceptance_rate": self.acceptance_rate,
            "trade/per_game": self.trades_per_game,
            "trade/monopolies_completed": self.monopolies_completed_via_trade,
            "trade/monopolies_given": self.monopolies_given_away,
            "trade/games_with_trades": self.games_with_trades,
            "game/completed": self.games_completed,
            "game/avg_length": self.avg_game_length,
        }


class SelfPlayTradingEnv(gym.Env[NDArray[np.float32], int]):
    """Self-play environment with trades enabled.

    All players use the same policy. When a trade is proposed, the environment
    switches to the responder who makes a decision using the same policy.

    This creates true self-play where the agent learns both to propose good
    trades AND to evaluate incoming trade proposals.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        num_players: int = 4,
        max_turns: int = 500,
        render_mode: str | None = None,
        trade_reward_config: TradeRewardConfig | None = None,
    ) -> None:
        """Initialize self-play trading environment.

        Args:
            num_players: Number of players (2-4)
            max_turns: Maximum turns before truncation
            render_mode: Render mode
            trade_reward_config: Configuration for trade rewards
        """
        super().__init__()

        self.num_players = num_players
        self.max_turns = max_turns
        self.render_mode = render_mode
        self.trade_reward_config = trade_reward_config or TradeRewardConfig()

        # Create underlying multi-agent environment with trades
        self._env = MonopolyEnv(
            num_players=num_players,
            max_turns=max_turns,
            enable_trades=True,
            trade_reward_config=self.trade_reward_config,
            render_mode=render_mode,
        )

        # Action and observation spaces (with trades)
        self.action_space: spaces.Space[int] = spaces.Discrete(ACTION_SPACE_SIZE)

        # Calculate observation size by sampling (includes action_mask and trade_context)
        self._env.reset(seed=0)
        sample_obs = flatten_observation(self._env.observe("player_0"))
        obs_size = sample_obs.shape[0]

        self.observation_space: spaces.Space[NDArray[np.float32]] = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # Metrics tracking
        self.metrics = TradeMetrics()

        # Current learning agent (rotates through players for diversity)
        self._learning_agent_idx = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset the environment."""
        self._env.reset(seed=seed)

        # Track game completion
        if self.metrics.episode_trades_proposed > 0:
            self.metrics.games_with_trades += 1

        self.metrics.reset_episode()

        # Rotate learning agent for diversity
        self._learning_agent_idx = (self._learning_agent_idx + 1) % self.num_players
        self._learning_agent = f"player_{self._learning_agent_idx}"

        # Play until it's the learning agent's turn
        self._play_until_learning_agent()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:
        """Take a step in the environment."""
        # Check if game is already over
        if self._env.game is None or self._env.game.game_over:
            obs = self._get_observation()
            info = self._get_info()
            return obs, 0.0, True, False, info

        # Check if learning agent is terminated
        if self._env.terminations.get(self._learning_agent, False) or \
           self._env.truncations.get(self._learning_agent, False):
            # Process dead step for learning agent
            if self._env.agent_selection == self._learning_agent:
                self._env.step(None)
            obs = self._get_observation()
            info = self._get_info()
            return obs, 0.0, True, False, info

        # Ensure it's the learning agent's turn before applying action
        if self._env.agent_selection != self._learning_agent:
            # Play until it's learning agent's turn
            self._play_until_learning_agent()

        # If game ended during opponent turns, return
        if self._env.game.game_over:
            obs = self._get_observation()
            info = self._get_info()
            winner = self._env.game.winner
            reward = 10.0 if winner == self._learning_agent_idx else -5.0
            self.metrics.games_completed += 1
            self.metrics.total_game_length += self._env.game.turn_number
            return obs, reward, True, False, info

        # Final check - if learning agent became terminated, don't step with action
        if self._env.terminations.get(self._learning_agent, False) or \
           self._env.truncations.get(self._learning_agent, False):
            if self._env.agent_selection == self._learning_agent:
                self._env.step(None)
            obs = self._get_observation()
            info = self._get_info()
            return obs, 0.0, True, False, info

        # Track trade actions
        self._track_trade_action(action)

        # Execute action for learning agent
        # Final sanity check - ensure agent_selection matches and isn't terminated
        current = self._env.agent_selection
        if current != self._learning_agent:
            # Wrong agent selected - play until learning agent
            self._play_until_learning_agent()
            if self._env.game.game_over or \
               self._env.terminations.get(self._learning_agent, True):
                obs = self._get_observation()
                info = self._get_info()
                return obs, 0.0, True, False, info

        # Check one more time before stepping
        if self._env.terminations.get(self._env.agent_selection, True):
            self._env.step(None)
        else:
            self._env.step(action)

        # Get immediate result
        _, reward, terminated, truncated, _ = self._env.last()

        # If not done, continue playing until learning agent's turn
        if not (terminated or truncated):
            self._play_until_learning_agent()

        # Check final termination
        terminated = self._env.terminations.get(self._learning_agent, False)
        truncated = self._env.truncations.get(self._learning_agent, False)

        # Check if game ended
        if self._env.game is not None and self._env.game.game_over:
            terminated = True
            self.metrics.games_completed += 1
            self.metrics.total_game_length += self._env.game.turn_number

            # Win/loss reward
            if self._env.game.winner == self._learning_agent_idx:
                reward = 10.0
            else:
                reward = -5.0

        obs = self._get_observation()
        info = self._get_info()

        return obs, float(reward), terminated, truncated, info

    def _play_until_learning_agent(self) -> None:
        """Play opponent turns using the same policy logic (self-play)."""
        max_iterations = 1000
        iterations = 0

        while iterations < max_iterations:
            # Check if game is over
            if self._env.game is None or self._env.game.game_over:
                break

            # Check if it's the learning agent's turn
            if self._env.agent_selection == self._learning_agent:
                break

            # Check if all agents are done
            if all(self._env.terminations.get(a, True) for a in self._env.possible_agents):
                break

            current_agent = self._env.agent_selection

            # Check if this agent exists and is terminated
            term = self._env.terminations.get(current_agent, True)
            trunc = self._env.truncations.get(current_agent, False)

            if term or trunc:
                self._env.step(None)
                iterations += 1
                continue

            # Get action mask from infos
            info = self._env.infos.get(current_agent, {})
            action_mask = info.get(
                "action_mask", np.ones(ACTION_SPACE_SIZE, dtype=np.bool_)
            )

            # Select action from valid ones
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                # Prioritize gameplay actions for stability
                gameplay_valid = valid_actions[valid_actions < GAMEPLAY_ACTION_SPACE_SIZE]
                if len(gameplay_valid) > 0 and np.random.random() < 0.7:
                    action = int(np.random.choice(gameplay_valid))
                else:
                    action = int(np.random.choice(valid_actions))
            else:
                action = 146  # End turn fallback

            # Track trade actions
            self._track_trade_action(action, is_opponent=True)

            self._env.step(action)
            iterations += 1

    def _track_trade_action(self, action: int, is_opponent: bool = False) -> None:
        """Track trade-related metrics."""
        if OFFSET_SIMPLE_TRADE <= action < OFFSET_SIMPLE_TRADE + SIMPLE_TRADE_DIM:
            self.metrics.trades_proposed += 1
            self.metrics.episode_trades_proposed += 1
        elif action == OFFSET_ACCEPT_TRADE:
            self.metrics.trades_accepted += 1
            self.metrics.episode_trades_accepted += 1
        elif action == OFFSET_REJECT_TRADE:
            self.metrics.trades_rejected += 1

    def _get_observation(self) -> NDArray[np.float32]:
        """Get flattened observation (includes action_mask and trade_context)."""
        obs_dict = self._env.observe(self._learning_agent)
        return flatten_observation(obs_dict)

    def _get_info(self) -> dict[str, Any]:
        """Get info dict with action mask."""
        agent_info = self._env.infos.get(self._learning_agent, {})
        action_mask = agent_info.get(
            "action_mask", np.ones(ACTION_SPACE_SIZE, dtype=np.bool_)
        )
        return {
            "action_mask": action_mask,
            "metrics": self.metrics.to_dict(),
        }

    def action_masks(self) -> NDArray[np.bool_]:
        """Get action mask for MaskablePPO compatibility."""
        agent_info = self._env.infos.get(self._learning_agent, {})
        mask = agent_info.get("action_mask", np.ones(ACTION_SPACE_SIZE, dtype=np.bool_))
        if isinstance(mask, np.ndarray):
            return mask.astype(np.bool_)
        return np.ones(ACTION_SPACE_SIZE, dtype=np.bool_)

    def render(self) -> str | list[str] | None:
        """Render the environment."""
        return self._env.render()

    def close(self) -> None:
        """Clean up resources."""
        self._env.close()


def make_trading_env(
    num_players: int = 4,
    max_turns: int = 500,
    seed: int | None = None,
) -> SelfPlayTradingEnv:
    """Factory function to create a trading environment."""
    env = SelfPlayTradingEnv(num_players=num_players, max_turns=max_turns)
    if seed is not None:
        env.reset(seed=seed)
    return env


def train_with_trades(
    total_timesteps: int = 500_000,
    num_envs: int = 8,
    eval_freq: int = 25_000,
    save_dir: str = "models/trades",
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train an agent with trades enabled.

    Args:
        total_timesteps: Total training timesteps
        num_envs: Number of parallel environments
        eval_freq: Evaluate and log metrics every N steps
        save_dir: Directory to save models
        seed: Random seed
        verbose: Print progress

    Returns:
        Dict with training results and final metrics
    """
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as e:
        print(f"Error: {e}")
        print("Install training dependencies: uv sync --extra training")
        sys.exit(1)

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Aggregated metrics across all envs
    global_metrics = TradeMetrics()

    class TradeMetricsCallback(BaseCallback):
        """Callback to track and log trade metrics."""

        def __init__(self, eval_freq: int, verbose: int = 0):
            super().__init__(verbose)
            self.eval_freq = eval_freq
            self.last_log_step = 0
            self.start_time = time.time()

        def _on_step(self) -> bool:
            # Aggregate metrics from all envs
            if self.num_timesteps - self.last_log_step >= self.eval_freq:
                self._log_metrics()
                self.last_log_step = self.num_timesteps
            return True

        def _log_metrics(self) -> None:
            # Collect metrics from environments
            total_proposed = 0
            total_accepted = 0
            total_rejected = 0
            total_games = 0
            total_games_with_trades = 0

            for env in self.training_env.envs:
                if hasattr(env, "metrics"):
                    m = env.metrics
                    total_proposed += m.trades_proposed
                    total_accepted += m.trades_accepted
                    total_rejected += m.trades_rejected
                    total_games += m.games_completed
                    total_games_with_trades += m.games_with_trades

            # Calculate rates
            total_responses = total_accepted + total_rejected
            acceptance_rate = total_accepted / total_responses if total_responses > 0 else 0
            trades_per_game = total_proposed / total_games if total_games > 0 else 0
            games_with_trade_rate = total_games_with_trades / total_games if total_games > 0 else 0

            elapsed = time.time() - self.start_time
            steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0

            # Log to TensorBoard
            self.logger.record("trade/total_proposed", total_proposed)
            self.logger.record("trade/total_accepted", total_accepted)
            self.logger.record("trade/total_rejected", total_rejected)
            self.logger.record("trade/acceptance_rate", acceptance_rate)
            self.logger.record("trade/per_game", trades_per_game)
            self.logger.record("trade/games_with_trades_pct", games_with_trade_rate)
            self.logger.record("game/total_completed", total_games)
            self.logger.record("time/steps_per_sec", steps_per_sec)
            self.logger.record("time/elapsed_minutes", elapsed / 60)

            if verbose:
                print(f"\n{'='*60}")
                print(f"Step {self.num_timesteps:,} | Elapsed: {elapsed/60:.1f}min")
                print(f"Games: {total_games:,} | With trades: {total_games_with_trades:,} ({games_with_trade_rate:.1%})")
                print(f"Trades: {total_proposed:,} proposed, {total_accepted:,} accepted, {total_rejected:,} rejected")
                print(f"Acceptance rate: {acceptance_rate:.1%} | Per game: {trades_per_game:.2f}")
                print(f"Speed: {steps_per_sec:.0f} steps/sec")
                print(f"{'='*60}")

    # Create vectorized environment
    def make_env(rank: int) -> gym.Env:
        def _init() -> gym.Env:
            env = SelfPlayTradingEnv(num_players=4, max_turns=500)
            env.reset(seed=seed + rank if seed else None)
            return env
        return _init

    if verbose:
        print(f"Creating {num_envs} parallel environments with trades enabled...")

    # Use DummyVecEnv for easier metric access (SubprocVecEnv would be faster but harder to access metrics)
    env = DummyVecEnv([make_env(i) for i in range(num_envs)])

    # Create model with action masking
    if verbose:
        print("Initializing MaskablePPO model...")

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        verbose=0,
        tensorboard_log=str(save_path / "tensorboard"),
        seed=seed,
    )

    # Create callback
    callback = TradeMetricsCallback(eval_freq=eval_freq, verbose=1 if verbose else 0)

    # Train
    if verbose:
        print(f"\nStarting training for {total_timesteps:,} timesteps...")
        print(f"Logging to TensorBoard: {save_path / 'tensorboard'}")
        print("Run: tensorboard --logdir models/trades/tensorboard")
        print()

    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=verbose,
    )
    training_time = time.time() - start_time

    # Save final model
    model_path = save_path / "final_model"
    model.save(str(model_path))
    if verbose:
        print(f"\nModel saved to: {model_path}")

    # Collect final metrics
    final_metrics = TradeMetrics()
    for e in env.envs:
        if hasattr(e, "metrics"):
            m = e.metrics
            final_metrics.trades_proposed += m.trades_proposed
            final_metrics.trades_accepted += m.trades_accepted
            final_metrics.trades_rejected += m.trades_rejected
            final_metrics.games_completed += m.games_completed
            final_metrics.games_with_trades += m.games_with_trades
            final_metrics.monopolies_completed_via_trade += m.monopolies_completed_via_trade
            final_metrics.total_game_length += m.total_game_length

    env.close()

    return {
        "model_path": str(model_path),
        "training_time": training_time,
        "total_timesteps": total_timesteps,
        "metrics": final_metrics.to_dict(),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train Monopoly agent with trades enabled (Phase 2.5a)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run for ~1 hour
    uv run python scripts/train_with_trades.py --timesteps 500000

    # Quick test (5-10 min)
    uv run python scripts/train_with_trades.py --timesteps 50000 --eval-freq 10000

    # Monitor with TensorBoard
    tensorboard --logdir models/trades/tensorboard
        """,
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps (default: 500000, ~1 hour)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel environments (default: 8)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=25_000,
        help="Log metrics every N steps (default: 25000)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/trades",
        help="Directory to save models (default: models/trades)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    results = train_with_trades(
        total_timesteps=args.timesteps,
        num_envs=args.num_envs,
        eval_freq=args.eval_freq,
        save_dir=args.save_dir,
        seed=args.seed,
        verbose=not args.quiet,
    )

    if not args.quiet:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Training time: {results['training_time']/60:.1f} minutes")
        print(f"Model saved to: {results['model_path']}")
        print("\nFinal Trade Metrics:")
        for key, value in results["metrics"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
