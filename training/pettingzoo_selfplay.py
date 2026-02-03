"""Self-play training for Monopoly using PettingZoo's native API.

This module provides proper multi-agent self-play training where all agents
share the same policy network. The key insight is that in self-play, we want
the agent to learn from playing against itself, so all players use identical
(or near-identical) policies.

Architecture:
- Uses MonopolyEnv (PettingZoo AEC) directly
- Wraps it as a Gymnasium env for SB3 compatibility
- All opponent actions come from the same policy (true self-play)
- Supports curriculum: start vs random/rule-based, then self-play

Usage:
    from training.pettingzoo_selfplay import SelfPlayTrainer, SelfPlayConfig

    config = SelfPlayConfig(
        total_timesteps=1_000_000,
        opponent_type="self",  # or "random", "rule_based"
    )
    trainer = SelfPlayTrainer(config)
    trainer.train()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from monopoly_gym import MonopolyEnv, GAMEPLAY_ACTION_SPACE_SIZE
from monopoly_gym.observation import flatten_observation, get_flat_observation_size


@dataclass
class SelfPlayConfig:
    """Configuration for self-play training."""

    # Training
    total_timesteps: int = 1_000_000
    num_envs: int = 8
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 128
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01

    # Environment
    num_players: int = 4
    max_turns: int = 500

    # Self-play
    opponent_type: str = "self"  # "self", "random", "rule_based", "mixed"
    update_opponent_freq: int = 10_000  # Update opponent policy every N steps

    # Evaluation
    eval_freq: int = 25_000
    eval_episodes: int = 50

    # Checkpointing
    save_freq: int = 50_000
    save_dir: str = "models/selfplay"

    # Misc
    seed: int = 42
    verbose: bool = True


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""

    total_steps: int = 0
    games_completed: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_game_length: int = 0

    # Evaluation results
    eval_results: list[dict[str, float]] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses + self.draws
        return self.wins / total if total > 0 else 0.0

    @property
    def avg_game_length(self) -> float:
        return self.total_game_length / self.games_completed if self.games_completed > 0 else 0.0


class SelfPlayEnv(gym.Env[NDArray[np.float32], int]):
    """Gymnasium wrapper for PettingZoo MonopolyEnv with self-play.

    This environment:
    1. Wraps the multi-agent MonopolyEnv
    2. Learning agent is always player 0
    3. Opponents use a provided policy function (for self-play)
    4. Falls back to random/rule-based when no policy provided
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        num_players: int = 4,
        max_turns: int = 500,
        opponent_type: str = "random",
        render_mode: str | None = None,
    ):
        super().__init__()

        self.num_players = num_players
        self.max_turns = max_turns
        self.opponent_type = opponent_type
        self.render_mode = render_mode

        # Create underlying PettingZoo env (no trades for now)
        self._env = MonopolyEnv(
            num_players=num_players,
            max_turns=max_turns,
            enable_trades=False,
            render_mode=render_mode,
        )

        # Learning agent is always player 0
        self._learning_agent = "player_0"
        self._learning_agent_id = 0

        # Action and observation spaces
        self.action_space = spaces.Discrete(GAMEPLAY_ACTION_SPACE_SIZE)

        # Calculate observation size
        self._obs_size = get_flat_observation_size(num_players)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_size,), dtype=np.float32
        )

        # Opponent policy (set externally for self-play)
        self._opponent_policy: Callable[[NDArray, NDArray], int] | None = None

        # Opponent agents (for non-self-play modes)
        self._opponents = self._create_opponents()

        # Metrics
        self.episode_length = 0
        self.episode_reward = 0.0

    def _create_opponents(self) -> dict[str, Any]:
        """Create opponent agents based on opponent_type."""
        from agents import RandomAgent, RuleBasedAgent, AggressiveAgent, ConservativeAgent

        opponents = {}
        for i in range(1, self.num_players):
            agent_id = f"player_{i}"
            if self.opponent_type == "random":
                opponents[agent_id] = RandomAgent(i)
            elif self.opponent_type == "rule_based":
                opponents[agent_id] = RuleBasedAgent(i)
            elif self.opponent_type == "aggressive":
                opponents[agent_id] = AggressiveAgent(i)
            elif self.opponent_type == "conservative":
                opponents[agent_id] = ConservativeAgent(i)
            elif self.opponent_type == "mixed":
                # Mix of different agents
                agent_types = [RandomAgent, RuleBasedAgent, AggressiveAgent, ConservativeAgent]
                agent_class = agent_types[(i - 1) % len(agent_types)]
                opponents[agent_id] = agent_class(i)
            else:
                # Default to random for "self" mode (will use policy when available)
                opponents[agent_id] = RandomAgent(i)
        return opponents

    def set_opponent_policy(self, policy_fn: Callable[[NDArray, NDArray], int] | None) -> None:
        """Set the opponent policy function for self-play.

        Args:
            policy_fn: Function that takes (observation, action_mask) and returns action.
                      If None, falls back to opponent_type agents.
        """
        self._opponent_policy = policy_fn

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset the environment."""
        self._env.reset(seed=seed)

        self.episode_length = 0
        self.episode_reward = 0.0

        # Play until it's the learning agent's turn
        self._play_opponent_turns()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Take a step with the learning agent's action."""
        self.episode_length += 1

        # Check if game already over
        if self._env.game is None or self._env.game.game_over:
            return self._handle_game_over()

        # Check if learning agent is terminated
        if self._is_agent_done(self._learning_agent):
            return self._handle_game_over()

        # Ensure it's learning agent's turn
        if self._env.agent_selection != self._learning_agent:
            self._play_opponent_turns()
            if self._env.game.game_over or self._is_agent_done(self._learning_agent):
                return self._handle_game_over()

        # Execute learning agent's action
        self._safe_step(self._learning_agent, action)

        # Play opponent turns
        self._play_opponent_turns()

        # Check termination
        terminated = self._env.game.game_over or self._is_agent_done(self._learning_agent)
        truncated = self._env.truncations.get(self._learning_agent, False)

        # Calculate reward
        reward = self._calculate_reward(terminated)
        self.episode_reward += reward

        obs = self._get_observation()
        info = self._get_info()

        if terminated or truncated:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.episode_length,
            }

        return obs, reward, terminated, truncated, info

    def _play_opponent_turns(self) -> None:
        """Play all opponent turns until it's learning agent's turn or game over."""
        max_iters = 1000
        iters = 0

        while iters < max_iters:
            # Check exit conditions
            if self._env.game is None or self._env.game.game_over:
                break
            if self._env.agent_selection == self._learning_agent:
                break
            if all(self._env.terminations.get(a, True) for a in self._env.possible_agents):
                break

            current_agent = self._env.agent_selection

            # Check if current agent is done
            if self._is_agent_done(current_agent):
                self._safe_step(current_agent, None)
                iters += 1
                continue

            # Get action for opponent
            action = self._get_opponent_action(current_agent)
            self._safe_step(current_agent, action)

            iters += 1

    def _get_opponent_action(self, agent: str) -> int:
        """Get action for an opponent agent."""
        # Get observation and mask
        obs_dict = self._env.observe(agent)
        obs = flatten_observation(obs_dict)

        info = self._env.infos.get(agent, {})
        mask = info.get("action_mask", np.ones(GAMEPLAY_ACTION_SPACE_SIZE, dtype=np.bool_))

        # Use self-play policy if available
        if self._opponent_policy is not None and self.opponent_type == "self":
            return self._opponent_policy(obs, mask)

        # Otherwise use agent-based opponents
        if agent in self._opponents:
            opponent = self._opponents[agent]
            return opponent.choose_action(None, mask, self._env.game)

        # Fallback: random valid action
        valid = np.where(mask)[0]
        return int(np.random.choice(valid)) if len(valid) > 0 else 0

    def _safe_step(self, agent: str, action: int | None) -> None:
        """Safely step the environment, handling edge cases."""
        # Ensure agent is in tracking dicts
        if agent not in self._env.terminations:
            self._env.terminations[agent] = True
        if agent not in self._env.truncations:
            self._env.truncations[agent] = False
        if agent not in self._env._cumulative_rewards:
            self._env._cumulative_rewards[agent] = 0.0
        if agent not in self._env.rewards:
            self._env.rewards[agent] = 0.0
        if agent not in self._env.infos:
            self._env.infos[agent] = {}

        # Check if current agent_selection is terminated
        current = self._env.agent_selection
        if current not in self._env.terminations:
            self._env.terminations[current] = True
        if current not in self._env.agents:
            self._env.agents.append(current)

        # If current agent is terminated, must pass None
        if self._env.terminations.get(current, True):
            self._env.step(None)
        elif action is None:
            self._env.step(None)
        else:
            self._env.step(action)

    def _is_agent_done(self, agent: str) -> bool:
        """Check if an agent is terminated or truncated."""
        return (
            self._env.terminations.get(agent, True) or
            self._env.truncations.get(agent, False)
        )

    def _calculate_reward(self, game_over: bool) -> float:
        """Calculate reward for the learning agent."""
        if not game_over:
            return 0.0

        if self._env.game is None:
            return 0.0

        winner = self._env.game.winner
        if winner == self._learning_agent_id:
            return 1.0  # Win
        elif winner is not None:
            return -1.0  # Loss
        else:
            # Truncation - reward based on relative net worth
            from monopoly_engine import calculate_net_worth
            my_worth = calculate_net_worth(
                self._env.game.players[self._learning_agent_id],
                self._env.game.property_manager,
            )
            # Compare to average opponent worth
            opp_worths = []
            for i in range(self.num_players):
                if i != self._learning_agent_id and not self._env.game.players[i].bankrupt:
                    opp_worths.append(calculate_net_worth(
                        self._env.game.players[i],
                        self._env.game.property_manager,
                    ))
            if opp_worths:
                avg_opp = sum(opp_worths) / len(opp_worths)
                # Normalize: positive if ahead, negative if behind
                if avg_opp > 0:
                    return 0.5 * (my_worth - avg_opp) / avg_opp
            return 0.0

    def _get_observation(self) -> NDArray[np.float32]:
        """Get flattened observation for learning agent."""
        if self._env.game is None or self._env.game.game_over:
            return np.zeros(self._obs_size, dtype=np.float32)

        obs_dict = self._env.observe(self._learning_agent)
        return flatten_observation(obs_dict)

    def _get_info(self) -> dict[str, Any]:
        """Get info dict with action mask."""
        if self._env.game is None or self._env.game.game_over:
            return {"action_mask": np.zeros(GAMEPLAY_ACTION_SPACE_SIZE, dtype=np.bool_)}

        info = self._env.infos.get(self._learning_agent, {})
        mask = info.get("action_mask", np.ones(GAMEPLAY_ACTION_SPACE_SIZE, dtype=np.bool_))
        return {"action_mask": mask}

    def _handle_game_over(self) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Handle game over state."""
        obs = self._get_observation()
        reward = self._calculate_reward(True)
        self.episode_reward += reward
        info = self._get_info()
        info["episode"] = {
            "r": self.episode_reward,
            "l": self.episode_length,
        }
        return obs, reward, True, False, info

    def action_masks(self) -> NDArray[np.bool_]:
        """Get action mask for MaskablePPO."""
        info = self._get_info()
        return info["action_mask"]

    def render(self) -> str | None:
        return self._env.render()

    def close(self) -> None:
        self._env.close()


class SelfPlayTrainer:
    """Trainer for self-play Monopoly using MaskablePPO."""

    def __init__(self, config: SelfPlayConfig):
        self.config = config
        self.metrics = TrainingMetrics()
        self._model = None
        self._vec_env = None

    def train(self) -> Any:
        """Run training and return the trained model."""
        try:
            from sb3_contrib import MaskablePPO
            from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
            from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
        except ImportError as e:
            raise ImportError(
                f"Training requires sb3-contrib: {e}\n"
                "Install with: uv add sb3-contrib"
            )

        config = self.config
        save_path = Path(config.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if config.verbose:
            print(f"Creating {config.num_envs} parallel environments...")
            print(f"Opponent type: {config.opponent_type}")

        # Create vectorized environment
        def make_env(rank: int) -> Callable[[], SelfPlayEnv]:
            def _init() -> SelfPlayEnv:
                env = SelfPlayEnv(
                    num_players=config.num_players,
                    max_turns=config.max_turns,
                    opponent_type=config.opponent_type,
                )
                env.reset(seed=config.seed + rank)
                return env
            return _init

        self._vec_env = DummyVecEnv([make_env(i) for i in range(config.num_envs)])

        if config.verbose:
            print("Creating MaskablePPO model...")

        self._model = MaskablePPO(
            "MlpPolicy",
            self._vec_env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            verbose=0,
            tensorboard_log=str(save_path / "tensorboard"),
            seed=config.seed,
        )

        # Create callback for evaluation and self-play updates
        callback = SelfPlayCallback(
            trainer=self,
            eval_freq=config.eval_freq,
            eval_episodes=config.eval_episodes,
            save_freq=config.save_freq,
            save_path=save_path,
            update_opponent_freq=config.update_opponent_freq,
            verbose=config.verbose,
        )

        if config.verbose:
            print(f"\nStarting training for {config.total_timesteps:,} timesteps...")
            print(f"TensorBoard: tensorboard --logdir {save_path / 'tensorboard'}")
            print()

        start_time = time.time()
        self._model.learn(
            total_timesteps=config.total_timesteps,
            callback=callback,
            progress_bar=config.verbose,
        )
        training_time = time.time() - start_time

        # Save final model
        final_path = save_path / "final_model"
        self._model.save(str(final_path))

        if config.verbose:
            print(f"\n{'='*60}")
            print("TRAINING COMPLETE")
            print(f"{'='*60}")
            print(f"Training time: {training_time/60:.1f} minutes")
            print(f"Final model: {final_path}")
            print(f"Games completed: {self.metrics.games_completed}")
            print(f"Win rate: {self.metrics.win_rate:.1%}")
            if self.metrics.eval_results:
                last_eval = self.metrics.eval_results[-1]
                print(f"Last eval vs random: {last_eval.get('vs_random', 0):.1%}")
                print(f"Last eval vs rule_based: {last_eval.get('vs_rule_based', 0):.1%}")

        return self._model

    def update_opponent_policies(self) -> None:
        """Update opponent policies to use current model (for self-play)."""
        if self._model is None or self.config.opponent_type != "self":
            return

        def policy_fn(obs: NDArray, mask: NDArray) -> int:
            """Policy function that uses the current model."""
            action, _ = self._model.predict(
                obs.reshape(1, -1),
                action_masks=mask.reshape(1, -1),
                deterministic=False,
            )
            return int(action[0])

        # Update policy in all environments
        for env in self._vec_env.envs:
            env.set_opponent_policy(policy_fn)

    def evaluate(
        self,
        opponent_type: str,
        n_episodes: int = 50,
    ) -> float:
        """Evaluate current model against specified opponent type."""
        if self._model is None:
            return 0.0

        # Create evaluation environment with shorter games for speed
        eval_env = SelfPlayEnv(
            num_players=self.config.num_players,
            max_turns=min(self.config.max_turns, 300),  # Cap at 300 for eval speed
            opponent_type=opponent_type,
        )

        wins = 0
        for ep in range(n_episodes):
            obs, info = eval_env.reset(seed=self.config.seed + 10000 + ep)
            done = False
            steps = 0
            max_steps = 500  # Prevent infinite loops

            while not done and steps < max_steps:
                mask = eval_env.action_masks()
                action, _ = self._model.predict(
                    obs.reshape(1, -1),
                    action_masks=mask.reshape(1, -1),
                    deterministic=True,
                )
                obs, reward, terminated, truncated, info = eval_env.step(int(action[0]))
                done = terminated or truncated
                steps += 1

            if reward > 0:
                wins += 1

        eval_env.close()
        return wins / n_episodes


class SelfPlayCallback:
    """Callback for self-play training with evaluation and opponent updates."""

    def __init__(
        self,
        trainer: SelfPlayTrainer,
        eval_freq: int,
        eval_episodes: int,
        save_freq: int,
        save_path: Path,
        update_opponent_freq: int,
        verbose: bool,
    ):
        self.trainer = trainer
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.save_freq = save_freq
        self.save_path = save_path
        self.update_opponent_freq = update_opponent_freq
        self.verbose = verbose

        self.n_calls = 0
        self.num_timesteps = 0
        self.last_eval = 0
        self.last_save = 0
        self.last_opponent_update = 0
        self.start_time = time.time()

    def __call__(self, locals_dict: dict, globals_dict: dict) -> bool:
        """Called at each training step."""
        self.n_calls += 1

        # Get model from locals
        model = locals_dict.get("self")
        if model is None:
            return True

        self.num_timesteps = model.num_timesteps

        # Update opponent policies for self-play
        if (self.num_timesteps - self.last_opponent_update >= self.update_opponent_freq
            and self.trainer.config.opponent_type == "self"):
            self.trainer.update_opponent_policies()
            self.last_opponent_update = self.num_timesteps
            if self.verbose:
                print(f"[{self.num_timesteps:,}] Updated opponent policies")

        # Evaluation (less frequent to speed up training)
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            self._evaluate()
            self.last_eval = self.num_timesteps

        # Save checkpoint
        if self.num_timesteps - self.last_save >= self.save_freq:
            self._save_checkpoint()
            self.last_save = self.num_timesteps

        return True

    def _evaluate(self) -> None:
        """Run evaluation against different opponent types."""
        results = {}

        # Use fewer episodes for faster eval during training
        eval_eps = min(self.eval_episodes, 20)

        for opp_type in ["random", "rule_based"]:
            win_rate = self.trainer.evaluate(opp_type, eval_eps)
            results[f"vs_{opp_type}"] = win_rate

        self.trainer.metrics.eval_results.append(results)

        if self.verbose:
            import sys
            elapsed = time.time() - self.start_time
            steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0
            print(f"\n{'='*60}")
            print(f"Step {self.num_timesteps:,} | {elapsed/60:.1f}min | {steps_per_sec:.0f} steps/s")
            print(f"Win rate vs random: {results['vs_random']:.1%}")
            print(f"Win rate vs rule_based: {results['vs_rule_based']:.1%}")
            print(f"{'='*60}\n")
            sys.stdout.flush()

        # Log to tensorboard
        if self.trainer._model is not None:
            for key, value in results.items():
                self.trainer._model.logger.record(f"eval/{key}", value)
            self.trainer._model.logger.dump(self.num_timesteps)

    def _save_checkpoint(self) -> None:
        """Save a checkpoint."""
        if self.trainer._model is None:
            return

        checkpoint_path = self.save_path / f"checkpoint_{self.num_timesteps}"
        self.trainer._model.save(str(checkpoint_path))

        if self.verbose:
            print(f"[{self.num_timesteps:,}] Saved checkpoint: {checkpoint_path}")


def make_selfplay_env(
    num_players: int = 4,
    max_turns: int = 500,
    opponent_type: str = "random",
    seed: int | None = None,
) -> SelfPlayEnv:
    """Factory function to create a self-play environment."""
    env = SelfPlayEnv(
        num_players=num_players,
        max_turns=max_turns,
        opponent_type=opponent_type,
    )
    if seed is not None:
        env.reset(seed=seed)
    return env
