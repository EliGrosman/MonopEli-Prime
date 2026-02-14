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
    num_envs: int = 16
    learning_rate: float = 3e-4
    lr_schedule: str = "linear"  # "linear" or "constant"
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr_min: float = 0.0  # Minimum LR floor for linear schedule
    policy_kwargs: dict[str, Any] | None = None  # Custom network architecture

    # Environment
    num_players: int = 4
    max_turns: int = 500
    reward_type: str = "sparse"  # "sparse", "dense", or "rank"
    terminal_win_reward: float = 5.0  # Dense mode terminal win reward
    terminal_loss_reward: float = -5.0  # Dense mode terminal loss reward
    worth_scale: float = 500.0  # Divisor for dense reward net worth delta

    # Self-play
    opponent_type: str = "self"  # "self", "random", "rule_based", "mixed"
    update_opponent_freq: int = 10_000  # Update opponent policy every N steps
    opponent_pool: list[str] | None = None  # Pool of opponent types to sample from
    opponent_weights: list[float] | None = None  # Sampling weights (uniform if None)

    # Evaluation
    eval_freq: int = 25_000
    eval_episodes: int = 50

    # Checkpointing
    save_freq: int = 50_000
    save_dir: str = "models/selfplay"
    checkpoint_min_win_rate: float = 0.0  # Min win rate vs random to save (0=disabled)
    collapse_detection: bool = True  # Stop training if win rate collapses to 0%
    load_model: str | None = None  # Path to pre-trained model to load

    # Vectorization
    use_subproc: bool = True  # Use SubprocVecEnv for true parallelism (auto-disabled for self-play)

    # Normalization
    normalize_env: bool = False  # Enable VecNormalize for obs/reward normalization

    # Separate value function learning rate
    vf_lr_multiplier: float = 1.0  # Value function LR = this * policy LR (1.0 = same LR)

    # Misc
    seed: int = 42
    verbose: bool = True
    diagnostic_logging: bool = True  # Enable diagnostic callbacks


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
        reward_type: str = "sparse",
        terminal_win_reward: float = 5.0,
        terminal_loss_reward: float = -5.0,
        worth_scale: float = 500.0,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.num_players = num_players
        self.max_turns = max_turns
        self.opponent_type = opponent_type
        self.reward_type = reward_type
        self.terminal_win_reward = terminal_win_reward
        self.terminal_loss_reward = terminal_loss_reward
        self.worth_scale = worth_scale
        self.render_mode = render_mode

        # Dense reward tracking
        self._prev_net_worth: int = 1500
        self._prev_num_properties: int = 0

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
        self._prev_net_worth = 1500
        self._prev_num_properties = 0

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
        """Calculate reward for the learning agent.

        Supports sparse (win/loss only) and dense (per-step shaping) modes.
        Dense rewards give the agent per-step signal about net worth and
        property acquisition, which is critical when training against strong
        opponents where wins are rare.
        """
        if self._env.game is None:
            return 0.0

        from monopoly_engine import calculate_net_worth

        player = self._env.game.players[self._learning_agent_id]
        pm = self._env.game.property_manager

        if game_over:
            winner = self._env.game.winner
            if winner == self._learning_agent_id:
                return self.terminal_win_reward if self.reward_type in ("dense", "rank") else 1.0
            elif winner is not None:
                return self.terminal_loss_reward if self.reward_type in ("dense", "rank") else -1.0
            else:
                # Truncation - reward based on relative net worth
                my_worth = calculate_net_worth(player, pm)
                opp_worths = []
                for i in range(self.num_players):
                    if i != self._learning_agent_id and not self._env.game.players[i].bankrupt:
                        opp_worths.append(calculate_net_worth(
                            self._env.game.players[i], pm,
                        ))
                if opp_worths:
                    avg_opp = sum(opp_worths) / len(opp_worths)
                    if avg_opp > 0:
                        return 0.5 * (my_worth - avg_opp) / avg_opp
                return 0.0

        # For sparse reward, no intermediate signal
        if self.reward_type == "sparse":
            return 0.0

        # Rank-based reward: relative position among alive players
        if self.reward_type == "rank":
            # Calculate all players' net worths
            worths = []
            for i in range(self.num_players):
                p = self._env.game.players[i]
                if p.bankrupt:
                    worths.append(0)
                else:
                    worths.append(calculate_net_worth(p, pm))

            my_worth = worths[self._learning_agent_id]

            # Calculate rank (0 = last, num_alive-1 = first)
            alive_worths = [w for w in worths if w > 0]
            num_alive = len(alive_worths)

            if num_alive <= 1:
                return 0.0

            # Count how many alive players I'm beating
            rank = sum(1 for w in alive_worths if my_worth > w)

            # Normalize to [-1, 1] range
            # rank=0 (last) -> -1.0, rank=num_alive-1 (first) -> +1.0
            normalized_rank = (2.0 * rank / (num_alive - 1)) - 1.0

            # Scale down to per-step reward magnitude (~0.01)
            reward = normalized_rank * 0.01

            # Bonus for property acquisition (player's own achievement)
            current_props = len(pm.get_owned_by(self._learning_agent_id))
            if current_props > self._prev_num_properties:
                reward += 0.05 * (current_props - self._prev_num_properties)
            self._prev_num_properties = current_props

            return reward

        # Dense reward: per-step shaping
        reward = 0.0

        # Net worth progress (scaled to ~0.01 per step)
        current_worth = calculate_net_worth(player, pm)
        worth_delta = current_worth - self._prev_net_worth
        reward += worth_delta / self.worth_scale
        self._prev_net_worth = current_worth

        # Property acquisition bonus
        current_props = len(pm.get_owned_by(self._learning_agent_id))
        if current_props > self._prev_num_properties:
            reward += 0.05 * (current_props - self._prev_num_properties)
        self._prev_num_properties = current_props

        return reward

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

        # Set up opponent pool for mixed training
        if config.opponent_pool is not None and len(config.opponent_pool) > 0:
            self._opponent_pool = config.opponent_pool
            self._opponent_weights = config.opponent_weights
            if config.verbose:
                print(f"Mixed opponent training: {self._opponent_pool}")
                if self._opponent_weights:
                    print(f"  Weights: {self._opponent_weights}")
                else:
                    print(f"  Weights: uniform")
        else:
            # Single opponent (backward compatible)
            self._opponent_pool = [config.opponent_type]
            self._opponent_weights = None
            if config.verbose:
                print(f"Single opponent training: {config.opponent_type}")

    def train(self) -> Any:
        """Run training and return the trained model."""
        try:
            from sb3_contrib import MaskablePPO
            from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
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
            # Show opponent info (pool or single type)
            if config.opponent_pool is not None and len(config.opponent_pool) > 0:
                pool_str = ", ".join(config.opponent_pool)
                if len(config.opponent_pool) > 1:
                    if config.opponent_weights:
                        weights_str = ", ".join(f"{w:.1f}" for w in config.opponent_weights)
                        print(f"Opponent pool: [{pool_str}] (weights: [{weights_str}])")
                    else:
                        print(f"Opponent pool: [{pool_str}] (uniform weights)")
                else:
                    print(f"Opponent type: {pool_str}")
            else:
                print(f"Opponent type: {config.opponent_type}")
            print(f"Reward type: {config.reward_type}")
            print(f"Players: {config.num_players}, Max turns: {config.max_turns}")
            print(f"n_steps: {config.n_steps}, batch_size: {config.batch_size}, "
                  f"n_epochs: {config.n_epochs}")
            print(f"ent_coef: {config.ent_coef}, lr: {config.learning_rate}, "
                  f"lr_schedule: {config.lr_schedule}")
            if config.lr_min > 0:
                print(f"lr_min: {config.lr_min}")
            if config.vf_coef != 0.5:
                print(f"vf_coef: {config.vf_coef}")
            if config.gamma != 0.99:
                print(f"gamma: {config.gamma}")
            if config.gae_lambda != 0.95:
                print(f"gae_lambda: {config.gae_lambda}")
            if config.policy_kwargs is not None:
                print(f"policy_kwargs: {config.policy_kwargs}")
            if config.worth_scale != 500.0:
                print(f"worth_scale: {config.worth_scale}")
            if config.checkpoint_min_win_rate > 0:
                print(f"Checkpoint min win rate: {config.checkpoint_min_win_rate:.0%}")

        # Create vectorized environment
        def make_env(rank: int) -> Callable[[], SelfPlayEnv]:
            """Create a single environment instance.

            Args:
                rank: Environment index (for seeding)

            Returns:
                Callable that creates and returns a SelfPlayEnv
            """
            def _init() -> SelfPlayEnv:
                # Sample opponent type from pool
                if len(self._opponent_pool) > 1:
                    opponent_type = np.random.choice(
                        self._opponent_pool,
                        p=self._opponent_weights,  # None = uniform
                    )
                else:
                    opponent_type = self._opponent_pool[0]

                env = SelfPlayEnv(
                    num_players=config.num_players,
                    max_turns=config.max_turns,
                    opponent_type=opponent_type,  # Sampled!
                    reward_type=config.reward_type,
                    terminal_win_reward=config.terminal_win_reward,
                    terminal_loss_reward=config.terminal_loss_reward,
                    worth_scale=config.worth_scale,
                )
                env.reset(seed=config.seed + rank)
                return env
            return _init

        # Use SubprocVecEnv for true parallelism when possible.
        # Self-play mode needs direct env access for policy updates, so it
        # must use DummyVecEnv. All other modes benefit from SubprocVecEnv.
        use_subproc = (
            config.use_subproc
            and config.num_envs > 1
            and config.opponent_type != "self"
        )
        env_fns = [make_env(i) for i in range(config.num_envs)]
        if use_subproc:
            self._vec_env = SubprocVecEnv(env_fns)
            if config.verbose:
                print(f"Using SubprocVecEnv ({config.num_envs} processes)")
        else:
            self._vec_env = DummyVecEnv(env_fns)
            if config.verbose:
                reason = "self-play" if config.opponent_type == "self" else "single env or disabled"
                print(f"Using DummyVecEnv ({reason})")

        # Add VecNormalize for observation and return normalization
        if config.normalize_env:
            self._vec_env = VecNormalize(
                self._vec_env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
                gamma=config.gamma,
            )
            if config.verbose:
                print("VecNormalize enabled (obs + reward normalization)")

        # Resolve learning rate schedule
        if config.lr_schedule == "linear":
            _base_lr = config.learning_rate
            _lr_min = config.lr_min
            def lr_schedule_fn(progress_remaining: float) -> float:
                return max(progress_remaining * _base_lr, _lr_min)
            lr_value: float | Callable[[float], float] = lr_schedule_fn
        elif config.lr_schedule == "cosine":
            _base_lr = config.learning_rate
            _lr_min = config.lr_min
            import math
            def lr_schedule_fn_cosine(progress_remaining: float) -> float:
                return _lr_min + 0.5 * (_base_lr - _lr_min) * (1 + math.cos(math.pi * (1 - progress_remaining)))
            lr_value = lr_schedule_fn_cosine
        else:
            lr_value = config.learning_rate

        if config.verbose:
            print("Creating MaskablePPO model...")

        self._model = MaskablePPO(
            "MlpPolicy",
            self._vec_env,
            learning_rate=lr_value,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            policy_kwargs=config.policy_kwargs,
            verbose=0,
            tensorboard_log=str(save_path / "tensorboard"),
            seed=config.seed,
        )

        # Load pre-trained model if specified
        if config.load_model:
            self._load_pretrained(Path(config.load_model))

        # Create main self-play callback
        main_callback = SelfPlayCallback(
            trainer=self,
            eval_freq=config.eval_freq,
            eval_episodes=config.eval_episodes,
            save_freq=config.save_freq,
            save_path=save_path,
            update_opponent_freq=config.update_opponent_freq,
            verbose=config.verbose,
        )

        # Create callback list (using SB3's CallbackList for proper lifecycle)
        from stable_baselines3.common.callbacks import CallbackList

        callbacks: list[Any] = [CallableCallbackAdapter(main_callback)]

        # Add diagnostic logging if enabled
        if config.diagnostic_logging:
            from training.callbacks import DiagnosticCallback
            callbacks.append(DiagnosticCallback(
                log_freq=1000,  # Log every 1000 steps
                verbose=1 if config.verbose else 0,
            ))
            if config.verbose:
                print("Diagnostic logging enabled (every 1000 steps)")

        # Add separate value function learning rate if configured
        if config.vf_lr_multiplier != 1.0:
            from training.callbacks import SeparateValueLRCallback
            callbacks.append(SeparateValueLRCallback(
                vf_lr_multiplier=config.vf_lr_multiplier,
                verbose=1 if config.verbose else 0,
            ))
            if config.verbose:
                print(f"Separate value function LR: {config.vf_lr_multiplier}x policy LR")

        callback = CallbackList(callbacks)

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

        # Save VecNormalize statistics alongside the model
        if isinstance(self._vec_env, VecNormalize):
            self._vec_env.save(str(save_path / "vecnormalize.pkl"))

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

    def _load_pretrained(self, path: Path) -> None:
        """Load a pre-trained model, with weight surgery for cross-player-count transfer.

        First attempts direct load (same observation dimensions). If that fails
        due to dimension mismatch (e.g., 2P model loaded into 4P env), performs
        weight surgery on the first linear layers to map observation features.

        Observation layout (from observation.py):
        - [0:33] = player_state (33 dims, fixed)
        - [33:33+N*34] = opponent_states (N = num_players-1, 34 features each)
        - [33+N*34:] = board(140) + game(4) + action_mask(149) = 293 dims (fixed)
        """
        from sb3_contrib import MaskablePPO

        if self._model is None or self._vec_env is None:
            return

        # Resolve path (handle both with and without .zip extension)
        model_path = path
        if not model_path.exists() and not model_path.with_suffix(".zip").exists():
            print(f"Warning: Model not found at {path}, skipping load")
            return
        # MaskablePPO.load handles .zip extension automatically
        load_str = str(model_path)

        if self.config.verbose:
            print(f"Loading pre-trained model from {path}...")

        try:
            # Try direct load (same observation dimensions)
            self._model = MaskablePPO.load(load_str, env=self._vec_env)
            if self.config.verbose:
                print("Direct model load successful (matching dimensions)")
            # Load VecNormalize statistics if they exist
            self._load_vecnormalize_stats(path)
            return
        except Exception as direct_err:
            if self.config.verbose:
                print(f"Direct load failed ({direct_err}), attempting weight surgery...")

        # Weight surgery for cross-player-count transfer
        try:
            import torch

            source_model = MaskablePPO.load(load_str)
            source_params = source_model.policy.state_dict()
            target_params = self._model.policy.state_dict()

            # Get observation sizes
            source_obs = source_model.observation_space.shape[0]
            target_obs = self._model.observation_space.shape[0]

            if self.config.verbose:
                print(f"Source obs size: {source_obs}, Target obs size: {target_obs}")

            # Observation layout constants
            player_state_size = 33  # Fixed
            tail_size = 293  # board(140) + game(4) + action_mask(149), fixed
            source_opp_size = source_obs - player_state_size - tail_size
            target_opp_size = target_obs - player_state_size - tail_size

            if self.config.verbose:
                print(f"Source opponent dims: {source_opp_size}, "
                      f"Target opponent dims: {target_opp_size}")

            # Process first linear layers that take observations as input
            for layer_name in list(source_params.keys()):
                source_tensor = source_params[layer_name]
                if layer_name not in target_params:
                    continue
                target_tensor = target_params[layer_name]

                # First linear layer weights have shape (out_features, in_features)
                # where in_features == observation size
                if (source_tensor.dim() == 2
                        and source_tensor.shape[1] == source_obs
                        and target_tensor.shape[1] == target_obs):
                    # Build new weight matrix
                    new_weight = torch.zeros_like(target_tensor)

                    # Copy player_state cols (0:33)
                    new_weight[:, :player_state_size] = \
                        source_tensor[:, :player_state_size]

                    # Copy opponent cols (as many as fit)
                    copy_opp = min(source_opp_size, target_opp_size)
                    new_weight[:, player_state_size:player_state_size + copy_opp] = \
                        source_tensor[:, player_state_size:player_state_size + copy_opp]
                    # New opponent cols (if target has more) are already zero-initialized

                    # Copy tail cols (board + game + action_mask)
                    new_weight[:, -tail_size:] = source_tensor[:, -tail_size:]

                    target_params[layer_name] = new_weight
                    if self.config.verbose:
                        print(f"Weight surgery on {layer_name}: "
                              f"{source_tensor.shape} -> {target_tensor.shape}")
                elif source_tensor.shape == target_tensor.shape:
                    # Same shape: copy directly (hidden layers, biases, output heads)
                    target_params[layer_name] = source_tensor

            self._model.policy.load_state_dict(target_params)
            if self.config.verbose:
                print("Weight surgery complete, model loaded successfully")

        except Exception as e:
            print(f"Warning: Model loading failed ({e}), starting from scratch")

    def _load_vecnormalize_stats(self, model_path: Path) -> None:
        """Load VecNormalize statistics if they exist alongside the model."""
        from stable_baselines3.common.vec_env import VecNormalize
        if not isinstance(self._vec_env, VecNormalize):
            return

        vecnorm_path = model_path.parent / "vecnormalize.pkl"
        if not vecnorm_path.exists():
            return

        try:
            saved_env = VecNormalize.load(str(vecnorm_path), self._vec_env.venv)
            self._vec_env.obs_rms = saved_env.obs_rms
            self._vec_env.ret_rms = saved_env.ret_rms
            if self.config.verbose:
                print(f"Loaded VecNormalize stats from {vecnorm_path}")
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Could not load VecNormalize stats: {e}")

    def evaluate(
        self,
        opponent_type: str,
        n_episodes: int = 50,
    ) -> float:
        """Evaluate current model against specified opponent type."""
        if self._model is None:
            return 0.0

        from stable_baselines3.common.vec_env import VecNormalize

        # Create evaluation environment with shorter games for speed
        eval_env = SelfPlayEnv(
            num_players=self.config.num_players,
            max_turns=min(self.config.max_turns, 300),  # Cap at 300 for eval speed
            opponent_type=opponent_type,
            terminal_win_reward=self.config.terminal_win_reward,
            terminal_loss_reward=self.config.terminal_loss_reward,
        )

        # If training uses VecNormalize, normalize eval observations using
        # the training env's running statistics (but don't update them)
        normalize_obs = isinstance(self._vec_env, VecNormalize)

        wins = 0
        for ep in range(n_episodes):
            obs, info = eval_env.reset(seed=self.config.seed + 10000 + ep)
            done = False
            steps = 0
            max_steps = 500  # Prevent infinite loops

            while not done and steps < max_steps:
                mask = eval_env.action_masks()

                # Normalize observation using training env's stats
                if normalize_obs:
                    obs = self._vec_env.normalize_obs(obs)

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
    """Callback for self-play training with evaluation and opponent updates.

    This callback is a plain callable (not a BaseCallback) because it needs
    to access the trainer object directly, which SB3's callback system doesn't
    support natively. It will be wrapped in a BaseCallbackAdapter.
    """

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

        self.num_timesteps = 0
        self.last_eval = 0
        self.last_save = 0
        self.last_opponent_update = 0
        self.start_time = time.time()
        self.consecutive_zero_evals: int = 0

    def __call__(self, locals_dict: dict, globals_dict: dict) -> bool:
        """Called at each training step."""
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

            # Collapse detection: check if win rate has dropped to 0%
            if self.trainer.metrics.eval_results:
                last_eval = self.trainer.metrics.eval_results[-1]
                vs_random = last_eval.get("vs_random", 0.0)
                if vs_random == 0.0 and self.num_timesteps >= 100_000:
                    self.consecutive_zero_evals += 1
                else:
                    self.consecutive_zero_evals = 0

                if (self.consecutive_zero_evals >= 3
                        and self.trainer.config.collapse_detection):
                    print(f"\n{'!'*60}")
                    print(f"COLLAPSE DETECTED at step {self.num_timesteps:,}")
                    print(f"Win rate vs random has been 0% for "
                          f"{self.consecutive_zero_evals} consecutive evals")
                    best_path = self.save_path / "best_model.zip"
                    if best_path.exists():
                        print(f"Loading best model from {best_path}")
                        try:
                            self.trainer._model = type(self.trainer._model).load(
                                str(self.save_path / "best_model"),
                                env=self.trainer._vec_env,
                            )
                        except ValueError:
                            # Optimizer state mismatch (e.g., separate value LR
                            # creates 2 param groups but load() expects 1).
                            # Fall back to loading just policy network weights.
                            model_cls = type(self.trainer._model)
                            _, params, _ = model_cls._load_from_file(
                                str(self.save_path / "best_model"),
                                device=self.trainer._model.device,
                            )
                            if "policy" in params:
                                self.trainer._model.policy.load_state_dict(
                                    params["policy"]
                                )
                                print("Restored policy weights (optimizer reset)")
                    print(f"Stopping training early.")
                    print(f"{'!'*60}\n")
                    return False

        # Save checkpoint
        if self.num_timesteps - self.last_save >= self.save_freq:
            self._save_checkpoint()
            self.last_save = self.num_timesteps

        return True

    def _evaluate(self) -> None:
        """Run evaluation against different opponent types."""
        results = {}

        eval_eps = self.eval_episodes

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
        """Save a checkpoint, gated on minimum win rate if configured."""
        if self.trainer._model is None:
            return

        min_wr = self.trainer.config.checkpoint_min_win_rate
        if min_wr > 0 and self.trainer.metrics.eval_results:
            last_eval = self.trainer.metrics.eval_results[-1]
            vs_random = last_eval.get("vs_random", 0.0)
            if vs_random < min_wr:
                if self.verbose:
                    print(f"[{self.num_timesteps:,}] Checkpoint skipped "
                          f"(win rate {vs_random:.0%} < {min_wr:.0%})")
                return

        checkpoint_path = self.save_path / f"checkpoint_{self.num_timesteps}"
        self.trainer._model.save(str(checkpoint_path))

        # Save VecNormalize stats with each checkpoint
        from stable_baselines3.common.vec_env import VecNormalize
        if isinstance(self.trainer._vec_env, VecNormalize):
            self.trainer._vec_env.save(
                str(self.save_path / f"vecnormalize_{self.num_timesteps}.pkl")
            )

        # Track best model
        if self.trainer.metrics.eval_results:
            last_eval = self.trainer.metrics.eval_results[-1]
            vs_random = last_eval.get("vs_random", 0.0)
            if not hasattr(self, "_best_win_rate"):
                self._best_win_rate = 0.0
            if vs_random > self._best_win_rate:
                self._best_win_rate = vs_random
                best_path = self.save_path / "best_model"
                self.trainer._model.save(str(best_path))
                if isinstance(self.trainer._vec_env, VecNormalize):
                    self.trainer._vec_env.save(
                        str(self.save_path / "best_vecnormalize.pkl")
                    )
                if self.verbose:
                    print(f"[{self.num_timesteps:,}] New best model! "
                          f"({vs_random:.0%} vs random)")

        if self.verbose:
            print(f"[{self.num_timesteps:,}] Saved checkpoint: {checkpoint_path}")


def make_selfplay_env(
    num_players: int = 4,
    max_turns: int = 500,
    opponent_type: str = "random",
    reward_type: str = "sparse",
    terminal_win_reward: float = 5.0,
    terminal_loss_reward: float = -5.0,
    worth_scale: float = 500.0,
    seed: int | None = None,
) -> SelfPlayEnv:
    """Factory function to create a self-play environment."""
    env = SelfPlayEnv(
        num_players=num_players,
        max_turns=max_turns,
        opponent_type=opponent_type,
        reward_type=reward_type,
        terminal_win_reward=terminal_win_reward,
        terminal_loss_reward=terminal_loss_reward,
        worth_scale=worth_scale,
    )
    if seed is not None:
        env.reset(seed=seed)
    return env


class CallableCallbackAdapter:
    """Adapter that wraps a plain callable callback into SB3's BaseCallback protocol.

    This allows legacy callable callbacks (like SelfPlayCallback) to work with
    SB3's CallbackList, which properly handles the callback lifecycle including
    init_callback, on_training_start, on_step, etc.
    """

    def __init__(self, callable_callback: Any):
        """Initialize with a callable callback.

        Args:
            callable_callback: A callable that takes (locals_dict, globals_dict)
                              and returns bool to continue/stop training.
        """
        self.callback = callable_callback
        self.locals: dict[str, Any] = {}
        self.globals: dict[str, Any] = {}
        self.n_calls = 0
        self.model = None
        self.logger = None

    def init_callback(self, model: Any) -> None:
        """Initialize the callback with the model.

        The callable callback doesn't need model initialization, so we just
        store the reference for potential use.
        """
        self.model = model
        self.logger = model.logger if hasattr(model, 'logger') else None

    def on_training_start(self, locals_dict: dict, globals_dict: dict) -> None:
        """Called at the start of training."""
        self.locals = locals_dict
        self.globals = globals_dict

    def on_rollout_start(self) -> None:
        """Called at the start of each rollout."""
        pass

    def on_step(self) -> bool:
        """Called at each training step.

        Returns True to continue training, False to stop.
        """
        self.n_calls += 1
        return self.callback(self.locals, self.globals)

    def on_training_end(self) -> None:
        """Called at the end of training."""
        pass

    def on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        pass

    def update_locals(self, locals_dict: dict[str, Any]) -> None:
        """Update locals dict (called by SB3 during training)."""
        self.locals = locals_dict
