"""Single-agent Gymnasium wrapper for Monopoly.

This module provides a Gymnasium-compatible environment that wraps the
multi-agent PettingZoo environment, allowing training with single-agent
RL libraries like Stable-Baselines3.
"""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from monopoly_gym import ACTION_SPACE_SIZE, MonopolyEnv
from monopoly_gym.observation import flatten_observation, get_flat_observation_size


class SingleAgentMonopolyEnv(gym.Env[NDArray[np.float32], int]):
    """Single-agent Gymnasium wrapper for Monopoly.

    This environment wraps the multi-agent MonopolyEnv to create a single-agent
    training environment. The learning agent is always player 0, while other
    players are controlled by opponent agents.

    Supports action masking through the info dict for use with MaskablePPO.

    Attributes:
        action_space: Discrete action space (149 actions in Phase 2)
        observation_space: Flattened observation space for compatibility
    """

    metadata = {
        "render_modes": ["human", "ansi"],
    }

    def __init__(
        self,
        num_players: int = 4,
        opponent_type: str = "random",
        max_turns: int = 1000,
        reward_type: str = "sparse",
        flatten_obs: bool = True,
        render_mode: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize single-agent Monopoly environment.

        Args:
            num_players: Total players including the learning agent (2-4)
            opponent_type: Type of opponent agents ("random", "rule_based")
            max_turns: Maximum turns before truncation
            reward_type: "sparse" or "dense"
            flatten_obs: Whether to flatten observations (required for MLP policies)
            render_mode: "human", "ansi", or None
            seed: Random seed
        """
        super().__init__()

        if not 2 <= num_players <= 4:
            raise ValueError(f"num_players must be 2-4, got {num_players}")

        self.num_players = num_players
        self.opponent_type = opponent_type
        self.max_turns = max_turns
        self.reward_type = reward_type
        self.flatten_obs = flatten_obs
        self.render_mode = render_mode
        self._seed = seed

        # Create underlying multi-agent environment
        self._env = MonopolyEnv(
            num_players=num_players,
            max_turns=max_turns,
            reward_type=reward_type,
            render_mode=render_mode,
        )

        # Create opponent agents (lazy import to avoid circular dependencies)
        self._opponents = self._create_opponents()

        # Define spaces
        self.action_space: spaces.Space[int] = spaces.Discrete(ACTION_SPACE_SIZE)

        if flatten_obs:
            flat_size = get_flat_observation_size(num_players)
            self.observation_space: spaces.Space[Any] = spaces.Box(
                low=-1.0, high=1.0, shape=(flat_size,), dtype=np.float32
            )
        else:
            self.observation_space = self._env._observation_space

        # Agent ID for the learning agent
        self._agent_id = "player_0"

    def _create_opponents(self) -> dict[str, Any]:
        """Create opponent agents based on opponent_type.

        Returns:
            Dictionary mapping agent IDs to agent instances.
        """
        # Import agents lazily to avoid circular imports
        from agents import RandomAgent

        opponents: dict[str, Any] = {}
        for i in range(1, self.num_players):
            agent_id = f"player_{i}"
            # Currently only RandomAgent is fully implemented
            # Other agent types fall back to RandomAgent
            if self.opponent_type == "random":
                opponents[agent_id] = RandomAgent(i, seed=self._seed)
            else:
                # For now, all other types default to random
                # TODO: Add RuleBasedAgent, AggressiveAgent, ConservativeAgent
                opponents[agent_id] = RandomAgent(i, seed=self._seed)
        return opponents

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        actual_seed = seed if seed is not None else self._seed
        self._env.reset(seed=actual_seed)

        # Reset opponent agents
        for agent in self._opponents.values():
            agent.reset()

        # Play opponent turns until it's player_0's turn
        self._play_until_agent_turn()

        # Get observation for learning agent
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Action index for the learning agent

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Check if the learning agent is already terminated/truncated
        agent_terminated = self._env.terminations.get(self._agent_id, False)
        agent_truncated = self._env.truncations.get(self._agent_id, False)

        if agent_terminated or agent_truncated:
            # Agent is done - pass None and return terminal state
            self._env.step(None)
            obs = self._get_observation()
            info = self._get_info()
            # Calculate final reward
            reward = 0.0
            if self._env.game is not None and self._env.game.game_over:
                if self._env.game.winner == 0:
                    reward = 1.0 if self.reward_type == "sparse" else 10.0
                else:
                    reward = -1.0 if self.reward_type == "sparse" else -1.0
            return obs, reward, agent_terminated, agent_truncated, info

        # Execute learning agent's action
        self._env.step(action)

        # Get reward for the learning agent
        _, reward, terminated, truncated, _ = self._env.last()

        # If not done, play opponent turns
        if not (terminated or truncated):
            self._play_until_agent_turn()

        # Check termination status again after opponent turns
        terminated = self._env.terminations.get(self._agent_id, False)
        truncated = self._env.truncations.get(self._agent_id, False)

        # Check if game ended during opponent turns
        if self._env.game is not None and self._env.game.game_over:
            terminated = True
            # Recalculate reward if game ended
            if self._env.game.winner == 0:
                reward = 1.0 if self.reward_type == "sparse" else 10.0
            else:
                reward = -1.0 if self.reward_type == "sparse" else -1.0

        obs = self._get_observation()
        info = self._get_info()

        return obs, float(reward), terminated, truncated, info

    def _play_until_agent_turn(self) -> None:
        """Play opponent turns until it's the learning agent's turn."""
        max_iterations = 1000  # Prevent infinite loops
        iterations = 0

        while (
            self._env.agent_selection != self._agent_id
            and not all(self._env.terminations.values())
            and iterations < max_iterations
        ):
            current_agent = self._env.agent_selection

            if current_agent in self._opponents:
                opponent = self._opponents[current_agent]
                obs, _, term, trunc, info = self._env.last()

                if term or trunc:
                    self._env.step(None)
                else:
                    action_mask = info.get(
                        "action_mask", np.ones(ACTION_SPACE_SIZE, dtype=np.bool_)
                    )
                    action = opponent.choose_action(obs, action_mask, self._env.game)
                    self._env.step(action)
            else:
                break

            iterations += 1

    def _get_observation(self) -> NDArray[np.float32]:
        """Get observation for the learning agent.

        Returns:
            Flattened observation array if flatten_obs=True, else dict observation.
        """
        obs = self._env.observe(self._agent_id)

        if self.flatten_obs:
            return flatten_observation(obs)
        # For non-flattened, we still return the flattened version
        # because the observation_space is defined as Box
        # Users who want dict obs should set flatten_obs=False and
        # handle the dict space themselves
        return flatten_observation(obs)

    def _get_info(self) -> dict[str, Any]:
        """Get info dict with action mask.

        Returns:
            Info dictionary containing the action mask for MaskablePPO.
        """
        agent_info = self._env.infos.get(self._agent_id, {})
        action_mask = agent_info.get(
            "action_mask", np.ones(ACTION_SPACE_SIZE, dtype=np.bool_)
        )
        return {
            "action_mask": action_mask,
        }

    def render(self) -> str | list[str] | None:  # type: ignore[override]
        """Render the environment.

        Returns:
            String representation if render_mode is "ansi", None otherwise.
        """
        return self._env.render()

    def close(self) -> None:
        """Clean up resources."""
        self._env.close()

    def action_masks(self) -> NDArray[np.bool_]:
        """Get the current action mask.

        This method is provided for compatibility with sb3-contrib's MaskablePPO.
        The action mask indicates which actions are valid (True) or invalid (False).

        Returns:
            Boolean array of shape (149,) indicating valid actions.
        """
        agent_info = self._env.infos.get(self._agent_id, {})
        mask = agent_info.get(
            "action_mask", np.ones(ACTION_SPACE_SIZE, dtype=np.bool_)
        )
        # Ensure it's a numpy array with correct dtype
        if isinstance(mask, np.ndarray):
            return mask.astype(np.bool_)
        return np.ones(ACTION_SPACE_SIZE, dtype=np.bool_)
