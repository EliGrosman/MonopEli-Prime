"""One learner decision through the next learner decision, shared by PPO/self-play."""

from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .action_space import GAMEPLAY_ACTION_SPACE_SIZE
from .env import MonopolyEnv
from .observation import flatten_observation, get_flat_observation_size


class SingleAgentMonopolyEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        num_players: int = 4,
        opponent_type: str = "random",
        max_turns: int = 1000,
        reward_type: str = "sparse",
        flatten_obs: bool = True,
        render_mode: str | None = None,
        seed: int | None = None,
        learner_seat: int = 0,
    ):
        super().__init__()
        if not 0 <= learner_seat < num_players:
            raise ValueError("Invalid learner seat")
        self.num_players, self.opponent_type = num_players, opponent_type
        self.max_turns, self.reward_type = max_turns, reward_type
        self.flatten_obs, self.render_mode = flatten_obs, render_mode
        self.learner_seat, self._seed = learner_seat, seed
        self._agent_id = f"player_{learner_seat}"
        self._env = MonopolyEnv(
            num_players, max_turns, reward_type, render_mode, cutoff_player=learner_seat
        )
        self.action_space = spaces.Discrete(GAMEPLAY_ACTION_SPACE_SIZE)
        self.observation_space = (
            spaces.Box(0, 1, (get_flat_observation_size(num_players),), dtype=np.float32)
            if flatten_obs
            else self._env.observation_space(self._agent_id)
        )
        self._opponent_policy: Callable | None = None
        self._stream = np.random.default_rng(seed)
        self._done = False

    def set_opponent_policy(self, policy_fn: Callable | None) -> None:
        self._opponent_policy = policy_fn

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._stream = np.random.default_rng(seed)
        episode = int(self._stream.integers(0, 2**32))
        seeds = np.random.SeedSequence(episode).spawn(self.num_players + 1)
        self._env.reset(seed=int(seeds[0].generate_state(1)[0]))
        from agents import AggressiveAgent, ConservativeAgent, RandomAgent, RuleBasedAgent

        types = {
            "random": RandomAgent,
            "self": RandomAgent,
            "rule_based": RuleBasedAgent,
            "aggressive": AggressiveAgent,
            "conservative": ConservativeAgent,
        }
        if self.opponent_type not in types:
            raise ValueError(f"Unknown opponent {self.opponent_type}")
        cls = types[self.opponent_type]
        self._opponents = {
            f"player_{i}": (
                cls(i, seed=int(seeds[i + 1].generate_state(1)[0]))
                if cls is RandomAgent
                else cls(i)
            )
            for i in range(self.num_players)
            if i != self.learner_seat
        }
        self.episode_reward, self.episode_length = 0.0, 0
        self._done = False
        from monopoly_engine.progress import ProgressGuard

        self._guard = ProgressGuard()
        self._reset_reward = self._advance()
        self._pending_terminal = self._flags()[0] or self._flags()[1]
        return self._get_observation(), {
            **self._get_info(),
            "pending_terminal": self._pending_terminal,
        }

    def _flags(self):
        return (self._env.terminations[self._agent_id], self._env.truncations[self._agent_id])

    def _advance(self) -> float:
        reward = 0.0
        for _ in range(10000):
            if any(self._flags()) or self._env.agent_selection == self._agent_id:
                return reward
            agent = self._env.agent_selection
            obs, _, terminated, truncated, info = self._env.last()
            if terminated or truncated:
                self._env.step(None)
                continue
            if self.opponent_type == "self" and self._opponent_policy is not None:
                action = int(self._opponent_policy(flatten_observation(obs), info["action_mask"]))
            else:
                action = self._opponents[agent].choose_action(
                    obs, info["action_mask"], self._env.game
                )
            self._guard.check(self._env.game)
            self._env.step(action)
            reward += self._env.rewards.get(self._agent_id, 0.0)
        raise RuntimeError("Stalled opponent interval; trajectory invalid")

    def step(self, action: int):
        if self._done:
            raise RuntimeError("Episode finished; reset before stepping again")
        if self._pending_terminal:
            reward = self._reset_reward
            self._pending_terminal = False
        else:
            self._guard.check(self._env.game)
            self._env.step(action)
            reward = self._env.rewards[self._agent_id] + self._advance()
        terminated, truncated = self._flags()
        self._done = terminated or truncated
        self.episode_length += 1
        self.episode_reward += reward
        info = self._get_info()
        if self._done:
            info["episode"] = {"r": self.episode_reward, "l": self.episode_length}
        return self._get_observation(), float(reward), terminated, truncated, info

    def _get_observation(self):
        obs = self._env.observe(self._agent_id)
        return flatten_observation(obs) if self.flatten_obs else obs

    def _get_info(self) -> dict[str, Any]:
        return dict(self._env.infos[self._agent_id])

    def action_masks(self):
        return self._get_info()["action_mask"].copy()

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
