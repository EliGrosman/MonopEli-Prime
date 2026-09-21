"""PettingZoo AEC adapter for the foundation-v1 engine decision contract."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from monopoly_engine import MonopolyGame

from .action_space import ACTION_VERSION, GAMEPLAY_ACTION_SPACE_SIZE, ActionEncoder
from .observation import OBSERVATION_VERSION, ObservationEncoder


class MonopolyEnv(AECEnv):
    metadata = {
        "name": "monopoly_foundation_v1",
        "render_modes": ["human", "ansi"],
        "is_parallelizable": False,
    }

    def __init__(
        self,
        num_players: int = 4,
        max_turns: int = 1000,
        reward_type: str = "sparse",
        render_mode: str | None = None,
        enable_trades: bool = False,
        trade_reward_config: Any = None,
        incremental_obs: bool = False,
        cutoff_player: int | None = None,
    ):
        super().__init__()
        if not 2 <= num_players <= 4:
            raise ValueError("num_players must be 2-4")
        if max_turns <= 0:
            raise ValueError("Expected a positive turn horizon")
        if reward_type != "sparse" or enable_trades:
            raise ValueError("foundation-v1 supports terminal-only sparse rewards and no trading")
        self.num_players, self.max_turns = num_players, max_turns
        self.reward_type, self.render_mode = reward_type, render_mode
        self.enable_trades, self.cutoff_player = False, cutoff_player
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agent_name_mapping = dict(zip(self.possible_agents, range(num_players)))
        self.obs_encoder = ObservationEncoder(num_players)
        self.action_encoder = ActionEncoder()
        self._observation_space = self.obs_encoder.get_observation_space()
        self._action_space = spaces.Discrete(GAMEPLAY_ACTION_SPACE_SIZE)
        self.game: MonopolyGame | None = None
        self._rng = np.random.default_rng()

    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_space

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_space

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.episode_seed = int(self._rng.integers(0, 2**32))
        self.game = MonopolyGame(self.num_players, seed=self.episode_seed)
        self.agents = self.possible_agents.copy()
        self.agent_selection = self.agents[0]
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self._paid: set[str] = set()
        self._update_infos()

    def observe(self, agent: str) -> dict[str, Any]:
        if self.game is None:
            raise RuntimeError("Reset before observing")
        return self.obs_encoder.encode(self.game, self.agent_name_mapping[agent])

    def step(self, action: int | None) -> None:
        if self.game is None or not self.agents:
            raise RuntimeError("Reset before stepping a finished environment")
        actor = self.agent_selection
        if self.terminations[actor] or self.truncations[actor]:
            self._was_dead_step(action)
            return
        if action is None or not self._action_space.contains(action):
            raise ValueError("A live decision requires an in-range action")
        pid = self.agent_name_mapping[actor]
        decoded = self.action_encoder.decode(int(action), pid, self.game)
        # Validate before consuming rewards: rejected decisions are mutation-free.
        valid, reason = decoded.validate(self.game)
        if not valid:
            raise ValueError(reason)
        self._cumulative_rewards[actor] = 0.0
        self._clear_rewards()
        self.game.apply_action(pid, decoded)
        for agent in self.agents:
            player = self.agent_name_mapping[agent]
            eliminated = self.game.players[player].bankrupt
            self.terminations[agent] = eliminated or self.game.game_over
            if agent not in self._paid and (eliminated or self.game.game_over):
                self.rewards[agent] = 1.0 if player == self.game.winner else -1.0
                self._paid.add(agent)
        focal = self.cutoff_player
        at_boundary = (
            focal is None or self.game.decision_player == focal or self.game.players[focal].bankrupt
        )
        if not self.game.game_over and self.game.turn_number >= self.max_turns and at_boundary:
            for agent in self.agents:
                self.truncations[agent] = not self.terminations[agent]
        self.agent_selection = f"player_{self.game.decision_player}"
        self._accumulate_rewards()
        self._update_infos()
        self._deads_step_first()

    def _update_infos(self) -> None:
        assert self.game is not None
        self.infos = {}
        for agent in self.agents:
            pid = self.agent_name_mapping[agent]
            self.infos[agent] = {
                "action_mask": self.action_encoder.get_action_mask(self.game, pid),
                "winner": self.game.winner,
                "game_over": self.game.game_over,
                "eliminated": self.game.players[pid].bankrupt,
                "elimination_order": self.game.state.elimination_order.copy(),
                "turns": self.game.turn_number,
                "horizon_overshoot": max(0, self.game.turn_number - self.max_turns),
                "rules_id": "foundation-v1",
                "action_version": ACTION_VERSION,
                "observation_version": OBSERVATION_VERSION,
                "episode_seed": self.episode_seed,
                "cutoff_reason": "turn_limit" if self.truncations.get(agent) else None,
            }

    def state(self) -> dict[str, Any]:
        assert self.game is not None
        return self.game.to_dict()

    def render(self) -> str | None:
        if self.render_mode == "ansi":
            return f"Turn {self.game.turn_number}: {self.game}"
        if self.render_mode == "human":
            print(self.game)
        return None

    def close(self) -> None:
        pass
