"""PettingZoo AEC Environment for Monopoly.

This module provides the MonopolyEnv class, a multi-agent turn-based environment
following the PettingZoo AEC (Agent Environment Cycle) API.

Phase 2.5a adds support for simple 1-for-1 property trades:
- When a player proposes a trade, the environment switches to the responder
- The responder can accept or reject the trade
- After response, control returns to the proposer's turn

Usage:
    from monopoly_gym import MonopolyEnv

    env = MonopolyEnv(num_players=4, enable_trades=True)
    env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            action_mask = info["action_mask"]
            action = policy(observation, action_mask)

        env.step(action)
"""

from __future__ import annotations

from typing import Any

from gymnasium import spaces
from pettingzoo import AECEnv  # type: ignore[import-untyped]
from pettingzoo.utils.agent_selector import AgentSelector  # type: ignore[import-untyped]

from monopoly_engine import MonopolyGame, RollDice, calculate_net_worth
from monopoly_engine.board import Board

from .action_space import (
    ACTION_SPACE_SIZE,
    GAMEPLAY_ACTION_SPACE_SIZE,
    OFFSET_ACCEPT_TRADE,
    OFFSET_END_TURN,
    OFFSET_REJECT_TRADE,
    OFFSET_SIMPLE_TRADE,
    SIMPLE_TRADE_DIM,
    ActionEncoder,
    decode_simple_trade,
    find_trade_for_player,
)
from .observation import ObservationEncoder
from .trades import (
    TradeRewardConfig,
    calculate_trade_rewards,
)


class MonopolyEnv(AECEnv):  # type: ignore[misc]
    """Multi-agent Monopoly environment following PettingZoo AEC API.

    This environment wraps the monopoly_engine game logic to provide
    a standard RL interface. It supports 2-4 players with configurable
    reward functions and game length limits.

    Attributes:
        possible_agents: List of all possible agent IDs (e.g., ["player_0", "player_1", ...])
        agents: List of currently active (non-bankrupt) agents
        observation_spaces: Dict mapping agent ID to observation space
        action_spaces: Dict mapping agent ID to action space
        game: The underlying MonopolyGame instance
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "monopoly_v1",
        "is_parallelizable": False,  # Turn-based game
    }

    def __init__(
        self,
        num_players: int = 4,
        max_turns: int = 1000,
        reward_type: str = "sparse",
        render_mode: str | None = None,
        enable_trades: bool = False,
        trade_reward_config: TradeRewardConfig | None = None,
    ) -> None:
        """Initialize the Monopoly environment.

        Args:
            num_players: Number of players (2-4)
            max_turns: Maximum turns before truncation
            reward_type: "sparse" (+1/-1 win/loss) or "dense" (incremental)
            render_mode: "human", "ansi", or None
            enable_trades: Enable Phase 2.5a trade actions (907 actions total)
            trade_reward_config: Configuration for trade reward shaping

        Raises:
            ValueError: If num_players is not 2-4
        """
        super().__init__()

        if not 2 <= num_players <= 4:
            raise ValueError(f"num_players must be 2-4, got {num_players}")

        self.num_players = num_players
        self.max_turns = max_turns
        self.reward_type = reward_type
        self.render_mode = render_mode
        self.enable_trades = enable_trades
        self.trade_reward_config = trade_reward_config or TradeRewardConfig()

        # Agent IDs
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}

        # Initialize encoders (with trade support if enabled)
        self.obs_encoder = ObservationEncoder(num_players, enable_trades=enable_trades)
        self.action_encoder = ActionEncoder(enable_trades=enable_trades)

        # Define spaces (same for all agents)
        self._observation_space: spaces.Dict = self.obs_encoder.get_observation_space()
        action_space_size = ACTION_SPACE_SIZE if enable_trades else GAMEPLAY_ACTION_SPACE_SIZE
        self._action_space: spaces.Space[Any] = spaces.Discrete(action_space_size)

        # These will be set in reset()
        self.game: MonopolyGame | None = None
        self._agent_selector: AgentSelector | None = None
        self.agent_selection: str = ""

        # Reward/termination tracking (set in reset)
        self.rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, Any]] = {}

        # For dense rewards
        self._prev_net_worth: dict[str, int] = {}

        # Trade state (Phase 2.5a)
        # When a trade is proposed, we track who needs to respond
        self._pending_trade_response: bool = False
        self._trade_proposer_agent: str | None = None
        self._trade_responder_agent: str | None = None

    @property
    def observation_spaces(self) -> dict[str, spaces.Space[Any]]:
        """Get observation spaces for all agents."""
        return {agent: self._observation_space for agent in self.possible_agents}

    @property
    def action_spaces(self) -> dict[str, spaces.Space[Any]]:
        """Get action spaces for all agents."""
        return {agent: self._action_space for agent in self.possible_agents}

    def observation_space(self, agent: str) -> spaces.Space[Any]:
        """Get observation space for a specific agent."""
        return self._observation_space

    def action_space(self, agent: str) -> spaces.Space[Any]:
        """Get action space for a specific agent."""
        return self._action_space

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
        """
        # Create new game
        self.game = MonopolyGame(
            num_players=self.num_players,
            seed=seed,
        )

        # Reset agent tracking
        self.agents = self.possible_agents.copy()
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

        # Reset reward/termination tracking
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Track previous state for dense reward calculation
        self._prev_net_worth = {
            agent: self._get_net_worth(self.agent_name_mapping[agent])
            for agent in self.agents
        }

        # Reset trade state
        self._pending_trade_response = False
        self._trade_proposer_agent = None
        self._trade_responder_agent = None

        # Clear action mask cache for new episode
        self.action_encoder.invalidate_cache()

        # Update infos with action masks
        self._update_infos()

        # Automatically roll dice for first player's turn
        self._handle_turn_start()

    def step(self, action: int | None) -> None:
        """Execute an action for the current agent.

        Args:
            action: Action index or None if terminated/truncated.
                   - 0-148: Gameplay actions
                   - 149-904: Trade proposals (if enable_trades)
                   - 905: Accept trade (if enable_trades)
                   - 906: Reject trade (if enable_trades)
        """
        if self.game is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        agent = self.agent_selection

        # Ensure agent is in tracking dicts (may be missing after bankruptcy)
        if agent not in self.terminations:
            self.terminations[agent] = True
        if agent not in self.truncations:
            self.truncations[agent] = False
        if agent not in self._cumulative_rewards:
            self._cumulative_rewards[agent] = 0.0
        if agent not in self.rewards:
            self.rewards[agent] = 0.0
        if agent not in self.infos:
            self.infos[agent] = {}

        # Handle terminated/truncated agent
        if self.terminations[agent] or self.truncations[agent]:
            # Ensure agent is in agents list for _was_dead_step
            if agent not in self.agents:
                self.agents.append(agent)
            self._was_dead_step(action)
            return

        player_id = self.agent_name_mapping[agent]

        # Clear previous rewards
        self._clear_rewards()

        # Handle trade response (if we're in that state)
        if self._pending_trade_response and self.enable_trades:
            self._handle_trade_response(action, agent, player_id)
            return

        # Decode and execute action
        if action is not None:
            # Validate action against cached mask (computed in previous _update_infos)
            cached_info = self.infos.get(agent, {})
            if "_needs_mask_update" not in cached_info and "action_mask" in cached_info:
                mask = cached_info["action_mask"]
            else:
                mask = self.action_encoder.get_action_mask(self.game, player_id)
            if not mask[action]:
                # Invalid action: small penalty and skip
                self.rewards[agent] = -0.01
                self._accumulate_rewards()
                self._advance_agent()
                self._update_infos()
                return

            # Check if this is a trade proposal (Phase 2.5a)
            if (
                self.enable_trades
                and OFFSET_SIMPLE_TRADE <= action < OFFSET_SIMPLE_TRADE + SIMPLE_TRADE_DIM
            ):
                self._handle_trade_proposal(action, agent, player_id)
                return

            # Decode action
            game_action = self.action_encoder.decode(action, player_id, self.game)

            # Execute action
            try:
                is_valid, error = game_action.validate(self.game)
                if is_valid:
                    game_action.execute(self.game)
                else:
                    # Validation failed (shouldn't happen if mask is correct)
                    self.rewards[agent] = -0.01
            except Exception:
                # Execution error
                self.rewards[agent] = -0.01

        # Calculate rewards
        self._calculate_rewards()

        # Check termination conditions
        self._check_terminations()

        # Accumulate rewards
        self._accumulate_rewards()

        # Advance to next agent
        self._advance_agent()

        # Update infos with new action masks
        self._update_infos()

        # If this was an EndTurn action, trigger the next player's turn start
        if action == OFFSET_END_TURN and not all(self.terminations.values()):
            self._handle_turn_start()

    def _handle_trade_proposal(
        self,
        action: int,
        agent: str,
        player_id: int,
    ) -> None:
        """Handle a trade proposal action.

        This creates a pending trade in the game state and switches
        control to the responder for them to accept/reject.

        Args:
            action: The trade proposal action index
            agent: The proposing agent name
            player_id: The proposing player ID
        """
        if self.game is None:
            return

        # Decode the trade action to get properties
        my_prop, their_prop = decode_simple_trade(action)

        # Find the owner of their_prop
        prop = self.game.property_manager.get(their_prop)
        if prop is None or prop.owner is None:
            # Invalid trade - property not owned
            self.rewards[agent] = -0.01
            self._accumulate_rewards()
            self._update_infos()
            return

        responder_id = prop.owner
        responder_agent = f"player_{responder_id}"

        # Create the pending trade in game state
        _trade_id = self.game.propose_trade(
            from_player=player_id,
            to_player=responder_id,
            give_properties=[my_prop],
            want_properties=[their_prop],
            give_money=0,
            want_money=0,
        )

        # Set up trade response state
        self._pending_trade_response = True
        self._trade_proposer_agent = agent
        self._trade_responder_agent = responder_agent

        # Switch to responder
        self.agent_selection = responder_agent

        # Update infos with trade response mask (only accept/reject valid)
        self._update_infos_for_trade_response()

    def _handle_trade_response(
        self,
        action: int | None,
        agent: str,
        player_id: int,
    ) -> None:
        """Handle accept/reject of a pending trade.

        Args:
            action: OFFSET_ACCEPT_TRADE or OFFSET_REJECT_TRADE (or None/invalid)
            agent: The responding agent name
            player_id: The responding player ID
        """
        if self.game is None or self._trade_proposer_agent is None:
            return

        proposer_agent = self._trade_proposer_agent
        proposer_id = self.agent_name_mapping[proposer_agent]

        # Find the pending trade
        trade_id = find_trade_for_player(self.game, player_id)
        if trade_id is None:
            # No trade to respond to - shouldn't happen
            self._clear_trade_state()
            self._update_infos()
            return

        trade = self.game.state.pending_trades.get(trade_id)
        if trade is None:
            self._clear_trade_state()
            self._update_infos()
            return

        # Get trade details before execution (for reward calculation)
        proposer_gave = trade["give_properties"][0] if trade["give_properties"] else 0
        proposer_got = trade["want_properties"][0] if trade["want_properties"] else 0

        accepted = False

        if action == OFFSET_ACCEPT_TRADE:
            # Accept the trade
            try:
                self.game.accept_trade(trade_id)
                accepted = True
            except ValueError:
                # Trade failed (e.g., player no longer has money/property)
                self.rewards[agent] = -0.01

        elif action == OFFSET_REJECT_TRADE:
            # Reject the trade
            self.game.reject_trade(trade_id)
            accepted = False

        else:
            # Invalid action during trade response - treat as reject
            if trade_id in self.game.state.pending_trades:
                self.game.reject_trade(trade_id)
            self.rewards[agent] = -0.01
            accepted = False

        # Calculate trade-specific rewards
        trade_rewards = calculate_trade_rewards(
            self.game,
            proposer_id=proposer_id,
            responder_id=player_id,
            proposer_gave=proposer_gave,
            proposer_got=proposer_got,
            accepted=accepted,
            config=self.trade_reward_config,
        )

        # Apply trade rewards
        for pid, reward in trade_rewards.items():
            agent_name = f"player_{pid}"
            if agent_name in self.rewards:
                self.rewards[agent_name] += reward

        # Clear trade state
        self._clear_trade_state()

        # Return to proposer's turn
        self.agent_selection = proposer_agent

        # Calculate regular rewards
        self._calculate_rewards()

        # Check termination conditions
        self._check_terminations()

        # Accumulate rewards
        self._accumulate_rewards()

        # Update infos for proposer's continued turn
        self._update_infos()

    def _clear_trade_state(self) -> None:
        """Clear the pending trade response state."""
        self._pending_trade_response = False
        self._trade_proposer_agent = None
        self._trade_responder_agent = None

    def _update_infos_for_trade_response(self) -> None:
        """Update infos when a player needs to respond to a trade.

        Only accept/reject actions are valid.
        """
        import numpy as np

        if self.game is None or self._trade_responder_agent is None:
            return

        responder_agent = self._trade_responder_agent
        responder_id = self.agent_name_mapping[responder_agent]

        # Create mask with only accept/reject valid
        mask = self.action_encoder.get_action_mask(
            self.game, responder_id, pending_trade_response=True
        )

        self.infos[responder_agent] = {
            "action_mask": mask,
            "pending_trade_response": True,
        }

        # Set placeholder for other agents
        placeholder = np.zeros(self.action_encoder.action_space_size, dtype=np.bool_)
        for agent in self.agents:
            if agent != responder_agent:
                self.infos[agent] = {
                    "action_mask": placeholder,
                    "_needs_mask_update": True,
                }

    def observe(self, agent: str) -> dict[str, Any]:
        """Get observation for a specific agent.

        Args:
            agent: Agent ID

        Returns:
            Observation dictionary
        """
        if self.game is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        player_id = self.agent_name_mapping[agent]

        # Check if this agent is responding to a trade
        pending_response = (
            self._pending_trade_response
            and self._trade_responder_agent == agent
        )

        return self.obs_encoder.encode(
            self.game, player_id, pending_trade_response=pending_response
        )

    def last(
        self,
        observe: bool = True,
    ) -> tuple[dict[str, Any] | None, float, bool, bool, dict[str, Any]]:
        """Get last observation, reward, termination, truncation, info.

        Args:
            observe: Whether to return observation (True) or None

        Returns:
            Tuple of (observation, reward, termination, truncation, info)
        """
        agent = self.agent_selection

        # Ensure agent is in tracking dicts (may be missing after bankruptcy)
        if agent not in self.terminations:
            self.terminations[agent] = True
        if agent not in self.truncations:
            self.truncations[agent] = False
        if agent not in self._cumulative_rewards:
            self._cumulative_rewards[agent] = 0.0
        if agent not in self.rewards:
            self.rewards[agent] = 0.0
        if agent not in self.infos:
            self.infos[agent] = {}

        if observe:
            observation = self.observe(agent)
        else:
            observation = None

        # Check if this agent is responding to a trade
        pending_response = (
            self._pending_trade_response
            and self._trade_responder_agent == agent
        )

        # Ensure action mask is computed (lazy)
        mask = self._get_agent_mask(agent, pending_trade_response=pending_response)
        info: dict[str, Any] = {"action_mask": mask}

        # Add trade context to info if responding
        if pending_response:
            info["pending_trade_response"] = True

        return (
            observation,
            self._cumulative_rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            info,
        )

    def _clear_rewards(self) -> None:
        """Clear rewards for all agents."""
        for agent in self.agents:
            self.rewards[agent] = 0.0

    def _accumulate_rewards(self) -> None:
        """Accumulate rewards into cumulative rewards."""
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]

    def _calculate_rewards(self) -> None:
        """Calculate rewards based on reward_type."""
        if self.game is None:
            return

        for agent in self.agents:
            player_id = self.agent_name_mapping[agent]

            if self.reward_type == "sparse":
                # Only reward on game end
                if self.game.game_over:
                    if self.game.winner == player_id:
                        self.rewards[agent] = 1.0
                    else:
                        self.rewards[agent] = -1.0
                else:
                    self.rewards[agent] = 0.0

            elif self.reward_type == "dense":
                # Incremental rewards based on net worth change
                current_worth = self._get_net_worth(player_id)
                prev_worth = self._prev_net_worth[agent]

                # Normalize reward
                reward = (current_worth - prev_worth) / 1000.0

                # Bonus for winning
                if self.game.game_over and self.game.winner == player_id:
                    reward += 10.0

                self.rewards[agent] = reward
                self._prev_net_worth[agent] = current_worth

    def _get_net_worth(self, player_id: int) -> int:
        """Calculate net worth for a player.

        Args:
            player_id: Player ID

        Returns:
            Net worth in dollars
        """
        if self.game is None:
            return 0

        return calculate_net_worth(
            self.game.players[player_id],
            self.game.property_manager,
        )

    def _check_terminations(self) -> None:
        """Check for game over and truncation conditions."""
        if self.game is None:
            return

        # Game over (winner determined)
        if self.game.game_over:
            for agent in self.agents:
                self.terminations[agent] = True

        # Max turns (truncation)
        if self.game.turn_number >= self.max_turns:
            for agent in self.agents:
                self.truncations[agent] = True

        # Individual bankruptcy
        for agent in self.agents:
            player_id = self.agent_name_mapping[agent]
            if self.game.players[player_id].bankrupt:
                self.terminations[agent] = True

    def _update_infos(self) -> None:
        """Update info dicts with action masks.

        For performance, only computes action mask for current agent.
        Other agents get masks computed lazily when accessed via infos property.
        """
        import numpy as np

        if self.game is None:
            return

        # Only compute mask for current agent (main performance optimization)
        current_agent = self.agent_selection
        if current_agent in self.agents:
            player_id = self.agent_name_mapping[current_agent]
            mask = self.action_encoder.get_action_mask(self.game, player_id)
            self.infos[current_agent] = {
                "action_mask": mask,
            }

        # Set placeholder for other agents (computed lazily in _get_agent_mask)
        # Placeholder mask allows all actions - real mask computed on access
        placeholder = np.ones(self.action_encoder.action_space_size, dtype=np.bool_)
        for agent in self.agents:
            if agent != current_agent:
                self.infos[agent] = {
                    "action_mask": placeholder,
                    "_needs_mask_update": True,
                }

    def _get_agent_mask(
        self,
        agent: str,
        pending_trade_response: bool = False,
    ) -> Any:
        """Get action mask for agent, computing if needed.

        Args:
            agent: Agent ID
            pending_trade_response: If True, only accept/reject are valid

        Returns:
            Action mask array
        """
        import numpy as np

        if self.game is None:
            return np.ones(self.action_encoder.action_space_size, dtype=np.bool_)

        info = self.infos.get(agent, {})

        # If mask needs update (not the current agent), compute it
        if info.get("_needs_mask_update", False) or pending_trade_response:
            player_id = self.agent_name_mapping[agent]
            mask = self.action_encoder.get_action_mask(
                self.game, player_id, pending_trade_response=pending_trade_response
            )
            self.infos[agent] = {"action_mask": mask}
            return mask

        # Return cached mask
        if "action_mask" in info:
            return info["action_mask"]

        # Fallback: compute mask
        player_id = self.agent_name_mapping[agent]
        mask = self.action_encoder.get_action_mask(
            self.game, player_id, pending_trade_response=pending_trade_response
        )
        self.infos[agent] = {"action_mask": mask}
        return mask

    def _handle_turn_start(self) -> None:
        """Handle automatic dice roll at the start of a turn.

        In Monopoly, each turn starts with rolling dice and moving.
        This happens automatically - the agent only chooses post-move actions.
        """
        if self.game is None:
            return

        # Get current player
        current_agent = self.agent_selection
        if current_agent not in self.agent_name_mapping:
            return

        player_id = self.agent_name_mapping[current_agent]
        player = self.game.players[player_id]

        # Don't roll if game is over or player is bankrupt
        if self.game.game_over or player.bankrupt:
            return

        # Don't roll if player is in jail (they get to choose actions first)
        # Actually, in Monopoly, if you're in jail you still roll to try to get out
        # The jail logic is handled in _handle_dice_roll

        # Execute dice roll
        roll_action = RollDice(player_id=player_id)
        is_valid, _ = roll_action.validate(self.game)
        if is_valid:
            roll_action.execute(self.game)

        # Check terminations after dice roll (bankruptcy might happen from rent)
        self._check_terminations()

        # Update infos with new action masks after movement
        self._update_infos()

    def _advance_agent(self) -> None:
        """Advance to the next non-terminated agent."""
        if self._agent_selector is None or self.game is None:
            return

        # Remove terminated agents from active list (but keep in terminations/truncations dicts)
        self.agents = [
            agent for agent in self.agents
            if not self.terminations.get(agent, True)
        ]

        # Ensure all possible agents remain in terminations/truncations dicts
        # (PettingZoo requires this for _was_dead_step)
        for agent in self.possible_agents:
            if agent not in self.terminations:
                self.terminations[agent] = True  # Assume terminated if missing
            if agent not in self.truncations:
                self.truncations[agent] = False

        if not self.agents:
            return

        # Rebuild agent selector with remaining agents
        self._agent_selector = AgentSelector(self.agents)

        # Sync with game's current player
        current_player = self.game.current_player
        current_agent = f"player_{current_player}"

        if current_agent in self.agents:
            # Find and select the matching agent
            while self._agent_selector.next() != current_agent:
                pass
            self.agent_selection = current_agent
        else:
            # Current player is bankrupt, just advance
            self.agent_selection = self._agent_selector.next()

    def render(self) -> str | None:
        """Render the environment.

        Returns:
            String representation if mode is "ansi", None otherwise
        """
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
            return None
        return None

    def _render_ansi(self) -> str:
        """Render game state as text.

        Returns:
            String representation of current game state
        """
        if self.game is None:
            return "No game in progress"

        lines = [
            f"=== Turn {self.game.turn_number} ===",
            f"Current agent: {self.agent_selection}",
            "",
        ]

        for i, player in enumerate(self.game.players):
            status = "BANKRUPT" if player.bankrupt else "active"
            jail = " (JAIL)" if player.in_jail else ""
            position = player.position
            space = Board.get_space(position)

            lines.append(
                f"Player {i}: ${player.money:,}, "
                f"pos {position} ({space.name}){jail} [{status}]"
            )

        # Show some property info
        lines.append("")
        owned_props = []
        for pos in [1, 3, 5, 6, 8, 9, 11, 13, 14, 15, 16, 18, 19]:
            prop = self.game.property_manager.get(pos)
            if prop and prop.owner is not None:
                space = Board.get_space(pos)
                owned_props.append(f"{space.name}(P{prop.owner})")

        if owned_props:
            lines.append(f"Properties: {', '.join(owned_props[:5])}...")

        return "\n".join(lines)

    def close(self) -> None:
        """Clean up resources."""
        pass

    def state(self) -> dict[str, Any]:
        """Get the full environment state.

        Returns:
            Dictionary with complete state for serialization
        """
        if self.game is None:
            return {}
        return self.game.to_dict()
