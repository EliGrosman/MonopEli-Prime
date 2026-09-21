"""
AI opponent orchestration service.

Integrates Phase 2 agents (RandomAgent, RuleBasedAgent, AggressiveAgent,
ConservativeAgent) for server-side AI opponent play.
"""

import asyncio
from typing import TYPE_CHECKING, Any

from agents.base import Agent
from agents.random_agent import RandomAgent
from agents.rule_based import AggressiveAgent, ConservativeAgent, RuleBasedAgent
from monopoly_engine.progress import ProgressGuard
from monopoly_gym.action_space import ActionEncoder

from ..config import get_settings

if TYPE_CHECKING:
    from monopoly_engine import MonopolyGame

    from .game_manager import GameManager


# Map AI type strings to agent classes
AI_TYPES: dict[str, type[Agent]] = {
    "random": RandomAgent,
    "rule_based": RuleBasedAgent,
    "aggressive": AggressiveAgent,
    "conservative": ConservativeAgent,
}


class AIManager:
    """Manages AI opponent turns for active games.

    This service orchestrates AI players by:
    - Creating agent instances for AI players when games start
    - Processing AI turns with configurable think delays
    - Executing AI actions through the game manager
    - Handling AI player lifecycle
    """

    def __init__(self, think_delay_ms: int | None = None) -> None:
        """Initialize the AI manager.

        Args:
            think_delay_ms: Milliseconds to wait before AI actions (for UX).
                          If None, uses config value.
        """
        settings = get_settings()
        self.think_delay_ms = (
            think_delay_ms if think_delay_ms is not None else settings.ai_think_delay_ms
        )
        # (game_id, player_id) -> Agent instance
        self._agents: dict[tuple[str, int], Agent] = {}
        self._encoder = ActionEncoder()
        self._guards: dict[str, ProgressGuard] = {}
        self._processing: set[str] = set()  # Games currently being processed

    def create_agent(
        self,
        game_id: str,
        player_id: int,
        ai_type: str = "rule_based",
        seed: int | None = None,
    ) -> Agent:
        """Create an AI agent for a player slot.

        Args:
            game_id: The game ID
            player_id: The player slot for this agent
            ai_type: Type of agent ("random", "rule_based", "aggressive", "conservative")

        Returns:
            The created agent instance

        Raises:
            ValueError: If ai_type is not recognized
        """
        agent_cls = AI_TYPES.get(ai_type)
        if agent_cls is None:
            raise ValueError(f"Unknown AI type: {ai_type}. Valid types: {list(AI_TYPES.keys())}")

        agent = (
            agent_cls(player_id=player_id, seed=seed)
            if agent_cls is RandomAgent
            else agent_cls(player_id=player_id)
        )
        self._agents[(game_id, player_id)] = agent
        return agent

    def get_agent(self, game_id: str, player_id: int) -> Agent | None:
        """Get an existing agent.

        Args:
            game_id: The game ID
            player_id: The player ID

        Returns:
            The Agent instance if it exists, None otherwise
        """
        return self._agents.get((game_id, player_id))

    def remove_agent(self, game_id: str, player_id: int) -> bool:
        """Remove an AI agent.

        Args:
            game_id: The game ID
            player_id: The player ID

        Returns:
            True if agent was removed, False if not found
        """
        key = (game_id, player_id)
        if key in self._agents:
            del self._agents[key]
            return True
        return False

    def remove_game_agents(self, game_id: str) -> int:
        """Remove all agents for a game.

        Args:
            game_id: The game ID

        Returns:
            Number of agents removed
        """
        self._guards.pop(game_id, None)
        to_remove = [key for key in self._agents if key[0] == game_id]
        for key in to_remove:
            del self._agents[key]
        return len(to_remove)

    def is_ai_player(self, game_id: str, player_id: int) -> bool:
        """Check if a player slot has an AI agent.

        Args:
            game_id: The game ID
            player_id: The player ID

        Returns:
            True if player has an AI agent
        """
        return (game_id, player_id) in self._agents

    async def process_ai_turn(
        self,
        game_manager: "GameManager",
        game_id: str,
        player_id: int,
        max_actions: int = 100,
    ) -> int:
        """Process an AI player's turn.

        Executes actions until the turn ends or an error occurs.
        This method is safe to call even if the current player is not AI.

        Args:
            game_manager: The game manager for executing actions
            game_id: The game ID
            player_id: The player ID
            max_actions: Maximum actions to take (safety limit)

        Returns:
            Number of actions taken
        """
        agent = self._agents.get((game_id, player_id))
        if agent is None:
            return 0

        active_game = await game_manager.get_game(game_id)
        if active_game is None:
            return 0

        actions_taken = 0

        while actions_taken < max_actions:
            game = active_game.game

            # Check if still this player's turn and game not over
            if game.decision_player != player_id or game.game_over:
                break

            # Artificial thinking delay for better UX
            if self.think_delay_ms > 0:
                await asyncio.sleep(self.think_delay_ms / 1000)

            # Get valid actions mask
            mask = self._encoder.get_action_mask(game, player_id)

            # Check if any actions are available
            if not mask.any():
                raise RuntimeError("Live AI decision has no legal actions")
            self._guards.setdefault(game_id, ProgressGuard()).check(game)

            # Get observation (simplified dict for agents)
            observation = self._get_observation(game, player_id)

            # Let agent choose action
            action_idx = agent.choose_action(
                observation=observation,
                action_mask=mask,
                game=game,
            )

            action = self._encoder.decode(action_idx, player_id, game)
            success, message = await game_manager.execute_action(game_id, action)
            actions_taken += 1
            if not success:
                raise RuntimeError(f"AI action failed: {message}")

            # Refresh game reference after action
            active_game = await game_manager.get_game(game_id)
            if active_game is None:
                break

        return actions_taken

    async def process_ai_turns_for_game(
        self,
        game_manager: "GameManager",
        game_id: str,
    ) -> None:
        """Process AI turns for a game until a human player's turn.

        This method loops through AI players, executing their turns,
        until it's a human player's turn or the game ends.

        Args:
            game_manager: The game manager
            game_id: The game ID
        """
        # Prevent concurrent processing of the same game
        if game_id in self._processing:
            return
        self._processing.add(game_id)

        try:
            while True:
                active_game = await game_manager.get_game(game_id)
                if active_game is None:
                    break

                game = active_game.game
                if game.game_over:
                    break

                current_player = game.decision_player

                # Check if current player is AI
                if not self.is_ai_player(game_id, current_player):
                    # Human player's turn - stop processing
                    break

                # Process AI turn
                count = await self.process_ai_turn(game_manager, game_id, current_player)
                if count == 0:
                    raise RuntimeError("AI made no progress")

                # Small delay between AI players for better UX
                await asyncio.sleep(0.1)

        finally:
            self._processing.discard(game_id)

    def _get_observation(self, game: "MonopolyGame", player_id: int) -> dict[str, Any]:
        """Get a simplified observation dict for an agent.

        Args:
            game: The game instance
            player_id: The player requesting observation

        Returns:
            Observation dictionary
        """
        state = game.state.to_dict()
        return {
            "players": state["players"],
            "properties": state["properties"],
            "current_player": state["current_player"],
            "turn_number": state["turn_number"],
            "houses_remaining": state["houses_remaining"],
            "hotels_remaining": state["hotels_remaining"],
            "player_id": player_id,
        }

    def get_ai_type_for_agent(self, game_id: str, player_id: int) -> str | None:
        """Get the AI type string for an agent.

        Args:
            game_id: The game ID
            player_id: The player ID

        Returns:
            AI type string if agent exists, None otherwise
        """
        agent = self._agents.get((game_id, player_id))
        if agent is None:
            return None

        # Reverse lookup the type from the agent class
        for type_name, agent_cls in AI_TYPES.items():
            if isinstance(agent, agent_cls):
                return type_name

        return "unknown"

    def agent_count(self) -> int:
        """Get total number of active AI agents.

        Returns:
            Count of active agents
        """
        return len(self._agents)
