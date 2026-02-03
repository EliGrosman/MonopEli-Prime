"""Observation encoding for the Monopoly RL environment.

This module provides the ObservationEncoder class that encodes game state
to numpy arrays suitable for reinforcement learning algorithms.

The observation space is a Dict containing:
1. player_state: Current player's state (money, position, jail status, properties)
2. opponent_states: Other players' states (stacked features)
3. board_state: Property ownership and development (28 properties x 5 features)
4. game_state: Global game information (turn number, resources, last roll)
5. action_mask: Valid actions (149-dim for gameplay, 907-dim with trades)
6. trade_context: (Phase 2.5a) Pending trade information for responding
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from monopoly_engine.game import MonopolyGame

# Constants for observation encoding
MAX_MONEY: int = 10000  # Normalization cap for money values
BOARD_SIZE: int = 40
NUM_PROPERTIES: int = 28  # All buyable positions
MAX_JAIL_TURNS: int = 3
MAX_JAIL_CARDS: int = 2
MAX_TURN_NUMBER: int = 1000  # Normalization cap for turn number
MAX_HOUSES: int = 32
MAX_HOTELS: int = 12
MAX_DICE_SUM: int = 12

# Property positions (all 28 buyable spaces) - must match action_space.py
PROPERTY_POSITIONS: tuple[int, ...] = (
    1, 3,  # Brown
    6, 8, 9,  # Light Blue
    11, 13, 14,  # Magenta
    16, 18, 19,  # Orange
    21, 23, 24,  # Red
    26, 27, 29,  # Yellow
    31, 32, 34,  # Green
    37, 39,  # Dark Blue
    5, 15, 25, 35,  # Railroads
    12, 28,  # Utilities
)

# Map from property position to index in observation (0-27)
PROPERTY_POS_TO_IDX: dict[int, int] = {
    pos: idx for idx, pos in enumerate(PROPERTY_POSITIONS)
}

# Inverse map: index to position
PROPERTY_IDX_TO_POS: dict[int, int] = {
    idx: pos for idx, pos in enumerate(PROPERTY_POSITIONS)
}


class ObservationEncoder:
    """Encodes Monopoly game state to numpy arrays for RL.

    The encoder creates a structured observation space with separate components
    for player state, opponent states, board state, and game state. This design
    allows RL algorithms to easily access different aspects of the game.

    Attributes:
        num_players: Total number of players in the game.
        max_opponents: Maximum number of opponents (num_players - 1).
        enable_trades: Whether trade context is included (Phase 2.5a).
    """

    def __init__(self, num_players: int, enable_trades: bool = False) -> None:
        """Initialize the observation encoder.

        Args:
            num_players: Total number of players (2-8).
            enable_trades: If True, include trade context in observations.

        Raises:
            ValueError: If num_players is not between 2 and 8.
        """
        if not 2 <= num_players <= 8:
            raise ValueError(f"num_players must be between 2 and 8, got {num_players}")

        self.num_players = num_players
        self.max_opponents = num_players - 1
        self.enable_trades = enable_trades

        # Features per opponent: money, position, in_jail, jail_turns, jail_cards,
        # bankrupt, plus 28-dim property ownership
        self._opponent_features = 6 + NUM_PROPERTIES

    def get_observation_space(self) -> spaces.Dict:
        """Define the gymnasium observation space.

        Returns:
            A gymnasium Dict space with the following components:
            - player_state: Dict with money, position, in_jail, jail_turns,
                           jail_cards, and properties_owned
            - opponent_states: Box of shape (max_opponents, features)
            - board_state: Box of shape (28, 5) - owner one-hot + houses normalized
            - game_state: Dict with turn_number, houses_remaining, hotels_remaining,
                         last_roll
            - action_mask: MultiBinary(149 or 907) for valid actions
            - trade_context: (Phase 2.5a) Dict with pending trade info
        """
        # Import action space size
        from .action_space import ACTION_SPACE_SIZE, GAMEPLAY_ACTION_SPACE_SIZE

        action_mask_size = (
            ACTION_SPACE_SIZE if self.enable_trades else GAMEPLAY_ACTION_SPACE_SIZE
        )

        obs_space: dict[str, spaces.Space[Any]] = {
            "player_state": spaces.Dict({
                # Money normalized to [0, 1] by dividing by MAX_MONEY
                "money": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                # Board position 0-39
                "position": spaces.Discrete(BOARD_SIZE),
                # Binary: in jail or not
                "in_jail": spaces.Discrete(2),
                # Jail turns 0-3
                "jail_turns": spaces.Discrete(MAX_JAIL_TURNS + 1),
                # Jail cards 0-2
                "jail_cards": spaces.Discrete(MAX_JAIL_CARDS + 1),
                # Binary vector: which of 28 properties are owned
                "properties_owned": spaces.MultiBinary(NUM_PROPERTIES),
            }),
            "opponent_states": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.max_opponents, self._opponent_features),
                dtype=np.float32,
            ),
            "board_state": spaces.Box(
                low=0.0,
                high=1.0,
                # 28 properties x 5 features (4-dim owner one-hot + normalized houses)
                shape=(NUM_PROPERTIES, 5),
                dtype=np.float32,
            ),
            "game_state": spaces.Dict({
                # Turn number normalized
                "turn_number": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                # Houses remaining normalized (0-32 -> 0-1)
                "houses_remaining": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                # Hotels remaining normalized (0-12 -> 0-1)
                "hotels_remaining": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                # Last dice roll normalized (0 if no roll, otherwise sum/12)
                "last_roll": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
            }),
            "action_mask": spaces.MultiBinary(action_mask_size),
        }

        # Add trade context if trades are enabled (Phase 2.5a)
        if self.enable_trades:
            obs_space["trade_context"] = spaces.Dict({
                # Is there a pending trade I need to respond to?
                "pending_trade": spaces.Discrete(2),  # 0 = no, 1 = yes
                # Property index being offered to me (0-27, or 28 for none)
                "offer_property_idx": spaces.Discrete(NUM_PROPERTIES + 1),
                # Property index they want from me (0-27, or 28 for none)
                "want_property_idx": spaces.Discrete(NUM_PROPERTIES + 1),
                # Who proposed the trade (0-3, or 4 for none)
                "proposer_id": spaces.Discrete(self.num_players + 1),
            })

        return spaces.Dict(obs_space)

    def encode(
        self,
        game: MonopolyGame,
        player_id: int,
        pending_trade_response: bool = False,
    ) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
        """Encode the full game state from a player's perspective.

        Args:
            game: The MonopolyGame instance to encode.
            player_id: The player ID whose perspective to encode from.
            pending_trade_response: If True, this player is responding to a trade.

        Returns:
            Dictionary matching the observation space structure with encoded
            numpy arrays.

        Raises:
            ValueError: If player_id is invalid.
        """
        if player_id < 0 or player_id >= self.num_players:
            raise ValueError(f"Invalid player_id: {player_id}")

        obs: dict[str, np.ndarray | dict[str, np.ndarray]] = {
            "player_state": self._encode_player_state(game, player_id),
            "opponent_states": self._encode_opponent_states(game, player_id),
            "board_state": self._encode_board_state(game, player_id),
            "game_state": self._encode_game_state(game),
            "action_mask": self._encode_action_mask(game, player_id, pending_trade_response),
        }

        # Add trade context if trades are enabled
        if self.enable_trades:
            obs["trade_context"] = self._encode_trade_context(game, player_id)

        return obs

    def _encode_player_state(
        self,
        game: MonopolyGame,
        player_id: int,
    ) -> dict[str, np.ndarray]:
        """Encode the current player's state.

        Args:
            game: The MonopolyGame instance.
            player_id: The player's ID.

        Returns:
            Dictionary with player state arrays.
        """
        player = game.players[player_id]

        # Get properties owned by this player
        owned_properties = game.property_manager.get_owned_by(player_id)
        properties_owned = np.zeros(NUM_PROPERTIES, dtype=np.int8)
        for pos in owned_properties:
            if pos in PROPERTY_POS_TO_IDX:
                properties_owned[PROPERTY_POS_TO_IDX[pos]] = 1

        return {
            "money": np.array(
                [min(player.money / MAX_MONEY, 1.0)],
                dtype=np.float32,
            ),
            "position": np.array(player.position, dtype=np.int64),
            "in_jail": np.array(1 if player.in_jail else 0, dtype=np.int64),
            "jail_turns": np.array(
                min(player.jail_turns, MAX_JAIL_TURNS),
                dtype=np.int64,
            ),
            "jail_cards": np.array(
                min(player.jail_cards, MAX_JAIL_CARDS),
                dtype=np.int64,
            ),
            "properties_owned": properties_owned,
        }

    def _encode_opponent_states(
        self,
        game: MonopolyGame,
        player_id: int,
    ) -> np.ndarray:
        """Encode all opponents' states.

        Opponents are ordered by their player ID, excluding the current player.
        Bankrupt players are still included but with bankrupt=1.
        If there are fewer opponents than max_opponents, remaining rows are zeros.

        Args:
            game: The MonopolyGame instance.
            player_id: The current player's ID.

        Returns:
            Array of shape (max_opponents, features) with opponent states.
        """
        opponent_states = np.zeros(
            (self.max_opponents, self._opponent_features),
            dtype=np.float32,
        )

        opponent_idx = 0
        for pid in range(self.num_players):
            if pid == player_id:
                continue

            player = game.players[pid]

            # Basic features
            opponent_states[opponent_idx, 0] = min(player.money / MAX_MONEY, 1.0)
            opponent_states[opponent_idx, 1] = player.position / (BOARD_SIZE - 1)
            opponent_states[opponent_idx, 2] = 1.0 if player.in_jail else 0.0
            opponent_states[opponent_idx, 3] = player.jail_turns / MAX_JAIL_TURNS
            opponent_states[opponent_idx, 4] = player.jail_cards / MAX_JAIL_CARDS
            opponent_states[opponent_idx, 5] = 1.0 if player.bankrupt else 0.0

            # Property ownership (binary)
            owned_properties = game.property_manager.get_owned_by(pid)
            for pos in owned_properties:
                if pos in PROPERTY_POS_TO_IDX:
                    opponent_states[opponent_idx, 6 + PROPERTY_POS_TO_IDX[pos]] = 1.0

            opponent_idx += 1

        return opponent_states

    def _encode_board_state(
        self,
        game: MonopolyGame,
        player_id: int,
    ) -> np.ndarray:
        """Encode the board state (property ownership and development).

        Each property has 5 features:
        - Owner one-hot (4 dims): [unowned, self, opponent, mortgaged]
          - unowned: property has no owner
          - self: current player owns it
          - opponent: another player owns it
          - mortgaged: property is mortgaged (any owner)
        - Houses normalized (1 dim): 0-5 houses normalized to 0-1

        Args:
            game: The MonopolyGame instance.
            player_id: The current player's ID.

        Returns:
            Array of shape (28, 5) with board state.
        """
        board_state = np.zeros((NUM_PROPERTIES, 5), dtype=np.float32)

        for idx, pos in enumerate(PROPERTY_POSITIONS):
            prop = game.property_manager.get(pos)
            if prop is None:
                # Should not happen, but handle gracefully
                board_state[idx, 0] = 1.0  # Mark as unowned
                continue

            if prop.owner is None:
                # Unowned
                board_state[idx, 0] = 1.0
            elif prop.mortgaged:
                # Mortgaged (regardless of owner)
                board_state[idx, 3] = 1.0
            elif prop.owner == player_id:
                # Self owns
                board_state[idx, 1] = 1.0
            else:
                # Opponent owns
                board_state[idx, 2] = 1.0

            # Normalized houses (0-5 -> 0-1)
            board_state[idx, 4] = prop.houses / 5.0

        return board_state

    def _encode_game_state(self, game: MonopolyGame) -> dict[str, np.ndarray]:
        """Encode global game state.

        Args:
            game: The MonopolyGame instance.

        Returns:
            Dictionary with game state arrays.
        """
        # Handle last_roll which may be None
        if game.last_roll is not None:
            last_roll_sum = sum(game.last_roll) / MAX_DICE_SUM
        else:
            last_roll_sum = 0.0

        return {
            "turn_number": np.array(
                [min(game.turn_number / MAX_TURN_NUMBER, 1.0)],
                dtype=np.float32,
            ),
            "houses_remaining": np.array(
                [game.houses_remaining / MAX_HOUSES],
                dtype=np.float32,
            ),
            "hotels_remaining": np.array(
                [game.hotels_remaining / MAX_HOTELS],
                dtype=np.float32,
            ),
            "last_roll": np.array([last_roll_sum], dtype=np.float32),
        }

    def _encode_action_mask(
        self,
        game: MonopolyGame,
        player_id: int,
        pending_trade_response: bool = False,
    ) -> np.ndarray:
        """Encode the action mask for valid actions.

        This creates a binary mask indicating which actions are valid in the
        current game state.

        Args:
            game: The MonopolyGame instance.
            player_id: The current player's ID.
            pending_trade_response: If True, only accept/reject are valid.

        Returns:
            Binary array of shape (149,) or (907,) with 1 for valid actions.
        """
        from .action_space import ActionEncoder

        encoder = ActionEncoder(enable_trades=self.enable_trades)
        mask = encoder.get_action_mask(game, player_id, pending_trade_response)
        return mask.astype(np.int8)

    def _encode_trade_context(
        self,
        game: MonopolyGame,
        player_id: int,
    ) -> dict[str, np.ndarray]:
        """Encode the trade context for a player.

        Args:
            game: The MonopolyGame instance.
            player_id: The current player's ID.

        Returns:
            Dictionary with trade context arrays.
        """
        # Check for pending trade directed at this player
        pending_trade = False
        offer_property_idx = NUM_PROPERTIES  # 28 = no property
        want_property_idx = NUM_PROPERTIES
        proposer_id = self.num_players  # 4 = no proposer (for 4-player game)

        for trade_id, trade in game.state.pending_trades.items():
            if trade["to_player"] == player_id:
                pending_trade = True

                # For simple 1-for-1 trades
                if trade["give_properties"] and trade["want_properties"]:
                    # What they're offering (give_properties from their perspective)
                    offer_pos = trade["give_properties"][0]
                    # What they want (want_properties from their perspective)
                    want_pos = trade["want_properties"][0]

                    # Convert positions to indices
                    if offer_pos in PROPERTY_POS_TO_IDX:
                        offer_property_idx = PROPERTY_POS_TO_IDX[offer_pos]
                    if want_pos in PROPERTY_POS_TO_IDX:
                        want_property_idx = PROPERTY_POS_TO_IDX[want_pos]

                proposer_id = trade["from_player"]
                break

        return {
            "pending_trade": np.array(1 if pending_trade else 0, dtype=np.int64),
            "offer_property_idx": np.array(offer_property_idx, dtype=np.int64),
            "want_property_idx": np.array(want_property_idx, dtype=np.int64),
            "proposer_id": np.array(proposer_id, dtype=np.int64),
        }


def flatten_observation(
    obs: dict[str, np.ndarray | dict[str, np.ndarray]],
) -> np.ndarray:
    """Flatten a structured observation dict to a single numpy array.

    This is useful for algorithms that require flat observation vectors
    (e.g., simple MLPs).

    Args:
        obs: The observation dictionary from ObservationEncoder.encode().

    Returns:
        1D numpy array with all observation components concatenated.
    """
    flat_parts: list[np.ndarray] = []

    # Player state
    player_state = obs["player_state"]
    if isinstance(player_state, dict):
        flat_parts.append(player_state["money"].flatten())
        flat_parts.append(
            np.array([player_state["position"]], dtype=np.float32) / (BOARD_SIZE - 1)
        )
        flat_parts.append(
            np.array([player_state["in_jail"]], dtype=np.float32)
        )
        flat_parts.append(
            np.array([player_state["jail_turns"]], dtype=np.float32) / MAX_JAIL_TURNS
        )
        flat_parts.append(
            np.array([player_state["jail_cards"]], dtype=np.float32) / MAX_JAIL_CARDS
        )
        flat_parts.append(player_state["properties_owned"].astype(np.float32))

    # Opponent states
    opponent_states = obs["opponent_states"]
    if isinstance(opponent_states, np.ndarray):
        flat_parts.append(opponent_states.flatten())

    # Board state
    board_state = obs["board_state"]
    if isinstance(board_state, np.ndarray):
        flat_parts.append(board_state.flatten())

    # Game state
    game_state = obs["game_state"]
    if isinstance(game_state, dict):
        flat_parts.append(game_state["turn_number"].flatten())
        flat_parts.append(game_state["houses_remaining"].flatten())
        flat_parts.append(game_state["hotels_remaining"].flatten())
        flat_parts.append(game_state["last_roll"].flatten())

    # Action mask
    action_mask = obs["action_mask"]
    if isinstance(action_mask, np.ndarray):
        flat_parts.append(action_mask.astype(np.float32))

    return np.concatenate(flat_parts)


def get_flat_observation_size(num_players: int) -> int:
    """Calculate the size of a flattened observation.

    Args:
        num_players: Total number of players (2-8).

    Returns:
        Total number of elements in a flattened observation.
    """
    if not 2 <= num_players <= 8:
        raise ValueError(f"num_players must be between 2 and 8, got {num_players}")

    max_opponents = num_players - 1
    opponent_features = 6 + NUM_PROPERTIES  # 34 features per opponent

    # Player state:
    # - money: 1
    # - position: 1
    # - in_jail: 1
    # - jail_turns: 1
    # - jail_cards: 1
    # - properties_owned: 28
    player_state_size = 1 + 1 + 1 + 1 + 1 + NUM_PROPERTIES  # 33

    # Opponent states: max_opponents * 34
    opponent_states_size = max_opponents * opponent_features

    # Board state: 28 properties * 5 features
    board_state_size = NUM_PROPERTIES * 5  # 140

    # Game state:
    # - turn_number: 1
    # - houses_remaining: 1
    # - hotels_remaining: 1
    # - last_roll: 1
    game_state_size = 4

    # Action mask: 149
    action_mask_size = 149

    return (
        player_state_size
        + opponent_states_size
        + board_state_size
        + game_state_size
        + action_mask_size
    )
