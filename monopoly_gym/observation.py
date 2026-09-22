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
OBSERVATION_VERSION = "observation-v2"
TRADE_OBSERVATION_VERSION = "observation-v3"
TRADE_ROW_FEATURES = 131

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
    1,
    3,  # Brown
    6,
    8,
    9,  # Light Blue
    11,
    13,
    14,  # Magenta
    16,
    18,
    19,  # Orange
    21,
    23,
    24,  # Red
    26,
    27,
    29,  # Yellow
    31,
    32,
    34,  # Green
    37,
    39,  # Dark Blue
    5,
    15,
    25,
    35,  # Railroads
    12,
    28,  # Utilities
)

# Map from property position to index in observation (0-27)
PROPERTY_POS_TO_IDX: dict[int, int] = {pos: idx for idx, pos in enumerate(PROPERTY_POSITIONS)}

# Inverse map: index to position
PROPERTY_IDX_TO_POS: dict[int, int] = {idx: pos for idx, pos in enumerate(PROPERTY_POSITIONS)}


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

    def __init__(
        self,
        num_players: int,
        enable_trades: bool = False,
        *,
        rules_id: str = "foundation-v1",
        action_encoder: Any = None,
    ) -> None:
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
        if enable_trades:
            raise ValueError("enable_trades is obsolete; select rules_id='foundation-trade-v1'")
        self.rules_id = rules_id
        self.enable_trades = rules_id == "foundation-trade-v1"
        self._action_encoder = action_encoder

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
        from .action_space import GAMEPLAY_ACTION_SPACE_SIZE, TRADE_ACTION_SPACE_SIZE

        action_mask_size = (
            TRADE_ACTION_SPACE_SIZE if self.enable_trades else GAMEPLAY_ACTION_SPACE_SIZE
        )

        obs_space: dict[str, spaces.Space[Any]] = {
            "player_state": spaces.Dict(
                {
                    # Money normalized to [0, 1] by dividing by MAX_MONEY
                    "money": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
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
                }
            ),
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
            "game_state": spaces.Dict(
                {
                    # Turn number normalized
                    "turn_number": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    # Houses remaining normalized (0-32 -> 0-1)
                    "houses_remaining": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    # Hotels remaining normalized (0-12 -> 0-1)
                    "hotels_remaining": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    # Last dice roll normalized (0 if no roll, otherwise sum/12)
                    "last_roll": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                }
            ),
            "decision_state": spaces.Box(
                0.0, 1.0, shape=(14 if self.enable_trades else 13,), dtype=np.float32
            ),
            "action_mask": spaces.MultiBinary(action_mask_size),
        }

        # Add trade context if trades are enabled (Phase 2.5a)
        if self.enable_trades:
            from .action_space import TRADE_CANDIDATE_COUNT

            obs_space["trade_context"] = spaces.Dict(
                {
                    "pending_offer": spaces.Box(
                        low=0.0, high=np.inf, shape=(TRADE_ROW_FEATURES,), dtype=np.float32
                    ),
                    "candidate_rows": spaces.Box(
                        low=0.0,
                        high=np.inf,
                        shape=(TRADE_CANDIDATE_COUNT, TRADE_ROW_FEATURES),
                        dtype=np.float32,
                    ),
                    "candidate_mask": spaces.MultiBinary(TRADE_CANDIDATE_COUNT),
                    "proposal_state": spaces.Box(
                        low=0.0, high=np.inf, shape=(11,), dtype=np.float32
                    ),
                }
            )

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
            "decision_state": self._encode_decision_state(game),
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

    def _encode_decision_state(self, game: MonopolyGame) -> np.ndarray:
        from monopoly_engine.foundation import PHASES

        phases = PHASES + (("trade_response",) if self.enable_trades else ())
        phase = [float(game.state.phase == item) for item in phases]
        debt = game.state.obligations[0] if game.state.obligations else None
        return np.asarray(
            phase
            + [
                game.current_player / 7,
                game.decision_player / 7,
                float(game.state.roll_owed),
                game.doubles_count / 3,
                min((debt["amount"] if debt else 0) / MAX_MONEY, 1),
                (debt["creditor"] + 1) / 8 if debt and debt["creditor"] is not None else 0,
                (game.players[game.current_player].position + 1) / 40
                if game.state.phase == "purchase_decision"
                else 0,
            ],
            dtype=np.float32,
        )

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

        encoder = self._action_encoder or ActionEncoder(rules_id=self.rules_id)
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
        from .action_space import TRADE_CANDIDATE_COUNT, ActionEncoder

        encoder = self._action_encoder or ActionEncoder(rules_id=self.rules_id)

        def row(
            proposer: int,
            recipient: int,
            give: list[int] | tuple[int, ...],
            want: list[int] | tuple[int, ...],
            give_money: int,
            want_money: int,
        ) -> np.ndarray:
            result = np.zeros(TRADE_ROW_FEATURES, dtype=np.float32)
            result[0] = 1.0
            result[1 + proposer] = 1.0
            result[9 + recipient] = 1.0
            for position in give:
                idx = PROPERTY_POS_TO_IDX[position]
                result[17 + idx] = 1.0
                result[73 + idx] = float(game.property_manager.properties[position].mortgaged)
            for position in want:
                idx = PROPERTY_POS_TO_IDX[position]
                result[45 + idx] = 1.0
                result[101 + idx] = float(game.property_manager.properties[position].mortgaged)
            result[129] = give_money / MAX_MONEY
            result[130] = want_money / MAX_MONEY
            return result

        pending = np.zeros(TRADE_ROW_FEATURES, dtype=np.float32)
        if game.state.pending_trades:
            offer = next(iter(game.state.pending_trades.values()))
            pending = row(
                offer["from_player"],
                offer["to_player"],
                offer["give_properties"],
                offer["want_properties"],
                offer["give_money"],
                offer["want_money"],
            )
        rows = np.zeros((TRADE_CANDIDATE_COUNT, TRADE_ROW_FEATURES), dtype=np.float32)
        candidate_mask = np.zeros(TRADE_CANDIDATE_COUNT, dtype=np.int8)
        for slot, action in enumerate(encoder.get_trade_candidates(game, player_id)):
            rows[slot] = row(
                action.player_id,
                action.to_player,
                action.give_properties,
                action.want_properties,
                action.give_money,
                action.want_money,
            )
            candidate_mask[slot] = 1
        used = game.state.trade_targets_this_turn
        proposal = np.zeros(11, dtype=np.float32)
        proposal[0] = max(0, 2 - len(used))
        proposal[1] = float(
            game.state.phase == "asset_management"
            and not game.state.roll_owed
            and game.decision_player == game.current_player
        )
        for pid in used:
            proposal[2 + pid] = 1.0
        proposal[10] = game.state.revision
        return {
            "pending_offer": pending,
            "candidate_rows": rows,
            "candidate_mask": candidate_mask,
            "proposal_state": proposal,
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
        flat_parts.append(np.array([player_state["position"]], dtype=np.float32) / (BOARD_SIZE - 1))
        flat_parts.append(np.array([player_state["in_jail"]], dtype=np.float32))
        flat_parts.append(np.array([player_state["jail_turns"]], dtype=np.float32) / MAX_JAIL_TURNS)
        flat_parts.append(np.array([player_state["jail_cards"]], dtype=np.float32) / MAX_JAIL_CARDS)
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

    flat_parts.append(np.asarray(obs["decision_state"], dtype=np.float32))
    trade_context = obs.get("trade_context")
    if isinstance(trade_context, dict):
        for key in ("pending_offer", "candidate_rows", "candidate_mask", "proposal_state"):
            value = trade_context.get(key)
            if isinstance(value, np.ndarray):
                flat_parts.append(value.astype(np.float32).flatten())
    return np.concatenate(flat_parts)


class IncrementalObservationEncoder:
    """Observation encoder with incremental update support.

    Wraps ObservationEncoder and caches observation arrays per player.
    Uses two strategies for efficient change detection:

    1. **Player-identity tracking**: opponent_states only changes when a
       different player acts. By tracking the last-encoded player_id,
       consecutive encodes for the same player skip the expensive
       opponent_states recomputation entirely (O(1) check).

    2. **Lightweight snapshots**: board_state and game_state use cheap
       tuple snapshots of scalar fields (owner, mortgaged, houses per
       property; turn_number, houses_remaining, etc.) for change detection.

    player_state is always recomputed (money changes almost every step,
    and encoding is cheap relative to the snapshot cost).

    Attributes:
        num_players: Total number of players in the game.
        enable_trades: Whether trade context is included.
    """

    def __init__(
        self,
        num_players: int,
        enable_trades: bool = False,
        *,
        rules_id: str = "foundation-v1",
    ) -> None:
        """Initialize the incremental observation encoder.

        Args:
            num_players: Total number of players (2-8).
            enable_trades: If True, include trade context in observations.
        """
        self._base = ObservationEncoder(num_players, enable_trades, rules_id=rules_id)
        self.num_players = num_players
        self.max_opponents = num_players - 1
        self.enable_trades = rules_id == "foundation-trade-v1"

        # Per-player cached observations
        self._cache: dict[int, dict[str, Any]] = {}

        # Track which player was last encoded — consecutive encodes for the
        # same player can skip opponent_states (no other player acted).
        self._last_encoded_pid: int | None = None

        # Lightweight snapshots for board and game state
        self._board_snap: dict[int, tuple[Any, ...]] = {}
        self._game_snap: tuple[Any, ...] | None = None

        # Per-section hit/miss counters
        self._section_hits: dict[str, int] = {
            "player_state": 0,
            "opponent_states": 0,
            "board_state": 0,
            "game_state": 0,
        }
        self._section_misses: dict[str, int] = {
            "player_state": 0,
            "opponent_states": 0,
            "board_state": 0,
            "game_state": 0,
        }

    def get_observation_space(self) -> spaces.Dict:
        """Define the gymnasium observation space (delegates to base encoder)."""
        return self._base.get_observation_space()

    def reset(self) -> None:
        """Clear all caches. Must be called on env.reset()."""
        self._cache.clear()
        self._board_snap.clear()
        self._game_snap = None
        self._last_encoded_pid = None

    def encode(
        self, game: MonopolyGame, player_id: int, pending_trade_response: bool = False
    ) -> dict[str, Any]:
        # Compatibility class: correctness reference, caching intentionally disabled.
        return self._base.encode(game, player_id, pending_trade_response)

    def _snap_board(self, game: MonopolyGame, player_id: int) -> tuple[Any, ...]:
        """Snapshot fields that board_state encoding depends on.

        Includes player_id since board encodes ownership relative to self.
        """
        parts: list[tuple[Any, ...]] = []
        for pos in PROPERTY_POSITIONS:
            prop = game.property_manager.get(pos)
            if prop is None:
                parts.append((None, False, 0))
            else:
                parts.append((prop.owner, prop.mortgaged, prop.houses))
        return (player_id, tuple(parts))

    def _snap_game(self, game: MonopolyGame) -> tuple[Any, ...]:
        """Snapshot fields that game_state encoding depends on."""
        return (
            game.turn_number,
            game.houses_remaining,
            game.hotels_remaining,
            game.last_roll,
        )

    @property
    def cache_stats(self) -> dict[str, Any]:
        """Return cache hit/miss statistics per section.

        Returns:
            Dictionary with hit rates and per-section breakdown.
        """
        stats: dict[str, Any] = {}
        for section in self._section_hits:
            total = self._section_hits[section] + self._section_misses[section]
            stats[section] = {
                "hits": self._section_hits[section],
                "misses": self._section_misses[section],
                "hit_rate": (self._section_hits[section] / total if total > 0 else 0.0),
            }
        total_hits = sum(self._section_hits.values())
        total_misses = sum(self._section_misses.values())
        total = total_hits + total_misses
        stats["overall"] = {
            "hits": total_hits,
            "misses": total_misses,
            "hit_rate": total_hits / total if total > 0 else 0.0,
        }
        return stats


def get_flat_observation_size(
    num_players: int,
    enable_trades: bool = False,
    *,
    rules_id: str = "foundation-v1",
) -> int:
    from monopoly_engine import MonopolyGame

    return int(
        flatten_observation(
            ObservationEncoder(num_players, enable_trades, rules_id=rules_id).encode(
                MonopolyGame(
                    num_players,
                    seed=0,
                    rules_id=rules_id,
                ),
                0,
            )
        ).size
    )
