"""Feature extraction for the MCTS value network.

Converts MonopolyGame states to feature vectors suitable for the
ValueNetwork. Reuses the ObservationEncoder from monopoly_gym but
excludes the action mask (applied separately to the policy head).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from monopoly_engine.game import MonopolyGame
from monopoly_gym.observation import (
    BOARD_SIZE,
    MAX_JAIL_CARDS,
    MAX_JAIL_TURNS,
    ObservationEncoder,
)


def extract_features(
    game: MonopolyGame,
    player_id: int,
    num_players: int = 4,
) -> NDArray[np.float32]:
    """Extract feature vector from game state for value network input.

    Uses the same encoding as monopoly_gym/observation.py but excludes
    the action mask (which is applied separately to the policy head).

    Feature layout (for 4-player game, 279 total):
        [0:33]    player_state (money, position, jail, properties)
        [33:135]  opponent_states (3 opponents * 34 features)
        [135:275] board_state (28 properties * 5 features)
        [275:279] game_state (turn, houses, hotels, last_roll)

    Args:
        game: The game state to encode.
        player_id: The player perspective.
        num_players: Total number of players.

    Returns:
        Feature vector of shape (feature_size,), all values in [0, 1].
    """
    encoder = ObservationEncoder(num_players)
    obs = encoder.encode(game, player_id)

    flat_parts: list[NDArray[np.float32]] = []

    # Player state
    ps = obs["player_state"]
    assert isinstance(ps, dict)
    flat_parts.append(ps["money"].flatten().astype(np.float32))
    flat_parts.append(np.array([float(ps["position"]) / (BOARD_SIZE - 1)], dtype=np.float32))
    flat_parts.append(np.array([float(ps["in_jail"])], dtype=np.float32))
    flat_parts.append(np.array([float(ps["jail_turns"]) / MAX_JAIL_TURNS], dtype=np.float32))
    flat_parts.append(np.array([float(ps["jail_cards"]) / MAX_JAIL_CARDS], dtype=np.float32))
    flat_parts.append(ps["properties_owned"].astype(np.float32))

    # Opponent states (already normalized in [0, 1])
    opponent_states = obs["opponent_states"]
    assert isinstance(opponent_states, np.ndarray)
    flat_parts.append(opponent_states.flatten().astype(np.float32))

    # Board state (already normalized in [0, 1])
    board_state = obs["board_state"]
    assert isinstance(board_state, np.ndarray)
    flat_parts.append(board_state.flatten().astype(np.float32))

    # Game state
    gs = obs["game_state"]
    assert isinstance(gs, dict)
    flat_parts.append(gs["turn_number"].flatten().astype(np.float32))
    flat_parts.append(gs["houses_remaining"].flatten().astype(np.float32))
    flat_parts.append(gs["hotels_remaining"].flatten().astype(np.float32))
    flat_parts.append(gs["last_roll"].flatten().astype(np.float32))

    # NOTE: action_mask is NOT included in features
    return np.concatenate(flat_parts)


def get_feature_size(num_players: int = 4) -> int:
    """Get the feature vector size for a given number of players.

    This is get_flat_observation_size(num_players) minus the action mask
    size (149). The action mask is applied separately to the policy head.

    Args:
        num_players: Number of players (2-8).

    Returns:
        Feature vector dimension.
    """
    # Frozen experimental feature layout; learned search is not certified.
    # Do not infer its size from the independently versioned Gym observation.
    return 33 + (num_players - 1) * 34 + 140 + 4
