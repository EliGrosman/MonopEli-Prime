"""Comprehensive tests for the observation module.

This module tests the ObservationEncoder class including:
- Observation space structure and dimensions
- Encoding of player state, opponent states, board state, game state
- Normalization bounds
- Flattening and size calculations
- Edge cases and multi-player scenarios
"""

import numpy as np
import pytest
from gymnasium import spaces
from hypothesis import given, settings
from hypothesis import strategies as st

from monopoly_engine import MonopolyGame
from monopoly_gym.observation import (
    BOARD_SIZE,
    MAX_DICE_SUM,
    MAX_HOTELS,
    MAX_HOUSES,
    MAX_JAIL_CARDS,
    MAX_JAIL_TURNS,
    MAX_MONEY,
    MAX_TURN_NUMBER,
    NUM_PROPERTIES,
    PROPERTY_POS_TO_IDX,
    PROPERTY_POSITIONS,
    ObservationEncoder,
    flatten_observation,
    get_flat_observation_size,
)


class TestObservationConstants:
    """Tests for observation space constants."""

    def test_max_money_constant(self) -> None:
        """MAX_MONEY should be a reasonable value for normalization."""
        assert MAX_MONEY == 10000
        assert MAX_MONEY > 0

    def test_board_size_constant(self) -> None:
        """BOARD_SIZE should be 40."""
        assert BOARD_SIZE == 40

    def test_num_properties_constant(self) -> None:
        """NUM_PROPERTIES should be 28 (all buyable)."""
        assert NUM_PROPERTIES == 28

    def test_property_positions_count(self) -> None:
        """Should have 28 property positions."""
        assert len(PROPERTY_POSITIONS) == 28

    def test_property_pos_to_idx_mapping(self) -> None:
        """PROPERTY_POS_TO_IDX should map all positions correctly."""
        for idx, pos in enumerate(PROPERTY_POSITIONS):
            assert PROPERTY_POS_TO_IDX[pos] == idx

    def test_property_positions_valid(self) -> None:
        """All property positions should be valid board positions."""
        for pos in PROPERTY_POSITIONS:
            assert 0 <= pos <= 39

    def test_max_jail_constants(self) -> None:
        """Jail-related constants should be correct."""
        assert MAX_JAIL_TURNS == 3
        assert MAX_JAIL_CARDS == 2

    def test_resource_constants(self) -> None:
        """Resource constants should match game rules."""
        assert MAX_HOUSES == 32
        assert MAX_HOTELS == 12

    def test_max_dice_sum(self) -> None:
        """Max dice sum should be 12 (6+6)."""
        assert MAX_DICE_SUM == 12


class TestObservationEncoderInit:
    """Tests for ObservationEncoder initialization."""

    def test_init_valid_player_counts(self) -> None:
        """Should accept 2-8 players."""
        for num_players in range(2, 9):
            encoder = ObservationEncoder(num_players)
            assert encoder.num_players == num_players
            assert encoder.max_opponents == num_players - 1

    def test_init_invalid_player_count_low(self) -> None:
        """Should reject fewer than 2 players."""
        with pytest.raises(ValueError, match="between 2 and 8"):
            ObservationEncoder(1)

    def test_init_invalid_player_count_high(self) -> None:
        """Should reject more than 8 players."""
        with pytest.raises(ValueError, match="between 2 and 8"):
            ObservationEncoder(9)

    def test_init_zero_players(self) -> None:
        """Should reject 0 players."""
        with pytest.raises(ValueError):
            ObservationEncoder(0)

    def test_init_negative_players(self) -> None:
        """Should reject negative players."""
        with pytest.raises(ValueError):
            ObservationEncoder(-1)


class TestGetObservationSpace:
    """Tests for observation space definition."""

    def test_returns_dict_space(self) -> None:
        """Should return a gymnasium Dict space."""
        encoder = ObservationEncoder(4)
        space = encoder.get_observation_space()
        assert isinstance(space, spaces.Dict)

    def test_has_required_keys(self) -> None:
        """Observation space should have all required keys."""
        encoder = ObservationEncoder(4)
        space = encoder.get_observation_space()

        required_keys = [
            "player_state",
            "opponent_states",
            "board_state",
            "game_state",
            "action_mask",
        ]
        for key in required_keys:
            assert key in space.spaces, f"Missing key: {key}"

    def test_player_state_structure(self) -> None:
        """Player state should have correct subspaces."""
        encoder = ObservationEncoder(4)
        space = encoder.get_observation_space()

        player_state = space.spaces["player_state"]
        assert isinstance(player_state, spaces.Dict)

        expected_keys = [
            "money",
            "position",
            "in_jail",
            "jail_turns",
            "jail_cards",
            "properties_owned",
        ]
        for key in expected_keys:
            assert key in player_state.spaces, f"Missing player state key: {key}"

    def test_player_state_money_shape(self) -> None:
        """Money should be Box(0, 1, shape=(1,))."""
        encoder = ObservationEncoder(4)
        space = encoder.get_observation_space()

        money_space = space.spaces["player_state"].spaces["money"]
        assert isinstance(money_space, spaces.Box)
        assert money_space.shape == (1,)
        assert money_space.low[0] == 0.0
        assert money_space.high[0] == 1.0

    def test_player_state_position_space(self) -> None:
        """Position should be Discrete(40)."""
        encoder = ObservationEncoder(4)
        space = encoder.get_observation_space()

        position_space = space.spaces["player_state"].spaces["position"]
        assert isinstance(position_space, spaces.Discrete)
        assert position_space.n == BOARD_SIZE

    def test_player_state_properties_owned_space(self) -> None:
        """Properties owned should be MultiBinary(28)."""
        encoder = ObservationEncoder(4)
        space = encoder.get_observation_space()

        props_space = space.spaces["player_state"].spaces["properties_owned"]
        assert isinstance(props_space, spaces.MultiBinary)
        assert props_space.n == NUM_PROPERTIES

    def test_opponent_states_shape(self) -> None:
        """Opponent states shape should be (max_opponents, features)."""
        for num_players in range(2, 9):
            encoder = ObservationEncoder(num_players)
            space = encoder.get_observation_space()

            opp_space = space.spaces["opponent_states"]
            assert isinstance(opp_space, spaces.Box)

            expected_shape = (num_players - 1, 6 + NUM_PROPERTIES)
            assert opp_space.shape == expected_shape

    def test_board_state_shape(self) -> None:
        """Board state should be (28, 5)."""
        encoder = ObservationEncoder(4)
        space = encoder.get_observation_space()

        board_space = space.spaces["board_state"]
        assert isinstance(board_space, spaces.Box)
        assert board_space.shape == (NUM_PROPERTIES, 5)

    def test_game_state_structure(self) -> None:
        """Game state should have correct subspaces."""
        encoder = ObservationEncoder(4)
        space = encoder.get_observation_space()

        game_state = space.spaces["game_state"]
        assert isinstance(game_state, spaces.Dict)

        expected_keys = [
            "turn_number",
            "houses_remaining",
            "hotels_remaining",
            "last_roll",
        ]
        for key in expected_keys:
            assert key in game_state.spaces, f"Missing game state key: {key}"

    def test_action_mask_space(self) -> None:
        """Action mask should be MultiBinary(158)."""
        encoder = ObservationEncoder(4)
        space = encoder.get_observation_space()

        mask_space = space.spaces["action_mask"]
        assert isinstance(mask_space, spaces.MultiBinary)
        assert mask_space.n == 158


class TestEncode:
    """Tests for encoding game state to observations."""

    def test_encode_returns_dict(self) -> None:
        """Encode should return a dictionary."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        obs = encoder.encode(game, player_id=0)
        assert isinstance(obs, dict)

    def test_encode_has_required_keys(self) -> None:
        """Encoded observation should have all required keys."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        obs = encoder.encode(game, player_id=0)

        required_keys = [
            "player_state",
            "opponent_states",
            "board_state",
            "game_state",
            "action_mask",
        ]
        for key in required_keys:
            assert key in obs, f"Missing key: {key}"

    def test_encode_invalid_player_id(self) -> None:
        """Should raise for invalid player_id."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        with pytest.raises(ValueError, match="Invalid player_id"):
            encoder.encode(game, player_id=4)

        with pytest.raises(ValueError, match="Invalid player_id"):
            encoder.encode(game, player_id=-1)


class TestEncodePlayerState:
    """Tests for encoding player state."""

    def test_player_state_money_normalized(self) -> None:
        """Money should be normalized to [0, 1]."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)
        game.players[0].money = 5000

        obs = encoder.encode(game, player_id=0)
        money = obs["player_state"]["money"]

        assert money.dtype == np.float32
        assert money.shape == (1,)
        assert 0 <= money[0] <= 1
        assert money[0] == pytest.approx(5000 / MAX_MONEY)

    def test_player_state_money_capped(self) -> None:
        """Money over MAX_MONEY should be capped at 1.0."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)
        game.players[0].money = 20000  # Over MAX_MONEY

        obs = encoder.encode(game, player_id=0)
        money = obs["player_state"]["money"]

        assert money[0] == 1.0

    def test_player_state_position(self) -> None:
        """Position should be the raw board position."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)
        game.players[0].position = 25

        obs = encoder.encode(game, player_id=0)
        position = obs["player_state"]["position"]

        assert position == 25

    def test_player_state_in_jail(self) -> None:
        """In jail should be 0 or 1."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        # Not in jail
        obs = encoder.encode(game, player_id=0)
        assert obs["player_state"]["in_jail"] == 0

        # In jail
        game.players[0].in_jail = True
        obs = encoder.encode(game, player_id=0)
        assert obs["player_state"]["in_jail"] == 1

    def test_player_state_jail_turns(self) -> None:
        """Jail turns should be capped at MAX_JAIL_TURNS."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)
        game.players[0].jail_turns = 5  # Over max

        obs = encoder.encode(game, player_id=0)
        jail_turns = obs["player_state"]["jail_turns"]

        assert jail_turns == MAX_JAIL_TURNS

    def test_player_state_jail_cards(self) -> None:
        """Jail cards should be capped at MAX_JAIL_CARDS."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)
        game.players[0].jail_cards = 5  # Over max

        obs = encoder.encode(game, player_id=0)
        jail_cards = obs["player_state"]["jail_cards"]

        assert jail_cards == MAX_JAIL_CARDS

    def test_player_state_properties_owned_empty(self) -> None:
        """Properties owned should be all zeros initially."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        obs = encoder.encode(game, player_id=0)
        props = obs["player_state"]["properties_owned"]

        assert props.dtype == np.int8
        assert props.shape == (NUM_PROPERTIES,)
        assert props.sum() == 0

    def test_player_state_properties_owned_with_property(self) -> None:
        """Properties owned should reflect ownership."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        # Give player Mediterranean (position 1)
        game.property_manager.properties[1].owner = 0

        obs = encoder.encode(game, player_id=0)
        props = obs["player_state"]["properties_owned"]

        idx = PROPERTY_POS_TO_IDX[1]
        assert props[idx] == 1
        assert props.sum() == 1


class TestEncodeOpponentStates:
    """Tests for encoding opponent states."""

    def test_opponent_states_shape(self) -> None:
        """Opponent states should have correct shape."""
        for num_players in range(2, 9):
            encoder = ObservationEncoder(num_players)
            game = MonopolyGame(num_players=num_players, seed=42)

            obs = encoder.encode(game, player_id=0)
            opp_states = obs["opponent_states"]

            expected_shape = (num_players - 1, 6 + NUM_PROPERTIES)
            assert opp_states.shape == expected_shape

    def test_opponent_states_excludes_current_player(self) -> None:
        """Opponent states should not include the current player."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        # Give distinct money to each player
        for i in range(4):
            game.players[i].money = (i + 1) * 1000

        obs = encoder.encode(game, player_id=0)
        opp_states = obs["opponent_states"]

        # Player 0 has $1000, should not appear in opponent states
        # Opponent 0 should be player 1 with $2000
        expected_money_0 = 2000 / MAX_MONEY
        assert opp_states[0, 0] == pytest.approx(expected_money_0)

    def test_opponent_states_features(self) -> None:
        """Opponent states should encode all features correctly."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        game.players[1].money = 3000
        game.players[1].position = 20
        game.players[1].in_jail = True
        game.players[1].jail_turns = 2
        game.players[1].jail_cards = 1
        game.players[1].bankrupt = False

        obs = encoder.encode(game, player_id=0)
        opp = obs["opponent_states"][0]  # Player 1 is first opponent

        # Feature 0: money normalized
        assert opp[0] == pytest.approx(3000 / MAX_MONEY)
        # Feature 1: position normalized
        assert opp[1] == pytest.approx(20 / (BOARD_SIZE - 1))
        # Feature 2: in_jail
        assert opp[2] == 1.0
        # Feature 3: jail_turns normalized
        assert opp[3] == pytest.approx(2 / MAX_JAIL_TURNS)
        # Feature 4: jail_cards normalized
        assert opp[4] == pytest.approx(1 / MAX_JAIL_CARDS)
        # Feature 5: bankrupt
        assert opp[5] == 0.0

    def test_opponent_states_bankrupt_player(self) -> None:
        """Bankrupt opponents should have bankrupt flag set."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        game.players[1].bankrupt = True

        obs = encoder.encode(game, player_id=0)
        opp = obs["opponent_states"][0]

        assert opp[5] == 1.0  # Bankrupt flag

    def test_opponent_states_property_ownership(self) -> None:
        """Opponent property ownership should be encoded."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        # Give player 1 a property
        game.property_manager.properties[1].owner = 1  # Mediterranean

        obs = encoder.encode(game, player_id=0)
        opp = obs["opponent_states"][0]

        # Property ownership starts at index 6
        prop_idx = PROPERTY_POS_TO_IDX[1]
        assert opp[6 + prop_idx] == 1.0


class TestEncodeBoardState:
    """Tests for encoding board state."""

    def test_board_state_shape(self) -> None:
        """Board state should be (28, 5)."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        obs = encoder.encode(game, player_id=0)
        board = obs["board_state"]
        assert isinstance(board, np.ndarray)

        assert board.shape == (NUM_PROPERTIES, 5)
        assert board.dtype == np.float32

    def test_board_state_unowned_property(self) -> None:
        """Unowned property should have one-hot [1, 0, 0, 0]."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        obs = encoder.encode(game, player_id=0)
        board = obs["board_state"]
        assert isinstance(board, np.ndarray)

        # First property (Mediterranean, index 0)
        assert board[0, 0] == 1.0  # Unowned
        assert board[0, 1] == 0.0  # Not self
        assert board[0, 2] == 0.0  # Not opponent
        assert board[0, 3] == 0.0  # Not mortgaged

    def test_board_state_self_owned_property(self) -> None:
        """Self-owned property should have one-hot [0, 1, 0, 0]."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        game.property_manager.properties[1].owner = 0  # Player 0 owns Mediterranean

        obs = encoder.encode(game, player_id=0)
        board = obs["board_state"]
        assert isinstance(board, np.ndarray)

        idx = PROPERTY_POS_TO_IDX[1]
        assert board[idx, 0] == 0.0  # Not unowned
        assert board[idx, 1] == 1.0  # Self owns
        assert board[idx, 2] == 0.0  # Not opponent
        assert board[idx, 3] == 0.0  # Not mortgaged

    def test_board_state_opponent_owned_property(self) -> None:
        """Opponent-owned property should have one-hot [0, 0, 1, 0]."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        game.property_manager.properties[1].owner = 1  # Player 1 owns Mediterranean

        obs = encoder.encode(game, player_id=0)
        board = obs["board_state"]
        assert isinstance(board, np.ndarray)

        idx = PROPERTY_POS_TO_IDX[1]
        assert board[idx, 0] == 0.0  # Not unowned
        assert board[idx, 1] == 0.0  # Not self
        assert board[idx, 2] == 1.0  # Opponent owns
        assert board[idx, 3] == 0.0  # Not mortgaged

    def test_board_state_mortgaged_property(self) -> None:
        """Mortgaged property should have one-hot [0, 0, 0, 1]."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        game.property_manager.properties[1].owner = 0
        game.property_manager.properties[1].mortgaged = True

        obs = encoder.encode(game, player_id=0)
        board = obs["board_state"]
        assert isinstance(board, np.ndarray)

        idx = PROPERTY_POS_TO_IDX[1]
        assert board[idx, 0] == 0.0
        assert board[idx, 1] == 0.0
        assert board[idx, 2] == 0.0
        assert board[idx, 3] == 1.0  # Mortgaged

    def test_board_state_houses_normalized(self) -> None:
        """Houses should be normalized to [0, 1]."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        # Set up a monopoly and add houses
        game.property_manager.properties[1].owner = 0  # Mediterranean
        game.property_manager.properties[3].owner = 0  # Baltic
        game.property_manager.properties[1].houses = 3

        obs = encoder.encode(game, player_id=0)
        board = obs["board_state"]
        assert isinstance(board, np.ndarray)

        idx = PROPERTY_POS_TO_IDX[1]
        assert board[idx, 4] == pytest.approx(3 / 5.0)

    def test_board_state_hotel(self) -> None:
        """Hotel (5 houses) should normalize to 1.0."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        game.property_manager.properties[1].owner = 0
        game.property_manager.properties[1].houses = 5  # Hotel

        obs = encoder.encode(game, player_id=0)
        board = obs["board_state"]
        assert isinstance(board, np.ndarray)

        idx = PROPERTY_POS_TO_IDX[1]
        assert board[idx, 4] == 1.0


class TestEncodeGameState:
    """Tests for encoding game state."""

    def test_game_state_structure(self) -> None:
        """Game state should have all required keys."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        obs = encoder.encode(game, player_id=0)
        game_state = obs["game_state"]

        assert "turn_number" in game_state
        assert "houses_remaining" in game_state
        assert "hotels_remaining" in game_state
        assert "last_roll" in game_state

    def test_game_state_turn_number_normalized(self) -> None:
        """Turn number should be normalized."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)
        game.turn_number = 500

        obs = encoder.encode(game, player_id=0)
        turn = obs["game_state"]["turn_number"]

        assert turn.shape == (1,)
        assert turn[0] == pytest.approx(500 / MAX_TURN_NUMBER)

    def test_game_state_turn_number_capped(self) -> None:
        """Turn number over max should be capped at 1.0."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)
        game.turn_number = 2000  # Over MAX_TURN_NUMBER

        obs = encoder.encode(game, player_id=0)
        turn = obs["game_state"]["turn_number"]

        assert turn[0] == 1.0

    def test_game_state_houses_remaining(self) -> None:
        """Houses remaining should be normalized."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        obs = encoder.encode(game, player_id=0)
        houses = obs["game_state"]["houses_remaining"]

        assert houses.shape == (1,)
        assert houses[0] == pytest.approx(game.houses_remaining / MAX_HOUSES)

    def test_game_state_hotels_remaining(self) -> None:
        """Hotels remaining should be normalized."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        obs = encoder.encode(game, player_id=0)
        hotels = obs["game_state"]["hotels_remaining"]

        assert hotels.shape == (1,)
        assert hotels[0] == pytest.approx(game.hotels_remaining / MAX_HOTELS)

    def test_game_state_last_roll_none(self) -> None:
        """Last roll should be 0 if no roll."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)
        game.last_roll = None

        obs = encoder.encode(game, player_id=0)
        last_roll = obs["game_state"]["last_roll"]

        assert last_roll.shape == (1,)
        assert last_roll[0] == 0.0

    def test_game_state_last_roll_normalized(self) -> None:
        """Last roll should be sum normalized by MAX_DICE_SUM."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)
        game.last_roll = (4, 5)  # Sum = 9

        obs = encoder.encode(game, player_id=0)
        last_roll = obs["game_state"]["last_roll"]

        assert last_roll[0] == pytest.approx(9 / MAX_DICE_SUM)


class TestEncodeActionMask:
    """Tests for action mask encoding."""

    def test_action_mask_shape(self) -> None:
        """Action mask should be shape (158,)."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        obs = encoder.encode(game, player_id=0)
        mask = obs["action_mask"]
        assert isinstance(mask, np.ndarray)

        assert mask.shape == (158,)
        assert mask.dtype == np.int8

    def test_action_mask_binary(self) -> None:
        """Action mask should be binary (0 or 1)."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        obs = encoder.encode(game, player_id=0)
        mask = obs["action_mask"]

        assert np.all((mask == 0) | (mask == 1))


class TestFlattenObservation:
    """Tests for flattening observations."""

    def test_flatten_returns_1d_array(self) -> None:
        """Flattened observation should be 1D."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        obs = encoder.encode(game, player_id=0)
        flat = flatten_observation(obs)

        assert flat.ndim == 1

    def test_flatten_dtype(self) -> None:
        """Flattened observation should be float32."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        obs = encoder.encode(game, player_id=0)
        flat = flatten_observation(obs)

        assert flat.dtype == np.float32

    def test_flatten_size_matches_calculation(self) -> None:
        """Flattened size should match get_flat_observation_size."""
        for num_players in range(2, 9):
            encoder = ObservationEncoder(num_players)
            game = MonopolyGame(num_players=num_players, seed=42)

            obs = encoder.encode(game, player_id=0)
            flat = flatten_observation(obs)

            expected_size = get_flat_observation_size(num_players)
            assert flat.shape[0] == expected_size, f"Mismatch for {num_players} players"

    def test_flatten_normalized(self) -> None:
        """Most flattened values should be in [0, 1]."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        obs = encoder.encode(game, player_id=0)
        flat = flatten_observation(obs)

        # Check bounds (allowing some tolerance for edge cases)
        assert flat.min() >= 0.0
        assert flat.max() <= 1.0


class TestGetFlatObservationSize:
    """Tests for flat observation size calculation."""

    def test_size_for_2_players(self) -> None:
        """Calculate size for 2 players."""
        size = get_flat_observation_size(2)
        # player_state: 33
        # opponent_states: 1 * 34 = 34
        # board_state: 140
        # game_state: 4
        # action_mask: 158
        expected = 33 + 34 + 140 + 4 + 158 + 13
        assert size == expected

    def test_size_for_4_players(self) -> None:
        """Calculate size for 4 players."""
        size = get_flat_observation_size(4)
        # opponent_states: 3 * 34 = 102
        expected = 33 + 102 + 140 + 4 + 158 + 13
        assert size == expected

    def test_size_for_8_players(self) -> None:
        """Calculate size for 8 players."""
        size = get_flat_observation_size(8)
        # opponent_states: 7 * 34 = 238
        expected = 33 + 238 + 140 + 4 + 158 + 13
        assert size == expected

    def test_size_invalid_player_count(self) -> None:
        """Should raise for invalid player count."""
        with pytest.raises(ValueError):
            get_flat_observation_size(1)

        with pytest.raises(ValueError):
            get_flat_observation_size(9)

    @pytest.mark.parametrize("num_players", range(2, 9))
    def test_size_increases_with_players(self, num_players: int) -> None:
        """Size should increase with number of players."""
        if num_players > 2:
            prev_size = get_flat_observation_size(num_players - 1)
            curr_size = get_flat_observation_size(num_players)
            assert curr_size > prev_size


class TestNormalizationBounds:
    """Tests for observation value normalization bounds."""

    def test_all_components_bounded(self) -> None:
        """All observation components should be bounded."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        # Set up various game states
        game.players[0].money = 15000  # Over max
        game.turn_number = 2000  # Over max

        obs = encoder.encode(game, player_id=0)

        # Check player_state
        player_state = obs["player_state"]
        assert isinstance(player_state, dict)
        assert 0 <= player_state["money"][0] <= 1  # type: ignore[index]
        assert 0 <= player_state["position"] <= 39  # type: ignore[operator]

        # Check opponent_states
        opponent_states = obs["opponent_states"]
        assert isinstance(opponent_states, np.ndarray)
        assert np.all(opponent_states >= 0)
        assert np.all(opponent_states <= 1)

        # Check board_state
        board_state = obs["board_state"]
        assert isinstance(board_state, np.ndarray)
        assert np.all(board_state >= 0)
        assert np.all(board_state <= 1)

        # Check game_state
        game_state = obs["game_state"]
        assert isinstance(game_state, dict)
        assert 0 <= game_state["turn_number"][0] <= 1  # type: ignore[index]
        assert 0 <= game_state["houses_remaining"][0] <= 1  # type: ignore[index]
        assert 0 <= game_state["hotels_remaining"][0] <= 1  # type: ignore[index]
        assert 0 <= game_state["last_roll"][0] <= 1  # type: ignore[index]


class TestPropertyBasedObservation:
    """Property-based tests using hypothesis."""

    @given(num_players=st.integers(2, 8))
    @settings(max_examples=7)
    def test_encoder_valid_for_all_player_counts(self, num_players: int) -> None:
        """Encoder should work for all valid player counts."""
        encoder = ObservationEncoder(num_players)
        game = MonopolyGame(num_players=num_players, seed=42)

        # Should not raise
        obs = encoder.encode(game, player_id=0)
        assert obs is not None

    @given(
        num_players=st.integers(2, 8),
        player_id=st.integers(0, 7),
    )
    @settings(max_examples=20)
    def test_encode_valid_player_ids(self, num_players: int, player_id: int) -> None:
        """Encode should work for valid player IDs."""
        if player_id >= num_players:
            return  # Skip invalid combinations

        encoder = ObservationEncoder(num_players)
        game = MonopolyGame(num_players=num_players, seed=42)

        # Should not raise
        obs = encoder.encode(game, player_id=player_id)
        assert obs is not None

    @given(
        money=st.integers(0, 50000),
        position=st.integers(0, 39),
        jail_turns=st.integers(0, 10),
        jail_cards=st.integers(0, 5),
    )
    @settings(max_examples=50)
    def test_encode_handles_various_player_states(
        self,
        money: int,
        position: int,
        jail_turns: int,
        jail_cards: int,
    ) -> None:
        """Encoder should handle various player states."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        game.players[0].money = money
        game.players[0].position = position
        game.players[0].jail_turns = jail_turns
        game.players[0].jail_cards = jail_cards

        # Should not raise
        obs = encoder.encode(game, player_id=0)
        assert obs is not None

        # Values should be bounded
        player_state = obs["player_state"]
        assert isinstance(player_state, dict)
        assert 0 <= player_state["money"][0] <= 1  # type: ignore[index]


class TestMultiPlayerScenarios:
    """Tests for multi-player scenarios."""

    def test_two_player_game(self) -> None:
        """Two-player game should work correctly."""
        encoder = ObservationEncoder(2)
        game = MonopolyGame(num_players=2, seed=42)

        obs = encoder.encode(game, player_id=0)
        opp_states = obs["opponent_states"]
        assert isinstance(opp_states, np.ndarray)

        # Should have 1 opponent
        assert opp_states.shape == (1, 6 + NUM_PROPERTIES)

    def test_eight_player_game(self) -> None:
        """Eight-player game should work correctly."""
        encoder = ObservationEncoder(8)
        game = MonopolyGame(num_players=8, seed=42)

        obs = encoder.encode(game, player_id=0)
        opp_states = obs["opponent_states"]
        assert isinstance(opp_states, np.ndarray)

        # Should have 7 opponents
        assert opp_states.shape == (7, 6 + NUM_PROPERTIES)

    def test_different_perspectives(self) -> None:
        """Different players should get different observations."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        # Give player 0 a property
        game.property_manager.properties[1].owner = 0

        obs0 = encoder.encode(game, player_id=0)
        obs1 = encoder.encode(game, player_id=1)
        board0 = obs0["board_state"]
        board1 = obs1["board_state"]
        assert isinstance(board0, np.ndarray)
        assert isinstance(board1, np.ndarray)

        # Player 0's self-owned property appears as opponent-owned to player 1
        idx = PROPERTY_POS_TO_IDX[1]

        # For player 0, position 1 is self-owned
        assert board0[idx, 1] == 1.0  # Self

        # For player 1, position 1 is opponent-owned
        assert board1[idx, 2] == 1.0  # Opponent


class TestEdgeCases:
    """Tests for edge cases."""

    def test_all_properties_owned(self) -> None:
        """Observation should handle all properties being owned."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        # Give all properties to player 0
        for pos in PROPERTY_POSITIONS:
            game.property_manager.properties[pos].owner = 0

        obs = encoder.encode(game, player_id=0)
        board_state = obs["board_state"]
        assert isinstance(board_state, np.ndarray)

        # All properties should show as self-owned
        assert board_state[:, 1].sum() == NUM_PROPERTIES

    def test_no_houses_remaining(self) -> None:
        """Observation should handle no houses remaining."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)
        game.houses_remaining = 0

        obs = encoder.encode(game, player_id=0)
        game_state = obs["game_state"]
        assert isinstance(game_state, dict)
        assert game_state["houses_remaining"][0] == 0.0  # type: ignore[index]

    def test_no_hotels_remaining(self) -> None:
        """Observation should handle no hotels remaining."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)
        game.hotels_remaining = 0

        obs = encoder.encode(game, player_id=0)
        game_state = obs["game_state"]
        assert isinstance(game_state, dict)
        assert game_state["hotels_remaining"][0] == 0.0  # type: ignore[index]

    def test_all_players_bankrupt_except_one(self) -> None:
        """Observation should handle multiple bankrupt players."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        for i in range(1, 4):
            game.players[i].bankrupt = True

        obs = encoder.encode(game, player_id=0)
        opp_states = obs["opponent_states"]
        assert isinstance(opp_states, np.ndarray)

        # All opponents should be marked bankrupt
        for i in range(3):
            assert opp_states[i, 5] == 1.0
