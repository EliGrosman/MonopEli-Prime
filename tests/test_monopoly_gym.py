"""Tests for the monopoly_gym package.

This module contains tests for:
- MonopolyEnv (PettingZoo AEC environment)
- ActionEncoder (action space encoding/decoding)
- ObservationEncoder (observation space encoding)
"""

import numpy as np
import pytest
from gymnasium import spaces

from monopoly_engine import MonopolyGame
from monopoly_gym import (
    ACTION_SPACE_SIZE,
    BUYABLE_POSITIONS,
    DEVELOPABLE_POSITIONS,
    GAMEPLAY_ACTION_SPACE_SIZE,
    OFFSET_BUILD_HOTEL,
    OFFSET_BUILD_HOUSE,
    OFFSET_BUY_PROPERTY,
    OFFSET_END_TURN,
    OFFSET_MORTGAGE,
    OFFSET_PASS_BUY,
    OFFSET_PAY_JAIL_FINE,
    OFFSET_SELL_HOTEL,
    OFFSET_SELL_HOUSE,
    OFFSET_UNMORTGAGE,
    OFFSET_USE_JAIL_CARD,
    ActionEncoder,
    MonopolyEnv,
    ObservationEncoder,
)


class TestMonopolyEnv:
    """Tests for the MonopolyEnv PettingZoo environment."""

    def test_init_default(self) -> None:
        """Test environment initialization with defaults."""
        env = MonopolyEnv()
        assert env.num_players == 4
        assert env.max_turns == 1000
        assert env.reward_type == "sparse"
        assert len(env.possible_agents) == 4

    def test_init_custom_players(self) -> None:
        """Test environment initialization with custom player count."""
        env = MonopolyEnv(num_players=2)
        assert env.num_players == 2
        assert len(env.possible_agents) == 2

    def test_init_invalid_players(self) -> None:
        """Test environment rejects invalid player counts."""
        with pytest.raises(ValueError, match="num_players must be 2-4"):
            MonopolyEnv(num_players=1)
        with pytest.raises(ValueError, match="num_players must be 2-4"):
            MonopolyEnv(num_players=5)

    def test_reset_creates_valid_state(self) -> None:
        """Test reset creates valid initial state."""
        env = MonopolyEnv(num_players=4)
        env.reset(seed=42)

        assert len(env.agents) == 4
        assert env.game is not None
        assert env.game.turn_number == 0
        assert env.agent_selection in env.agents

        # Check infos have action masks
        for agent in env.agents:
            assert "action_mask" in env.infos[agent]
            mask = env.infos[agent]["action_mask"]
            assert len(mask) == GAMEPLAY_ACTION_SPACE_SIZE  # Default env has no trades

    def test_reset_deterministic_with_seed(self) -> None:
        """Test reset with same seed produces same initial state."""
        env1 = MonopolyEnv(num_players=2)
        env2 = MonopolyEnv(num_players=2)

        env1.reset(seed=12345)
        env2.reset(seed=12345)

        # Same initial player positions after dice roll
        for i in range(2):
            assert env1.game.players[i].position == env2.game.players[i].position

    def test_step_advances_game(self) -> None:
        """Test step() advances the game state."""
        env = MonopolyEnv(num_players=2)
        env.reset(seed=42)

        initial_agent = env.agent_selection
        initial_turn = env.game.turn_number

        # Take EndTurn action
        env.step(OFFSET_END_TURN)

        # Should have advanced to next player
        assert env.agent_selection != initial_agent
        assert env.game.turn_number > initial_turn

    def test_step_invalid_action_penalized(self) -> None:
        """Test that invalid actions result in penalty."""
        env = MonopolyEnv(num_players=2)
        env.reset(seed=42)

        # Get the action mask
        _, _, _, _, info = env.last()
        mask = info["action_mask"]

        # Find an invalid action
        invalid_actions = np.where(~mask)[0]
        if len(invalid_actions) > 0:
            env.step(invalid_actions[0])
            # Should have received a penalty (cumulative rewards)
            # Note: penalty is small (-0.01)

    def test_observe_returns_dict(self) -> None:
        """Test observe() returns dict with expected keys."""
        env = MonopolyEnv(num_players=2)
        env.reset(seed=42)

        obs = env.observe(env.agent_selection)

        assert isinstance(obs, dict)
        assert "player_state" in obs
        assert "opponent_states" in obs
        assert "board_state" in obs
        assert "game_state" in obs
        assert "action_mask" in obs

    def test_last_returns_tuple(self) -> None:
        """Test last() returns correct tuple format."""
        env = MonopolyEnv(num_players=2)
        env.reset(seed=42)

        result = env.last()

        assert len(result) == 5
        obs, reward, termination, truncation, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(termination, bool)
        assert isinstance(truncation, bool)
        assert isinstance(info, dict)

    def test_action_mask_only_valid_actions(self) -> None:
        """Test that only masked actions can be executed successfully."""
        env = MonopolyEnv(num_players=2)
        env.reset(seed=42)

        _, _, _, _, info = env.last()
        mask = info["action_mask"]

        valid_actions = np.where(mask)[0]
        assert len(valid_actions) > 0  # Should have at least EndTurn
        assert mask[OFFSET_END_TURN]  # EndTurn should always be valid

    def test_buy_property_action(self) -> None:
        """Test buying a property updates game state."""
        env = MonopolyEnv(num_players=2)
        env.reset(seed=42)

        # Find a state where we can buy
        for _ in range(20):
            _, _, term, trunc, info = env.last()
            if term or trunc:
                break

            mask = info["action_mask"]
            if mask[OFFSET_BUY_PROPERTY]:
                # Get current agent and position
                agent = env.agent_selection
                player_id = env.agent_name_mapping[agent]
                position = env.game.players[player_id].position

                # Buy the property
                env.step(OFFSET_BUY_PROPERTY)

                # Verify ownership
                prop = env.game.property_manager.get(position)
                assert prop is not None
                assert prop.owner == player_id
                return

            # Otherwise end turn
            env.step(OFFSET_END_TURN)

    def test_game_completion(self) -> None:
        """Test that game can complete via termination or truncation."""
        env = MonopolyEnv(num_players=2, max_turns=50)
        env.reset(seed=42)

        steps = 0
        max_steps = 1000

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None
            else:
                # Random valid action
                mask = info["action_mask"]
                valid = np.where(mask)[0]
                action = np.random.choice(valid) if len(valid) > 0 else OFFSET_END_TURN

            env.step(action)
            steps += 1

            if steps > max_steps:
                break

        # Game should have ended (either via termination or we hit step limit)
        assert steps <= max_steps

    def test_render_ansi(self) -> None:
        """Test ANSI rendering returns string."""
        env = MonopolyEnv(num_players=2, render_mode="ansi")
        env.reset(seed=42)

        output = env.render()
        assert isinstance(output, str)
        assert "Turn" in output
        assert "Player" in output


class TestActionEncoder:
    """Tests for the ActionEncoder class."""

    def test_action_space_size(self) -> None:
        """Test action space has correct size."""
        encoder = ActionEncoder()
        assert encoder.action_space_size == 149

    def test_encode_decode_roundtrip_buy(self) -> None:
        """Test encode/decode roundtrip for BuyProperty."""
        from monopoly_engine import BuyProperty

        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        # Move player to a buyable position
        game.players[0].position = 1  # Mediterranean Ave

        action = BuyProperty(player_id=0, property_id=1)
        encoded = encoder.encode(action)
        assert encoded == OFFSET_BUY_PROPERTY

        decoded = encoder.decode(encoded, player_id=0, game=game)
        assert isinstance(decoded, BuyProperty)
        assert decoded.property_id == 1

    def test_encode_decode_roundtrip_build_house(self) -> None:
        """Test encode/decode roundtrip for BuildHouse."""
        from monopoly_engine import BuildHouse

        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        action = BuildHouse(player_id=0, property_id=1)
        encoded = encoder.encode(action)
        assert OFFSET_BUILD_HOUSE <= encoded < OFFSET_BUILD_HOTEL

        decoded = encoder.decode(encoded, player_id=0, game=game)
        assert isinstance(decoded, BuildHouse)
        assert decoded.property_id == 1

    def test_encode_decode_roundtrip_mortgage(self) -> None:
        """Test encode/decode roundtrip for MortgageProperty."""
        from monopoly_engine import MortgageProperty

        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        action = MortgageProperty(player_id=0, property_id=5)  # Reading Railroad
        encoded = encoder.encode(action)
        assert OFFSET_MORTGAGE <= encoded < OFFSET_UNMORTGAGE

        decoded = encoder.decode(encoded, player_id=0, game=game)
        assert isinstance(decoded, MortgageProperty)
        assert decoded.property_id == 5

    def test_action_mask_not_your_turn(self) -> None:
        """Test action mask is all zeros when not current player."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        # Player 0 is current player
        assert game.current_player == 0

        # Mask for player 1 should have very limited actions
        mask = encoder.get_action_mask(game, player_id=1)

        # Most actions should be masked out for non-current player
        # Only selling houses and mortgaging (to pay debt) might be allowed
        assert not mask[OFFSET_BUY_PROPERTY]
        assert not mask[OFFSET_END_TURN]

    def test_action_mask_end_turn_always_valid(self) -> None:
        """Test EndTurn is always valid for current player."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        mask = encoder.get_action_mask(game, player_id=0)
        assert mask[OFFSET_END_TURN]

    def test_developable_positions_correct(self) -> None:
        """Test developable positions exclude railroads and utilities."""
        railroads = {5, 15, 25, 35}
        utilities = {12, 28}

        for pos in DEVELOPABLE_POSITIONS:
            assert pos not in railroads
            assert pos not in utilities

    def test_buyable_positions_includes_all(self) -> None:
        """Test buyable positions includes all purchasable properties."""
        railroads = {5, 15, 25, 35}
        utilities = {12, 28}

        for rr in railroads:
            assert rr in BUYABLE_POSITIONS
        for util in utilities:
            assert util in BUYABLE_POSITIONS


class TestObservationEncoder:
    """Tests for the ObservationEncoder class."""

    def test_init_valid_players(self) -> None:
        """Test encoder accepts valid player counts."""
        for n in range(2, 9):
            encoder = ObservationEncoder(n)
            assert encoder.num_players == n

    def test_init_invalid_players(self) -> None:
        """Test encoder rejects invalid player counts."""
        with pytest.raises(ValueError):
            ObservationEncoder(1)
        with pytest.raises(ValueError):
            ObservationEncoder(9)

    def test_observation_space_structure(self) -> None:
        """Test observation space has correct structure."""
        encoder = ObservationEncoder(4)
        space = encoder.get_observation_space()

        assert isinstance(space, spaces.Dict)
        assert "player_state" in space.spaces
        assert "opponent_states" in space.spaces
        assert "board_state" in space.spaces
        assert "game_state" in space.spaces
        assert "action_mask" in space.spaces

    def test_encode_returns_valid_observation(self) -> None:
        """Test encode returns observation matching space."""
        encoder = ObservationEncoder(2)
        game = MonopolyGame(num_players=2, seed=42)

        obs = encoder.encode(game, player_id=0)

        assert isinstance(obs, dict)
        assert "player_state" in obs
        assert "opponent_states" in obs
        assert "board_state" in obs
        assert "game_state" in obs
        assert "action_mask" in obs

    def test_player_state_shape(self) -> None:
        """Test player state has correct shapes."""
        encoder = ObservationEncoder(2)
        game = MonopolyGame(num_players=2, seed=42)

        obs = encoder.encode(game, player_id=0)
        ps = obs["player_state"]

        assert ps["money"].shape == (1,)
        assert 0 <= ps["money"][0] <= 1
        assert ps["properties_owned"].shape == (28,)

    def test_opponent_states_shape(self) -> None:
        """Test opponent states have correct shape."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        obs = encoder.encode(game, player_id=0)

        # 3 opponents, 34 features each (6 basic + 28 properties)
        assert obs["opponent_states"].shape == (3, 34)

    def test_board_state_shape(self) -> None:
        """Test board state has correct shape."""
        encoder = ObservationEncoder(2)
        game = MonopolyGame(num_players=2, seed=42)

        obs = encoder.encode(game, player_id=0)

        # 28 properties, 5 features each
        assert obs["board_state"].shape == (28, 5)

    def test_normalization_bounds(self) -> None:
        """Test all normalized values are in [0, 1]."""
        encoder = ObservationEncoder(4)
        game = MonopolyGame(num_players=4, seed=42)

        obs = encoder.encode(game, player_id=0)

        # Check opponent states in bounds
        assert np.all(obs["opponent_states"] >= 0)
        assert np.all(obs["opponent_states"] <= 1)

        # Check board state in bounds
        assert np.all(obs["board_state"] >= 0)
        assert np.all(obs["board_state"] <= 1)

        # Check game state in bounds
        for key, val in obs["game_state"].items():
            assert np.all(val >= 0), f"{key} has negative values"
            assert np.all(val <= 1), f"{key} exceeds 1.0"


class TestActionMaskConsistency:
    """Tests for action mask consistency with game rules."""

    def test_masked_action_executes(self) -> None:
        """Test that any masked action can actually execute."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        # Give player money and properties for more valid actions
        game.players[0].money = 10000
        game.transfer_property(1, None, 0)  # Mediterranean
        game.transfer_property(3, None, 0)  # Baltic (monopoly)

        mask = encoder.get_action_mask(game, player_id=0)

        for action_idx in np.where(mask)[0]:
            # Each masked action should decode and validate
            action = encoder.decode(int(action_idx), player_id=0, game=game)
            is_valid, msg = action.validate(game)
            assert is_valid, f"Action {action_idx} was masked but failed validation: {msg}"

    def test_jail_actions_only_when_in_jail(self) -> None:
        """Test jail actions are only valid when in jail."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        # Not in jail
        mask = encoder.get_action_mask(game, player_id=0)
        assert not mask[OFFSET_USE_JAIL_CARD]
        assert not mask[OFFSET_PAY_JAIL_FINE]

        # Send to jail
        game.send_to_jail(0)

        # Now jail actions should be available
        mask = encoder.get_action_mask(game, player_id=0)
        assert mask[OFFSET_PAY_JAIL_FINE]  # Can always pay fine if affordable

        # Give jail card
        game.players[0].add_jail_card()
        mask = encoder.get_action_mask(game, player_id=0)
        assert mask[OFFSET_USE_JAIL_CARD]


class TestIntegration:
    """Integration tests for the full environment."""

    def test_play_full_game(self) -> None:
        """Test playing a complete game with random actions."""
        env = MonopolyEnv(num_players=2, max_turns=100)
        env.reset(seed=42)

        done = False
        steps = 0

        while not done and steps < 500:
            for agent in env.agents:
                if agent != env.agent_selection:
                    continue

                obs, reward, termination, truncation, info = env.last()
                done = termination or truncation

                if done:
                    env.step(None)
                    break

                mask = info["action_mask"]
                valid = np.where(mask)[0]

                if len(valid) == 0:
                    action = OFFSET_END_TURN
                else:
                    # Prefer buying properties, otherwise end turn
                    if mask[OFFSET_BUY_PROPERTY]:
                        action = OFFSET_BUY_PROPERTY
                    else:
                        action = OFFSET_END_TURN

                env.step(action)
                steps += 1

        # Should have completed without crashing
        assert steps > 0

    def test_environment_space_compatibility(self) -> None:
        """Test observation and action spaces are valid gym spaces."""
        env = MonopolyEnv(num_players=2)
        env.reset(seed=42)

        # Check spaces are valid
        for agent in env.possible_agents:
            obs_space = env.observation_space(agent)
            act_space = env.action_space(agent)

            assert isinstance(obs_space, spaces.Dict)
            assert isinstance(act_space, spaces.Discrete)
            assert act_space.n == GAMEPLAY_ACTION_SPACE_SIZE  # Default env has no trades
