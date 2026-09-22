"""Tests for the rewards module.

This module contains comprehensive tests for:
- RewardConfig dataclass
- calculate_sparse_reward function
- calculate_dense_reward function
- count_monopolies function
- RewardTracker class
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from monopoly_engine import MonopolyGame, calculate_net_worth
from monopoly_engine.types import PROPERTY_GROUPS, PropertyColor
from monopoly_gym.rewards import (
    RewardConfig,
    RewardTracker,
    calculate_dense_reward,
    calculate_sparse_reward,
    count_monopolies,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def game() -> MonopolyGame:
    """Create a basic 2-player game."""
    return MonopolyGame(num_players=2, seed=42)


@pytest.fixture
def game_4_players() -> MonopolyGame:
    """Create a 4-player game."""
    return MonopolyGame(num_players=4, seed=42)


@pytest.fixture
def default_config() -> RewardConfig:
    """Create default reward configuration."""
    return RewardConfig()


@pytest.fixture
def custom_config() -> RewardConfig:
    """Create custom reward configuration."""
    return RewardConfig(
        win_reward=10.0,
        loss_reward=-5.0,
        net_worth_scale=0.01,
        monopoly_bonus=1.0,
        property_bonus=0.2,
        building_bonus=0.1,
        bankruptcy_penalty=-10.0,
        invalid_action_penalty=-0.1,
    )


@pytest.fixture
def game_with_brown_monopoly(game: MonopolyGame) -> MonopolyGame:
    """Create a game where player 0 has the brown monopoly."""
    # Give player 0 both brown properties (positions 1 and 3)
    game.transfer_property(1, None, 0)  # Mediterranean Ave
    game.transfer_property(3, None, 0)  # Baltic Ave
    return game


@pytest.fixture
def game_with_light_blue_monopoly(game: MonopolyGame) -> MonopolyGame:
    """Create a game where player 0 has the light blue monopoly."""
    # Give player 0 all light blue properties (positions 6, 8, 9)
    for pos in PROPERTY_GROUPS[PropertyColor.LIGHT_BLUE]:
        game.transfer_property(pos, None, 0)
    return game


@pytest.fixture
def game_with_multiple_monopolies(game: MonopolyGame) -> MonopolyGame:
    """Create a game where player 0 has multiple monopolies."""
    # Brown monopoly (2 properties)
    game.transfer_property(1, None, 0)
    game.transfer_property(3, None, 0)
    # Light blue monopoly (3 properties)
    for pos in PROPERTY_GROUPS[PropertyColor.LIGHT_BLUE]:
        game.transfer_property(pos, None, 0)
    return game


@pytest.fixture
def game_with_railroads(game: MonopolyGame) -> MonopolyGame:
    """Create a game where player 0 owns all railroads."""
    for pos in PROPERTY_GROUPS[PropertyColor.RAILROAD]:
        game.transfer_property(pos, None, 0)
    return game


@pytest.fixture
def game_with_utilities(game: MonopolyGame) -> MonopolyGame:
    """Create a game where player 0 owns both utilities."""
    for pos in PROPERTY_GROUPS[PropertyColor.UTILITY]:
        game.transfer_property(pos, None, 0)
    return game


@pytest.fixture
def game_over_with_winner(game: MonopolyGame) -> MonopolyGame:
    """Create a game that is over with player 0 as winner."""
    # Mark player 1 as bankrupt
    game.players[1].bankrupt = True
    game.players[1].money = 0
    game.state.game_over = True
    game.state.winner = 0
    return game


@pytest.fixture
def game_over_player_lost(game: MonopolyGame) -> MonopolyGame:
    """Create a game that is over with player 0 as loser."""
    # Mark player 0 as bankrupt
    game.players[0].bankrupt = True
    game.players[0].money = 0
    game.state.game_over = True
    game.state.winner = 1
    return game


# =============================================================================
# TestRewardConfig
# =============================================================================


class TestRewardConfig:
    """Tests for RewardConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that default values are correct."""
        config = RewardConfig()
        assert config.win_reward == 1.0
        assert config.loss_reward == -1.0
        assert config.net_worth_scale == 0.001
        assert config.monopoly_bonus == 0.5
        assert config.property_bonus == 0.1
        assert config.building_bonus == 0.05
        assert config.bankruptcy_penalty == -2.0
        assert config.invalid_action_penalty == -0.01

    def test_custom_values(self) -> None:
        """Test that custom values can be set."""
        config = RewardConfig(
            win_reward=100.0,
            loss_reward=-50.0,
            net_worth_scale=0.1,
            monopoly_bonus=5.0,
            property_bonus=1.0,
            building_bonus=0.5,
            bankruptcy_penalty=-20.0,
            invalid_action_penalty=-0.5,
        )
        assert config.win_reward == 100.0
        assert config.loss_reward == -50.0
        assert config.net_worth_scale == 0.1
        assert config.monopoly_bonus == 5.0
        assert config.property_bonus == 1.0
        assert config.building_bonus == 0.5
        assert config.bankruptcy_penalty == -20.0
        assert config.invalid_action_penalty == -0.5

    def test_all_fields_accessible(self) -> None:
        """Test that all fields are accessible."""
        config = RewardConfig()
        # Access all fields to ensure they exist
        fields = [
            config.win_reward,
            config.loss_reward,
            config.net_worth_scale,
            config.monopoly_bonus,
            config.property_bonus,
            config.building_bonus,
            config.bankruptcy_penalty,
            config.invalid_action_penalty,
        ]
        assert all(isinstance(f, float) for f in fields)

    def test_fields_are_floats(self) -> None:
        """Test that all config fields are floats."""
        config = RewardConfig()
        assert isinstance(config.win_reward, float)
        assert isinstance(config.loss_reward, float)
        assert isinstance(config.net_worth_scale, float)
        assert isinstance(config.monopoly_bonus, float)
        assert isinstance(config.property_bonus, float)
        assert isinstance(config.building_bonus, float)
        assert isinstance(config.bankruptcy_penalty, float)
        assert isinstance(config.invalid_action_penalty, float)

    def test_partial_custom_values(self) -> None:
        """Test setting only some custom values."""
        config = RewardConfig(win_reward=5.0, loss_reward=-3.0)
        assert config.win_reward == 5.0
        assert config.loss_reward == -3.0
        # Other values should be default
        assert config.net_worth_scale == 0.001
        assert config.monopoly_bonus == 0.5


# =============================================================================
# TestCalculateSparseReward
# =============================================================================


class TestCalculateSparseReward:
    """Tests for sparse reward calculation."""

    def test_returns_zero_when_game_not_over(self, game: MonopolyGame) -> None:
        """Test returns 0 when game is not over."""
        assert not game.game_over
        reward = calculate_sparse_reward(game, player_id=0)
        assert reward == 0.0

    def test_returns_zero_for_all_players_during_game(self, game_4_players: MonopolyGame) -> None:
        """Test returns 0 for all players when game is active."""
        for player_id in range(4):
            reward = calculate_sparse_reward(game_4_players, player_id=player_id)
            assert reward == 0.0

    def test_returns_win_reward_when_player_is_winner(
        self, game_over_with_winner: MonopolyGame
    ) -> None:
        """Test returns win_reward when player is the winner."""
        reward = calculate_sparse_reward(game_over_with_winner, player_id=0)
        assert reward == 1.0  # Default win_reward

    def test_returns_loss_reward_when_player_is_not_winner(
        self, game_over_with_winner: MonopolyGame
    ) -> None:
        """Test returns loss_reward when player is not the winner."""
        reward = calculate_sparse_reward(game_over_with_winner, player_id=1)
        assert reward == -1.0  # Default loss_reward

    def test_with_custom_config_win(
        self, game_over_with_winner: MonopolyGame, custom_config: RewardConfig
    ) -> None:
        """Test with custom config for winner."""
        reward = calculate_sparse_reward(game_over_with_winner, player_id=0, config=custom_config)
        assert reward == custom_config.win_reward
        assert reward == 10.0

    def test_with_custom_config_loss(
        self, game_over_with_winner: MonopolyGame, custom_config: RewardConfig
    ) -> None:
        """Test with custom config for loser."""
        reward = calculate_sparse_reward(game_over_with_winner, player_id=1, config=custom_config)
        assert reward == custom_config.loss_reward
        assert reward == -5.0

    def test_with_none_config_uses_defaults(self, game_over_with_winner: MonopolyGame) -> None:
        """Test that None config uses default values."""
        reward = calculate_sparse_reward(game_over_with_winner, player_id=0, config=None)
        assert reward == 1.0  # Default win_reward

    def test_returns_loss_for_bankrupt_player(self, game_over_player_lost: MonopolyGame) -> None:
        """Test returns loss_reward for bankrupt player."""
        reward = calculate_sparse_reward(game_over_player_lost, player_id=0)
        assert reward == -1.0


# =============================================================================
# TestCalculateDenseReward
# =============================================================================


class TestCalculateDenseReward:
    """Tests for dense reward calculation."""

    def test_returns_tuple(self, game: MonopolyGame) -> None:
        """Test returns (reward, current_net_worth) tuple."""
        result = calculate_dense_reward(game, player_id=0, prev_net_worth=1500)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reward_is_float(self, game: MonopolyGame) -> None:
        """Test that reward is a float."""
        reward, _ = calculate_dense_reward(game, player_id=0, prev_net_worth=1500)
        assert isinstance(reward, float)

    def test_current_net_worth_is_int(self, game: MonopolyGame) -> None:
        """Test that current_net_worth is an int."""
        _, current_worth = calculate_dense_reward(game, player_id=0, prev_net_worth=1500)
        assert isinstance(current_worth, int)

    def test_reward_scales_with_positive_net_worth_change(self, game: MonopolyGame) -> None:
        """Test reward scales positively with net worth increase."""
        # Give player some property to increase net worth
        game.transfer_property(1, None, 0)  # Mediterranean Ave (cost 60)

        # Start from lower net worth
        reward, current = calculate_dense_reward(game, player_id=0, prev_net_worth=1500)

        # Current net worth includes property value
        assert current > 1500
        assert reward > 0

    def test_reward_scales_with_negative_net_worth_change(self, game: MonopolyGame) -> None:
        """Test reward scales negatively with net worth decrease."""
        # Reduce player money
        game.players[0].money = 1000

        # Start from higher net worth
        reward, current = calculate_dense_reward(game, player_id=0, prev_net_worth=1500)

        # Current net worth is lower
        assert current < 1500
        assert reward < 0

    def test_win_bonus_at_game_end(self, game_over_with_winner: MonopolyGame) -> None:
        """Test win bonus is added at game end."""
        config = RewardConfig()
        reward, _ = calculate_dense_reward(
            game_over_with_winner,
            player_id=0,
            prev_net_worth=1500,
            config=config,
        )
        # Winner gets win_reward * 10 bonus
        assert reward >= config.win_reward * 10

    def test_loss_penalty_at_game_end(self, game_over_with_winner: MonopolyGame) -> None:
        """Test loss penalty is added at game end."""
        config = RewardConfig()
        prev_worth = 0  # Simulate bankrupt player
        reward, _ = calculate_dense_reward(
            game_over_with_winner,
            player_id=1,
            prev_net_worth=prev_worth,
            config=config,
        )
        # Loser gets loss_reward penalty
        assert reward <= config.loss_reward

    def test_bankruptcy_penalty(self, game: MonopolyGame) -> None:
        """Test bankruptcy penalty is applied."""
        # Mark player as bankrupt
        game.players[0].bankrupt = True
        game.players[0].money = 0

        config = RewardConfig()
        reward, _ = calculate_dense_reward(game, player_id=0, prev_net_worth=0, config=config)

        # Should include bankruptcy penalty
        assert reward <= config.bankruptcy_penalty

    def test_with_custom_config(self, game: MonopolyGame) -> None:
        """Test dense reward with custom config."""
        custom = RewardConfig(net_worth_scale=0.01)  # 10x default

        # Give player property to increase net worth
        game.transfer_property(1, None, 0)

        default_reward, _ = calculate_dense_reward(game, player_id=0, prev_net_worth=1500)
        custom_reward, _ = calculate_dense_reward(
            game, player_id=0, prev_net_worth=1500, config=custom
        )

        # Custom reward should be scaled differently
        # (approximately 10x if only net worth change matters)
        if default_reward != 0:
            assert abs(custom_reward) > abs(default_reward)

    def test_with_none_config_uses_defaults(self, game: MonopolyGame) -> None:
        """Test that None config uses default values."""
        reward, _ = calculate_dense_reward(game, player_id=0, prev_net_worth=1500, config=None)
        # Should work without error
        assert isinstance(reward, float)

    def test_returns_accurate_current_net_worth(self, game: MonopolyGame) -> None:
        """Test that current_net_worth matches calculate_net_worth."""
        _, current = calculate_dense_reward(game, player_id=0, prev_net_worth=1500)

        expected = calculate_net_worth(game.players[0], game.property_manager)
        assert current == expected


# =============================================================================
# TestCountMonopolies
# =============================================================================


class TestCountMonopolies:
    """Tests for monopoly counting."""

    def test_returns_zero_with_no_monopolies(self, game: MonopolyGame) -> None:
        """Test returns 0 when player has no monopolies."""
        count = count_monopolies(game, player_id=0)
        assert count == 0

    def test_returns_zero_with_no_properties(self, game: MonopolyGame) -> None:
        """Test returns 0 when player owns no properties."""
        count = count_monopolies(game, player_id=0)
        assert count == 0

    def test_counts_single_monopoly(self, game_with_brown_monopoly: MonopolyGame) -> None:
        """Test counts single monopoly correctly."""
        count = count_monopolies(game_with_brown_monopoly, player_id=0)
        assert count == 1

    def test_counts_multiple_monopolies(self, game_with_multiple_monopolies: MonopolyGame) -> None:
        """Test counts multiple monopolies correctly."""
        count = count_monopolies(game_with_multiple_monopolies, player_id=0)
        assert count == 2  # Brown and Light Blue

    def test_excludes_railroad_monopoly(self, game_with_railroads: MonopolyGame) -> None:
        """Test does not count railroad 'monopoly'."""
        count = count_monopolies(game_with_railroads, player_id=0)
        assert count == 0

    def test_excludes_utility_monopoly(self, game_with_utilities: MonopolyGame) -> None:
        """Test does not count utility 'monopoly'."""
        count = count_monopolies(game_with_utilities, player_id=0)
        assert count == 0

    def test_partial_color_group_not_counted(self, game: MonopolyGame) -> None:
        """Test partial ownership of a color group is not counted."""
        # Own only one brown property
        game.transfer_property(1, None, 0)  # Mediterranean Ave only
        count = count_monopolies(game, player_id=0)
        assert count == 0

    def test_different_players_monopolies(self, game: MonopolyGame) -> None:
        """Test each player's monopolies counted separately."""
        # Player 0 gets brown
        game.transfer_property(1, None, 0)
        game.transfer_property(3, None, 0)

        # Player 1 gets light blue
        for pos in PROPERTY_GROUPS[PropertyColor.LIGHT_BLUE]:
            game.transfer_property(pos, None, 1)

        assert count_monopolies(game, player_id=0) == 1
        assert count_monopolies(game, player_id=1) == 1

    def test_all_eight_color_monopolies(self, game: MonopolyGame) -> None:
        """Test can count all 8 color monopolies."""
        # Give player 0 all color properties
        color_groups = [
            PropertyColor.BROWN,
            PropertyColor.LIGHT_BLUE,
            PropertyColor.MAGENTA,
            PropertyColor.ORANGE,
            PropertyColor.RED,
            PropertyColor.YELLOW,
            PropertyColor.GREEN,
            PropertyColor.DARK_BLUE,
        ]

        for color in color_groups:
            for pos in PROPERTY_GROUPS[color]:
                game.transfer_property(pos, None, 0)

        count = count_monopolies(game, player_id=0)
        assert count == 8

    def test_mixed_ownership_not_counted(self, game: MonopolyGame) -> None:
        """Test mixed ownership of color group not counted."""
        # Player 0 owns Mediterranean, Player 1 owns Baltic
        game.transfer_property(1, None, 0)
        game.transfer_property(3, None, 1)

        assert count_monopolies(game, player_id=0) == 0
        assert count_monopolies(game, player_id=1) == 0


# =============================================================================
# TestRewardTracker
# =============================================================================


class TestRewardTracker:
    """Tests for RewardTracker class."""

    def test_initialization_with_num_players(self) -> None:
        """Test initialization with num_players."""
        tracker = RewardTracker(num_players=4)
        assert tracker.num_players == 4

    def test_initialization_with_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = RewardConfig(win_reward=5.0)
        tracker = RewardTracker(num_players=2, config=config)
        assert tracker.config.win_reward == 5.0

    def test_initialization_creates_default_config(self) -> None:
        """Test initialization creates default config if not provided."""
        tracker = RewardTracker(num_players=2)
        assert isinstance(tracker.config, RewardConfig)
        assert tracker.config.win_reward == 1.0

    def test_reset_initializes_tracking_state(self, game: MonopolyGame) -> None:
        """Test reset initializes tracking state."""
        tracker = RewardTracker(num_players=2)
        tracker.reset(game)

        # Should have tracked net worth for both players
        assert 0 in tracker._prev_net_worth
        assert 1 in tracker._prev_net_worth

        # Should have tracked monopolies for both players
        assert 0 in tracker._prev_monopolies
        assert 1 in tracker._prev_monopolies

    def test_reset_clears_previous_state(self, game: MonopolyGame) -> None:
        """Test reset clears any previous state."""
        tracker = RewardTracker(num_players=2)

        # First reset
        tracker.reset(game)
        tracker._prev_net_worth[0] = 9999  # Modify state

        # Second reset should clear
        tracker.reset(game)
        assert tracker._prev_net_worth[0] != 9999

    def test_calculate_reward_sparse_mode(self, game: MonopolyGame) -> None:
        """Test calculate_reward in sparse mode."""
        tracker = RewardTracker(num_players=2)
        tracker.reset(game)

        reward = tracker.calculate_reward(game, player_id=0, reward_type="sparse")
        assert reward == 0.0  # Game not over

    def test_calculate_reward_sparse_mode_win(self, game_over_with_winner: MonopolyGame) -> None:
        """Test calculate_reward in sparse mode when winning."""
        tracker = RewardTracker(num_players=2)
        tracker.reset(game_over_with_winner)

        reward = tracker.calculate_reward(game_over_with_winner, player_id=0, reward_type="sparse")
        assert reward == 1.0

    def test_calculate_reward_dense_mode(self, game: MonopolyGame) -> None:
        """Test calculate_reward in dense mode."""
        tracker = RewardTracker(num_players=2)
        tracker.reset(game)

        reward = tracker.calculate_reward(game, player_id=0, reward_type="dense")
        assert isinstance(reward, float)

    def test_calculate_reward_dense_mode_tracks_net_worth(self, game: MonopolyGame) -> None:
        """Test dense mode updates tracked net worth."""
        tracker = RewardTracker(num_players=2)
        tracker.reset(game)

        initial_worth = tracker._prev_net_worth[0]

        # Give player some property
        game.transfer_property(1, None, 0)

        tracker.calculate_reward(game, player_id=0, reward_type="dense")

        # Net worth should have been updated
        assert tracker._prev_net_worth[0] != initial_worth

    def test_monopoly_bonus_when_completing_monopoly(self, game: MonopolyGame) -> None:
        """Test monopoly bonus is given when completing a monopoly."""
        tracker = RewardTracker(num_players=2)
        tracker.reset(game)

        # Give player first brown property
        game.transfer_property(1, None, 0)
        tracker.calculate_reward(game, player_id=0, reward_type="dense")

        # Record monopoly count (should be 0)
        assert tracker._prev_monopolies[0] == 0

        # Complete the monopoly
        game.transfer_property(3, None, 0)
        reward = tracker.calculate_reward(game, player_id=0, reward_type="dense")

        # Should have gotten monopoly bonus
        assert tracker._prev_monopolies[0] == 1
        # Reward should include bonus
        assert reward > 0  # Net worth didn't change, but got monopoly bonus

    def test_state_updates_between_calls(self, game: MonopolyGame) -> None:
        """Test state updates correctly between calls."""
        tracker = RewardTracker(num_players=2)
        tracker.reset(game)

        # First call
        tracker.calculate_reward(game, player_id=0, reward_type="dense")
        first_worth = tracker._prev_net_worth[0]

        # Modify game state
        game.players[0].money += 500

        # Second call
        tracker.calculate_reward(game, player_id=0, reward_type="dense")
        second_worth = tracker._prev_net_worth[0]

        assert second_worth != first_worth

    def test_with_custom_config_affects_rewards(self, game_over_with_winner: MonopolyGame) -> None:
        """Test custom config affects calculated rewards."""
        config = RewardConfig(win_reward=100.0)
        tracker = RewardTracker(num_players=2, config=config)
        tracker.reset(game_over_with_winner)

        reward = tracker.calculate_reward(game_over_with_winner, player_id=0, reward_type="sparse")
        assert reward == 100.0

    def test_invalid_reward_type_raises_valueerror(self, game: MonopolyGame) -> None:
        """Test invalid reward_type raises ValueError."""
        tracker = RewardTracker(num_players=2)
        tracker.reset(game)

        with pytest.raises(ValueError, match="Unknown reward_type"):
            tracker.calculate_reward(game, player_id=0, reward_type="invalid")

    def test_invalid_reward_type_error_message(self, game: MonopolyGame) -> None:
        """Test error message includes the invalid type."""
        tracker = RewardTracker(num_players=2)
        tracker.reset(game)

        with pytest.raises(ValueError) as exc_info:
            tracker.calculate_reward(game, player_id=0, reward_type="foobar")
        assert "foobar" in str(exc_info.value)

    def test_dense_reward_uses_default_prev_net_worth(self, game: MonopolyGame) -> None:
        """Test dense reward uses default if player not in tracking dict."""
        tracker = RewardTracker(num_players=2)
        # Don't call reset, so _prev_net_worth is empty

        # Should still work, using default of 1500
        reward = tracker.calculate_reward(game, player_id=0, reward_type="dense")
        assert isinstance(reward, float)

    def test_tracks_multiple_players(self, game_4_players: MonopolyGame) -> None:
        """Test tracking works for multiple players."""
        tracker = RewardTracker(num_players=4)
        tracker.reset(game_4_players)

        # All players should be tracked
        for player_id in range(4):
            reward = tracker.calculate_reward(
                game_4_players, player_id=player_id, reward_type="dense"
            )
            assert isinstance(reward, float)


# =============================================================================
# TestRewardIntegration
# =============================================================================


class TestRewardIntegration:
    """Integration tests with game environment."""

    def test_rewards_work_with_monopoly_game(self, game: MonopolyGame) -> None:
        """Test rewards work with MonopolyGame."""
        tracker = RewardTracker(num_players=2)
        tracker.reset(game)

        # Take some actions
        game.transfer_property(1, None, 0)
        game.transfer_property(3, None, 0)

        # Calculate rewards should work
        sparse = tracker.calculate_reward(game, player_id=0, reward_type="sparse")
        dense = tracker.calculate_reward(game, player_id=0, reward_type="dense")

        assert isinstance(sparse, float)
        assert isinstance(dense, float)

    def test_reward_tracking_across_multiple_steps(self, game: MonopolyGame) -> None:
        """Test reward tracking across multiple steps."""
        tracker = RewardTracker(num_players=2)
        tracker.reset(game)

        rewards = []
        for i in range(5):
            # Simulate some changes
            game.players[0].money += 100 * i

            reward = tracker.calculate_reward(game, player_id=0, reward_type="dense")
            rewards.append(reward)

        # First reward should be 0 (no change from reset)
        # Subsequent rewards should be positive (increasing money)
        assert all(isinstance(r, float) for r in rewards)

    def test_rewards_are_reasonable_values(self, game: MonopolyGame) -> None:
        """Test rewards are reasonable values (not NaN, inf, etc.)."""
        import math

        tracker = RewardTracker(num_players=2)
        tracker.reset(game)

        # Test various game states
        # State 1: No change
        reward = tracker.calculate_reward(game, player_id=0, reward_type="dense")
        assert not math.isnan(reward)
        assert not math.isinf(reward)

        # State 2: Zero money
        game.players[0].money = 0
        reward = tracker.calculate_reward(game, player_id=0, reward_type="dense")
        assert not math.isnan(reward)
        assert not math.isinf(reward)

        # State 3: Lots of money
        game.players[0].money = 10000
        reward = tracker.calculate_reward(game, player_id=0, reward_type="dense")
        assert not math.isnan(reward)
        assert not math.isinf(reward)

        # State 4: Property ownership
        game.transfer_property(1, None, 0)
        reward = tracker.calculate_reward(game, player_id=0, reward_type="dense")
        assert not math.isnan(reward)
        assert not math.isinf(reward)

    def test_sparse_reward_zero_during_game(self, game: MonopolyGame) -> None:
        """Test sparse reward is always 0 during active game."""
        tracker = RewardTracker(num_players=2)
        tracker.reset(game)

        # Make various changes
        game.players[0].money = 5000
        game.transfer_property(1, None, 0)

        for _ in range(10):
            reward = tracker.calculate_reward(game, player_id=0, reward_type="sparse")
            assert reward == 0.0

    def test_sparse_reward_nonzero_at_game_end(self, game_over_with_winner: MonopolyGame) -> None:
        """Test sparse reward is nonzero at game end."""
        tracker = RewardTracker(num_players=2)
        tracker.reset(game_over_with_winner)

        winner_reward = tracker.calculate_reward(
            game_over_with_winner, player_id=0, reward_type="sparse"
        )
        loser_reward = tracker.calculate_reward(
            game_over_with_winner, player_id=1, reward_type="sparse"
        )

        assert winner_reward != 0.0
        assert loser_reward != 0.0
        assert winner_reward > loser_reward

    def test_dense_reward_reflects_property_acquisition(self, game: MonopolyGame) -> None:
        """Test dense reward reflects property acquisition."""
        tracker = RewardTracker(num_players=2)
        tracker.reset(game)

        # Baseline reward
        baseline = tracker.calculate_reward(game, player_id=0, reward_type="dense")

        # Acquire property (increases net worth)
        game.transfer_property(37, None, 0)  # Park Place (expensive)

        # New reward should be positive (net worth increased)
        reward = tracker.calculate_reward(game, player_id=0, reward_type="dense")
        assert reward > baseline

    def test_dense_reward_reflects_money_loss(self, game: MonopolyGame) -> None:
        """Test dense reward reflects money loss."""
        tracker = RewardTracker(num_players=2)
        tracker.reset(game)

        # Baseline
        tracker.calculate_reward(game, player_id=0, reward_type="dense")

        # Lose money
        game.players[0].money -= 500

        # Should get negative reward
        reward = tracker.calculate_reward(game, player_id=0, reward_type="dense")
        assert reward < 0


# =============================================================================
# Property-based tests using hypothesis
# =============================================================================


class TestRewardConfigHypothesis:
    """Property-based tests for RewardConfig."""

    @given(
        win_reward=st.floats(min_value=-100, max_value=100, allow_nan=False),
        loss_reward=st.floats(min_value=-100, max_value=100, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_config_accepts_any_float_values(self, win_reward: float, loss_reward: float) -> None:
        """Test config accepts any float values."""
        config = RewardConfig(win_reward=win_reward, loss_reward=loss_reward)
        assert config.win_reward == win_reward
        assert config.loss_reward == loss_reward


class TestSparseRewardHypothesis:
    """Property-based tests for sparse reward."""

    @given(player_id=st.integers(min_value=0, max_value=1))
    @settings(max_examples=20)
    def test_sparse_reward_always_zero_during_game(self, player_id: int) -> None:
        """Test sparse reward is always 0 during active game."""
        game = MonopolyGame(num_players=2, seed=42)
        reward = calculate_sparse_reward(game, player_id=player_id)
        assert reward == 0.0

    @given(
        win_reward=st.floats(min_value=0.1, max_value=100, allow_nan=False),
        loss_reward=st.floats(min_value=-100, max_value=-0.1, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_winner_gets_positive_loser_gets_negative(
        self, win_reward: float, loss_reward: float
    ) -> None:
        """Test winner gets positive, loser gets negative reward."""
        game = MonopolyGame(num_players=2, seed=42)
        game.players[1].bankrupt = True
        game.state.game_over = True
        game.state.winner = 0

        config = RewardConfig(win_reward=win_reward, loss_reward=loss_reward)

        winner = calculate_sparse_reward(game, 0, config)
        loser = calculate_sparse_reward(game, 1, config)

        assert winner > 0
        assert loser < 0
        assert winner > loser


class TestDenseRewardHypothesis:
    """Property-based tests for dense reward."""

    @given(prev_net_worth=st.integers(min_value=0, max_value=10000))
    @settings(max_examples=30)
    def test_dense_reward_handles_various_prev_worth(self, prev_net_worth: int) -> None:
        """Test dense reward handles various previous net worth values."""
        game = MonopolyGame(num_players=2, seed=42)
        reward, current = calculate_dense_reward(game, 0, prev_net_worth)

        assert isinstance(reward, float)
        assert isinstance(current, int)


class TestCountMonopoliesHypothesis:
    """Property-based tests for count_monopolies."""

    @given(player_id=st.integers(min_value=0, max_value=1))
    @settings(max_examples=20)
    def test_count_monopolies_non_negative(self, player_id: int) -> None:
        """Test count_monopolies always returns non-negative."""
        game = MonopolyGame(num_players=2, seed=42)
        count = count_monopolies(game, player_id)
        assert count >= 0

    @given(player_id=st.integers(min_value=0, max_value=1))
    @settings(max_examples=20)
    def test_count_monopolies_max_eight(self, player_id: int) -> None:
        """Test count_monopolies returns at most 8."""
        game = MonopolyGame(num_players=2, seed=42)
        count = count_monopolies(game, player_id)
        assert count <= 8


class TestRewardTrackerHypothesis:
    """Property-based tests for RewardTracker."""

    @given(num_players=st.integers(min_value=2, max_value=4))
    @settings(max_examples=20)
    def test_tracker_handles_various_player_counts(self, num_players: int) -> None:
        """Test tracker works with various player counts."""
        game = MonopolyGame(num_players=num_players, seed=42)
        tracker = RewardTracker(num_players=num_players)
        tracker.reset(game)

        for player_id in range(num_players):
            reward = tracker.calculate_reward(game, player_id, "dense")
            assert isinstance(reward, float)

    @given(reward_type=st.sampled_from(["sparse", "dense"]))
    @settings(max_examples=20)
    def test_tracker_valid_reward_types(self, reward_type: str) -> None:
        """Test tracker works with valid reward types."""
        game = MonopolyGame(num_players=2, seed=42)
        tracker = RewardTracker(num_players=2)
        tracker.reset(game)

        reward = tracker.calculate_reward(game, 0, reward_type)
        assert isinstance(reward, float)
