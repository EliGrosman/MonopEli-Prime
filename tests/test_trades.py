"""Tests for Phase 2.5a trade functionality.

This module tests:
- Trade encoding/decoding
- Trade action masking
- Trade execution in the environment
- Trade reward calculation
"""

import pytest

from monopoly_engine import MonopolyGame
from monopoly_gym import (
    OFFSET_SIMPLE_TRADE,
    SIMPLE_TRADE_DIM,
    ActionEncoder,
    MonopolyEnv,
    TradeRewardConfig,
    calculate_trade_rewards,
    decode_simple_trade,
    encode_simple_trade,
    get_simple_trade_mask,
)
from monopoly_gym.trades import (
    BUYABLE_POSITIONS,
    _completed_monopoly_with_property,
    find_trade_for_player,
    get_trade_response_mask,
)


class TestTradeEncoding:
    """Tests for trade encoding/decoding."""

    def test_encode_decode_roundtrip_all_pairs(self) -> None:
        """Encoding then decoding should return the original pair."""
        for my_prop in BUYABLE_POSITIONS:
            for their_prop in BUYABLE_POSITIONS:
                if my_prop != their_prop:
                    action = encode_simple_trade(my_prop, their_prop)
                    decoded = decode_simple_trade(action)
                    assert decoded == (my_prop, their_prop), (
                        f"Roundtrip failed for ({my_prop}, {their_prop}): "
                        f"action={action}, decoded={decoded}"
                    )

    def test_encode_produces_valid_range(self) -> None:
        """All encoded actions should be in the valid trade action range."""
        for my_prop in BUYABLE_POSITIONS:
            for their_prop in BUYABLE_POSITIONS:
                if my_prop != their_prop:
                    action = encode_simple_trade(my_prop, their_prop)
                    assert OFFSET_SIMPLE_TRADE <= action < OFFSET_SIMPLE_TRADE + SIMPLE_TRADE_DIM

    def test_encode_unique_actions(self) -> None:
        """Each property pair should map to a unique action index."""
        seen_actions: set[int] = set()
        for my_prop in BUYABLE_POSITIONS:
            for their_prop in BUYABLE_POSITIONS:
                if my_prop != their_prop:
                    action = encode_simple_trade(my_prop, their_prop)
                    assert action not in seen_actions, f"Duplicate action {action}"
                    seen_actions.add(action)

        # Should have exactly 28 * 27 = 756 unique actions
        assert len(seen_actions) == SIMPLE_TRADE_DIM

    def test_encode_same_property_raises(self) -> None:
        """Cannot trade a property with itself."""
        with pytest.raises(ValueError, match="Cannot trade property with itself"):
            encode_simple_trade(1, 1)

    def test_encode_invalid_property_raises(self) -> None:
        """Cannot encode non-buyable positions."""
        with pytest.raises(ValueError, match="not a buyable property"):
            encode_simple_trade(0, 1)  # Position 0 is Go
        with pytest.raises(ValueError, match="not a buyable property"):
            encode_simple_trade(1, 10)  # Position 10 is Jail

    def test_decode_invalid_action_raises(self) -> None:
        """Cannot decode actions outside the trade range."""
        with pytest.raises(ValueError, match="not a simple trade action"):
            decode_simple_trade(0)  # Buy property action
        with pytest.raises(ValueError, match="not a simple trade action"):
            decode_simple_trade(OFFSET_SIMPLE_TRADE + SIMPLE_TRADE_DIM)  # Accept trade


class TestTradeMask:
    """Tests for trade action masking."""

    @pytest.fixture
    def game_with_properties(self) -> MonopolyGame:
        """Create a game where players own some properties."""
        game = MonopolyGame(num_players=4, seed=42)

        # Player 0 owns Mediterranean (1) and Oriental (6)
        game.property_manager.get(1).owner = 0  # type: ignore[union-attr]
        game.property_manager.get(6).owner = 0  # type: ignore[union-attr]

        # Player 1 owns Baltic (3) and Vermont (8)
        game.property_manager.get(3).owner = 1  # type: ignore[union-attr]
        game.property_manager.get(8).owner = 1  # type: ignore[union-attr]

        # Player 2 owns St. Charles (11)
        game.property_manager.get(11).owner = 2  # type: ignore[union-attr]

        return game

    def test_mask_enables_valid_trades(self, game_with_properties: MonopolyGame) -> None:
        """Mask should enable trades I can actually make."""
        game = game_with_properties
        mask = get_simple_trade_mask(game, player_id=0)

        # Player 0 owns 1, 6. Can trade with player 1 (owns 3, 8) or player 2 (owns 11)
        # Valid: (1,3), (1,8), (1,11), (6,3), (6,8), (6,11)
        for my_prop in [1, 6]:
            for their_prop in [3, 8, 11]:
                action = encode_simple_trade(my_prop, their_prop)
                mask_idx = action - OFFSET_SIMPLE_TRADE
                assert mask[mask_idx], f"Trade ({my_prop}, {their_prop}) should be valid"

    def test_mask_disables_trades_i_dont_own(self, game_with_properties: MonopolyGame) -> None:
        """Cannot propose to give away property I don't own."""
        game = game_with_properties
        mask = get_simple_trade_mask(game, player_id=0)

        # Player 0 doesn't own Baltic (3), so can't offer it
        action = encode_simple_trade(3, 1)  # Try to trade Baltic for Mediterranean
        mask_idx = action - OFFSET_SIMPLE_TRADE
        assert not mask[mask_idx], "Cannot trade property I don't own"

    def test_mask_disables_trades_for_unowned(self, game_with_properties: MonopolyGame) -> None:
        """Cannot request property that no opponent owns."""
        game = game_with_properties
        mask = get_simple_trade_mask(game, player_id=0)

        # Park Place (37) is unowned
        action = encode_simple_trade(1, 37)
        mask_idx = action - OFFSET_SIMPLE_TRADE
        assert not mask[mask_idx], "Cannot trade for unowned property"

    def test_mask_disables_trades_with_houses(self, game_with_properties: MonopolyGame) -> None:
        """Cannot trade properties that have houses."""
        game = game_with_properties

        # Give player 0 monopoly on Brown and build a house
        game.property_manager.get(3).owner = 0  # type: ignore[union-attr]
        game.property_manager.get(1).houses = 1  # type: ignore[union-attr]

        # Give player 1 something to trade
        game.property_manager.get(6).owner = 1  # type: ignore[union-attr]

        mask = get_simple_trade_mask(game, player_id=0)

        # Can't trade Mediterranean (1) - it has a house
        action = encode_simple_trade(1, 6)
        mask_idx = action - OFFSET_SIMPLE_TRADE
        assert not mask[mask_idx], "Cannot trade property with houses"

        # But can still trade Baltic (3) - no house
        action = encode_simple_trade(3, 6)
        mask_idx = action - OFFSET_SIMPLE_TRADE
        assert mask[mask_idx], "Can trade property without houses"

    def test_mask_disables_trades_with_mortgaged(self, game_with_properties: MonopolyGame) -> None:
        """Cannot trade mortgaged properties."""
        game = game_with_properties

        # Mortgage Mediterranean
        game.property_manager.get(1).mortgaged = True  # type: ignore[union-attr]

        mask = get_simple_trade_mask(game, player_id=0)

        # Can't trade Mediterranean (1) - it's mortgaged
        action = encode_simple_trade(1, 3)
        mask_idx = action - OFFSET_SIMPLE_TRADE
        assert not mask[mask_idx], "Cannot trade mortgaged property"

    def test_mask_disables_trades_with_bankrupt_opponent(
        self, game_with_properties: MonopolyGame
    ) -> None:
        """Cannot trade with bankrupt opponents."""
        game = game_with_properties

        # Player 1 goes bankrupt
        game.players[1].bankrupt = True

        mask = get_simple_trade_mask(game, player_id=0)

        # Can't trade for Baltic (3) - owner is bankrupt
        action = encode_simple_trade(1, 3)
        mask_idx = action - OFFSET_SIMPLE_TRADE
        assert not mask[mask_idx], "Cannot trade with bankrupt player"

    def test_no_valid_trades_returns_empty_mask(self) -> None:
        """Player with no properties has no valid trades."""
        game = MonopolyGame(num_players=2, seed=42)
        # No one owns anything
        mask = get_simple_trade_mask(game, player_id=0)
        assert not mask.any(), "No trades should be valid"


class TestTradeResponseMask:
    """Tests for accept/reject action masking."""

    def test_response_mask_with_pending_trade(self) -> None:
        """Response mask should enable accept/reject when trade is pending."""
        game = MonopolyGame(num_players=2, seed=42, rules_id="foundation-trade-v1")
        game.state.phase, game.state.roll_owed = "asset_management", False

        # Player 0 owns Mediterranean, player 1 owns Baltic
        game.property_manager.get(1).owner = 0  # type: ignore[union-attr]
        game.property_manager.get(3).owner = 1  # type: ignore[union-attr]

        # Create a pending trade from player 0 to player 1
        game.propose_trade(
            from_player=0,
            to_player=1,
            give_properties=[1],
            want_properties=[3],
            give_money=0,
            want_money=0,
        )

        mask = get_trade_response_mask(game, player_id=1)
        assert mask[0], "Should be able to accept"
        assert mask[1], "Should be able to reject"

    def test_response_mask_without_pending_trade(self) -> None:
        """Response mask should disable all when no trade is pending."""
        game = MonopolyGame(num_players=2, seed=42)
        mask = get_trade_response_mask(game, player_id=0)
        assert not mask[0], "Cannot accept without pending trade"
        assert not mask[1], "Cannot reject without pending trade"

    def test_response_mask_for_wrong_player(self) -> None:
        """Response mask should only enable for the trade recipient."""
        game = MonopolyGame(num_players=3, seed=42, rules_id="foundation-trade-v1")
        game.state.phase, game.state.roll_owed = "asset_management", False

        game.property_manager.get(1).owner = 0  # type: ignore[union-attr]
        game.property_manager.get(3).owner = 1  # type: ignore[union-attr]

        # Trade from 0 to 1
        game.propose_trade(
            from_player=0,
            to_player=1,
            give_properties=[1],
            want_properties=[3],
            give_money=0,
            want_money=0,
        )

        # Player 2 cannot respond
        mask = get_trade_response_mask(game, player_id=2)
        assert not mask[0], "Player 2 cannot accept trade meant for player 1"
        assert not mask[1], "Player 2 cannot reject trade meant for player 1"


class TestTradeRewards:
    """Tests for trade reward calculation."""

    def test_rejected_trade_penalty_for_proposer(self) -> None:
        """Proposer should get small penalty when trade is rejected."""
        game = MonopolyGame(num_players=2, seed=42)
        game.property_manager.get(1).owner = 0  # type: ignore[union-attr]
        game.property_manager.get(3).owner = 1  # type: ignore[union-attr]

        rewards = calculate_trade_rewards(
            game,
            proposer_id=0,
            responder_id=1,
            proposer_gave=1,
            proposer_got=3,
            accepted=False,
        )

        assert rewards[0] < 0, "Proposer should get penalty for rejection"
        assert rewards[1] == 0, "Responder should get no reward for rejection"

    def test_monopoly_completion_bonus(self) -> None:
        """Completing a monopoly via trade should give bonus."""
        game = MonopolyGame(num_players=2, seed=42)

        # Player 0 owns Mediterranean, player 1 owns Baltic
        # Trade completes Brown monopoly for player 0
        game.property_manager.get(1).owner = 0  # type: ignore[union-attr]
        game.property_manager.get(3).owner = 0  # type: ignore[union-attr]  # After trade

        config = TradeRewardConfig(monopoly_completion_bonus=2.0)
        rewards = calculate_trade_rewards(
            game,
            proposer_id=0,
            responder_id=1,
            proposer_gave=6,  # Gave Oriental
            proposer_got=3,  # Got Baltic (completes Brown)
            accepted=True,
            config=config,
        )

        assert rewards[0] >= 2.0, "Should get monopoly bonus"

    def test_gave_monopoly_penalty(self) -> None:
        """Giving opponent a monopoly should incur penalty."""
        game = MonopolyGame(num_players=2, seed=42)

        # After trade, player 1 has Brown monopoly
        game.property_manager.get(1).owner = 1  # type: ignore[union-attr]
        game.property_manager.get(3).owner = 1  # type: ignore[union-attr]

        config = TradeRewardConfig(gave_monopoly_penalty=-1.5)
        rewards = calculate_trade_rewards(
            game,
            proposer_id=0,
            responder_id=1,
            proposer_gave=1,  # Gave Mediterranean (completing Brown for opponent)
            proposer_got=6,  # Got Oriental
            accepted=True,
            config=config,
        )

        # Proposer gave away monopoly to responder
        assert rewards[0] <= -1.5, "Should get penalty for giving opponent monopoly"


def _v3_trade_setup() -> tuple[MonopolyGame, ActionEncoder]:
    game = MonopolyGame(num_players=2, seed=42, rules_id="foundation-trade-v1")
    game.state.phase = "asset_management"
    game.state.roll_owed = False
    for position in (1, 6):
        game.property_manager.properties[position].owner = 0
    for position in (3, 8, 9):
        game.property_manager.properties[position].owner = 1
    encoder = ActionEncoder(rules_id="foundation-trade-v1")
    assert encoder.get_trade_candidates(game, 0)
    return game, encoder


def _v3_env_setup() -> MonopolyEnv:
    env = MonopolyEnv(num_players=2, rules_id="foundation-trade-v1")
    env.reset(options={"engine_seed": 42})
    game, _ = _v3_trade_setup()
    env.game = game
    env.agent_selection = "player_0"
    env.action_encoder.invalidate_cache()
    env._update_infos()
    return env


class TestActionEncoderWithTrades:
    """Tests for the bounded action-v3 trade representation."""

    def test_action_space_size_with_trades(self) -> None:
        encoder = ActionEncoder(rules_id="foundation-trade-v1")
        assert encoder.action_space_size == 192

    def test_action_space_size_without_trades(self) -> None:
        """Without trades, action space should be 149."""
        encoder = ActionEncoder(enable_trades=False)
        assert encoder.action_space_size == 158

    def test_decode_trade_action(self) -> None:
        game, encoder = _v3_trade_setup()
        candidate = encoder.get_trade_candidates(game, 0)[0]
        assert encoder.decode(160, 0, game) == candidate

    def test_get_action_mask_includes_trades(self) -> None:
        game, encoder = _v3_trade_setup()
        mask = encoder.get_action_mask(game, player_id=0)
        assert mask[160:192].any()

    def test_get_action_mask_trade_response_mode(self) -> None:
        game, encoder = _v3_trade_setup()
        game.apply_action(0, encoder.get_trade_candidates(game, 0)[0])
        mask = encoder.get_action_mask(game, 1)
        assert mask[158] and mask[159]
        assert mask.sum() == 2


class TestMonopolyEnvWithTrades:
    """Tests for AEC action/observation-v3 trade support."""

    def test_env_creation_with_trades(self) -> None:
        env = MonopolyEnv(num_players=2, rules_id="foundation-trade-v1")
        assert env.enable_trades
        assert env.action_encoder.action_space_size == 192

    def test_env_action_space_with_trades(self) -> None:
        env = MonopolyEnv(num_players=2, rules_id="foundation-trade-v1")
        env.reset(seed=42)
        assert env._action_space.n == 192  # type: ignore[attr-defined]

    def test_env_trade_proposal_flow(self) -> None:
        env = _v3_env_setup()
        env.step(160)
        assert env.agent_selection == "player_1"
        assert env.game.state.phase == "trade_response"

    def test_env_trade_accept_flow(self) -> None:
        env = _v3_env_setup()
        offered = env.action_encoder.decode(160, 0, env.game)
        env.step(160)
        env.step(158)
        for position in offered.give_properties:
            assert env.game.property_manager.get(position).owner == 1  # type: ignore[union-attr]
        for position in offered.want_properties:
            assert env.game.property_manager.get(position).owner == 0  # type: ignore[union-attr]
        assert env.agent_selection == "player_0"

    def test_env_trade_reject_flow(self) -> None:
        env = _v3_env_setup()
        before = [(p.owner, p.mortgaged) for p in env.game.property_manager.properties.values()]
        env.step(160)
        env.step(159)
        after = [(p.owner, p.mortgaged) for p in env.game.property_manager.properties.values()]
        assert after == before
        assert env.agent_selection == "player_0"

    def test_env_trade_observation_includes_context(self) -> None:
        env = _v3_env_setup()
        obs = env.observe("player_0")
        assert "trade_context" in obs

    def test_env_trade_mask_during_response(self) -> None:
        env = _v3_env_setup()
        env.step(160)
        _, _, _, _, info = env.last()
        mask = info["action_mask"]
        assert mask[158] and mask[159]
        assert mask.sum() == 2


class TestFindTradeForPlayer:
    """Tests for find_trade_for_player utility."""

    def test_finds_pending_trade(self) -> None:
        """Should find trade directed at player."""
        game = MonopolyGame(num_players=2, seed=42, rules_id="foundation-trade-v1")
        game.state.phase, game.state.roll_owed = "asset_management", False
        game.property_manager.get(1).owner = 0  # type: ignore[union-attr]
        game.property_manager.get(3).owner = 1  # type: ignore[union-attr]

        trade_id = game.propose_trade(
            from_player=0,
            to_player=1,
            give_properties=[1],
            want_properties=[3],
            give_money=0,
            want_money=0,
        )

        found_id = find_trade_for_player(game, player_id=1)
        assert found_id == trade_id

    def test_returns_none_when_no_trade(self) -> None:
        """Should return None when no pending trade for player."""
        game = MonopolyGame(num_players=2, seed=42)
        assert find_trade_for_player(game, player_id=0) is None


class TestCompletedMonopolyWithProperty:
    """Tests for monopoly completion detection."""

    def test_detects_completed_monopoly(self) -> None:
        """Should detect when player has full monopoly."""
        game = MonopolyGame(num_players=2, seed=42)

        # Player 0 owns both Brown properties
        game.property_manager.get(1).owner = 0  # type: ignore[union-attr]
        game.property_manager.get(3).owner = 0  # type: ignore[union-attr]

        pm = game.property_manager
        assert _completed_monopoly_with_property(pm, 0, 1)
        assert _completed_monopoly_with_property(pm, 0, 3)

    def test_incomplete_monopoly_returns_false(self) -> None:
        """Should return False when monopoly is not complete."""
        game = MonopolyGame(num_players=2, seed=42)

        # Player 0 only owns one Brown property
        game.property_manager.get(1).owner = 0  # type: ignore[union-attr]
        game.property_manager.get(3).owner = 1  # type: ignore[union-attr]

        pm = game.property_manager
        assert not _completed_monopoly_with_property(pm, 0, 1)

    def test_railroad_not_buildable_monopoly(self) -> None:
        """Railroads don't form buildable monopolies."""
        game = MonopolyGame(num_players=2, seed=42)

        # Player 0 owns all railroads
        for pos in [5, 15, 25, 35]:
            game.property_manager.get(pos).owner = 0  # type: ignore[union-attr]

        pm = game.property_manager
        # This function checks for building monopolies, not railroad sets
        # Railroads don't have PropertySpace type
        assert not _completed_monopoly_with_property(pm, 0, 5)
