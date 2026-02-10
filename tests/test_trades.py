"""Tests for Phase 2.5a trade functionality.

This module tests:
- Trade encoding/decoding
- Trade action masking
- Trade execution in the environment
- Trade reward calculation
"""

import numpy as np
import pytest

from monopoly_engine import MonopolyGame
from monopoly_gym import (
    MonopolyEnv,
    OFFSET_SIMPLE_TRADE,
    OFFSET_ACCEPT_TRADE,
    OFFSET_REJECT_TRADE,
    SIMPLE_TRADE_DIM,
    encode_simple_trade,
    decode_simple_trade,
    get_simple_trade_mask,
    calculate_trade_rewards,
    TradeRewardConfig,
    ActionEncoder,
)
from monopoly_gym.trades import (
    BUYABLE_POSITIONS,
    find_trade_for_player,
    get_trade_response_mask,
    _completed_monopoly_with_property,
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

    def test_mask_disables_trades_with_mortgaged(
        self, game_with_properties: MonopolyGame
    ) -> None:
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
        game = MonopolyGame(num_players=2, seed=42)

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
        game = MonopolyGame(num_players=3, seed=42)

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
            proposer_got=3,   # Got Baltic (completes Brown)
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
            proposer_got=6,   # Got Oriental
            accepted=True,
            config=config,
        )

        # Proposer gave away monopoly to responder
        assert rewards[0] <= -1.5, "Should get penalty for giving opponent monopoly"


class TestActionEncoderWithTrades:
    """Tests for ActionEncoder with trade support."""

    def test_action_space_size_with_trades(self) -> None:
        """With trades enabled, action space should be 907."""
        encoder = ActionEncoder(enable_trades=True)
        assert encoder.action_space_size == 907

    def test_action_space_size_without_trades(self) -> None:
        """Without trades, action space should be 149."""
        encoder = ActionEncoder(enable_trades=False)
        assert encoder.action_space_size == 149

    def test_decode_trade_action(self) -> None:
        """Should decode trade action to ProposeTrade."""
        game = MonopolyGame(num_players=2, seed=42)
        game.property_manager.get(1).owner = 0  # type: ignore[union-attr]
        game.property_manager.get(3).owner = 1  # type: ignore[union-attr]

        encoder = ActionEncoder(enable_trades=True)
        action_idx = encode_simple_trade(1, 3)

        decoded = encoder.decode(action_idx, player_id=0, game=game)

        from monopoly_engine import ProposeTrade

        assert isinstance(decoded, ProposeTrade)
        assert decoded.give_properties == [1]
        assert decoded.want_properties == [3]
        assert decoded.to_player == 1

    def test_get_action_mask_includes_trades(self) -> None:
        """Action mask should include trade actions when enabled."""
        game = MonopolyGame(num_players=2, seed=42)
        game.property_manager.get(1).owner = 0  # type: ignore[union-attr]
        game.property_manager.get(3).owner = 1  # type: ignore[union-attr]

        encoder = ActionEncoder(enable_trades=True)
        mask = encoder.get_action_mask(game, player_id=0)

        # Should have some trade actions enabled
        trade_mask = mask[OFFSET_SIMPLE_TRADE:OFFSET_SIMPLE_TRADE + SIMPLE_TRADE_DIM]
        assert trade_mask.sum() > 0, "Should have valid trade actions"

    def test_get_action_mask_trade_response_mode(self) -> None:
        """In trade response mode, only accept/reject should be valid."""
        game = MonopolyGame(num_players=2, seed=42)
        game.property_manager.get(1).owner = 0  # type: ignore[union-attr]
        game.property_manager.get(3).owner = 1  # type: ignore[union-attr]

        # Create pending trade
        game.propose_trade(
            from_player=0,
            to_player=1,
            give_properties=[1],
            want_properties=[3],
            give_money=0,
            want_money=0,
        )

        encoder = ActionEncoder(enable_trades=True)
        mask = encoder.get_action_mask(game, player_id=1, pending_trade_response=True)

        # Only accept/reject should be valid
        assert mask[OFFSET_ACCEPT_TRADE], "Accept should be valid"
        assert mask[OFFSET_REJECT_TRADE], "Reject should be valid"

        # Gameplay actions should not be valid
        assert not mask[:OFFSET_SIMPLE_TRADE].any(), "Gameplay actions should be invalid"

        # Trade proposal actions should not be valid
        trade_mask = mask[OFFSET_SIMPLE_TRADE:OFFSET_SIMPLE_TRADE + SIMPLE_TRADE_DIM]
        assert not trade_mask.any(), "Trade proposals should be invalid during response"


class TestMonopolyEnvWithTrades:
    """Tests for MonopolyEnv with trade support."""

    def test_env_creation_with_trades(self) -> None:
        """Environment should support enable_trades parameter."""
        env = MonopolyEnv(num_players=2, enable_trades=True)
        assert env.enable_trades
        assert env.action_encoder.action_space_size == 907

    def test_env_action_space_with_trades(self) -> None:
        """Action space should reflect trade actions when enabled."""
        env = MonopolyEnv(num_players=2, enable_trades=True)
        env.reset(seed=42)
        assert env._action_space.n == 907  # type: ignore[attr-defined]

    def test_env_trade_proposal_flow(self) -> None:
        """Trade proposal should switch to responder."""
        env = MonopolyEnv(num_players=2, enable_trades=True)
        env.reset(seed=42)

        assert env.game is not None

        # Give players properties
        env.game.property_manager.get(1).owner = 0  # type: ignore[union-attr]
        env.game.property_manager.get(3).owner = 1  # type: ignore[union-attr]
        env._update_infos()  # Refresh cached action mask after state change

        # Player 0 starts
        assert env.agent_selection == "player_0"

        # Player 0 proposes trade
        trade_action = encode_simple_trade(1, 3)
        env.step(trade_action)

        # Should switch to player 1 for response
        assert env.agent_selection == "player_1"
        assert env._pending_trade_response

    def test_env_trade_accept_flow(self) -> None:
        """Accepting trade should execute swap and return to proposer."""
        env = MonopolyEnv(num_players=2, enable_trades=True)
        env.reset(seed=42)

        assert env.game is not None

        # Give players properties
        env.game.property_manager.get(1).owner = 0  # type: ignore[union-attr]
        env.game.property_manager.get(3).owner = 1  # type: ignore[union-attr]
        env._update_infos()  # Refresh cached action mask after state change

        # Player 0 proposes trade: Mediterranean for Baltic
        trade_action = encode_simple_trade(1, 3)
        env.step(trade_action)

        # Player 1 accepts
        env.step(OFFSET_ACCEPT_TRADE)

        # Properties should be swapped
        assert env.game.property_manager.get(1).owner == 1  # type: ignore[union-attr]
        assert env.game.property_manager.get(3).owner == 0  # type: ignore[union-attr]

        # Should return to player 0's turn
        assert env.agent_selection == "player_0"
        assert not env._pending_trade_response

    def test_env_trade_reject_flow(self) -> None:
        """Rejecting trade should keep properties unchanged and return to proposer."""
        env = MonopolyEnv(num_players=2, enable_trades=True)
        env.reset(seed=42)

        assert env.game is not None

        # Give players properties
        env.game.property_manager.get(1).owner = 0  # type: ignore[union-attr]
        env.game.property_manager.get(3).owner = 1  # type: ignore[union-attr]
        env._update_infos()  # Refresh cached action mask after state change

        # Player 0 proposes trade
        trade_action = encode_simple_trade(1, 3)
        env.step(trade_action)

        # Player 1 rejects
        env.step(OFFSET_REJECT_TRADE)

        # Properties should be unchanged
        assert env.game.property_manager.get(1).owner == 0  # type: ignore[union-attr]
        assert env.game.property_manager.get(3).owner == 1  # type: ignore[union-attr]

        # Should return to player 0's turn
        assert env.agent_selection == "player_0"
        assert not env._pending_trade_response

    def test_env_trade_observation_includes_context(self) -> None:
        """Observation should include trade context when trades enabled."""
        env = MonopolyEnv(num_players=2, enable_trades=True)
        env.reset(seed=42)

        obs = env.observe("player_0")
        assert "trade_context" in obs

    def test_env_trade_mask_during_response(self) -> None:
        """During trade response, only accept/reject should be valid."""
        env = MonopolyEnv(num_players=2, enable_trades=True)
        env.reset(seed=42)

        assert env.game is not None

        # Give players properties
        env.game.property_manager.get(1).owner = 0  # type: ignore[union-attr]
        env.game.property_manager.get(3).owner = 1  # type: ignore[union-attr]
        env._update_infos()  # Refresh cached action mask after state change

        # Player 0 proposes trade
        trade_action = encode_simple_trade(1, 3)
        env.step(trade_action)

        # Get mask for player 1 (responder)
        _, _, _, _, info = env.last()
        mask = info["action_mask"]

        # Only accept/reject should be valid
        assert mask[OFFSET_ACCEPT_TRADE]
        assert mask[OFFSET_REJECT_TRADE]
        assert not mask[:OFFSET_SIMPLE_TRADE].any()


class TestFindTradeForPlayer:
    """Tests for find_trade_for_player utility."""

    def test_finds_pending_trade(self) -> None:
        """Should find trade directed at player."""
        game = MonopolyGame(num_players=2, seed=42)
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
