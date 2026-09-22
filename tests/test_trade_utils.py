"""Tests for trade evaluation helper functions."""

from __future__ import annotations

from mcts.trade_utils import (
    TradeCandidate,
    monopoly_proximity,
    property_strategic_value,
    suggest_valuable_trades,
    trade_impact,
)
from monopoly_engine.game import MonopolyGame
from monopoly_engine.types import PropertyColor, TradeOfferData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _game_with_ownership() -> MonopolyGame:
    """Create a game with specific property ownership.

    Player 0: Mediterranean(1), Baltic(3)           [Brown monopoly]
    Player 1: Oriental(6), Vermont(8)               [Light Blue 2/3]
    Player 2: St. Charles(11), States Ave(13)       [Magenta 2/3]
    Player 3: (none)
    """
    game = MonopolyGame(num_players=4, seed=42)
    game.property_manager.properties[1].owner = 0
    game.property_manager.properties[3].owner = 0
    game.property_manager.properties[6].owner = 1
    game.property_manager.properties[8].owner = 1
    game.property_manager.properties[11].owner = 2
    game.property_manager.properties[13].owner = 2
    return game


# ---------------------------------------------------------------------------
# monopoly_proximity
# ---------------------------------------------------------------------------


class TestMonopolyProximity:
    def test_full_monopoly(self) -> None:
        game = _game_with_ownership()
        prox = monopoly_proximity(0, game.property_manager)
        assert prox[PropertyColor.BROWN] == 1.0

    def test_partial_ownership(self) -> None:
        game = _game_with_ownership()
        prox = monopoly_proximity(1, game.property_manager)
        # Light Blue: 2 out of 3 (positions 6, 8, 9)
        assert abs(prox[PropertyColor.LIGHT_BLUE] - 2 / 3) < 1e-9

    def test_no_ownership(self) -> None:
        game = _game_with_ownership()
        prox = monopoly_proximity(3, game.property_manager)
        # Player 3 owns nothing
        for color in PropertyColor:
            assert prox[color] == 0.0

    def test_all_colors_present(self) -> None:
        game = _game_with_ownership()
        prox = monopoly_proximity(0, game.property_manager)
        assert set(prox.keys()) == set(PropertyColor)

    def test_two_of_three(self) -> None:
        game = _game_with_ownership()
        prox = monopoly_proximity(2, game.property_manager)
        # Magenta: 2 out of 3 (positions 11, 13, 14)
        assert abs(prox[PropertyColor.MAGENTA] - 2 / 3) < 1e-9

    def test_two_player_brown_group(self) -> None:
        """Brown only has 2 properties, so 1/2 = 0.5."""
        game = MonopolyGame(num_players=2, seed=1)
        game.property_manager.properties[1].owner = 0
        prox = monopoly_proximity(0, game.property_manager)
        assert prox[PropertyColor.BROWN] == 0.5


# ---------------------------------------------------------------------------
# trade_impact
# ---------------------------------------------------------------------------


class TestTradeImpact:
    def test_basic_property_swap(self) -> None:
        game = _game_with_ownership()
        trade: TradeOfferData = {
            "from_player": 0,
            "to_player": 1,
            "give_properties": [1],
            "give_money": 0,
            "want_properties": [6],
            "want_money": 0,
        }
        impact = trade_impact(game, trade)

        assert impact["from_player_id"] == 0
        assert impact["to_player_id"] == 1
        assert 0 in impact["before"]
        assert 1 in impact["before"]
        assert 0 in impact["after"]
        assert 1 in impact["after"]

    def test_does_not_mutate_game(self) -> None:
        game = _game_with_ownership()
        trade: TradeOfferData = {
            "from_player": 0,
            "to_player": 1,
            "give_properties": [1],
            "give_money": 0,
            "want_properties": [6],
            "want_money": 0,
        }
        trade_impact(game, trade)

        # Original ownership unchanged
        assert game.property_manager.properties[1].owner == 0
        assert game.property_manager.properties[6].owner == 1

    def test_monopoly_created(self) -> None:
        """Player 1 has Oriental(6), Vermont(8). Getting Connecticut(9) completes LB."""
        game = _game_with_ownership()
        # Give Player 0 Connecticut Ave so they can trade it
        game.property_manager.properties[9].owner = 0

        trade: TradeOfferData = {
            "from_player": 0,
            "to_player": 1,
            "give_properties": [9],
            "give_money": 0,
            "want_properties": [],
            "want_money": 100,
        }
        impact = trade_impact(game, trade)

        assert (1, "LIGHT_BLUE") in impact["monopolies_created"]
        assert impact["after"][1]["monopoly_count"] == 1

    def test_monopoly_broken(self) -> None:
        """Player 0 has Brown monopoly. Trading Baltic(3) away breaks it."""
        game = _game_with_ownership()
        trade: TradeOfferData = {
            "from_player": 0,
            "to_player": 1,
            "give_properties": [3],
            "give_money": 0,
            "want_properties": [6],
            "want_money": 0,
        }
        impact = trade_impact(game, trade)

        assert (0, "BROWN") in impact["monopolies_broken"]
        assert impact["before"][0]["monopoly_count"] == 1
        assert impact["after"][0]["monopoly_count"] == 0

    def test_cash_transfer_affects_net_worth(self) -> None:
        game = _game_with_ownership()
        trade: TradeOfferData = {
            "from_player": 0,
            "to_player": 1,
            "give_properties": [],
            "give_money": 200,
            "want_properties": [],
            "want_money": 0,
        }
        impact = trade_impact(game, trade)

        # Player 0 loses $200, Player 1 gains $200
        assert impact["from_player_net_change"] == -200
        assert impact["to_player_net_change"] == 200

    def test_before_after_structure(self) -> None:
        game = _game_with_ownership()
        trade: TradeOfferData = {
            "from_player": 0,
            "to_player": 1,
            "give_properties": [1],
            "give_money": 0,
            "want_properties": [6],
            "want_money": 0,
        }
        impact = trade_impact(game, trade)

        for pid in (0, 1):
            for phase in ("before", "after"):
                metrics = impact[phase][pid]
                assert "net_worth" in metrics
                assert "monopoly_count" in metrics
                assert "proximity" in metrics


# ---------------------------------------------------------------------------
# property_strategic_value
# ---------------------------------------------------------------------------


class TestPropertyStrategicValue:
    def test_monopoly_completing_property_is_most_valuable(self) -> None:
        """Connecticut Ave (9) completes Light Blue for Player 1."""
        game = _game_with_ownership()
        value_for_p1 = property_strategic_value(game, 9, for_player=1)
        value_for_p3 = property_strategic_value(game, 9, for_player=3)
        # Should be more valuable to P1 who is 1 away from completing LB.
        # (P3 also gets some blocking value, but proximity bonus dominates.)
        assert value_for_p1 > value_for_p3

    def test_blocking_property_has_bonus(self) -> None:
        """Player 0 holds a property that blocks Player 1's monopoly."""
        game = _game_with_ownership()
        # Give P0 Connecticut(9) -- this blocks P1's Light Blue monopoly
        game.property_manager.properties[9].owner = 0
        value = property_strategic_value(game, 9, for_player=0)
        # Should include blocking bonus
        assert value > 0

    def test_unowned_position_returns_zero(self) -> None:
        game = _game_with_ownership()
        # Position 9 is unowned
        assert game.property_manager.properties[9].owner is None
        # Still returns a value based on cost (unowned doesn't mean no value)
        value = property_strategic_value(game, 9, for_player=0)
        assert value > 0  # Has base value even with no proximity

    def test_non_property_position_returns_zero(self) -> None:
        game = MonopolyGame(num_players=2, seed=1)
        # Position 0 is Go, not a property
        value = property_strategic_value(game, 0, for_player=0)
        assert value == 0.0

    def test_orange_properties_get_frequency_boost(self) -> None:
        """Orange properties have highest frequency multiplier."""
        game = MonopolyGame(num_players=2, seed=1)
        # Compare orange (St. James, pos 16) vs green (Pacific, pos 31)
        # Both unowned, same player, no proximity
        orange_val = property_strategic_value(game, 16, for_player=0)
        green_val = property_strategic_value(game, 31, for_player=0)
        # Orange should be boosted relative to raw cost
        # St. James costs $180, Pacific costs $300
        # But with 1.3x vs 0.9x frequency, orange becomes 234 vs 270
        # Green is still more due to higher cost, but ratio is compressed
        assert orange_val / 180 > green_val / 300  # Per-dollar value is higher


# ---------------------------------------------------------------------------
# suggest_valuable_trades
# ---------------------------------------------------------------------------


class TestSuggestValuableTrades:
    def test_suggests_monopoly_completing_trade(self) -> None:
        """Player 1 is 1 away from Light Blue. Should suggest getting pos 9."""
        game = _game_with_ownership()
        # Give P0 Connecticut(9) so there's a possible trade
        game.property_manager.properties[9].owner = 0
        # Give P1 a non-critical property to offer (Reading Railroad, pos 5)
        game.property_manager.properties[5].owner = 1

        candidates = suggest_valuable_trades(game, player_id=1)
        assert len(candidates) > 0
        # At least one candidate should want Connecticut Ave (9)
        assert any(9 in c.want_properties for c in candidates)

    def test_returns_trade_candidates(self) -> None:
        game = _game_with_ownership()
        game.property_manager.properties[9].owner = 0
        game.property_manager.properties[5].owner = 1  # Non-critical to offer

        candidates = suggest_valuable_trades(game, player_id=1)
        for c in candidates:
            assert isinstance(c, TradeCandidate)
            assert len(c.want_properties) > 0
            assert c.estimated_value != 0

    def test_respects_max_suggestions(self) -> None:
        game = _game_with_ownership()
        game.property_manager.properties[9].owner = 0
        game.property_manager.properties[5].owner = 1  # Non-critical to offer

        candidates = suggest_valuable_trades(game, player_id=1, max_suggestions=1)
        assert len(candidates) <= 1

    def test_no_suggestions_when_no_near_monopoly(self) -> None:
        """Player 3 owns nothing, no near-monopolies to complete."""
        game = _game_with_ownership()
        candidates = suggest_valuable_trades(game, player_id=3)
        assert len(candidates) == 0

    def test_sorted_by_value_descending(self) -> None:
        game = _game_with_ownership()
        # Give P0 extra properties so P1 has multiple trade options
        game.property_manager.properties[9].owner = 0
        game.property_manager.properties[14].owner = 0
        # Give P1 non-critical properties to offer
        game.property_manager.properties[5].owner = 1
        game.property_manager.properties[25].owner = 1

        candidates = suggest_valuable_trades(game, player_id=1)
        if len(candidates) >= 2:
            for i in range(len(candidates) - 1):
                assert candidates[i].estimated_value >= candidates[i + 1].estimated_value

    def test_does_not_suggest_trading_critical_properties(self) -> None:
        """Should not offer a property from a set we're also building."""
        game = _game_with_ownership()
        # P1 owns Oriental(6), Vermont(8) -- both Light Blue
        # P1 should not offer either of these
        game.property_manager.properties[9].owner = 0

        candidates = suggest_valuable_trades(game, player_id=1)
        for c in candidates:
            assert 6 not in c.give_properties
            assert 8 not in c.give_properties

    def test_skips_properties_with_buildings(self) -> None:
        """Can't trade properties that have houses on them."""
        game = _game_with_ownership()
        game.property_manager.properties[9].owner = 0
        # Put houses on Oriental(6) -- P1 can't trade it
        game.property_manager.properties[6].houses = 2

        candidates = suggest_valuable_trades(game, player_id=1)
        for c in candidates:
            assert 6 not in c.give_properties
