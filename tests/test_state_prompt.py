"""Tests for LLM state serialization."""

from __future__ import annotations

from mcts.llm.state_prompt import (
    serialize_game_state,
    serialize_player_summary,
    serialize_property_landscape,
    serialize_trade_proposal,
)
from monopoly_engine.game import MonopolyGame
from monopoly_engine.types import TradeOfferData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _game_with_ownership() -> MonopolyGame:
    """Create a 4-player game with specific property ownership.

    Player 0: Mediterranean(1), Baltic(3), Reading RR(5)
    Player 1: Oriental(6), Vermont(8)
    Player 2: St. Charles(11), States Ave(13)
    Player 3: (none)
    """
    game = MonopolyGame(num_players=4, seed=42)
    game.property_manager.properties[1].owner = 0
    game.property_manager.properties[3].owner = 0
    game.property_manager.properties[5].owner = 0
    game.property_manager.properties[6].owner = 1
    game.property_manager.properties[8].owner = 1
    game.property_manager.properties[11].owner = 2
    game.property_manager.properties[13].owner = 2
    return game


# ---------------------------------------------------------------------------
# serialize_game_state
# ---------------------------------------------------------------------------

class TestSerializeGameState:
    def test_includes_player_identity(self) -> None:
        game = _game_with_ownership()
        text = serialize_game_state(game, 0)
        assert "You are Player 0" in text
        assert "$1500" in text

    def test_includes_player_properties(self) -> None:
        game = _game_with_ownership()
        text = serialize_game_state(game, 0)
        assert "Mediterranean Avenue [1]" in text
        assert "Baltic Avenue [3]" in text
        assert "Reading Railroad [5]" in text

    def test_includes_opponents(self) -> None:
        game = _game_with_ownership()
        text = serialize_game_state(game, 0)
        assert "Player 1" in text
        assert "Player 2" in text
        assert "Player 3" in text

    def test_shows_near_monopoly_hints_for_opponents(self) -> None:
        game = _game_with_ownership()
        text = serialize_game_state(game, 0)
        # Player 1 has 2/3 Light Blue
        assert "LIGHT_BLUE: owns 2/3" in text

    def test_shows_monopoly_hint_for_own_properties(self) -> None:
        game = _game_with_ownership()
        text = serialize_game_state(game, 0)
        # Player 0 has Brown monopoly
        assert "BROWN: MONOPOLY" in text

    def test_includes_unowned_properties(self) -> None:
        game = _game_with_ownership()
        text = serialize_game_state(game, 0)
        assert "UNOWNED:" in text
        assert "Connecticut Avenue [9]" in text

    def test_includes_game_info(self) -> None:
        game = _game_with_ownership()
        text = serialize_game_state(game, 0)
        assert "Turn 0" in text
        assert "position 0" in text

    def test_includes_position_instruction(self) -> None:
        game = _game_with_ownership()
        text = serialize_game_state(game, 0)
        assert "position numbers" in text

    def test_perspective_player_shown_first(self) -> None:
        game = _game_with_ownership()
        text = serialize_game_state(game, 1)
        # Player 1's properties should come before opponents
        your_idx = text.index("YOUR PROPERTIES:")
        opp_idx = text.index("OPPONENTS:")
        assert your_idx < opp_idx

    def test_bankrupt_opponents_excluded(self) -> None:
        game = _game_with_ownership()
        game.players[3].bankrupt = True
        text = serialize_game_state(game, 0)
        # Player index 3 (named "Player 4") should not appear as an opponent
        opp_section = text[text.index("OPPONENTS:"):text.index("UNOWNED:")]
        # The opponent lines start with "- Player <id>", bankrupt pid=3 excluded
        assert "- Player 3" not in opp_section

    def test_jail_status_shown(self) -> None:
        game = _game_with_ownership()
        game.players[0].in_jail = True
        game.players[0].position = 10
        text = serialize_game_state(game, 0)
        assert "IN JAIL" in text

    def test_output_under_2000_chars_for_typical_game(self) -> None:
        """Rough proxy for <1000 tokens (avg ~4 chars/token)."""
        game = _game_with_ownership()
        text = serialize_game_state(game, 0)
        assert len(text) < 4000


# ---------------------------------------------------------------------------
# serialize_player_summary
# ---------------------------------------------------------------------------

class TestSerializePlayerSummary:
    def test_includes_money(self) -> None:
        game = _game_with_ownership()
        text = serialize_player_summary(game, 0)
        assert "$1500" in text

    def test_includes_net_worth(self) -> None:
        game = _game_with_ownership()
        text = serialize_player_summary(game, 0)
        assert "net worth" in text

    def test_includes_properties(self) -> None:
        game = _game_with_ownership()
        text = serialize_player_summary(game, 0)
        assert "Mediterranean Avenue [1]" in text

    def test_includes_monopoly_info(self) -> None:
        game = _game_with_ownership()
        text = serialize_player_summary(game, 0)
        assert "BROWN" in text

    def test_shows_jail_status(self) -> None:
        game = _game_with_ownership()
        game.players[1].in_jail = True
        text = serialize_player_summary(game, 1)
        assert "In jail" in text

    def test_shows_bankrupt_status(self) -> None:
        game = _game_with_ownership()
        game.players[3].bankrupt = True
        text = serialize_player_summary(game, 3)
        assert "Bankrupt" in text


# ---------------------------------------------------------------------------
# serialize_property_landscape
# ---------------------------------------------------------------------------

class TestSerializePropertyLandscape:
    def test_groups_by_color(self) -> None:
        game = _game_with_ownership()
        text = serialize_property_landscape(game)
        assert "BROWN" in text
        assert "LIGHT_BLUE" in text
        assert "MAGENTA" in text

    def test_shows_monopoly_status(self) -> None:
        game = _game_with_ownership()
        text = serialize_property_landscape(game)
        assert "MONOPOLY" in text  # Player 0 has brown monopoly

    def test_shows_near_monopoly(self) -> None:
        game = _game_with_ownership()
        text = serialize_property_landscape(game)
        # Player 1 has 2/3 Light Blue
        assert "2/3" in text

    def test_shows_unowned(self) -> None:
        game = _game_with_ownership()
        text = serialize_property_landscape(game)
        assert "unowned" in text

    def test_shows_houses(self) -> None:
        game = _game_with_ownership()
        game.property_manager.properties[1].houses = 3
        text = serialize_property_landscape(game)
        assert "3h" in text

    def test_shows_hotel(self) -> None:
        game = _game_with_ownership()
        game.property_manager.properties[1].houses = 5
        text = serialize_property_landscape(game)
        assert "HOTEL" in text

    def test_shows_mortgaged(self) -> None:
        game = _game_with_ownership()
        game.property_manager.properties[5].mortgaged = True
        text = serialize_property_landscape(game)
        assert "[M]" in text


# ---------------------------------------------------------------------------
# serialize_trade_proposal
# ---------------------------------------------------------------------------

class TestSerializeTradeProposal:
    def test_basic_trade(self) -> None:
        game = _game_with_ownership()
        trade: TradeOfferData = {
            "from_player": 0,
            "to_player": 1,
            "give_properties": [3],
            "give_money": 0,
            "want_properties": [6],
            "want_money": 0,
        }
        text = serialize_trade_proposal(trade, game)
        assert "Player 0" in text
        assert "Player 1" in text
        assert "Baltic Avenue [3]" in text
        assert "Oriental Avenue [6]" in text

    def test_with_cash(self) -> None:
        game = _game_with_ownership()
        trade: TradeOfferData = {
            "from_player": 0,
            "to_player": 1,
            "give_properties": [3],
            "give_money": 100,
            "want_properties": [6],
            "want_money": 0,
        }
        text = serialize_trade_proposal(trade, game)
        assert "$100" in text

    def test_nothing_offered(self) -> None:
        game = _game_with_ownership()
        trade: TradeOfferData = {
            "from_player": 0,
            "to_player": 1,
            "give_properties": [],
            "give_money": 0,
            "want_properties": [6],
            "want_money": 0,
        }
        text = serialize_trade_proposal(trade, game)
        assert "(nothing)" in text

    def test_multi_property_trade(self) -> None:
        game = _game_with_ownership()
        trade: TradeOfferData = {
            "from_player": 0,
            "to_player": 1,
            "give_properties": [1, 3],
            "give_money": 50,
            "want_properties": [6, 8],
            "want_money": 0,
        }
        text = serialize_trade_proposal(trade, game)
        assert "Mediterranean Avenue [1]" in text
        assert "Baltic Avenue [3]" in text
        assert "Oriental Avenue [6]" in text
        assert "Vermont Avenue [8]" in text
        assert "$50" in text
