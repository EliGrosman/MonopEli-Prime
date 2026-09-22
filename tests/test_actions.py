"""Tests for actions.py module.

These tests verify that all action classes properly validate and execute
their respective game operations. Phase orchestration is tested separately in tests/foundation.
"""

from typing import Any

import pytest

from monopoly_engine import (
    JAIL_FINE,
    PROPERTY_GROUPS,
    AcceptTrade,
    BuildHotel,
    BuildHouse,
    BuyProperty,
    DeclareBankruptcy,
    EndTurn,
    MonopolyGame,
    MortgageProperty,
    PayJailFine,
    Player,
    PropertyColor,
    PropertyManager,
    ProposeTrade,
    RejectTrade,
    RollDice,
    SellHotel,
    SellHouse,
    UnmortgageProperty,
    UseJailCard,
)


class MockGame(MonopolyGame):
    """Action-mechanics fixture; foundation tests exercise authoritative phases."""

    def __init__(self, players, property_manager):
        super().__init__(len(players), seed=1, rules_id="foundation-trade-v1")
        self.state.phase = "asset_management"
        self.state.roll_owed = False
        self.pending_trades = self.state.pending_trades
        self.next_trade_id = 0
        self.players = self.state.players = players
        self.property_manager = self.state.property_manager = property_manager

    def validate_phase(self, action):
        return True, ""

    def _return_jail_card(self, player_id):
        # These payload unit tests do not draw decks. Provenance is tested
        # with real decks in tests/foundation.
        pass

    def _add_pending_trade(
        self,
        from_player: int,
        to_player: int,
        give_properties: list[int],
        give_money: int,
        want_properties: list[int],
        want_money: int,
    ) -> None:
        """Mock trade proposal handler."""
        trade_id = self.next_trade_id
        self.next_trade_id += 1
        self.pending_trades[trade_id] = {
            "from_player": from_player,
            "to_player": to_player,
            "give_properties": give_properties,
            "give_money": give_money,
            "want_properties": want_properties,
            "want_money": want_money,
        }

    def _get_pending_trade(self, trade_id: int) -> dict[str, Any] | None:
        """Mock get pending trade."""
        return self.pending_trades.get(trade_id)

    def _remove_pending_trade(self, trade_id: int) -> None:
        """Mock remove pending trade."""
        self.pending_trades.pop(trade_id, None)


@pytest.fixture
def mock_game() -> MockGame:
    """Create a mock game with two players."""
    players = [
        Player(id=0, name="Player 1", money=1500),
        Player(id=1, name="Player 2", money=1500),
    ]
    return MockGame(
        players=players,
        property_manager=PropertyManager(),
    )


@pytest.fixture
def mock_game_four_players() -> MockGame:
    """Create a mock game with four players."""
    players = [Player(id=i, name=f"Player {i + 1}", money=1500) for i in range(4)]
    return MockGame(
        players=players,
        property_manager=PropertyManager(),
    )


class TestRollDice:
    """Tests for RollDice action."""

    def test_validate_success(self, mock_game: MockGame) -> None:
        """Valid dice roll should pass validation."""
        action = RollDice(player_id=0)
        valid, error = action.validate(mock_game)
        assert valid
        assert error == ""

    def test_validate_not_your_turn(self, mock_game: MockGame) -> None:
        """Cannot roll dice when it's not your turn."""
        action = RollDice(player_id=1)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "Not your turn" in error

    def test_validate_bankrupt_player(self, mock_game: MockGame) -> None:
        """Bankrupt player cannot roll dice."""
        mock_game.players[0].bankrupt = True
        action = RollDice(player_id=0)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "bankrupt" in error.lower()

    def test_to_dict(self) -> None:
        """Test serialization."""
        action = RollDice(player_id=0)
        data = action.to_dict()
        assert data["type"] == "RollDice"
        assert data["player_id"] == 0


class TestBuyProperty:
    """Tests for BuyProperty action."""

    def test_validate_success(self, mock_game: MockGame) -> None:
        """Valid property purchase should pass validation."""
        mock_game.players[0].position = 1  # Mediterranean
        action = BuyProperty(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert valid
        assert error == ""

    def test_validate_not_your_turn(self, mock_game: MockGame) -> None:
        """Cannot buy property when it's not your turn."""
        mock_game.players[1].position = 1
        action = BuyProperty(player_id=1, property_id=1)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "Not your turn" in error

    def test_validate_not_on_property(self, mock_game: MockGame) -> None:
        """Cannot buy property if not standing on it."""
        mock_game.players[0].position = 0  # GO
        action = BuyProperty(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "Not on this property" in error

    def test_validate_already_owned(self, mock_game: MockGame) -> None:
        """Cannot buy property that's already owned."""
        mock_game.players[0].position = 1
        mock_game.property_manager.properties[1].owner = 1
        action = BuyProperty(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "owned" in error.lower()

    def test_validate_insufficient_funds(self, mock_game: MockGame) -> None:
        """Cannot buy property without enough money."""
        mock_game.players[0].position = 1
        mock_game.players[0].money = 10  # Mediterranean costs $60
        action = BuyProperty(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "funds" in error.lower()

    def test_execute_purchase(self, mock_game: MockGame) -> None:
        """Execute should buy the property."""
        mock_game.players[0].position = 1
        initial_money = mock_game.players[0].money
        action = BuyProperty(player_id=0, property_id=1)
        action.execute(mock_game)

        prop = mock_game.property_manager.properties[1]
        assert prop.owner == 0
        assert mock_game.players[0].money == initial_money - 60  # Mediterranean cost

    def test_to_dict(self) -> None:
        """Test serialization."""
        action = BuyProperty(player_id=0, property_id=1)
        data = action.to_dict()
        assert data["type"] == "BuyProperty"
        assert data["player_id"] == 0
        assert data["property_id"] == 1


class TestBuildHouse:
    """Tests for BuildHouse action."""

    def test_validate_success(self, mock_game: MockGame) -> None:
        """Valid house building should pass validation."""
        # Give player monopoly on brown
        for pos in PROPERTY_GROUPS[PropertyColor.BROWN]:
            mock_game.property_manager.properties[pos].owner = 0
        mock_game.players[0].money = 1500

        action = BuildHouse(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert valid
        assert error == ""

    def test_validate_not_your_turn(self, mock_game: MockGame) -> None:
        """Cannot build when it's not your turn."""
        action = BuildHouse(player_id=1, property_id=1)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "Not your turn" in error

    def test_validate_no_monopoly(self, mock_game: MockGame) -> None:
        """Cannot build without a monopoly."""
        mock_game.property_manager.properties[1].owner = 0
        action = BuildHouse(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "monopoly" in error.lower()

    def test_validate_mortgaged(self, mock_game: MockGame) -> None:
        """Cannot build on mortgaged property."""
        for pos in PROPERTY_GROUPS[PropertyColor.BROWN]:
            mock_game.property_manager.properties[pos].owner = 0
        mock_game.property_manager.properties[1].mortgaged = True

        action = BuildHouse(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "mortgaged" in error.lower()

    def test_validate_already_hotel(self, mock_game: MockGame) -> None:
        """Cannot build when property already has a hotel."""
        for pos in PROPERTY_GROUPS[PropertyColor.BROWN]:
            mock_game.property_manager.properties[pos].owner = 0
        mock_game.property_manager.properties[1].houses = 5

        action = BuildHouse(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "hotel" in error.lower()

    def test_validate_even_building(self, mock_game: MockGame) -> None:
        """Must build evenly across color group."""
        for pos in PROPERTY_GROUPS[PropertyColor.BROWN]:
            mock_game.property_manager.properties[pos].owner = 0
        mock_game.property_manager.properties[1].houses = 2
        mock_game.property_manager.properties[3].houses = 0

        action = BuildHouse(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "evenly" in error.lower()

    def test_validate_no_houses_remaining(self, mock_game: MockGame) -> None:
        """Cannot build when no houses remain."""
        for pos in PROPERTY_GROUPS[PropertyColor.BROWN]:
            mock_game.property_manager.properties[pos].owner = 0
        mock_game.houses_remaining = 0

        action = BuildHouse(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "houses remaining" in error.lower()

    def test_execute_build_house(self, mock_game: MockGame) -> None:
        """Execute should build a house."""
        for pos in PROPERTY_GROUPS[PropertyColor.BROWN]:
            mock_game.property_manager.properties[pos].owner = 0

        initial_houses = mock_game.houses_remaining
        initial_money = mock_game.players[0].money
        action = BuildHouse(player_id=0, property_id=1)
        action.execute(mock_game)

        prop = mock_game.property_manager.properties[1]
        assert prop.houses == 1
        assert mock_game.houses_remaining == initial_houses - 1
        assert mock_game.players[0].money == initial_money - 50  # Brown house cost

    def test_execute_build_hotel(self, mock_game: MockGame) -> None:
        """Execute should build a hotel when at 4 houses."""
        for pos in PROPERTY_GROUPS[PropertyColor.BROWN]:
            mock_game.property_manager.properties[pos].owner = 0
        mock_game.property_manager.properties[1].houses = 4
        mock_game.property_manager.properties[3].houses = 4

        initial_houses = mock_game.houses_remaining
        initial_hotels = mock_game.hotels_remaining
        action = BuildHouse(player_id=0, property_id=1)
        action.execute(mock_game)

        prop = mock_game.property_manager.properties[1]
        assert prop.houses == 5
        assert mock_game.houses_remaining == initial_houses + 4
        assert mock_game.hotels_remaining == initial_hotels - 1

    def test_to_dict(self) -> None:
        """Test serialization."""
        action = BuildHouse(player_id=0, property_id=1)
        data = action.to_dict()
        assert data["type"] == "BuildHouse"
        assert data["property_id"] == 1


class TestBuildHotel:
    """Tests for BuildHotel action."""

    def test_validate_success(self, mock_game: MockGame) -> None:
        """Valid hotel building should pass validation."""
        for pos in PROPERTY_GROUPS[PropertyColor.BROWN]:
            prop = mock_game.property_manager.properties[pos]
            prop.owner = 0
            prop.houses = 4

        action = BuildHotel(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert valid
        assert error == ""

    def test_validate_not_4_houses(self, mock_game: MockGame) -> None:
        """Cannot build hotel without exactly 4 houses."""
        for pos in PROPERTY_GROUPS[PropertyColor.BROWN]:
            prop = mock_game.property_manager.properties[pos]
            prop.owner = 0
            prop.houses = 2

        action = BuildHotel(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "4 houses" in error.lower()

    def test_execute_build_hotel(self, mock_game: MockGame) -> None:
        """Execute should build a hotel."""
        for pos in PROPERTY_GROUPS[PropertyColor.BROWN]:
            prop = mock_game.property_manager.properties[pos]
            prop.owner = 0
            prop.houses = 4

        initial_houses = mock_game.houses_remaining
        initial_hotels = mock_game.hotels_remaining
        action = BuildHotel(player_id=0, property_id=1)
        action.execute(mock_game)

        prop = mock_game.property_manager.properties[1]
        assert prop.houses == 5
        assert mock_game.houses_remaining == initial_houses + 4
        assert mock_game.hotels_remaining == initial_hotels - 1


class TestSellHouse:
    """Tests for SellHouse action."""

    def test_validate_success(self, mock_game: MockGame) -> None:
        """Valid house selling should pass validation."""
        for pos in PROPERTY_GROUPS[PropertyColor.BROWN]:
            prop = mock_game.property_manager.properties[pos]
            prop.owner = 0
            prop.houses = 2

        action = SellHouse(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert valid
        assert error == ""

    def test_validate_no_houses(self, mock_game: MockGame) -> None:
        """Cannot sell houses if property has none."""
        mock_game.property_manager.properties[1].owner = 0
        action = SellHouse(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "No houses" in error

    def test_validate_even_selling(self, mock_game: MockGame) -> None:
        """Must sell evenly across color group."""
        for pos in PROPERTY_GROUPS[PropertyColor.BROWN]:
            mock_game.property_manager.properties[pos].owner = 0
        mock_game.property_manager.properties[1].houses = 2
        mock_game.property_manager.properties[3].houses = 4

        action = SellHouse(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "evenly" in error.lower()

    def test_execute_sell_house(self, mock_game: MockGame) -> None:
        """Execute should sell a house."""
        for pos in PROPERTY_GROUPS[PropertyColor.BROWN]:
            prop = mock_game.property_manager.properties[pos]
            prop.owner = 0
            prop.houses = 2

        initial_houses = mock_game.houses_remaining
        initial_money = mock_game.players[0].money
        action = SellHouse(player_id=0, property_id=1)
        action.execute(mock_game)

        prop = mock_game.property_manager.properties[1]
        assert prop.houses == 1
        assert mock_game.houses_remaining == initial_houses + 1
        assert mock_game.players[0].money == initial_money + 25  # Half of $50

    def test_execute_sell_hotel(self, mock_game: MockGame) -> None:
        """Execute should downgrade hotel to 4 houses."""
        for pos in PROPERTY_GROUPS[PropertyColor.BROWN]:
            prop = mock_game.property_manager.properties[pos]
            prop.owner = 0
            prop.houses = 5

        initial_hotels = mock_game.hotels_remaining
        action = SellHouse(player_id=0, property_id=1)
        action.execute(mock_game)

        prop = mock_game.property_manager.properties[1]
        assert prop.houses == 4
        assert mock_game.hotels_remaining == initial_hotels + 1

    def test_execute_sell_hotel_no_houses(self, mock_game: MockGame) -> None:
        """Execute should handle selling hotel when no houses available."""
        for pos in PROPERTY_GROUPS[PropertyColor.BROWN]:
            prop = mock_game.property_manager.properties[pos]
            prop.owner = 0
            prop.houses = 5

        mock_game.houses_remaining = 0
        action = SellHouse(player_id=0, property_id=1)
        action.execute(mock_game)

        prop = mock_game.property_manager.properties[1]
        assert prop.houses == 0
        assert mock_game.houses_remaining == 0


class TestSellHotel:
    """Tests for SellHotel action."""

    def test_validate_success(self, mock_game: MockGame) -> None:
        """Valid hotel selling should pass validation."""
        for pos in PROPERTY_GROUPS[PropertyColor.BROWN]:
            prop = mock_game.property_manager.properties[pos]
            prop.owner = 0
            prop.houses = 5

        action = SellHotel(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert valid
        assert error == ""

    def test_validate_no_hotel(self, mock_game: MockGame) -> None:
        """Cannot sell hotel if property doesn't have one."""
        mock_game.property_manager.properties[1].owner = 0
        mock_game.property_manager.properties[1].houses = 4
        action = SellHotel(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "hotel" in error.lower()

    def test_execute_sell_hotel(self, mock_game: MockGame) -> None:
        """Execute should sell hotel and return to 4 houses."""
        for pos in PROPERTY_GROUPS[PropertyColor.BROWN]:
            prop = mock_game.property_manager.properties[pos]
            prop.owner = 0
            prop.houses = 5

        initial_hotels = mock_game.hotels_remaining
        action = SellHotel(player_id=0, property_id=1)
        action.execute(mock_game)

        prop = mock_game.property_manager.properties[1]
        assert prop.houses == 4
        assert mock_game.hotels_remaining == initial_hotels + 1


class TestMortgageProperty:
    """Tests for MortgageProperty action."""

    def test_validate_success(self, mock_game: MockGame) -> None:
        """Valid mortgage should pass validation."""
        mock_game.property_manager.properties[1].owner = 0
        action = MortgageProperty(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert valid
        assert error == ""

    def test_validate_not_owner(self, mock_game: MockGame) -> None:
        """Cannot mortgage property you don't own."""
        mock_game.property_manager.properties[1].owner = 1
        action = MortgageProperty(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "own" in error.lower()

    def test_validate_already_mortgaged(self, mock_game: MockGame) -> None:
        """Cannot mortgage already mortgaged property."""
        prop = mock_game.property_manager.properties[1]
        prop.owner = 0
        prop.mortgaged = True
        action = MortgageProperty(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "already mortgaged" in error.lower()

    def test_validate_has_houses(self, mock_game: MockGame) -> None:
        """Cannot mortgage property with houses."""
        prop = mock_game.property_manager.properties[1]
        prop.owner = 0
        prop.houses = 2
        action = MortgageProperty(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "houses" in error.lower()

    def test_execute_mortgage(self, mock_game: MockGame) -> None:
        """Execute should mortgage the property."""
        mock_game.property_manager.properties[1].owner = 0
        initial_money = mock_game.players[0].money
        action = MortgageProperty(player_id=0, property_id=1)
        action.execute(mock_game)

        prop = mock_game.property_manager.properties[1]
        assert prop.mortgaged
        assert mock_game.players[0].money == initial_money + 30  # Mediterranean mortgage


class TestUnmortgageProperty:
    """Tests for UnmortgageProperty action."""

    def test_validate_success(self, mock_game: MockGame) -> None:
        """Valid unmortgage should pass validation."""
        prop = mock_game.property_manager.properties[1]
        prop.owner = 0
        prop.mortgaged = True
        action = UnmortgageProperty(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert valid
        assert error == ""

    def test_validate_not_mortgaged(self, mock_game: MockGame) -> None:
        """Cannot unmortgage property that isn't mortgaged."""
        mock_game.property_manager.properties[1].owner = 0
        action = UnmortgageProperty(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "not mortgaged" in error.lower()

    def test_validate_insufficient_funds(self, mock_game: MockGame) -> None:
        """Cannot unmortgage without enough money."""
        prop = mock_game.property_manager.properties[1]
        prop.owner = 0
        prop.mortgaged = True
        mock_game.players[0].money = 10
        action = UnmortgageProperty(player_id=0, property_id=1)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "funds" in error.lower()

    def test_execute_unmortgage(self, mock_game: MockGame) -> None:
        """Execute should unmortgage the property."""
        prop = mock_game.property_manager.properties[1]
        prop.owner = 0
        prop.mortgaged = True
        initial_money = mock_game.players[0].money
        action = UnmortgageProperty(player_id=0, property_id=1)
        action.execute(mock_game)

        assert not prop.mortgaged
        # 110% of $30 = $33
        assert mock_game.players[0].money == initial_money - 33


class TestProposeTrade:
    """Tests for ProposeTrade action."""

    def test_validate_success(self, mock_game: MockGame) -> None:
        """Valid trade proposal should pass validation."""
        mock_game.property_manager.properties[1].owner = 0
        mock_game.property_manager.properties[3].owner = 1
        action = ProposeTrade(
            player_id=0,
            to_player=1,
            give_properties=[1],
            give_money=100,
            want_properties=[3],
            want_money=0,
        )
        valid, error = action.validate(mock_game)
        assert valid
        assert error == ""

    def test_validate_trade_with_self(self, mock_game: MockGame) -> None:
        """Cannot trade with yourself."""
        action = ProposeTrade(
            player_id=0,
            to_player=0,
            give_properties=[],
            give_money=100,
            want_properties=[],
            want_money=0,
        )
        valid, error = action.validate(mock_game)
        assert not valid
        assert "yourself" in error.lower()

    def test_validate_invalid_player(self, mock_game: MockGame) -> None:
        """Cannot trade with invalid player ID."""
        action = ProposeTrade(
            player_id=0,
            to_player=5,
            give_properties=[],
            give_money=100,
            want_properties=[],
            want_money=0,
        )
        valid, error = action.validate(mock_game)
        assert not valid
        assert "Invalid player" in error

    def test_validate_bankrupt_target(self, mock_game: MockGame) -> None:
        """Cannot trade with bankrupt player."""
        mock_game.players[1].bankrupt = True
        action = ProposeTrade(
            player_id=0,
            to_player=1,
            give_properties=[],
            give_money=100,
            want_properties=[],
            want_money=0,
        )
        valid, error = action.validate(mock_game)
        assert not valid
        assert "bankrupt" in error.lower()

    def test_validate_dont_own_property(self, mock_game: MockGame) -> None:
        """Cannot trade property you don't own."""
        mock_game.property_manager.properties[1].owner = 1
        action = ProposeTrade(
            player_id=0,
            to_player=1,
            give_properties=[1],
            give_money=0,
            want_properties=[],
            want_money=0,
        )
        valid, error = action.validate(mock_game)
        assert not valid
        assert "own" in error.lower()

    def test_execute_propose_trade(self, mock_game: MockGame) -> None:
        """Execute should create a pending trade."""
        mock_game.property_manager.properties[1].owner = 0
        mock_game.property_manager.properties[3].owner = 1
        action = ProposeTrade(
            player_id=0,
            to_player=1,
            give_properties=[1],
            give_money=100,
            want_properties=[3],
            want_money=0,
        )
        action.execute(mock_game)

        assert len(mock_game.pending_trades) == 1
        trade = mock_game.pending_trades[0]
        assert trade["from_player"] == 0
        assert trade["to_player"] == 1
        assert trade["give_properties"] == [1]
        assert trade["give_money"] == 100

    def test_to_dict(self) -> None:
        """Test serialization."""
        action = ProposeTrade(
            player_id=0,
            to_player=1,
            give_properties=[1, 3],
            give_money=100,
            want_properties=[5],
            want_money=50,
        )
        data = action.to_dict()
        assert data["type"] == "ProposeTrade"
        assert data["to_player"] == 1
        assert data["give_properties"] == [1, 3]
        assert data["give_money"] == 100


class TestAcceptTrade:
    """Tests for AcceptTrade action."""

    def test_validate_success(self, mock_game: MockGame) -> None:
        """Valid trade acceptance should pass validation."""
        mock_game.property_manager.properties[1].owner = 0
        mock_game.property_manager.properties[3].owner = 1
        mock_game.pending_trades[0] = {
            "from_player": 0,
            "to_player": 1,
            "give_properties": [1],
            "give_money": 100,
            "want_properties": [3],
            "want_money": 0,
        }
        mock_game.state.phase = "trade_response"
        action = AcceptTrade(player_id=1, trade_id=0)
        valid, error = action.validate(mock_game)
        assert valid
        assert error == ""

    def test_validate_trade_doesnt_exist(self, mock_game: MockGame) -> None:
        """Cannot accept non-existent trade."""
        action = AcceptTrade(player_id=1, trade_id=99)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "pending" in error.lower()

    def test_validate_trade_not_for_you(self, mock_game: MockGame) -> None:
        """Cannot accept trade not directed to you."""
        mock_game.pending_trades[0] = {
            "from_player": 0,
            "to_player": 1,
            "give_properties": [],
            "give_money": 100,
            "want_properties": [],
            "want_money": 0,
        }
        mock_game.state.phase = "trade_response"
        action = AcceptTrade(player_id=0, trade_id=0)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "another recipient" in error.lower()

    def test_execute_accept_trade(self, mock_game: MockGame) -> None:
        """Execute should transfer properties and money."""
        mock_game.property_manager.properties[1].owner = 0
        mock_game.property_manager.properties[3].owner = 1
        mock_game.pending_trades[0] = {
            "from_player": 0,
            "to_player": 1,
            "give_properties": [1],
            "give_money": 100,
            "want_properties": [3],
            "want_money": 0,
        }
        mock_game.state.phase = "trade_response"

        p0_money = mock_game.players[0].money
        p1_money = mock_game.players[1].money

        action = AcceptTrade(player_id=1, trade_id=0)
        action.execute(mock_game)

        # Properties swapped
        assert mock_game.property_manager.properties[1].owner == 1
        assert mock_game.property_manager.properties[3].owner == 0

        # Money transferred
        assert mock_game.players[0].money == p0_money - 100
        assert mock_game.players[1].money == p1_money + 100

        # Trade removed
        assert 0 not in mock_game.pending_trades


class TestRejectTrade:
    """Tests for RejectTrade action."""

    def test_validate_success(self, mock_game: MockGame) -> None:
        """Valid trade rejection should pass validation."""
        mock_game.pending_trades[0] = {
            "from_player": 0,
            "to_player": 1,
            "give_properties": [],
            "give_money": 100,
            "want_properties": [],
            "want_money": 0,
        }
        mock_game.state.phase = "trade_response"
        action = RejectTrade(player_id=1, trade_id=0)
        valid, error = action.validate(mock_game)
        assert valid
        assert error == ""

    def test_validate_trade_doesnt_exist(self, mock_game: MockGame) -> None:
        """Cannot reject non-existent trade."""
        action = RejectTrade(player_id=1, trade_id=99)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "pending" in error.lower()

    def test_execute_reject_trade(self, mock_game: MockGame) -> None:
        """Execute should remove the trade."""
        mock_game.pending_trades[0] = {
            "from_player": 0,
            "to_player": 1,
            "give_properties": [],
            "give_money": 100,
            "want_properties": [],
            "want_money": 0,
        }
        mock_game.state.phase = "trade_response"
        action = RejectTrade(player_id=1, trade_id=0)
        action.execute(mock_game)

        assert 0 not in mock_game.pending_trades


class TestPayJailFine:
    """Tests for PayJailFine action."""

    def test_validate_success(self, mock_game: MockGame) -> None:
        """Valid jail fine payment should pass validation."""
        mock_game.players[0].in_jail = True
        action = PayJailFine(player_id=0)
        valid, error = action.validate(mock_game)
        assert valid
        assert error == ""

    def test_validate_not_in_jail(self, mock_game: MockGame) -> None:
        """Cannot pay jail fine when not in jail."""
        action = PayJailFine(player_id=0)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "not in jail" in error.lower()

    def test_validate_insufficient_funds(self, mock_game: MockGame) -> None:
        """Cannot pay jail fine without enough money."""
        mock_game.players[0].in_jail = True
        mock_game.players[0].money = 10
        action = PayJailFine(player_id=0)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "funds" in error.lower()

    def test_execute_pay_fine(self, mock_game: MockGame) -> None:
        """Execute should pay fine and get out of jail."""
        mock_game.players[0].in_jail = True
        mock_game.players[0].jail_turns = 2
        initial_money = mock_game.players[0].money
        action = PayJailFine(player_id=0)
        action.execute(mock_game)

        player = mock_game.players[0]
        assert not player.in_jail
        assert player.jail_turns == 0
        assert player.money == initial_money - JAIL_FINE


class TestUseJailCard:
    """Tests for UseJailCard action."""

    def test_validate_success(self, mock_game: MockGame) -> None:
        """Valid jail card use should pass validation."""
        mock_game.players[0].in_jail = True
        mock_game.players[0].jail_cards = 1
        action = UseJailCard(player_id=0)
        valid, error = action.validate(mock_game)
        assert valid
        assert error == ""

    def test_validate_not_in_jail(self, mock_game: MockGame) -> None:
        """Cannot use jail card when not in jail."""
        mock_game.players[0].jail_cards = 1
        action = UseJailCard(player_id=0)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "not in jail" in error.lower()

    def test_validate_no_card(self, mock_game: MockGame) -> None:
        """Cannot use jail card if you don't have one."""
        mock_game.players[0].in_jail = True
        action = UseJailCard(player_id=0)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "card" in error.lower()

    def test_execute_use_card(self, mock_game: MockGame) -> None:
        """Execute should use card and get out of jail."""
        mock_game.players[0].in_jail = True
        mock_game.players[0].jail_cards = 1
        action = UseJailCard(player_id=0)
        action.execute(mock_game)

        player = mock_game.players[0]
        assert not player.in_jail
        assert player.jail_turns == 0
        assert player.jail_cards == 0


class TestDeclareBankruptcy:
    """Tests for DeclareBankruptcy action."""

    def test_validate_success(self, mock_game: MockGame) -> None:
        """Can always declare bankruptcy."""
        mock_game.state.obligations = [{"debtor": 0, "amount": 10000, "creditor": None}]
        action = DeclareBankruptcy(player_id=0)
        valid, error = action.validate(mock_game)
        assert valid
        assert error == ""

    def test_validate_already_bankrupt(self, mock_game: MockGame) -> None:
        """Cannot declare bankruptcy twice."""
        mock_game.players[0].bankrupt = True
        mock_game.state.obligations = [{"debtor": 0, "amount": 10000, "creditor": None}]
        action = DeclareBankruptcy(player_id=0)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "already bankrupt" in error.lower()

    def test_execute_bankruptcy(self, mock_game: MockGame) -> None:
        """Execute should mark player bankrupt and clear properties."""
        # Give player some properties
        mock_game.property_manager.properties[1].owner = 0
        mock_game.property_manager.properties[3].owner = 0
        mock_game.property_manager.properties[3].houses = 2

        mock_game.state.obligations = [{"debtor": 0, "amount": 10000, "creditor": None}]
        action = DeclareBankruptcy(player_id=0)
        action.execute(mock_game)

        player = mock_game.players[0]
        assert player.bankrupt

        # Properties should be cleared
        assert mock_game.property_manager.properties[1].owner is None
        assert mock_game.property_manager.properties[3].owner is None
        assert mock_game.property_manager.properties[3].houses == 0

    def test_execute_bankruptcy_ends_game(self, mock_game: MockGame) -> None:
        """Execute should end game when only one player left."""
        mock_game.players[0].bankrupt = False
        mock_game.players[1].bankrupt = False

        mock_game.state.obligations = [{"debtor": 0, "amount": 10000, "creditor": None}]
        action = DeclareBankruptcy(player_id=0)
        action.execute(mock_game)

        assert mock_game.game_over
        assert mock_game.winner == 1

    def test_execute_bankruptcy_returns_jail_cards(self, mock_game: MockGame) -> None:
        """Execute should return jail cards to deck."""
        mock_game.players[0].jail_cards = 2
        mock_game.state.jail_card_sources[0] = ["chance", "chest"]
        mock_game.state.obligations = [{"debtor": 0, "amount": 10000, "creditor": None}]
        action = DeclareBankruptcy(player_id=0)
        action.execute(mock_game)

        assert mock_game.players[0].jail_cards == 0


class TestEndTurn:
    """Tests for EndTurn action."""

    def test_validate_success(self, mock_game: MockGame) -> None:
        """Valid turn end should pass validation."""
        action = EndTurn(player_id=0)
        valid, error = action.validate(mock_game)
        assert valid
        assert error == ""

    def test_validate_not_your_turn(self, mock_game: MockGame) -> None:
        """Cannot end turn when it's not your turn."""
        action = EndTurn(player_id=1)
        valid, error = action.validate(mock_game)
        assert not valid
        assert "Not your turn" in error

    def test_execute_end_turn(self, mock_game: MockGame) -> None:
        """Execute should advance to next player."""
        mock_game.doubles_count = 2
        mock_game.last_roll = (3, 4)
        initial_turn = mock_game.turn_number

        action = EndTurn(player_id=0)
        action.execute(mock_game)

        assert mock_game.current_player == 1
        assert mock_game.turn_number == initial_turn + 1
        assert mock_game.doubles_count == 0
        assert mock_game.last_roll is None

    def test_execute_skip_bankrupt_players(self, mock_game_four_players: MockGame) -> None:
        """Execute should skip bankrupt players."""
        mock_game_four_players.players[1].bankrupt = True
        mock_game_four_players.players[2].bankrupt = True

        action = EndTurn(player_id=0)
        action.execute(mock_game_four_players)

        # Should skip players 1 and 2, go to 3
        assert mock_game_four_players.current_player == 3

    def test_execute_wrap_around(self, mock_game_four_players: MockGame) -> None:
        """Execute should wrap around to player 0."""
        mock_game_four_players.current_player = 3

        action = EndTurn(player_id=3)
        action.execute(mock_game_four_players)

        assert mock_game_four_players.current_player == 0
