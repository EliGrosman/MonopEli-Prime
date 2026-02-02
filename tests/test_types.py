"""
Comprehensive tests for the types module.

This module tests all enums, TypedDicts, and protocols defined in types.py,
including edge cases and serialization compatibility.
"""

import pytest
from monopoly_engine.types import (
    SpaceType,
    PropertyColor,
    ActionType,
    CardType,
    PropertyData,
    PlayerState,
    GameState,
    TradeOffer,
    CardData,
    SpaceData,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestSpaceType:
    """Test the SpaceType enum."""

    def test_all_space_types_exist(self, all_space_types: list[SpaceType]) -> None:
        """Verify all expected space types are defined."""
        assert len(all_space_types) == 10
        assert SpaceType.GO in all_space_types
        assert SpaceType.PROPERTY in all_space_types
        assert SpaceType.RAILROAD in all_space_types
        assert SpaceType.UTILITY in all_space_types
        assert SpaceType.CHANCE in all_space_types
        assert SpaceType.COMMUNITY_CHEST in all_space_types
        assert SpaceType.TAX in all_space_types
        assert SpaceType.JAIL in all_space_types
        assert SpaceType.GO_TO_JAIL in all_space_types
        assert SpaceType.FREE_PARKING in all_space_types

    def test_space_type_uniqueness(self, all_space_types: list[SpaceType]) -> None:
        """Verify each space type has a unique value."""
        values = [st.value for st in all_space_types]
        assert len(values) == len(set(values))

    def test_space_type_names(self) -> None:
        """Verify space type names are as expected."""
        assert SpaceType.GO.name == "GO"
        assert SpaceType.PROPERTY.name == "PROPERTY"
        assert SpaceType.RAILROAD.name == "RAILROAD"
        assert SpaceType.UTILITY.name == "UTILITY"
        assert SpaceType.CHANCE.name == "CHANCE"
        assert SpaceType.COMMUNITY_CHEST.name == "COMMUNITY_CHEST"
        assert SpaceType.TAX.name == "TAX"
        assert SpaceType.JAIL.name == "JAIL"
        assert SpaceType.GO_TO_JAIL.name == "GO_TO_JAIL"
        assert SpaceType.FREE_PARKING.name == "FREE_PARKING"

    def test_space_type_equality(self) -> None:
        """Verify space type equality comparisons work correctly."""
        assert SpaceType.GO == SpaceType.GO
        assert SpaceType.GO != SpaceType.PROPERTY
        assert SpaceType.RAILROAD != SpaceType.UTILITY


class TestPropertyColor:
    """Test the PropertyColor enum."""

    def test_all_property_colors_exist(self, all_property_colors: list[PropertyColor]) -> None:
        """Verify all expected property colors are defined."""
        assert len(all_property_colors) == 10
        assert PropertyColor.BROWN in all_property_colors
        assert PropertyColor.LIGHT_BLUE in all_property_colors
        assert PropertyColor.MAGENTA in all_property_colors
        assert PropertyColor.ORANGE in all_property_colors
        assert PropertyColor.RED in all_property_colors
        assert PropertyColor.YELLOW in all_property_colors
        assert PropertyColor.GREEN in all_property_colors
        assert PropertyColor.BLUE in all_property_colors
        assert PropertyColor.RAILROAD in all_property_colors
        assert PropertyColor.UTILITY in all_property_colors

    def test_property_color_uniqueness(self, all_property_colors: list[PropertyColor]) -> None:
        """Verify each property color has a unique value."""
        values = [pc.value for pc in all_property_colors]
        assert len(values) == len(set(values))

    def test_property_color_names(self) -> None:
        """Verify property color names are as expected."""
        assert PropertyColor.BROWN.name == "BROWN"
        assert PropertyColor.LIGHT_BLUE.name == "LIGHT_BLUE"
        assert PropertyColor.MAGENTA.name == "MAGENTA"
        assert PropertyColor.ORANGE.name == "ORANGE"
        assert PropertyColor.RED.name == "RED"
        assert PropertyColor.YELLOW.name == "YELLOW"
        assert PropertyColor.GREEN.name == "GREEN"
        assert PropertyColor.BLUE.name == "BLUE"
        assert PropertyColor.RAILROAD.name == "RAILROAD"
        assert PropertyColor.UTILITY.name == "UTILITY"

    def test_property_color_equality(self) -> None:
        """Verify property color equality comparisons work correctly."""
        assert PropertyColor.BROWN == PropertyColor.BROWN
        assert PropertyColor.BROWN != PropertyColor.BLUE
        assert PropertyColor.RAILROAD != PropertyColor.UTILITY


class TestActionType:
    """Test the ActionType enum."""

    def test_all_action_types_exist(self, all_action_types: list[ActionType]) -> None:
        """Verify all expected action types are defined."""
        assert len(all_action_types) == 15
        assert ActionType.ROLL_DICE in all_action_types
        assert ActionType.BUY_PROPERTY in all_action_types
        assert ActionType.BUILD_HOUSE in all_action_types
        assert ActionType.BUILD_HOTEL in all_action_types
        assert ActionType.SELL_HOUSE in all_action_types
        assert ActionType.SELL_HOTEL in all_action_types
        assert ActionType.MORTGAGE_PROPERTY in all_action_types
        assert ActionType.UNMORTGAGE_PROPERTY in all_action_types
        assert ActionType.PROPOSE_TRADE in all_action_types
        assert ActionType.ACCEPT_TRADE in all_action_types
        assert ActionType.REJECT_TRADE in all_action_types
        assert ActionType.PAY_JAIL_FINE in all_action_types
        assert ActionType.USE_JAIL_CARD in all_action_types
        assert ActionType.DECLARE_BANKRUPTCY in all_action_types
        assert ActionType.END_TURN in all_action_types

    def test_action_type_uniqueness(self, all_action_types: list[ActionType]) -> None:
        """Verify each action type has a unique value."""
        values = [at.value for at in all_action_types]
        assert len(values) == len(set(values))

    def test_action_type_names(self) -> None:
        """Verify action type names are as expected."""
        assert ActionType.ROLL_DICE.name == "ROLL_DICE"
        assert ActionType.BUY_PROPERTY.name == "BUY_PROPERTY"
        assert ActionType.BUILD_HOUSE.name == "BUILD_HOUSE"
        assert ActionType.MORTGAGE_PROPERTY.name == "MORTGAGE_PROPERTY"
        assert ActionType.END_TURN.name == "END_TURN"


class TestCardType:
    """Test the CardType enum."""

    def test_all_card_types_exist(self, all_card_types: list[CardType]) -> None:
        """Verify all expected card types are defined."""
        assert len(all_card_types) == 5
        assert CardType.MOVE in all_card_types
        assert CardType.MONEY in all_card_types
        assert CardType.GET_OUT_OF_JAIL in all_card_types
        assert CardType.PAY_PER_HOUSE in all_card_types
        assert CardType.GO_TO_JAIL in all_card_types

    def test_card_type_uniqueness(self, all_card_types: list[CardType]) -> None:
        """Verify each card type has a unique value."""
        values = [ct.value for ct in all_card_types]
        assert len(values) == len(set(values))

    def test_card_type_names(self) -> None:
        """Verify card type names are as expected."""
        assert CardType.MOVE.name == "MOVE"
        assert CardType.MONEY.name == "MONEY"
        assert CardType.GET_OUT_OF_JAIL.name == "GET_OUT_OF_JAIL"
        assert CardType.PAY_PER_HOUSE.name == "PAY_PER_HOUSE"
        assert CardType.GO_TO_JAIL.name == "GO_TO_JAIL"


# =============================================================================
# TypedDict Tests
# =============================================================================


class TestPropertyData:
    """Test the PropertyData TypedDict."""

    def test_property_data_creation(self, sample_property_data: PropertyData) -> None:
        """Verify PropertyData can be created with all required fields."""
        assert sample_property_data["position"] == 1
        assert sample_property_data["name"] == "Mediterranean Avenue"
        assert sample_property_data["color"] == PropertyColor.BROWN
        assert sample_property_data["cost"] == 60
        assert sample_property_data["rent"] == [2, 10, 30, 90, 160, 250]
        assert sample_property_data["house_cost"] == 50
        assert sample_property_data["mortgage_value"] == 30

    def test_property_data_all_fields_required(self) -> None:
        """Verify all fields are present in PropertyData."""
        # This tests that we can create a complete PropertyData
        data: PropertyData = {
            "position": 1,
            "name": "Test",
            "color": PropertyColor.BROWN,
            "cost": 100,
            "rent": [5, 10, 15, 20, 25, 30],
            "house_cost": 50,
            "mortgage_value": 50,
        }
        assert data["position"] == 1

    def test_property_data_rent_list_length(self) -> None:
        """Verify rent list can have different lengths for different property types."""
        # Street property with 6 rent values
        street: PropertyData = {
            "position": 1,
            "name": "Street",
            "color": PropertyColor.BROWN,
            "cost": 60,
            "rent": [2, 10, 30, 90, 160, 250],
            "house_cost": 50,
            "mortgage_value": 30,
        }
        assert len(street["rent"]) == 6

        # Railroad with 4 rent values (conceptually)
        railroad: PropertyData = {
            "position": 5,
            "name": "Railroad",
            "color": PropertyColor.RAILROAD,
            "cost": 200,
            "rent": [25, 50, 100, 200],
            "house_cost": 0,
            "mortgage_value": 100,
        }
        assert len(railroad["rent"]) == 4


class TestPlayerState:
    """Test the PlayerState TypedDict."""

    def test_player_state_creation(self, sample_player_state: PlayerState) -> None:
        """Verify PlayerState can be created with all required fields."""
        assert sample_player_state["id"] == 0
        assert sample_player_state["name"] == "Player 1"
        assert sample_player_state["money"] == 1500
        assert sample_player_state["position"] == 0
        assert sample_player_state["properties"] == []
        assert sample_player_state["houses"] == {}
        assert sample_player_state["jail_cards"] == 0
        assert sample_player_state["in_jail"] is False
        assert sample_player_state["jail_turns"] == 0
        assert sample_player_state["bankrupt"] is False

    def test_player_state_with_properties(self) -> None:
        """Verify PlayerState can track owned properties and houses."""
        state: PlayerState = {
            "id": 0,
            "name": "Test Player",
            "money": 1000,
            "position": 5,
            "properties": [1, 3, 6],
            "houses": {1: 2, 3: 3},
            "jail_cards": 1,
            "in_jail": False,
            "jail_turns": 0,
            "bankrupt": False,
        }
        assert len(state["properties"]) == 3
        assert state["houses"][1] == 2
        assert state["houses"][3] == 3
        assert state["jail_cards"] == 1

    def test_player_state_in_jail(self) -> None:
        """Verify PlayerState can represent a player in jail."""
        state: PlayerState = {
            "id": 0,
            "name": "Jailed Player",
            "money": 1500,
            "position": 10,
            "properties": [],
            "houses": {},
            "jail_cards": 0,
            "in_jail": True,
            "jail_turns": 2,
            "bankrupt": False,
        }
        assert state["in_jail"] is True
        assert state["jail_turns"] == 2

    def test_player_state_bankrupt(self) -> None:
        """Verify PlayerState can represent a bankrupt player."""
        state: PlayerState = {
            "id": 0,
            "name": "Bankrupt Player",
            "money": 0,
            "position": 15,
            "properties": [],
            "houses": {},
            "jail_cards": 0,
            "in_jail": False,
            "jail_turns": 0,
            "bankrupt": True,
        }
        assert state["bankrupt"] is True
        assert state["money"] == 0


class TestGameState:
    """Test the GameState TypedDict."""

    def test_game_state_creation(self, sample_game_state: GameState) -> None:
        """Verify GameState can be created with all required fields."""
        assert len(sample_game_state["players"]) == 2
        assert sample_game_state["current_player"] == 0
        assert sample_game_state["turn_number"] == 0
        assert sample_game_state["houses_remaining"] == 32
        assert sample_game_state["hotels_remaining"] == 12
        assert sample_game_state["chance_deck"] == []
        assert sample_game_state["community_chest_deck"] == []
        assert sample_game_state["last_roll"] is None
        assert sample_game_state["doubles_count"] == 0
        assert sample_game_state["game_over"] is False
        assert sample_game_state["winner"] is None

    def test_game_state_with_last_roll(self) -> None:
        """Verify GameState can track the last dice roll."""
        state: GameState = {
            "players": [],
            "current_player": 0,
            "turn_number": 5,
            "houses_remaining": 32,
            "hotels_remaining": 12,
            "chance_deck": [],
            "community_chest_deck": [],
            "last_roll": (3, 4),
            "doubles_count": 0,
            "game_over": False,
            "winner": None,
        }
        assert state["last_roll"] == (3, 4)
        assert state["turn_number"] == 5

    def test_game_state_with_doubles(self) -> None:
        """Verify GameState can track consecutive doubles."""
        state: GameState = {
            "players": [],
            "current_player": 0,
            "turn_number": 10,
            "houses_remaining": 32,
            "hotels_remaining": 12,
            "chance_deck": [],
            "community_chest_deck": [],
            "last_roll": (5, 5),
            "doubles_count": 2,
            "game_over": False,
            "winner": None,
        }
        assert state["last_roll"] == (5, 5)
        assert state["doubles_count"] == 2

    def test_game_state_game_over(self) -> None:
        """Verify GameState can represent a finished game."""
        state: GameState = {
            "players": [],
            "current_player": 0,
            "turn_number": 100,
            "houses_remaining": 32,
            "hotels_remaining": 12,
            "chance_deck": [],
            "community_chest_deck": [],
            "last_roll": (2, 3),
            "doubles_count": 0,
            "game_over": True,
            "winner": 1,
        }
        assert state["game_over"] is True
        assert state["winner"] == 1

    def test_game_state_house_depletion(self) -> None:
        """Verify GameState can track house/hotel depletion."""
        state: GameState = {
            "players": [],
            "current_player": 0,
            "turn_number": 50,
            "houses_remaining": 5,
            "hotels_remaining": 2,
            "chance_deck": [],
            "community_chest_deck": [],
            "last_roll": None,
            "doubles_count": 0,
            "game_over": False,
            "winner": None,
        }
        assert state["houses_remaining"] == 5
        assert state["hotels_remaining"] == 2


class TestTradeOffer:
    """Test the TradeOffer TypedDict."""

    def test_trade_offer_creation(self, sample_trade_offer: TradeOffer) -> None:
        """Verify TradeOffer can be created with all required fields."""
        assert sample_trade_offer["from_player"] == 0
        assert sample_trade_offer["to_player"] == 1
        assert sample_trade_offer["from_properties"] == [1]
        assert sample_trade_offer["to_properties"] == [3]
        assert sample_trade_offer["from_money"] == 100
        assert sample_trade_offer["to_money"] == 50

    def test_trade_offer_properties_only(self) -> None:
        """Verify TradeOffer can represent a property-only trade."""
        trade: TradeOffer = {
            "from_player": 0,
            "to_player": 1,
            "from_properties": [1, 3],
            "to_properties": [6, 8, 9],
            "from_money": 0,
            "to_money": 0,
        }
        assert len(trade["from_properties"]) == 2
        assert len(trade["to_properties"]) == 3
        assert trade["from_money"] == 0
        assert trade["to_money"] == 0

    def test_trade_offer_money_only(self) -> None:
        """Verify TradeOffer can represent a money-only trade."""
        trade: TradeOffer = {
            "from_player": 0,
            "to_player": 1,
            "from_properties": [],
            "to_properties": [],
            "from_money": 500,
            "to_money": 0,
        }
        assert len(trade["from_properties"]) == 0
        assert len(trade["to_properties"]) == 0
        assert trade["from_money"] == 500

    def test_trade_offer_complex(self) -> None:
        """Verify TradeOffer can represent complex multi-asset trades."""
        trade: TradeOffer = {
            "from_player": 0,
            "to_player": 1,
            "from_properties": [1, 3],
            "to_properties": [37, 39],
            "from_money": 1000,
            "to_money": 200,
        }
        assert trade["from_properties"] == [1, 3]
        assert trade["to_properties"] == [37, 39]
        assert trade["from_money"] == 1000
        assert trade["to_money"] == 200


class TestCardData:
    """Test the CardData TypedDict."""

    def test_card_data_creation(self, sample_card_data: CardData) -> None:
        """Verify CardData can be created with all required fields."""
        assert sample_card_data["id"] == 1
        assert sample_card_data["deck"] == "chance"
        assert sample_card_data["type"] == CardType.MONEY
        assert sample_card_data["description"] == "Bank pays you dividend of $50"
        assert sample_card_data["value"] == 50
        assert sample_card_data["keep"] is False

    def test_card_data_get_out_of_jail(self) -> None:
        """Verify CardData can represent a Get Out of Jail Free card."""
        card: CardData = {
            "id": 5,
            "deck": "community_chest",
            "type": CardType.GET_OUT_OF_JAIL,
            "description": "Get Out of Jail Free",
            "value": 0,
            "keep": True,
        }
        assert card["type"] == CardType.GET_OUT_OF_JAIL
        assert card["keep"] is True

    def test_card_data_move_card(self) -> None:
        """Verify CardData can represent a movement card."""
        card: CardData = {
            "id": 10,
            "deck": "chance",
            "type": CardType.MOVE,
            "description": "Advance to Go",
            "value": 0,
            "keep": False,
        }
        assert card["type"] == CardType.MOVE
        assert card["value"] == 0

    def test_card_data_pay_per_house(self) -> None:
        """Verify CardData can represent a pay-per-house card."""
        card: CardData = {
            "id": 15,
            "deck": "chance",
            "type": CardType.PAY_PER_HOUSE,
            "description": "Make general repairs: $25 per house, $100 per hotel",
            "value": 25,
            "keep": False,
        }
        assert card["type"] == CardType.PAY_PER_HOUSE
        assert card["value"] == 25


class TestSpaceData:
    """Test the SpaceData TypedDict."""

    def test_space_data_creation(self, sample_space_data: SpaceData) -> None:
        """Verify SpaceData can be created with all required fields."""
        assert sample_space_data["position"] == 4
        assert sample_space_data["name"] == "Income Tax"
        assert sample_space_data["type"] == SpaceType.TAX
        assert sample_space_data["value"] == 200

    def test_space_data_special_spaces(self) -> None:
        """Verify SpaceData can represent special spaces with no value."""
        go: SpaceData = {
            "position": 0,
            "name": "GO",
            "type": SpaceType.GO,
            "value": 0,
        }
        assert go["type"] == SpaceType.GO
        assert go["value"] == 0

        jail: SpaceData = {
            "position": 10,
            "name": "Just Visiting",
            "type": SpaceType.JAIL,
            "value": 0,
        }
        assert jail["type"] == SpaceType.JAIL
        assert jail["value"] == 0

    def test_space_data_tax_spaces(self) -> None:
        """Verify SpaceData correctly represents both tax spaces."""
        income_tax: SpaceData = {
            "position": 4,
            "name": "Income Tax",
            "type": SpaceType.TAX,
            "value": 200,
        }
        assert income_tax["value"] == 200

        luxury_tax: SpaceData = {
            "position": 38,
            "name": "Luxury Tax",
            "type": SpaceType.TAX,
            "value": 100,
        }
        assert luxury_tax["value"] == 100


# =============================================================================
# Protocol Tests
# =============================================================================


class TestSerializableProtocol:
    """Test that objects can implement the Serializable protocol."""

    def test_serializable_methods_exist(self) -> None:
        """Verify Serializable protocol defines expected methods."""
        from monopoly_engine.types import Serializable

        # Protocol defines abstract methods, so we can't instantiate it
        # But we can check that it has the expected attributes
        assert hasattr(Serializable, "to_dict")
        assert hasattr(Serializable, "from_dict")


class TestValidateableProtocol:
    """Test that objects can implement the Validateable protocol."""

    def test_validateable_method_exists(self) -> None:
        """Verify Validateable protocol defines expected method."""
        from monopoly_engine.types import Validateable

        assert hasattr(Validateable, "validate")


class TestExecutableProtocol:
    """Test that objects can implement the Executable protocol."""

    def test_executable_method_exists(self) -> None:
        """Verify Executable protocol defines expected method."""
        from monopoly_engine.types import Executable

        assert hasattr(Executable, "execute")


# =============================================================================
# Integration Tests
# =============================================================================


class TestTypeCompatibility:
    """Test that types work together correctly."""

    def test_game_state_with_all_player_states(self) -> None:
        """Verify GameState can hold multiple complete PlayerStates."""
        players: list[PlayerState] = [
            {
                "id": i,
                "name": f"Player {i + 1}",
                "money": 1500,
                "position": 0,
                "properties": [],
                "houses": {},
                "jail_cards": 0,
                "in_jail": False,
                "jail_turns": 0,
                "bankrupt": False,
            }
            for i in range(4)
        ]

        game: GameState = {
            "players": players,
            "current_player": 0,
            "turn_number": 0,
            "houses_remaining": 32,
            "hotels_remaining": 12,
            "chance_deck": list(range(16)),
            "community_chest_deck": list(range(16)),
            "last_roll": None,
            "doubles_count": 0,
            "game_over": False,
            "winner": None,
        }

        assert len(game["players"]) == 4
        assert all(p["money"] == 1500 for p in game["players"])

    def test_enum_in_typed_dict(self) -> None:
        """Verify enums can be used in TypedDict fields."""
        card: CardData = {
            "id": 1,
            "deck": "chance",
            "type": CardType.MONEY,
            "description": "Test",
            "value": 100,
            "keep": False,
        }

        # Verify we can compare enum values
        assert card["type"] == CardType.MONEY
        assert card["type"] != CardType.MOVE

        # Verify enum name access
        assert card["type"].name == "MONEY"
