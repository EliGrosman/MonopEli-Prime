"""Actions for the Monopoly game engine.

This module defines all possible player actions and their validation/execution logic.
Each action validates against game state before execution.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .types import JAIL_FINE

if TYPE_CHECKING:
    from .game import MonopolyGame


@dataclass
class Action(ABC):
    """Base class for all player actions.

    Each action must implement validate() and execute() methods.
    Validation checks if the action is legal in the current game state.
    Execution assumes validation has passed and modifies game state.
    """

    player_id: int

    @abstractmethod
    def validate(self, game: "MonopolyGame") -> tuple[bool, str]:
        """Validate action against game state.

        Args:
            game: The current game state

        Returns:
            Tuple of (is_valid, error_message)
            If valid, error_message is empty string
        """
        pass

    @abstractmethod
    def execute(self, game: "MonopolyGame") -> None:
        """Execute the action, modifying game state.

        This method assumes validate() has already passed.

        Args:
            game: The current game state to modify
        """
        pass

    def to_dict(self) -> dict[str, int | str | list[int]]:
        """Serialize action to JSON-compatible dict."""
        return {
            "type": self.__class__.__name__,
            "player_id": self.player_id,
        }


@dataclass
class RollDice(Action):
    """Roll dice to move around the board.

    This is the primary action that advances the game turn.
    """

    def validate(self, game: "MonopolyGame") -> tuple[bool, str]:
        """Validate dice roll action."""
        phase_valid, phase_error = game.validate_phase(self)
        if not phase_valid:
            return False, phase_error
        if game.current_player != self.player_id:
            return False, "Not your turn"

        player = game.players[self.player_id]
        if player.bankrupt:
            return False, "Player is bankrupt"

        return True, ""

    def execute(self, game: "MonopolyGame") -> None:
        """Roll dice and handle movement."""
        # Dice rolling logic will be implemented in game.py
        # This action triggers the game's dice roll handler
        game._handle_dice_roll(self.player_id)


@dataclass
class BuyProperty(Action):
    """Purchase an unowned property the player is standing on."""

    property_id: int

    def validate(self, game: "MonopolyGame") -> tuple[bool, str]:
        """Validate property purchase."""
        phase_valid, phase_error = game.validate_phase(self)
        if not phase_valid:
            return False, phase_error
        if game.current_player != self.player_id:
            return False, "Not your turn"

        player = game.players[self.player_id]
        if player.bankrupt:
            return False, "Player is bankrupt"

        if player.position != self.property_id:
            return False, "Not on this property"

        from .rules import can_buy_property

        can_buy, reason = can_buy_property(player, game.property_manager, self.property_id)
        if not can_buy:
            return False, reason

        return True, ""

    def execute(self, game: "MonopolyGame") -> None:
        """Purchase the property."""
        from .rules import get_property_cost

        player = game.players[self.player_id]
        prop = game.property_manager.properties[self.property_id]
        cost = get_property_cost(self.property_id)

        player.remove_money(cost)
        prop.owner = self.player_id

    def to_dict(self) -> dict[str, int | str | list[int]]:
        """Serialize with property_id."""
        data = super().to_dict()
        data["property_id"] = self.property_id
        return data


@dataclass
class BuildHouse(Action):
    """Build a house on a property in a monopoly."""

    property_id: int

    def validate(self, game: "MonopolyGame") -> tuple[bool, str]:
        """Validate house building."""
        phase_valid, phase_error = game.validate_phase(self)
        if not phase_valid:
            return False, phase_error
        if game.current_player != self.player_id:
            return False, "Not your turn"

        player = game.players[self.player_id]
        if player.bankrupt:
            return False, "Player is bankrupt"

        from .rules import can_build_house

        can_build, reason = can_build_house(
            player,
            game.property_manager,
            self.property_id,
            game.houses_remaining,
            game.hotels_remaining,
        )
        if not can_build:
            return False, reason

        return True, ""

    def execute(self, game: "MonopolyGame") -> None:
        """Build a house on the property."""
        from .rules import get_building_cost

        player = game.players[self.player_id]
        prop = game.property_manager.properties[self.property_id]
        cost = get_building_cost(self.property_id)

        player.remove_money(cost)

        # Check if building a hotel (4 houses -> 5)
        if prop.houses == 4:
            # Return 4 houses to the bank, take 1 hotel
            game.houses_remaining += 4
            game.hotels_remaining -= 1
            prop.houses = 5
        else:
            # Take 1 house from the bank
            game.houses_remaining -= 1
            prop.houses += 1

    def to_dict(self) -> dict[str, int | str | list[int]]:
        """Serialize with property_id."""
        data = super().to_dict()
        data["property_id"] = self.property_id
        return data


@dataclass
class BuildHotel(Action):
    """Build a hotel on a property (shortcut for BuildHouse when at 4 houses)."""

    property_id: int

    def validate(self, game: "MonopolyGame") -> tuple[bool, str]:
        """Validate hotel building."""
        phase_valid, phase_error = game.validate_phase(self)
        if not phase_valid:
            return False, phase_error
        if game.current_player != self.player_id:
            return False, "Not your turn"

        player = game.players[self.player_id]
        if player.bankrupt:
            return False, "Player is bankrupt"

        prop = game.property_manager.properties.get(self.property_id)
        if prop is None:
            return False, "Invalid property"

        if prop.houses != 4:
            return False, "Must have exactly 4 houses to build a hotel"

        from .rules import can_build_house

        can_build, reason = can_build_house(
            player,
            game.property_manager,
            self.property_id,
            game.houses_remaining,
            game.hotels_remaining,
        )
        if not can_build:
            return False, reason

        return True, ""

    def execute(self, game: "MonopolyGame") -> None:
        """Build a hotel on the property."""
        from .rules import get_building_cost

        player = game.players[self.player_id]
        prop = game.property_manager.properties[self.property_id]
        cost = get_building_cost(self.property_id)

        player.remove_money(cost)

        # Return 4 houses, take 1 hotel
        game.houses_remaining += 4
        game.hotels_remaining -= 1
        prop.houses = 5

    def to_dict(self) -> dict[str, int | str | list[int]]:
        """Serialize with property_id."""
        data = super().to_dict()
        data["property_id"] = self.property_id
        return data


@dataclass
class SellHouse(Action):
    """Sell a house from a property back to the bank."""

    property_id: int

    def validate(self, game: "MonopolyGame") -> tuple[bool, str]:
        """Validate house selling."""
        phase_valid, phase_error = game.validate_phase(self)
        if not phase_valid:
            return False, phase_error
        player = game.players[self.player_id]
        if player.bankrupt:
            return False, "Player is bankrupt"

        from .rules import can_sell_house

        can_sell, reason = can_sell_house(player, game.property_manager, self.property_id)
        if not can_sell:
            return False, reason

        return True, ""

    def execute(self, game: "MonopolyGame") -> None:
        """Sell a house from the property."""
        from .rules import get_house_sale_value

        player = game.players[self.player_id]
        prop = game.property_manager.properties[self.property_id]
        sale_value = get_house_sale_value(self.property_id)

        player.add_money(sale_value)

        # Check if selling a hotel (5 -> 4)
        if prop.houses == 5:
            # Return hotel, take 4 houses
            game.hotels_remaining += 1
            # Only take houses if available
            houses_to_take = min(4, game.houses_remaining)
            if houses_to_take < 4:
                # Not enough houses - property goes to 4 minus shortfall
                prop.houses = houses_to_take
                game.houses_remaining = 0
            else:
                game.houses_remaining -= 4
                prop.houses = 4
        else:
            # Return 1 house
            game.houses_remaining += 1
            prop.houses -= 1

    def to_dict(self) -> dict[str, int | str | list[int]]:
        """Serialize with property_id."""
        data = super().to_dict()
        data["property_id"] = self.property_id
        return data


@dataclass
class SellHotel(Action):
    """Sell a hotel from a property (shortcut for SellHouse when at hotel)."""

    property_id: int

    def validate(self, game: "MonopolyGame") -> tuple[bool, str]:
        """Validate hotel selling."""
        phase_valid, phase_error = game.validate_phase(self)
        if not phase_valid:
            return False, phase_error
        player = game.players[self.player_id]
        if player.bankrupt:
            return False, "Player is bankrupt"

        prop = game.property_manager.properties.get(self.property_id)
        if prop is None:
            return False, "Invalid property"

        if prop.houses != 5:
            return False, "Property does not have a hotel"

        from .rules import can_sell_house

        can_sell, reason = can_sell_house(player, game.property_manager, self.property_id)
        if not can_sell:
            return False, reason

        return True, ""

    def execute(self, game: "MonopolyGame") -> None:
        """Sell the hotel from the property."""
        from .rules import get_house_sale_value

        player = game.players[self.player_id]
        prop = game.property_manager.properties[self.property_id]
        sale_value = get_house_sale_value(self.property_id)

        player.add_money(sale_value)

        # Return hotel, take 4 houses
        game.hotels_remaining += 1
        houses_to_take = min(4, game.houses_remaining)
        if houses_to_take < 4:
            # Not enough houses - property goes to fewer houses
            prop.houses = houses_to_take
            game.houses_remaining = 0
        else:
            game.houses_remaining -= 4
            prop.houses = 4

    def to_dict(self) -> dict[str, int | str | list[int]]:
        """Serialize with property_id."""
        data = super().to_dict()
        data["property_id"] = self.property_id
        return data


@dataclass
class MortgageProperty(Action):
    """Mortgage a property to receive cash from the bank."""

    property_id: int

    def validate(self, game: "MonopolyGame") -> tuple[bool, str]:
        """Validate property mortgaging."""
        phase_valid, phase_error = game.validate_phase(self)
        if not phase_valid:
            return False, phase_error
        player = game.players[self.player_id]
        if player.bankrupt:
            return False, "Player is bankrupt"

        from .rules import can_mortgage_property

        can_mortgage, reason = can_mortgage_property(
            player, game.property_manager, self.property_id
        )
        if not can_mortgage:
            return False, reason

        return True, ""

    def execute(self, game: "MonopolyGame") -> None:
        """Mortgage the property."""
        from .rules import get_mortgage_value

        player = game.players[self.player_id]
        prop = game.property_manager.properties[self.property_id]
        mortgage_value = get_mortgage_value(self.property_id)

        player.add_money(mortgage_value)
        prop.mortgaged = True

    def to_dict(self) -> dict[str, int | str | list[int]]:
        """Serialize with property_id."""
        data = super().to_dict()
        data["property_id"] = self.property_id
        return data


@dataclass
class UnmortgageProperty(Action):
    """Unmortgage a property by paying 110% of mortgage value."""

    property_id: int

    def validate(self, game: "MonopolyGame") -> tuple[bool, str]:
        """Validate property unmortgaging."""
        phase_valid, phase_error = game.validate_phase(self)
        if not phase_valid:
            return False, phase_error
        if game.current_player != self.player_id:
            return False, "Not your turn"

        player = game.players[self.player_id]
        if player.bankrupt:
            return False, "Player is bankrupt"

        from .rules import can_unmortgage_property

        can_unmortgage, reason = can_unmortgage_property(
            player, game.property_manager, self.property_id
        )
        if not can_unmortgage:
            return False, reason

        return True, ""

    def execute(self, game: "MonopolyGame") -> None:
        """Unmortgage the property."""
        from .rules import get_unmortgage_cost

        player = game.players[self.player_id]
        prop = game.property_manager.properties[self.property_id]
        unmortgage_cost = get_unmortgage_cost(self.property_id)

        player.remove_money(unmortgage_cost)
        prop.mortgaged = False

    def to_dict(self) -> dict[str, int | str | list[int]]:
        """Serialize with property_id."""
        data = super().to_dict()
        data["property_id"] = self.property_id
        return data


@dataclass
class ProposeTrade(Action):
    """Propose a trade to another player.

    Trades can include properties and money going both directions.
    """

    to_player: int
    give_properties: list[int] = field(default_factory=list)
    give_money: int = 0
    want_properties: list[int] = field(default_factory=list)
    want_money: int = 0

    def validate(self, game: "MonopolyGame") -> tuple[bool, str]:
        """Validate trade proposal."""
        phase_valid, phase_error = game.validate_phase(self)
        if not phase_valid:
            return False, phase_error
        if game.current_player != self.player_id:
            return False, "Not your turn"

        from_player = game.players[self.player_id]
        if from_player.bankrupt:
            return False, "Player is bankrupt"

        if self.to_player == self.player_id:
            return False, "Cannot trade with yourself"

        if self.to_player < 0 or self.to_player >= len(game.players):
            return False, "Invalid player ID"

        to_player = game.players[self.to_player]
        if to_player.bankrupt:
            return False, "Cannot trade with bankrupt player"

        # Validate the trade using rules
        from .rules import validate_trade

        is_valid, reason = validate_trade(
            from_player,
            to_player,
            game.property_manager,
            self.give_properties,
            self.give_money,
            self.want_properties,
            self.want_money,
        )
        if not is_valid:
            return False, reason

        return True, ""

    def execute(self, game: "MonopolyGame") -> None:
        """Create a pending trade offer.

        The trade is stored in game state for the other player to accept/reject.
        """
        # Store the pending trade in game state
        # This will be implemented when game.py is created
        game._add_pending_trade(
            from_player=self.player_id,
            to_player=self.to_player,
            give_properties=self.give_properties,
            give_money=self.give_money,
            want_properties=self.want_properties,
            want_money=self.want_money,
        )

    def to_dict(self) -> dict[str, int | str | list[int]]:
        """Serialize with all trade parameters."""
        data = super().to_dict()
        data["to_player"] = self.to_player
        data["give_properties"] = self.give_properties
        data["give_money"] = self.give_money
        data["want_properties"] = self.want_properties
        data["want_money"] = self.want_money
        return data


@dataclass
class AcceptTrade(Action):
    """Accept a pending trade offer."""

    trade_id: int

    def validate(self, game: "MonopolyGame") -> tuple[bool, str]:
        """Validate trade acceptance."""
        phase_valid, phase_error = game.validate_phase(self)
        if not phase_valid:
            return False, phase_error
        player = game.players[self.player_id]
        if player.bankrupt:
            return False, "Player is bankrupt"

        # Check if trade exists and is directed to this player
        trade = game._get_pending_trade(self.trade_id)
        if trade is None:
            return False, "Trade does not exist"

        if trade["to_player"] != self.player_id:
            return False, "Trade is not for you"

        # Re-validate the trade (money/properties may have changed)
        from .rules import validate_trade

        from_player = game.players[trade["from_player"]]
        to_player = game.players[trade["to_player"]]

        is_valid, reason = validate_trade(
            from_player,
            to_player,
            game.property_manager,
            trade["give_properties"],
            trade["give_money"],
            trade["want_properties"],
            trade["want_money"],
        )
        if not is_valid:
            return False, f"Trade no longer valid: {reason}"

        return True, ""

    def execute(self, game: "MonopolyGame") -> None:
        """Execute the trade, transferring properties and money."""
        trade = game._get_pending_trade(self.trade_id)
        if trade is None:
            return

        from_player = game.players[trade["from_player"]]
        to_player = game.players[trade["to_player"]]

        # Transfer properties from from_player to to_player
        for pos in trade["give_properties"]:
            prop = game.property_manager.properties[pos]
            prop.owner = to_player.id

        # Transfer properties from to_player to from_player
        for pos in trade["want_properties"]:
            prop = game.property_manager.properties[pos]
            prop.owner = from_player.id

        # Transfer money
        if trade["give_money"] > 0:
            from_player.remove_money(trade["give_money"])
            to_player.add_money(trade["give_money"])

        if trade["want_money"] > 0:
            to_player.remove_money(trade["want_money"])
            from_player.add_money(trade["want_money"])

        # Remove the trade from pending
        game._remove_pending_trade(self.trade_id)

    def to_dict(self) -> dict[str, int | str | list[int]]:
        """Serialize with trade_id."""
        data = super().to_dict()
        data["trade_id"] = self.trade_id
        return data


@dataclass
class RejectTrade(Action):
    """Reject a pending trade offer."""

    trade_id: int

    def validate(self, game: "MonopolyGame") -> tuple[bool, str]:
        """Validate trade rejection."""
        phase_valid, phase_error = game.validate_phase(self)
        if not phase_valid:
            return False, phase_error
        player = game.players[self.player_id]
        if player.bankrupt:
            return False, "Player is bankrupt"

        # Check if trade exists and is directed to this player
        trade = game._get_pending_trade(self.trade_id)
        if trade is None:
            return False, "Trade does not exist"

        if trade["to_player"] != self.player_id:
            return False, "Trade is not for you"

        return True, ""

    def execute(self, game: "MonopolyGame") -> None:
        """Remove the trade from pending offers."""
        game._remove_pending_trade(self.trade_id)

    def to_dict(self) -> dict[str, int | str | list[int]]:
        """Serialize with trade_id."""
        data = super().to_dict()
        data["trade_id"] = self.trade_id
        return data


@dataclass
class PayJailFine(Action):
    """Pay $50 to get out of jail."""

    def validate(self, game: "MonopolyGame") -> tuple[bool, str]:
        """Validate jail fine payment."""
        phase_valid, phase_error = game.validate_phase(self)
        if not phase_valid:
            return False, phase_error
        if game.current_player != self.player_id:
            return False, "Not your turn"

        player = game.players[self.player_id]
        if player.bankrupt:
            return False, "Player is bankrupt"

        if not player.in_jail:
            return False, "Player is not in jail"

        if player.money < JAIL_FINE:
            return False, "Insufficient funds to pay jail fine"

        return True, ""

    def execute(self, game: "MonopolyGame") -> None:
        """Pay the fine and get out of jail."""
        player = game.players[self.player_id]

        player.remove_money(JAIL_FINE)
        player.in_jail = False
        player.jail_turns = 0


@dataclass
class UseJailCard(Action):
    """Use a "Get Out of Jail Free" card."""

    def validate(self, game: "MonopolyGame") -> tuple[bool, str]:
        """Validate jail card usage."""
        phase_valid, phase_error = game.validate_phase(self)
        if not phase_valid:
            return False, phase_error
        if game.current_player != self.player_id:
            return False, "Not your turn"

        player = game.players[self.player_id]
        if player.bankrupt:
            return False, "Player is bankrupt"

        if not player.in_jail:
            return False, "Player is not in jail"

        if player.jail_cards <= 0:
            return False, "Player does not have a Get Out of Jail Free card"

        return True, ""

    def execute(self, game: "MonopolyGame") -> None:
        """Use the card and get out of jail."""
        player = game.players[self.player_id]

        player.jail_cards -= 1
        player.in_jail = False
        player.jail_turns = 0

        # Return card to appropriate deck
        # This will be handled by game.py when implemented
        game._return_jail_card(self.player_id)


@dataclass
class DeclareBankruptcy(Action):
    """Declare bankruptcy and exit the game.

    All assets are transferred to the creditor (or bank if no creditor).
    """

    def validate(self, game: "MonopolyGame") -> tuple[bool, str]:
        """Validate bankruptcy declaration."""
        phase_valid, phase_error = game.validate_phase(self)
        if not phase_valid:
            return False, phase_error
        player = game.players[self.player_id]

        if player.bankrupt:
            return False, "Player is already bankrupt"

        # Player can always declare bankruptcy if they choose
        # (though they may not be forced to yet)
        return True, ""

    def execute(self, game: "MonopolyGame") -> None:
        debt = game.state.obligations[0]
        game.handle_bankruptcy(self.player_id, debt["creditor"])


@dataclass
class EndTurn(Action):
    """End the current player's turn and advance to the next player."""

    def validate(self, game: "MonopolyGame") -> tuple[bool, str]:
        """Validate turn ending."""
        phase_valid, phase_error = game.validate_phase(self)
        if not phase_valid:
            return False, phase_error
        if game.current_player != self.player_id:
            return False, "Not your turn"

        player = game.players[self.player_id]
        if player.bankrupt:
            return False, "Player is bankrupt"

        # Check if player has any outstanding obligations
        # (This would be determined by game phase/state in game.py)
        # For now, we allow ending turn at any time
        return True, ""

    def execute(self, game: "MonopolyGame") -> None:
        game.end_turn()


@dataclass
class PassBuy(Action):
    """Decline the pending purchase without ending the turn."""

    def validate(self, game: "MonopolyGame") -> tuple[bool, str]:
        return game.validate_phase(self)

    def execute(self, game: "MonopolyGame") -> None:
        game.state.phase = "asset_management"


@dataclass
class SellBuildingGroup(Action):
    """Liquidate all development in a color group; property_id identifies its first space."""

    property_id: int

    def validate(self, game: "MonopolyGame") -> tuple[bool, str]:
        from .foundation import group_properties

        valid, error = game.validate_phase(self)
        if not valid:
            return valid, error
        props = group_properties(game, self.property_id)
        if not props or any(p.owner != self.player_id for p in props):
            return False, "Must own the color group"
        return (True, "") if any(p.houses for p in props) else (False, "No buildings")

    def execute(self, game: "MonopolyGame") -> None:
        from .foundation import group_properties
        from .rules import get_house_sale_value

        for prop in group_properties(game, self.property_id):
            game.players[self.player_id].add_money(
                prop.houses * get_house_sale_value(prop.position)
            )
            if prop.houses == 5:
                game.hotels_remaining += 1
            else:
                game.houses_remaining += prop.houses
            prop.houses = 0

    def to_dict(self) -> dict[str, int | str | list[int]]:
        return {**super().to_dict(), "property_id": self.property_id}
