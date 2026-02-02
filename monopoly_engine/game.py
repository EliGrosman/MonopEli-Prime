"""Main game controller for the Monopoly game engine.

This module provides the MonopolyGame class that orchestrates all game logic
and centralizes all state mutations. All game state changes should flow through
this class's methods, not by directly modifying state objects.
"""

import random
from typing import Any

from .state import GameState
from .player import Player
from .property import PropertyManager, Property
from .board import Board, PropertySpace, RailroadSpace, UtilitySpace
from .cards import CardDeck, Card, CHANCE_CARDS, COMMUNITY_CHEST_CARDS
from .rules import (
    calculate_rent,
    can_build_house,
    can_sell_house,
    get_building_cost,
    get_house_sale_value,
    get_mortgage_value,
    get_unmortgage_cost,
)
from .types import (
    TradeOfferData,
    CardType,
    SpaceType,
    GO_SALARY,
    JAIL_POSITION,
    GO_TO_JAIL_POSITION,
    JAIL_FINE,
    MAX_JAIL_TURNS,
    INCOME_TAX_AMOUNT,
    LUXURY_TAX_AMOUNT,
    BOARD_SIZE,
)
from .exceptions import (
    InvalidPlayerError,
    InvalidPropertyError,
    GameOverError,
)


class MonopolyGame:
    """Main game controller that orchestrates all game logic.

    This class is the central orchestrator for the Monopoly game engine.
    All state mutations should flow through this class's methods to maintain
    consistency and enable proper event logging.

    The game maintains:
    - Player positions, money, and jail status
    - Property ownership and buildings
    - Card decks and draws
    - Turn order and game flow
    - Event logging for debugging/replay

    Attributes:
        state: The complete game state container
        rng: Random number generator for dice rolls and shuffling
        players: Convenience alias for state.players
        property_manager: Convenience alias for state.property_manager
        board: Reference to the static board definition
    """

    def __init__(
        self,
        num_players: int = 2,
        player_names: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize a new Monopoly game.

        Args:
            num_players: Number of players (2-8)
            player_names: Optional list of player names (auto-generated if not provided)
            seed: Random seed for deterministic gameplay (None = random seed)

        Raises:
            ValueError: If num_players is invalid or player_names length doesn't match
        """
        if not 2 <= num_players <= 8:
            raise ValueError("Number of players must be between 2 and 8")

        if player_names is not None and len(player_names) != num_players:
            raise ValueError(
                f"Expected {num_players} player names, got {len(player_names)}"
            )

        # Initialize random number generator
        self.rng = random.Random(seed)

        # Create players
        if player_names is None:
            player_names = [f"Player {i + 1}" for i in range(num_players)]

        players = [Player(id=i, name=name) for i, name in enumerate(player_names)]

        # Create property manager
        property_manager = PropertyManager()

        # Create card decks with the same seed for determinism
        chance_deck = CardDeck.create_chance_deck(seed=self.rng.randint(0, 1000000))
        chest_deck = CardDeck.create_community_chest_deck(
            seed=self.rng.randint(0, 1000000)
        )

        # Initialize game state
        self.state = GameState(
            players=players,
            property_manager=property_manager,
            chance_deck=chance_deck,
            chest_deck=chest_deck,
        )

        # Convenience aliases
        self.players = self.state.players
        self.property_manager = self.state.property_manager
        self.board = Board()

        self.log_event(f"Game initialized with {num_players} players")

    # Convenience properties

    @property
    def current_player(self) -> int:
        """Get the current player's ID."""
        return self.state.current_player

    @current_player.setter
    def current_player(self, value: int) -> None:
        """Set the current player's ID."""
        self.state.current_player = value

    @property
    def houses_remaining(self) -> int:
        """Get number of houses remaining in the bank."""
        return self.state.houses_remaining

    @houses_remaining.setter
    def houses_remaining(self, value: int) -> None:
        """Set number of houses remaining in the bank."""
        self.state.houses_remaining = value

    @property
    def hotels_remaining(self) -> int:
        """Get number of hotels remaining in the bank."""
        return self.state.hotels_remaining

    @hotels_remaining.setter
    def hotels_remaining(self, value: int) -> None:
        """Set number of hotels remaining in the bank."""
        self.state.hotels_remaining = value

    @property
    def doubles_count(self) -> int:
        """Get the current doubles count."""
        return self.state.doubles_count

    @doubles_count.setter
    def doubles_count(self, value: int) -> None:
        """Set the current doubles count."""
        self.state.doubles_count = value

    @property
    def last_roll(self) -> tuple[int, int] | None:
        """Get the last dice roll."""
        return self.state.last_roll

    @last_roll.setter
    def last_roll(self, value: tuple[int, int] | None) -> None:
        """Set the last dice roll."""
        self.state.last_roll = value

    @property
    def turn_number(self) -> int:
        """Get the current turn number."""
        return self.state.turn_number

    @turn_number.setter
    def turn_number(self, value: int) -> None:
        """Set the current turn number."""
        self.state.turn_number = value

    @property
    def game_over(self) -> bool:
        """Check if the game is over."""
        return self.state.game_over

    @game_over.setter
    def game_over(self, value: bool) -> None:
        """Set game over status."""
        self.state.game_over = value

    @property
    def winner(self) -> int | None:
        """Get the winner's player ID."""
        return self.state.winner

    @winner.setter
    def winner(self, value: int | None) -> None:
        """Set the winner's player ID."""
        self.state.winner = value

    # Movement orchestration methods

    def move_player(self, player_id: int, spaces: int) -> tuple[int, bool]:
        """Move a player by a number of spaces.

        Orchestrates:
        - Movement around the board
        - Passing GO detection
        - GO salary collection ($200)
        - Event logging

        Args:
            player_id: The player to move
            spaces: Number of spaces to move (can be negative)

        Returns:
            Tuple of (new_position, passed_go)

        Raises:
            InvalidPlayerError: If player_id is invalid
        """
        player = self._get_player(player_id)
        old_pos = player.position

        new_pos, passed_go = player.move(spaces)

        # Collect GO salary if passed GO
        if passed_go:
            player.add_money(GO_SALARY)
            self.log_event(
                f"{player.name} passed GO and collected ${GO_SALARY}"
            )

        self.log_event(
            f"{player.name} moved from {old_pos} to {new_pos} ({spaces} spaces)"
        )

        return new_pos, passed_go

    def move_player_to(self, player_id: int, position: int) -> bool:
        """Move a player directly to a position.

        Used for Chance/Community Chest cards that move to specific locations.

        Args:
            player_id: The player to move
            position: Target position (0-39)

        Returns:
            True if player passed GO

        Raises:
            InvalidPlayerError: If player_id is invalid
        """
        player = self._get_player(player_id)
        old_pos = player.position

        passed_go = player.move_to(position)

        # Collect GO salary if passed GO
        if passed_go:
            player.add_money(GO_SALARY)
            self.log_event(
                f"{player.name} passed GO and collected ${GO_SALARY}"
            )

        space = Board.get_space(position)
        self.log_event(
            f"{player.name} moved from {old_pos} to {position} ({space.name})"
        )

        return passed_go

    def send_to_jail(self, player_id: int) -> None:
        """Send a player to jail.

        Moves player to jail position, sets in_jail flag, and resets doubles count.

        Args:
            player_id: The player to send to jail

        Raises:
            InvalidPlayerError: If player_id is invalid
        """
        player = self._get_player(player_id)

        player.go_to_jail()
        self.state.doubles_count = 0

        self.log_event(f"{player.name} went to JAIL")

    # House/Hotel inventory management

    def allocate_house(self, property_id: int) -> bool:
        """Allocate a house from the bank to a property.

        Args:
            property_id: The property position

        Returns:
            True if house was allocated, False if no houses available

        Raises:
            InvalidPropertyError: If property_id is invalid
        """
        prop = self._get_property(property_id)

        if self.state.houses_remaining <= 0:
            return False

        prop.houses += 1
        self.state.houses_remaining -= 1

        space = Board.get_space(property_id)
        owner = self._get_player(prop.owner)  # type: ignore
        self.log_event(
            f"{owner.name} built house on {space.name} "
            f"({prop.houses} houses, {self.state.houses_remaining} remaining)"
        )

        return True

    def return_house(self, property_id: int) -> None:
        """Return a house from a property to the bank.

        Args:
            property_id: The property position

        Raises:
            InvalidPropertyError: If property_id is invalid
            ValueError: If property has no houses
        """
        prop = self._get_property(property_id)

        if prop.houses == 0:
            raise ValueError(f"Property {property_id} has no houses to return")

        if prop.houses == 5:
            raise ValueError(
                f"Property {property_id} has a hotel, use return_hotel() instead"
            )

        prop.houses -= 1
        self.state.houses_remaining += 1

        space = Board.get_space(property_id)
        owner = self._get_player(prop.owner)  # type: ignore
        self.log_event(
            f"{owner.name} sold house on {space.name} "
            f"({prop.houses} houses, {self.state.houses_remaining} remaining)"
        )

    def allocate_hotel(self, property_id: int) -> bool:
        """Allocate a hotel from the bank to a property.

        Returns 4 houses to the bank and allocates a hotel.

        Args:
            property_id: The property position

        Returns:
            True if hotel was allocated, False if no hotels available

        Raises:
            InvalidPropertyError: If property_id is invalid
            ValueError: If property doesn't have 4 houses
        """
        prop = self._get_property(property_id)

        if prop.houses != 4:
            raise ValueError(
                f"Property {property_id} must have exactly 4 houses to build hotel"
            )

        if self.state.hotels_remaining <= 0:
            return False

        # Return 4 houses to bank
        self.state.houses_remaining += 4

        # Allocate hotel
        prop.houses = 5
        self.state.hotels_remaining -= 1

        space = Board.get_space(property_id)
        owner = self._get_player(prop.owner)  # type: ignore
        self.log_event(
            f"{owner.name} built hotel on {space.name} "
            f"({self.state.hotels_remaining} hotels remaining)"
        )

        return True

    def return_hotel(self, property_id: int) -> bool:
        """Return a hotel from a property to the bank.

        Returns hotel to bank and allocates 4 houses (if available).
        If 4 houses aren't available, the hotel is still returned but
        the property is left with 0 buildings.

        Args:
            property_id: The property position

        Returns:
            True if 4 houses were allocated, False if not enough houses available

        Raises:
            InvalidPropertyError: If property_id is invalid
            ValueError: If property has no hotel
        """
        prop = self._get_property(property_id)

        if prop.houses != 5:
            raise ValueError(f"Property {property_id} has no hotel to return")

        # Return hotel to bank
        prop.houses = 0
        self.state.hotels_remaining += 1

        # Try to allocate 4 houses
        if self.state.houses_remaining >= 4:
            prop.houses = 4
            self.state.houses_remaining -= 4
            success = True
        else:
            # Not enough houses, property left with 0 buildings
            success = False

        space = Board.get_space(property_id)
        owner = self._get_player(prop.owner)  # type: ignore

        if success:
            self.log_event(
                f"{owner.name} sold hotel on {space.name} for 4 houses"
            )
        else:
            self.log_event(
                f"{owner.name} sold hotel on {space.name} "
                f"but only {self.state.houses_remaining} houses available"
            )

        return success

    # Property ownership management

    def transfer_property(
        self,
        property_id: int,
        from_player: int | None,
        to_player: int | None,
    ) -> None:
        """Transfer property ownership.

        Centralizes property transfers with validation and logging.

        Args:
            property_id: The property position
            from_player: Previous owner ID (None if from bank)
            to_player: New owner ID (None if to bank)

        Raises:
            InvalidPropertyError: If property_id is invalid
            InvalidPlayerError: If player IDs are invalid
        """
        prop = self._get_property(property_id)
        space = Board.get_space(property_id)

        # Validate players
        if from_player is not None:
            from_name = self._get_player(from_player).name
        else:
            from_name = "Bank"

        if to_player is not None:
            to_name = self._get_player(to_player).name
        else:
            to_name = "Bank"

        # Transfer ownership
        prop.owner = to_player

        self.log_event(
            f"{space.name} transferred from {from_name} to {to_name}"
        )

    def reset_property(self, property_id: int) -> None:
        """Reset a property to unowned state.

        Clears ownership, houses, and mortgage status.
        Used during bankruptcy.

        Args:
            property_id: The property position

        Raises:
            InvalidPropertyError: If property_id is invalid
        """
        prop = self._get_property(property_id)
        space = Board.get_space(property_id)

        # Return buildings to bank
        if prop.houses == 5:
            self.state.hotels_remaining += 1
        elif prop.houses > 0:
            self.state.houses_remaining += prop.houses

        # Reset property
        prop.owner = None
        prop.houses = 0
        prop.mortgaged = False

        self.log_event(f"{space.name} reset to unowned state")

    # Trade management

    def propose_trade(
        self,
        from_player: int,
        to_player: int,
        give_properties: list[int],
        give_money: int,
        want_properties: list[int],
        want_money: int,
    ) -> int:
        """Propose a trade between players.

        Creates a TradeOfferData and stores it in state.pending_trades.

        Args:
            from_player: Player initiating the trade
            to_player: Player receiving the trade offer
            give_properties: Properties the initiator is offering
            give_money: Money the initiator is offering
            want_properties: Properties the initiator wants
            want_money: Money the initiator wants

        Returns:
            The trade ID

        Raises:
            InvalidPlayerError: If player IDs are invalid
        """
        # Validate players
        from_p = self._get_player(from_player)
        to_p = self._get_player(to_player)

        # Generate unique trade ID
        trade_id = len(self.state.pending_trades)

        # Create trade offer
        trade_offer: TradeOfferData = {
            "from_player": from_player,
            "to_player": to_player,
            "give_properties": give_properties,
            "give_money": give_money,
            "want_properties": want_properties,
            "want_money": want_money,
        }

        self.state.pending_trades[trade_id] = trade_offer

        self.log_event(
            f"{from_p.name} proposed trade #{trade_id} to {to_p.name}"
        )

        return trade_id

    def accept_trade(self, trade_id: int) -> None:
        """Accept and execute a trade.

        Transfers properties and money, then removes trade from pending.

        Args:
            trade_id: The trade ID

        Raises:
            ValueError: If trade_id is invalid or trade cannot be executed
        """
        if trade_id not in self.state.pending_trades:
            raise ValueError(f"Trade {trade_id} not found")

        trade = self.state.pending_trades[trade_id]

        from_player = self._get_player(trade["from_player"])
        to_player = self._get_player(trade["to_player"])

        # Validate funds
        if from_player.money < trade["give_money"]:
            raise ValueError(
                f"{from_player.name} doesn't have ${trade['give_money']}"
            )
        if to_player.money < trade["want_money"]:
            raise ValueError(
                f"{to_player.name} doesn't have ${trade['want_money']}"
            )

        # Transfer money
        if trade["give_money"] > 0:
            from_player.remove_money(trade["give_money"])
            to_player.add_money(trade["give_money"])

        if trade["want_money"] > 0:
            to_player.remove_money(trade["want_money"])
            from_player.add_money(trade["want_money"])

        # Transfer properties
        for prop_id in trade["give_properties"]:
            self.transfer_property(prop_id, trade["from_player"], trade["to_player"])

        for prop_id in trade["want_properties"]:
            self.transfer_property(prop_id, trade["to_player"], trade["from_player"])

        # Remove from pending
        del self.state.pending_trades[trade_id]

        self.log_event(
            f"Trade #{trade_id} executed: {from_player.name} <-> {to_player.name}"
        )

    def reject_trade(self, trade_id: int) -> None:
        """Reject a trade offer.

        Args:
            trade_id: The trade ID

        Raises:
            ValueError: If trade_id is invalid
        """
        if trade_id not in self.state.pending_trades:
            raise ValueError(f"Trade {trade_id} not found")

        trade = self.state.pending_trades[trade_id]
        from_player = self._get_player(trade["from_player"])
        to_player = self._get_player(trade["to_player"])

        del self.state.pending_trades[trade_id]

        self.log_event(
            f"Trade #{trade_id} rejected by {to_player.name}"
        )

    def get_pending_trade(self, trade_id: int) -> TradeOfferData | None:
        """Get a pending trade offer.

        Args:
            trade_id: The trade ID

        Returns:
            The trade offer data, or None if not found
        """
        return self.state.pending_trades.get(trade_id)

    # Bankruptcy handling

    def handle_bankruptcy(
        self,
        player_id: int,
        creditor_id: int | None = None,
    ) -> None:
        """Handle player bankruptcy.

        Orchestrates:
        - Clearing all property ownership
        - Returning houses/hotels to bank
        - Transferring assets to creditor or bank
        - Marking player as bankrupt
        - Checking for winner
        - Event logging

        Args:
            player_id: The bankrupt player
            creditor_id: The creditor (None if bank)

        Raises:
            InvalidPlayerError: If player IDs are invalid
        """
        player = self._get_player(player_id)

        if creditor_id is not None:
            creditor = self._get_player(creditor_id)
            creditor_name = creditor.name
        else:
            creditor = None
            creditor_name = "Bank"

        self.log_event(f"{player.name} declared bankruptcy to {creditor_name}")

        # Get all properties owned by the player
        owned_properties = self.property_manager.get_owned_by(player_id)

        # Return all buildings and transfer properties
        for prop_id in owned_properties:
            prop = self.property_manager.get(prop_id)
            assert prop is not None

            # Return buildings to bank
            if prop.houses == 5:
                self.state.hotels_remaining += 1
            elif prop.houses > 0:
                self.state.houses_remaining += prop.houses

            # Transfer property to creditor or reset to unowned
            if creditor_id is not None:
                # Creditor gets property (unmortgaged)
                prop.owner = creditor_id
                prop.houses = 0
                prop.mortgaged = False

                space = Board.get_space(prop_id)
                self.log_event(
                    f"{space.name} transferred to {creditor_name}"
                )
            else:
                # Bank gets property (reset to unowned)
                self.reset_property(prop_id)

        # Transfer money to creditor
        if player.money > 0:
            if creditor is not None:
                creditor.add_money(player.money)
                self.log_event(
                    f"${player.money} transferred to {creditor_name}"
                )

        # Mark player as bankrupt
        player.declare_bankrupt()

        # Check for winner
        winner_id = self.state.check_winner()
        if winner_id is not None:
            self.state.game_over = True
            self.state.winner = winner_id
            winner = self._get_player(winner_id)
            self.log_event(f"GAME OVER: {winner.name} wins!")

    # Turn management

    def end_turn(self) -> None:
        """End the current turn and advance to the next player.

        Resets turn-specific state like doubles count and advances
        to the next active player.
        """
        current = self.state.get_current_player()

        # Reset turn-specific state
        self.state.doubles_count = 0
        self.state.last_roll = None

        # Advance to next player
        next_player = self.state.next_player()
        self.state.turn_number += 1

        self.log_event(
            f"{current.name} ended turn. {next_player.name}'s turn begins."
        )

    def next_player(self) -> Player:
        """Advance to the next active player.

        Skips bankrupt players.

        Returns:
            The next active player
        """
        return self.state.next_player()

    # Game flow methods

    def roll_dice(self) -> tuple[int, int]:
        """Roll two dice.

        Returns:
            Tuple of (die1, die2)
        """
        die1 = self.rng.randint(1, 6)
        die2 = self.rng.randint(1, 6)
        return die1, die2

    def _handle_dice_roll(self, player_id: int) -> None:
        """Handle dice roll logic.

        Orchestrates:
        - Dice rolling
        - Doubles detection (3 in a row = jail)
        - Jail release on doubles
        - Player movement
        - Landing logic

        Args:
            player_id: The player rolling dice
        """
        player = self._get_player(player_id)

        # Roll dice
        die1, die2 = self.roll_dice()
        is_doubles = die1 == die2
        total = die1 + die2

        self.state.last_roll = (die1, die2)

        self.log_event(
            f"{player.name} rolled {die1} + {die2} = {total}"
            + (" (DOUBLES)" if is_doubles else "")
        )

        # Handle jail
        if player.in_jail:
            if is_doubles:
                # Doubles releases from jail
                player.leave_jail()
                self.state.doubles_count = 0
                self.log_event(f"{player.name} rolled doubles and got out of jail!")

                # Move normally
                self.move_player(player_id, total)
                self._handle_landing(player_id)
            else:
                # Increment jail turns
                jail_turns = player.increment_jail_turns()

                if jail_turns >= MAX_JAIL_TURNS:
                    # Must pay fine after 3 turns
                    if player.money >= JAIL_FINE:
                        player.remove_money(JAIL_FINE)
                        player.leave_jail()
                        self.log_event(
                            f"{player.name} paid ${JAIL_FINE} fine after 3 turns in jail"
                        )

                        # Move normally
                        self.move_player(player_id, total)
                        self._handle_landing(player_id)
                    else:
                        # Can't pay fine, bankruptcy
                        self.log_event(
                            f"{player.name} cannot afford jail fine"
                        )
                        self.handle_bankruptcy(player_id)
                else:
                    self.log_event(
                        f"{player.name} stays in jail (turn {jail_turns}/{MAX_JAIL_TURNS})"
                    )
            return

        # Handle doubles
        if is_doubles:
            self.state.doubles_count += 1

            if self.state.doubles_count >= 3:
                # Three doubles in a row = jail
                self.log_event(
                    f"{player.name} rolled 3 doubles in a row!"
                )
                self.send_to_jail(player_id)
                return
        else:
            self.state.doubles_count = 0

        # Move player
        self.move_player(player_id, total)

        # Handle landing
        self._handle_landing(player_id)

    def _handle_landing(self, player_id: int) -> None:
        """Handle landing on a space.

        Orchestrates:
        - Rent payment
        - Card drawing
        - Tax payment
        - GO TO JAIL
        - Bankruptcy checks

        Args:
            player_id: The player who landed
        """
        player = self._get_player(player_id)
        space = Board.get_space(player.position)

        self.log_event(f"{player.name} landed on {space.name}")

        # Handle different space types
        if space.space_type == SpaceType.PROPERTY:
            self._handle_property_landing(player_id)

        elif space.space_type == SpaceType.RAILROAD:
            self._handle_railroad_landing(player_id)

        elif space.space_type == SpaceType.UTILITY:
            self._handle_utility_landing(player_id)

        elif space.space_type == SpaceType.CHANCE:
            self._handle_chance_card(player_id)

        elif space.space_type == SpaceType.COMMUNITY_CHEST:
            self._handle_community_chest_card(player_id)

        elif space.space_type == SpaceType.INCOME_TAX:
            self._handle_tax(player_id, INCOME_TAX_AMOUNT, "Income Tax")

        elif space.space_type == SpaceType.LUXURY_TAX:
            self._handle_tax(player_id, LUXURY_TAX_AMOUNT, "Luxury Tax")

        elif space.space_type == SpaceType.GO_TO_JAIL:
            self.send_to_jail(player_id)

    def _handle_property_landing(self, player_id: int) -> None:
        """Handle landing on a property."""
        player = self._get_player(player_id)
        prop = self.property_manager.get(player.position)

        if prop is None or not prop.is_owned or prop.owner == player_id:
            return

        # Calculate and pay rent
        rent = calculate_rent(
            self.property_manager,
            player.position,
            dice_roll=sum(self.state.last_roll) if self.state.last_roll else 0,
        )

        if rent > 0:
            owner = self._get_player(prop.owner)  # type: ignore

            if not player.remove_money(rent):
                # Can't afford rent, bankruptcy
                self.log_event(
                    f"{player.name} cannot afford ${rent} rent"
                )
                self.handle_bankruptcy(player_id, prop.owner)
            else:
                owner.add_money(rent)
                self.log_event(
                    f"{player.name} paid ${rent} rent to {owner.name}"
                )

    def _handle_railroad_landing(self, player_id: int) -> None:
        """Handle landing on a railroad."""
        # Same logic as property
        self._handle_property_landing(player_id)

    def _handle_utility_landing(self, player_id: int) -> None:
        """Handle landing on a utility."""
        # Same logic as property
        self._handle_property_landing(player_id)

    def _handle_chance_card(self, player_id: int) -> None:
        """Handle drawing a Chance card."""
        card = self.state.chance_deck.draw()
        player = self._get_player(player_id)

        self.log_event(f"{player.name} drew Chance: {card.text}")

        self._execute_card(player_id, card)

    def _handle_community_chest_card(self, player_id: int) -> None:
        """Handle drawing a Community Chest card."""
        card = self.state.chest_deck.draw()
        player = self._get_player(player_id)

        self.log_event(f"{player.name} drew Community Chest: {card.text}")

        self._execute_card(player_id, card)

    def _execute_card(self, player_id: int, card: Card) -> None:
        """Execute a card's effect."""
        player = self._get_player(player_id)

        if card.card_type == CardType.MOVE:
            if card.move_to is not None:
                self.move_player_to(player_id, card.move_to)
                self._handle_landing(player_id)

        elif card.card_type == CardType.MOVE_NEAREST:
            if card.move_to_nearest == "railroad":
                self._move_to_nearest_railroad(player_id, from_card=True)
            elif card.move_to_nearest == "utility":
                self._move_to_nearest_utility(player_id, from_card=True)

        elif card.card_type == CardType.MOVE_BACK:
            if card.move_spaces is not None:
                self.move_player(player_id, card.move_spaces)
                self._handle_landing(player_id)

        elif card.card_type == CardType.COLLECT:
            if card.amount is not None:
                player.add_money(card.amount)
                self.log_event(f"{player.name} collected ${card.amount}")

        elif card.card_type == CardType.PAY:
            if card.amount is not None:
                if not player.remove_money(card.amount):
                    self.log_event(
                        f"{player.name} cannot afford ${card.amount}"
                    )
                    self.handle_bankruptcy(player_id)
                else:
                    self.log_event(f"{player.name} paid ${card.amount}")

        elif card.card_type == CardType.PAY_PER_BUILDING:
            if card.per_house is not None and card.per_hotel is not None:
                self.pay_house_hotel_repairs(
                    player_id,
                    card.per_house,
                    card.per_hotel,
                )

        elif card.card_type == CardType.COLLECT_FROM_PLAYERS:
            if card.amount is not None:
                total = 0
                for other in self.state.players:
                    if other.id != player_id and not other.bankrupt:
                        if other.remove_money(card.amount):
                            total += card.amount
                            player.add_money(card.amount)
                        else:
                            # Other player bankrupt to this player
                            self.handle_bankruptcy(other.id, player_id)

                self.log_event(
                    f"{player.name} collected ${total} from other players"
                )

        elif card.card_type == CardType.PAY_TO_PLAYERS:
            if card.amount is not None:
                for other in self.state.players:
                    if other.id != player_id and not other.bankrupt:
                        if not player.remove_money(card.amount):
                            # Player bankrupt
                            self.handle_bankruptcy(player_id)
                            return
                        other.add_money(card.amount)

                total = card.amount * (len(self.state.get_active_players()) - 1)
                self.log_event(
                    f"{player.name} paid ${total} to other players"
                )

        elif card.card_type == CardType.GET_OUT_OF_JAIL:
            player.add_jail_card()
            self.log_event(f"{player.name} received Get Out of Jail Free card")

        elif card.card_type == CardType.GO_TO_JAIL:
            self.send_to_jail(player_id)

    def _move_to_nearest_railroad(
        self,
        player_id: int,
        from_card: bool = False,
    ) -> None:
        """Move player to nearest railroad."""
        player = self._get_player(player_id)
        railroads = [5, 15, 25, 35]

        # Find nearest railroad ahead
        for railroad in railroads:
            if railroad > player.position:
                self.move_player_to(player_id, railroad)
                self._handle_landing(player_id)
                return

        # Wrap around to first railroad
        self.move_player_to(player_id, 5)
        self._handle_landing(player_id)

    def _move_to_nearest_utility(
        self,
        player_id: int,
        from_card: bool = False,
    ) -> None:
        """Move player to nearest utility."""
        player = self._get_player(player_id)
        utilities = [12, 28]

        # Find nearest utility ahead
        for utility in utilities:
            if utility > player.position:
                self.move_player_to(player_id, utility)
                self._handle_landing(player_id)
                return

        # Wrap around to first utility
        self.move_player_to(player_id, 12)
        self._handle_landing(player_id)

    def _handle_tax(self, player_id: int, amount: int, tax_name: str) -> None:
        """Handle tax payment."""
        player = self._get_player(player_id)

        if not player.remove_money(amount):
            self.log_event(f"{player.name} cannot afford ${amount} {tax_name}")
            self.handle_bankruptcy(player_id)
        else:
            self.log_event(f"{player.name} paid ${amount} {tax_name}")

    def pay_house_hotel_repairs(
        self,
        player_id: int,
        per_house: int,
        per_hotel: int,
    ) -> None:
        """Calculate and pay house/hotel repairs.

        Used for Chance/Community Chest cards that charge per building.

        Args:
            player_id: The player paying repairs
            per_house: Cost per house
            per_hotel: Cost per hotel
        """
        player = self._get_player(player_id)

        houses = 0
        hotels = 0

        for prop in self.property_manager.properties.values():
            if prop.owner == player_id:
                if prop.houses == 5:
                    hotels += 1
                else:
                    houses += prop.houses

        total = (houses * per_house) + (hotels * per_hotel)

        if total > 0:
            if not player.remove_money(total):
                self.log_event(
                    f"{player.name} cannot afford ${total} repairs "
                    f"({houses} houses @ ${per_house}, {hotels} hotels @ ${per_hotel})"
                )
                self.handle_bankruptcy(player_id)
            else:
                self.log_event(
                    f"{player.name} paid ${total} repairs "
                    f"({houses} houses @ ${per_house}, {hotels} hotels @ ${per_hotel})"
                )

    # Serialization

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete game state to JSON-compatible dict.

        Returns:
            Dictionary with all game state fields
        """
        return self.state.to_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MonopolyGame":
        """Reconstruct game from serialized state.

        Args:
            data: Dictionary containing serialized game state

        Returns:
            Reconstructed MonopolyGame instance
        """
        # Create game with minimal initialization
        game = cls.__new__(cls)

        # Reconstruct state
        game.state = GameState.from_dict(data)

        # Set up convenience aliases
        game.players = game.state.players
        game.property_manager = game.state.property_manager
        game.board = Board()

        # Create RNG (state is already deterministic from serialization)
        game.rng = random.Random()

        return game

    # Logging

    def log_event(self, message: str) -> None:
        """Add an event to the event log.

        Args:
            message: Description of the game event
        """
        self.state.log_event(message)

    # Helper methods for actions.py compatibility
    # TODO: Refactor actions.py to use public orchestration methods directly

    def _add_pending_trade(
        self,
        from_player: int,
        to_player: int,
        give_properties: list[int],
        give_money: int,
        want_properties: list[int],
        want_money: int,
    ) -> int:
        """Add a pending trade (compatibility method for actions.py)."""
        return self.propose_trade(
            from_player,
            to_player,
            give_properties,
            give_money,
            want_properties,
            want_money,
        )

    def _get_pending_trade(self, trade_id: int) -> TradeOfferData | None:
        """Get a pending trade (compatibility method for actions.py)."""
        return self.get_pending_trade(trade_id)

    def _remove_pending_trade(self, trade_id: int) -> None:
        """Remove a pending trade (compatibility method for actions.py)."""
        if trade_id in self.state.pending_trades:
            del self.state.pending_trades[trade_id]

    def _return_jail_card(self, player_id: int | None = None) -> None:
        """Return a jail card to the deck (compatibility method for actions.py).

        Args:
            player_id: Optional player ID (for compatibility with different calling conventions)
        """
        # Return card to appropriate deck
        # In Monopoly, there are 2 Get Out of Jail Free cards (1 Chance, 1 Community Chest)
        # For simplicity, we return to the deck with fewer cards
        if len(self.state.chance_deck.cards) < len(self.state.chest_deck.cards):
            # Return to Chance deck
            for card in CHANCE_CARDS:
                if card.card_type == CardType.GET_OUT_OF_JAIL:
                    self.state.chance_deck.return_jail_card(card)
                    break
        else:
            # Return to Community Chest deck
            for card in COMMUNITY_CHEST_CARDS:
                if card.card_type == CardType.GET_OUT_OF_JAIL:
                    self.state.chest_deck.return_jail_card(card)
                    break

    def _return_jail_cards(self, player_id: int) -> None:
        """Return all jail cards to decks (compatibility method for actions.py)."""
        player = self._get_player(player_id)
        while player.jail_cards > 0:
            player.jail_cards -= 1
            self._return_jail_card()

    # Core helper methods

    def _get_player(self, player_id: int) -> Player:
        """Get a player by ID.

        Args:
            player_id: The player's ID

        Returns:
            The Player object

        Raises:
            InvalidPlayerError: If player_id is invalid
        """
        player = self.state.get_player(player_id)
        if player is None:
            raise InvalidPlayerError(f"Invalid player ID: {player_id}")
        return player

    def _get_property(self, property_id: int) -> "Property":
        """Get a property by position.

        Args:
            property_id: The property position

        Returns:
            The Property object

        Raises:
            InvalidPropertyError: If property_id is invalid
        """
        prop = self.property_manager.get(property_id)
        if prop is None:
            raise InvalidPropertyError(f"Invalid property ID: {property_id}")
        return prop

    def __str__(self) -> str:
        """Human-readable string representation."""
        return str(self.state)
