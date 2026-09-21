"""Legacy compatibility tests.

This module tests that the new monopoly_engine implementation behaves correctly
compared to the legacy MonopEli-Server implementation, while also fixing known bugs.
"""

import pytest
from monopoly_engine.game import MonopolyGame
from monopoly_engine.actions import (
    RollDice,
    BuyProperty,
    BuildHouse,
    BuildHotel,
    MortgageProperty,
    UnmortgageProperty,
    PayJailFine,
    UseJailCard,
    DeclareBankruptcy,
)
from monopoly_engine.types import PropertyColor
from monopoly_engine.rules import calculate_rent


class TestDiceRolling:
    """Test that dice rolling is fixed (not 1-7 like in legacy)."""

    def test_dice_range_is_correct(self) -> None:
        """Verify dice roll in range 1-6, not 1-7 as in legacy bug."""
        game = MonopolyGame(num_players=2, seed=42)

        # Roll dice many times and check range
        for _ in range(1000):
            d1, d2 = game.roll_dice()
            assert 1 <= d1 <= 6, f"Die 1 out of range: {d1}"
            assert 1 <= d2 <= 6, f"Die 2 out of range: {d2}"
            assert 2 <= d1 + d2 <= 12, f"Total out of range: {d1 + d2}"


class TestRentCalculation:
    """Test rent calculation matches legacy behavior."""

    def test_property_base_rent_no_monopoly(self) -> None:
        """Test base rent on property without monopoly."""
        game = MonopolyGame(num_players=2, seed=42)

        # Buy Mediterranean Ave (position 1) - base rent is $2
        game.state.players[0].position = 1
        game.state.players[0].money = 1500
        BuyProperty(player_id=0, property_id=1).execute(game)

        # Calculate rent for player 1 landing on it
        rent = calculate_rent(game.state.property_manager, 1)
        assert rent == 2, f"Expected base rent of $2, got ${rent}"

    def test_property_monopoly_doubles_rent(self) -> None:
        """Test that owning a monopoly doubles base rent when no houses."""
        game = MonopolyGame(num_players=2, seed=42)

        # Give player 0 both Brown properties (Mediterranean and Baltic)
        game.state.property_manager.properties[1].owner = 0  # Mediterranean
        game.state.property_manager.properties[3].owner = 0  # Baltic

        # Calculate rent - should be 2x base rent
        rent = calculate_rent(game.state.property_manager, 1)
        assert rent == 4, f"Expected doubled rent of $4 for monopoly, got ${rent}"

    def test_property_rent_with_houses(self) -> None:
        """Test rent calculation with houses."""
        game = MonopolyGame(num_players=2, seed=42)

        # Give player 0 monopoly and add houses
        game.state.property_manager.properties[1].owner = 0  # Mediterranean
        game.state.property_manager.properties[3].owner = 0  # Baltic
        game.state.property_manager.properties[1].houses = 2  # 2 houses

        # Rent with 2 houses on Mediterranean should be $30
        rent = calculate_rent(game.state.property_manager, 1)
        assert rent == 30, f"Expected rent of $30 with 2 houses, got ${rent}"

    def test_property_rent_with_hotel(self) -> None:
        """Test rent calculation with hotel (houses=5)."""
        game = MonopolyGame(num_players=2, seed=42)

        # Give player 0 monopoly and hotel
        game.state.property_manager.properties[1].owner = 0  # Mediterranean
        game.state.property_manager.properties[3].owner = 0  # Baltic
        game.state.property_manager.properties[1].houses = 5  # Hotel

        # Rent with hotel on Mediterranean should be $250
        rent = calculate_rent(game.state.property_manager, 1)
        assert rent == 250, f"Expected hotel rent of $250, got ${rent}"

    def test_mortgaged_property_no_rent(self) -> None:
        """Test that mortgaged properties don't collect rent."""
        game = MonopolyGame(num_players=2, seed=42)

        # Give player 0 Mediterranean and mortgage it
        game.state.property_manager.properties[1].owner = 0
        game.state.property_manager.properties[1].mortgaged = True

        # Rent should be 0
        rent = calculate_rent(game.state.property_manager, 1)
        assert rent == 0, f"Expected no rent on mortgaged property, got ${rent}"

    def test_railroad_rent_by_count(self) -> None:
        """Test railroad rent based on number owned."""
        game = MonopolyGame(num_players=2, seed=42)

        # Reading Railroad is position 5
        game.state.property_manager.properties[5].owner = 0

        # 1 railroad: $25
        rent = calculate_rent(game.state.property_manager, 5)
        assert rent == 25, f"Expected $25 for 1 railroad, got ${rent}"

        # Add second railroad (Pennsylvania at 15)
        game.state.property_manager.properties[15].owner = 0
        rent = calculate_rent(game.state.property_manager, 5)
        assert rent == 50, f"Expected $50 for 2 railroads, got ${rent}"

        # Add third railroad (B&O at 25)
        game.state.property_manager.properties[25].owner = 0
        rent = calculate_rent(game.state.property_manager, 5)
        assert rent == 100, f"Expected $100 for 3 railroads, got ${rent}"

        # Add fourth railroad (Short Line at 35)
        game.state.property_manager.properties[35].owner = 0
        rent = calculate_rent(game.state.property_manager, 5)
        assert rent == 200, f"Expected $200 for 4 railroads, got ${rent}"

    def test_utility_rent_one_owned(self) -> None:
        """Test utility rent with one owned (4x dice roll)."""
        game = MonopolyGame(num_players=2, seed=42)

        # Electric Company is position 12
        game.state.property_manager.properties[12].owner = 0

        # Set last roll to (3, 4) = 7
        game.state.last_roll = (3, 4)

        # Rent should be 4 * 7 = 28
        rent = calculate_rent(game.state.property_manager, 12, dice_roll=7)
        assert rent == 28, f"Expected 4x dice roll = $28, got ${rent}"

    def test_utility_rent_both_owned(self) -> None:
        """Test utility rent with both owned (10x dice roll)."""
        game = MonopolyGame(num_players=2, seed=42)

        # Give player 0 both utilities
        game.state.property_manager.properties[12].owner = 0  # Electric
        game.state.property_manager.properties[28].owner = 0  # Water Works

        # Set last roll to (3, 4) = 7
        game.state.last_roll = (3, 4)

        # Rent should be 10 * 7 = 70
        rent = calculate_rent(game.state.property_manager, 12, dice_roll=7)
        assert rent == 70, f"Expected 10x dice roll = $70, got ${rent}"


class TestMovementAndGO:
    """Test movement and GO passing logic."""

    def test_passing_go_awards_200(self) -> None:
        """Test that passing GO awards $200."""
        game = MonopolyGame(num_players=2, seed=42)

        # Put player at position 38 (Luxury Tax)
        game.state.players[0].position = 38
        initial_money = game.state.players[0].money

        # Move 4 spaces (would land on 42 -> wraps to 2)
        new_pos, passed_go = game.move_player(0, 4)

        assert passed_go, "Should have passed GO"
        assert new_pos == 2, f"Expected position 2, got {new_pos}"
        assert game.state.players[0].money == initial_money + 200, "Should award $200 for passing GO"

    def test_landing_on_go_awards_200(self) -> None:
        """Test that landing exactly on GO awards $200."""
        game = MonopolyGame(num_players=2, seed=42)

        game.state.players[0].position = 38
        initial_money = game.state.players[0].money

        # Move 2 spaces to land exactly on GO
        new_pos, passed_go = game.move_player(0, 2)

        assert passed_go, "Should have passed GO (landed on it)"
        assert new_pos == 0, f"Expected position 0 (GO), got {new_pos}"
        assert game.state.players[0].money == initial_money + 200, "Should award $200 for landing on GO"

    def test_not_passing_go_no_money(self) -> None:
        """Test that not passing GO doesn't award money."""
        game = MonopolyGame(num_players=2, seed=42)

        game.state.players[0].position = 10
        initial_money = game.state.players[0].money

        # Move 5 spaces (10 -> 15), doesn't cross GO
        new_pos, passed_go = game.move_player(0, 5)

        assert not passed_go, "Should not have passed GO"
        assert new_pos == 15, f"Expected position 15, got {new_pos}"
        assert game.state.players[0].money == initial_money, "Should not award money"


class TestBuildingRules:
    """Test house and hotel building rules."""

    def test_building_requires_monopoly(self) -> None:
        """Test that you must own monopoly to build."""
        game = MonopolyGame(num_players=2, seed=42)

        # Give player 0 just Mediterranean (not both Browns)
        game.state.property_manager.properties[1].owner = 0
        game.state.players[0].money = 1000

        # Try to build house - should fail
        action = BuildHouse(player_id=0, property_id=1)
        valid, msg = action.validate(game)

        assert not valid, "Should not be able to build without monopoly"
        assert "monopoly" in msg.lower(), f"Error should mention monopoly, got: {msg}"

    def test_building_requires_unmortgaged_properties(self) -> None:
        """Test that you cannot build on a mortgaged property."""
        game = MonopolyGame(num_players=2, seed=42)

        # Give player 0 both Browns and mortgage Mediterranean
        game.state.property_manager.properties[1].owner = 0  # Mediterranean
        game.state.property_manager.properties[3].owner = 0  # Baltic
        game.state.property_manager.properties[1].mortgaged = True
        game.state.players[0].money = 1000

        # Try to build house on mortgaged property - should fail
        action = BuildHouse(player_id=0, property_id=1)
        valid, msg = action.validate(game)

        assert not valid, "Should not be able to build on mortgaged property"
        assert "mortgaged" in msg.lower(), f"Error should mention mortgage, got: {msg}"

    def test_house_limit_32(self) -> None:
        """Test that house limit is 32."""
        game = MonopolyGame(num_players=2, seed=42)

        # Set houses remaining to 0
        game.state.houses_remaining = 0

        # Give player 0 monopoly
        game.state.property_manager.properties[1].owner = 0
        game.state.property_manager.properties[3].owner = 0
        game.state.players[0].money = 1000

        # Try to build house - should fail
        action = BuildHouse(player_id=0, property_id=1)
        valid, msg = action.validate(game)

        assert not valid, "Should not be able to build when no houses available"
        assert "no houses" in msg.lower() or "shortage" in msg.lower(), f"Error should mention shortage, got: {msg}"

    def test_hotel_limit_12(self) -> None:
        """Test that hotel limit is 12."""
        game = MonopolyGame(num_players=2, seed=42)

        # Set hotels remaining to 0
        game.state.hotels_remaining = 0

        # Give player 0 monopoly with 4 houses on BOTH properties (to satisfy even building)
        game.state.property_manager.properties[1].owner = 0
        game.state.property_manager.properties[3].owner = 0
        game.state.property_manager.properties[1].houses = 4
        game.state.property_manager.properties[3].houses = 4  # Even building
        game.state.players[0].money = 1000

        # Try to build hotel - should fail
        action = BuildHotel(player_id=0, property_id=1)
        valid, msg = action.validate(game)

        assert not valid, "Should not be able to build hotel when none available"
        assert "no hotels" in msg.lower() or "shortage" in msg.lower(), f"Error should mention shortage, got: {msg}"

    def test_hotel_returns_4_houses(self) -> None:
        """Test that building hotel returns 4 houses to bank."""
        game = MonopolyGame(num_players=2, seed=42)

        # Give player 0 monopoly with 4 houses
        game.state.property_manager.properties[1].owner = 0
        game.state.property_manager.properties[3].owner = 0
        game.state.property_manager.properties[1].houses = 4
        game.state.players[0].money = 1000

        initial_houses = game.state.houses_remaining

        # Build hotel
        action = BuildHotel(player_id=0, property_id=1)
        action.execute(game)

        # Should have returned 4 houses to bank
        assert game.state.houses_remaining == initial_houses + 4, "Building hotel should return 4 houses"
        assert game.state.property_manager.properties[1].houses == 5, "Hotel should be houses=5"


class TestMortgaging:
    """Test mortgage and unmortgage mechanics."""

    def test_unmortgage_costs_110_percent(self) -> None:
        """Test that unmortgaging costs 110% of mortgage value."""
        game = MonopolyGame(num_players=2, seed=42)

        # Give player 0 Mediterranean (mortgage value $30)
        game.state.property_manager.properties[1].owner = 0
        game.state.property_manager.properties[1].mortgaged = True
        game.state.players[0].money = 100

        # Unmortgage should cost $33 (110% of $30)
        from monopoly_engine.rules import get_unmortgage_cost

        cost = get_unmortgage_cost(1)  # Pass position, not property
        assert cost == 33, f"Expected unmortgage cost of $33, got ${cost}"

    def test_mortgage_gives_50_percent(self) -> None:
        """Test that mortgaging gives 50% of purchase price."""
        game = MonopolyGame(num_players=2, seed=42)

        # Mediterranean costs $60, mortgage value should be $30
        from monopoly_engine.rules import get_mortgage_value

        mortgage_value = get_mortgage_value(1)  # Pass position, not property
        assert mortgage_value == 30, f"Expected mortgage value of $30, got ${mortgage_value}"


class TestJailMechanics:
    """Test jail mechanics."""

    def test_jail_position_is_10(self) -> None:
        """Test that jail position is 10."""
        game = MonopolyGame(num_players=2, seed=42)

        # Send player to jail
        game.send_to_jail(0)

        # Should be at position 10 (jail)
        assert game.state.players[0].position == 10, "Jail position should be 10"
        assert game.state.players[0].in_jail, "Player should be in jail"
        assert game.state.players[0].jail_turns == 0, "Jail turn counter should start at 0"

    def test_jail_fine_is_50(self) -> None:
        """Test that jail fine is $50."""
        game = MonopolyGame(num_players=2, seed=42)

        # Send player to jail
        game.send_to_jail(0)
        game.state.players[0].money = 100

        # Pay fine
        action = PayJailFine(player_id=0)
        action.execute(game)

        # Should have paid $50
        assert game.state.players[0].money == 50, "Should have paid $50 jail fine"
        assert not game.state.players[0].in_jail, "Should be out of jail"

    def test_jail_card_releases_player(self) -> None:
        """Test that using Get Out of Jail Free card releases player."""
        game = MonopolyGame(num_players=2, seed=42)

        # Send player to jail and give them a card
        game.send_to_jail(0)
        game.state.players[0].jail_cards = 1
        game.state.jail_card_sources[0] = ["chance"]

        # Use card
        action = UseJailCard(player_id=0)
        action.execute(game)

        # Should be out of jail
        assert not game.state.players[0].in_jail, "Should be out of jail"
        assert game.state.players[0].jail_cards == 0, "Card should be used"

    def test_jail_auto_release_after_3_turns(self) -> None:
        """Test that player is automatically released after 3 turns in jail."""
        game = MonopolyGame(num_players=2, seed=42)

        # Send player to jail
        game.send_to_jail(0)

        # Simulate 3 turns (jail_turns goes 0 -> 1 -> 2, then auto-release)
        game.state.players[0].jail_turns = 2
        game.state.phase = "jail_decision"
        game.roll_dice = lambda: (1, 2)

        # Roll dice should trigger auto-release
        action = RollDice(player_id=0)
        game.apply_action(0, action)

        # Should be out of jail
        assert not game.state.players[0].in_jail, "Should be auto-released after 3 turns"


class TestBankruptcy:
    """Test bankruptcy mechanics."""

    def test_bankruptcy_to_bank_returns_properties(self) -> None:
        """Test that bankruptcy to bank returns properties unmortgaged."""
        game = MonopolyGame(num_players=2, seed=42)

        # Give player 0 some properties
        game.state.property_manager.properties[1].owner = 0  # Mediterranean
        game.state.property_manager.properties[3].owner = 0  # Baltic
        game.state.players[0].money = -100  # In debt

        # Declare bankruptcy to bank
        game.handle_bankruptcy(player_id=0, creditor_id=None)

        # Properties should be back to unowned
        assert game.state.property_manager.properties[1].owner is None, "Property should return to bank"
        assert game.state.property_manager.properties[3].owner is None, "Property should return to bank"
        assert not game.state.property_manager.properties[1].mortgaged, "Property should be unmortgaged"
        assert game.state.players[0].bankrupt, "Player should be marked bankrupt"

    def test_bankruptcy_to_player_transfers_assets(self) -> None:
        """Test that bankruptcy to another player transfers all assets."""
        game = MonopolyGame(num_players=2, seed=42)

        # Give player 0 some properties and money
        game.state.property_manager.properties[1].owner = 0  # Mediterranean
        game.state.property_manager.properties[3].owner = 0  # Baltic
        game.state.players[0].money = 100

        # Declare bankruptcy to player 1
        game.handle_bankruptcy(player_id=0, creditor_id=1)

        # Player 1 should receive properties and money
        assert game.state.property_manager.properties[1].owner == 1, "Property should transfer to creditor"
        assert game.state.property_manager.properties[3].owner == 1, "Property should transfer to creditor"
        assert game.state.players[1].money == 1600, "Creditor should receive money (1500 + 100)"
        assert game.state.players[0].bankrupt, "Player should be marked bankrupt"


class TestTaxes:
    """Test tax mechanics."""

    def test_income_tax_is_200(self) -> None:
        """Test that Income Tax is $200."""
        from monopoly_engine.types import INCOME_TAX_AMOUNT

        assert INCOME_TAX_AMOUNT == 200, "Income tax should be $200"

    def test_luxury_tax_is_100(self) -> None:
        """Test that Luxury Tax is $100."""
        from monopoly_engine.types import LUXURY_TAX_AMOUNT

        assert LUXURY_TAX_AMOUNT == 100, "Luxury tax should be $100"


class TestInventoryManagement:
    """Test house and hotel inventory management."""

    def test_initial_house_count_is_32(self) -> None:
        """Test that game starts with 32 houses."""
        game = MonopolyGame(num_players=2, seed=42)
        assert game.state.houses_remaining == 32, "Should start with 32 houses"

    def test_initial_hotel_count_is_12(self) -> None:
        """Test that game starts with 12 hotels."""
        game = MonopolyGame(num_players=2, seed=42)
        assert game.state.hotels_remaining == 12, "Should start with 12 hotels"

    def test_building_house_decrements_inventory(self) -> None:
        """Test that building a house decrements inventory."""
        game = MonopolyGame(num_players=2, seed=42)

        # Give player monopoly
        game.state.property_manager.properties[1].owner = 0
        game.state.property_manager.properties[3].owner = 0
        game.state.players[0].money = 1000

        initial_houses = game.state.houses_remaining

        # Build house
        action = BuildHouse(player_id=0, property_id=1)
        action.execute(game)

        assert game.state.houses_remaining == initial_houses - 1, "House inventory should decrease"

    def test_selling_house_increments_inventory(self) -> None:
        """Test that selling a house returns it to inventory."""
        game = MonopolyGame(num_players=2, seed=42)

        # Give player monopoly with a house
        game.state.property_manager.properties[1].owner = 0
        game.state.property_manager.properties[3].owner = 0
        game.state.property_manager.properties[1].houses = 1
        game.state.players[0].money = 1000

        initial_houses = game.state.houses_remaining

        # Sell house
        from monopoly_engine.actions import SellHouse

        action = SellHouse(player_id=0, property_id=1)
        action.execute(game)

        assert game.state.houses_remaining == initial_houses + 1, "House inventory should increase"
