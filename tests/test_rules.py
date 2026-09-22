"""Tests for rules.py module."""

from monopoly_engine import (
    PROPERTY_GROUPS,
    Player,
    PropertyColor,
    PropertyManager,
    calculate_net_worth,
    calculate_rent,
    can_afford_rent,
    can_build_house,
    can_buy_property,
    can_mortgage_property,
    can_sell_house,
    get_buildable_properties,
    get_mortgage_value,
    get_mortgageable_properties,
    get_property_cost,
    get_sellable_houses,
    get_unmortgage_cost,
    get_unmortgageable_properties,
    is_bankrupt,
    validate_trade,
)


class TestCalculateRent:
    """Tests for rent calculation."""

    def test_rent_unowned_property(self, property_manager: PropertyManager) -> None:
        """Unowned property should have 0 rent."""
        rent = calculate_rent(property_manager, 1)
        assert rent == 0

    def test_rent_mortgaged_property(self, property_manager: PropertyManager) -> None:
        """Mortgaged property should have 0 rent."""
        property_manager.properties[1].owner = 0
        property_manager.properties[1].mortgaged = True
        rent = calculate_rent(property_manager, 1)
        assert rent == 0

    def test_rent_basic_property(self, property_manager: PropertyManager) -> None:
        """Property without monopoly should have base rent."""
        property_manager.properties[1].owner = 0
        rent = calculate_rent(property_manager, 1)
        assert rent == 2  # Mediterranean base rent

    def test_rent_monopoly_no_houses(self, property_manager_with_owner: PropertyManager) -> None:
        """Monopoly without houses should have double rent."""
        rent = calculate_rent(property_manager_with_owner, 1)
        assert rent == 4  # Mediterranean base rent * 2

    def test_rent_with_houses(self, property_manager_with_houses: PropertyManager) -> None:
        """Property with houses should use house rent."""
        # Light blue with 2 houses
        rent = calculate_rent(property_manager_with_houses, 6)
        assert rent == 90  # Oriental with 2 houses

    def test_rent_with_hotel(self, property_manager_with_monopoly: PropertyManager) -> None:
        """Property with hotel should use hotel rent."""
        property_manager_with_monopoly.properties[6].houses = 5
        rent = calculate_rent(property_manager_with_monopoly, 6)
        assert rent == 550  # Oriental hotel rent

    def test_rent_railroad_one_owned(self, property_manager: PropertyManager) -> None:
        """Railroad rent with 1 owned should be $25."""
        property_manager.properties[5].owner = 0
        rent = calculate_rent(property_manager, 5)
        assert rent == 25

    def test_rent_railroad_all_owned(self, property_manager: PropertyManager) -> None:
        """Railroad rent with all 4 owned should be $200."""
        for pos in [5, 15, 25, 35]:
            property_manager.properties[pos].owner = 0
        rent = calculate_rent(property_manager, 5)
        assert rent == 200

    def test_rent_railroad_from_card(self, property_manager: PropertyManager) -> None:
        """Railroad rent from Chance card should be double."""
        property_manager.properties[5].owner = 0
        rent = calculate_rent(property_manager, 5, is_from_card=True)
        assert rent == 50  # $25 * 2

    def test_rent_utility_one_owned(self, property_manager: PropertyManager) -> None:
        """Utility rent with 1 owned should be 4x dice roll."""
        property_manager.properties[12].owner = 0
        rent = calculate_rent(property_manager, 12, dice_roll=7)
        assert rent == 28  # 4 * 7

    def test_rent_utility_both_owned(self, property_manager: PropertyManager) -> None:
        """Utility rent with both owned should be 10x dice roll."""
        property_manager.properties[12].owner = 0
        property_manager.properties[28].owner = 0
        rent = calculate_rent(property_manager, 12, dice_roll=7)
        assert rent == 70  # 10 * 7

    def test_rent_utility_from_card(self, property_manager: PropertyManager) -> None:
        """Utility rent from Chance card should be 10x dice roll."""
        property_manager.properties[12].owner = 0
        rent = calculate_rent(property_manager, 12, dice_roll=7, is_from_card=True)
        assert rent == 70  # 10 * 7 regardless of how many owned


class TestCanBuyProperty:
    """Tests for property purchase validation."""

    def test_can_buy_unowned(
        self, player_with_money: Player, property_manager: PropertyManager
    ) -> None:
        """Should be able to buy unowned property."""
        can, reason = can_buy_property(player_with_money, property_manager, 1)
        assert can
        assert reason == ""

    def test_cannot_buy_owned(
        self, player_with_money: Player, property_manager: PropertyManager
    ) -> None:
        """Should not be able to buy owned property."""
        property_manager.properties[1].owner = 1
        can, reason = can_buy_property(player_with_money, property_manager, 1)
        assert not can
        assert "owned" in reason.lower()

    def test_cannot_buy_non_property(
        self, player_with_money: Player, property_manager: PropertyManager
    ) -> None:
        """Should not be able to buy non-property space."""
        can, reason = can_buy_property(player_with_money, property_manager, 0)
        assert not can

    def test_cannot_buy_without_funds(
        self, player: Player, property_manager: PropertyManager
    ) -> None:
        """Should not be able to buy without sufficient funds."""
        player.money = 50  # Mediterranean costs $60
        can, reason = can_buy_property(player, property_manager, 1)
        assert not can
        assert "funds" in reason.lower()


class TestGetPropertyCost:
    """Tests for property cost lookup."""

    def test_property_cost(self) -> None:
        """Should return correct property cost."""
        assert get_property_cost(1) == 60  # Mediterranean
        assert get_property_cost(39) == 400  # Boardwalk

    def test_railroad_cost(self) -> None:
        """Should return correct railroad cost."""
        assert get_property_cost(5) == 200

    def test_utility_cost(self) -> None:
        """Should return correct utility cost."""
        assert get_property_cost(12) == 150

    def test_non_property_cost(self) -> None:
        """Non-properties should return 0."""
        assert get_property_cost(0) == 0  # GO
        assert get_property_cost(7) == 0  # Chance


class TestCanBuildHouse:
    """Tests for house building validation."""

    def test_can_build_with_monopoly(
        self,
        player_with_money: Player,
        property_manager_with_monopoly: PropertyManager,
    ) -> None:
        """Should be able to build with monopoly."""
        can, reason = can_build_house(player_with_money, property_manager_with_monopoly, 6, 32, 12)
        assert can

    def test_cannot_build_without_monopoly(
        self, player_with_money: Player, property_manager: PropertyManager
    ) -> None:
        """Should not be able to build without monopoly."""
        property_manager.properties[6].owner = player_with_money.id
        can, reason = can_build_house(player_with_money, property_manager, 6, 32, 12)
        assert not can
        assert "monopoly" in reason.lower()

    def test_cannot_build_on_mortgaged(
        self,
        player_with_money: Player,
        property_manager_with_monopoly: PropertyManager,
    ) -> None:
        """Should not be able to build on mortgaged property."""
        property_manager_with_monopoly.properties[6].mortgaged = True
        can, reason = can_build_house(player_with_money, property_manager_with_monopoly, 6, 32, 12)
        assert not can
        assert "mortgaged" in reason.lower()

    def test_cannot_build_unevenly(
        self,
        player_with_money: Player,
        property_manager_with_monopoly: PropertyManager,
    ) -> None:
        """Should not be able to build unevenly."""
        property_manager_with_monopoly.properties[6].houses = 1
        can, reason = can_build_house(player_with_money, property_manager_with_monopoly, 6, 32, 12)
        assert not can
        assert "evenly" in reason.lower()

    def test_cannot_build_without_houses(
        self,
        player_with_money: Player,
        property_manager_with_monopoly: PropertyManager,
    ) -> None:
        """Should not be able to build when no houses available."""
        can, reason = can_build_house(player_with_money, property_manager_with_monopoly, 6, 0, 12)
        assert not can
        assert "houses" in reason.lower()

    def test_cannot_build_hotel_without_hotels(
        self,
        player_with_money: Player,
        property_manager_with_houses: PropertyManager,
    ) -> None:
        """Should not be able to build hotel when none available."""
        # Set all to 4 houses
        for pos in PROPERTY_GROUPS[PropertyColor.LIGHT_BLUE]:
            property_manager_with_houses.properties[pos].houses = 4
        can, reason = can_build_house(player_with_money, property_manager_with_houses, 6, 32, 0)
        assert not can
        assert "hotel" in reason.lower()


class TestCanSellHouse:
    """Tests for house selling validation."""

    def test_can_sell_house(
        self,
        player: Player,
        property_manager_with_houses: PropertyManager,
    ) -> None:
        """Should be able to sell house when evenly built."""
        player.id = 0
        can, reason = can_sell_house(player, property_manager_with_houses, 6)
        assert can

    def test_cannot_sell_no_houses(
        self,
        player: Player,
        property_manager_with_monopoly: PropertyManager,
    ) -> None:
        """Should not be able to sell when no houses."""
        player.id = 0
        can, reason = can_sell_house(player, property_manager_with_monopoly, 6)
        assert not can
        assert "houses" in reason.lower()

    def test_cannot_sell_unevenly(
        self,
        player: Player,
        property_manager_with_houses: PropertyManager,
    ) -> None:
        """Should not be able to sell unevenly."""
        player.id = 0
        # Make one property have more houses
        property_manager_with_houses.properties[6].houses = 1
        can, reason = can_sell_house(player, property_manager_with_houses, 6)
        assert not can
        assert "evenly" in reason.lower()


class TestMortgage:
    """Tests for mortgage functionality."""

    def test_can_mortgage(self, player: Player, property_manager: PropertyManager) -> None:
        """Should be able to mortgage owned property."""
        property_manager.properties[1].owner = player.id
        can, reason = can_mortgage_property(player, property_manager, 1)
        assert can

    def test_cannot_mortgage_with_houses(
        self,
        player: Player,
        property_manager_with_houses: PropertyManager,
    ) -> None:
        """Should not be able to mortgage with houses."""
        player.id = 0
        can, reason = can_mortgage_property(player, property_manager_with_houses, 6)
        assert not can
        assert "houses" in reason.lower()

    def test_cannot_mortgage_already_mortgaged(
        self, player: Player, property_manager: PropertyManager
    ) -> None:
        """Should not be able to mortgage already mortgaged property."""
        property_manager.properties[1].owner = player.id
        property_manager.properties[1].mortgaged = True
        can, reason = can_mortgage_property(player, property_manager, 1)
        assert not can
        assert "mortgaged" in reason.lower()

    def test_mortgage_value(self) -> None:
        """Mortgage value should be correct."""
        assert get_mortgage_value(1) == 30  # Mediterranean
        assert get_mortgage_value(5) == 100  # Railroad
        assert get_mortgage_value(12) == 75  # Utility

    def test_unmortgage_cost(self) -> None:
        """Unmortgage cost should be 110% of mortgage value."""
        assert get_unmortgage_cost(1) == 33  # 30 * 1.1 = 33
        assert get_unmortgage_cost(5) == 110  # 100 * 1.1 = 110


class TestNetWorth:
    """Tests for net worth calculation."""

    def test_net_worth_cash_only(self, player: Player, property_manager: PropertyManager) -> None:
        """Net worth with no properties should be cash."""
        worth = calculate_net_worth(player, property_manager)
        assert worth == player.money

    def test_net_worth_with_property(
        self, player: Player, property_manager: PropertyManager
    ) -> None:
        """Net worth should include property mortgage value."""
        property_manager.properties[1].owner = player.id
        worth = calculate_net_worth(player, property_manager)
        assert worth == player.money + 30  # Mediterranean mortgage value

    def test_net_worth_mortgaged_property(
        self, player: Player, property_manager: PropertyManager
    ) -> None:
        """Mortgaged property should not add to net worth."""
        property_manager.properties[1].owner = player.id
        property_manager.properties[1].mortgaged = True
        worth = calculate_net_worth(player, property_manager)
        assert worth == player.money

    def test_net_worth_with_houses(
        self, player: Player, property_manager_with_houses: PropertyManager
    ) -> None:
        """Net worth should include house sale value."""
        player.id = 0
        worth = calculate_net_worth(player, property_manager_with_houses)
        # Cash + mortgage values + house values
        # 6 houses at $50/2 = $150 sale value
        assert worth > player.money


class TestBankruptcy:
    """Tests for bankruptcy checks."""

    def test_can_afford_with_cash(self, player: Player, property_manager: PropertyManager) -> None:
        """Should be able to afford rent with cash."""
        can, shortfall = can_afford_rent(player, property_manager, 100)
        assert can
        assert shortfall == 0

    def test_cannot_afford_exceeds_net_worth(
        self, player: Player, property_manager: PropertyManager
    ) -> None:
        """Should not be able to afford if exceeds net worth."""
        can, shortfall = can_afford_rent(player, property_manager, 10000)
        assert not can
        assert shortfall > 0

    def test_is_bankrupt(self, player: Player, property_manager: PropertyManager) -> None:
        """is_bankrupt should check net worth vs debt."""
        assert not is_bankrupt(player, property_manager, 100)
        assert is_bankrupt(player, property_manager, 10000)


class TestValidateTrade:
    """Tests for trade validation."""

    def test_valid_trade(
        self, two_players: list[Player], property_manager: PropertyManager
    ) -> None:
        """Should validate a proper trade."""
        property_manager.properties[1].owner = 0
        property_manager.properties[3].owner = 1
        valid, reason = validate_trade(
            two_players[0],
            two_players[1],
            property_manager,
            give_properties=[1],
            give_money=0,
            want_properties=[3],
            want_money=0,
        )
        assert valid

    def test_trade_with_houses_invalid(
        self, two_players: list[Player], property_manager: PropertyManager
    ) -> None:
        """Should reject trade of property with houses."""
        property_manager.properties[1].owner = 0
        property_manager.properties[1].houses = 1
        valid, reason = validate_trade(
            two_players[0],
            two_players[1],
            property_manager,
            give_properties=[1],
            give_money=0,
            want_properties=[],
            want_money=0,
        )
        assert not valid
        assert "houses" in reason.lower()

    def test_trade_insufficient_funds(
        self, two_players: list[Player], property_manager: PropertyManager
    ) -> None:
        """Should reject trade when insufficient funds."""
        two_players[0].money = 10
        valid, reason = validate_trade(
            two_players[0],
            two_players[1],
            property_manager,
            give_properties=[],
            give_money=100,
            want_properties=[],
            want_money=0,
        )
        assert not valid
        assert "funds" in reason.lower()


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_buildable_properties(
        self,
        player_with_money: Player,
        property_manager_with_monopoly: PropertyManager,
    ) -> None:
        """get_buildable_properties should list valid positions."""
        player_with_money.id = 0
        buildable = get_buildable_properties(
            player_with_money, property_manager_with_monopoly, 32, 12
        )
        # Should be able to build on all 3 light blue properties
        assert len(buildable) == 3

    def test_get_sellable_houses(
        self, player: Player, property_manager_with_houses: PropertyManager
    ) -> None:
        """get_sellable_houses should list valid positions."""
        player.id = 0
        sellable = get_sellable_houses(player, property_manager_with_houses)
        assert len(sellable) == 3

    def test_get_mortgageable_properties(
        self, player: Player, property_manager: PropertyManager
    ) -> None:
        """get_mortgageable_properties should list valid positions."""
        property_manager.properties[1].owner = player.id
        property_manager.properties[5].owner = player.id
        mortgageable = get_mortgageable_properties(player, property_manager)
        assert len(mortgageable) == 2

    def test_get_unmortgageable_properties(
        self, player_with_money: Player, property_manager: PropertyManager
    ) -> None:
        """get_unmortgageable_properties should list valid positions."""
        property_manager.properties[1].owner = player_with_money.id
        property_manager.properties[1].mortgaged = True
        unmortgageable = get_unmortgageable_properties(player_with_money, property_manager)
        assert 1 in unmortgageable
