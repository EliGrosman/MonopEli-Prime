"""
Comprehensive tests for the board module.

This module tests the board layout, space definitions, property groups,
and utility methods for accessing board information.
"""

import pytest
from monopoly_engine.board import (
    Board,
    Space,
    PropertySpace,
    RailroadSpace,
    UtilitySpace,
    TaxSpace,
)
from monopoly_engine.types import SpaceType, PropertyColor


# =============================================================================
# Space Class Tests
# =============================================================================


class TestSpace:
    """Test the base Space class."""

    def test_space_creation(self) -> None:
        """Verify Space can be created with required attributes."""
        space = Space(position=0, name="GO", type=SpaceType.GO)
        assert space.position == 0
        assert space.name == "GO"
        assert space.type == SpaceType.GO

    def test_space_immutability(self) -> None:
        """Verify Space is immutable (frozen dataclass)."""
        space = Space(position=0, name="GO", type=SpaceType.GO)
        with pytest.raises(AttributeError):
            space.position = 1  # type: ignore


class TestPropertySpace:
    """Test the PropertySpace class."""

    def test_property_space_creation(self) -> None:
        """Verify PropertySpace can be created with all attributes."""
        space = PropertySpace(
            position=1,
            name="Mediterranean Avenue",
            type=SpaceType.PROPERTY,
            color=PropertyColor.BROWN,
            cost=60,
            rent=(2, 10, 30, 90, 160, 250),
            house_cost=50,
            mortgage_value=30,
        )
        assert space.position == 1
        assert space.name == "Mediterranean Avenue"
        assert space.type == SpaceType.PROPERTY
        assert space.color == PropertyColor.BROWN
        assert space.cost == 60
        assert space.rent == (2, 10, 30, 90, 160, 250)
        assert space.house_cost == 50
        assert space.mortgage_value == 30

    def test_property_space_rent_tuple(self) -> None:
        """Verify rent is stored as a tuple with 6 values."""
        space = Board.get_space(1)
        assert isinstance(space, PropertySpace)
        assert len(space.rent) == 6
        assert space.rent[0] < space.rent[1] < space.rent[5]

    def test_property_space_immutability(self) -> None:
        """Verify PropertySpace is immutable."""
        space = Board.get_space(1)
        assert isinstance(space, PropertySpace)
        with pytest.raises(AttributeError):
            space.cost = 100  # type: ignore


class TestRailroadSpace:
    """Test the RailroadSpace class."""

    def test_railroad_space_creation(self) -> None:
        """Verify RailroadSpace can be created with all attributes."""
        space = RailroadSpace(
            position=5,
            name="Reading Railroad",
            type=SpaceType.RAILROAD,
            cost=200,
            rent=(25, 50, 100, 200),
            mortgage_value=100,
        )
        assert space.position == 5
        assert space.name == "Reading Railroad"
        assert space.type == SpaceType.RAILROAD
        assert space.cost == 200
        assert space.rent == (25, 50, 100, 200)
        assert space.mortgage_value == 100

    def test_railroad_rent_tuple(self) -> None:
        """Verify railroad rent is stored as a tuple with 4 values."""
        space = Board.get_space(5)
        assert isinstance(space, RailroadSpace)
        assert len(space.rent) == 4
        assert space.rent[0] == 25
        assert space.rent[1] == 50
        assert space.rent[2] == 100
        assert space.rent[3] == 200


class TestUtilitySpace:
    """Test the UtilitySpace class."""

    def test_utility_space_creation(self) -> None:
        """Verify UtilitySpace can be created with all attributes."""
        space = UtilitySpace(
            position=12,
            name="Electric Company",
            type=SpaceType.UTILITY,
            cost=150,
            mortgage_value=75,
        )
        assert space.position == 12
        assert space.name == "Electric Company"
        assert space.type == SpaceType.UTILITY
        assert space.cost == 150
        assert space.mortgage_value == 75

    def test_utility_has_no_rent_tuple(self) -> None:
        """Verify utilities don't have a rent tuple (calculated dynamically)."""
        space = Board.get_space(12)
        assert isinstance(space, UtilitySpace)
        # Utilities don't have a 'rent' attribute
        assert not hasattr(space, "rent")


class TestTaxSpace:
    """Test the TaxSpace class."""

    def test_tax_space_creation(self) -> None:
        """Verify TaxSpace can be created with all attributes."""
        space = TaxSpace(
            position=4,
            name="Income Tax",
            type=SpaceType.TAX,
            amount=200,
        )
        assert space.position == 4
        assert space.name == "Income Tax"
        assert space.type == SpaceType.TAX
        assert space.amount == 200

    def test_both_tax_spaces(self) -> None:
        """Verify both tax spaces have correct amounts."""
        income_tax = Board.get_space(4)
        assert isinstance(income_tax, TaxSpace)
        assert income_tax.amount == 200

        luxury_tax = Board.get_space(38)
        assert isinstance(luxury_tax, TaxSpace)
        assert luxury_tax.amount == 100


# =============================================================================
# Board Class Tests
# =============================================================================


class TestBoardStructure:
    """Test the Board class structure and layout."""

    def test_board_has_40_spaces(self) -> None:
        """Verify the board has exactly 40 spaces."""
        assert len(Board.SPACES) == 40

    def test_all_positions_sequential(self) -> None:
        """Verify all positions are sequential from 0 to 39."""
        for i, space in enumerate(Board.SPACES):
            assert space.position == i

    def test_board_spaces_immutable(self) -> None:
        """Verify Board.SPACES tuple is immutable."""
        with pytest.raises(TypeError):
            Board.SPACES[0] = Space(0, "Test", SpaceType.GO)  # type: ignore

    def test_corner_spaces(self) -> None:
        """Verify the four corner spaces are correct."""
        go = Board.get_space(0)
        assert go.name == "GO"
        assert go.type == SpaceType.GO

        jail = Board.get_space(10)
        assert jail.name == "Just Visiting"
        assert jail.type == SpaceType.JAIL

        free_parking = Board.get_space(20)
        assert free_parking.name == "Free Parking"
        assert free_parking.type == SpaceType.FREE_PARKING

        go_to_jail = Board.get_space(30)
        assert go_to_jail.name == "Go to Jail"
        assert go_to_jail.type == SpaceType.GO_TO_JAIL


class TestBoardSpaceCounts:
    """Test that the board has the correct number of each space type."""

    def test_property_count(self) -> None:
        """Verify there are 22 street properties."""
        properties = [s for s in Board.SPACES if isinstance(s, PropertySpace)]
        assert len(properties) == 22

    def test_railroad_count(self) -> None:
        """Verify there are 4 railroads."""
        railroads = [s for s in Board.SPACES if isinstance(s, RailroadSpace)]
        assert len(railroads) == 4

    def test_utility_count(self) -> None:
        """Verify there are 2 utilities."""
        utilities = [s for s in Board.SPACES if isinstance(s, UtilitySpace)]
        assert len(utilities) == 2

    def test_chance_count(self) -> None:
        """Verify there are 3 Chance spaces."""
        chances = [s for s in Board.SPACES if s.type == SpaceType.CHANCE]
        assert len(chances) == 3
        assert all(c.position in [7, 22, 36] for c in chances)

    def test_community_chest_count(self) -> None:
        """Verify there are 3 Community Chest spaces."""
        chests = [s for s in Board.SPACES if s.type == SpaceType.COMMUNITY_CHEST]
        assert len(chests) == 3
        assert all(c.position in [2, 17, 33] for c in chests)

    def test_tax_count(self) -> None:
        """Verify there are 2 tax spaces."""
        taxes = [s for s in Board.SPACES if isinstance(s, TaxSpace)]
        assert len(taxes) == 2

    def test_total_buyable_properties(self) -> None:
        """Verify there are 28 total buyable properties (22 + 4 + 2)."""
        buyable = [s for s in Board.SPACES if hasattr(s, "cost")]
        assert len(buyable) == 28


class TestBoardGetSpace:
    """Test the Board.get_space() method."""

    def test_get_space_valid_position(self) -> None:
        """Verify get_space returns correct space for valid positions."""
        space = Board.get_space(0)
        assert space.name == "GO"
        assert space.position == 0

        space = Board.get_space(39)
        assert space.name == "Boardwalk"
        assert space.position == 39

    def test_get_space_wraps_around(self) -> None:
        """Verify get_space wraps around for positions >= 40."""
        space_0 = Board.get_space(0)
        space_40 = Board.get_space(40)
        assert space_0.position == space_40.position
        assert space_0.name == space_40.name

        space_1 = Board.get_space(1)
        space_41 = Board.get_space(41)
        assert space_1.position == space_41.position

        space_39 = Board.get_space(39)
        space_79 = Board.get_space(79)
        assert space_39.position == space_79.position

    def test_get_space_large_position(self) -> None:
        """Verify get_space handles very large positions."""
        space = Board.get_space(1000)
        expected_pos = 1000 % 40
        assert space.position == expected_pos


class TestPropertyGroups:
    """Test property color groups and Board.get_property_group()."""

    def test_brown_properties(self) -> None:
        """Verify brown property group has 2 properties."""
        brown = Board.get_property_group(PropertyColor.BROWN)
        assert brown == [1, 3]
        assert len(brown) == 2

        # Verify they're actually brown
        for pos in brown:
            space = Board.get_space(pos)
            assert isinstance(space, PropertySpace)
            assert space.color == PropertyColor.BROWN

    def test_light_blue_properties(self) -> None:
        """Verify light blue property group has 3 properties."""
        light_blue = Board.get_property_group(PropertyColor.LIGHT_BLUE)
        assert light_blue == [6, 8, 9]
        assert len(light_blue) == 3

    def test_magenta_properties(self) -> None:
        """Verify magenta property group has 3 properties."""
        magenta = Board.get_property_group(PropertyColor.MAGENTA)
        assert magenta == [11, 13, 14]
        assert len(magenta) == 3

    def test_orange_properties(self) -> None:
        """Verify orange property group has 3 properties."""
        orange = Board.get_property_group(PropertyColor.ORANGE)
        assert orange == [16, 18, 19]
        assert len(orange) == 3

    def test_red_properties(self) -> None:
        """Verify red property group has 3 properties."""
        red = Board.get_property_group(PropertyColor.RED)
        assert red == [21, 23, 24]
        assert len(red) == 3

    def test_yellow_properties(self) -> None:
        """Verify yellow property group has 3 properties."""
        yellow = Board.get_property_group(PropertyColor.YELLOW)
        assert yellow == [26, 27, 29]
        assert len(yellow) == 3

    def test_green_properties(self) -> None:
        """Verify green property group has 3 properties."""
        green = Board.get_property_group(PropertyColor.GREEN)
        assert green == [31, 32, 34]
        assert len(green) == 3

    def test_blue_properties(self) -> None:
        """Verify blue property group has 2 properties."""
        blue = Board.get_property_group(PropertyColor.BLUE)
        assert blue == [37, 39]
        assert len(blue) == 2

    def test_all_property_groups_have_correct_counts(self) -> None:
        """Verify all color groups have expected number of properties."""
        # Brown and Blue have 2 properties
        assert len(Board.get_property_group(PropertyColor.BROWN)) == 2
        assert len(Board.get_property_group(PropertyColor.BLUE)) == 2

        # All others have 3 properties
        for color in [
            PropertyColor.LIGHT_BLUE,
            PropertyColor.MAGENTA,
            PropertyColor.ORANGE,
            PropertyColor.RED,
            PropertyColor.YELLOW,
            PropertyColor.GREEN,
        ]:
            assert len(Board.get_property_group(color)) == 3

    def test_property_groups_sorted(self) -> None:
        """Verify property group positions are returned in sorted order."""
        for color in [
            PropertyColor.BROWN,
            PropertyColor.LIGHT_BLUE,
            PropertyColor.MAGENTA,
            PropertyColor.ORANGE,
            PropertyColor.RED,
            PropertyColor.YELLOW,
            PropertyColor.GREEN,
            PropertyColor.BLUE,
        ]:
            group = Board.get_property_group(color)
            assert group == sorted(group)


class TestRailroadsAndUtilities:
    """Test railroad and utility groups."""

    def test_get_railroads(self) -> None:
        """Verify get_railroads returns all 4 railroad positions."""
        railroads = Board.get_railroads()
        assert railroads == [5, 15, 25, 35]
        assert len(railroads) == 4

        # Verify they're actually railroads
        for pos in railroads:
            space = Board.get_space(pos)
            assert isinstance(space, RailroadSpace)

    def test_railroad_names(self) -> None:
        """Verify railroad names are correct."""
        names = [
            "Reading Railroad",
            "Pennsylvania Railroad",
            "B & O Railroad",
            "Short Line",
        ]
        for pos, name in zip([5, 15, 25, 35], names):
            space = Board.get_space(pos)
            assert isinstance(space, RailroadSpace)
            assert space.name == name

    def test_get_utilities(self) -> None:
        """Verify get_utilities returns both utility positions."""
        utilities = Board.get_utilities()
        assert utilities == [12, 28]
        assert len(utilities) == 2

        # Verify they're actually utilities
        for pos in utilities:
            space = Board.get_space(pos)
            assert isinstance(space, UtilitySpace)

    def test_utility_names(self) -> None:
        """Verify utility names are correct."""
        electric = Board.get_space(12)
        assert isinstance(electric, UtilitySpace)
        assert electric.name == "Electric Company"

        water = Board.get_space(28)
        assert isinstance(water, UtilitySpace)
        assert water.name == "Water Works"

    def test_railroads_as_property_group(self) -> None:
        """Verify railroads don't appear in get_property_group for color groups."""
        # Railroads shouldn't be in any street property color group
        for color in [
            PropertyColor.BROWN,
            PropertyColor.LIGHT_BLUE,
            PropertyColor.MAGENTA,
            PropertyColor.ORANGE,
            PropertyColor.RED,
            PropertyColor.YELLOW,
            PropertyColor.GREEN,
            PropertyColor.BLUE,
        ]:
            group = Board.get_property_group(color)
            for pos in group:
                space = Board.get_space(pos)
                assert isinstance(space, PropertySpace)
                assert not isinstance(space, RailroadSpace)


# =============================================================================
# Property Details Tests
# =============================================================================


class TestPropertyPrices:
    """Test that property prices are correct."""

    def test_brown_properties_cheap(self) -> None:
        """Verify brown properties are the cheapest."""
        med_ave = Board.get_space(1)
        baltic = Board.get_space(3)
        assert isinstance(med_ave, PropertySpace)
        assert isinstance(baltic, PropertySpace)
        assert med_ave.cost == 60
        assert baltic.cost == 60

    def test_blue_properties_expensive(self) -> None:
        """Verify blue properties are the most expensive."""
        park_place = Board.get_space(37)
        boardwalk = Board.get_space(39)
        assert isinstance(park_place, PropertySpace)
        assert isinstance(boardwalk, PropertySpace)
        assert park_place.cost == 350
        assert boardwalk.cost == 400

    def test_railroad_prices_uniform(self) -> None:
        """Verify all railroads cost $200."""
        for pos in Board.get_railroads():
            railroad = Board.get_space(pos)
            assert isinstance(railroad, RailroadSpace)
            assert railroad.cost == 200

    def test_utility_prices_uniform(self) -> None:
        """Verify both utilities cost $150."""
        for pos in Board.get_utilities():
            utility = Board.get_space(pos)
            assert isinstance(utility, UtilitySpace)
            assert utility.cost == 150


class TestPropertyRent:
    """Test property rent values."""

    def test_rent_increases_with_houses(self) -> None:
        """Verify rent increases with each house."""
        for i in range(1, 40):
            space = Board.get_space(i)
            if isinstance(space, PropertySpace):
                # Rent should increase with each house
                assert space.rent[0] < space.rent[1] < space.rent[2]
                assert space.rent[2] < space.rent[3] < space.rent[4]
                assert space.rent[4] < space.rent[5]

    def test_railroad_rent_progression(self) -> None:
        """Verify railroad rent doubles with each additional railroad."""
        railroad = Board.get_space(5)
        assert isinstance(railroad, RailroadSpace)
        assert railroad.rent[0] == 25
        assert railroad.rent[1] == 50
        assert railroad.rent[2] == 100
        assert railroad.rent[3] == 200

    def test_brown_rent_values(self) -> None:
        """Verify specific rent values for brown properties."""
        med_ave = Board.get_space(1)
        assert isinstance(med_ave, PropertySpace)
        assert med_ave.rent[0] == 2  # Base rent

        baltic = Board.get_space(3)
        assert isinstance(baltic, PropertySpace)
        assert baltic.rent[0] == 4  # Base rent


class TestMortgageValues:
    """Test property mortgage values."""

    def test_mortgage_half_of_cost(self) -> None:
        """Verify mortgage values are half the cost for properties."""
        for i in range(1, 40):
            space = Board.get_space(i)
            if isinstance(space, PropertySpace):
                assert space.mortgage_value == space.cost // 2

    def test_railroad_mortgage_values(self) -> None:
        """Verify railroad mortgage values."""
        for pos in Board.get_railroads():
            railroad = Board.get_space(pos)
            assert isinstance(railroad, RailroadSpace)
            assert railroad.mortgage_value == 100

    def test_utility_mortgage_values(self) -> None:
        """Verify utility mortgage values."""
        for pos in Board.get_utilities():
            utility = Board.get_space(pos)
            assert isinstance(utility, UtilitySpace)
            assert utility.mortgage_value == 75


class TestHouseCosts:
    """Test house building costs."""

    def test_brown_and_light_blue_house_costs(self) -> None:
        """Verify cheapest properties have $50 house cost."""
        for pos in [1, 3, 6, 8, 9]:
            space = Board.get_space(pos)
            assert isinstance(space, PropertySpace)
            assert space.house_cost == 50

    def test_green_and_blue_house_costs(self) -> None:
        """Verify most expensive properties have $200 house cost."""
        for pos in [31, 32, 34, 37, 39]:
            space = Board.get_space(pos)
            assert isinstance(space, PropertySpace)
            assert space.house_cost == 200

    def test_house_costs_increase_with_property_value(self) -> None:
        """Verify house costs generally increase with property cost."""
        # Check a few specific examples
        brown = Board.get_space(1)
        assert isinstance(brown, PropertySpace)
        assert brown.house_cost == 50

        red = Board.get_space(21)
        assert isinstance(red, PropertySpace)
        assert red.house_cost == 150

        blue = Board.get_space(39)
        assert isinstance(blue, PropertySpace)
        assert blue.house_cost == 200


# =============================================================================
# Edge Cases and Validation
# =============================================================================


class TestBoardEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_negative_position_wraps(self) -> None:
        """Verify negative positions wrap around correctly."""
        # Python modulo handles this naturally
        space = Board.get_space(-1)
        assert space.position == 39

        space = Board.get_space(-40)
        assert space.position == 0

    def test_all_spaces_have_names(self) -> None:
        """Verify all spaces have non-empty names."""
        for space in Board.SPACES:
            assert space.name
            assert len(space.name) > 0

    def test_all_buyable_spaces_have_positive_costs(self) -> None:
        """Verify all buyable properties have positive costs."""
        for space in Board.SPACES:
            if hasattr(space, "cost"):
                assert space.cost > 0

    def test_no_duplicate_positions(self) -> None:
        """Verify no two spaces have the same position."""
        positions = [space.position for space in Board.SPACES]
        assert len(positions) == len(set(positions))

    def test_property_group_returns_empty_for_invalid_color(self) -> None:
        """Verify property group handling for special colors."""
        # RAILROAD and UTILITY are special - they're not street properties
        # but get_property_group still works on them
        railroads = Board.get_property_group(PropertyColor.RAILROAD)
        assert len(railroads) == 0  # PropertySpace only, not RailroadSpace

        utilities = Board.get_property_group(PropertyColor.UTILITY)
        assert len(utilities) == 0  # PropertySpace only, not UtilitySpace
