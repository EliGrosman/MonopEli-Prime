"""Tests for board.py module."""

from monopoly_engine import (
    Board,
    PropertyColor,
    PropertySpace,
    RailroadSpace,
    SpaceType,
    TaxSpace,
    UtilitySpace,
)


class TestBoard:
    """Tests for Board class."""

    def test_board_has_40_spaces(self) -> None:
        """Board should have exactly 40 spaces."""
        assert len(Board.SPACES) == 40

    def test_go_is_first_space(self) -> None:
        """GO should be at position 0."""
        space = Board.get_space(0)
        assert space.name == "GO"
        assert space.space_type == SpaceType.GO

    def test_boardwalk_is_last_property(self) -> None:
        """Boardwalk should be at position 39."""
        space = Board.get_space(39)
        assert isinstance(space, PropertySpace)
        assert space.name == "Boardwalk"
        assert space.color == PropertyColor.DARK_BLUE

    def test_get_space_wraps_around(self) -> None:
        """get_space should wrap positions > 39."""
        assert Board.get_space(40) == Board.get_space(0)
        assert Board.get_space(41) == Board.get_space(1)
        assert Board.get_space(80) == Board.get_space(0)

    def test_jail_at_position_10(self) -> None:
        """Jail should be at position 10."""
        space = Board.get_space(10)
        assert space.space_type == SpaceType.JAIL
        assert "Jail" in space.name

    def test_go_to_jail_at_position_30(self) -> None:
        """Go To Jail should be at position 30."""
        space = Board.get_space(30)
        assert space.space_type == SpaceType.GO_TO_JAIL

    def test_free_parking_at_position_20(self) -> None:
        """Free Parking should be at position 20."""
        space = Board.get_space(20)
        assert space.space_type == SpaceType.FREE_PARKING


class TestPropertySpaces:
    """Tests for property spaces on the board."""

    def test_mediterranean_avenue(self) -> None:
        """Mediterranean Avenue should be correctly defined."""
        space = Board.get_space(1)
        assert isinstance(space, PropertySpace)
        assert space.name == "Mediterranean Avenue"
        assert space.color == PropertyColor.BROWN
        assert space.cost == 60
        assert space.rent[0] == 2  # Base rent
        assert space.house_cost == 50
        assert space.mortgage_value == 30

    def test_boardwalk(self) -> None:
        """Boardwalk should be correctly defined."""
        space = Board.get_space(39)
        assert isinstance(space, PropertySpace)
        assert space.name == "Boardwalk"
        assert space.color == PropertyColor.DARK_BLUE
        assert space.cost == 400
        assert space.rent[0] == 50  # Base rent
        assert space.rent[5] == 2000  # Hotel rent
        assert space.house_cost == 200
        assert space.mortgage_value == 200

    def test_property_rent_increases_with_houses(self) -> None:
        """Property rent should increase with houses."""
        space = Board.get_space(1)
        assert isinstance(space, PropertySpace)
        assert len(space.rent) == 6  # Base + 4 houses + hotel
        # Rent should increase with each house
        for i in range(1, 6):
            assert space.rent[i] > space.rent[i - 1]

    def test_all_properties_have_valid_data(self, all_property_positions: list[int]) -> None:
        """All properties should have valid cost, rent, etc."""
        for pos in all_property_positions:
            space = Board.get_space(pos)
            assert isinstance(space, PropertySpace)
            assert space.cost > 0
            assert len(space.rent) == 6
            assert all(r >= 0 for r in space.rent)
            assert space.house_cost > 0
            assert space.mortgage_value > 0


class TestRailroadSpaces:
    """Tests for railroad spaces on the board."""

    def test_reading_railroad(self) -> None:
        """Reading Railroad should be correctly defined."""
        space = Board.get_space(5)
        assert isinstance(space, RailroadSpace)
        assert space.name == "Reading Railroad"
        assert space.cost == 200
        assert space.rent == (25, 50, 100, 200)
        assert space.mortgage_value == 100

    def test_all_railroads_same_cost(self) -> None:
        """All railroads should cost $200."""
        for pos in [5, 15, 25, 35]:
            space = Board.get_space(pos)
            assert isinstance(space, RailroadSpace)
            assert space.cost == 200

    def test_railroad_rent_structure(self) -> None:
        """Railroad rent should double with each owned."""
        space = Board.get_space(5)
        assert isinstance(space, RailroadSpace)
        assert space.rent[0] == 25  # 1 owned
        assert space.rent[1] == 50  # 2 owned
        assert space.rent[2] == 100  # 3 owned
        assert space.rent[3] == 200  # 4 owned


class TestUtilitySpaces:
    """Tests for utility spaces on the board."""

    def test_electric_company(self) -> None:
        """Electric Company should be correctly defined."""
        space = Board.get_space(12)
        assert isinstance(space, UtilitySpace)
        assert space.name == "Electric Company"
        assert space.cost == 150
        assert space.mortgage_value == 75

    def test_water_works(self) -> None:
        """Water Works should be correctly defined."""
        space = Board.get_space(28)
        assert isinstance(space, UtilitySpace)
        assert space.name == "Water Works"
        assert space.cost == 150
        assert space.mortgage_value == 75


class TestTaxSpaces:
    """Tests for tax spaces on the board."""

    def test_income_tax(self) -> None:
        """Income Tax should be at position 4."""
        space = Board.get_space(4)
        assert isinstance(space, TaxSpace)
        assert space.name == "Income Tax"
        assert space.amount == 200

    def test_luxury_tax(self) -> None:
        """Luxury Tax should be at position 38."""
        space = Board.get_space(38)
        assert isinstance(space, TaxSpace)
        assert space.name == "Luxury Tax"
        assert space.amount == 100


class TestChanceSpaces:
    """Tests for Chance spaces on the board."""

    def test_chance_positions(self) -> None:
        """Chance spaces should be at positions 7, 22, 36."""
        for pos in [7, 22, 36]:
            space = Board.get_space(pos)
            assert space.space_type == SpaceType.CHANCE


class TestCommunityChestSpaces:
    """Tests for Community Chest spaces on the board."""

    def test_community_chest_positions(self) -> None:
        """Community Chest should be at positions 2, 17, 33."""
        for pos in [2, 17, 33]:
            space = Board.get_space(pos)
            assert space.space_type == SpaceType.COMMUNITY_CHEST


class TestBoardUtilityMethods:
    """Tests for Board utility methods."""

    def test_get_property_group(self) -> None:
        """get_property_group should return correct positions."""
        brown = Board.get_property_group(PropertyColor.BROWN)
        assert brown == (1, 3)

        railroads = Board.get_property_group(PropertyColor.RAILROAD)
        assert railroads == (5, 15, 25, 35)

    def test_is_property(self) -> None:
        """is_property should correctly identify properties."""
        assert Board.is_property(1)  # Mediterranean
        assert Board.is_property(39)  # Boardwalk
        assert not Board.is_property(5)  # Railroad
        assert not Board.is_property(12)  # Utility
        assert not Board.is_property(0)  # GO

    def test_is_railroad(self) -> None:
        """is_railroad should correctly identify railroads."""
        assert Board.is_railroad(5)
        assert Board.is_railroad(15)
        assert Board.is_railroad(25)
        assert Board.is_railroad(35)
        assert not Board.is_railroad(1)
        assert not Board.is_railroad(12)

    def test_is_utility(self) -> None:
        """is_utility should correctly identify utilities."""
        assert Board.is_utility(12)
        assert Board.is_utility(28)
        assert not Board.is_utility(1)
        assert not Board.is_utility(5)

    def test_is_buyable(self) -> None:
        """is_buyable should identify all purchasable spaces."""
        assert Board.is_buyable(1)  # Property
        assert Board.is_buyable(5)  # Railroad
        assert Board.is_buyable(12)  # Utility
        assert not Board.is_buyable(0)  # GO
        assert not Board.is_buyable(7)  # Chance
        assert not Board.is_buyable(4)  # Income Tax

    def test_get_buyable_positions(self) -> None:
        """get_buyable_positions should return all 28 buyable positions."""
        positions = Board.get_buyable_positions()
        assert len(positions) == 28  # 22 properties + 4 railroads + 2 utilities

    def test_get_nearest_railroad(self) -> None:
        """get_nearest_railroad should return correct railroad."""
        assert Board.get_nearest_railroad(0) == 5  # From GO
        assert Board.get_nearest_railroad(5) == 15  # From Reading
        assert Board.get_nearest_railroad(36) == 5  # Wrap around

    def test_get_nearest_utility(self) -> None:
        """get_nearest_utility should return correct utility."""
        assert Board.get_nearest_utility(0) == 12  # Electric Company
        assert Board.get_nearest_utility(15) == 28  # Water Works
        assert Board.get_nearest_utility(30) == 12  # Wrap around

    def test_calculate_distance(self) -> None:
        """calculate_distance should correctly measure board distance."""
        assert Board.calculate_distance(0, 10) == 10
        assert Board.calculate_distance(35, 5) == 10  # Wrap around
        assert Board.calculate_distance(0, 0) == 0


class TestSpaceSerialization:
    """Tests for space serialization."""

    def test_property_space_to_dict(self) -> None:
        """PropertySpace.to_dict should include all fields."""
        space = Board.get_space(1)
        assert isinstance(space, PropertySpace)
        data = space.to_dict()
        assert data["position"] == 1
        assert data["name"] == "Mediterranean Avenue"
        assert data["space_type"] == "PROPERTY"
        assert data["color"] == "BROWN"
        assert data["cost"] == 60
        assert data["rent"] == [2, 10, 30, 90, 160, 250]

    def test_railroad_space_to_dict(self) -> None:
        """RailroadSpace.to_dict should include all fields."""
        space = Board.get_space(5)
        assert isinstance(space, RailroadSpace)
        data = space.to_dict()
        assert data["position"] == 5
        assert data["name"] == "Reading Railroad"
        assert data["space_type"] == "RAILROAD"
        assert data["cost"] == 200
        assert data["rent"] == [25, 50, 100, 200]

    def test_board_to_dict(self) -> None:
        """Board.to_dict should serialize entire board."""
        data = Board.to_dict()
        assert "spaces" in data
        assert "property_groups" in data
        assert len(data["spaces"]) == 40
