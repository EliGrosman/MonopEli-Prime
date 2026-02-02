"""Tests for property.py module."""

import pytest

from monopoly_engine import (
    Property,
    PropertyManager,
    PropertyColor,
    PROPERTY_GROUPS,
)


class TestProperty:
    """Tests for Property class."""

    def test_default_property(self) -> None:
        """Property should have sensible defaults."""
        prop = Property(position=1)
        assert prop.position == 1
        assert prop.owner is None
        assert prop.houses == 0
        assert not prop.mortgaged

    def test_is_owned(self) -> None:
        """is_owned should return True when owner is set."""
        prop = Property(position=1)
        assert not prop.is_owned
        prop.owner = 0
        assert prop.is_owned

    def test_has_hotel(self) -> None:
        """has_hotel should return True when houses == 5."""
        prop = Property(position=1, owner=0)
        assert not prop.has_hotel
        prop.houses = 4
        assert not prop.has_hotel
        prop.houses = 5
        assert prop.has_hotel

    def test_has_houses(self) -> None:
        """has_houses should return True for 1-4 houses."""
        prop = Property(position=1, owner=0)
        assert not prop.has_houses
        prop.houses = 1
        assert prop.has_houses
        prop.houses = 4
        assert prop.has_houses
        prop.houses = 5  # Hotel
        assert not prop.has_houses

    def test_has_buildings(self) -> None:
        """has_buildings should return True for any houses/hotel."""
        prop = Property(position=1, owner=0)
        assert not prop.has_buildings
        prop.houses = 1
        assert prop.has_buildings
        prop.houses = 5
        assert prop.has_buildings

    def test_color_property(self) -> None:
        """color should return the property's color."""
        prop = Property(position=1)  # Mediterranean
        assert prop.color == PropertyColor.BROWN
        prop2 = Property(position=39)  # Boardwalk
        assert prop2.color == PropertyColor.DARK_BLUE

    def test_can_build(self) -> None:
        """can_build should check basic requirements."""
        prop = Property(position=1)
        assert not prop.can_build()  # Not owned

        prop.owner = 0
        assert prop.can_build()

        prop.mortgaged = True
        assert not prop.can_build()

        prop.mortgaged = False
        prop.houses = 5
        assert not prop.can_build()  # Already has hotel

    def test_can_mortgage(self) -> None:
        """can_mortgage should check basic requirements."""
        prop = Property(position=1)
        assert not prop.can_mortgage()  # Not owned

        prop.owner = 0
        assert prop.can_mortgage()

        prop.houses = 1
        assert not prop.can_mortgage()

        prop.houses = 0
        prop.mortgaged = True
        assert not prop.can_mortgage()

    def test_can_unmortgage(self) -> None:
        """can_unmortgage should check if mortgaged."""
        prop = Property(position=1, owner=0)
        assert not prop.can_unmortgage()

        prop.mortgaged = True
        assert prop.can_unmortgage()

        prop.owner = None
        assert not prop.can_unmortgage()

    def test_to_dict(self) -> None:
        """to_dict should serialize property state."""
        prop = Property(position=1, owner=0, houses=2, mortgaged=False)
        data = prop.to_dict()
        assert data["position"] == 1
        assert data["owner"] == 0
        assert data["houses"] == 2
        assert data["mortgaged"] is False

    def test_from_dict(self) -> None:
        """from_dict should deserialize property state."""
        data = {"position": 1, "owner": 0, "houses": 2, "mortgaged": True}
        prop = Property.from_dict(data)
        assert prop.position == 1
        assert prop.owner == 0
        assert prop.houses == 2
        assert prop.mortgaged is True


class TestPropertyManager:
    """Tests for PropertyManager class."""

    def test_init_creates_28_properties(
        self, property_manager: PropertyManager
    ) -> None:
        """PropertyManager should initialize with 28 properties."""
        assert len(property_manager.properties) == 28

    def test_all_properties_start_unowned(
        self, property_manager: PropertyManager
    ) -> None:
        """All properties should start unowned."""
        for prop in property_manager.properties.values():
            assert not prop.is_owned

    def test_get_valid_position(
        self, property_manager: PropertyManager
    ) -> None:
        """get should return property for valid position."""
        prop = property_manager.get(1)
        assert prop is not None
        assert prop.position == 1

    def test_get_invalid_position(
        self, property_manager: PropertyManager
    ) -> None:
        """get should return None for non-property position."""
        assert property_manager.get(0) is None  # GO
        assert property_manager.get(7) is None  # Chance
        assert property_manager.get(10) is None  # Jail

    def test_get_owned_by(
        self, property_manager_with_owner: PropertyManager
    ) -> None:
        """get_owned_by should return owned positions."""
        player_0_props = property_manager_with_owner.get_owned_by(0)
        assert set(player_0_props) == {1, 3}

        player_1_props = property_manager_with_owner.get_owned_by(1)
        assert player_1_props == [5]

        player_2_props = property_manager_with_owner.get_owned_by(2)
        assert player_2_props == []

    def test_get_unowned(
        self, property_manager_with_owner: PropertyManager
    ) -> None:
        """get_unowned should return unowned positions."""
        unowned = property_manager_with_owner.get_unowned()
        assert 1 not in unowned  # Owned by player 0
        assert 3 not in unowned  # Owned by player 0
        assert 5 not in unowned  # Owned by player 1
        assert 6 in unowned  # Not owned

    def test_has_monopoly_true(
        self, property_manager_with_owner: PropertyManager
    ) -> None:
        """has_monopoly should return True when player owns all in group."""
        assert property_manager_with_owner.has_monopoly(0, PropertyColor.BROWN)

    def test_has_monopoly_false(
        self, property_manager: PropertyManager
    ) -> None:
        """has_monopoly should return False when player doesn't own all."""
        # Only own one property of the brown group
        property_manager.properties[1].owner = 0
        assert not property_manager.has_monopoly(0, PropertyColor.BROWN)

    def test_get_monopolies(
        self, property_manager_with_monopoly: PropertyManager
    ) -> None:
        """get_monopolies should return all monopolies for a player."""
        monopolies = property_manager_with_monopoly.get_monopolies(0)
        assert PropertyColor.LIGHT_BLUE in monopolies

    def test_count_railroads_owned(
        self, property_manager: PropertyManager
    ) -> None:
        """count_railroads_owned should count player's railroads."""
        assert property_manager.count_railroads_owned(0) == 0

        property_manager.properties[5].owner = 0
        assert property_manager.count_railroads_owned(0) == 1

        property_manager.properties[15].owner = 0
        assert property_manager.count_railroads_owned(0) == 2

    def test_count_utilities_owned(
        self, property_manager: PropertyManager
    ) -> None:
        """count_utilities_owned should count player's utilities."""
        assert property_manager.count_utilities_owned(0) == 0

        property_manager.properties[12].owner = 0
        assert property_manager.count_utilities_owned(0) == 1

        property_manager.properties[28].owner = 0
        assert property_manager.count_utilities_owned(0) == 2

    def test_count_houses_in_group(
        self, property_manager_with_houses: PropertyManager
    ) -> None:
        """count_houses_in_group should count houses in a monopoly."""
        # Light blue has 3 properties with 2 houses each = 6 houses
        count = property_manager_with_houses.count_houses_in_group(
            0, PropertyColor.LIGHT_BLUE
        )
        assert count == 6

    def test_count_hotels_in_group(
        self, property_manager_with_monopoly: PropertyManager
    ) -> None:
        """count_hotels_in_group should count hotels."""
        # No hotels initially
        count = property_manager_with_monopoly.count_hotels_in_group(
            0, PropertyColor.LIGHT_BLUE
        )
        assert count == 0

        # Add a hotel to one property
        property_manager_with_monopoly.properties[6].houses = 5
        count = property_manager_with_monopoly.count_hotels_in_group(
            0, PropertyColor.LIGHT_BLUE
        )
        assert count == 1

    def test_get_min_houses_in_group(
        self, property_manager_with_monopoly: PropertyManager
    ) -> None:
        """get_min_houses_in_group should return minimum."""
        # All start with 0
        assert property_manager_with_monopoly.get_min_houses_in_group(
            PropertyColor.LIGHT_BLUE
        ) == 0

        # Add houses unevenly
        property_manager_with_monopoly.properties[6].houses = 2
        property_manager_with_monopoly.properties[8].houses = 1
        assert property_manager_with_monopoly.get_min_houses_in_group(
            PropertyColor.LIGHT_BLUE
        ) == 0

    def test_get_max_houses_in_group(
        self, property_manager_with_houses: PropertyManager
    ) -> None:
        """get_max_houses_in_group should return maximum."""
        # All have 2 houses
        assert property_manager_with_houses.get_max_houses_in_group(
            PropertyColor.LIGHT_BLUE
        ) == 2

    def test_can_build_house_on(
        self, property_manager_with_monopoly: PropertyManager
    ) -> None:
        """can_build_house_on should check build eligibility."""
        # Can build with monopoly
        assert property_manager_with_monopoly.can_build_house_on(0, 6)

        # Can't build without monopoly
        property_manager_with_monopoly.properties[9].owner = 1
        assert not property_manager_with_monopoly.can_build_house_on(0, 6)

    def test_can_build_evenly(
        self, property_manager_with_monopoly: PropertyManager
    ) -> None:
        """can_build_house_on should enforce even building."""
        # Add 2 houses to first property only
        property_manager_with_monopoly.properties[6].houses = 2

        # Can't build more on 6 until others catch up
        assert not property_manager_with_monopoly.can_build_house_on(0, 6)

        # Can build on 8 (has 0 houses)
        assert property_manager_with_monopoly.can_build_house_on(0, 8)

    def test_can_sell_house_on(
        self, property_manager_with_houses: PropertyManager
    ) -> None:
        """can_sell_house_on should check sell eligibility."""
        # All have equal houses, can sell from any
        assert property_manager_with_houses.can_sell_house_on(0, 6)

    def test_can_sell_evenly(
        self, property_manager_with_houses: PropertyManager
    ) -> None:
        """can_sell_house_on should enforce even selling."""
        # Make one property have fewer houses
        property_manager_with_houses.properties[6].houses = 1

        # Can't sell from properties with fewer houses
        assert not property_manager_with_houses.can_sell_house_on(0, 6)

        # Can sell from properties with more houses
        assert property_manager_with_houses.can_sell_house_on(0, 8)

    def test_transfer_property(
        self, property_manager: PropertyManager
    ) -> None:
        """transfer_property should change owner."""
        property_manager.properties[1].owner = 0
        assert property_manager.transfer_property(1, 1)
        assert property_manager.properties[1].owner == 1

    def test_reset_property(
        self, property_manager: PropertyManager
    ) -> None:
        """reset_property should clear all state."""
        prop = property_manager.properties[1]
        prop.owner = 0
        prop.houses = 3
        prop.mortgaged = True

        assert property_manager.reset_property(1)
        assert prop.owner is None
        assert prop.houses == 0
        assert not prop.mortgaged

    def test_to_dict(
        self, property_manager_with_owner: PropertyManager
    ) -> None:
        """to_dict should serialize all properties."""
        data = property_manager_with_owner.to_dict()
        assert "1" in data
        assert data["1"]["owner"] == 0

    def test_from_dict(self) -> None:
        """from_dict should deserialize properties."""
        data = {
            "1": {"position": 1, "owner": 0, "houses": 2, "mortgaged": False}
        }
        manager = PropertyManager.from_dict(data)
        assert manager.properties[1].owner == 0
        assert manager.properties[1].houses == 2
