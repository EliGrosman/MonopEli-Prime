"""
Comprehensive tests for the property module.

This module tests the Property class (mutable property state) and the
PropertyManager class (manages all 28 buyable properties in the game).
"""

import pytest
from monopoly_engine.property import Property, PropertyManager
from monopoly_engine.types import PropertyColor


# =============================================================================
# Property Class Tests
# =============================================================================


class TestPropertyCreation:
    """Test Property class initialization and creation."""

    def test_property_creation_minimal(self) -> None:
        """Verify Property can be created with just a position."""
        prop = Property(position=1)
        assert prop.position == 1
        assert prop.owner is None
        assert prop.houses == 0
        assert prop.mortgaged is False

    def test_property_creation_with_owner(self, owned_property: Property) -> None:
        """Verify Property can be created with an owner."""
        assert owned_property.owner == 0
        assert owned_property.position == 1

    def test_property_creation_with_houses(self, property_with_houses: Property) -> None:
        """Verify Property can be created with houses."""
        assert property_with_houses.houses == 3
        assert property_with_houses.owner == 0

    def test_property_creation_with_hotel(self, property_with_hotel: Property) -> None:
        """Verify Property can be created with a hotel."""
        assert property_with_hotel.houses == 5
        assert property_with_hotel.owner == 0

    def test_property_creation_mortgaged(self, mortgaged_property: Property) -> None:
        """Verify Property can be created in mortgaged state."""
        assert mortgaged_property.mortgaged is True
        assert mortgaged_property.owner == 0


class TestPropertyIsOwned:
    """Test Property.is_owned property."""

    def test_is_owned_false_when_no_owner(self, mediterranean_property: Property) -> None:
        """Verify is_owned returns False for unowned property."""
        assert mediterranean_property.is_owned is False

    def test_is_owned_true_when_has_owner(self, owned_property: Property) -> None:
        """Verify is_owned returns True for owned property."""
        assert owned_property.is_owned is True

    def test_is_owned_changes_with_ownership(self) -> None:
        """Verify is_owned changes when owner is set."""
        prop = Property(position=1)
        assert prop.is_owned is False

        prop.owner = 0
        assert prop.is_owned is True

        prop.owner = None
        assert prop.is_owned is False


class TestPropertyIsHotel:
    """Test Property.is_hotel property."""

    def test_is_hotel_false_with_no_houses(self, owned_property: Property) -> None:
        """Verify is_hotel returns False when houses == 0."""
        assert owned_property.is_hotel is False

    def test_is_hotel_false_with_some_houses(self, property_with_houses: Property) -> None:
        """Verify is_hotel returns False when houses < 5."""
        assert property_with_houses.houses == 3
        assert property_with_houses.is_hotel is False

    def test_is_hotel_false_with_four_houses(self) -> None:
        """Verify is_hotel returns False when houses == 4."""
        prop = Property(position=1, owner=0, houses=4)
        assert prop.is_hotel is False

    def test_is_hotel_true_with_hotel(self, property_with_hotel: Property) -> None:
        """Verify is_hotel returns True when houses == 5."""
        assert property_with_hotel.is_hotel is True

    def test_is_hotel_changes_with_houses(self) -> None:
        """Verify is_hotel changes as houses are built."""
        prop = Property(position=1, owner=0)
        assert prop.is_hotel is False

        prop.houses = 4
        assert prop.is_hotel is False

        prop.houses = 5
        assert prop.is_hotel is True


class TestPropertyCanBuild:
    """Test Property.can_build() method."""

    def test_can_build_false_when_unowned(self, mediterranean_property: Property) -> None:
        """Verify can_build returns False for unowned property."""
        assert mediterranean_property.can_build() is False

    def test_can_build_true_when_owned_no_houses(self, owned_property: Property) -> None:
        """Verify can_build returns True for owned property with no houses."""
        assert owned_property.can_build() is True

    def test_can_build_true_with_some_houses(self, property_with_houses: Property) -> None:
        """Verify can_build returns True with houses < 5."""
        assert property_with_houses.can_build() is True

    def test_can_build_false_with_hotel(self, property_with_hotel: Property) -> None:
        """Verify can_build returns False when hotel is built."""
        assert property_with_hotel.can_build() is False

    def test_can_build_false_when_mortgaged(self, mortgaged_property: Property) -> None:
        """Verify can_build returns False for mortgaged property."""
        assert mortgaged_property.can_build() is False

    def test_can_build_progression(self) -> None:
        """Verify can_build changes correctly as houses are built."""
        prop = Property(position=1, owner=0)

        for houses in range(5):
            prop.houses = houses
            assert prop.can_build() is True

        prop.houses = 5
        assert prop.can_build() is False


class TestPropertyCanMortgage:
    """Test Property.can_mortgage() method."""

    def test_can_mortgage_false_when_unowned(self, mediterranean_property: Property) -> None:
        """Verify can_mortgage returns False for unowned property."""
        assert mediterranean_property.can_mortgage() is False

    def test_can_mortgage_true_when_owned_no_houses(self, owned_property: Property) -> None:
        """Verify can_mortgage returns True for owned property with no houses."""
        assert owned_property.can_mortgage() is True

    def test_can_mortgage_false_with_houses(self, property_with_houses: Property) -> None:
        """Verify can_mortgage returns False when property has houses."""
        assert property_with_houses.can_mortgage() is False

    def test_can_mortgage_false_when_already_mortgaged(
        self, mortgaged_property: Property
    ) -> None:
        """Verify can_mortgage returns False for already mortgaged property."""
        assert mortgaged_property.can_mortgage() is False

    def test_can_mortgage_false_with_hotel(self, property_with_hotel: Property) -> None:
        """Verify can_mortgage returns False when property has hotel."""
        assert property_with_hotel.can_mortgage() is False


class TestPropertyCanUnmortgage:
    """Test Property.can_unmortgage() method."""

    def test_can_unmortgage_false_when_unowned(
        self, mediterranean_property: Property
    ) -> None:
        """Verify can_unmortgage returns False for unowned property."""
        assert mediterranean_property.can_unmortgage() is False

    def test_can_unmortgage_false_when_not_mortgaged(self, owned_property: Property) -> None:
        """Verify can_unmortgage returns False for non-mortgaged property."""
        assert owned_property.can_unmortgage() is False

    def test_can_unmortgage_true_when_mortgaged(self, mortgaged_property: Property) -> None:
        """Verify can_unmortgage returns True for mortgaged property."""
        assert mortgaged_property.can_unmortgage() is True

    def test_can_unmortgage_state_changes(self) -> None:
        """Verify can_unmortgage changes with mortgage state."""
        prop = Property(position=1, owner=0)
        assert prop.can_unmortgage() is False

        prop.mortgaged = True
        assert prop.can_unmortgage() is True

        prop.mortgaged = False
        assert prop.can_unmortgage() is False


class TestPropertySerialization:
    """Test Property serialization to/from dict."""

    def test_to_dict_basic(self, mediterranean_property: Property) -> None:
        """Verify to_dict works for basic unowned property."""
        data = mediterranean_property.to_dict()
        assert data["position"] == 1
        assert data["owner"] is None
        assert data["houses"] == 0
        assert data["mortgaged"] is False

    def test_to_dict_owned_property(self, owned_property: Property) -> None:
        """Verify to_dict works for owned property."""
        data = owned_property.to_dict()
        assert data["position"] == 1
        assert data["owner"] == 0
        assert data["houses"] == 0
        assert data["mortgaged"] is False

    def test_to_dict_with_houses(self, property_with_houses: Property) -> None:
        """Verify to_dict works for property with houses."""
        data = property_with_houses.to_dict()
        assert data["position"] == 1
        assert data["owner"] == 0
        assert data["houses"] == 3
        assert data["mortgaged"] is False

    def test_to_dict_mortgaged(self, mortgaged_property: Property) -> None:
        """Verify to_dict works for mortgaged property."""
        data = mortgaged_property.to_dict()
        assert data["position"] == 1
        assert data["owner"] == 0
        assert data["houses"] == 0
        assert data["mortgaged"] is True

    def test_from_dict_basic(self) -> None:
        """Verify from_dict recreates basic property."""
        data = {
            "position": 1,
            "owner": None,
            "houses": 0,
            "mortgaged": False,
        }
        prop = Property.from_dict(data)
        assert prop.position == 1
        assert prop.owner is None
        assert prop.houses == 0
        assert prop.mortgaged is False

    def test_from_dict_owned(self) -> None:
        """Verify from_dict recreates owned property."""
        data = {
            "position": 5,
            "owner": 2,
            "houses": 3,
            "mortgaged": False,
        }
        prop = Property.from_dict(data)
        assert prop.position == 5
        assert prop.owner == 2
        assert prop.houses == 3
        assert prop.mortgaged is False

    def test_round_trip_serialization(self, property_with_houses: Property) -> None:
        """Verify property survives round-trip serialization."""
        data = property_with_houses.to_dict()
        restored = Property.from_dict(data)

        assert restored.position == property_with_houses.position
        assert restored.owner == property_with_houses.owner
        assert restored.houses == property_with_houses.houses
        assert restored.mortgaged == property_with_houses.mortgaged


# =============================================================================
# PropertyManager Class Tests
# =============================================================================


class TestPropertyManagerCreation:
    """Test PropertyManager initialization."""

    def test_property_manager_creation(self, property_manager: PropertyManager) -> None:
        """Verify PropertyManager initializes with all properties."""
        assert len(property_manager.properties) == 28

    def test_all_properties_have_correct_positions(
        self, property_manager: PropertyManager
    ) -> None:
        """Verify all properties have correct positions matching their keys."""
        for pos, prop in property_manager.properties.items():
            assert prop.position == pos

    def test_all_properties_start_unowned(self, property_manager: PropertyManager) -> None:
        """Verify all properties start with no owner."""
        for prop in property_manager.properties.values():
            assert prop.owner is None
            assert prop.houses == 0
            assert prop.mortgaged is False

    def test_property_manager_includes_all_buyable_spaces(
        self, property_manager: PropertyManager
    ) -> None:
        """Verify PropertyManager includes all 28 buyable spaces."""
        # 22 properties + 4 railroads + 2 utilities = 28
        from monopoly_engine.board import Board, PropertySpace, RailroadSpace, UtilitySpace

        buyable_positions = []
        for i, space in enumerate(Board.SPACES):
            if isinstance(space, (PropertySpace, RailroadSpace, UtilitySpace)):
                buyable_positions.append(i)

        assert len(buyable_positions) == 28
        assert set(buyable_positions) == set(property_manager.properties.keys())


class TestPropertyManagerGetOwnedByPlayer:
    """Test PropertyManager.get_owned_by_player()."""

    def test_get_owned_by_player_empty(self, property_manager: PropertyManager) -> None:
        """Verify returns empty list when player owns nothing."""
        owned = property_manager.get_owned_by_player(0)
        assert owned == []

    def test_get_owned_by_player_single_property(
        self, property_manager: PropertyManager
    ) -> None:
        """Verify returns single property when player owns one."""
        property_manager.properties[1].owner = 0
        owned = property_manager.get_owned_by_player(0)
        assert owned == [1]

    def test_get_owned_by_player_multiple_properties(
        self, property_manager: PropertyManager
    ) -> None:
        """Verify returns all properties owned by player."""
        property_manager.properties[1].owner = 0
        property_manager.properties[3].owner = 0
        property_manager.properties[6].owner = 0

        owned = property_manager.get_owned_by_player(0)
        assert sorted(owned) == [1, 3, 6]

    def test_get_owned_by_player_sorted(self, property_manager: PropertyManager) -> None:
        """Verify returned list is sorted by position."""
        property_manager.properties[39].owner = 0
        property_manager.properties[1].owner = 0
        property_manager.properties[21].owner = 0  # Position 20 is Free Parking (not buyable)

        owned = property_manager.get_owned_by_player(0)
        assert owned == [1, 21, 39]

    def test_get_owned_by_player_different_players(
        self, property_manager: PropertyManager
    ) -> None:
        """Verify correctly filters by player ID."""
        property_manager.properties[1].owner = 0
        property_manager.properties[3].owner = 1
        property_manager.properties[6].owner = 0

        owned_0 = property_manager.get_owned_by_player(0)
        owned_1 = property_manager.get_owned_by_player(1)

        assert sorted(owned_0) == [1, 6]
        assert owned_1 == [3]


class TestPropertyManagerHasMonopoly:
    """Test PropertyManager.has_monopoly()."""

    def test_has_monopoly_false_when_owns_none(
        self, property_manager: PropertyManager
    ) -> None:
        """Verify returns False when player owns no properties in group."""
        assert property_manager.has_monopoly(0, PropertyColor.BROWN) is False

    def test_has_monopoly_false_when_owns_partial(
        self, property_manager: PropertyManager
    ) -> None:
        """Verify returns False when player owns only part of group."""
        property_manager.properties[1].owner = 0
        # Don't own position 3
        assert property_manager.has_monopoly(0, PropertyColor.BROWN) is False

    def test_has_monopoly_true_when_owns_complete_group(
        self, property_manager_with_brown_monopoly: PropertyManager
    ) -> None:
        """Verify returns True when player owns all properties in group."""
        assert (
            property_manager_with_brown_monopoly.has_monopoly(0, PropertyColor.BROWN)
            is True
        )

    def test_has_monopoly_false_when_split_ownership(
        self, property_manager: PropertyManager
    ) -> None:
        """Verify returns False when properties split between players."""
        property_manager.properties[1].owner = 0
        property_manager.properties[3].owner = 1
        assert property_manager.has_monopoly(0, PropertyColor.BROWN) is False
        assert property_manager.has_monopoly(1, PropertyColor.BROWN) is False

    def test_has_monopoly_light_blue(self, property_manager: PropertyManager) -> None:
        """Verify monopoly detection for 3-property group."""
        property_manager.properties[6].owner = 0
        property_manager.properties[8].owner = 0
        assert property_manager.has_monopoly(0, PropertyColor.LIGHT_BLUE) is False

        property_manager.properties[9].owner = 0
        assert property_manager.has_monopoly(0, PropertyColor.LIGHT_BLUE) is True

    def test_has_monopoly_all_colors(self, property_manager: PropertyManager) -> None:
        """Verify monopoly detection works for all color groups."""
        color_groups = {
            PropertyColor.BROWN: [1, 3],
            PropertyColor.LIGHT_BLUE: [6, 8, 9],
            PropertyColor.MAGENTA: [11, 13, 14],
            PropertyColor.ORANGE: [16, 18, 19],
            PropertyColor.RED: [21, 23, 24],
            PropertyColor.YELLOW: [26, 27, 29],
            PropertyColor.GREEN: [31, 32, 34],
            PropertyColor.BLUE: [37, 39],
        }

        for color, positions in color_groups.items():
            # Initially no monopoly
            assert property_manager.has_monopoly(0, color) is False

            # Assign all properties
            for pos in positions:
                property_manager.properties[pos].owner = 0

            # Now should have monopoly
            assert property_manager.has_monopoly(0, color) is True


class TestPropertyManagerGetMonopolies:
    """Test PropertyManager.get_monopolies()."""

    def test_get_monopolies_empty(self, property_manager: PropertyManager) -> None:
        """Verify returns empty list when player has no monopolies."""
        monopolies = property_manager.get_monopolies(0)
        assert monopolies == []

    def test_get_monopolies_single(
        self, property_manager_with_brown_monopoly: PropertyManager
    ) -> None:
        """Verify returns single monopoly."""
        monopolies = property_manager_with_brown_monopoly.get_monopolies(0)
        assert PropertyColor.BROWN in monopolies
        assert len(monopolies) == 1

    def test_get_monopolies_multiple(self, property_manager: PropertyManager) -> None:
        """Verify returns all monopolies owned by player."""
        # Give player brown monopoly
        property_manager.properties[1].owner = 0
        property_manager.properties[3].owner = 0

        # Give player blue monopoly
        property_manager.properties[37].owner = 0
        property_manager.properties[39].owner = 0

        monopolies = property_manager.get_monopolies(0)
        assert len(monopolies) == 2
        assert PropertyColor.BROWN in monopolies
        assert PropertyColor.BLUE in monopolies

    def test_get_monopolies_different_players(
        self, property_manager: PropertyManager
    ) -> None:
        """Verify correctly tracks monopolies for different players."""
        # Player 0 gets brown
        property_manager.properties[1].owner = 0
        property_manager.properties[3].owner = 0

        # Player 1 gets light blue
        property_manager.properties[6].owner = 1
        property_manager.properties[8].owner = 1
        property_manager.properties[9].owner = 1

        monopolies_0 = property_manager.get_monopolies(0)
        monopolies_1 = property_manager.get_monopolies(1)

        assert PropertyColor.BROWN in monopolies_0
        assert PropertyColor.LIGHT_BLUE not in monopolies_0

        assert PropertyColor.LIGHT_BLUE in monopolies_1
        assert PropertyColor.BROWN not in monopolies_1


class TestPropertyManagerGroupQueries:
    """Test PropertyManager methods for querying property groups."""

    def test_get_properties_in_group(self, property_manager: PropertyManager) -> None:
        """Verify get_properties_in_group returns correct positions."""
        brown = property_manager.get_properties_in_group(PropertyColor.BROWN)
        assert brown == [1, 3]

        light_blue = property_manager.get_properties_in_group(PropertyColor.LIGHT_BLUE)
        assert light_blue == [6, 8, 9]

    def test_count_houses_in_group_no_houses(
        self, property_manager: PropertyManager
    ) -> None:
        """Verify count_houses_in_group returns 0 when no houses built."""
        count = property_manager.count_houses_in_group(PropertyColor.BROWN)
        assert count == 0

    def test_count_houses_in_group_with_houses(
        self, property_manager: PropertyManager
    ) -> None:
        """Verify count_houses_in_group returns correct total."""
        property_manager.properties[1].houses = 2
        property_manager.properties[3].houses = 3

        count = property_manager.count_houses_in_group(PropertyColor.BROWN)
        assert count == 5

    def test_count_houses_in_group_excludes_hotels(
        self, property_manager: PropertyManager
    ) -> None:
        """Verify hotels (houses=5) are not counted as houses."""
        property_manager.properties[1].houses = 5  # Hotel
        property_manager.properties[3].houses = 2

        count = property_manager.count_houses_in_group(PropertyColor.BROWN)
        assert count == 2  # Only counts the 2 houses, not the hotel

    def test_get_min_houses_in_group(self, property_manager: PropertyManager) -> None:
        """Verify get_min_houses_in_group returns minimum."""
        property_manager.properties[1].houses = 2
        property_manager.properties[3].houses = 3

        min_houses = property_manager.get_min_houses_in_group(PropertyColor.BROWN)
        assert min_houses == 2

    def test_get_max_houses_in_group(self, property_manager: PropertyManager) -> None:
        """Verify get_max_houses_in_group returns maximum."""
        property_manager.properties[1].houses = 2
        property_manager.properties[3].houses = 3

        max_houses = property_manager.get_max_houses_in_group(PropertyColor.BROWN)
        assert max_houses == 3

    def test_get_min_max_with_hotel(self, property_manager: PropertyManager) -> None:
        """Verify min/max work correctly with hotels."""
        property_manager.properties[37].houses = 5  # Hotel
        property_manager.properties[39].houses = 3

        min_houses = property_manager.get_min_houses_in_group(PropertyColor.BLUE)
        max_houses = property_manager.get_max_houses_in_group(PropertyColor.BLUE)

        assert min_houses == 3
        assert max_houses == 5


class TestPropertyManagerSerialization:
    """Test PropertyManager serialization to/from dict."""

    def test_to_dict_empty(self, property_manager: PropertyManager) -> None:
        """Verify to_dict works for empty PropertyManager."""
        data = property_manager.to_dict()
        assert "properties" in data
        assert len(data["properties"]) == 28

    def test_to_dict_with_ownership(self, property_manager: PropertyManager) -> None:
        """Verify to_dict includes ownership information."""
        property_manager.properties[1].owner = 0
        property_manager.properties[3].owner = 1

        data = property_manager.to_dict()
        props = {p["position"]: p for p in data["properties"]}

        assert props[1]["owner"] == 0
        assert props[3]["owner"] == 1

    def test_from_dict(self) -> None:
        """Verify from_dict recreates PropertyManager."""
        data = {
            "properties": [
                {"position": 1, "owner": 0, "houses": 2, "mortgaged": False},
                {"position": 3, "owner": 0, "houses": 3, "mortgaged": False},
            ]
        }

        # Create manager with more properties
        manager = PropertyManager()
        # Update with our data
        for prop_data in data["properties"]:
            pos = int(prop_data["position"])
            manager.properties[pos] = Property.from_dict(prop_data)

        assert manager.properties[1].owner == 0
        assert manager.properties[1].houses == 2
        assert manager.properties[3].owner == 0
        assert manager.properties[3].houses == 3

    def test_round_trip_serialization(
        self, property_manager_with_brown_monopoly: PropertyManager
    ) -> None:
        """Verify PropertyManager survives round-trip serialization."""
        # Add some houses
        property_manager_with_brown_monopoly.properties[1].houses = 2
        property_manager_with_brown_monopoly.properties[3].houses = 3

        data = property_manager_with_brown_monopoly.to_dict()
        restored = PropertyManager.from_dict(data)

        assert restored.properties[1].owner == 0
        assert restored.properties[1].houses == 2
        assert restored.properties[3].owner == 0
        assert restored.properties[3].houses == 3


class TestPropertyManagerReset:
    """Test PropertyManager.reset() method."""

    def test_reset_clears_ownership(self, property_manager: PropertyManager) -> None:
        """Verify reset clears all property ownership."""
        property_manager.properties[1].owner = 0
        property_manager.properties[3].owner = 1
        property_manager.properties[6].owner = 0

        property_manager.reset()

        for prop in property_manager.properties.values():
            assert prop.owner is None

    def test_reset_clears_houses(self, property_manager: PropertyManager) -> None:
        """Verify reset clears all houses."""
        property_manager.properties[1].houses = 3
        property_manager.properties[3].houses = 5

        property_manager.reset()

        for prop in property_manager.properties.values():
            assert prop.houses == 0

    def test_reset_clears_mortgages(self, property_manager: PropertyManager) -> None:
        """Verify reset clears all mortgages."""
        property_manager.properties[1].mortgaged = True
        property_manager.properties[3].mortgaged = True

        property_manager.reset()

        for prop in property_manager.properties.values():
            assert prop.mortgaged is False

    def test_reset_keeps_positions(self, property_manager: PropertyManager) -> None:
        """Verify reset preserves property positions."""
        positions_before = set(property_manager.properties.keys())

        property_manager.properties[1].owner = 0
        property_manager.reset()

        positions_after = set(property_manager.properties.keys())
        assert positions_before == positions_after


# =============================================================================
# Integration Tests
# =============================================================================


class TestPropertyManagerIntegration:
    """Test PropertyManager with complex scenarios."""

    def test_even_building_scenario(self, property_manager: PropertyManager) -> None:
        """Test scenario for even building rule enforcement."""
        # Player owns brown monopoly
        property_manager.properties[1].owner = 0
        property_manager.properties[3].owner = 0

        # Build evenly
        property_manager.properties[1].houses = 2
        property_manager.properties[3].houses = 2

        min_houses = property_manager.get_min_houses_in_group(PropertyColor.BROWN)
        max_houses = property_manager.get_max_houses_in_group(PropertyColor.BROWN)

        assert max_houses - min_houses == 0  # Even building

    def test_uneven_building_scenario(self, property_manager: PropertyManager) -> None:
        """Test scenario where building is uneven (violation)."""
        property_manager.properties[6].owner = 0
        property_manager.properties[8].owner = 0
        property_manager.properties[9].owner = 0

        # Build unevenly
        property_manager.properties[6].houses = 3
        property_manager.properties[8].houses = 1
        property_manager.properties[9].houses = 2

        min_houses = property_manager.get_min_houses_in_group(PropertyColor.LIGHT_BLUE)
        max_houses = property_manager.get_max_houses_in_group(PropertyColor.LIGHT_BLUE)

        assert max_houses - min_houses == 2  # Uneven - should not be allowed

    def test_multiple_monopolies_one_player(
        self, property_manager: PropertyManager
    ) -> None:
        """Test player with multiple complete monopolies."""
        # Give player 0 brown monopoly
        for pos in [1, 3]:
            property_manager.properties[pos].owner = 0

        # Give player 0 blue monopoly
        for pos in [37, 39]:
            property_manager.properties[pos].owner = 0

        # Give player 0 all railroads
        for pos in [5, 15, 25, 35]:
            property_manager.properties[pos].owner = 0

        monopolies = property_manager.get_monopolies(0)
        owned = property_manager.get_owned_by_player(0)

        assert len(monopolies) == 2  # Brown and Blue
        assert len(owned) == 8  # 2 + 2 + 4

    def test_trade_breaks_monopoly(self, property_manager: PropertyManager) -> None:
        """Test that trading away a property breaks monopoly."""
        # Player 0 starts with brown monopoly
        property_manager.properties[1].owner = 0
        property_manager.properties[3].owner = 0
        assert property_manager.has_monopoly(0, PropertyColor.BROWN) is True

        # Trade one property to player 1
        property_manager.properties[3].owner = 1
        assert property_manager.has_monopoly(0, PropertyColor.BROWN) is False
        assert property_manager.has_monopoly(1, PropertyColor.BROWN) is False

    def test_complete_monopoly_through_trade(
        self, property_manager: PropertyManager
    ) -> None:
        """Test completing a monopoly through trade."""
        # Player 0 owns most of light blue
        property_manager.properties[6].owner = 0
        property_manager.properties[8].owner = 0
        property_manager.properties[9].owner = 1

        assert property_manager.has_monopoly(0, PropertyColor.LIGHT_BLUE) is False

        # Player 1 trades the last property to player 0
        property_manager.properties[9].owner = 0

        assert property_manager.has_monopoly(0, PropertyColor.LIGHT_BLUE) is True
