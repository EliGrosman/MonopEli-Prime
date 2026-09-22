"""Tests for types.py module."""

from monopoly_engine import (
    BOARD_SIZE,
    GO_SALARY,
    JAIL_POSITION,
    POSITION_TO_COLOR,
    PROPERTY_GROUPS,
    STARTING_MONEY,
    TOTAL_HOTELS,
    TOTAL_HOUSES,
    ActionType,
    CardType,
    GamePhase,
    PropertyColor,
    SpaceType,
)


class TestSpaceType:
    """Tests for SpaceType enum."""

    def test_all_space_types_defined(self) -> None:
        """Verify all expected space types exist."""
        expected = [
            "GO",
            "PROPERTY",
            "RAILROAD",
            "UTILITY",
            "CHANCE",
            "COMMUNITY_CHEST",
            "INCOME_TAX",
            "LUXURY_TAX",
            "JAIL",
            "GO_TO_JAIL",
            "FREE_PARKING",
        ]
        actual = [st.name for st in SpaceType]
        for name in expected:
            assert name in actual

    def test_space_types_are_unique(self) -> None:
        """Verify all space type values are unique."""
        values = [st.value for st in SpaceType]
        assert len(values) == len(set(values))


class TestPropertyColor:
    """Tests for PropertyColor enum."""

    def test_all_property_colors_defined(self) -> None:
        """Verify all expected property colors exist."""
        expected = [
            "BROWN",
            "LIGHT_BLUE",
            "MAGENTA",
            "ORANGE",
            "RED",
            "YELLOW",
            "GREEN",
            "DARK_BLUE",
            "RAILROAD",
            "UTILITY",
        ]
        actual = [pc.name for pc in PropertyColor]
        for name in expected:
            assert name in actual

    def test_property_colors_are_unique(self) -> None:
        """Verify all property color values are unique."""
        values = [pc.value for pc in PropertyColor]
        assert len(values) == len(set(values))


class TestActionType:
    """Tests for ActionType enum."""

    def test_all_action_types_defined(self) -> None:
        """Verify key action types exist."""
        expected = [
            "ROLL_DICE",
            "BUY_PROPERTY",
            "BUILD_HOUSE",
            "MORTGAGE_PROPERTY",
            "END_TURN",
            "DECLARE_BANKRUPTCY",
        ]
        actual = [at.name for at in ActionType]
        for name in expected:
            assert name in actual


class TestCardType:
    """Tests for CardType enum."""

    def test_all_card_types_defined(self) -> None:
        """Verify all expected card types exist."""
        expected = [
            "MOVE",
            "MOVE_NEAREST",
            "MOVE_BACK",
            "COLLECT",
            "PAY",
            "PAY_PER_BUILDING",
            "COLLECT_FROM_PLAYERS",
            "PAY_TO_PLAYERS",
            "GET_OUT_OF_JAIL",
            "GO_TO_JAIL",
        ]
        actual = [ct.name for ct in CardType]
        for name in expected:
            assert name in actual


class TestGamePhase:
    """Tests for GamePhase enum."""

    def test_key_phases_defined(self) -> None:
        """Verify key game phases exist."""
        expected = [
            "WAITING_FOR_ROLL",
            "ROLLED",
            "LANDED",
            "PURCHASE_DECISION",
            "IN_JAIL",
            "GAME_OVER",
        ]
        actual = [gp.name for gp in GamePhase]
        for name in expected:
            assert name in actual


class TestConstants:
    """Tests for game constants."""

    def test_board_size(self) -> None:
        """Board should have 40 spaces."""
        assert BOARD_SIZE == 40

    def test_starting_money(self) -> None:
        """Players should start with $1500."""
        assert STARTING_MONEY == 1500

    def test_go_salary(self) -> None:
        """GO salary should be $200."""
        assert GO_SALARY == 200

    def test_jail_position(self) -> None:
        """Jail should be at position 10."""
        assert JAIL_POSITION == 10

    def test_total_houses(self) -> None:
        """Game should have 32 houses."""
        assert TOTAL_HOUSES == 32

    def test_total_hotels(self) -> None:
        """Game should have 12 hotels."""
        assert TOTAL_HOTELS == 12


class TestPropertyGroups:
    """Tests for property group mappings."""

    def test_all_colors_have_groups(self) -> None:
        """Every property color should have positions defined."""
        for color in PropertyColor:
            assert color in PROPERTY_GROUPS

    def test_brown_group_has_two_properties(self) -> None:
        """Brown group should have 2 properties."""
        assert len(PROPERTY_GROUPS[PropertyColor.BROWN]) == 2
        assert PROPERTY_GROUPS[PropertyColor.BROWN] == (1, 3)

    def test_dark_blue_group_has_two_properties(self) -> None:
        """Dark blue group should have 2 properties."""
        assert len(PROPERTY_GROUPS[PropertyColor.DARK_BLUE]) == 2
        assert PROPERTY_GROUPS[PropertyColor.DARK_BLUE] == (37, 39)

    def test_railroad_group_has_four_positions(self) -> None:
        """Railroad group should have 4 positions."""
        assert len(PROPERTY_GROUPS[PropertyColor.RAILROAD]) == 4
        assert PROPERTY_GROUPS[PropertyColor.RAILROAD] == (5, 15, 25, 35)

    def test_utility_group_has_two_positions(self) -> None:
        """Utility group should have 2 positions."""
        assert len(PROPERTY_GROUPS[PropertyColor.UTILITY]) == 2
        assert PROPERTY_GROUPS[PropertyColor.UTILITY] == (12, 28)

    def test_standard_groups_have_three_properties(self) -> None:
        """Most color groups should have 3 properties."""
        three_prop_colors = [
            PropertyColor.LIGHT_BLUE,
            PropertyColor.MAGENTA,
            PropertyColor.ORANGE,
            PropertyColor.RED,
            PropertyColor.YELLOW,
            PropertyColor.GREEN,
        ]
        for color in three_prop_colors:
            assert len(PROPERTY_GROUPS[color]) == 3

    def test_all_positions_mapped(self) -> None:
        """All property positions should be in POSITION_TO_COLOR."""
        for color, positions in PROPERTY_GROUPS.items():
            for pos in positions:
                assert pos in POSITION_TO_COLOR
                assert POSITION_TO_COLOR[pos] == color

    def test_no_duplicate_positions(self) -> None:
        """No position should appear in multiple groups."""
        all_positions: list[int] = []
        for positions in PROPERTY_GROUPS.values():
            all_positions.extend(positions)
        assert len(all_positions) == len(set(all_positions))
