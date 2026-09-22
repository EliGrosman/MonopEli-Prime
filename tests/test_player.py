"""Tests for player.py module."""

import pytest

from monopoly_engine import JAIL_POSITION, STARTING_MONEY, Player


class TestPlayerBasics:
    """Tests for basic Player functionality."""

    def test_default_player(self) -> None:
        """Player should have sensible defaults."""
        player = Player(id=0, name="Test")
        assert player.id == 0
        assert player.name == "Test"
        assert player.money == STARTING_MONEY
        assert player.position == 0
        assert player.jail_cards == 0
        assert not player.in_jail
        assert player.jail_turns == 0
        assert not player.bankrupt

    def test_custom_starting_money(self) -> None:
        """Player can start with custom money."""
        player = Player(id=0, name="Rich", money=5000)
        assert player.money == 5000


class TestPlayerMovement:
    """Tests for player movement."""

    def test_move_forward(self, player: Player) -> None:
        """move should update position correctly."""
        new_pos, passed_go = player.move(5)
        assert new_pos == 5
        assert player.position == 5
        assert not passed_go

    def test_move_wraps_around(self, player: Player) -> None:
        """move should wrap around the board."""
        player.position = 38
        new_pos, passed_go = player.move(5)
        assert new_pos == 3
        assert player.position == 3
        assert passed_go

    def test_move_exactly_to_go(self, player: Player) -> None:
        """Moving exactly to GO should count as passing."""
        player.position = 35
        new_pos, passed_go = player.move(5)
        assert new_pos == 0
        assert passed_go

    def test_move_backward(self, player: Player) -> None:
        """move with negative spaces should move backward."""
        player.position = 5
        new_pos, passed_go = player.move(-3)
        assert new_pos == 2
        assert not passed_go

    def test_move_backward_wraps(self, player: Player) -> None:
        """Backward movement should wrap correctly."""
        player.position = 2
        new_pos, passed_go = player.move(-5)
        assert new_pos == 37
        assert not passed_go

    def test_move_to_position(self, player: Player) -> None:
        """move_to should go directly to position."""
        passed = player.move_to(24)
        assert player.position == 24
        assert not passed

    def test_move_to_passes_go(self, player: Player) -> None:
        """move_to should detect passing GO."""
        player.position = 30
        passed = player.move_to(5)
        assert player.position == 5
        assert passed

    def test_move_to_jail_no_go(self, player: Player) -> None:
        """Moving directly to jail shouldn't count as passing GO."""
        player.position = 30
        passed = player.move_to(JAIL_POSITION)
        assert player.position == JAIL_POSITION
        assert not passed


class TestPlayerMoney:
    """Tests for player money management."""

    def test_add_money(self, player: Player) -> None:
        """add_money should increase balance."""
        initial = player.money
        player.add_money(100)
        assert player.money == initial + 100

    def test_add_money_negative_raises(self, player: Player) -> None:
        """add_money should reject negative amounts."""
        with pytest.raises(ValueError):
            player.add_money(-100)

    def test_remove_money_success(self, player: Player) -> None:
        """remove_money should decrease balance when affordable."""
        initial = player.money
        result = player.remove_money(100)
        assert result is True
        assert player.money == initial - 100

    def test_remove_money_insufficient(self, player: Player) -> None:
        """remove_money should fail when not affordable."""
        result = player.remove_money(10000)
        assert result is False
        assert player.money == STARTING_MONEY  # Unchanged

    def test_remove_money_negative_raises(self, player: Player) -> None:
        """remove_money should reject negative amounts."""
        with pytest.raises(ValueError):
            player.remove_money(-100)

    def test_force_remove_money(self, player: Player) -> None:
        """force_remove_money should allow negative balance."""
        player.force_remove_money(2000)
        assert player.money == STARTING_MONEY - 2000

    def test_can_afford(self, player: Player) -> None:
        """can_afford should check balance."""
        assert player.can_afford(100)
        assert player.can_afford(STARTING_MONEY)
        assert not player.can_afford(STARTING_MONEY + 1)


class TestPlayerJail:
    """Tests for player jail functionality."""

    def test_go_to_jail(self, player: Player) -> None:
        """go_to_jail should set jail state."""
        player.position = 30  # Go To Jail space
        player.go_to_jail()
        assert player.position == JAIL_POSITION
        assert player.in_jail
        assert player.jail_turns == 0

    def test_leave_jail(self, player: Player) -> None:
        """leave_jail should clear jail state."""
        player.go_to_jail()
        player.jail_turns = 2
        player.leave_jail()
        assert not player.in_jail
        assert player.jail_turns == 0

    def test_increment_jail_turns(self, player: Player) -> None:
        """increment_jail_turns should count turns."""
        player.go_to_jail()
        assert player.increment_jail_turns() == 1
        assert player.increment_jail_turns() == 2
        assert player.increment_jail_turns() == 3

    def test_use_jail_card_success(self, player: Player) -> None:
        """use_jail_card should use card and leave jail."""
        player.go_to_jail()
        player.jail_cards = 1
        assert player.use_jail_card()
        assert not player.in_jail
        assert player.jail_cards == 0

    def test_use_jail_card_no_card(self, player: Player) -> None:
        """use_jail_card should fail without cards."""
        player.go_to_jail()
        assert not player.use_jail_card()
        assert player.in_jail

    def test_add_jail_card(self, player: Player) -> None:
        """add_jail_card should increase count."""
        assert player.jail_cards == 0
        player.add_jail_card()
        assert player.jail_cards == 1
        player.add_jail_card()
        assert player.jail_cards == 2


class TestPlayerBankruptcy:
    """Tests for player bankruptcy."""

    def test_declare_bankrupt(self, player: Player) -> None:
        """declare_bankrupt should set state."""
        player.declare_bankrupt()
        assert player.bankrupt
        assert player.money == 0

    def test_is_active(self, player: Player) -> None:
        """is_active should reflect bankruptcy."""
        assert player.is_active()
        player.declare_bankrupt()
        assert not player.is_active()


class TestPlayerSerialization:
    """Tests for player serialization."""

    def test_to_dict(self, player: Player) -> None:
        """to_dict should serialize all fields."""
        player.money = 1000
        player.position = 24
        player.jail_cards = 1
        player.in_jail = True
        player.jail_turns = 2

        data = player.to_dict()
        assert data["id"] == 0
        assert data["name"] == "Test Player"
        assert data["money"] == 1000
        assert data["position"] == 24
        assert data["jail_cards"] == 1
        assert data["in_jail"] is True
        assert data["jail_turns"] == 2
        assert data["bankrupt"] is False

    def test_from_dict(self) -> None:
        """from_dict should deserialize correctly."""
        data = {
            "id": 1,
            "name": "Loaded Player",
            "money": 2000,
            "position": 15,
            "jail_cards": 2,
            "in_jail": True,
            "jail_turns": 1,
            "bankrupt": False,
        }
        player = Player.from_dict(data)
        assert player.id == 1
        assert player.name == "Loaded Player"
        assert player.money == 2000
        assert player.position == 15
        assert player.jail_cards == 2
        assert player.in_jail
        assert player.jail_turns == 1
        assert not player.bankrupt

    def test_from_dict_defaults(self) -> None:
        """from_dict should handle missing optional fields."""
        data = {"id": 0, "name": "Minimal"}
        player = Player.from_dict(data)
        assert player.money == STARTING_MONEY
        assert player.position == 0


class TestPlayerStr:
    """Tests for player string representation."""

    def test_str_active(self, player: Player) -> None:
        """Active player should show status."""
        s = str(player)
        assert "Test Player" in s
        assert "$1500" in s
        assert "active" in s

    def test_str_in_jail(self, player: Player) -> None:
        """Jailed player should show status."""
        player.go_to_jail()
        s = str(player)
        assert "IN JAIL" in s

    def test_str_bankrupt(self, player: Player) -> None:
        """Bankrupt player should show status."""
        player.declare_bankrupt()
        s = str(player)
        assert "BANKRUPT" in s
