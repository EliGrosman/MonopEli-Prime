"""Player state management for the Monopoly game engine.

This module handles individual player state including position, money,
jail status, and bankruptcy.
"""

from dataclasses import dataclass, field

from .types import BOARD_SIZE, JAIL_POSITION, STARTING_MONEY, PlayerStateData


@dataclass
class Player:
    """Mutable player state.

    Tracks a player's position, money, jail status, and other state.
    """

    id: int
    name: str
    money: int = STARTING_MONEY
    position: int = 0
    jail_cards: int = 0
    in_jail: bool = False
    jail_turns: int = 0
    bankrupt: bool = False

    # Internal state tracking
    _net_worth_cache: int | None = field(default=None, init=False, repr=False)

    def move(self, spaces: int) -> tuple[int, bool]:
        """Move player by a number of spaces.

        Args:
            spaces: Number of spaces to move (can be negative for "Go Back" cards)

        Returns:
            Tuple of (new_position, passed_go)
        """
        old_pos = self.position
        new_pos = (self.position + spaces) % BOARD_SIZE

        # Check if passed GO (only for forward movement)
        passed_go = False
        if spaces > 0 and new_pos < old_pos:
            passed_go = True

        self.position = new_pos
        self._net_worth_cache = None
        return new_pos, passed_go

    def move_to(self, position: int) -> bool:
        """Move player directly to a position.

        Args:
            position: Target position (0-39)

        Returns:
            True if player passed GO
        """
        old_pos = self.position
        self.position = position % BOARD_SIZE

        # Passed GO if new position is before old position
        # (assuming we always move forward around the board)
        passed_go = self.position < old_pos and self.position != JAIL_POSITION

        self._net_worth_cache = None
        return passed_go

    def add_money(self, amount: int) -> None:
        """Add money to player's balance."""
        if amount < 0:
            raise ValueError("Cannot add negative money. Use remove_money() instead.")
        self.money += amount
        self._net_worth_cache = None

    def remove_money(self, amount: int) -> bool:
        """Remove money from player's balance.

        Args:
            amount: Amount to remove

        Returns:
            True if player had sufficient funds, False otherwise
        """
        if amount < 0:
            raise ValueError("Cannot remove negative money. Use add_money() instead.")

        if self.money >= amount:
            self.money -= amount
            self._net_worth_cache = None
            return True
        return False

    def force_remove_money(self, amount: int) -> None:
        """Remove money from player's balance, allowing negative balance.

        Used when player owes money but may need to sell assets.
        """
        if amount < 0:
            raise ValueError("Cannot remove negative money.")
        self.money -= amount
        self._net_worth_cache = None

    def can_afford(self, amount: int) -> bool:
        """Check if player can afford an amount."""
        return self.money >= amount

    def go_to_jail(self) -> None:
        """Send player to jail."""
        self.position = JAIL_POSITION
        self.in_jail = True
        self.jail_turns = 0
        self._net_worth_cache = None

    def leave_jail(self) -> None:
        """Release player from jail."""
        self.in_jail = False
        self.jail_turns = 0

    def increment_jail_turns(self) -> int:
        """Increment jail turn counter and return new value."""
        self.jail_turns += 1
        return self.jail_turns

    def use_jail_card(self) -> bool:
        """Use a Get Out of Jail Free card.

        Returns:
            True if card was available and used, False otherwise
        """
        if self.jail_cards > 0:
            self.jail_cards -= 1
            self.leave_jail()
            return True
        return False

    def add_jail_card(self) -> None:
        """Add a Get Out of Jail Free card."""
        self.jail_cards += 1

    def declare_bankrupt(self) -> None:
        """Mark player as bankrupt."""
        self.bankrupt = True
        self.money = 0

    def is_active(self) -> bool:
        """Check if player is still in the game."""
        return not self.bankrupt

    def to_dict(self) -> PlayerStateData:
        """Serialize player to JSON-compatible dict."""
        return PlayerStateData(
            id=self.id,
            name=self.name,
            money=self.money,
            position=self.position,
            jail_cards=self.jail_cards,
            in_jail=self.in_jail,
            jail_turns=self.jail_turns,
            bankrupt=self.bankrupt,
        )

    @classmethod
    def from_dict(cls, data: PlayerStateData) -> "Player":
        """Create Player from serialized dict."""
        return cls(
            id=data["id"],
            name=data["name"],
            money=data.get("money", STARTING_MONEY),
            position=data.get("position", 0),
            jail_cards=data.get("jail_cards", 0),
            in_jail=data.get("in_jail", False),
            jail_turns=data.get("jail_turns", 0),
            bankrupt=data.get("bankrupt", False),
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        status = "BANKRUPT" if self.bankrupt else ("IN JAIL" if self.in_jail else "active")
        return f"{self.name} (${self.money}, pos {self.position}, {status})"
