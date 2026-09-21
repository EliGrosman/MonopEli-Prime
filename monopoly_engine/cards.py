"""Chance and Community Chest card definitions.

This module defines all Chance and Community Chest cards and provides
a CardDeck class for managing shuffled decks.
"""

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .types import CardData, CardType

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class Card:
    """A Chance or Community Chest card.

    Cards are immutable and define their effects through data attributes.
    The actual effect execution is handled by the game engine.
    """

    id: int
    text: str
    card_type: CardType

    # Optional attributes based on card type
    move_to: int | None = None  # For MOVE cards
    move_to_nearest: str | None = None  # "railroad" or "utility" for MOVE_NEAREST
    move_spaces: int | None = None  # For MOVE_BACK (negative for backward)
    amount: int | None = None  # Money amount for COLLECT/PAY
    per_house: int | None = None  # For PAY_PER_BUILDING
    per_hotel: int | None = None  # For PAY_PER_BUILDING
    collect_go: bool = True  # Whether to collect $200 if passing GO

    def to_dict(self) -> CardData:
        """Serialize to JSON-compatible dict."""
        data: CardData = {
            "id": self.id,
            "text": self.text,
            "card_type": self.card_type.name,
        }
        if self.move_to is not None:
            data["move_to"] = self.move_to
        if self.move_to_nearest is not None:
            data["move_to_type"] = self.move_to_nearest
        if self.move_spaces is not None:
            data["move_spaces"] = self.move_spaces
        if self.amount is not None:
            data["amount"] = self.amount
        if self.per_house is not None:
            data["per_house"] = self.per_house
        if self.per_hotel is not None:
            data["per_hotel"] = self.per_hotel
        return data


# All 16 Chance cards
CHANCE_CARDS: tuple[Card, ...] = (
    Card(
        id=0,
        text="Advance to Boardwalk",
        card_type=CardType.MOVE,
        move_to=39,
        collect_go=False,
    ),
    Card(
        id=1,
        text="Advance to GO (Collect $200)",
        card_type=CardType.MOVE,
        move_to=0,
        collect_go=True,
    ),
    Card(
        id=2,
        text="Advance to Illinois Avenue. If you pass GO, collect $200",
        card_type=CardType.MOVE,
        move_to=24,
        collect_go=True,
    ),
    Card(
        id=3,
        text="Advance to St. Charles Place. If you pass GO, collect $200",
        card_type=CardType.MOVE,
        move_to=11,
        collect_go=True,
    ),
    Card(
        id=4,
        text=(
            "Advance to the nearest Railroad. If unowned, you may buy it from the Bank. "
            "If owned, pay owner twice the rental."
        ),
        card_type=CardType.MOVE_NEAREST,
        move_to_nearest="railroad",
    ),
    Card(
        id=5,
        text=(
            "Advance to the nearest Railroad. If unowned, you may buy it from the Bank. "
            "If owned, pay owner twice the rental."
        ),
        card_type=CardType.MOVE_NEAREST,
        move_to_nearest="railroad",
    ),
    Card(
        id=6,
        text=(
            "Advance to the nearest Utility. If unowned, you may buy it from the Bank. "
            "If owned, throw dice and pay owner 10 times amount thrown."
        ),
        card_type=CardType.MOVE_NEAREST,
        move_to_nearest="utility",
    ),
    Card(
        id=7,
        text="Bank pays you dividend of $50",
        card_type=CardType.COLLECT,
        amount=50,
    ),
    Card(
        id=8,
        text="Get Out of Jail Free",
        card_type=CardType.GET_OUT_OF_JAIL,
    ),
    Card(
        id=9,
        text="Go Back 3 Spaces",
        card_type=CardType.MOVE_BACK,
        move_spaces=-3,
    ),
    Card(
        id=10,
        text="Go to Jail. Go directly to Jail, do not pass GO, do not collect $200",
        card_type=CardType.GO_TO_JAIL,
    ),
    Card(
        id=11,
        text=(
            "Make general repairs on all your property. "
            "For each house pay $25. For each hotel pay $100."
        ),
        card_type=CardType.PAY_PER_BUILDING,
        per_house=25,
        per_hotel=100,
    ),
    Card(
        id=12,
        text="Speeding fine $15",
        card_type=CardType.PAY,
        amount=15,
    ),
    Card(
        id=13,
        text="Take a trip to Reading Railroad. If you pass GO, collect $200",
        card_type=CardType.MOVE,
        move_to=5,
        collect_go=True,
    ),
    Card(
        id=14,
        text="You have been elected Chairman of the Board. Pay each player $50",
        card_type=CardType.PAY_TO_PLAYERS,
        amount=50,
    ),
    Card(
        id=15,
        text="Your building loan matures. Collect $150",
        card_type=CardType.COLLECT,
        amount=150,
    ),
)


# All 16 Community Chest cards
COMMUNITY_CHEST_CARDS: tuple[Card, ...] = (
    Card(
        id=0,
        text="Advance to GO (Collect $200)",
        card_type=CardType.MOVE,
        move_to=0,
        collect_go=True,
    ),
    Card(
        id=1,
        text="Bank error in your favor. Collect $200",
        card_type=CardType.COLLECT,
        amount=200,
    ),
    Card(
        id=2,
        text="Doctor's fee. Pay $50",
        card_type=CardType.PAY,
        amount=50,
    ),
    Card(
        id=3,
        text="From sale of stock you get $50",
        card_type=CardType.COLLECT,
        amount=50,
    ),
    Card(
        id=4,
        text="Get Out of Jail Free",
        card_type=CardType.GET_OUT_OF_JAIL,
    ),
    Card(
        id=5,
        text="Go to Jail. Go directly to Jail, do not pass GO, do not collect $200",
        card_type=CardType.GO_TO_JAIL,
    ),
    Card(
        id=6,
        text="Holiday fund matures. Receive $100",
        card_type=CardType.COLLECT,
        amount=100,
    ),
    Card(
        id=7,
        text="Income tax refund. Collect $20",
        card_type=CardType.COLLECT,
        amount=20,
    ),
    Card(
        id=8,
        text="It is your birthday. Collect $10 from every player",
        card_type=CardType.COLLECT_FROM_PLAYERS,
        amount=10,
    ),
    Card(
        id=9,
        text="Life insurance matures. Collect $100",
        card_type=CardType.COLLECT,
        amount=100,
    ),
    Card(
        id=10,
        text="Pay hospital fees of $100",
        card_type=CardType.PAY,
        amount=100,
    ),
    Card(
        id=11,
        text="Pay school fees of $50",
        card_type=CardType.PAY,
        amount=50,
    ),
    Card(
        id=12,
        text="Receive $25 consultancy fee",
        card_type=CardType.COLLECT,
        amount=25,
    ),
    Card(
        id=13,
        text="You are assessed for street repair. $40 per house. $115 per hotel",
        card_type=CardType.PAY_PER_BUILDING,
        per_house=40,
        per_hotel=115,
    ),
    Card(
        id=14,
        text="You have won second prize in a beauty contest. Collect $10",
        card_type=CardType.COLLECT,
        amount=10,
    ),
    Card(
        id=15,
        text="You inherit $100",
        card_type=CardType.COLLECT,
        amount=100,
    ),
)


@dataclass
class CardDeck:
    """A shuffled deck of cards with discard pile.

    The deck automatically reshuffles when empty (except for
    Get Out of Jail Free cards which are held by players).
    """

    cards: list[Card] = field(default_factory=list)
    discard: list[Card] = field(default_factory=list)
    _rng: random.Random = field(default_factory=random.Random, repr=False)
    _auto_shuffle: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        """Shuffle the deck after initialization if auto_shuffle is enabled."""
        if self._auto_shuffle and self.cards:
            self.shuffle()

    @classmethod
    def create_chance_deck(cls, seed: int | None = None) -> "CardDeck":
        """Create a shuffled Chance deck."""
        rng = random.Random(seed) if seed is not None else random.Random()
        deck = cls(cards=list(CHANCE_CARDS), _rng=rng, _auto_shuffle=False)
        deck.shuffle()
        return deck

    @classmethod
    def create_community_chest_deck(cls, seed: int | None = None) -> "CardDeck":
        """Create a shuffled Community Chest deck."""
        rng = random.Random(seed) if seed is not None else random.Random()
        deck = cls(cards=list(COMMUNITY_CHEST_CARDS), _rng=rng, _auto_shuffle=False)
        deck.shuffle()
        return deck

    def shuffle(self) -> None:
        """Shuffle the draw pile."""
        self._rng.shuffle(self.cards)

    def draw(self) -> Card:
        """Draw a card from the deck.

        If the deck is empty, reshuffles the discard pile.
        Get Out of Jail Free cards are not added to discard.
        """
        if not self.cards:
            if not self.discard:
                raise RuntimeError("No cards available in deck or discard pile")
            self.cards = self.discard
            self.discard = []
            self.shuffle()

        card = self.cards.pop()

        # Get Out of Jail Free cards are kept by the player
        if card.card_type != CardType.GET_OUT_OF_JAIL:
            self.discard.append(card)

        return card

    def return_jail_card(self, card: Card) -> None:
        """Return a Get Out of Jail Free card to the deck.

        Called when a player uses their jail card.
        """
        if card.card_type != CardType.GET_OUT_OF_JAIL:
            raise ValueError("Can only return Get Out of Jail Free cards")
        self.discard.append(card)

    def peek(self) -> Card | None:
        """Peek at the top card without drawing it."""
        if not self.cards:
            return None
        return self.cards[-1]

    def cards_remaining(self) -> int:
        """Return number of cards in draw pile."""
        return len(self.cards)

    def total_cards(self) -> int:
        """Return total number of cards in deck and discard."""
        return len(self.cards) + len(self.discard)

    def to_dict(self) -> dict[str, list[int]]:
        """Serialize deck state to JSON-compatible dict."""
        return {
            "draw_pile": [card.id for card in self.cards],
            "discard_pile": [card.id for card in self.discard],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, list[int]],
        card_source: tuple[Card, ...],
        seed: int | None = None,
    ) -> "CardDeck":
        """Create CardDeck from serialized dict.

        Args:
            data: Serialized deck data
            card_source: The full set of cards (CHANCE_CARDS or COMMUNITY_CHEST_CARDS)
            seed: Optional random seed
        """
        card_map = {card.id: card for card in card_source}
        deck = cls(
            cards=[card_map[card_id] for card_id in data.get("draw_pile", [])],
            discard=[card_map[card_id] for card_id in data.get("discard_pile", [])],
            _auto_shuffle=False,
        )
        if seed is not None:
            deck._rng = random.Random(seed)
        return deck


def get_chance_card(card_id: int) -> Card:
    """Get a Chance card by ID."""
    for card in CHANCE_CARDS:
        if card.id == card_id:
            return card
    raise ValueError(f"Invalid Chance card ID: {card_id}")


def get_community_chest_card(card_id: int) -> Card:
    """Get a Community Chest card by ID."""
    for card in COMMUNITY_CHEST_CARDS:
        if card.id == card_id:
            return card
    raise ValueError(f"Invalid Community Chest card ID: {card_id}")
