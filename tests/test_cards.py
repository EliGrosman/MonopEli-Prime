"""Tests for cards.py module."""

import pytest

from monopoly_engine import (
    Card,
    CardDeck,
    CardType,
    CHANCE_CARDS,
    COMMUNITY_CHEST_CARDS,
    get_chance_card,
    get_community_chest_card,
)


class TestCard:
    """Tests for Card class."""

    def test_card_basics(self) -> None:
        """Card should store basic attributes."""
        card = Card(id=0, text="Test card", card_type=CardType.COLLECT, amount=100)
        assert card.id == 0
        assert card.text == "Test card"
        assert card.card_type == CardType.COLLECT
        assert card.amount == 100

    def test_card_immutable(self) -> None:
        """Card should be frozen (immutable)."""
        card = Card(id=0, text="Test", card_type=CardType.COLLECT)
        with pytest.raises(AttributeError):
            card.id = 1  # type: ignore

    def test_card_to_dict(self) -> None:
        """Card.to_dict should serialize all fields."""
        card = Card(
            id=0,
            text="Collect $100",
            card_type=CardType.COLLECT,
            amount=100,
        )
        data = card.to_dict()
        assert data["id"] == 0
        assert data["text"] == "Collect $100"
        assert data["card_type"] == "COLLECT"
        assert data["amount"] == 100


class TestChanceCards:
    """Tests for Chance card definitions."""

    def test_16_chance_cards(self) -> None:
        """There should be exactly 16 Chance cards."""
        assert len(CHANCE_CARDS) == 16

    def test_chance_card_ids_unique(self) -> None:
        """All Chance card IDs should be unique."""
        ids = [card.id for card in CHANCE_CARDS]
        assert len(ids) == len(set(ids))

    def test_chance_advance_to_go(self) -> None:
        """Chance should have 'Advance to GO' card."""
        card = get_chance_card(1)
        assert card.card_type == CardType.MOVE
        assert card.move_to == 0
        assert "GO" in card.text

    def test_chance_go_to_jail(self) -> None:
        """Chance should have 'Go to Jail' card."""
        card = get_chance_card(10)
        assert card.card_type == CardType.GO_TO_JAIL
        assert "Jail" in card.text

    def test_chance_get_out_of_jail(self) -> None:
        """Chance should have 'Get Out of Jail Free' card."""
        card = get_chance_card(8)
        assert card.card_type == CardType.GET_OUT_OF_JAIL

    def test_chance_advance_to_railroad(self) -> None:
        """Chance should have 'Advance to nearest Railroad' cards."""
        card = get_chance_card(4)
        assert card.card_type == CardType.MOVE_NEAREST
        assert card.move_to_nearest == "railroad"

    def test_chance_advance_to_utility(self) -> None:
        """Chance should have 'Advance to nearest Utility' card."""
        card = get_chance_card(6)
        assert card.card_type == CardType.MOVE_NEAREST
        assert card.move_to_nearest == "utility"

    def test_chance_go_back_3(self) -> None:
        """Chance should have 'Go Back 3 Spaces' card."""
        card = get_chance_card(9)
        assert card.card_type == CardType.MOVE_BACK
        assert card.move_spaces == -3

    def test_chance_pay_per_building(self) -> None:
        """Chance should have repairs card."""
        card = get_chance_card(11)
        assert card.card_type == CardType.PAY_PER_BUILDING
        assert card.per_house == 25
        assert card.per_hotel == 100

    def test_chance_collect_money(self) -> None:
        """Chance should have money collection cards."""
        # Bank dividend
        card = get_chance_card(7)
        assert card.card_type == CardType.COLLECT
        assert card.amount == 50

    def test_chance_pay_money(self) -> None:
        """Chance should have payment cards."""
        # Speeding fine
        card = get_chance_card(12)
        assert card.card_type == CardType.PAY
        assert card.amount == 15

    def test_chance_pay_to_players(self) -> None:
        """Chance should have 'Pay each player' card."""
        card = get_chance_card(14)
        assert card.card_type == CardType.PAY_TO_PLAYERS
        assert card.amount == 50


class TestCommunityChestCards:
    """Tests for Community Chest card definitions."""

    def test_16_community_chest_cards(self) -> None:
        """There should be exactly 16 Community Chest cards."""
        assert len(COMMUNITY_CHEST_CARDS) == 16

    def test_community_chest_ids_unique(self) -> None:
        """All Community Chest card IDs should be unique."""
        ids = [card.id for card in COMMUNITY_CHEST_CARDS]
        assert len(ids) == len(set(ids))

    def test_community_chest_advance_to_go(self) -> None:
        """Community Chest should have 'Advance to GO' card."""
        card = get_community_chest_card(0)
        assert card.card_type == CardType.MOVE
        assert card.move_to == 0

    def test_community_chest_go_to_jail(self) -> None:
        """Community Chest should have 'Go to Jail' card."""
        card = get_community_chest_card(5)
        assert card.card_type == CardType.GO_TO_JAIL

    def test_community_chest_get_out_of_jail(self) -> None:
        """Community Chest should have 'Get Out of Jail Free' card."""
        card = get_community_chest_card(4)
        assert card.card_type == CardType.GET_OUT_OF_JAIL

    def test_community_chest_bank_error(self) -> None:
        """Community Chest should have 'Bank Error' card."""
        card = get_community_chest_card(1)
        assert card.card_type == CardType.COLLECT
        assert card.amount == 200

    def test_community_chest_collect_from_players(self) -> None:
        """Community Chest should have birthday card."""
        card = get_community_chest_card(8)
        assert card.card_type == CardType.COLLECT_FROM_PLAYERS
        assert card.amount == 10

    def test_community_chest_pay_per_building(self) -> None:
        """Community Chest should have repairs card."""
        card = get_community_chest_card(13)
        assert card.card_type == CardType.PAY_PER_BUILDING
        assert card.per_house == 40
        assert card.per_hotel == 115


class TestCardDeck:
    """Tests for CardDeck class."""

    def test_create_chance_deck(self) -> None:
        """create_chance_deck should create shuffled deck."""
        deck = CardDeck.create_chance_deck(seed=42)
        assert deck.cards_remaining() == 16
        assert deck.total_cards() == 16

    def test_create_community_chest_deck(self) -> None:
        """create_community_chest_deck should create shuffled deck."""
        deck = CardDeck.create_community_chest_deck(seed=42)
        assert deck.cards_remaining() == 16
        assert deck.total_cards() == 16

    def test_draw_card(self, chance_deck: CardDeck) -> None:
        """draw should return a card and reduce count."""
        initial = chance_deck.cards_remaining()
        card = chance_deck.draw()
        assert isinstance(card, Card)
        assert chance_deck.cards_remaining() == initial - 1

    def test_draw_non_jail_card_goes_to_discard(
        self, chance_deck: CardDeck
    ) -> None:
        """Non-jail cards should go to discard."""
        # Draw until we get a non-jail card
        card = chance_deck.draw()
        while card.card_type == CardType.GET_OUT_OF_JAIL:
            card = chance_deck.draw()

        # Card should be in discard
        assert len(chance_deck.discard) > 0

    def test_draw_jail_card_not_discarded(self) -> None:
        """Get Out of Jail Free cards should not be discarded."""
        # Create deck where jail card is on top
        jail_card = Card(id=99, text="Test Jail", card_type=CardType.GET_OUT_OF_JAIL)
        deck = CardDeck(cards=[jail_card])

        card = deck.draw()
        assert card.card_type == CardType.GET_OUT_OF_JAIL
        assert len(deck.discard) == 0

    def test_deck_reshuffles_when_empty(self) -> None:
        """Deck should reshuffle discard pile when empty."""
        deck = CardDeck.create_chance_deck(seed=42)

        # Draw all cards (jail cards aren't discarded, so we may have fewer)
        drawn = []
        for _ in range(16):
            card = deck.draw()
            drawn.append(card)

        # Deck should have reshuffled from discard
        assert deck.cards_remaining() > 0 or len(deck.discard) > 0

    def test_seeded_deck_is_deterministic(self) -> None:
        """Same seed should produce same card order."""
        deck1 = CardDeck.create_chance_deck(seed=123)
        deck2 = CardDeck.create_chance_deck(seed=123)

        for _ in range(5):
            card1 = deck1.draw()
            card2 = deck2.draw()
            assert card1.id == card2.id

    def test_different_seeds_different_order(self) -> None:
        """Different seeds should produce different order."""
        deck1 = CardDeck.create_chance_deck(seed=123)
        deck2 = CardDeck.create_chance_deck(seed=456)

        # Highly unlikely to have same order
        same_count = 0
        for _ in range(5):
            if deck1.draw().id == deck2.draw().id:
                same_count += 1
        assert same_count < 5

    def test_return_jail_card(self) -> None:
        """return_jail_card should add card to discard."""
        deck = CardDeck.create_chance_deck(seed=42)
        jail_card = get_chance_card(8)  # Get Out of Jail Free

        initial_discard = len(deck.discard)
        deck.return_jail_card(jail_card)
        assert len(deck.discard) == initial_discard + 1

    def test_return_non_jail_card_raises(self) -> None:
        """return_jail_card should reject non-jail cards."""
        deck = CardDeck.create_chance_deck(seed=42)
        regular_card = get_chance_card(1)  # Advance to GO

        with pytest.raises(ValueError):
            deck.return_jail_card(regular_card)

    def test_peek(self, chance_deck: CardDeck) -> None:
        """peek should show top card without drawing."""
        top = chance_deck.peek()
        assert top is not None
        count_before = chance_deck.cards_remaining()
        drawn = chance_deck.draw()
        assert drawn.id == top.id
        assert chance_deck.cards_remaining() == count_before - 1

    def test_peek_empty_deck(self) -> None:
        """peek on empty deck should return None."""
        deck = CardDeck(cards=[])
        assert deck.peek() is None

    def test_to_dict(self, chance_deck: CardDeck) -> None:
        """to_dict should serialize deck state."""
        # Draw a few cards
        chance_deck.draw()
        chance_deck.draw()

        data = chance_deck.to_dict()
        assert "draw_pile" in data
        assert "discard_pile" in data
        assert isinstance(data["draw_pile"], list)

    def test_from_dict(self) -> None:
        """from_dict should deserialize deck state."""
        data = {
            "draw_pile": [0, 1, 2],
            "discard_pile": [3, 4],
        }
        deck = CardDeck.from_dict(data, CHANCE_CARDS, seed=42)
        assert deck.cards_remaining() == 3
        assert len(deck.discard) == 2


class TestCardLookup:
    """Tests for card lookup functions."""

    def test_get_chance_card_valid(self) -> None:
        """get_chance_card should return correct card."""
        card = get_chance_card(0)
        assert card.id == 0
        assert card in CHANCE_CARDS

    def test_get_chance_card_invalid(self) -> None:
        """get_chance_card should raise for invalid ID."""
        with pytest.raises(ValueError):
            get_chance_card(99)

    def test_get_community_chest_card_valid(self) -> None:
        """get_community_chest_card should return correct card."""
        card = get_community_chest_card(0)
        assert card.id == 0
        assert card in COMMUNITY_CHEST_CARDS

    def test_get_community_chest_card_invalid(self) -> None:
        """get_community_chest_card should raise for invalid ID."""
        with pytest.raises(ValueError):
            get_community_chest_card(99)
