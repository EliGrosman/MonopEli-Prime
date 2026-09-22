"""Comprehensive tests for the action_space module.

This module tests the ActionEncoder class including:
- Encode/decode roundtrips for all 11 action types
- Position index mappings
- Action mask generation for various game states
- Action name generation
- Edge cases and error handling
- Property-based tests using hypothesis
"""

from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from monopoly_engine import (
    BuildHotel,
    BuildHouse,
    BuyProperty,
    EndTurn,
    MonopolyGame,
    MortgageProperty,
    PayJailFine,
    SellHotel,
    SellHouse,
    UnmortgageProperty,
    UseJailCard,
)
from monopoly_engine.actions import PassBuy
from monopoly_gym.action_space import (
    _BUYABLE_TO_INDEX,
    _DEVELOPABLE_TO_INDEX,
    ACTION_SPACE_SIZE,
    BUYABLE_POSITIONS,
    DEVELOPABLE_POSITIONS,
    GAMEPLAY_ACTION_SPACE_SIZE,
    OFFSET_BUILD_HOTEL,
    OFFSET_BUILD_HOUSE,
    OFFSET_BUY_PROPERTY,
    OFFSET_END_TURN,
    OFFSET_MORTGAGE,
    OFFSET_PASS_BUY,
    OFFSET_PAY_JAIL_FINE,
    OFFSET_SELL_HOTEL,
    OFFSET_SELL_HOUSE,
    OFFSET_UNMORTGAGE,
    OFFSET_USE_JAIL_CARD,
    ActionEncoder,
)

if TYPE_CHECKING:
    pass


class TestActionSpaceConstants:
    """Tests for action space constants and mappings."""

    def test_action_space_size(self) -> None:
        """Gameplay action space should have 158 dimensions, full space 907."""
        assert GAMEPLAY_ACTION_SPACE_SIZE == 158
        assert ACTION_SPACE_SIZE == 158  # With Phase 2.5a trades

    def test_buyable_positions_count(self) -> None:
        """Should have exactly 28 buyable positions."""
        assert len(BUYABLE_POSITIONS) == 28

    def test_developable_positions_count(self) -> None:
        """Should have exactly 22 developable positions."""
        assert len(DEVELOPABLE_POSITIONS) == 22

    def test_developable_subset_of_buyable(self) -> None:
        """All developable positions should be buyable positions."""
        for pos in DEVELOPABLE_POSITIONS:
            assert pos in BUYABLE_POSITIONS, f"Position {pos} is developable but not buyable"

    def test_buyable_to_index_mapping(self) -> None:
        """BUYABLE_TO_INDEX mapping should be consistent."""
        for idx, pos in enumerate(BUYABLE_POSITIONS):
            assert _BUYABLE_TO_INDEX[pos] == idx

    def test_developable_to_index_mapping(self) -> None:
        """DEVELOPABLE_TO_INDEX mapping should be consistent."""
        for idx, pos in enumerate(DEVELOPABLE_POSITIONS):
            assert _DEVELOPABLE_TO_INDEX[pos] == idx

    def test_offset_ranges_non_overlapping(self) -> None:
        """Action offset ranges should not overlap."""
        ranges = [
            (OFFSET_BUY_PROPERTY, OFFSET_BUY_PROPERTY + 1),
            (OFFSET_PASS_BUY, OFFSET_PASS_BUY + 1),
            (OFFSET_BUILD_HOUSE, OFFSET_BUILD_HOTEL),
            (OFFSET_BUILD_HOTEL, OFFSET_SELL_HOUSE),
            (OFFSET_SELL_HOUSE, OFFSET_SELL_HOTEL),
            (OFFSET_SELL_HOTEL, OFFSET_MORTGAGE),
            (OFFSET_MORTGAGE, OFFSET_UNMORTGAGE),
            (OFFSET_UNMORTGAGE, OFFSET_END_TURN),
            (OFFSET_END_TURN, OFFSET_END_TURN + 1),
            (OFFSET_USE_JAIL_CARD, OFFSET_USE_JAIL_CARD + 1),
            (OFFSET_PAY_JAIL_FINE, OFFSET_PAY_JAIL_FINE + 1),
        ]

        # Sort by start position
        ranges.sort()

        # Check no overlaps
        for i in range(len(ranges) - 1):
            assert ranges[i][1] <= ranges[i + 1][0], (
                f"Range {ranges[i]} overlaps with {ranges[i + 1]}"
            )

    def test_offset_ranges_cover_action_space(self) -> None:
        """Action offsets should cover the entire action space."""
        # Last action should be at index 148
        assert OFFSET_PAY_JAIL_FINE == 148

    def test_buyable_positions_valid_board_positions(self) -> None:
        """All buyable positions should be valid board positions (0-39)."""
        for pos in BUYABLE_POSITIONS:
            assert 0 <= pos <= 39, f"Invalid board position: {pos}"

    def test_developable_excludes_railroads_utilities(self) -> None:
        """Developable positions should exclude railroads and utilities."""
        railroads = {5, 15, 25, 35}
        utilities = {12, 28}

        for pos in DEVELOPABLE_POSITIONS:
            assert pos not in railroads, f"Railroad at {pos} should not be developable"
            assert pos not in utilities, f"Utility at {pos} should not be developable"


class TestActionEncoderInit:
    """Tests for ActionEncoder initialization."""

    def test_init_sets_action_space_size(self) -> None:
        """Encoder should have correct action space size attribute."""
        encoder = ActionEncoder()
        assert encoder.action_space_size == 158

    def test_encoder_is_stateless(self) -> None:
        """Multiple encoder instances should be equivalent."""
        encoder1 = ActionEncoder()
        encoder2 = ActionEncoder()
        assert encoder1.action_space_size == encoder2.action_space_size


class TestEncodeAction:
    """Tests for encoding Action objects to indices."""

    def test_encode_buy_property(self) -> None:
        """BuyProperty should encode to index 0."""
        encoder = ActionEncoder()
        action = BuyProperty(player_id=0, property_id=1)
        assert encoder.encode(action) == OFFSET_BUY_PROPERTY

    def test_encode_end_turn(self) -> None:
        """EndTurn should encode to index 146."""
        encoder = ActionEncoder()
        action = EndTurn(player_id=0)
        assert encoder.encode(action) == OFFSET_END_TURN

    def test_encode_use_jail_card(self) -> None:
        """UseJailCard should encode to index 147."""
        encoder = ActionEncoder()
        action = UseJailCard(player_id=0)
        assert encoder.encode(action) == OFFSET_USE_JAIL_CARD

    def test_encode_pay_jail_fine(self) -> None:
        """PayJailFine should encode to index 148."""
        encoder = ActionEncoder()
        action = PayJailFine(player_id=0)
        assert encoder.encode(action) == OFFSET_PAY_JAIL_FINE

    @pytest.mark.parametrize("prop_idx,prop_pos", enumerate(DEVELOPABLE_POSITIONS))
    def test_encode_build_house_all_positions(self, prop_idx: int, prop_pos: int) -> None:
        """BuildHouse should encode correctly for all developable positions."""
        encoder = ActionEncoder()
        action = BuildHouse(player_id=0, property_id=prop_pos)
        expected = OFFSET_BUILD_HOUSE + prop_idx
        assert encoder.encode(action) == expected

    @pytest.mark.parametrize("prop_idx,prop_pos", enumerate(DEVELOPABLE_POSITIONS))
    def test_encode_build_hotel_all_positions(self, prop_idx: int, prop_pos: int) -> None:
        """BuildHotel should encode correctly for all developable positions."""
        encoder = ActionEncoder()
        action = BuildHotel(player_id=0, property_id=prop_pos)
        expected = OFFSET_BUILD_HOTEL + prop_idx
        assert encoder.encode(action) == expected

    @pytest.mark.parametrize("prop_idx,prop_pos", enumerate(DEVELOPABLE_POSITIONS))
    def test_encode_sell_house_all_positions(self, prop_idx: int, prop_pos: int) -> None:
        """SellHouse should encode correctly for all developable positions."""
        encoder = ActionEncoder()
        action = SellHouse(player_id=0, property_id=prop_pos)
        expected = OFFSET_SELL_HOUSE + prop_idx
        assert encoder.encode(action) == expected

    @pytest.mark.parametrize("prop_idx,prop_pos", enumerate(DEVELOPABLE_POSITIONS))
    def test_encode_sell_hotel_all_positions(self, prop_idx: int, prop_pos: int) -> None:
        """SellHotel should encode correctly for all developable positions."""
        encoder = ActionEncoder()
        action = SellHotel(player_id=0, property_id=prop_pos)
        expected = OFFSET_SELL_HOTEL + prop_idx
        assert encoder.encode(action) == expected

    @pytest.mark.parametrize("prop_idx,prop_pos", enumerate(BUYABLE_POSITIONS))
    def test_encode_mortgage_all_positions(self, prop_idx: int, prop_pos: int) -> None:
        """MortgageProperty should encode correctly for all buyable positions."""
        encoder = ActionEncoder()
        action = MortgageProperty(player_id=0, property_id=prop_pos)
        expected = OFFSET_MORTGAGE + prop_idx
        assert encoder.encode(action) == expected

    @pytest.mark.parametrize("prop_idx,prop_pos", enumerate(BUYABLE_POSITIONS))
    def test_encode_unmortgage_all_positions(self, prop_idx: int, prop_pos: int) -> None:
        """UnmortgageProperty should encode correctly for all buyable positions."""
        encoder = ActionEncoder()
        action = UnmortgageProperty(player_id=0, property_id=prop_pos)
        expected = OFFSET_UNMORTGAGE + prop_idx
        assert encoder.encode(action) == expected

    def test_encode_invalid_build_house_position(self) -> None:
        """BuildHouse on non-developable position should raise ValueError."""
        encoder = ActionEncoder()
        # Position 5 is a railroad, not developable
        action = BuildHouse(player_id=0, property_id=5)
        with pytest.raises(ValueError, match="Cannot build"):
            encoder.encode(action)

    def test_encode_invalid_mortgage_position(self) -> None:
        """MortgageProperty on non-buyable position should raise ValueError."""
        encoder = ActionEncoder()
        # Position 0 is GO, not buyable
        action = MortgageProperty(player_id=0, property_id=0)
        with pytest.raises(ValueError, match="Cannot mortgage"):
            encoder.encode(action)


class TestDecodeAction:
    """Tests for decoding action indices to Action objects."""

    def test_decode_buy_property(self) -> None:
        """Index 0 should decode to BuyProperty at player position."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)
        # Move player to a property position
        game.players[0].position = 1

        action = encoder.decode(OFFSET_BUY_PROPERTY, player_id=0, game=game)
        assert isinstance(action, BuyProperty)
        assert action.player_id == 0
        assert action.property_id == 1  # Player's current position

    def test_decode_pass_buy(self) -> None:
        encoder = ActionEncoder()
        game = MonopolyGame(2, seed=42)
        assert isinstance(encoder.decode(OFFSET_PASS_BUY, 0, game), PassBuy)

    def test_decode_end_turn(self) -> None:
        """Index 146 should decode to EndTurn."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        action = encoder.decode(OFFSET_END_TURN, player_id=0, game=game)
        assert isinstance(action, EndTurn)
        assert action.player_id == 0

    def test_decode_use_jail_card(self) -> None:
        """Index 147 should decode to UseJailCard."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        action = encoder.decode(OFFSET_USE_JAIL_CARD, player_id=0, game=game)
        assert isinstance(action, UseJailCard)
        assert action.player_id == 0

    def test_decode_pay_jail_fine(self) -> None:
        """Index 148 should decode to PayJailFine."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        action = encoder.decode(OFFSET_PAY_JAIL_FINE, player_id=0, game=game)
        assert isinstance(action, PayJailFine)
        assert action.player_id == 0

    @pytest.mark.parametrize("prop_idx,prop_pos", enumerate(DEVELOPABLE_POSITIONS))
    def test_decode_build_house_all_positions(self, prop_idx: int, prop_pos: int) -> None:
        """BuildHouse should decode correctly for all developable positions."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)
        action_idx = OFFSET_BUILD_HOUSE + prop_idx

        action = encoder.decode(action_idx, player_id=0, game=game)
        assert isinstance(action, BuildHouse)
        assert action.property_id == prop_pos

    @pytest.mark.parametrize("prop_idx,prop_pos", enumerate(DEVELOPABLE_POSITIONS))
    def test_decode_build_hotel_all_positions(self, prop_idx: int, prop_pos: int) -> None:
        """BuildHotel should decode correctly for all developable positions."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)
        action_idx = OFFSET_BUILD_HOTEL + prop_idx

        action = encoder.decode(action_idx, player_id=0, game=game)
        assert isinstance(action, BuildHotel)
        assert action.property_id == prop_pos

    @pytest.mark.parametrize("prop_idx,prop_pos", enumerate(BUYABLE_POSITIONS))
    def test_decode_mortgage_all_positions(self, prop_idx: int, prop_pos: int) -> None:
        """MortgageProperty should decode correctly for all buyable positions."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)
        action_idx = OFFSET_MORTGAGE + prop_idx

        action = encoder.decode(action_idx, player_id=0, game=game)
        assert isinstance(action, MortgageProperty)
        assert action.property_id == prop_pos

    def test_decode_negative_index_raises(self) -> None:
        """Negative action index should raise ValueError."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        with pytest.raises(ValueError, match="out of range"):
            encoder.decode(-1, player_id=0, game=game)

    def test_decode_out_of_range_index_raises(self) -> None:
        encoder = ActionEncoder()
        game = MonopolyGame(2, seed=42)
        for index in (-1, 158, 907):
            with pytest.raises(ValueError):
                encoder.decode(index, 0, game)
        with pytest.raises(ValueError, match="Trading is disabled"):
            ActionEncoder(enable_trades=True)


class TestEncodeDecodeRoundtrip:
    """Tests for encode/decode roundtrip consistency."""

    def test_roundtrip_buy_property(self) -> None:
        """BuyProperty should roundtrip correctly."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)
        game.players[0].position = 1  # Mediterranean

        original = BuyProperty(player_id=0, property_id=1)
        encoded = encoder.encode(original)
        decoded = encoder.decode(encoded, player_id=0, game=game)

        assert isinstance(decoded, BuyProperty)
        assert decoded.property_id == original.property_id

    def test_roundtrip_end_turn(self) -> None:
        """EndTurn should roundtrip correctly."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        original = EndTurn(player_id=0)
        encoded = encoder.encode(original)
        decoded = encoder.decode(encoded, player_id=0, game=game)

        assert isinstance(decoded, EndTurn)

    @pytest.mark.parametrize("prop_pos", DEVELOPABLE_POSITIONS)
    def test_roundtrip_build_house(self, prop_pos: int) -> None:
        """BuildHouse should roundtrip for all developable properties."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        original = BuildHouse(player_id=0, property_id=prop_pos)
        encoded = encoder.encode(original)
        decoded = encoder.decode(encoded, player_id=0, game=game)

        assert isinstance(decoded, BuildHouse)
        assert decoded.property_id == original.property_id

    @pytest.mark.parametrize("prop_pos", BUYABLE_POSITIONS)
    def test_roundtrip_mortgage(self, prop_pos: int) -> None:
        """MortgageProperty should roundtrip for all buyable properties."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        original = MortgageProperty(player_id=0, property_id=prop_pos)
        encoded = encoder.encode(original)
        decoded = encoder.decode(encoded, player_id=0, game=game)

        assert isinstance(decoded, MortgageProperty)
        assert decoded.property_id == original.property_id


class TestGetActionMask:
    """Tests for action mask generation."""

    def test_mask_shape(self) -> None:
        """Action mask should have shape (158,)."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        mask = encoder.get_action_mask(game, player_id=0)
        assert mask.shape == (158,)
        assert mask.dtype == np.bool_

    def test_mask_end_turn_always_valid_for_current_player(self) -> None:
        encoder = ActionEncoder()
        game = MonopolyGame(2, seed=42)
        assert not encoder.get_action_mask(game, 0)[OFFSET_END_TURN]
        game.state.phase = "asset_management"
        game.state.roll_owed = False
        assert encoder.get_action_mask(game, 0)[OFFSET_END_TURN]

    def test_mask_all_invalid_for_non_current_player(self) -> None:
        """Non-current player should have limited valid actions."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        # Get the non-current player
        non_current = 1 - game.current_player

        mask = encoder.get_action_mask(game, player_id=non_current)

        # Buy, build, end turn should not be valid
        assert not mask[OFFSET_BUY_PROPERTY]
        assert not mask[OFFSET_END_TURN]

    def test_mask_all_invalid_for_bankrupt_player(self) -> None:
        """Bankrupt player should have no valid actions."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        # Make player 0 bankrupt
        game.players[0].bankrupt = True

        mask = encoder.get_action_mask(game, player_id=0)
        assert not mask.any()  # All actions invalid

    def test_mask_buy_property_when_on_unowned_property(self) -> None:
        """Buy property should be valid when on unowned property."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        # Move current player to Mediterranean (position 1)
        current = game.current_player
        game.players[current].position = 1
        game.state.phase = "purchase_decision"

        mask = encoder.get_action_mask(game, player_id=current)
        assert mask[OFFSET_BUY_PROPERTY]
        assert mask[OFFSET_PASS_BUY]

    def test_mask_buy_property_invalid_when_owned(self) -> None:
        """Buy property should be invalid when property is owned."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        # Move current player to Mediterranean and have someone own it
        current = game.current_player
        game.players[current].position = 1
        game.state.phase = "purchase_decision"
        game.property_manager.properties[1].owner = 1  # Player 1 owns it

        mask = encoder.get_action_mask(game, player_id=current)
        assert not mask[OFFSET_BUY_PROPERTY]

    def test_mask_buy_property_invalid_when_cannot_afford(self) -> None:
        """Buy property should be invalid when player cannot afford it."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        current = game.current_player
        game.players[current].position = 39  # Boardwalk ($400)
        game.state.phase = "purchase_decision"
        game.players[current].money = 100  # Not enough

        mask = encoder.get_action_mask(game, player_id=current)
        assert not mask[OFFSET_BUY_PROPERTY]
        # Pass buy should still be available
        assert mask[OFFSET_PASS_BUY]

    def test_mask_jail_card_when_in_jail_with_card(self) -> None:
        """Use jail card should be valid when in jail with card."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        current = game.current_player
        game.players[current].in_jail = True
        game.state.phase = "jail_decision"
        game.players[current].jail_cards = 1

        mask = encoder.get_action_mask(game, player_id=current)
        assert mask[OFFSET_USE_JAIL_CARD]

    def test_mask_jail_card_invalid_when_not_in_jail(self) -> None:
        """Use jail card should be invalid when not in jail."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        current = game.current_player
        game.players[current].in_jail = False
        game.players[current].jail_cards = 1

        mask = encoder.get_action_mask(game, player_id=current)
        assert not mask[OFFSET_USE_JAIL_CARD]

    def test_mask_jail_card_invalid_when_no_cards(self) -> None:
        """Use jail card should be invalid when no cards."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        current = game.current_player
        game.players[current].in_jail = True
        game.state.phase = "jail_decision"
        game.players[current].jail_cards = 0

        mask = encoder.get_action_mask(game, player_id=current)
        assert not mask[OFFSET_USE_JAIL_CARD]

    def test_mask_pay_jail_fine_when_in_jail_can_afford(self) -> None:
        """Pay jail fine should be valid when in jail and can afford."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        current = game.current_player
        game.players[current].in_jail = True
        game.state.phase = "jail_decision"
        game.players[current].money = 100  # Can afford $50

        mask = encoder.get_action_mask(game, player_id=current)
        assert mask[OFFSET_PAY_JAIL_FINE]

    def test_mask_pay_jail_fine_invalid_cannot_afford(self) -> None:
        """Pay jail fine should be invalid when cannot afford."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        current = game.current_player
        game.players[current].in_jail = True
        game.state.phase = "jail_decision"
        game.players[current].money = 10  # Cannot afford $50

        mask = encoder.get_action_mask(game, player_id=current)
        assert not mask[OFFSET_PAY_JAIL_FINE]

    def test_mask_mortgage_when_owns_unmortgaged_property(self) -> None:
        """Mortgage should be valid when owning unmortgaged property."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        current = game.current_player
        # Give player ownership of Mediterranean (position 1)
        game.property_manager.properties[1].owner = current

        mask = encoder.get_action_mask(game, player_id=current)

        # Find the mortgage action index for position 1
        prop_idx = _BUYABLE_TO_INDEX[1]
        assert mask[OFFSET_MORTGAGE + prop_idx]

    def test_mask_unmortgage_when_owns_mortgaged_property(self) -> None:
        """Unmortgage should be valid when owning mortgaged property and can afford."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        current = game.current_player
        # Give player ownership and mortgage Mediterranean (position 1)
        game.property_manager.properties[1].owner = current
        game.property_manager.properties[1].mortgaged = True

        mask = encoder.get_action_mask(game, player_id=current)

        # Find the unmortgage action index for position 1
        prop_idx = _BUYABLE_TO_INDEX[1]
        assert mask[OFFSET_UNMORTGAGE + prop_idx]


class TestGetActionName:
    """Tests for human-readable action names."""

    def test_action_name_buy_property(self) -> None:
        """Buy property action name."""
        encoder = ActionEncoder()
        name = encoder.get_action_name(OFFSET_BUY_PROPERTY)
        assert name == "Buy Property"

    def test_action_name_pass_buy(self) -> None:
        """Pass buy action name."""
        encoder = ActionEncoder()
        name = encoder.get_action_name(OFFSET_PASS_BUY)
        assert "Pass" in name or "Auction" in name

    def test_action_name_end_turn(self) -> None:
        """End turn action name."""
        encoder = ActionEncoder()
        name = encoder.get_action_name(OFFSET_END_TURN)
        assert name == "End Turn"

    def test_action_name_use_jail_card(self) -> None:
        """Use jail card action name."""
        encoder = ActionEncoder()
        name = encoder.get_action_name(OFFSET_USE_JAIL_CARD)
        assert "Jail" in name

    def test_action_name_pay_jail_fine(self) -> None:
        """Pay jail fine action name."""
        encoder = ActionEncoder()
        name = encoder.get_action_name(OFFSET_PAY_JAIL_FINE)
        assert "Jail" in name and "$50" in name

    def test_action_name_build_house_includes_property_name(self) -> None:
        """Build house action name should include property name."""
        encoder = ActionEncoder()
        # First developable property is Mediterranean (position 1)
        name = encoder.get_action_name(OFFSET_BUILD_HOUSE)
        assert "Build House" in name
        assert "Mediterranean" in name

    def test_action_name_mortgage_includes_property_name(self) -> None:
        """Mortgage action name should include property name."""
        encoder = ActionEncoder()
        # First buyable property is Mediterranean (position 1)
        name = encoder.get_action_name(OFFSET_MORTGAGE)
        assert "Mortgage" in name
        assert "Mediterranean" in name

    def test_action_name_unknown_action(self) -> None:
        """Unknown action index should return descriptive name."""
        encoder = ActionEncoder()
        name = encoder.get_action_name(999)
        assert "Unknown" in name or "999" in name


class TestActionMaskConsistency:
    """Tests for action mask and execution consistency."""

    def test_masked_buy_property_executes(self) -> None:
        """If buy property is masked as valid, it should execute."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        current = game.current_player
        game.players[current].position = 1  # Mediterranean
        game.state.phase = "purchase_decision"

        mask = encoder.get_action_mask(game, player_id=current)

        if mask[OFFSET_BUY_PROPERTY]:
            action = encoder.decode(OFFSET_BUY_PROPERTY, current, game)
            is_valid, _ = action.validate(game)
            assert is_valid, "Masked action failed validation"

    def test_masked_end_turn_executes(self) -> None:
        """If end turn is masked as valid, it should execute."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        current = game.current_player
        mask = encoder.get_action_mask(game, player_id=current)

        if mask[OFFSET_END_TURN]:
            action = encoder.decode(OFFSET_END_TURN, current, game)
            is_valid, _ = action.validate(game)
            assert is_valid, "Masked EndTurn failed validation"

    def test_masked_mortgage_executes(self) -> None:
        """If mortgage is masked as valid, it should execute."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        current = game.current_player
        # Give player a property
        game.property_manager.properties[1].owner = current

        mask = encoder.get_action_mask(game, player_id=current)
        prop_idx = _BUYABLE_TO_INDEX[1]
        action_idx = OFFSET_MORTGAGE + prop_idx

        if mask[action_idx]:
            action = encoder.decode(action_idx, current, game)
            is_valid, _ = action.validate(game)
            assert is_valid, "Masked MortgageProperty failed validation"


class TestPropertyBasedTests:
    """Property-based tests using hypothesis."""

    @given(action_idx=st.integers(0, 148))
    @settings(max_examples=158)
    def test_all_valid_indices_decode_without_error(self, action_idx: int) -> None:
        """All valid action indices should decode without raising."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        # Should not raise
        action = encoder.decode(action_idx, player_id=0, game=game)
        assert action is not None

    @given(action_idx=st.integers(-1000, -1) | st.integers(158, 1000))
    def test_invalid_indices_raise(self, action_idx: int) -> None:
        """Invalid action indices should raise ValueError."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        with pytest.raises(ValueError):
            encoder.decode(action_idx, player_id=0, game=game)

    @given(player_id=st.integers(0, 7))
    @settings(max_examples=8)
    def test_mask_always_has_valid_shape(self, player_id: int) -> None:
        """Action mask should always have correct shape."""
        encoder = ActionEncoder()
        num_players = max(player_id + 1, 2)
        game = MonopolyGame(num_players=num_players, seed=42)

        mask = encoder.get_action_mask(game, player_id=player_id)
        assert mask.shape == (158,)
        assert mask.dtype == np.bool_

    @given(
        prop_idx=st.integers(0, 21),  # 22 developable positions
        action_type=st.sampled_from(["build_house", "build_hotel", "sell_house", "sell_hotel"]),
    )
    @settings(max_examples=50)
    def test_building_actions_roundtrip(self, prop_idx: int, action_type: str) -> None:
        """Building actions should roundtrip correctly."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        prop_pos = DEVELOPABLE_POSITIONS[prop_idx]

        original: BuildHouse | BuildHotel | SellHouse | SellHotel
        if action_type == "build_house":
            original = BuildHouse(player_id=0, property_id=prop_pos)
        elif action_type == "build_hotel":
            original = BuildHotel(player_id=0, property_id=prop_pos)
        elif action_type == "sell_house":
            original = SellHouse(player_id=0, property_id=prop_pos)
        else:
            original = SellHotel(player_id=0, property_id=prop_pos)

        encoded = encoder.encode(original)
        decoded = encoder.decode(encoded, player_id=0, game=game)

        assert type(decoded) is type(original)
        assert hasattr(decoded, "property_id") and decoded.property_id == original.property_id

    @given(prop_idx=st.integers(0, 27))  # 28 buyable positions
    @settings(max_examples=28)
    def test_mortgage_actions_roundtrip(self, prop_idx: int) -> None:
        """Mortgage/unmortgage actions should roundtrip correctly."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        prop_pos = BUYABLE_POSITIONS[prop_idx]

        # Test mortgage
        original_mortgage = MortgageProperty(player_id=0, property_id=prop_pos)
        encoded = encoder.encode(original_mortgage)
        decoded = encoder.decode(encoded, player_id=0, game=game)
        assert isinstance(decoded, MortgageProperty)
        assert decoded.property_id == prop_pos

        # Test unmortgage
        original_unmortgage = UnmortgageProperty(player_id=0, property_id=prop_pos)
        encoded = encoder.encode(original_unmortgage)
        decoded = encoder.decode(encoded, player_id=0, game=game)
        assert isinstance(decoded, UnmortgageProperty)
        assert decoded.property_id == prop_pos


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_mask_with_no_properties_owned(self) -> None:
        """Mask should work correctly when player owns no properties."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        current = game.current_player
        mask = encoder.get_action_mask(game, player_id=current)

        # No mortgage/unmortgage actions should be valid
        for idx in range(OFFSET_MORTGAGE, OFFSET_END_TURN):
            assert not mask[idx], f"Action {idx} should be invalid with no properties"

    def test_mask_with_all_properties_mortgaged(self) -> None:
        """Mask should handle all properties being mortgaged."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=2, seed=42)

        current = game.current_player
        # Give player all properties and mortgage them
        for pos in BUYABLE_POSITIONS:
            game.property_manager.properties[pos].owner = current
            game.property_manager.properties[pos].mortgaged = True

        mask = encoder.get_action_mask(game, player_id=current)

        # No build actions should be valid on mortgaged properties
        for idx in range(OFFSET_BUILD_HOUSE, OFFSET_MORTGAGE):
            assert not mask[idx], f"Build action {idx} should be invalid on mortgaged"

    def test_encode_preserves_player_id(self) -> None:
        """Encoding should work regardless of player_id in action."""
        encoder = ActionEncoder()

        # Encoding doesn't depend on player_id
        action1 = EndTurn(player_id=0)
        action2 = EndTurn(player_id=5)

        assert encoder.encode(action1) == encoder.encode(action2)

    def test_decode_uses_provided_player_id(self) -> None:
        """Decoding should use the provided player_id."""
        encoder = ActionEncoder()
        game = MonopolyGame(num_players=4, seed=42)

        action0 = encoder.decode(OFFSET_END_TURN, player_id=0, game=game)
        action3 = encoder.decode(OFFSET_END_TURN, player_id=3, game=game)

        assert action0.player_id == 0
        assert action3.player_id == 3


class TestFreshActionMasks:
    @pytest.mark.parametrize("cash", [0, 59, 60, 1500])
    def test_purchase_mask_changes_without_invalidation(self, cash):
        encoder = ActionEncoder()
        game = MonopolyGame(2, seed=42)
        game.players[0].position = 1
        game.state.phase = "purchase_decision"
        previous = encoder.get_action_mask(game, 0)
        game.players[0].money = cash
        current = encoder.get_action_mask(game, 0)
        assert bool(current[OFFSET_BUY_PROPERTY]) == (cash >= 60)
        assert previous[OFFSET_BUY_PROPERTY]
        assert current[OFFSET_PASS_BUY]
        assert not current[OFFSET_END_TURN]
        assert not encoder.get_action_mask(game, 1).any()
        np.testing.assert_array_equal(current, ActionEncoder().get_action_mask(game, 0))

    def test_mask_arrays_do_not_alias(self):
        encoder = ActionEncoder()
        game = MonopolyGame(2, seed=42)
        first = encoder.get_action_mask(game, 0)
        expected = first.copy()
        second = encoder.get_action_mask(game, 0)
        assert not np.shares_memory(first, second)
        game.players[0].in_jail = True
        game.state.phase = "jail_decision"
        third = encoder.get_action_mask(game, 0)
        assert third[OFFSET_PAY_JAIL_FINE]
        np.testing.assert_array_equal(first, expected)
