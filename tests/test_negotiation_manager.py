"""Tests for the NegotiationManager multi-round trade orchestration."""

from __future__ import annotations

import pytest

from mcts.negotiation import NegotiationManager, NegotiationStatus
from monopoly_engine.game import MonopolyGame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_game_with_properties() -> MonopolyGame:
    """Create a 4-player game and assign properties for trading.

    Ownership:
        Player 0: Mediterranean Ave (1), Baltic Ave (3)   [Brown monopoly]
        Player 1: Oriental Ave (6), Vermont Ave (8)       [Light Blue partial]
        Player 2: St. Charles Place (11)                   [Magenta partial]
        Player 3: (none)
    """
    game = MonopolyGame(num_players=4, seed=42)
    # Assign properties directly via property_manager
    game.property_manager.properties[1].owner = 0
    game.property_manager.properties[3].owner = 0
    game.property_manager.properties[6].owner = 1
    game.property_manager.properties[8].owner = 1
    game.property_manager.properties[11].owner = 2
    return game


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestNegotiationManagerInit:
    def test_default_max_rounds(self) -> None:
        manager = NegotiationManager()
        assert manager._max_rounds == 3

    def test_custom_max_rounds(self) -> None:
        manager = NegotiationManager(max_rounds=5)
        assert manager._max_rounds == 5

    def test_starts_empty(self) -> None:
        manager = NegotiationManager()
        assert manager.get_active_negotiations() == []


# ---------------------------------------------------------------------------
# Starting negotiations
# ---------------------------------------------------------------------------

class TestStartNegotiation:
    def test_basic_start(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )

        assert neg_id == 0
        record = manager.get_negotiation(neg_id)
        assert record is not None
        assert record.status == NegotiationStatus.PENDING
        assert record.original_proposer == 0
        assert record.current_proposer == 0
        assert record.current_responder == 1
        assert record.round_number == 1
        assert len(record.history) == 1
        assert record.current_trade_id is not None

    def test_creates_engine_trade(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )

        record = manager.get_negotiation(neg_id)
        assert record is not None
        trade = game.get_pending_trade(record.current_trade_id)  # type: ignore[arg-type]
        assert trade is not None
        assert trade["from_player"] == 0
        assert trade["to_player"] == 1
        assert trade["give_properties"] == [1]
        assert trade["want_properties"] == [6]

    def test_with_cash(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], give_money=100,
            want_properties=[6],
        )

        record = manager.get_negotiation(neg_id)
        assert record is not None
        assert record.history[0]["give_money"] == 100

    def test_cash_only_trade(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            want_properties=[6], give_money=200,
        )

        record = manager.get_negotiation(neg_id)
        assert record is not None
        assert record.history[0]["give_properties"] == []
        assert record.history[0]["give_money"] == 200

    def test_invalid_trade_raises(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        # Player 0 doesn't own property 6
        with pytest.raises(ValueError, match="Invalid trade"):
            manager.start_negotiation(
                game, from_player=0, to_player=1,
                give_properties=[6], want_properties=[1],
            )

    def test_sequential_ids(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        id1 = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )
        id2 = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[3], want_properties=[8],
        )

        assert id1 == 0
        assert id2 == 1


# ---------------------------------------------------------------------------
# Counter-proposals
# ---------------------------------------------------------------------------

class TestCounterPropose:
    def test_basic_counter(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )

        result = manager.counter_propose(
            game, neg_id, player_id=1,
            give_properties=[6], give_money=50,
            want_properties=[1, 3],
        )

        assert result is True
        record = manager.get_negotiation(neg_id)
        assert record is not None
        assert record.round_number == 2
        assert record.current_proposer == 1
        assert record.current_responder == 0
        assert record.status == NegotiationStatus.PENDING
        assert len(record.history) == 2

    def test_counter_rejects_old_engine_trade(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )

        record = manager.get_negotiation(neg_id)
        assert record is not None
        old_trade_id = record.current_trade_id
        assert old_trade_id is not None

        # Verify old trade exists before counter
        assert game.get_pending_trade(old_trade_id) is not None

        manager.counter_propose(
            game, neg_id, player_id=1,
            give_properties=[6], want_properties=[1],
        )

        # After counter, only 1 pending trade should exist (the new one).
        # The engine may reuse trade IDs, so we check that the remaining
        # trade is the counter-proposal, not the original.
        assert len(game.state.pending_trades) == 1
        new_trade_id = record.current_trade_id
        assert new_trade_id is not None
        new_trade = game.get_pending_trade(new_trade_id)
        assert new_trade is not None
        assert new_trade["from_player"] == 1  # Counter-proposer

    def test_counter_creates_new_engine_trade(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )

        manager.counter_propose(
            game, neg_id, player_id=1,
            give_properties=[6], want_properties=[1],
        )

        record = manager.get_negotiation(neg_id)
        assert record is not None
        new_trade = game.get_pending_trade(record.current_trade_id)  # type: ignore[arg-type]
        assert new_trade is not None
        assert new_trade["from_player"] == 1
        assert new_trade["to_player"] == 0

    def test_wrong_player_cannot_counter(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )

        # Player 0 is proposer, not responder
        result = manager.counter_propose(
            game, neg_id, player_id=0,
            give_properties=[1], want_properties=[6],
        )
        assert result is False

    def test_max_rounds_enforced(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager(max_rounds=2)

        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )

        # Round 1 -> 2 (OK, max_rounds=2)
        result = manager.counter_propose(
            game, neg_id, player_id=1,
            give_properties=[6], want_properties=[1],
        )
        assert result is True

        # Round 2 -> 3 (exceeds max_rounds=2, should fail)
        result = manager.counter_propose(
            game, neg_id, player_id=0,
            give_properties=[1], want_properties=[6],
        )
        assert result is False

        record = manager.get_negotiation(neg_id)
        assert record is not None
        assert record.status == NegotiationStatus.EXPIRED

    def test_counter_on_nonexistent_negotiation(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        result = manager.counter_propose(
            game, 999, player_id=1,
            give_properties=[6], want_properties=[1],
        )
        assert result is False

    def test_counter_on_closed_negotiation(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )
        manager.accept(game, neg_id, player_id=1)

        result = manager.counter_propose(
            game, neg_id, player_id=0,
            give_properties=[1], want_properties=[6],
        )
        assert result is False

    def test_invalid_counter_proposal(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )

        # Player 1 tries to give property 11 which they don't own
        result = manager.counter_propose(
            game, neg_id, player_id=1,
            give_properties=[11], want_properties=[1],
        )
        assert result is False


# ---------------------------------------------------------------------------
# Accept
# ---------------------------------------------------------------------------

class TestAccept:
    def test_accept_executes_trade(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )

        result = manager.accept(game, neg_id, player_id=1)
        assert result is True

        # Verify ownership transferred
        assert game.property_manager.properties[1].owner == 1
        assert game.property_manager.properties[6].owner == 0

        record = manager.get_negotiation(neg_id)
        assert record is not None
        assert record.status == NegotiationStatus.ACCEPTED

    def test_accept_counter_proposal(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )
        manager.counter_propose(
            game, neg_id, player_id=1,
            give_properties=[6], give_money=50,
            want_properties=[1, 3],
        )

        result = manager.accept(game, neg_id, player_id=0)
        assert result is True

        # Player 1 gave prop 6 + $50, got props 1 and 3
        assert game.property_manager.properties[6].owner == 0
        assert game.property_manager.properties[1].owner == 1
        assert game.property_manager.properties[3].owner == 1

        record = manager.get_negotiation(neg_id)
        assert record is not None
        assert record.status == NegotiationStatus.ACCEPTED

    def test_wrong_player_cannot_accept(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )

        # Player 2 is not involved
        result = manager.accept(game, neg_id, player_id=2)
        assert result is False

    def test_accept_nonexistent_negotiation(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        result = manager.accept(game, 999, player_id=0)
        assert result is False

    def test_accept_already_closed(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )
        manager.reject(game, neg_id, player_id=1)

        result = manager.accept(game, neg_id, player_id=1)
        assert result is False


# ---------------------------------------------------------------------------
# Reject
# ---------------------------------------------------------------------------

class TestReject:
    def test_basic_reject(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )

        result = manager.reject(game, neg_id, player_id=1)
        assert result is True

        record = manager.get_negotiation(neg_id)
        assert record is not None
        assert record.status == NegotiationStatus.REJECTED

        # Engine trade should be cleaned up
        assert record.current_trade_id is None

    def test_reject_removes_engine_trade(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )

        record = manager.get_negotiation(neg_id)
        assert record is not None
        trade_id = record.current_trade_id

        manager.reject(game, neg_id, player_id=1)

        assert game.get_pending_trade(trade_id) is None  # type: ignore[arg-type]

    def test_wrong_player_cannot_reject(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )

        result = manager.reject(game, neg_id, player_id=0)
        assert result is False

    def test_no_ownership_change_after_reject(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )
        manager.reject(game, neg_id, player_id=1)

        # Ownership unchanged
        assert game.property_manager.properties[1].owner == 0
        assert game.property_manager.properties[6].owner == 1


# ---------------------------------------------------------------------------
# Query methods
# ---------------------------------------------------------------------------

class TestQueryMethods:
    def test_get_pending_for_player(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )
        manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[3], want_properties=[8],
        )

        pending = manager.get_pending_for_player(1)
        assert len(pending) == 2

        pending_0 = manager.get_pending_for_player(0)
        assert len(pending_0) == 0

    def test_get_pending_excludes_closed(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )
        manager.reject(game, neg_id, player_id=1)

        pending = manager.get_pending_for_player(1)
        assert len(pending) == 0

    def test_get_negotiation_returns_none_for_missing(self) -> None:
        manager = NegotiationManager()
        assert manager.get_negotiation(999) is None

    def test_get_active_negotiations(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        neg1 = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )
        manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[3], want_properties=[8],
        )

        assert len(manager.get_active_negotiations()) == 2

        manager.reject(game, neg1, player_id=1)
        assert len(manager.get_active_negotiations()) == 1


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_all(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )

        manager.reset()

        assert manager.get_active_negotiations() == []
        assert manager.get_negotiation(0) is None

    def test_reset_resets_id_counter(self) -> None:
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )
        manager.reset()

        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )
        assert neg_id == 0


# ---------------------------------------------------------------------------
# Full negotiation flows
# ---------------------------------------------------------------------------

class TestFullNegotiationFlow:
    def test_propose_counter_accept(self) -> None:
        """Full 3-step negotiation: propose -> counter -> accept."""
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        # Player 0 proposes: give Baltic(3) for Oriental(6)
        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[3], want_properties=[6],
        )

        # Player 1 counters: wants both brown props for Oriental
        manager.counter_propose(
            game, neg_id, player_id=1,
            give_properties=[6],
            want_properties=[1, 3],
        )

        # Player 0 accepts the counter
        result = manager.accept(game, neg_id, player_id=0)
        assert result is True

        record = manager.get_negotiation(neg_id)
        assert record is not None
        assert record.status == NegotiationStatus.ACCEPTED
        assert record.round_number == 2
        assert len(record.history) == 2

        # Verify final ownership
        assert game.property_manager.properties[1].owner == 1
        assert game.property_manager.properties[3].owner == 1
        assert game.property_manager.properties[6].owner == 0

    def test_propose_counter_counter_accept(self) -> None:
        """4-step: propose -> counter -> counter -> accept."""
        game = _setup_game_with_properties()
        manager = NegotiationManager(max_rounds=4)

        # Round 1: P0 proposes Baltic(3) for Oriental(6)
        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[3], want_properties=[6],
        )

        # Round 2: P1 counters: wants both browns + $100
        manager.counter_propose(
            game, neg_id, player_id=1,
            give_properties=[6],
            want_properties=[1, 3], want_money=100,
        )

        # Round 3: P0 re-counters: both browns but only $50
        manager.counter_propose(
            game, neg_id, player_id=0,
            give_properties=[1, 3], give_money=50,
            want_properties=[6],
        )

        # P1 accepts
        result = manager.accept(game, neg_id, player_id=1)
        assert result is True

        record = manager.get_negotiation(neg_id)
        assert record is not None
        assert record.round_number == 3
        assert len(record.history) == 3

    def test_propose_reject(self) -> None:
        """Simple 2-step: propose -> reject."""
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )
        result = manager.reject(game, neg_id, player_id=1)
        assert result is True

        record = manager.get_negotiation(neg_id)
        assert record is not None
        assert record.status == NegotiationStatus.REJECTED

        # No ownership changes
        assert game.property_manager.properties[1].owner == 0
        assert game.property_manager.properties[6].owner == 1

    def test_multiple_concurrent_negotiations(self) -> None:
        """Two negotiations happening at the same time."""
        game = _setup_game_with_properties()
        manager = NegotiationManager()

        neg1 = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[6],
        )
        neg2 = manager.start_negotiation(
            game, from_player=0, to_player=2,
            give_properties=[3], want_properties=[11],
        )

        # Reject first, accept second
        manager.reject(game, neg1, player_id=1)
        manager.accept(game, neg2, player_id=2)

        r1 = manager.get_negotiation(neg1)
        r2 = manager.get_negotiation(neg2)
        assert r1 is not None and r1.status == NegotiationStatus.REJECTED
        assert r2 is not None and r2.status == NegotiationStatus.ACCEPTED

        # Only second trade executed
        assert game.property_manager.properties[1].owner == 0  # Unchanged
        assert game.property_manager.properties[3].owner == 2  # Traded
        assert game.property_manager.properties[11].owner == 0  # Traded
