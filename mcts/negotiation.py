"""Multi-round trade negotiation manager using existing engine primitives.

This module orchestrates trade negotiations WITHOUT modifying the game engine.
Counter-proposals are modeled as: reject current trade -> propose new trade.
The engine sees standard ProposeTrade/AcceptTrade/RejectTrade actions;
NegotiationManager tracks that they belong to the same negotiation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from monopoly_engine.game import MonopolyGame
from monopoly_engine.rules import validate_trade
from monopoly_engine.types import TradeOfferData


class NegotiationStatus(Enum):
    """Status of a trade negotiation."""

    PENDING = auto()  # Awaiting response
    ACCEPTED = auto()  # Trade accepted and executed
    REJECTED = auto()  # Trade rejected, negotiation over
    COUNTERED = auto()  # Counter-proposal made (new proposal created)
    EXPIRED = auto()  # Max rounds reached without resolution


@dataclass
class NegotiationRecord:
    """Tracks a multi-round trade negotiation (external to engine).

    Each record corresponds to one logical negotiation between two players,
    which may span multiple engine-level ProposeTrade/RejectTrade cycles.
    """

    negotiation_id: int
    original_proposer: int  # Player who initiated the negotiation
    current_proposer: int  # Who made the latest proposal
    current_responder: int  # Who needs to respond
    round_number: int = 0
    max_rounds: int = 3
    status: NegotiationStatus = NegotiationStatus.PENDING
    history: list[TradeOfferData] = field(default_factory=list)
    current_trade_id: int | None = None  # Engine's trade ID for current proposal


class NegotiationManager:
    """Orchestrates multi-round trade negotiations using existing engine actions.

    This class does NOT modify the game engine. It uses the existing
    game.propose_trade(), game.accept_trade(), and game.reject_trade()
    methods to implement negotiation. Counter-proposals are modeled as:
    reject current trade -> propose new trade.

    Usage::

        manager = NegotiationManager(max_rounds=3)

        # Player 0 proposes to Player 1
        neg_id = manager.start_negotiation(
            game, from_player=0, to_player=1,
            give_properties=[1], want_properties=[3],
        )

        # Player 1 counter-proposes (internally: reject + new propose)
        manager.counter_propose(
            game, neg_id, player_id=1,
            give_properties=[3], give_money=50,
            want_properties=[1],
        )

        # Player 0 accepts the counter
        manager.accept(game, neg_id, player_id=0)

    Important: One NegotiationManager instance must be shared across all
    agents in the same game.
    """

    def __init__(self, max_rounds: int = 3) -> None:
        self._negotiations: dict[int, NegotiationRecord] = {}
        self._next_id: int = 0
        self._max_rounds = max_rounds

    def start_negotiation(
        self,
        game: MonopolyGame,
        from_player: int,
        to_player: int,
        give_properties: list[int] | None = None,
        give_money: int = 0,
        want_properties: list[int] | None = None,
        want_money: int = 0,
    ) -> int:
        """Start a new negotiation. Executes propose_trade on the engine.

        Args:
            game: The game instance.
            from_player: Player initiating the trade.
            to_player: Player receiving the trade offer.
            give_properties: Positions the initiator offers.
            give_money: Cash the initiator offers.
            want_properties: Positions the initiator wants.
            want_money: Cash the initiator wants.

        Returns:
            Negotiation ID (for tracking, distinct from engine's trade_id).

        Raises:
            ValueError: If the trade fails engine-level validation.
        """
        give_props = give_properties or []
        want_props = want_properties or []

        # Validate via engine rules before proposing
        from_p = game.players[from_player]
        to_p = game.players[to_player]
        valid, reason = validate_trade(
            from_p,
            to_p,
            game.property_manager,
            give_props,
            give_money,
            want_props,
            want_money,
        )
        if not valid:
            raise ValueError(f"Invalid trade: {reason}")

        # Execute on engine
        trade_id = game.propose_trade(
            from_player,
            to_player,
            give_props,
            give_money,
            want_props,
            want_money,
        )

        # Build the trade data for history tracking
        trade_data: TradeOfferData = {
            "from_player": from_player,
            "to_player": to_player,
            "give_properties": give_props,
            "give_money": give_money,
            "want_properties": want_props,
            "want_money": want_money,
        }

        # Create negotiation record
        neg_id = self._next_id
        self._next_id += 1

        record = NegotiationRecord(
            negotiation_id=neg_id,
            original_proposer=from_player,
            current_proposer=from_player,
            current_responder=to_player,
            round_number=1,
            max_rounds=self._max_rounds,
            status=NegotiationStatus.PENDING,
            history=[trade_data],
            current_trade_id=trade_id,
        )
        self._negotiations[neg_id] = record
        return neg_id

    def counter_propose(
        self,
        game: MonopolyGame,
        negotiation_id: int,
        player_id: int,
        give_properties: list[int] | None = None,
        give_money: int = 0,
        want_properties: list[int] | None = None,
        want_money: int = 0,
    ) -> bool:
        """Counter-propose: rejects current trade, creates new proposal.

        Internally executes: reject_trade(current) -> propose_trade(new).

        Args:
            game: The game instance.
            negotiation_id: ID of the ongoing negotiation.
            player_id: The player making the counter-proposal (must be
                the current responder).
            give_properties: Positions the counter-proposer offers.
            give_money: Cash the counter-proposer offers.
            want_properties: Positions the counter-proposer wants.
            want_money: Cash the counter-proposer wants.

        Returns:
            True if counter-proposal was successfully made, False if
            max rounds exceeded or negotiation is not in PENDING state.
        """
        record = self._negotiations.get(negotiation_id)
        if record is None:
            return False

        if record.status != NegotiationStatus.PENDING:
            return False

        if record.current_responder != player_id:
            return False

        if record.round_number >= record.max_rounds:
            record.status = NegotiationStatus.EXPIRED
            # Reject the pending engine trade
            if record.current_trade_id is not None:
                trade = game.get_pending_trade(record.current_trade_id)
                if trade is not None:
                    game.reject_trade(record.current_trade_id)
            return False

        give_props = give_properties or []
        want_props = want_properties or []

        # Determine the other player (who will receive the counter)
        other_player = record.current_proposer

        # Validate the counter-proposal
        from_p = game.players[player_id]
        to_p = game.players[other_player]
        valid, reason = validate_trade(
            from_p,
            to_p,
            game.property_manager,
            give_props,
            give_money,
            want_props,
            want_money,
        )
        if not valid:
            return False

        # Step 1: Reject current trade on the engine
        if record.current_trade_id is not None:
            trade = game.get_pending_trade(record.current_trade_id)
            if trade is not None:
                game.reject_trade(record.current_trade_id)

        # Step 2: Propose new trade on the engine
        new_trade_id = game.propose_trade(
            player_id,
            other_player,
            give_props,
            give_money,
            want_props,
            want_money,
        )

        # Build trade data for history
        trade_data: TradeOfferData = {
            "from_player": player_id,
            "to_player": other_player,
            "give_properties": give_props,
            "give_money": give_money,
            "want_properties": want_props,
            "want_money": want_money,
        }

        # Update negotiation record
        record.history.append(trade_data)
        record.round_number += 1
        record.current_proposer = player_id
        record.current_responder = other_player
        record.current_trade_id = new_trade_id
        record.status = NegotiationStatus.PENDING

        return True

    def accept(
        self,
        game: MonopolyGame,
        negotiation_id: int,
        player_id: int,
    ) -> bool:
        """Accept the current proposal. Executes accept_trade on the engine.

        Args:
            game: The game instance.
            negotiation_id: ID of the negotiation.
            player_id: The accepting player (must be current responder).

        Returns:
            True if trade was accepted and executed, False otherwise.
        """
        record = self._negotiations.get(negotiation_id)
        if record is None:
            return False

        if record.status != NegotiationStatus.PENDING:
            return False

        if record.current_responder != player_id:
            return False

        if record.current_trade_id is None:
            return False

        # Verify trade still exists in engine
        trade = game.get_pending_trade(record.current_trade_id)
        if trade is None:
            record.status = NegotiationStatus.REJECTED
            return False

        try:
            game.accept_trade(record.current_trade_id)
        except ValueError:
            # Trade became invalid (e.g., player lost money/property)
            record.status = NegotiationStatus.REJECTED
            return False

        record.status = NegotiationStatus.ACCEPTED
        record.current_trade_id = None
        return True

    def reject(
        self,
        game: MonopolyGame,
        negotiation_id: int,
        player_id: int,
    ) -> bool:
        """Reject the current proposal. Executes reject_trade on the engine.

        Args:
            game: The game instance.
            negotiation_id: ID of the negotiation.
            player_id: The rejecting player (must be current responder).

        Returns:
            True if trade was rejected, False otherwise.
        """
        record = self._negotiations.get(negotiation_id)
        if record is None:
            return False

        if record.status != NegotiationStatus.PENDING:
            return False

        if record.current_responder != player_id:
            return False

        if record.current_trade_id is not None:
            trade = game.get_pending_trade(record.current_trade_id)
            if trade is not None:
                game.reject_trade(record.current_trade_id)

        record.status = NegotiationStatus.REJECTED
        record.current_trade_id = None
        return True

    def get_negotiation(self, negotiation_id: int) -> NegotiationRecord | None:
        """Get the current state of a negotiation."""
        return self._negotiations.get(negotiation_id)

    def get_pending_for_player(self, player_id: int) -> list[NegotiationRecord]:
        """Get all negotiations awaiting this player's response."""
        return [
            record
            for record in self._negotiations.values()
            if record.status == NegotiationStatus.PENDING and record.current_responder == player_id
        ]

    def get_active_negotiations(self) -> list[NegotiationRecord]:
        """Get all negotiations that are still pending."""
        return [
            record
            for record in self._negotiations.values()
            if record.status == NegotiationStatus.PENDING
        ]

    def reset(self) -> None:
        """Clear all negotiations (call at game start)."""
        self._negotiations.clear()
        self._next_id = 0
