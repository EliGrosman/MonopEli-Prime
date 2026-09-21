"""Authoritative trade decisions for the opt-in foundation-trade-v1 ruleset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .types import PROPERTY_GROUPS, TradeOfferData

if TYPE_CHECKING:
    from .game import MonopolyGame


def _terms(action: Any) -> TradeOfferData:
    return TradeOfferData(
        from_player=action.player_id,
        to_player=action.to_player,
        give_properties=list(action.give_properties),
        give_money=action.give_money,
        want_properties=list(action.want_properties),
        want_money=action.want_money,
    )


def _tradeable_property(game: MonopolyGame, position: int, owner: int) -> tuple[bool, str]:
    prop = game.property_manager.get(position)
    if prop is None or prop.owner != owner:
        return False, f"Player {owner} does not own property {position}"
    if prop.color is not None:
        group = PROPERTY_GROUPS.get(prop.color, ())
        if any(game.property_manager.properties[p].houses for p in group):
            return False, "Developed color groups cannot be traded"
    return True, ""


def validate_terms(game: MonopolyGame, offer: TradeOfferData) -> tuple[bool, str]:
    proposer, recipient = offer["from_player"], offer["to_player"]
    if not 0 <= proposer < len(game.players):
        return False, "Invalid player ID"
    if proposer == recipient:
        return False, "Cannot trade with yourself"
    if not 0 <= recipient < len(game.players):
        return False, "Invalid player ID"
    if game.players[proposer].bankrupt or game.players[recipient].bankrupt:
        return False, "Bankrupt players cannot trade"

    give = offer["give_properties"]
    want = offer["want_properties"]
    if not isinstance(give, list) or not isinstance(want, list):
        return False, "Property bundles must be lists"
    if len(give) > 2 or len(want) > 2:
        return False, "A trade may include at most two properties per side"
    if not give and not want:
        return False, "At least one property must change hands"
    combined = give + want
    if any(not isinstance(pos, int) for pos in combined) or len(set(combined)) != len(combined):
        return False, "Trade properties must be distinct integer positions"

    give_money, want_money = offer["give_money"], offer["want_money"]
    if type(give_money) is not int or type(want_money) is not int:
        return False, "Trade cash must be an integer"
    if give_money < 0 or want_money < 0 or (give_money and want_money):
        return False, "Cash must be nonnegative and flow in only one direction"
    if give_money > game.players[proposer].money:
        return False, "Proposer has insufficient cash"
    if want_money > game.players[recipient].money:
        return False, "Recipient has insufficient cash"

    for position in give:
        valid, reason = _tradeable_property(game, position, proposer)
        if not valid:
            return valid, reason
    for position in want:
        valid, reason = _tradeable_property(game, position, recipient)
        if not valid:
            return valid, reason
    return True, ""


def validate_proposal(game: MonopolyGame, action: Any) -> tuple[bool, str]:
    if game.rules_id != "foundation-trade-v1":
        return False, "Trading is disabled in foundation-v1"
    if game.state.pending_trades:
        return False, "Another trade response is pending"
    if action.to_player in game.state.trade_targets_this_turn:
        return False, "This opponent already received an offer this turn"
    if len(game.state.trade_targets_this_turn) >= 2:
        return False, "Trade proposal budget exhausted for this turn"
    return validate_terms(game, _terms(action))


def propose(game: MonopolyGame, action: Any) -> int:
    offer = _terms(action)
    trade_id = game.state.next_trade_id
    game.state.next_trade_id += 1
    offer["trade_id"] = trade_id
    offer["created_revision"] = game.state.revision + 1
    game.state.pending_trades[trade_id] = offer
    game.state.trade_targets_this_turn.append(action.to_player)
    game.state.trade_resume_phase = game.state.phase
    game.state.phase = "trade_response"
    game.state.decision_player = action.to_player
    event = {"type": "trade_proposed", **offer}
    game.log_structured_event(event)
    game.log_event(f"trade proposed #{trade_id}: {action.player_id} -> {action.to_player}")
    return trade_id


def pending_for(game: MonopolyGame, player_id: int, trade_id: int) -> TradeOfferData | None:
    offer = game.state.pending_trades.get(trade_id)
    return offer if offer is not None and offer["to_player"] == player_id else None


def validate_response(
    game: MonopolyGame, player_id: int, trade_id: int, *, accepting: bool
) -> tuple[bool, str]:
    if game.rules_id != "foundation-trade-v1" or game.state.phase != "trade_response":
        return False, "No trade response is pending"
    offer = pending_for(game, player_id, trade_id)
    if offer is None:
        return False, "Trade does not exist or belongs to another recipient"
    return validate_terms(game, offer) if accepting else (True, "")


def _resume(game: MonopolyGame, trade_id: int) -> None:
    offer = game.state.pending_trades.pop(trade_id)
    game.state.decision_player = offer["from_player"]
    game.state.phase = game.state.trade_resume_phase or "asset_management"
    game.state.trade_resume_phase = None


def accept(game: MonopolyGame, player_id: int, trade_id: int) -> None:
    offer = pending_for(game, player_id, trade_id)
    if offer is None:
        raise ValueError("Trade does not exist")
    proposer, recipient = offer["from_player"], offer["to_player"]
    for position in offer["give_properties"]:
        game.property_manager.properties[position].owner = recipient
    for position in offer["want_properties"]:
        game.property_manager.properties[position].owner = proposer
    if offer["give_money"]:
        game.players[proposer].remove_money(offer["give_money"])
        game.players[recipient].add_money(offer["give_money"])
    if offer["want_money"]:
        game.players[recipient].remove_money(offer["want_money"])
        game.players[proposer].add_money(offer["want_money"])
    game.log_structured_event({"type": "trade_accepted", "trade_id": trade_id, **offer})
    game.log_event(f"trade accepted #{trade_id}: {proposer} <-> {recipient}")
    _resume(game, trade_id)


def reject(game: MonopolyGame, player_id: int, trade_id: int) -> None:
    offer = pending_for(game, player_id, trade_id)
    if offer is None:
        raise ValueError("Trade does not exist")
    game.log_structured_event({"type": "trade_rejected", "trade_id": trade_id, **offer})
    game.log_event(f"trade rejected #{trade_id}: {player_id}")
    _resume(game, trade_id)
