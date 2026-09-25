"""Readable public activity derived from authoritative, committed transitions."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from monopoly_engine.actions import Action
from monopoly_engine.board import Board
from monopoly_engine.cards import CHANCE_CARDS, COMMUNITY_CHEST_CARDS
from monopoly_engine.decision import DecisionView
from monopoly_engine.foundation import TransitionResult
from monopoly_engine.rules import (
    get_building_cost,
    get_house_sale_value,
    get_mortgage_value,
    get_property_cost,
    get_unmortgage_cost,
)

from ..models.game import GameActivity


def describe_activity(
    game_id: str,
    action: Action,
    before: DecisionView,
    after: DecisionView,
    result: TransitionResult,
) -> GameActivity:
    """Format public outcomes; never project raw agent/request metadata."""
    names = {player.id: player.name for player in before.players}
    actor = action.player_id
    name = names[actor]
    kind = type(action).__name__
    position = getattr(action, "property_id", None)
    property_name = Board.get_space(position).name if position is not None else "property"
    summary = f"{name} completed {re.sub(r'(?<!^)(?=[A-Z])', ' ', kind).lower()}"
    if kind == "RollDice" and after.last_roll:
        first, second = after.last_roll
        summary = f"{name} rolled {first} + {second} = {first + second}"
    elif kind == "BuyProperty":
        summary = f"{name} bought {property_name} for ${get_property_cost(position):,}"
    elif kind == "PassBuy":
        target = before.purchase_property
        label = Board.get_space(target).name if target is not None else "the property"
        summary = f"{name} passed on {label}"
    elif kind in {"BuildHouse", "BuildHotel"}:
        building = "a hotel" if kind == "BuildHotel" else "a house"
        summary = f"{name} built {building} on {property_name} for ${get_building_cost(position):,}"
    elif kind in {"SellHouse", "SellHotel"}:
        building = "a hotel" if kind == "SellHotel" else "a house"
        summary = (
            f"{name} sold {building} on {property_name} for ${get_house_sale_value(position):,}"
        )
    elif kind == "SellBuildingGroup":
        summary = f"{name} sold buildings in the {property_name} group"
    elif kind == "MortgageProperty":
        summary = f"{name} mortgaged {property_name} for ${get_mortgage_value(position):,}"
    elif kind == "UnmortgageProperty":
        summary = f"{name} redeemed {property_name} for ${get_unmortgage_cost(position):,}"
    elif kind == "PayJailFine":
        summary = f"{name} paid $50 to leave jail"
    elif kind == "UseJailCard":
        summary = f"{name} used a Get Out of Jail Free card"
    elif kind == "DeclareBankruptcy":
        summary = f"{name} declared bankruptcy"
    elif kind == "EndTurn":
        summary = f"{name} ended their turn"

    details: list[str] = []
    for event in result.structured_events:
        event_type = event.get("type")
        if event_type not in {"trade_proposed", "trade_accepted", "trade_rejected"}:
            continue
        proposer, recipient = names[event["from_player"]], names[event["to_player"]]
        if event_type == "trade_proposed":
            summary = f"{proposer} offered a trade to {recipient}"
        else:
            verb = "accepted" if event_type == "trade_accepted" else "rejected"
            summary = f"{name} {verb} {proposer}’s trade"
        for person, prop_key, cash_key in (
            (proposer, "give_properties", "give_money"),
            (recipient, "want_properties", "want_money"),
        ):
            terms = [Board.get_space(p).name for p in event.get(prop_key, [])]
            if event.get(cash_key):
                terms.append(f"${event[cash_key]:,}")
            details.append(f"{person} gives {', '.join(terms) or 'nothing'}")

    event_position = next(p.position for p in before.players if p.id == actor)
    for event in result.events:
        movement = re.search(r" moved from \d+ to (\d+) ", event)
        if movement:
            event_position = int(movement.group(1))
        payment = re.fullmatch(r"payment (\d+) -> (None|\d+): (\d+)", event)
        card = re.fullmatch(r"card (\d+): (\d+)", event)
        if payment:
            payer, creditor, amount = payment.groups()
            payee = "the bank" if creditor == "None" else names[int(creditor)]
            details.append(f"{names[int(payer)]} paid ${int(amount):,} to {payee}")
        elif card:
            player_id, card_id = map(int, card.groups())
            deck = COMMUNITY_CHEST_CARDS if event_position in (2, 17, 33) else CHANCE_CARDS
            drawn = next((c for c in deck if c.id == card_id), None)
            if drawn:
                details.append(f"{names[player_id]} drew a card: {drawn.text}")
        elif "passed GO and collected" in event:
            details.append(event)

    for old, new in zip(before.players, after.players, strict=True):
        if old.position != new.position:
            details.append(f"{new.name} moved to {Board.get_space(new.position).name}")
        if not old.in_jail and new.in_jail:
            details.append(f"{new.name} went to jail")
        elif old.in_jail and not new.in_jail and not new.bankrupt:
            details.append(f"{new.name} left jail")
        if kind == "RollDice" and old.in_jail and new.in_jail and old.id == actor:
            details.append(f"{new.name} remains in jail")
        if old.money != new.money:
            change = new.money - old.money
            details.append(f"{new.name} cash: {change:+,} → ${new.money:,}")
        if not old.bankrupt and new.bankrupt:
            details.append(f"{new.name} was eliminated")
    if after.debt:
        details.append(f"{names[after.debt.debtor]} must raise ${after.debt.amount:,}")
    if result.winner is not None:
        details.append(f"{names[result.winner]} won the game")

    return GameActivity(
        id=f"{game_id}:{result.revision}",
        actor=actor,
        revision=result.revision,
        turn_number=before.turn_number,
        action_type=kind,
        summary=summary,
        details=list(dict.fromkeys(details)),
        occurred_at=datetime.now(UTC),
    )
