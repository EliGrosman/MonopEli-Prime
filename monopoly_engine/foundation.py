"""Authoritative foundation-v1 decision transitions. No I/O or reward policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .board import Board
from .cards import CHANCE_CARDS, COMMUNITY_CHEST_CARDS
from .rules import calculate_rent, get_house_sale_value, get_mortgage_value
from .types import (
    INCOME_TAX_AMOUNT,
    JAIL_FINE,
    LUXURY_TAX_AMOUNT,
    MAX_JAIL_TURNS,
    PROPERTY_GROUPS,
    CardType,
    PropertyColor,
    SpaceType,
)

if TYPE_CHECKING:
    from .actions import Action
    from .cards import Card
    from .game import MonopolyGame
    from .property import Property

RULES_ID = "foundation-v1"
TRADE_RULES_ID = "foundation-trade-v1"
RULES_IDS = (RULES_ID, TRADE_RULES_ID)
PHASES = (
    "pre_roll",
    "jail_decision",
    "purchase_decision",
    "asset_management",
    "debt_resolution",
    "terminal",
)
GROUP_STARTS = (1, 6, 11, 16, 21, 26, 31, 37)


@dataclass(frozen=True)
class TransitionResult:
    actor: int
    events: tuple[str, ...]
    decision_player: int
    phase: str
    revision: int
    eliminations: tuple[int, ...]
    winner: int | None
    structured_events: tuple[dict[str, Any], ...] = ()


def tuples(value: Any) -> Any:
    """Restore JSON-round-tripped Random state without executable serialization."""
    return tuple(tuples(v) for v in value) if isinstance(value, (tuple, list)) else value


def group_properties(game: MonopolyGame, position: int) -> list[Property]:
    prop = game.property_manager.get(position)
    if prop is None or prop.color in (None, PropertyColor.RAILROAD, PropertyColor.UTILITY):
        return []
    assert prop.color is not None
    return [game.property_manager.properties[p] for p in PROPERTY_GROUPS[prop.color]]


def liquidation_value(game: MonopolyGame, pid: int) -> int:
    # Full-group sales make this attainable even when hotel downgrades lack houses.
    value = game.players[pid].money
    for prop in game.property_manager.properties.values():
        if prop.owner == pid:
            value += prop.houses * get_house_sale_value(prop.position) if prop.houses else 0
            if not prop.mortgaged:
                value += get_mortgage_value(prop.position)
    return value


def validate_phase(game: MonopolyGame, action: Action) -> tuple[bool, str]:
    s = game.state
    pid = action.player_id
    if not 0 <= pid < len(game.players) or game.game_over:
        return False, "Invalid player or finished game"
    if pid != s.decision_player or game.players[pid].bankrupt:
        return False, "Not your turn: another player owns this decision"
    name = type(action).__name__
    if name == "ProposeTrade":
        return (
            game.rules_id == TRADE_RULES_ID
            and s.phase == "asset_management"
            and not s.roll_owed
            and pid == game.current_player,
            "Trade proposals require the turn owner's asset-management phase",
        )
    if name in ("AcceptTrade", "RejectTrade"):
        return (s.phase == "trade_response", "No trade response is pending")
    selling = ("SellHouse", "SellHotel", "SellBuildingGroup", "MortgageProperty")
    if name in ("SellHouse", "SellHotel"):
        prop = game.property_manager.get(getattr(action, "property_id", -1))
        if prop and prop.houses == 5 and game.houses_remaining < 4:
            return False, "Not enough houses; sell the whole group"
    if s.phase == "debt_resolution":
        if name == "DeclareBankruptcy":
            return (
                liquidation_value(game, pid) < s.obligations[0]["amount"],
                "Assets can satisfy the obligation",
            )
        return (name in selling, "Resolve the debt before other actions")
    if name == "DeclareBankruptcy":
        return False, "No insolvent obligation"
    if name == "RollDice":
        return (s.roll_owed and s.phase != "purchase_decision", "No roll is due")
    if name in ("BuyProperty", "PassBuy"):
        return (s.phase == "purchase_decision", "No pending purchase")
    if name in ("PayJailFine", "UseJailCard"):
        return (s.phase == "jail_decision", "Jail choice must precede rolling")
    if name == "EndTurn":
        return (
            s.phase == "asset_management" and not s.roll_owed,
            "Resolve mandatory decisions before ending the turn",
        )
    if name in selling + ("BuildHouse", "BuildHotel", "UnmortgageProperty"):
        if name in ("SellHouse", "SellHotel"):
            prop = game.property_manager.get(getattr(action, "property_id", -1))
            if prop and prop.houses == 5 and game.houses_remaining < 4:
                return False, "Not enough houses; sell the whole group"
        return True, ""
    return False, "Unsupported foundation action"


def apply_action(game: MonopolyGame, pid: int, action: Action) -> TransitionResult:
    if pid != action.player_id:
        raise ValueError("Action actor mismatch")
    valid, reason = action.validate(game)
    if not valid:
        raise ValueError(reason)
    before_events = len(game.state.event_log)
    before_structured = len(game.state.structured_event_log)
    before_eliminations = len(game.state.elimination_order)
    action.execute(game)
    name = type(action).__name__
    if name == "BuyProperty":
        game.state.phase = "asset_management"
    elif name in ("PayJailFine", "UseJailCard"):
        game.state.phase = "pre_roll"
    settle(game)
    game.state.revision += 1
    game.log_event(f"decision {pid}: {name}")
    s = game.state
    return TransitionResult(
        pid,
        tuple(s.event_log[before_events:]),
        s.decision_player,
        s.phase,
        s.revision,
        tuple(s.elimination_order[before_eliminations:]),
        game.winner,
        tuple(s.structured_event_log[before_structured:]),
    )


def end_turn(game: MonopolyGame) -> None:
    s = game.state
    if game.game_over:
        return
    s.next_player()
    s.turn_number += 1
    s.last_roll = None
    s.doubles_count = 0
    s.trade_targets_this_turn.clear()
    s.roll_owed = True
    s.decision_player = game.current_player
    s.phase = "jail_decision" if game.players[game.current_player].in_jail else "pre_roll"


def charge(
    game: MonopolyGame,
    pid: int,
    amount: int,
    creditor: int | None,
    continuation: dict[str, Any] | None = None,
) -> None:
    if amount:
        game.state.obligations.append({"debtor": pid, "amount": amount, "creditor": creditor})
    if continuation is not None:
        game.state.continuation.append(continuation)


def settle(game: MonopolyGame) -> None:
    s = game.state
    while not game.game_over:
        if s.obligations:
            debt = s.obligations[0]
            pid, amount, creditor = debt["debtor"], debt["amount"], debt["creditor"]
            player = game.players[pid]
            if player.bankrupt:
                s.obligations.pop(0)
                continue
            if player.money >= amount:
                player.remove_money(amount)
                if creditor is not None and not game.players[creditor].bankrupt:
                    game.players[creditor].add_money(amount)
                game.log_event(f"payment {pid} -> {creditor}: {amount}")
                s.obligations.pop(0)
                continue
            if liquidation_value(game, pid) < amount:
                bankrupt(game, pid, creditor)
                s.obligations.pop(0)
                continue
            s.decision_player = pid
            s.phase = "debt_resolution"
            return
        if s.continuation:
            task = s.continuation.pop(0)
            if task["kind"] == "jail_move" and not game.players[task["player"]].bankrupt:
                game.players[task["player"]].leave_jail()
                game.move_player(task["player"], task["spaces"])
                land(game, task["player"])
            continue
        if game.players[game.current_player].bankrupt:
            end_turn(game)
        elif s.phase != "trade_response":
            s.decision_player = game.current_player
            if s.phase == "debt_resolution":
                s.phase = "asset_management"
        return
    s.phase = "terminal"
    s.obligations.clear()
    s.continuation.clear()


def bankrupt(game: MonopolyGame, pid: int, creditor: int | None = None) -> None:
    p = game.players[pid]
    if p.bankrupt:
        return
    proceeds = p.money
    for prop in game.property_manager.properties.values():
        if prop.owner != pid:
            continue
        if prop.houses:
            proceeds += prop.houses * get_house_sale_value(prop.position)
        if prop.houses == 5:
            game.hotels_remaining += 1
        else:
            game.houses_remaining += prop.houses
        prop.houses = 0
        prop.owner = creditor
        if creditor is None:
            prop.mortgaged = False
    if creditor is not None:
        game.players[creditor].add_money(proceeds)
        sources = game.state.jail_card_sources.pop(pid, [])
        game.state.jail_card_sources.setdefault(creditor, []).extend(sources)
        game.players[creditor].jail_cards += p.jail_cards
    else:
        for _ in range(p.jail_cards):
            return_jail_card(game, pid)
    p.jail_cards = 0
    p.declare_bankrupt()
    game.state.elimination_order.append(pid)
    game.log_event(f"eliminated {pid}, creditor {creditor}")
    active = [p.id for p in game.players if not p.bankrupt]
    if len(active) == 1:
        game.game_over = True
        game.winner = active[0]
        game.state.phase = "terminal"


def return_jail_card(game: MonopolyGame, pid: int) -> None:
    sources = game.state.jail_card_sources.get(pid, [])
    if not sources:
        raise ValueError("Missing jail-card provenance in legacy state")
    source = sources.pop(0)
    deck = game.state.chance_deck if source == "chance" else game.state.chest_deck
    cards = CHANCE_CARDS if source == "chance" else COMMUNITY_CHEST_CARDS
    deck.return_jail_card(next(c for c in cards if c.card_type == CardType.GET_OUT_OF_JAIL))


def roll(game: MonopolyGame, pid: int) -> None:
    s, p = game.state, game.players[pid]
    dice = game.roll_dice()
    s.last_roll = dice
    s.roll_owed = False
    s.phase = "asset_management"
    doubles = dice[0] == dice[1]
    total = sum(dice)
    game.log_event(f"roll {pid}: {dice[0]}, {dice[1]}")
    if p.in_jail:
        if doubles:
            p.leave_jail()
            s.doubles_count = 0
        elif p.increment_jail_turns() >= MAX_JAIL_TURNS:
            charge(
                game, pid, JAIL_FINE, None, {"kind": "jail_move", "player": pid, "spaces": total}
            )
            return
        else:
            return
    else:
        s.doubles_count = s.doubles_count + 1 if doubles else 0
        if s.doubles_count == 3:
            game.send_to_jail(pid)
            return
        s.roll_owed = doubles
    game.move_player(pid, total)
    land(game, pid)


def property_landing(game: MonopolyGame, pid: int, special: bool = False) -> None:
    p = game.players[pid]
    prop = game.property_manager.get(p.position)
    if prop is None:
        return
    if prop.owner is None:
        game.state.phase = "purchase_decision"
    elif prop.owner != pid:
        dice = sum(game.last_roll or (0, 0))
        if special and prop.color == PropertyColor.UTILITY and not prop.mortgaged:
            dice = sum(game.roll_dice())
            game.log_event(f"utility rent dice: {dice}")
        charge(
            game, pid, calculate_rent(game.property_manager, p.position, dice, special), prop.owner
        )


def land(game: MonopolyGame, pid: int) -> None:
    p = game.players[pid]
    kind = Board.get_space(p.position).space_type
    if kind in (SpaceType.PROPERTY, SpaceType.RAILROAD, SpaceType.UTILITY):
        property_landing(game, pid)
    elif kind in (SpaceType.CHANCE, SpaceType.COMMUNITY_CHEST):
        source = "chance" if kind == SpaceType.CHANCE else "chest"
        deck = game.state.chance_deck if source == "chance" else game.state.chest_deck
        card = deck.draw()
        if card.card_type == CardType.GET_OUT_OF_JAIL:
            game.state.jail_card_sources.setdefault(pid, []).append(source)
        execute_card(game, pid, card)
    elif kind == SpaceType.INCOME_TAX:
        charge(game, pid, INCOME_TAX_AMOUNT, None)
    elif kind == SpaceType.LUXURY_TAX:
        charge(game, pid, LUXURY_TAX_AMOUNT, None)
    elif kind == SpaceType.GO_TO_JAIL:
        game.send_to_jail(pid)
        game.state.roll_owed = False


def execute_card(game: MonopolyGame, pid: int, card: Card) -> None:
    p = game.players[pid]
    kind = card.card_type
    game.log_event(f"card {pid}: {card.id}")
    if kind == CardType.MOVE and card.move_to is not None:
        game.move_player_to(pid, card.move_to)
        land(game, pid)
    elif kind == CardType.MOVE_BACK and card.move_spaces is not None:
        game.move_player(pid, card.move_spaces)
        land(game, pid)
    elif kind == CardType.MOVE_NEAREST:
        positions = (5, 15, 25, 35) if card.move_to_nearest == "railroad" else (12, 28)
        target = next((pos for pos in positions if pos > p.position), positions[0])
        game.move_player_to(pid, target)
        property_landing(game, pid, special=True)
    elif kind == CardType.COLLECT:
        p.add_money(card.amount or 0)
    elif kind == CardType.PAY:
        charge(game, pid, card.amount or 0, None)
    elif kind == CardType.PAY_PER_BUILDING:
        cost = sum(
            (card.per_hotel or 0) if prop.houses == 5 else prop.houses * (card.per_house or 0)
            for prop in game.property_manager.properties.values()
            if prop.owner == pid
        )
        charge(game, pid, cost, None)
    elif kind in (CardType.COLLECT_FROM_PLAYERS, CardType.PAY_TO_PLAYERS):
        for other in game.players:
            if other.id != pid and not other.bankrupt:
                debtor, creditor = (
                    (other.id, pid) if kind == CardType.COLLECT_FROM_PLAYERS else (pid, other.id)
                )
                charge(game, debtor, card.amount or 0, creditor)
    elif kind == CardType.GET_OUT_OF_JAIL:
        p.add_jail_card()
    elif kind == CardType.GO_TO_JAIL:
        game.send_to_jail(pid)
        game.state.roll_owed = False


def legal_actions(game: MonopolyGame, pid: int) -> list[Action]:
    """Engine-owned legal decisions, independent of any RL index representation."""
    from .actions import (
        AcceptTrade,
        BuildHotel,
        BuildHouse,
        BuyProperty,
        EndTurn,
        MortgageProperty,
        PassBuy,
        PayJailFine,
        RejectTrade,
        RollDice,
        SellBuildingGroup,
        SellHotel,
        SellHouse,
        UnmortgageProperty,
        UseJailCard,
    )

    if game.game_over or pid != game.decision_player or game.players[pid].bankrupt:
        return []
    if game.state.phase == "trade_response":
        trade_id = next(iter(game.state.pending_trades), -1)
        response_candidates: list[Action] = [AcceptTrade(pid, trade_id), RejectTrade(pid, trade_id)]
        return [action for action in response_candidates if action.validate(game)[0]]
    candidates: list[Action] = [
        RollDice(pid),
        BuyProperty(pid, game.players[pid].position),
        PassBuy(pid),
        EndTurn(pid),
        PayJailFine(pid),
        UseJailCard(pid),
    ]
    for position, prop in game.property_manager.properties.items():
        if prop.owner != pid:
            continue
        candidates.extend([MortgageProperty(pid, position), UnmortgageProperty(pid, position)])
        if prop.color not in (PropertyColor.RAILROAD, PropertyColor.UTILITY):
            candidates.extend(
                [
                    BuildHouse(pid, position),
                    BuildHotel(pid, position),
                    SellHouse(pid, position),
                    SellHotel(pid, position),
                ]
            )
    candidates.extend(SellBuildingGroup(pid, pos) for pos in GROUP_STARTS)
    return [action for action in candidates if action.validate(game)[0]]
