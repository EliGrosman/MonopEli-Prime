"""Stable public decision contract shared by every game consumer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from .board import Board
from .foundation import TRADE_RULES_ID, legal_actions
from .rules import get_property_cost, get_unmortgage_cost
from .trading import tradeable_property

if TYPE_CHECKING:
    from .game import MonopolyGame

DECISION_CONTRACT_VERSION = "decision-contract-v1"


@dataclass(frozen=True)
class PublicPlayer:
    id: int
    name: str
    money: int
    position: int
    in_jail: bool
    jail_turns: int
    jail_cards: int
    bankrupt: bool


@dataclass(frozen=True)
class PublicProperty:
    position: int
    name: str
    owner: int | None
    houses: int
    mortgaged: bool
    price: int
    redemption_cost: int
    group: str | None


@dataclass(frozen=True)
class PublicOffer:
    trade_id: int
    created_revision: int
    from_player: int
    to_player: int
    give_properties: tuple[int, ...]
    give_money: int
    want_properties: tuple[int, ...]
    want_money: int


@dataclass(frozen=True)
class PublicDebt:
    debtor: int
    amount: int
    creditor: int | None


@dataclass(frozen=True)
class PublicAction:
    type: str
    player_id: int
    property_id: int | None = None


@dataclass(frozen=True)
class TradeCapability:
    can_propose: bool
    proposals_remaining: int
    used_recipients: tuple[int, ...]
    eligible_recipients: tuple[int, ...]
    tradeable_properties: tuple[int, ...]


@dataclass(frozen=True)
class DecisionView:
    contract_version: str
    rules_id: str
    revision: int
    viewer_id: int | None
    turn_owner: int
    decision_player: int
    phase: str
    turn_number: int
    roll_owed: bool
    doubles_count: int
    last_roll: tuple[int, int] | None
    houses_remaining: int
    hotels_remaining: int
    debt: PublicDebt | None
    purchase_property: int | None
    players: tuple[PublicPlayer, ...]
    properties: tuple[PublicProperty, ...]
    legal_actions: tuple[PublicAction, ...]
    trade: TradeCapability
    pending_offer: PublicOffer | None

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation."""
        return asdict(self)


def _pending_offer(game: MonopolyGame) -> PublicOffer | None:
    if not game.state.pending_trades:
        return None
    trade_id, offer = next(iter(game.state.pending_trades.items()))
    return PublicOffer(
        trade_id=trade_id,
        created_revision=offer["created_revision"],
        from_player=offer["from_player"],
        to_player=offer["to_player"],
        give_properties=tuple(offer["give_properties"]),
        give_money=offer["give_money"],
        want_properties=tuple(offer["want_properties"]),
        want_money=offer["want_money"],
    )


def build_decision_view(game: MonopolyGame, viewer_id: int | None) -> DecisionView:
    """Build the complete public view without exposing replay-private state."""
    if viewer_id is not None and not 0 <= viewer_id < len(game.players):
        raise ValueError("Invalid viewer ID")
    actor = game.decision_player
    used = tuple(game.state.trade_targets_this_turn)
    can_propose = (
        game.rules_id == TRADE_RULES_ID
        and game.state.phase == "asset_management"
        and not game.state.roll_owed
        and actor == game.current_player
        and len(used) < 2
    )
    eligible = tuple(
        player.id
        for player in game.players
        if can_propose and player.id != actor and not player.bankrupt and player.id not in used
    )
    tradeable = tuple(
        position
        for position, prop in sorted(game.property_manager.properties.items())
        if prop.owner is not None and tradeable_property(game, position, prop.owner)[0]
    )
    properties = tuple(
        PublicProperty(
            position=position,
            name=Board.get_space(position).name,
            owner=prop.owner,
            houses=prop.houses,
            mortgaged=prop.mortgaged,
            price=get_property_cost(position),
            redemption_cost=get_unmortgage_cost(position),
            group=prop.color.name.lower() if prop.color is not None else None,
        )
        for position, prop in sorted(game.property_manager.properties.items())
    )
    debt = PublicDebt(**game.state.obligations[0]) if game.state.obligations else None
    purchase = (
        game.players[game.current_player].position
        if game.state.phase == "purchase_decision"
        else None
    )
    return DecisionView(
        contract_version=DECISION_CONTRACT_VERSION,
        rules_id=game.rules_id,
        revision=game.state.revision,
        viewer_id=viewer_id,
        turn_owner=game.current_player,
        decision_player=actor,
        phase=game.state.phase,
        turn_number=game.turn_number,
        roll_owed=game.state.roll_owed,
        doubles_count=game.doubles_count,
        last_roll=game.last_roll,
        houses_remaining=game.houses_remaining,
        hotels_remaining=game.hotels_remaining,
        debt=debt,
        purchase_property=purchase,
        players=tuple(
            PublicPlayer(
                id=player.id,
                name=player.name,
                money=player.money,
                position=player.position,
                in_jail=player.in_jail,
                jail_turns=player.jail_turns,
                jail_cards=player.jail_cards,
                bankrupt=player.bankrupt,
            )
            for player in game.players
        ),
        properties=properties,
        legal_actions=tuple(
            PublicAction(
                type=type(action).__name__,
                player_id=action.player_id,
                property_id=getattr(action, "property_id", None),
            )
            for action in legal_actions(game, actor)
        ),
        trade=TradeCapability(can_propose, max(0, 2 - len(used)), used, eligible, tradeable),
        pending_offer=_pending_offer(game),
    )
