"""Deterministic, public-state trading wrappers for headless experiments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Literal

from monopoly_engine import AcceptTrade, EndTurn, ProposeTrade, RejectTrade
from monopoly_engine.board import Board, PropertySpace, RailroadSpace, UtilitySpace
from monopoly_engine.types import PROPERTY_GROUPS, PropertyColor, TradeOfferData

if TYPE_CHECKING:
    from monopoly_engine import Action, MonopolyGame
from monopoly_gym.action_space import ActionEncoder

from .base import Agent

MIN_GAIN = 25
MAX_CASH_ADJUSTMENT = 500
CASH_RESERVE = 200
REGULAR_COLORS = tuple(
    color
    for color in PROPERTY_GROUPS
    if color not in (PropertyColor.RAILROAD, PropertyColor.UTILITY)
)


@dataclass(frozen=True)
class ScoredOffer:
    action: ProposeTrade
    proposer_gain: int
    recipient_gain: int


def _structural_signature(game: MonopolyGame) -> tuple[tuple[int | None, bool, int], ...]:
    return tuple(
        (prop.owner, prop.mortgaged, prop.houses)
        for prop in game.property_manager.properties.values()
    )


def _complete_groups(
    game: MonopolyGame, player_id: int, owners: dict[int, int] | None = None
) -> set[PropertyColor]:
    result = set()
    for color in REGULAR_COLORS:
        positions = PROPERTY_GROUPS[color]
        if all(
            (owners or {}).get(pos, game.property_manager.properties[pos].owner) == player_id
            and not game.property_manager.properties[pos].mortgaged
            for pos in positions
        ):
            result.add(color)
    return result


def _strategic_value(
    game: MonopolyGame, player_id: int, owners: dict[int, int] | None = None
) -> int:
    value = game.players[player_id].money
    for position, prop in game.property_manager.properties.items():
        owner = (owners or {}).get(position, prop.owner)
        if owner != player_id:
            continue
        space = Board.get_space(position)
        if not isinstance(space, (PropertySpace, RailroadSpace, UtilitySpace)):
            continue
        value += space.cost
        if prop.mortgaged:
            from monopoly_engine.rules import get_unmortgage_cost

            value -= get_unmortgage_cost(position)
    for color in _complete_groups(game, player_id, owners):
        value += 2 * sum(Board.get_space(pos).cost for pos in PROPERTY_GROUPS[color])
    return value


def _tradeable_positions(game: MonopolyGame, player_id: int) -> list[int]:
    result = []
    for position in sorted(game.property_manager.get_owned_by(player_id)):
        prop = game.property_manager.properties[position]
        group = PROPERTY_GROUPS.get(prop.color, ()) if prop.color is not None else ()
        if not any(game.property_manager.properties[pos].houses for pos in group):
            result.append(position)
    return result


def _bundles(positions: list[int]) -> list[tuple[int, ...]]:
    return [()] + [(p,) for p in positions] + list(combinations(positions, 2))


def _ownership_after(offer: TradeOfferData) -> dict[int, int]:
    owners = {position: offer["to_player"] for position in offer["give_properties"]}
    owners.update({position: offer["from_player"] for position in offer["want_properties"]})
    return owners


def _property_gains(game: MonopolyGame, offer: TradeOfferData) -> tuple[int, int] | None:
    proposer, recipient = offer["from_player"], offer["to_player"]
    before_p = _complete_groups(game, proposer)
    before_r = _complete_groups(game, recipient)
    owners = _ownership_after(offer)
    after_p = _complete_groups(game, proposer, owners)
    after_r = _complete_groups(game, recipient, owners)
    if not before_p <= after_p or not before_r <= after_r:
        return None
    if not (after_p - before_p or after_r - before_r):
        return None
    return (
        _strategic_value(game, proposer, owners) - _strategic_value(game, proposer),
        _strategic_value(game, recipient, owners) - _strategic_value(game, recipient),
    )


def _cash_adjustment(
    game: MonopolyGame, proposer: int, recipient: int, proposer_delta: int, recipient_delta: int
) -> tuple[int, int, int] | None:
    proposer_capacity = min(
        MAX_CASH_ADJUSTMENT, max(0, game.players[proposer].money - CASH_RESERVE)
    )
    recipient_capacity = min(
        MAX_CASH_ADJUSTMENT, max(0, game.players[recipient].money - CASH_RESERVE)
    )
    lower = max(MIN_GAIN - recipient_delta, -recipient_capacity)
    upper = min(proposer_delta - MIN_GAIN, proposer_capacity)
    if lower > upper:
        return None
    ideal = (proposer_delta - recipient_delta) / 2
    candidates = {int(lower), int(upper), int(ideal // 1), int(-(-ideal // 1))}
    feasible = [cash for cash in candidates if lower <= cash <= upper]
    cash = min(
        feasible, key=lambda x: (abs((proposer_delta - x) - (recipient_delta + x)), abs(x), x)
    )
    return cash, proposer_delta - cash, recipient_delta + cash


def score_offer(game: MonopolyGame, offer: TradeOfferData) -> tuple[int, int] | None:
    deltas = _property_gains(game, offer)
    if deltas is None:
        return None
    cash = offer["give_money"] - offer["want_money"]
    proposer_gain, recipient_gain = deltas[0] - cash, deltas[1] + cash
    if proposer_gain < MIN_GAIN or recipient_gain < MIN_GAIN or abs(cash) > MAX_CASH_ADJUSTMENT:
        return None
    if cash:
        payer = offer["from_player"] if cash > 0 else offer["to_player"]
        if game.players[payer].money - abs(cash) < CASH_RESERVE:
            return None
    return proposer_gain, recipient_gain


class TradingAgent(Agent):
    """Wrap an indexed heuristic with authoritative native trade decisions."""

    def __init__(self, base, response: Literal["mutual", "reject", "alternate"] = "mutual") -> None:
        super().__init__(base.player_id, name=f"Trading{base.name}")
        self.base = base
        self.response = response
        self.stats: Counter[str] = Counter()
        self._resume_action: Action | None = None
        self._template_cache: dict[
            tuple, list[tuple[int, tuple[int, ...], tuple[int, ...], int, int]]
        ] = {}

    def choose_action(self, observation, action_mask, game) -> int:
        """Compatibility path for ordinary indexed decisions."""
        return int(self.base.choose_action(observation, action_mask, game))

    def reset(self) -> None:
        self.base.reset()
        self.stats.clear()
        self._resume_action = None
        self._template_cache.clear()

    def _templates(
        self, game: MonopolyGame
    ) -> list[tuple[int, tuple[int, ...], tuple[int, ...], int, int]]:
        key = (self.player_id, _structural_signature(game))
        if key in self._template_cache:
            self.stats["template_cache_hits"] += 1
            return self._template_cache[key]
        templates: set[tuple[int, tuple[int, ...], tuple[int, ...]]] = set()
        my_positions = _tradeable_positions(game, self.player_id)
        give_bundles = _bundles(my_positions)
        for opponent in range(len(game.players)):
            if opponent == self.player_id or game.players[opponent].bankrupt:
                continue
            their_positions = _tradeable_positions(game, opponent)
            want_bundles = _bundles(their_positions)
            for color in REGULAR_COLORS:
                positions = PROPERTY_GROUPS[color]
                mine_missing = tuple(
                    pos
                    for pos in positions
                    if game.property_manager.properties[pos].owner != self.player_id
                )
                if 0 < len(mine_missing) <= 2 and all(
                    pos in their_positions for pos in mine_missing
                ):
                    for give in give_bundles:
                        templates.add((opponent, give, mine_missing))
                their_missing = tuple(
                    pos
                    for pos in positions
                    if game.property_manager.properties[pos].owner != opponent
                )
                if 0 < len(their_missing) <= 2 and all(
                    pos in my_positions for pos in their_missing
                ):
                    for want in want_bundles:
                        templates.add((opponent, their_missing, want))
        viable = []
        for opponent, give, want in sorted(templates):
            offer = TradeOfferData(
                from_player=self.player_id,
                to_player=opponent,
                give_properties=list(give),
                give_money=0,
                want_properties=list(want),
                want_money=0,
            )
            deltas = _property_gains(game, offer)
            if deltas is None:
                self.stats["rejected_no_new_group_or_breaks_group"] += 1
            else:
                viable.append((opponent, give, want, *deltas))
        self._template_cache[key] = viable
        self.stats["template_cache_misses"] += 1
        return viable

    def candidates(self, game: MonopolyGame, *, use_cache: bool = True) -> list[ScoredOffer]:
        templates = self._templates(game) if use_cache else self._uncached_templates(game)
        offers = []
        for opponent, give, want, proposer_delta, recipient_delta in templates:
            if opponent in game.state.trade_targets_this_turn:
                continue
            adjusted = _cash_adjustment(
                game, self.player_id, opponent, proposer_delta, recipient_delta
            )
            if adjusted is None:
                self.stats["rejected_no_mutual_cash_adjustment"] += 1
                continue
            cash, proposer_gain, recipient_gain = adjusted
            action = ProposeTrade(
                self.player_id, opponent, list(give), max(cash, 0), list(want), max(-cash, 0)
            )
            valid, _ = action.validate(game)
            if valid:
                offers.append(ScoredOffer(action, proposer_gain, recipient_gain))
        offers.sort(
            key=lambda item: (
                -item.proposer_gain,
                -item.recipient_gain,
                item.action.to_player,
                item.action.give_properties,
                item.action.want_properties,
                item.action.give_money,
                item.action.want_money,
            )
        )
        return offers

    def _uncached_templates(self, game: MonopolyGame):
        cache, self._template_cache = self._template_cache, {}
        try:
            return self._templates(game)
        finally:
            self._template_cache = cache

    def choose_native_action(self, game: MonopolyGame, encoder: ActionEncoder) -> Action:
        if game.state.phase == "trade_response":
            trade_id, offer = next(iter(game.state.pending_trades.items()))
            if (
                self.response == "reject"
                or (self.response == "alternate" and trade_id % 2 == 1)
                or score_offer(game, offer) is None
            ):
                self.stats["offers_rejected"] += 1
                return RejectTrade(self.player_id, trade_id)
            self.stats["offers_accepted"] += 1
            return AcceptTrade(self.player_id, trade_id)
        if self._resume_action is not None:
            action, self._resume_action = self._resume_action, None
            return action
        action = self.base.choose_decision(game.decision_view(self.player_id))
        if isinstance(action, EndTurn) and len(game.state.trade_targets_this_turn) < 2:
            candidates = self.candidates(game)
            if candidates:
                self.stats["offers_proposed"] += 1
                self._resume_action = action
                return candidates[0].action
        return action
