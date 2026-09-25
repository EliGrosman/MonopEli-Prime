"""Guided, public-state-only Jev policy for native engine commands."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from typing import Any, Literal

from agents.rule_based import RuleBasedAgent
from monopoly_engine import (
    AcceptTrade,
    Action,
    Board,
    BuildHotel,
    BuildHouse,
    BuyProperty,
    DecisionView,
    DeclareBankruptcy,
    EndTurn,
    MortgageProperty,
    PayJailFine,
    ProposeTrade,
    RejectTrade,
    RollDice,
    SellHotel,
    SellHouse,
    UnmortgageProperty,
    UseJailCard,
)
from monopoly_engine.actions import PassBuy, SellBuildingGroup
from monopoly_engine.board import PropertySpace, RailroadSpace, UtilitySpace
from monopoly_engine.rules import get_building_cost, get_house_sale_value, get_mortgage_value

from .provider import JevRuntime, ProviderError
from .types import (
    AgentInspection,
    ChoiceAnswer,
    DecisionCallBudget,
    DecisionOutcome,
    QuestionBatch,
    StrategyMemory,
    StrategyUpdate,
    TradeMemoryEntry,
)

PROMPT_PROTOCOL_VERSION = "guided-jev-v1"
RESERVE_OPTIONS = (0, 100, 200, 300, 500, 750)
REGULAR_GROUPS = (
    "brown",
    "light_blue",
    "magenta",
    "orange",
    "red",
    "yellow",
    "green",
    "dark_blue",
)


@dataclass(frozen=True)
class TradeSuggestion:
    key: str
    action: ProposeTrade
    proposer_gain: int
    recipient_gain: int


def _economic_fingerprint(view: DecisionView) -> str:
    actor_cash = view.players[view.decision_player].money
    payload = {
        "actor": view.decision_player,
        "cash_band": actor_cash // 100,
        "debt": asdict(view.debt) if view.debt else None,
        "properties": [
            (item.position, item.owner, item.houses, item.mortgaged)
            for item in view.properties
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _fingerprint(view: DecisionView) -> str:
    payload = {
        "economic": _economic_fingerprint(view),
        "phase": view.phase,
        "turn_owner": view.turn_owner,
        "decision_player": view.decision_player,
        "positions": [player.position for player in view.players],
        "legal_actions": [asdict(action) for action in view.legal_actions],
        "pending_offer": asdict(view.pending_offer) if view.pending_offer else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def public_request_state(view: DecisionView, memory: StrategyMemory) -> dict[str, Any]:
    """Build the exact allowlisted payload sent to Jev."""
    properties: list[dict[str, Any]] = []
    for item in view.properties:
        space = Board.get_space(item.position)
        static: dict[str, Any] = {
            "position": item.position,
            "name": item.name,
            "price": item.price,
            "mortgage_value": get_mortgage_value(item.position),
            "redemption_cost": item.redemption_cost,
            "group": item.group,
        }
        if isinstance(space, PropertySpace):
            static.update(
                {
                    "building_cost": space.house_cost,
                    "rents": list(space.rent),
                }
            )
        elif isinstance(space, RailroadSpace):
            static["rents"] = list(space.rent)
        elif isinstance(space, UtilitySpace):
            static["rent_rule"] = "four or ten times a fresh dice roll"
        properties.append(
            {
                **static,
                "owner": item.owner,
                "houses": item.houses,
                "mortgaged": item.mortgaged,
            }
        )
    return {
        "protocol_version": PROMPT_PROTOCOL_VERSION,
        "decision": {
            "contract_version": view.contract_version,
            "rules_id": view.rules_id,
            "revision": view.revision,
            "turn_owner": view.turn_owner,
            "decision_player": view.decision_player,
            "phase": view.phase,
            "turn_number": view.turn_number,
            "roll_owed": view.roll_owed,
            "doubles_count": view.doubles_count,
            "last_roll": list(view.last_roll) if view.last_roll else None,
            "houses_remaining": view.houses_remaining,
            "hotels_remaining": view.hotels_remaining,
            "debt": asdict(view.debt) if view.debt else None,
            "purchase_property": view.purchase_property,
        },
        "players": [
            {
                "id": player.id,
                "label": f"Seat {player.id + 1}",
                "money": player.money,
                "position": player.position,
                "in_jail": player.in_jail,
                "jail_turns": player.jail_turns,
                "jail_cards": player.jail_cards,
                "bankrupt": player.bankrupt,
            }
            for player in view.players
        ],
        "properties": properties,
        "legal_actions": [asdict(action) for action in view.legal_actions],
        "trade": asdict(view.trade),
        "pending_offer": asdict(view.pending_offer) if view.pending_offer else None,
        "strategy": memory.public_dict(),
    }


class _DecisionSession:
    def __init__(
        self,
        agent: GuidedJevAgent,
        view: DecisionView,
        memory: StrategyMemory,
        deadline_seconds: float,
    ) -> None:
        self.agent = agent
        self.view = view
        self.memory = memory
        self.call_budget = DecisionCallBudget()
        self.deadline = time.monotonic() + deadline_seconds
        self.elapsed_seconds = 0.0
        self.model: str | None = None
        self.answer_reference: str | None = None

    async def choice(
        self,
        question_id: str,
        instructions: str,
        criteria: dict[str, Any],
        *,
        extra_state: dict[str, Any] | None = None,
    ) -> str:
        state = public_request_state(self.view, self.memory)
        if extra_state:
            state["guided_step"] = extra_state
        request = QuestionBatch(
            state=state,
            model=self.agent.model,
            questions={
                question_id: {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": criteria,
                }
            },
        )
        result = await self.agent.runtime.evaluate(
            self.agent.game_id,
            self.agent.session_budget_id,
            request,
            self.call_budget,
            self.deadline,
        )
        answer = result.answers[question_id]
        if not isinstance(answer, ChoiceAnswer):
            raise ProviderError("unexpected_answer_type")
        self.elapsed_seconds += result.elapsed_seconds
        self.model = result.model
        self.answer_reference = result.request_id
        return answer.choice

    async def strategy(
        self, objective_criteria: dict[str, Any]
    ) -> tuple[str, int]:
        state = public_request_state(self.view, self.memory)
        request = QuestionBatch(
            state=state,
            model=self.agent.model,
            questions={
                "objective": {
                    "type": "choice",
                    "instructions": (
                        "Choose the most useful current Monopoly objective. These are preferences, "
                        "not game rules. Use only the public position in state."
                    ),
                    "criteria": objective_criteria,
                },
                "cash_reserve": {
                    "type": "choice",
                    "instructions": (
                        "Choose a soft cash reserve target for the acting player. It may be "
                        "exceeded when a legal opportunity or mandatory payment justifies it."
                    ),
                    "criteria": {
                        f"reserve_{amount}": f"Prefer to retain about ${amount}."
                        for amount in RESERVE_OPTIONS
                    },
                },
            },
        )
        result = await self.agent.runtime.evaluate(
            self.agent.game_id,
            self.agent.session_budget_id,
            request,
            self.call_budget,
            self.deadline,
        )
        objective = result.answers["objective"]
        reserve = result.answers["cash_reserve"]
        if not isinstance(objective, ChoiceAnswer) or not isinstance(reserve, ChoiceAnswer):
            raise ProviderError("unexpected_answer_type")
        self.elapsed_seconds += result.elapsed_seconds
        self.model = result.model
        self.answer_reference = result.request_id
        return objective.choice, int(reserve.choice.removeprefix("reserve_"))


class GuidedJevAgent:
    """Stateful Jev policy that accepts only detached public decisions."""

    def __init__(
        self,
        player_id: int,
        game_id: str,
        runtime: JevRuntime,
        *,
        model: str = "jev-1.13.0",
        session_budget_id: str | None = None,
    ) -> None:
        self.player_id = player_id
        self.game_id = game_id
        self.runtime = runtime
        self.model = model
        self.session_budget_id = session_budget_id or game_id
        self.memory = StrategyMemory()
        self.inspection = AgentInspection(player_id=player_id)
        self.stats: Counter[str] = Counter()
        self._suppressed_trade_fingerprints: set[str] = set()
        self._turn = -1
        self._optional_actions_this_turn = 0
        self._seen_fingerprints: Counter[str] = Counter()

    def reset(self) -> None:
        self.memory = StrategyMemory()
        self.inspection = AgentInspection(player_id=self.player_id)
        self.stats.clear()
        self._suppressed_trade_fingerprints.clear()
        self._turn = -1
        self._optional_actions_this_turn = 0
        self._seen_fingerprints.clear()

    def _needs_strategy_refresh(self, view: DecisionView, fingerprint: str) -> bool:
        if self.memory.strategy_version == 0:
            return True
        if view.phase == "debt_resolution" and self.memory.last_strategy_revision != view.revision:
            return True
        if view.turn_number - self.memory.last_strategy_own_turn >= 5:
            return True
        return fingerprint != self.memory.observed_public_fingerprint

    @staticmethod
    def _group_positions(view: DecisionView) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {group: [] for group in REGULAR_GROUPS}
        for prop in view.properties:
            if prop.group in result:
                result[prop.group].append(prop.position)
        return result

    def _objective_catalog(
        self, view: DecisionView
    ) -> tuple[dict[str, Any], dict[str, tuple[str, str, str | None]]]:
        criteria: dict[str, Any] = {
            "acquire": "Acquire affordable properties and look for a viable color group.",
            "liquidity": "Restore liquidity now, then return to acquisition or development.",
        }
        values: dict[str, tuple[str, str, str | None]] = {
            "acquire": (
                "Acquire useful properties",
                "Pursue a viable color group",
                None,
            ),
            "liquidity": (
                "Restore liquidity",
                "Return to acquisition or development",
                None,
            ),
        }
        props = {item.position: item for item in view.properties}
        for group, positions in self._group_positions(view).items():
            owned = [pos for pos in positions if props[pos].owner == self.player_id]
            if owned and len(owned) < len(positions):
                key = f"complete_{group}"
                criteria[key] = f"Complete the {group.replace('_', ' ')} group, then develop it."
                values[key] = (
                    f"Complete {group.replace('_', ' ')}",
                    f"Develop {group.replace('_', ' ')} rent income",
                    group,
                )
            if len(owned) == len(positions):
                mortgaged = any(props[pos].mortgaged for pos in positions)
                key = f"redeem_{group}" if mortgaged else f"develop_{group}"
                verb = "Redeem" if mortgaged else "Develop"
                criteria[key] = f"{verb} the {group.replace('_', ' ')} group while preserving cash."
                values[key] = (
                    f"{verb} {group.replace('_', ' ')}",
                    f"Maintain {group.replace('_', ' ')} income and liquidity",
                    group,
                )
        return criteria, values

    async def _refresh_strategy(
        self, session: _DecisionSession, fingerprint: str
    ) -> StrategyUpdate:
        criteria, values = self._objective_catalog(session.view)
        objective_id, reserve = await session.strategy(criteria)
        short, long, group = values[objective_id]
        updated = StrategyUpdate(
            objective_id=objective_id,
            short_term_objective=short,
            long_term_objective=long,
            target_group=group,
            cash_reserve_target=reserve,
            revision=session.view.revision,
            own_turn=session.view.turn_number,
            public_fingerprint=fingerprint,
        )
        session.memory = replace(
            session.memory,
            objective_id=objective_id,
            short_term_objective=short,
            long_term_objective=long,
            target_group=group,
            cash_reserve_target=reserve,
        )
        return updated

    @staticmethod
    def _action_key(item: Any) -> str:
        base = "".join(
            f"_{char.lower()}" if char.isupper() else char for char in item.type
        ).lstrip("_")
        return f"{base}_{item.property_id}" if item.property_id is not None else base

    def _action_from_public(self, item: Any, view: DecisionView) -> Action:
        kwargs: dict[str, int] = {"player_id": item.player_id}
        if item.property_id is not None:
            kwargs["property_id"] = item.property_id
        classes: dict[str, type[Action]] = {
            "RollDice": RollDice,
            "BuyProperty": BuyProperty,
            "PassBuy": PassBuy,
            "BuildHouse": BuildHouse,
            "BuildHotel": BuildHotel,
            "SellHouse": SellHouse,
            "SellHotel": SellHotel,
            "SellBuildingGroup": SellBuildingGroup,
            "MortgageProperty": MortgageProperty,
            "UnmortgageProperty": UnmortgageProperty,
            "PayJailFine": PayJailFine,
            "UseJailCard": UseJailCard,
            "DeclareBankruptcy": DeclareBankruptcy,
            "EndTurn": EndTurn,
        }
        if item.type == "AcceptTrade" and view.pending_offer:
            return AcceptTrade(item.player_id, view.pending_offer.trade_id)
        if item.type == "RejectTrade" and view.pending_offer:
            return RejectTrade(item.player_id, view.pending_offer.trade_id)
        return classes[item.type](**kwargs)

    def _action_description(self, item: Any, view: DecisionView) -> str:
        simple = {
            "RollDice": "Roll the dice and advance the current turn.",
            "PassBuy": "Decline this purchase; this does not end the turn.",
            "PayJailFine": "Pay the $50 jail fine.",
            "UseJailCard": "Use a Get Out of Jail Free card.",
            "DeclareBankruptcy": (
                "Declare bankruptcy because attainable assets cannot satisfy the debt."
            ),
            "EndTurn": "End the current turn.",
            "AcceptTrade": "Accept and execute the exact pending trade.",
            "RejectTrade": "Reject the pending trade and transfer nothing.",
        }
        if item.type in simple:
            return simple[item.type]
        prop = next(
            (candidate for candidate in view.properties if candidate.position == item.property_id),
            None,
        )
        name = prop.name if prop else f"property {item.property_id}"
        descriptions = {
            "BuyProperty": f"Buy {name} for ${prop.price if prop else 0}.",
            "BuildHouse": f"Develop {name} for ${get_building_cost(item.property_id)}.",
            "BuildHotel": f"Build a hotel on {name} for ${get_building_cost(item.property_id)}.",
            "SellHouse": (
                f"Sell one building step from {name} for "
                f"${get_house_sale_value(item.property_id)}."
            ),
            "SellHotel": (
                f"Sell the hotel from {name} for ${get_house_sale_value(item.property_id)}."
            ),
            "SellBuildingGroup": f"Liquidate all buildings in the group beginning at {name}.",
            "MortgageProperty": f"Mortgage {name} for ${get_mortgage_value(item.property_id)}.",
            "UnmortgageProperty": f"Redeem {name} for ${prop.redemption_cost if prop else 0}.",
        }
        return descriptions[item.type]

    @staticmethod
    def _phase_instruction(phase: str) -> str:
        return {
            "pre_roll": "Choose whether to manage an asset first or roll now.",
            "jail_decision": (
                "Choose whether to attempt doubles, pay, use a card, or manage an asset first."
            ),
            "purchase_decision": (
                "Choose whether to buy the pending property, decline it, or manage an asset first. "
                "The reserve is a preference, not a rule."
            ),
            "asset_management": (
                "Choose the most useful asset action, whether to consider a trade, or whether "
                "to end."
            ),
            "debt_resolution": (
                "Choose the legal liquidation that best meets the obligation while preserving "
                "useful assets."
            ),
            "trade_response": (
                "Choose whether to accept or reject the complete pending terms. There is no "
                "counteroffer."
            ),
        }[phase]

    def _fallback(self, view: DecisionView, reason: str) -> Action:
        if view.pending_offer is not None:
            return RejectTrade(self.player_id, view.pending_offer.trade_id)
        return RuleBasedAgent(self.player_id).choose_decision(view)

    @staticmethod
    def _complete_groups(
        view: DecisionView, player_id: int, owners: dict[int, int] | None = None
    ) -> set[str]:
        groups: dict[str, list[Any]] = {name: [] for name in REGULAR_GROUPS}
        for prop in view.properties:
            if prop.group in groups:
                groups[prop.group].append(prop)
        return {
            name
            for name, props in groups.items()
            if props
            and all(
                (owners or {}).get(prop.position, prop.owner) == player_id
                and not prop.mortgaged
                for prop in props
            )
        }

    @classmethod
    def _strategic_value(
        cls, view: DecisionView, player_id: int, owners: dict[int, int] | None = None
    ) -> int:
        value = view.players[player_id].money
        for prop in view.properties:
            if (owners or {}).get(prop.position, prop.owner) == player_id:
                value += prop.price - (prop.redemption_cost if prop.mortgaged else 0)
        complete = cls._complete_groups(view, player_id, owners)
        for group in complete:
            value += 2 * sum(prop.price for prop in view.properties if prop.group == group)
        return value

    @staticmethod
    def _bundles(positions: list[int]) -> list[tuple[int, ...]]:
        return [()] + [(position,) for position in positions] + list(combinations(positions, 2))

    def _trade_suggestions(self, view: DecisionView) -> list[TradeSuggestion]:
        actor = self.player_id
        tradeable = set(view.trade.tradeable_properties)
        my_positions = sorted(
            prop.position
            for prop in view.properties
            if prop.owner == actor and prop.position in tradeable
        )
        templates: set[tuple[int, tuple[int, ...], tuple[int, ...]]] = set()
        groups = self._group_positions(view)
        props = {item.position: item for item in view.properties}
        for recipient in view.trade.eligible_recipients:
            their_positions = sorted(
                prop.position
                for prop in view.properties
                if prop.owner == recipient and prop.position in tradeable
            )
            for positions in groups.values():
                mine_missing = tuple(pos for pos in positions if props[pos].owner != actor)
                if 0 < len(mine_missing) <= 2 and all(
                    pos in their_positions for pos in mine_missing
                ):
                    for give in self._bundles(my_positions):
                        templates.add((recipient, give, mine_missing))
                their_missing = tuple(pos for pos in positions if props[pos].owner != recipient)
                if 0 < len(their_missing) <= 2 and all(
                    pos in my_positions for pos in their_missing
                ):
                    for want in self._bundles(their_positions):
                        templates.add((recipient, their_missing, want))
        suggestions: list[TradeSuggestion] = []
        for index, (recipient, give, want) in enumerate(sorted(templates)):
            owners = {position: recipient for position in give}
            owners.update({position: actor for position in want})
            before_a = self._complete_groups(view, actor)
            before_r = self._complete_groups(view, recipient)
            after_a = self._complete_groups(view, actor, owners)
            after_r = self._complete_groups(view, recipient, owners)
            if not before_a <= after_a or not before_r <= after_r:
                continue
            if not (after_a - before_a or after_r - before_r):
                continue
            delta_a = self._strategic_value(view, actor, owners) - self._strategic_value(
                view, actor
            )
            delta_r = self._strategic_value(
                view, recipient, owners
            ) - self._strategic_value(view, recipient)
            cap_a = min(500, max(0, view.players[actor].money - 200))
            cap_r = min(500, max(0, view.players[recipient].money - 200))
            lower = max(25 - delta_r, -cap_r)
            upper = min(delta_a - 25, cap_a)
            if lower > upper:
                continue
            ideal = (delta_a - delta_r) / 2
            cash_candidates = {lower, upper, math.floor(ideal), math.ceil(ideal)}
            cash = min(
                (int(value) for value in cash_candidates if lower <= value <= upper),
                key=lambda value: (abs((delta_a - value) - (delta_r + value)), abs(value), value),
            )
            action = ProposeTrade(
                actor,
                recipient,
                list(give),
                max(cash, 0),
                list(want),
                max(-cash, 0),
            )
            suggestions.append(
                TradeSuggestion(
                    key=f"suggestion_{index}",
                    action=action,
                    proposer_gain=delta_a - cash,
                    recipient_gain=delta_r + cash,
                )
            )
        suggestions.sort(
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
        return [
            replace(item, key=f"suggestion_{index}")
            for index, item in enumerate(suggestions[:8])
        ]

    @staticmethod
    def _offer_description(action: ProposeTrade, view: DecisionView) -> str:
        names = {item.position: item.name for item in view.properties}
        give = [names[pos] for pos in action.give_properties]
        want = [names[pos] for pos in action.want_properties]
        return (
            f"Offer Seat {action.to_player + 1}: give properties {give or ['none']} and "
            f"${action.give_money}; request properties {want or ['none']} and ${action.want_money}."
        )

    def _partial_offer(
        self,
        recipient: int,
        give: list[int],
        want: list[int],
        *,
        cash_direction: str | None = None,
        cash_range: tuple[int, int] | None = None,
        give_money: int | None = None,
        want_money: int | None = None,
    ) -> dict[str, Any]:
        """Return a detached snapshot for one dependent trade question."""
        return {
            "proposer": self.player_id,
            "recipient": recipient,
            "give_properties": list(give),
            "want_properties": list(want),
            "cash_direction": cash_direction,
            "cash_range": (
                None
                if cash_range is None
                else {"minimum": cash_range[0], "maximum": cash_range[1]}
            ),
            "give_money": give_money,
            "want_money": want_money,
        }

    async def _choose_integer(
        self,
        session: _DecisionSession,
        payer: int,
        maximum: int,
        recipient: int,
        give: list[int],
        want: list[int],
        cash_direction: str,
    ) -> tuple[int, tuple[int, int]]:
        low, high = 1, maximum
        while high - low + 1 > 255:
            width = math.ceil((high - low + 1) / 255)
            ranges: dict[str, str] = {}
            endpoints: dict[str, tuple[int, int]] = {}
            cursor = low
            index = 0
            while cursor <= high:
                end = min(high, cursor + width - 1)
                key = f"range_{index}"
                ranges[key] = f"Choose a cash amount from ${cursor} through ${end}, inclusive."
                endpoints[key] = (cursor, end)
                cursor = end + 1
                index += 1
            selected = await session.choice(
                "cash_range",
                f"Choose the cash range for Seat {payer + 1}; exact amount is selected next.",
                ranges,
                extra_state={
                    "partial_offer": self._partial_offer(
                        recipient,
                        give,
                        want,
                        cash_direction=cash_direction,
                        cash_range=(low, high),
                        give_money=None if cash_direction == "proposer_pays" else 0,
                        want_money=None if cash_direction == "recipient_pays" else 0,
                    )
                },
            )
            low, high = endpoints[selected]
        criteria = {f"cash_{amount}": f"Use exactly ${amount}." for amount in range(low, high + 1)}
        selected = await session.choice(
            "cash_amount",
            f"Choose the exact integer cash amount for Seat {payer + 1}.",
            criteria,
            extra_state={
                "partial_offer": self._partial_offer(
                    recipient,
                    give,
                    want,
                    cash_direction=cash_direction,
                    cash_range=(low, high),
                    give_money=None if cash_direction == "proposer_pays" else 0,
                    want_money=None if cash_direction == "recipient_pays" else 0,
                )
            },
        )
        return int(selected.removeprefix("cash_")), (low, high)

    async def _construct_offer(
        self, session: _DecisionSession, recipient: int
    ) -> ProposeTrade | None:
        view = session.view
        tradeable = set(view.trade.tradeable_properties)
        mine = [
            prop
            for prop in view.properties
            if prop.owner == self.player_id and prop.position in tradeable
        ]
        theirs = [
            prop
            for prop in view.properties
            if prop.owner == recipient and prop.position in tradeable
        ]
        give_criteria = {"none": "Give no property in this first slot."}
        give_criteria.update(
            {f"property_{prop.position}": f"Give {prop.name}." for prop in mine}
        )
        first_give_key = await session.choice(
            "first_give_property",
            f"Choose the first property to give Seat {recipient + 1}, or none.",
            give_criteria,
            extra_state={
                "partial_offer": self._partial_offer(recipient, [], [])
            },
        )
        give = [] if first_give_key == "none" else [int(first_give_key.removeprefix("property_"))]
        want_criteria = {} if not give else {"none": "Request no property in this first slot."}
        want_criteria.update(
            {f"property_{prop.position}": f"Request {prop.name}." for prop in theirs}
        )
        if not want_criteria:
            return None
        first_want_key = await session.choice(
            "first_want_property",
            (
                f"Choose the first property to request from Seat {recipient + 1}. "
                "A property is required if none is given."
            ),
            want_criteria,
            extra_state={
                "partial_offer": self._partial_offer(recipient, give, [])
            },
        )
        want = [] if first_want_key == "none" else [int(first_want_key.removeprefix("property_"))]
        give_second = {"none": "Do not add a second given property."}
        give_second.update(
            {
                f"property_{prop.position}": f"Also give {prop.name}."
                for prop in mine
                if prop.position not in give
            }
        )
        second_give_key = await session.choice(
            "second_give_property",
            "Choose an optional second property to give.",
            give_second,
            extra_state={
                "partial_offer": self._partial_offer(recipient, give, want)
            },
        )
        if second_give_key != "none":
            give.append(int(second_give_key.removeprefix("property_")))

        want_second = {"none": "Do not add a second requested property."}
        want_second.update(
            {
                f"property_{prop.position}": f"Also request {prop.name}."
                for prop in theirs
                if prop.position not in want
            }
        )
        second_want_key = await session.choice(
            "second_want_property",
            "Choose an optional second property to request.",
            want_second,
            extra_state={
                "partial_offer": self._partial_offer(recipient, give, want)
            },
        )
        if second_want_key != "none":
            want.append(int(second_want_key.removeprefix("property_")))

        direction = {
            "none": "No cash changes hands.",
            "proposer_pays": f"Seat {self.player_id + 1} pays cash.",
            "recipient_pays": f"Seat {recipient + 1} pays cash.",
        }
        cash_direction = await session.choice(
            "cash_direction",
            "Choose whether cash is included and its one permitted direction.",
            direction,
            extra_state={
                "partial_offer": self._partial_offer(recipient, give, want)
            },
        )
        give_money = want_money = 0
        selected_cash_range: tuple[int, int] | None = None
        if cash_direction == "proposer_pays":
            maximum = view.players[self.player_id].money
            if maximum:
                give_money, selected_cash_range = await self._choose_integer(
                    session,
                    self.player_id,
                    maximum,
                    recipient,
                    give,
                    want,
                    cash_direction,
                )
        elif cash_direction == "recipient_pays":
            maximum = view.players[recipient].money
            if maximum:
                want_money, selected_cash_range = await self._choose_integer(
                    session,
                    recipient,
                    maximum,
                    recipient,
                    give,
                    want,
                    cash_direction,
                )
        action = ProposeTrade(
            self.player_id, recipient, give, give_money, want, want_money
        )
        final = await session.choice(
            "complete_offer",
            "Choose whether to propose these exact complete terms or abandon them.",
            {
                "propose": self._offer_description(action, view),
                "abandon": "Do not consume an engine proposal or transfer anything.",
            },
            extra_state={
                "partial_offer": self._partial_offer(
                    recipient,
                    give,
                    want,
                    cash_direction=cash_direction,
                    cash_range=selected_cash_range,
                    give_money=give_money,
                    want_money=want_money,
                ),
                "complete_offer": action.to_dict(),
            },
        )
        return action if final == "propose" else None

    async def _trade_decision(
        self, session: _DecisionSession
    ) -> ProposeTrade | None:
        suggestions = self._trade_suggestions(session.view)
        criteria: dict[str, str] = {
            item.key: (
                self._offer_description(item.action, session.view)
                + (
                    f" Suggestion heuristic gains: proposer {item.proposer_gain}, "
                    f"recipient {item.recipient_gain}."
                )
            )
            for item in suggestions
        }
        criteria["skip_trade"] = "Do not make a trade at this unchanged position."
        criteria.update(
            {
                f"construct_offer_to_{recipient}": (
                    f"Construct another complete legal offer to Seat {recipient + 1}; "
                    "suggestions do not limit legality."
                )
                for recipient in session.view.trade.eligible_recipients
            }
        )
        selected = await session.choice(
            "trade_path",
            "Choose a complete suggested offer, construct another legal offer, or skip trading.",
            criteria,
            extra_state={"suggestion_policy": "foundation-benefit-candidates-v1 assumptions only"},
        )
        if selected == "skip_trade":
            return None
        if selected.startswith("suggestion_"):
            return next(item.action for item in suggestions if item.key == selected)
        recipient = int(selected.removeprefix("construct_offer_to_"))
        return await self._construct_offer(session, recipient)

    async def decide(self, view: DecisionView) -> DecisionOutcome:
        """Select one revision-bound command from detached public state."""
        if view.decision_player != self.player_id or view.phase == "terminal":
            return DecisionOutcome(None, "fallback", "No live decision", "not_decision_owner")
        fingerprint = _fingerprint(view)
        economic_fingerprint = _economic_fingerprint(view)
        if view.turn_number != self._turn:
            self._turn = view.turn_number
            self._optional_actions_this_turn = 0
            self._seen_fingerprints.clear()
        self._seen_fingerprints[fingerprint] += 1
        forced_fallback = (
            view.phase != "debt_resolution"
            and (
                self._optional_actions_this_turn >= 20
                or self._seen_fingerprints[fingerprint] >= 3
            )
        )
        deadline_seconds = 30.0 if view.trade.can_propose else 12.0
        session = _DecisionSession(self, view, replace(self.memory), deadline_seconds)
        staged: StrategyUpdate | None = None
        command: Action | None
        try:
            if forced_fallback:
                raise ProviderError("local_progress_guard")
            options: dict[str, Any | None] = {
                self._action_key(item): item for item in view.legal_actions
            }
            can_trade = (
                view.trade.can_propose
                and bool(view.trade.eligible_recipients)
                and bool(view.trade.tradeable_properties)
                and fingerprint not in self._suppressed_trade_fingerprints
            )
            if can_trade:
                options["consider_trade"] = None
            if len(options) == 1:
                key, item = next(iter(options.items()))
                if key == "consider_trade":
                    raise ProviderError("no_ordinary_continuation")
                assert item is not None
                command = self._action_from_public(item, view)
                summary = self._summary(command, session.memory)
                return DecisionOutcome(
                    command,
                    "forced",
                    summary,
                    memory_version=self.memory.strategy_version,
                )
            if self._needs_strategy_refresh(view, economic_fingerprint):
                staged = await self._refresh_strategy(session, economic_fingerprint)
            criteria = {
                key: (
                    "Consider whether a complete trade is worthwhile before choosing exact terms."
                    if key == "consider_trade"
                    else self._action_description(item, view)
                )
                for key, item in options.items()
            }
            selected = await session.choice(
                "action",
                self._phase_instruction(view.phase),
                criteria,
            )
            if selected == "consider_trade":
                command = await self._trade_decision(session)
                if command is None:
                    self._suppressed_trade_fingerprints.add(fingerprint)
                    ordinary = {key: item for key, item in options.items() if item is not None}
                    selected = await session.choice(
                        "post_trade_action",
                        (
                            "Trading was skipped or abandoned. Choose the legal non-trade "
                            "continuation."
                        ),
                        {
                            key: self._action_description(item, view)
                            for key, item in ordinary.items()
                        },
                    )
                    command = self._action_from_public(ordinary[selected], view)
            else:
                item = options[selected]
                assert item is not None
                command = self._action_from_public(item, view)
            return DecisionOutcome(
                command,
                "jev",
                self._summary(command, session.memory),
                staged_strategy=staged,
                provider_model=session.model,
                answer_reference=session.answer_reference,
                attempts=session.call_budget.attempts,
                input_tokens=session.call_budget.input_tokens,
                elapsed_seconds=session.elapsed_seconds,
                memory_version=self.memory.strategy_version,
            )
        except ProviderError as error:
            command = self._fallback(view, error.code)
            self.stats[f"fallback_{error.code}"] += 1
            return DecisionOutcome(
                command,
                "fallback",
                self._summary(command, self.memory),
                fallback_reason=error.code,
                attempts=session.call_budget.attempts,
                input_tokens=session.call_budget.input_tokens,
                elapsed_seconds=session.elapsed_seconds,
                memory_version=self.memory.strategy_version,
            )

    @staticmethod
    def _summary(command: Action, memory: StrategyMemory) -> str:
        name = type(command).__name__
        if isinstance(command, ProposeTrade):
            action = f"Proposed a complete offer to Seat {command.to_player + 1}"
        elif isinstance(command, (AcceptTrade, RejectTrade)):
            action = (
                "Accepted the pending offer"
                if isinstance(command, AcceptTrade)
                else "Rejected the pending offer"
            )
        elif hasattr(command, "property_id"):
            position = getattr(command, "property_id")
            action_name = name.replace("Property", "").replace("House", " house")
            action = f"{action_name} at {Board.get_space(position).name}"
        else:
            action = name.replace("Dice", " dice").replace("Turn", " turn")
        return f"{action}; objective: {memory.short_term_objective}."

    def commit_outcome(self, outcome: DecisionOutcome, revision: int, turn_number: int) -> None:
        if outcome.staged_strategy is not None:
            update = outcome.staged_strategy
            self.memory.objective_id = update.objective_id
            self.memory.short_term_objective = update.short_term_objective
            self.memory.long_term_objective = update.long_term_objective
            self.memory.target_group = update.target_group
            self.memory.cash_reserve_target = update.cash_reserve_target
            self.memory.last_strategy_revision = update.revision
            self.memory.last_strategy_own_turn = update.own_turn
            self.memory.observed_public_fingerprint = update.public_fingerprint
            self.memory.strategy_version += 1
        if outcome.command is not None and not isinstance(outcome.command, (RollDice, EndTurn)):
            self._optional_actions_this_turn += 1
        self.inspection.basis_revision = revision
        self.inspection.status = "fallback" if outcome.source == "fallback" else "idle"
        self.inspection.latest_summary = outcome.summary
        self.inspection.fallback_reason = outcome.fallback_reason
        self.inspection.short_term_objective = self.memory.short_term_objective
        self.inspection.long_term_objective = self.memory.long_term_objective
        self.inspection.cash_reserve_target = self.memory.cash_reserve_target
        self.inspection.sequence += 1

    def observe_transition(
        self,
        before: DecisionView,
        after: DecisionView,
        structured_events: list[dict[str, Any]],
    ) -> None:
        new_fingerprint = _economic_fingerprint(after)
        if new_fingerprint != self.memory.observed_public_fingerprint:
            self.memory.observed_public_fingerprint = ""
        for event in structured_events:
            event_type = event.get("type")
            if event_type not in {"trade_proposed", "trade_accepted", "trade_rejected"}:
                continue
            from_player = event.get("from_player")
            to_player = event.get("to_player")
            if not isinstance(from_player, int) or not isinstance(to_player, int):
                continue
            if self.player_id not in (from_player, to_player):
                continue
            outcomes: dict[str, Literal["proposed", "accepted", "rejected"]] = {
                "trade_proposed": "proposed",
                "trade_accepted": "accepted",
                "trade_rejected": "rejected",
            }
            outcome = outcomes[event_type]
            entry = TradeMemoryEntry(
                revision=after.revision,
                trade_id=int(event["trade_id"]),
                role="proposer" if from_player == self.player_id else "recipient",
                outcome=outcome,
                from_player=from_player,
                to_player=to_player,
                give_properties=tuple(event.get("give_properties", [])),
                give_money=int(event.get("give_money", 0)),
                want_properties=tuple(event.get("want_properties", [])),
                want_money=int(event.get("want_money", 0)),
            )
            self.memory.recent_trades = (self.memory.recent_trades + [entry])[-8:]
