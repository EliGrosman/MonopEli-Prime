"""Diagnostic-only economic observer; certified rules/policies remain untouched.

Hooks are process-local and MUST NOT be used concurrently in threads. They intercept
semantic charge/card/movement boundaries, retain obligation provenance across debt
choices, and reconcile a transaction ledger against cash after every decision.
No RNG calls, engine state edits, or additional engine events are made.
"""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack, contextmanager
from functools import wraps
from typing import Any
from unittest.mock import patch

from evaluation import runner
from monopoly_engine import foundation
from monopoly_engine.board import Board
from monopoly_engine.rules import (
    get_building_cost,
    get_house_sale_value,
    get_mortgage_value,
    get_property_cost,
    get_unmortgage_cost,
)
from monopoly_engine.types import GO_SALARY, JAIL_FINE, PROPERTY_GROUPS, CardType, PropertyColor

VERSION = "economic-diagnostics-v1"
COLORS = [c for c in PROPERTY_GROUPS if c not in (PropertyColor.RAILROAD, PropertyColor.UTILITY)]


def snapshot(game):
    pm = game.property_manager
    groups = []
    for color in COLORS:
        props = [pm.properties[pos] for pos in PROPERTY_GROUPS[color]]
        owners = [p.owner for p in props]
        distinct = set(owners) - {None}
        groups.append(
            {
                "color": color.name,
                "positions": [p.position for p in props],
                "owners": owners,
                "complete_owner": next(iter(distinct))
                if len(distinct) == 1 and None not in owners
                else None,
                "fragmented": len(distinct) > 1,
                "buildings": [p.houses for p in props],
                "mortgages": [p.mortgaged for p in props],
            }
        )
    return {
        "turn": game.turn_number,
        "revision": game.state.revision,
        "cash": [p.money for p in game.players],
        "survivors": [p.id for p in game.players if not p.bankrupt],
        "groups": groups,
        "properties": [p.to_dict() for p in pm.properties.values()],
        "complete_groups": sum(g["complete_owner"] is not None for g in groups),
        "fragmented_groups": sum(g["fragmented"] for g in groups),
        "unowned": pm.get_unowned(),
        "building_units": sum(p.houses for p in pm.properties.values()),
        "mortgages": sum(p.mortgaged for p in pm.properties.values()),
        "houses_remaining": game.houses_remaining,
        "hotels_remaining": game.hotels_remaining,
    }


def structure(game):
    return tuple(
        (p.owner, p.houses, p.mortgaged) for p in game.property_manager.properties.values()
    )


class EconomicObserver:
    def __init__(self, capture=False):
        self.capture = capture
        self.context = "unknown"
        self.obligations = {}
        self.flows = Counter()
        self.player_flows = Counter()
        self.unmortgage_cycles = []
        self.mortgaged = set()
        self.actions = Counter()
        self.transactions = []
        self.eliminations = []
        self.snapshots = []
        self.declines = []
        self.rebuilds = []
        self.sold = set()
        self.defects = []
        self.last_structure_turn = 0
        self.last_transfer_turn = 0
        self.last_sample = -50
        self.ledger_checks = 0
        self.peak_buildings = 0

    def transfer(self, source, target, amount, cause):
        if not amount:
            return
        assert amount >= 0 and cause != "unknown", (amount, cause)
        if source is not None:
            self.expected[source] -= amount
        if target is not None:
            self.expected[target] += amount
        direction = (
            "bank_income"
            if source is None
            else ("bank_payment" if target is None else "player_transfer")
        )
        self.flows[f"{direction}:{cause}"] += amount
        if source is not None:
            self.player_flows[f"{source}:out:{direction}:{cause}"] += amount
        if target is not None:
            self.player_flows[f"{target}:in:{direction}:{cause}"] += amount
        self.last_transfer_turn = self.game.turn_number
        if self.capture:
            self.transactions.append(
                {
                    "revision": self.game.state.revision + 1,
                    "turn": self.game.turn_number,
                    "source": source,
                    "target": target,
                    "amount": amount,
                    "cause": cause,
                }
            )

    @contextmanager
    def cause(self, label):
        previous, self.context = self.context, label
        try:
            yield
        finally:
            self.context = previous

    def attach(self, game, stack):
        self.game = game
        self.expected = [p.money for p in game.players]
        self.snapshots.append(snapshot(game))
        original_apply, original_log = game.apply_action, game.log_event

        def log(message):
            if message.startswith("payment "):
                debt = game.state.obligations[0]
                label = self.obligations.pop(id(debt))
                creditor = debt["creditor"]
                if creditor is not None and game.players[creditor].bankrupt:
                    creditor = None
                self.transfer(debt["debtor"], creditor, debt["amount"], label)
            original_log(message)

        def apply(pid, action):
            before = structure(game)
            valid, reason = action.validate(game)
            if not valid:
                raise ValueError(reason)
            self.before_action(pid, action)
            result = original_apply(pid, action)
            assert self.expected == [p.money for p in game.players], (
                "Unreconciled cash",
                self.expected,
                [p.money for p in game.players],
                result.events,
            )
            self.ledger_checks += 1
            if structure(game) != before or result.eliminations:
                self.last_structure_turn = game.turn_number
            for eliminated in result.eliminations:
                self.eliminations.append(
                    {"player": eliminated, "turn": game.turn_number, "revision": result.revision}
                )
            self.peak_buildings = max(
                self.peak_buildings,
                sum(p.houses for p in game.property_manager.properties.values()),
            )
            if game.turn_number - self.last_sample >= 50:
                self.sample()
            return result

        stack.enter_context(patch.object(game, "log_event", log))
        stack.enter_context(patch.object(game, "apply_action", apply))
        # Movement's returned flag is the engine's actual GO-payment decision.
        for method in ("move_player", "move_player_to"):
            original = getattr(game, method)

            def movement(pid, value, original=original, method=method):
                result = original(pid, value)
                passed = result[1] if method == "move_player" else result
                if passed:
                    self.transfer(None, pid, GO_SALARY, "go")
                return result

            stack.enter_context(patch.object(game, method, movement))
        return game

    def before_action(self, pid, action):
        game = self.game
        name, pos = type(action).__name__, getattr(action, "property_id", None)
        self.actions[f"{pid}:{name}"] += 1
        if name == "BuyProperty":
            self.transfer(pid, None, get_property_cost(pos), "purchase")
        elif name in ("BuildHouse", "BuildHotel"):
            self.transfer(pid, None, get_building_cost(pos), "build")
            if pos in self.sold:
                self.rebuilds.append({"turn": game.turn_number, "player": pid, "position": pos})
                self.sold.remove(pos)
            siblings = foundation.group_properties(game, pos)
            if any(p.mortgaged for p in siblings):
                self.defects.append(
                    {
                        "kind": "build_with_mortgaged_sibling",
                        "revision": game.state.revision + 1,
                        "position": pos,
                    }
                )
        elif name in ("SellHouse", "SellHotel", "SellBuildingGroup"):
            props = (
                foundation.group_properties(game, pos)
                if name == "SellBuildingGroup"
                else [game.property_manager.properties[pos]]
            )
            for prop in props:
                units = prop.houses if name == "SellBuildingGroup" else 1
                self.transfer(None, pid, units * get_house_sale_value(prop.position), "sale")
                if units:
                    self.sold.add(prop.position)
        elif name == "MortgageProperty":
            self.transfer(None, pid, get_mortgage_value(pos), "mortgage")
            self.mortgaged.add(pos)
        elif name == "UnmortgageProperty":
            self.transfer(pid, None, get_unmortgage_cost(pos), "unmortgage")
            if pos in self.mortgaged:
                self.unmortgage_cycles.append(
                    {"turn": game.turn_number, "player": pid, "position": pos}
                )
                self.mortgaged.remove(pos)
        elif name == "PayJailFine":
            self.transfer(pid, None, JAIL_FINE, "jail_fine")
        elif name == "EndTurn":
            builds = [
                a.property_id
                for a in foundation.legal_actions(game, pid)
                if type(a).__name__ == "BuildHouse"
            ]
            if builds:
                self.declines.append(
                    {
                        "turn": game.turn_number,
                        "player": pid,
                        "cash": game.players[pid].money,
                        "positions": builds,
                    }
                )

    def sample(self):
        s = snapshot(self.game)
        s["flows"] = dict(self.flows)
        s["player_flows"] = dict(self.player_flows)
        s["turns_since_structure_change"] = self.game.turn_number - self.last_structure_turn
        self.snapshots.append(s)
        self.last_sample = self.game.turn_number

    def finish(self):
        self.sample()
        final = self.snapshots[-1]
        # Descriptive, overlapping flags, not assertions of infinite stalemate or causality.
        labels = []
        if final["complete_groups"] == 0 and final["fragmented_groups"]:
            labels.append("no_complete_color_group")
        if final["complete_groups"] and final["building_units"] == 0:
            labels.append("complete_group_but_no_buildings_at_stop")
        if final["building_units"]:
            labels.append("developed_at_stop")
        if not final["unowned"]:
            labels.append("all_properties_owned")
        if final["turns_since_structure_change"] >= 500:
            labels.append("structure_static_500_turns")
        if self.declines:
            labels.append("legal_build_declined_at_turn_end")
        if self.rebuilds:
            labels.append("sold_then_rebuilt")
        if self.defects:
            labels.append("observed_build_validation_defect")
        return {
            "version": VERSION,
            "labels": labels,
            "final": final,
            "peak_building_units": self.peak_buildings,
            "actions": dict(self.actions),
            "flows": dict(self.flows),
            "player_flows": dict(self.player_flows),
            "mortgage_unmortgage_cycles": self.unmortgage_cycles,
            "snapshots": self.snapshots,
            "eliminations": self.eliminations,
            "declined_builds": self.declines,
            "rebuilds": self.rebuilds,
            "defects": self.defects,
            "ledger_checks": self.ledger_checks,
            "turns_since_cash_transfer": self.game.turn_number - self.last_transfer_turn,
            **({"transactions": self.transactions} if self.capture else {}),
        }

    @contextmanager
    def installed(self):
        """Intercept only within one synchronous diagnostic run; always restore hooks."""
        with ExitStack() as stack:
            factory = runner.MonopolyGame
            stack.enter_context(
                patch.object(
                    runner, "MonopolyGame", lambda *a, **kw: self.attach(factory(*a, **kw), stack)
                )
            )
            original_charge, original_bankrupt = foundation.charge, foundation.bankrupt
            original_card = foundation.execute_card

            def charge(game, pid, amount, creditor, continuation=None):
                original_charge(game, pid, amount, creditor, continuation)
                if amount:
                    self.obligations[id(game.state.obligations[-1])] = self.context

            def bankrupt(game, pid, creditor=None):
                if not game.players[pid].bankrupt:
                    sale = sum(
                        p.houses * get_house_sale_value(p.position)
                        for p in game.property_manager.properties.values()
                        if p.owner == pid
                    )
                    # The engine credits salvage directly to a player creditor. A bank
                    # bankruptcy destroys buildings without any cash credit: do not
                    # invent a virtual sale/payment pair that inflates bank turnover.
                    if creditor is not None:
                        self.transfer(None, creditor, sale, "bankruptcy_building_sale")
                    self.transfer(pid, creditor, game.players[pid].money, "bankruptcy")
                    if game.state.obligations:
                        self.obligations.pop(id(game.state.obligations[0]), None)
                return original_bankrupt(game, pid, creditor)

            def card(game, pid, card):
                label = "card:" + card.card_type.name.lower()
                with self.cause(label):
                    if card.card_type == CardType.COLLECT:
                        self.transfer(None, pid, card.amount or 0, label)
                    return original_card(game, pid, card)

            stack.enter_context(patch.object(foundation, "charge", charge))
            stack.enter_context(patch.object(foundation, "bankrupt", bankrupt))
            stack.enter_context(patch.object(foundation, "execute_card", card))
            for name, label in (
                ("property_landing", "rent"),
                ("roll", "jail_fine"),
                ("land", "landing"),
            ):
                original = getattr(foundation, name)

                @wraps(original)
                def wrapped(game, pid, *args, original=original, label=label, **kwargs):
                    cause = (
                        "tax:" + Board.get_space(game.players[pid].position).space_type.name
                        if label == "landing"
                        else label
                    )
                    with self.cause(cause):
                        return original(game, pid, *args, **kwargs)

                stack.enter_context(patch.object(foundation, name, wrapped))
            yield self


def diagnose_game(**kwargs: Any):
    observer = EconomicObserver(capture=kwargs.get("capture", False))
    with observer.installed():
        record = runner.play_game(**kwargs)
        record["economics"] = observer.finish()
        if "replay" in record:
            record["final_snapshot"] = observer.game.to_dict()
    return record
