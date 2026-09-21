"""Accounting/parity diagnostics and retained correctness reproducers for milestone 1b."""

from contextlib import ExitStack

import pytest

from agents import AggressiveAgent, ConservativeAgent, RuleBasedAgent
from evaluation.economics import EconomicObserver, diagnose_game
from evaluation.runner import invariant, play_game
from monopoly_engine import MonopolyGame, MortgageProperty, RollDice, foundation
from monopoly_engine.actions import BuildHouse
from monopoly_gym.action_space import ActionEncoder


@pytest.mark.parametrize("players", [2, 4])
@pytest.mark.parametrize("opponent", ["rule_based", "random"])
def test_observer_preserves_exact_trace_and_every_cash_balance(players, opponent):
    options = dict(
        policies=["rule_based"] + [opponent] * (players - 1),
        seed=12000000,
        max_turns=1000,
        capture=True,
    )
    measured = diagnose_game(**options)
    reference = play_game(**options)
    assert measured["status"] in ("completed", "cutoff")
    for field in ("trace_sha256", "winner", "diagnostic_cash", "replay", "decisions", "turns"):
        assert measured[field] == reference[field]
    assert measured["economics"]["ledger_checks"] == measured["decisions"]
    for i in range(players):
        cash = 1500
        for transaction in measured["economics"]["transactions"]:
            if transaction["source"] == i:
                cash -= transaction["amount"]
            if transaction["target"] == i:
                cash += transaction["amount"]
        assert cash == measured["diagnostic_cash"][i]


def test_delayed_rent_keeps_cause_through_liquidation():
    game = MonopolyGame(2, seed=12)
    game.players[0].money = 0
    game.players[0].position = 39
    game.property_manager.properties[39].owner = 1
    game.property_manager.properties[37].owner = 0
    observer = EconomicObserver(capture=True)
    with observer.installed(), ExitStack() as stack:
        observer.attach(game, stack)
        foundation.property_landing(game, 0)
        foundation.settle(game)
        assert game.state.phase == "debt_resolution"
        game.apply_action(0, MortgageProperty(0, 37))
        assert observer.flows["player_transfer:rent"] == 50
        assert observer.flows["bank_income:mortgage"] == 175
        assert [p.money for p in game.players] == [125, 1550]
        assert observer.ledger_checks == 1
        invariant(game)


def test_bankruptcy_ledger_separates_salvage_and_creditor_payment():
    game = MonopolyGame(2, seed=12)
    game.players[0].money = 1
    for pos in (1, 3):
        game.property_manager.properties[pos].owner = 0
        game.property_manager.properties[pos].houses = 1
    game.houses_remaining = 30
    game.property_manager.properties[39].owner = 1
    game.property_manager.properties[39].houses = 5
    game.hotels_remaining = 11
    game.players[0].position = 36
    game.roll_dice = lambda: (1, 2)
    observer = EconomicObserver(capture=True)
    with observer.installed(), ExitStack() as stack:
        observer.attach(game, stack)
        game.apply_action(0, RollDice(0))
        assert game.winner == 1
        assert observer.flows["bank_income:bankruptcy_building_sale"] == 50
        assert observer.flows["player_transfer:bankruptcy"] == 1
        assert observer.flows["player_transfer:rent"] == 0  # $2,000 demand was never paid.
        invariant(game)


@pytest.mark.parametrize("agent_class", [RuleBasedAgent, AggressiveAgent, ConservativeAgent])
def test_rich_monopoly_owner_develops_to_hotels(agent_class):
    game = MonopolyGame(2, seed=12)
    game.players[0].money = 10000
    for pos in (16, 18, 19):
        game.property_manager.properties[pos].owner = 0
    game.state.phase, game.state.roll_owed = "asset_management", False
    declined = MonopolyGame.from_dict(game.to_dict())
    agent, encoder = agent_class(0), ActionEncoder()
    actions = []
    while game.decision_player == 0:
        mask = encoder.get_action_mask(game, 0)
        action = encoder.decode(agent.choose_action(None, mask, game), 0, game)
        actions.append(type(action).__name__)
        game.apply_action(0, action)
        invariant(game)
        assert len(actions) <= 16
    assert actions == ["BuildHouse"] * 15 + ["EndTurn"]
    assert game.players[0].money == 8500
    assert game.houses_remaining == 32 and game.hotels_remaining == 9
    from monopoly_engine import EndTurn

    declined.apply_action(0, EndTurn(0))
    for candidate in (game, declined):
        candidate.players[1].position = 13
        candidate.players[1].money = 500
        candidate.roll_dice = lambda: (1, 2)
        candidate.apply_action(1, RollDice(1))
    assert game.winner == 0 and game.game_over
    assert declined.winner is None and declined.players[1].money == 472


def test_build_threshold_decline_is_observed_at_end_turn():
    game = MonopolyGame(2, seed=12)
    for pos in (1, 3):
        game.property_manager.properties[pos].owner = 0
    game.players[0].money = 200
    game.state.phase, game.state.roll_owed = "asset_management", False
    observer = EconomicObserver()
    with observer.installed(), ExitStack() as stack:
        observer.attach(game, stack)
        encoder = ActionEncoder()
        mask = encoder.get_action_mask(game, 0)
        action = encoder.decode(RuleBasedAgent(0).choose_action(None, mask, game), 0, game)
        assert type(action).__name__ == "EndTurn"
        game.apply_action(0, action)
        assert observer.declines == [{"turn": 0, "player": 0, "cash": 200, "positions": [1, 3]}]


@pytest.mark.xfail(
    strict=True, reason="1b confirmed defect: mortgage on sibling must block building"
)
def test_mortgaged_sibling_must_block_engine_and_mask_build():
    game = MonopolyGame(2, seed=12)
    for pos in (1, 3):
        game.property_manager.properties[pos].owner = 0
    game.property_manager.properties[3].mortgaged = True
    action = BuildHouse(0, 1)
    encoder = ActionEncoder()
    assert not action.validate(game)[0]
    assert not encoder.get_action_mask(game, 0)[2]
    before = game.to_dict()
    with pytest.raises(ValueError):
        game.apply_action(0, action)
    assert game.to_dict() == before


def test_bank_bankruptcy_does_not_invent_salvage_cash_flows():
    game = MonopolyGame(2, seed=12)
    game.players[0].money = 1
    for pos in (1, 3):
        game.property_manager.properties[pos].owner = 0
        game.property_manager.properties[pos].houses = 1
    game.houses_remaining = 30
    game.players[0].position = 1
    game.roll_dice = lambda: (1, 2)  # $200 income tax exceeds all liquidatable assets.
    observer = EconomicObserver(capture=True)
    with observer.installed(), ExitStack() as stack:
        observer.attach(game, stack)
        game.apply_action(0, RollDice(0))
        assert game.winner == 1
        assert observer.flows["bank_income:bankruptcy_building_sale"] == 0
        assert observer.flows["bank_payment:bankruptcy"] == 1
        invariant(game)


def test_hooks_restore_after_failure_and_unexplained_cash_is_rejected():
    game = MonopolyGame(2, seed=12)
    original_charge = foundation.charge
    original_apply = game.apply_action

    def unexplained(pid, action):
        result = original_apply(pid, action)
        game.players[pid].add_money(1)
        return result

    game.apply_action = unexplained
    observer = EconomicObserver()
    with pytest.raises(AssertionError, match="Unreconciled cash"):
        with observer.installed(), ExitStack() as stack:
            observer.attach(game, stack)
            game.apply_action(0, RollDice(0))
    assert foundation.charge is original_charge
    assert game.apply_action is unexplained
