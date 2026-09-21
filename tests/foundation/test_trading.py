"""Authoritative foundation-trade-v1 decisions and deterministic policy tests."""

import json

import pytest

from agents import RuleBasedAgent
from agents.trading_agent import TradingAgent, score_offer
from evaluation.native_actions import decode_native_action
from evaluation.runner import invariant, play_game
from monopoly_engine import (
    AcceptTrade,
    BuildHouse,
    EndTurn,
    MonopolyGame,
    ProposeTrade,
    RejectTrade,
)
from monopoly_engine.types import TradeOfferData
from monopoly_gym.action_space import ActionEncoder


def trade_game(players=2):
    game = MonopolyGame(players, seed=17, rules_id="foundation-trade-v1")
    game.state.phase = "asset_management"
    game.state.roll_owed = False
    return game


def test_offer_acceptance_is_authoritative_atomic_and_replayable():
    game = trade_game()
    game.property_manager.properties[1].owner = 0
    game.property_manager.properties[6].owner = 0
    game.property_manager.properties[3].owner = 1
    game.property_manager.properties[3].mortgaged = True
    game.property_manager.properties[8].owner = 1
    before_rng = game.rng.getstate()
    proposal = ProposeTrade(0, 1, [6], 100, [3], 0)
    proposed = game.apply_action(0, proposal)
    assert game.current_player == 0
    assert game.decision_player == 1 and game.state.phase == "trade_response"
    assert game.rng.getstate() == before_rng
    assert proposed.structured_events[0]["type"] == "trade_proposed"
    snapshot = json.loads(json.dumps(game.to_dict()))

    accepted = game.apply_action(1, AcceptTrade(1, 0))
    assert [p.money for p in game.players] == [1400, 1600]
    assert game.property_manager.properties[6].owner == 1
    assert game.property_manager.properties[3].owner == 0
    assert game.property_manager.properties[3].mortgaged
    assert game.decision_player == 0 and game.state.phase == "asset_management"
    assert not game.state.pending_trades
    assert accepted.structured_events[0]["type"] == "trade_accepted"
    assert game.rng.getstate() == before_rng
    invariant(game)

    restored = MonopolyGame.from_dict(snapshot)
    replayed = restored.apply_action(1, AcceptTrade(1, 0))
    assert accepted.events == replayed.events
    assert accepted.structured_events == replayed.structured_events
    assert game.to_dict() == restored.to_dict()


def test_rejection_resumes_turn_and_offer_budget_is_per_unique_opponent():
    game = trade_game(4)
    for position, owner in ((1, 0), (3, 1), (6, 2), (8, 3)):
        game.property_manager.properties[position].owner = owner
    game.apply_action(0, ProposeTrade(0, 1, [1], 0, [3], 0))
    game.apply_action(1, RejectTrade(1, 0))
    assert game.decision_player == 0 and game.state.phase == "asset_management"
    assert not ProposeTrade(0, 1, [1], 0, [3], 0).validate(game)[0]
    game.apply_action(0, ProposeTrade(0, 2, [1], 0, [6], 0))
    game.apply_action(2, RejectTrade(2, 1))
    assert not ProposeTrade(0, 3, [1], 0, [8], 0).validate(game)[0]
    game.apply_action(0, EndTurn(0))
    assert game.state.trade_targets_this_turn == []


@pytest.mark.parametrize(
    "action",
    [
        ProposeTrade(0, 1, [], 0, [], 0),
        ProposeTrade(0, 0, [1], 0, [], 0),
        ProposeTrade(0, 1, [1, 1], 0, [], 0),
        ProposeTrade(0, 1, [1], 1, [3], 1),
        ProposeTrade(0, 1, [1, 6, 8], 0, [3], 0),
    ],
)
def test_invalid_offers_leave_complete_state_unchanged(action):
    game = trade_game()
    for position, owner in ((1, 0), (6, 0), (8, 0), (3, 1)):
        game.property_manager.properties[position].owner = owner
    before = game.to_dict()
    with pytest.raises(ValueError):
        game.apply_action(0, action)
    assert game.to_dict() == before


def test_developed_group_cannot_be_traded_and_stale_acceptance_does_not_mutate():
    game = trade_game()
    for position in (1, 3):
        game.property_manager.properties[position].owner = 0
    game.property_manager.properties[1].houses = 1
    game.houses_remaining -= 1
    game.property_manager.properties[6].owner = 0
    game.property_manager.properties[8].owner = 1
    assert not ProposeTrade(0, 1, [1], 0, [8], 0).validate(game)[0]
    game.apply_action(0, ProposeTrade(0, 1, [6], 0, [8], 100))
    game.players[1].money = 50
    before = game.to_dict()
    with pytest.raises(ValueError):
        game.apply_action(1, AcceptTrade(1, 0))
    assert game.to_dict() == before
    assert RejectTrade(1, 0).validate(game)[0]


def test_default_rules_and_environment_encoder_keep_trading_disabled():
    game = MonopolyGame(2, seed=17)
    game.state.phase = "asset_management"
    game.state.roll_owed = False
    game.property_manager.properties[1].owner = 0
    game.property_manager.properties[3].owner = 1
    action = ProposeTrade(0, 1, [1], 0, [3], 0)
    assert not action.validate(game)[0]
    with pytest.raises(ValueError):
        ActionEncoder().encode(action)
    with pytest.raises(ValueError):
        game.apply_action(0, action)


def test_mutual_policy_finds_one_and_two_property_bundles_without_mutation():
    game = trade_game()
    for position, owner in ((1, 0), (6, 0), (8, 0), (3, 1), (9, 1)):
        game.property_manager.properties[position].owner = owner
    agent = TradingAgent(RuleBasedAgent(0))
    before = game.to_dict()
    cached = agent.candidates(game)
    reference = agent.candidates(game, use_cache=False)
    assert cached == reference
    assert game.to_dict() == before
    assert any(
        len(c.action.give_properties) == 2 and c.action.want_properties == [3] for c in cached
    )
    assert any(
        len(c.action.give_properties) == len(c.action.want_properties) == 1 for c in cached
    )
    assert any(c.action.give_money or c.action.want_money for c in cached)
    assert all(
        score_offer(
            game,
            TradeOfferData(
                from_player=0,
                to_player=c.action.to_player,
                give_properties=c.action.give_properties,
                give_money=c.action.give_money,
                want_properties=c.action.want_properties,
                want_money=c.action.want_money,
            ),
        )
        is not None
        for c in cached
    )


def test_responder_rejects_offer_that_breaks_reserve_or_existing_group():
    game = trade_game()
    game.players[0].money = 210
    for position, owner in ((1, 0), (6, 0), (3, 1), (8, 1), (9, 1)):
        game.property_manager.properties[position].owner = owner
    reserve_breaking = TradeOfferData(
        from_player=0,
        to_player=1,
        give_properties=[6],
        give_money=25,
        want_properties=[3],
        want_money=0,
    )
    assert score_offer(game, reserve_breaking) is None
    game.property_manager.properties[3].owner = 0
    group_breaking = TradeOfferData(
        from_player=0,
        to_player=1,
        give_properties=[1],
        give_money=0,
        want_properties=[8],
        want_money=0,
    )
    assert score_offer(game, group_breaking) is None


def test_trade_enables_development_and_changes_survival():
    game = trade_game(3)
    game.players[0].money = 1000
    game.players[1].money = 1
    for position, owner in ((1, 0), (6, 0), (3, 2), (8, 2), (9, 2)):
        game.property_manager.properties[position].owner = owner
    game.apply_action(0, ProposeTrade(0, 2, [6], 0, [3], 0))
    game.apply_action(2, AcceptTrade(2, 0))
    game.apply_action(0, BuildHouse(0, 1))
    game.apply_action(0, BuildHouse(0, 3))
    game.players[1].position = 1
    from monopoly_engine import foundation

    foundation.property_landing(game, 1)
    foundation.settle(game)
    assert game.players[1].bankrupt


@pytest.mark.parametrize("players", [2, 4])
def test_rejection_control_preserves_ordinary_gameplay(players):
    options = dict(policies=["rule_based"] * players, seed=13000000, max_turns=1000)
    baseline = play_game(**options)
    rejection = play_game(
        **options, rules_id="foundation-trade-v1", trading="reject", capture=True
    )
    for field in (
        "status",
        "winner",
        "turns",
        "diagnostic_cash",
        "elimination_order",
        "gameplay_trace_sha256",
    ):
        assert baseline[field] == rejection[field]
    replay = rejection["replay"]
    restored = MonopolyGame.from_dict(replay["initial"])
    encoder = ActionEncoder()
    for item in replay["trace"]:
        action = (
            decode_native_action(item["native_action"])
            if "native_action" in item
            else encoder.decode(item["action"], item["actor"], restored)
        )
        result = restored.apply_action(item["actor"], action)
        assert result.events == tuple(item["events"])
        assert result.structured_events == tuple(item.get("structured_events", ()))
    assert restored.winner == rejection["winner"]
    assert restored.state.turn_number == rejection["turns"]
