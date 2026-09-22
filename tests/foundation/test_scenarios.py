"""Consequential rules scenarios; exact accounting rather than reward-sign proxies."""

import json

import pytest

from evaluation.runner import invariant
from monopoly_engine import BuyProperty, EndTurn, MonopolyGame, MortgageProperty, RollDice
from monopoly_engine.actions import PassBuy, UseJailCard
from monopoly_engine.cards import CardType
from monopoly_engine.foundation import charge, settle
from monopoly_gym import SingleAgentMonopolyEnv


def put_card_on_top(deck, predicate):
    card = next(c for c in deck.cards if predicate(c))
    deck.cards.remove(card)
    deck.cards.append(card)


@pytest.mark.parametrize("kind,position,rent", [("railroad", 15, 50), ("utility", 12, 60)])
def test_special_card_rent_survives_chained_landing(kind, position, rent):
    game = MonopolyGame(2, seed=2)
    game.property_manager.properties[position].owner = 1
    put_card_on_top(
        game.state.chance_deck,
        lambda c: c.card_type == CardType.MOVE_NEAREST and c.move_to_nearest == kind,
    )
    rolls = iter([(3, 4), (2, 4)])
    game.roll_dice = lambda: next(rolls)
    game.apply_action(0, RollDice(0))
    assert game.players[0].position == position
    assert [p.money for p in game.players] == [1500 - rent, 1500 + rent]
    assert game.state.phase == "asset_management"
    invariant(game)


def test_card_payment_interrupts_turn_and_resumes_in_stable_seat_order():
    game = MonopolyGame(4, seed=2)
    put_card_on_top(game.state.chest_deck, lambda c: c.card_type == CardType.COLLECT_FROM_PLAYERS)
    amount = game.state.chest_deck.peek().amount
    game.players[1].money = 0
    game.players[2].money = 0
    game.property_manager.properties[39].owner = 2
    game.roll_dice = lambda: (1, 1)
    game.apply_action(0, RollDice(0))
    assert game.state.elimination_order == [1]
    assert game.decision_player == 2 and game.current_player == 0
    assert [d["debtor"] for d in game.state.obligations] == [2, 3]
    snapshot = json.loads(json.dumps(game.to_dict()))
    restored = MonopolyGame.from_dict(snapshot)
    for g in (game, restored):
        g.apply_action(2, MortgageProperty(2, 39))
    assert game.to_dict() == restored.to_dict()
    assert [p.money for p in game.players] == [1500 + amount * 2, 0, 200 - amount, 1500 - amount]
    assert game.decision_player == 0 and game.state.roll_owed
    invariant(game)


def test_jail_card_provenance_survives_creditor_bankruptcy_transfer():
    game = MonopolyGame(3, seed=2)
    deck = game.state.chance_deck
    put_card_on_top(deck, lambda c: c.card_type == CardType.GET_OUT_OF_JAIL)
    game.roll_dice = lambda: (3, 4)
    game.apply_action(0, RollDice(0))
    assert game.players[0].jail_cards == 1
    game.players[0].money = 0
    charge(game, 0, 1, 1)
    settle(game)
    assert game.players[1].jail_cards == 1 and game.players[0].jail_cards == 0
    game.send_to_jail(1)
    game.state.phase = "jail_decision"
    game.apply_action(1, UseJailCard(1))
    assert not game.players[1].in_jail
    assert game.players[1].jail_cards == 0
    assert deck.total_cards() == 16
    invariant(game)


def test_purchase_choice_changes_actual_survival_outcome():
    # Buying Mediterranean leaves player 0 with $1. Its $2 rent eliminates
    # player 1 on their next landing; declining yields no winner.
    game = MonopolyGame(2, seed=2)
    game.players[0].money = 61
    game.players[1].money = 1
    game.players[0].position = 1
    game.players[1].position = 38
    game.state.phase = "purchase_decision"
    game.state.roll_owed = False
    declined = MonopolyGame.from_dict(game.to_dict())
    game.apply_action(0, BuyProperty(0, 1))
    declined.apply_action(0, PassBuy(0))
    for g in (game, declined):
        g.apply_action(0, EndTurn(0))
        # Avoid GO credit in this controlled landing comparison.
        g.players[1].position = 1
        from monopoly_engine.foundation import property_landing

        property_landing(g, 1)
        settle(g)
    assert game.game_over and game.winner == 0
    assert not declined.game_over and declined.winner is None


def test_elimination_before_first_learner_decision_is_delivered_once():
    env = SingleAgentMonopolyEnv(2, learner_seat=1)
    original_reset = env._env.reset

    def reset_scenario(*args, **kwargs):
        original_reset(*args, **kwargs)
        game = env._env.game
        put_card_on_top(
            game.state.chest_deck, lambda c: c.card_type == CardType.COLLECT_FROM_PLAYERS
        )
        game.players[1].money = 0
        game.roll_dice = lambda: (1, 1)

    env._env.reset = reset_scenario
    _, info = env.reset(seed=5)
    assert info["pending_terminal"] and info["winner"] == 0
    before = env._env.game.to_dict()
    _, reward, term, trunc, _ = env.step(999999)
    assert (reward, term, trunc) == (-1, True, False)
    assert env._env.game.to_dict() == before
    with pytest.raises(RuntimeError):
        env.step(0)
