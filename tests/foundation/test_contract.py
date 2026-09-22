import json

import numpy as np
import pytest

from monopoly_engine import EndTurn, MonopolyGame, MortgageProperty, RollDice
from monopoly_engine.actions import PassBuy, SellBuildingGroup
from monopoly_engine.foundation import charge, settle
from monopoly_gym import MonopolyEnv, ObservationEncoder, SingleAgentMonopolyEnv
from monopoly_gym.action_space import OFFSET_ROLL_DICE, ActionEncoder
from monopoly_gym.observation import IncrementalObservationEncoder, flatten_observation


def test_invalid_decision_preserves_rng_and_state():
    game = MonopolyGame(2, seed=4)
    before = game.to_dict()
    with pytest.raises(ValueError):
        game.apply_action(0, EndTurn(0))
    assert before == game.to_dict()


def test_pass_purchase_preserves_doubles_roll():
    game = MonopolyGame(2, seed=4)
    game.roll_dice = lambda: (3, 3)
    game.apply_action(0, RollDice(0))
    assert game.state.phase == "purchase_decision"
    game.apply_action(0, PassBuy(0))
    assert game.current_player == 0
    assert game.state.roll_owed
    assert not EndTurn(0).validate(game)[0]


def test_three_doubles_and_no_extra_jail_roll():
    game = MonopolyGame(2, seed=4)
    game.roll_dice = lambda: (3, 3)
    for _ in range(3):
        game.apply_action(0, RollDice(0))
        if game.state.phase == "purchase_decision":
            game.apply_action(0, PassBuy(0))
    assert game.players[0].in_jail and not game.state.roll_owed
    assert EndTurn(0).validate(game)[0]


def test_out_of_turn_debt_rescue():
    game = MonopolyGame(3, seed=1)
    game.state.phase = "asset_management"
    game.players[1].money = 0
    game.property_manager.properties[39].owner = 1
    charge(game, 1, 100, 0)
    settle(game)
    assert game.decision_player == 1 and game.current_player == 0
    game.apply_action(1, MortgageProperty(1, 39))
    assert game.players[1].money == 100
    assert game.players[0].money == 1600
    assert game.decision_player == 0


def test_insolvent_bankruptcy_preserves_mortgage_and_buildings():
    game = MonopolyGame(3, seed=1)
    game.players[1].money = 5
    p = game.property_manager.properties[1]
    p.owner, p.mortgaged = 1, True
    charge(game, 1, 100, 0)
    settle(game)
    assert game.players[1].bankrupt
    assert p.owner == 0 and p.mortgaged
    assert game.players[0].money == 1505
    assert game.state.elimination_order == [1]


def test_hotel_group_liquidation_conserves_inventory():
    game = MonopolyGame(2, seed=2)
    for pos in (1, 3):
        p = game.property_manager.properties[pos]
        p.owner, p.houses = 0, 5
    game.hotels_remaining = 10
    old = game.houses_remaining
    game.apply_action(0, SellBuildingGroup(0, 1))
    assert game.houses_remaining == old and game.hotels_remaining == 12
    assert game.players[0].money == 1750


def test_jail_third_failure_resumes_move_after_debt():
    game = MonopolyGame(2, seed=2)
    p = game.players[0]
    p.go_to_jail()
    p.jail_turns, p.money = 2, 0
    game.state.phase = "jail_decision"
    game.property_manager.properties[39].owner = 0
    game.roll_dice = lambda: (1, 2)
    game.apply_action(0, RollDice(0))
    assert game.state.phase == "debt_resolution" and p.position == 10
    game.apply_action(0, MortgageProperty(0, 39))
    assert not p.in_jail and p.position == 13 and p.money == 150
    assert not game.state.roll_owed


def test_snapshot_continuation_preserves_rng_and_decks():
    game = MonopolyGame(2, seed=15)
    restored = MonopolyGame.from_dict(json.loads(json.dumps(game.to_dict())))
    enc = ActionEncoder()
    for _ in range(60):
        mask = enc.get_action_mask(game, game.decision_player)
        idx = OFFSET_ROLL_DICE if mask[OFFSET_ROLL_DICE] else int(np.flatnonzero(mask)[0])
        for g in (game, restored):
            g.apply_action(g.decision_player, enc.decode(idx, g.decision_player, g))
        assert game.to_dict() == restored.to_dict()


def test_observation_fresh_and_immutable():
    game = MonopolyGame(2, seed=1)
    encoder = IncrementalObservationEncoder(2)
    old = flatten_observation(encoder.encode(game, 0)).copy()
    game.players[1].money -= 200
    game.turn_number += 1
    encoder.encode(game, 1)
    new = flatten_observation(encoder.encode(game, 0))
    fresh = flatten_observation(ObservationEncoder(2).encode(game, 0))
    assert not np.array_equal(old, new)
    np.testing.assert_array_equal(new, fresh)


def test_terminal_reward_delivered_once_and_dead_agents_drain():
    env = MonopolyEnv(2)
    env.reset(seed=1)
    game = env.game
    game.players[0].money = 0
    game.players[0].position = 1
    game.property_manager.properties[3].owner = 1
    game.roll_dice = lambda: (1, 1)
    env.step(OFFSET_ROLL_DICE)
    assert env.rewards == {"player_0": -1, "player_1": 1}
    observed = {}
    while env.agents:
        agent = env.agent_selection
        _, reward, terminated, truncated, _ = env.last()
        observed[agent] = reward
        assert terminated and not truncated
        env.step(None)
    assert observed == {"player_0": -1, "player_1": 1}


def test_wrapper_cutoff_is_not_win_and_has_observation():
    from agents import RuleBasedAgent

    env = SingleAgentMonopolyEnv(2, "rule_based", max_turns=2)
    obs, info = env.reset(seed=2)
    policy = RuleBasedAgent(0)
    for _ in range(100):
        action = policy.choose_action(None, env.action_masks(), env._env.game)
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            break
    assert trunc and not term and reward == 0 and info["winner"] is None
    assert np.any(obs) and env.observation_space.contains(obs)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_seed_stream_reproducible_but_advances():
    a, b = SingleAgentMonopolyEnv(2), SingleAgentMonopolyEnv(2)
    first = a.reset(seed=123)[0]
    np.testing.assert_array_equal(first, b.reset(seed=123)[0])
    second = a.reset()[0]
    np.testing.assert_array_equal(second, b.reset()[0])
    assert a._env.episode_seed != SingleAgentMonopolyEnv(2, seed=123)._seed
    assert a._env.episode_seed == b._env.episode_seed


def test_selfplay_uses_identical_contract():
    from agents import RuleBasedAgent
    from training.pettingzoo_selfplay import SelfPlayEnv

    a = SingleAgentMonopolyEnv(2, "rule_based", max_turns=5)
    b = SelfPlayEnv(2, opponent_type="rule_based", max_turns=5)
    np.testing.assert_array_equal(a.reset(seed=6)[0], b.reset(seed=6)[0])
    p = RuleBasedAgent(0)
    for _ in range(100):
        action = p.choose_action(None, a.action_masks(), a._env.game)
        x, y = a.step(action), b.step(action)
        np.testing.assert_array_equal(x[0], y[0])
        assert x[1:4] == y[1:4]
        if x[2] or x[3]:
            break


def test_learner_wrappers_accept_the_frozen_engine_seed():
    from training.pettingzoo_selfplay import SelfPlayEnv

    for wrapper in (SingleAgentMonopolyEnv, SelfPlayEnv):
        env = wrapper(
            2,
            opponent_type="trading_rule_based",
            rules_id="foundation-trade-v1",
        )
        env.reset(options={"engine_seed": 19_000_007})
        assert env._env.episode_seed == 19_000_007
        assert env._env.game.to_dict() == MonopolyGame(
            2, seed=19_000_007, rules_id="foundation-trade-v1"
        ).to_dict()
        assert env.action_space.n == 192


def test_masked_gym_and_aec_api_contracts():
    from gymnasium.utils.env_checker import check_env
    from pettingzoo.test import api_test

    api_test(MonopolyEnv(2, max_turns=10), num_cycles=50)
    env = SingleAgentMonopolyEnv(2, max_turns=10)
    # Gym's checker samples unmasked actions; supply its legal sampling driver.
    env.action_space.sample = lambda: OFFSET_ROLL_DICE
    check_env(env, skip_render_check=True)


def test_sb3_timeout_observation_and_smoke_save_load(tmp_path):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from agents import RuleBasedAgent

    base = SingleAgentMonopolyEnv(2, "rule_based", max_turns=2)
    vec = DummyVecEnv([lambda: base])
    vec.reset()
    policy = RuleBasedAgent(0)
    for _ in range(50):
        action = policy.choose_action(None, base.action_masks(), base._env.game)
        obs, reward, done, infos = vec.step([action])
        if done[0]:
            assert infos[0]["TimeLimit.truncated"]
            assert np.any(infos[0]["terminal_observation"])
            assert reward[0] == 0
            break
    else:
        pytest.fail("Expected bounded cutoff")
    model = MaskablePPO(
        "MlpPolicy",
        vec,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy_kwargs={"net_arch": [16]},
        seed=12,
        device="cpu",
    )
    model.learn(16)
    path = tmp_path / "smoke"
    model.save(path)
    loaded = MaskablePPO.load(path, env=vec)
    assert loaded.action_space.n == 158
    vec.close()
