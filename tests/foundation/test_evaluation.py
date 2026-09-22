import json

import numpy as np
import pytest

from evaluation.runner import play_game, summarize
from monopoly_engine import MonopolyGame
from monopoly_gym.action_space import ActionEncoder


def test_repeated_seed_outcomes_and_trace_equal():
    a = play_game(["rule_based", "random"], 998, max_turns=100)
    b = play_game(["rule_based", "random"], 998, max_turns=100)
    for field in ("runtime_seconds", "inference_seconds"):
        a.pop(field)
        b.pop(field)
    assert a == b
    assert a["status"] not in ("error", "stalled")


def test_cutoffs_never_wins():
    result = play_game(["rule_based"] * 4, 10, max_turns=1)
    assert result["status"] == "cutoff"
    assert not result["win"] and not result["loss"] and result["winner"] is None
    report = summarize([result])
    assert report["win_rate"] == 0
    assert not report["ready"]


def test_bad_policy_is_error_not_fallback_win():
    class BadPolicy:
        def reset(self):
            pass

        def choose_action(self, *args):
            return 9999

    result = play_game(["bad", "bad"], 13, policy_instances=[BadPolicy(), BadPolicy()])
    assert result["status"] == "error" and not result["win"]
    assert "replay" in result


def test_saved_replay_reproduces_events_and_revision():
    record = play_game(["rule_based", "random"], 21, max_turns=50, capture=True)
    game = MonopolyGame.from_dict(json.loads(json.dumps(record["replay"]["initial"])))
    encoder = ActionEncoder()
    for item in record["replay"]["trace"]:
        result = game.apply_action(
            item["actor"], encoder.decode(item["action"], item["actor"], game)
        )
        assert result.events == tuple(item["events"])
        assert result.revision == item["revision"]
    assert game.winner == record["winner"]


def test_masks_execute_every_advertised_action_on_clone():
    from agents import RuleBasedAgent

    for players in (2, 4):
        game = MonopolyGame(players, seed=77)
        agents = [RuleBasedAgent(i) for i in range(players)]
        encoder = ActionEncoder()
        for _ in range(200):
            pid = game.decision_player
            mask = encoder.get_action_mask(game, pid)
            for index in np.flatnonzero(mask):
                clone = MonopolyGame.from_dict(game.to_dict())
                clone.apply_action(pid, encoder.decode(int(index), pid, clone))
            action = agents[pid].choose_action(None, mask, game)
            game.apply_action(pid, encoder.decode(action, pid, game))
            if game.game_over:
                break


def test_learned_search_rejected():
    from mcts.search import MCTSConfig, MCTSSearch

    with pytest.raises(NotImplementedError, match="disconnected"):
        MCTSSearch(MCTSConfig(use_value_network=True), value_network=object())


def test_api_and_direct_engine_transition_parity():
    from api.models.action import ActionRequest
    from api.models.game import GameState
    from monopoly_engine import RollDice

    a = MonopolyGame(2, seed=91)
    b = MonopolyGame.from_dict(a.to_dict())
    action = ActionRequest(action_type="roll_dice").to_engine_action(0)
    a.apply_action(0, action)
    b.apply_action(0, RollDice(0))
    assert a.to_dict() == b.to_dict()
    public = GameState.from_engine(a).model_dump()
    assert public["rules_id"] == "foundation-v1"
    assert public["decision_player"] == a.decision_player
    assert public["legal_actions"]
    assert "rng_states" not in public and "chance_deck" not in public


@pytest.mark.asyncio
async def test_api_aec_and_engine_scripted_trace_parity():
    import re

    from agents import RuleBasedAgent
    from api.models.action import ActionRequest
    from api.services.game_manager import GameManager
    from monopoly_gym import MonopolyEnv

    manager = GameManager()
    game_id = await manager.create_game(num_players=2, seed=33)
    active = await manager.get_game(game_id)
    direct = MonopolyGame.from_dict(active.game.to_dict())
    env = MonopolyEnv(2)
    env.reset(seed=33)
    env.game = MonopolyGame.from_dict(direct.to_dict())
    env._update_infos()
    encoder = ActionEncoder()
    for _ in range(200):
        pid = direct.decision_player
        index = RuleBasedAgent(pid).choose_action(
            None, encoder.get_action_mask(direct, pid), direct
        )
        action = encoder.decode(index, pid, direct)
        payload = {"action_type": re.sub(r"(?<!^)(?=[A-Z])", "_", type(action).__name__).lower()}
        if hasattr(action, "property_id"):
            payload["property_position"] = action.property_id
        success, reason = await manager.execute_action(
            game_id, ActionRequest(**payload).to_engine_action(pid)
        )
        assert success, reason
        direct.apply_action(pid, action)
        assert env.agent_selection == f"player_{pid}"
        env.step(index)
        assert direct.to_dict() == env.game.to_dict() == active.game.to_dict()
        if direct.game_over:
            break
        while env.terminations[env.agent_selection]:
            env.step(None)
