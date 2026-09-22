"""Regressions for learner trade boundaries and the consumer certification gate."""

import json
from types import SimpleNamespace

import pytest

from api.routers.websocket import _handle_action
from api.services.game_manager import GameManager
from monopoly_engine import MonopolyGame, ProposeTrade
from monopoly_engine.progress import ProgressGuard
from monopoly_gym import SingleAgentMonopolyEnv
from monopoly_gym.action_space import OFFSET_ACCEPT_TRADE_V3, OFFSET_REJECT_TRADE_V3
from scripts.certify_trading_consumers import CONSUMERS, run_job
from scripts.certify_trading_integration import run_game
from scripts.verify_trading_integration import verify_consumers, verify_learner_coverage
from training.pettingzoo_selfplay import SelfPlayEnv


@pytest.mark.parametrize("wrapper", [SingleAgentMonopolyEnv, SelfPlayEnv])
@pytest.mark.parametrize("players,seat", [(2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3)])
@pytest.mark.parametrize("response", [OFFSET_ACCEPT_TRADE_V3, OFFSET_REJECT_TRADE_V3])
def test_out_of_turn_response_resolves_before_learner_horizon(wrapper, players, seat, response):
    env = wrapper(
        players,
        opponent_type="trading_rule_based",
        learner_seat=seat,
        max_turns=1,
        rules_id="foundation-trade-v1",
    )
    env.reset(options={"engine_seed": 101, "policy_seeds": list(range(players))})
    game = MonopolyGame(players, seed=101, rules_id="foundation-trade-v1")
    proposer = (seat + 1) % players
    game.current_player = proposer
    game.state.decision_player = proposer
    game.state.phase = "asset_management"
    game.state.roll_owed = False
    game.state.turn_number = 1
    game.property_manager.properties[1].owner = proposer
    game.property_manager.properties[3].owner = seat
    game.apply_action(proposer, ProposeTrade(proposer, seat, [1], 0, [3], 0))
    env._env.game = game
    env._env.agent_selection = f"player_{seat}"
    env._env.terminations = dict.fromkeys(env._env.agents, False)
    env._env.truncations = dict.fromkeys(env._env.agents, False)
    env._env.action_encoder.invalidate_cache()
    env._env._update_infos()
    env._done = env._pending_terminal = False
    env._guard = ProgressGuard()
    observation, reward, terminated, truncated, info = env.step(response)
    assert not game.state.pending_trades
    assert game.property_manager.properties[1].owner == (
        seat if response == OFFSET_ACCEPT_TRADE_V3 else proposer
    )
    assert reward == 0 and not terminated and truncated
    assert info["winner"] is None and env.observation_space.contains(observation)
    snapshot = game.to_dict()
    with pytest.raises(RuntimeError, match="Episode finished"):
        env.step(response)
    assert game.to_dict() == snapshot


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_seeded_consumer_matches_reference_including_terminal_reward(consumer):
    # A fresh development seed, with a nonzero learner seat and random opponents.
    record = run_game(
        {
            "policies": ["random", "rule_based"],
            "seed": 18_100_001,
            "focal_seat": 1,
            "rules_id": "foundation-trade-v1",
            "response": "mutual",
            "label": "2p-rule_based-vs-random",
            "arm": "mutual",
            "capture": False,
        }
    )
    assert record["status"] == "completed"
    result = run_job((record, consumer))
    assert result["trace_sha256"] == record["trace_sha256"]
    if consumer != "api_bot":
        assert result["learner_reward"] == (1.0 if record["winner"] == 1 else -1.0)


@pytest.mark.parametrize("policy_seeds", [[1], [1, -2], [True, 2], [1, "2"]])
def test_reject_invalid_explicit_policy_seed_schedule(policy_seeds):
    env = SingleAgentMonopolyEnv(2, rules_id="foundation-trade-v1")
    with pytest.raises(ValueError, match="policy_seeds"):
        env.reset(options={"policy_seeds": policy_seeds})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "player_id,data",
    [
        (None, {"action_type": "roll_dice"}),
        (0, {"action_type": "propose_trade", "give_money": 1.5}),
        (0, {"action_type": "unknown"}),
    ],
)
async def test_early_api_rejections_preserve_request_correlation(player_id, data):
    messages = []

    async def send_text(text):
        messages.append(json.loads(text))

    await _handle_action(
        SimpleNamespace(send_text=send_text),
        SimpleNamespace(player_id=player_id, game_id="unused", session_id="unused"),
        {**data, "request_id": "request-to-clear"},
        GameManager(),
    )
    result = messages[0]["data"]
    assert not result["success"] and result["request_id"] == "request-to-clear"
    assert result["error_code"]


def test_consumer_verifier_requires_full_seed_blocks(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "purpose": "milestone1d-shared-consumer-parity",
                "consumers": CONSUMERS,
                "blocks_per_matchup": 1,
            }
        )
    )
    with pytest.raises(AssertionError):
        verify_consumers(tmp_path, tmp_path)


@pytest.mark.parametrize("missing", [None, "responses", "wins", "early_elimination"])
def test_consumer_verifier_requires_response_and_terminal_lifecycle_coverage(missing):
    records = [
        {
            "consumer": consumer,
            "label": "4p-rule_based-vs-rule_based",
            "learner_responses": 0 if missing == "responses" else 1,
            "learner_reward": -1 if missing == "wins" else reward,
            "learner_terminal_revision": 100 if missing == "early_elimination" else 70,
            "decisions": 100,
        }
        for consumer in ("single_agent", "self_play")
        for reward in (-1, 1)
    ]
    if missing is None:
        verify_learner_coverage(records)
    else:
        with pytest.raises(AssertionError):
            verify_learner_coverage(records)
