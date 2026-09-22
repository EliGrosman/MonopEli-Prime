"""Shared-consumer regressions for decision-contract-v1 and action/observation-v3."""

import json

import numpy as np
import pytest

from agents.base import Agent
from agents.trading_agent import TradingAgent
from api.models.action import ActionRequest
from api.models.game import GameState
from api.services.game_manager import GameManager
from monopoly_engine import (
    AcceptTrade,
    EndTurn,
    MonopolyGame,
    MortgageProperty,
    ProposeTrade,
    RejectTrade,
)
from monopoly_gym import MonopolyEnv
from monopoly_gym.action_space import (
    BUYABLE_POSITIONS,
    OFFSET_ACCEPT_TRADE_V3,
    OFFSET_END_TURN,
    OFFSET_MORTGAGE,
    OFFSET_REJECT_TRADE_V3,
    OFFSET_TRADE_CANDIDATE,
    TRADE_ACTION_SPACE_SIZE,
    ActionEncoder,
)
from monopoly_gym.observation import (
    ObservationEncoder,
    flatten_observation,
    get_flat_observation_size,
)
from training.compatibility import (
    learner_contract,
    validate_checkpoint_metadata,
    write_checkpoint_metadata,
)


def prepared_game() -> MonopolyGame:
    game = MonopolyGame(2, seed=41, rules_id="foundation-trade-v1")
    game.state.phase = "asset_management"
    game.state.roll_owed = False
    game.property_manager.properties[1].owner = 0
    game.property_manager.properties[6].owner = 0
    game.property_manager.properties[3].owner = 1
    game.property_manager.properties[8].owner = 1
    game.property_manager.properties[9].owner = 1
    return game


def test_public_contract_is_detached_complete_and_private_free():
    game = prepared_game()
    view = game.decision_view(0)
    payload = view.to_dict()
    assert payload["contract_version"] == "decision-contract-v1"
    assert payload["viewer_id"] == 0
    assert payload["turn_owner"] == 0 == payload["decision_player"]
    assert payload["trade"]["can_propose"]
    assert payload["trade"]["eligible_recipients"] == (1,)
    encoded = json.dumps(payload)
    assert "rng" not in encoded and "deck" not in encoded and "seed" not in encoded
    payload["players"][0]["money"] = 0
    assert game.players[0].money == 1500


def test_revision_and_response_phase_rejections_are_nonmutating():
    game = prepared_game()
    before = game.to_dict()
    with pytest.raises(ValueError, match="Stale decision revision"):
        game.apply_action(0, ProposeTrade(0, 1, [1], 0, [3], 0), expected_revision=99)
    assert game.to_dict() == before
    game.apply_action(0, ProposeTrade(0, 1, [1], 0, [3], 0), expected_revision=0)
    pending = game.to_dict()
    assert not MortgageProperty(1, 3).validate(game)[0]
    with pytest.raises(ValueError):
        game.apply_action(1, MortgageProperty(1, 3))
    assert game.to_dict() == pending


def test_v3_mapping_and_observation_terms_are_stable():
    env = MonopolyEnv(2, rules_id="foundation-trade-v1")
    env.reset(options={"engine_seed": 3})
    env.game = prepared_game()
    env.agent_selection = "player_0"
    env._update_infos()
    obs = env.observe("player_0")
    mask = env.infos["player_0"]["action_mask"]
    candidates = env.action_encoder.get_trade_candidates(env.game, 0)
    assert len(mask) == TRADE_ACTION_SPACE_SIZE
    assert candidates
    assert mask[OFFSET_TRADE_CANDIDATE]
    first = env.action_encoder.decode(OFFSET_TRADE_CANDIDATE, 0, env.game)
    assert first == candidates[0]
    assert np.array_equal(obs["trade_context"]["candidate_mask"], mask[160:192])
    assert flatten_observation(obs).size == get_flat_observation_size(
        2, rules_id="foundation-trade-v1"
    )
    env.step(OFFSET_TRADE_CANDIDATE)
    response = env.observe("player_1")
    response_mask = env.infos["player_1"]["action_mask"]
    assert response_mask[OFFSET_ACCEPT_TRADE_V3]
    assert response_mask[OFFSET_REJECT_TRADE_V3]
    assert response["trade_context"]["pending_offer"][0] == 1


@pytest.mark.asyncio
async def test_api_engine_and_aec_trade_parity():
    manager = GameManager()
    game_id = await manager.create_game(2, seed=41, rules_id="foundation-trade-v1")
    active = await manager.get_game(game_id)
    assert active is not None
    active.game = prepared_game()
    direct = MonopolyGame.from_dict(active.game.to_dict())
    env = MonopolyEnv(2, rules_id="foundation-trade-v1")
    env.reset(options={"engine_seed": 41})
    env.game = MonopolyGame.from_dict(active.game.to_dict())
    env.agent_selection = "player_0"
    env._update_infos()

    candidate = env.action_encoder.get_trade_candidates(env.game, 0)[0]
    request = ActionRequest(
        action_type="propose_trade",
        contract_version="decision-contract-v1",
        expected_revision=0,
        request_id="proposal-1",
        to_player=candidate.to_player,
        give_properties=candidate.give_properties,
        give_money=candidate.give_money,
        want_properties=candidate.want_properties,
        want_money=candidate.want_money,
    )
    action = request.to_engine_action(0)
    success, reason = await manager.execute_action(
        game_id,
        action,
        expected_revision=request.expected_revision,
        request_id=request.request_id,
    )
    assert success, reason
    direct.apply_action(0, action, expected_revision=0)
    slot = env.action_encoder.encode_current(action, env.game, 0)
    env.step(slot)
    assert active.game.to_dict() == direct.to_dict() == env.game.to_dict()
    state = GameState.from_engine(active.game).model_dump()
    assert state["rules_id"] == "foundation-trade-v1"
    assert state["decision_contract"]["viewer_id"] is None
    assert state["decision_contract"]["pending_offer"]["trade_id"] == 0

    response = ActionRequest(
        action_type="reject_trade",
        contract_version="decision-contract-v1",
        expected_revision=1,
        request_id="response-1",
        trade_id=0,
    ).to_engine_action(1)
    success, reason = await manager.execute_action(
        game_id, response, expected_revision=1, request_id="response-1"
    )
    assert success, reason
    direct.apply_action(1, RejectTrade(1, 0), expected_revision=1)
    env.step(OFFSET_REJECT_TRADE_V3)
    assert active.game.to_dict() == direct.to_dict() == env.game.to_dict()


def test_acceptance_preserves_mortgage_and_exact_cash():
    game = prepared_game()
    game.property_manager.properties[3].mortgaged = True
    game.apply_action(0, ProposeTrade(0, 1, [6], 100, [3], 0))
    game.apply_action(1, AcceptTrade(1, 0))
    assert [player.money for player in game.players] == [1400, 1600]
    assert game.property_manager.properties[3].owner == 0
    assert game.property_manager.properties[3].mortgaged


def test_checkpoint_contract_rejects_missing_and_incompatible_metadata(tmp_path):
    model = tmp_path / "policy"
    with pytest.raises(ValueError, match="metadata is missing"):
        validate_checkpoint_metadata(model, "foundation-trade-v1", 4)
    write_checkpoint_metadata(model, "foundation-trade-v1", 4)
    validate_checkpoint_metadata(model, "foundation-trade-v1", 4)
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_checkpoint_metadata(model, "foundation-trade-v1", 2)
    assert learner_contract("foundation-v1", 4)["action_version"] == "action-v2"
    with pytest.raises(ValueError, match="enable_trades is obsolete"):
        ObservationEncoder(2, enable_trades=True)


def test_restored_game_does_not_mutate_its_source_snapshot():
    game = prepared_game()
    snapshot = game.to_dict()
    frozen = json.dumps(snapshot, sort_keys=True)
    restored = MonopolyGame.from_dict(snapshot)
    restored.apply_action(0, ProposeTrade(0, 1, [1], 0, [3], 0))
    restored.apply_action(1, RejectTrade(1, 0))
    assert json.dumps(snapshot, sort_keys=True) == frozen


def test_rejected_offer_resumes_preselected_policy_action_without_new_choice():
    class EndTurnPolicy(Agent):
        def __init__(self):
            super().__init__(0, "end-turn")
            self.calls = 0

        def choose_action(self, observation, action_mask, game):
            self.calls += 1
            return OFFSET_END_TURN

        def reset(self):
            self.calls = 0

    game = prepared_game()
    encoder = ActionEncoder(rules_id="foundation-trade-v1")
    base = EndTurnPolicy()
    agent = TradingAgent(base, response="reject")
    proposal = agent.choose_native_action(game, encoder)
    assert isinstance(proposal, ProposeTrade)
    assert base.calls == 1
    game.apply_action(0, proposal)
    game.apply_action(1, RejectTrade(1, 0))
    resumed = agent.choose_native_action(game, encoder)
    assert isinstance(resumed, EndTurn)
    assert base.calls == 1


def test_aec_disabled_response_action_preserves_state_and_reward_buffers():
    env = MonopolyEnv(2, rules_id="foundation-trade-v1")
    env.reset(options={"engine_seed": 41})
    env.game = prepared_game()
    env.agent_selection = "player_0"
    env._update_infos()
    env.step(OFFSET_TRADE_CANDIDATE)
    env.rewards["player_0"] = 0.25
    env._cumulative_rewards["player_0"] = 0.5
    snapshot = json.dumps(env.game.to_dict(), sort_keys=True)
    rewards = env.rewards.copy(), env._cumulative_rewards.copy()
    mortgage_baltic = OFFSET_MORTGAGE + BUYABLE_POSITIONS.index(3)
    with pytest.raises(ValueError, match="not legal"):
        env.step(mortgage_baltic)
    assert json.dumps(env.game.to_dict(), sort_keys=True) == snapshot
    assert (env.rewards, env._cumulative_rewards) == rewards
