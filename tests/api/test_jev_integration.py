"""Asynchronous API orchestration tests for the guided Jev agent."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agents.jev import FakeProvider
from agents.jev.types import QuestionBatch
from api.config import Settings
from api.models.lobby import LobbySettings
from api.services.ai_manager import AIManager
from api.services.game_manager import GameManager
from api.services.lobby_manager import LobbyManager
from monopoly_engine import ProposeTrade, RollDice


def answer_first(request: QuestionBatch) -> dict[str, Any]:
    answers = {}
    for question_id, question in request.questions.items():
        criteria = question["criteria"]
        selected = next(iter(criteria))
        if question_id == "cash_reserve":
            selected = "reserve_200"
        answers[question_id] = {
            "type": "choice",
            "choice": selected,
            "probabilities": {
                key: 1.0 if key == selected else 0.0 for key in criteria
            },
            "confidence": 1.0,
        }
    return {
        "model": "jev-1.13.0",
        "answers": answers,
        "usage": {"input_tokens": 50, "output_tokens": 1},
    }


@pytest.mark.asyncio
async def test_stale_result_does_not_execute_or_update_strategy() -> None:
    provider = FakeProvider([answer_first], delay_seconds=0.05)
    ai = AIManager(think_delay_ms=0, settings=Settings(), provider=provider)
    games = GameManager()
    game_id = await games.create_game(2, seed=12, rules_id="foundation-trade-v1")
    active = await games.get_game(game_id)
    assert active is not None
    active.game.property_manager.properties[1].owner = 0
    agent = ai.create_agent(game_id, 0, "jev")
    task = asyncio.create_task(ai._process_jev_decision(games, game_id, 0, agent, 1))
    while not provider.requests:
        await asyncio.sleep(0)
    success, reason = await games.execute_action(
        game_id, RollDice(0), expected_revision=0, request_id="human-won-race"
    )
    assert success, reason
    committed, message = await task
    assert not committed
    assert "Stale" in message
    assert agent.memory.strategy_version == 0
    assert len(active.agent_action_log) == 1
    await ai.shutdown()


@pytest.mark.asyncio
async def test_successful_commit_records_provenance_and_updates_memory_once() -> None:
    provider = FakeProvider([answer_first])
    ai = AIManager(think_delay_ms=0, settings=Settings(), provider=provider)
    games = GameManager()
    games.set_ai_manager(ai)
    game_id = await games.create_game(2, seed=13, rules_id="foundation-trade-v1")
    agent = ai.create_agent(game_id, 0, "jev")
    committed, reason = await ai._process_jev_decision(games, game_id, 0, agent, 1)
    assert committed, reason
    active = await games.get_game(game_id)
    assert active is not None
    record = active.agent_action_log[0]
    assert record["source"] == "forced"
    assert record["prompt_protocol"] == "guided-jev-v1"
    assert record["before_revision"] == 0
    assert record["after_revision"] == 1
    assert record["action"]["type"] == "RollDice"
    assert agent.memory.strategy_version == 0
    state = await games.get_game_state(game_id)
    assert state is not None
    assert state.players[0].is_ai is False
    assert state.agent_inspections[0].basis_revision == 1
    await ai.shutdown()


@pytest.mark.asyncio
async def test_lobby_requires_trading_rules_and_reports_provider_availability() -> None:
    ai = AIManager(think_delay_ms=0, settings=Settings(), provider=FakeProvider())
    lobbies = LobbyManager()
    lobbies.set_ai_manager(ai)
    lobby_id, _ = await lobbies.create_lobby("Jev", "Human", "session")
    added, message, _ = await lobbies.add_ai_player(lobby_id, "session", "jev")
    assert not added
    assert message == "Jev requires foundation-trade-v1"
    updated, message = await lobbies.update_settings(
        lobby_id,
        "session",
        LobbySettings(rules_id="foundation-trade-v1"),
    )
    assert updated, message
    added, message, slot = await lobbies.add_ai_player(lobby_id, "session", "jev")
    assert added, message
    assert slot == 1
    state = await lobbies.get_lobby_state(lobby_id)
    assert state is not None and state.jev_available
    reverted, message = await lobbies.update_settings(
        lobby_id,
        "session",
        LobbySettings(rules_id="foundation-v1"),
    )
    assert not reverted
    assert "Remove Jev" in message
    await ai.shutdown()


@pytest.mark.asyncio
async def test_human_offer_to_jev_accepts_and_resumes_the_proposer() -> None:
    def accept_answer(request: QuestionBatch) -> dict[str, Any]:
        payload = answer_first(request)
        if "action" in request.questions:
            criteria = request.questions["action"]["criteria"]
            selected = next(key for key in criteria if key.startswith("accept_trade"))
            payload["answers"]["action"] = {
                "type": "choice",
                "choice": selected,
                "probabilities": {
                    key: 1.0 if key == selected else 0.0 for key in criteria
                },
                "confidence": 1.0,
            }
        return payload

    provider = FakeProvider(default_step=accept_answer)
    ai = AIManager(think_delay_ms=0, settings=Settings(), provider=provider)
    games = GameManager(background_ai_scheduling=False)
    games.set_ai_manager(ai)
    game_id = await games.create_game(2, seed=15, rules_id="foundation-trade-v1")
    active = await games.get_game(game_id)
    assert active is not None
    active.game.state.phase = "asset_management"
    active.game.state.roll_owed = False
    active.game.property_manager.properties[1].owner = 0
    active.game.property_manager.properties[3].owner = 1
    agent = ai.create_agent(game_id, 1, "jev")
    proposed, reason = await games.execute_action(
        game_id,
        ProposeTrade(0, 1, [1], 100, [3], 0),
        expected_revision=0,
        request_id="human-proposal",
    )
    assert proposed, reason
    assert active.game.decision_player == 1
    taken = await ai.process_ai_turn(games, game_id, 1, max_actions=1)
    assert taken == 1
    assert active.game.decision_player == 0
    assert active.game.property_manager.properties[1].owner == 1
    assert active.game.property_manager.properties[3].owner == 0
    assert [player.money for player in active.game.players] == [1400, 1600]
    assert [entry.outcome for entry in agent.memory.recent_trades] == [
        "proposed",
        "accepted",
    ]
    await ai.shutdown()


@pytest.mark.asyncio
async def test_game_deletion_cancels_stalled_provider_task() -> None:
    provider = FakeProvider([answer_first], delay_seconds=10)
    ai = AIManager(think_delay_ms=0, settings=Settings(), provider=provider)
    games = GameManager()
    games.set_ai_manager(ai)
    game_id = await games.create_game(2, seed=14, rules_id="foundation-trade-v1")
    ai.create_agent(game_id, 0, "jev")
    ai.schedule_ai_turns(games, game_id)
    while not provider.requests:
        await asyncio.sleep(0)
    assert await games.delete_game(game_id)
    await asyncio.sleep(0)
    assert not ai.is_ai_player(game_id, 0)
    assert game_id not in ai._tasks or ai._tasks[game_id].cancelled()
    await ai.shutdown()
