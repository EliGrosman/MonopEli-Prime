"""Offline contracts for the guided Jev policy and provider boundary."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from agents.jev import FakeProvider, GuidedJevAgent, JevRuntime
from agents.jev.provider import (
    BudgetLimits,
    ProviderError,
    TypeSafeProvider,
    validate_provider_payload,
)
from agents.jev.types import DecisionCallBudget, QuestionBatch, StrategyMemory
from monopoly_engine import EndTurn, MonopolyGame, ProposeTrade, RejectTrade


def prepared_trade_game() -> MonopolyGame:
    game = MonopolyGame(
        2,
        player_names=["PRIVATE_PLAYER_NAME", "OPPONENT_CANARY"],
        seed=41,
        rules_id="foundation-trade-v1",
    )
    game.state.phase = "asset_management"
    game.state.roll_owed = False
    game.property_manager.properties[1].owner = 0
    game.property_manager.properties[6].owner = 0
    game.property_manager.properties[3].owner = 1
    game.property_manager.properties[8].owner = 1
    return game


def choice_payload(
    request: QuestionBatch,
    selector: Callable[[str, dict[str, Any]], str] | None = None,
) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    for question_id, question in request.questions.items():
        criteria = question["criteria"]
        selected = selector(question_id, criteria) if selector else next(iter(criteria))
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
        "usage": {"input_tokens": 100, "output_tokens": 2},
        "request_id": "offline-answer",
    }


def simple_request() -> QuestionBatch:
    return QuestionBatch(
        state={},
        questions={
            "action": {
                "type": "choice",
                "instructions": "Choose.",
                "criteria": {"a": "A", "b": "B"},
            }
        },
    )


def test_provider_validation_rejects_unknown_choice_and_invalid_probabilities() -> None:
    request = QuestionBatch(
        state={},
        questions={
            "action": {
                "type": "choice",
                "instructions": "Choose.",
                "criteria": {"a": "A", "b": "B"},
            }
        },
    )
    unknown = choice_payload(request)
    unknown["answers"]["action"]["choice"] = "other"
    with pytest.raises(ProviderError, match="unknown_choice"):
        validate_provider_payload(request, unknown)
    invalid = choice_payload(request)
    invalid["answers"]["action"]["probabilities"] = {"a": 0.4, "b": 0.4}
    with pytest.raises(ProviderError, match="malformed_probability_sum"):
        validate_provider_payload(request, invalid)
    missing = choice_payload(request)
    missing["answers"]["action"]["probabilities"] = {"a": 1.0}
    with pytest.raises(ProviderError, match="malformed_probabilities"):
        validate_provider_payload(request, missing)
    extra = choice_payload(request)
    extra["answers"]["action"]["probabilities"]["other"] = 0.0
    with pytest.raises(ProviderError, match="malformed_probabilities"):
        validate_provider_payload(request, extra)


def test_probability_validation_has_bounded_absolute_tolerance(
    caplog: pytest.LogCaptureFixture,
) -> None:
    criteria = {f"option_{index}": str(index) for index in range(255)}
    request = QuestionBatch(
        state={},
        questions={
            "large_choice": {
                "type": "choice",
                "instructions": "Choose.",
                "criteria": criteria,
            }
        },
    )
    payload = choice_payload(request)
    payload["answers"]["large_choice"]["probabilities"] = {
        key: 1 / len(criteria) for key in criteria
    }
    validate_provider_payload(request, payload)

    payload["answers"]["large_choice"]["probabilities"] = {
        key: 0.0 for key in criteria
    }
    payload["answers"]["large_choice"]["probabilities"]["option_0"] = 1.0
    payload["answers"]["large_choice"]["probabilities"]["option_1"] = 0.0001
    validate_provider_payload(request, payload)

    payload["answers"]["large_choice"]["probabilities"]["option_0"] = 0.9999
    payload["answers"]["large_choice"]["probabilities"]["option_1"] = 0.0
    validate_provider_payload(request, payload)
    payload["answers"]["large_choice"]["probabilities"]["option_0"] = 0.999899
    with pytest.raises(ProviderError, match="malformed_probability_sum"):
        validate_provider_payload(request, payload)
    caplog.clear()

    sentinel = "PRIVATE_OPTION_SENTINEL"
    payload["answers"]["large_choice"]["probabilities"]["option_0"] = 1.0
    payload["answers"]["large_choice"]["probabilities"]["option_1"] = 0.000101
    payload["answers"]["large_choice"]["probabilities"][sentinel] = payload[
        "answers"
    ]["large_choice"]["probabilities"].pop("option_254")
    request.questions["large_choice"]["criteria"][sentinel] = request.questions[
        "large_choice"
    ]["criteria"].pop("option_254")
    with pytest.raises(ProviderError, match="malformed_probability_sum") as caught:
        validate_provider_payload(request, payload)
    assert str(caught.value) == "malformed_probability_sum"
    assert sentinel not in caplog.text
    assert "question_type=choice" in caplog.text
    assert "option_count=255" in caplog.text
    assert "probability_sum=" in caplog.text
    assert "deviation=" in caplog.text


def test_score_probability_boundaries_and_invalid_values() -> None:
    request = QuestionBatch(
        state={},
        questions={
            "score": {
                "type": "score",
                "instructions": "Score.",
                "criteria": [str(index) for index in range(10)],
            }
        },
    )
    probabilities = {str(index): 0.1 for index in range(10)}
    payload: dict[str, Any] = {
        "model": "jev-1.13.0",
        "answers": {
            "score": {
                "type": "score",
                "score": 4.5,
                "probabilities": probabilities,
                "confidence": 0.5,
                "legend": {str(index): str(index) for index in range(10)},
            }
        },
        "usage": {"input_tokens": 10, "output_tokens": 1},
    }
    validate_provider_payload(request, payload)
    for invalid in (-0.01, float("nan"), float("inf")):
        payload["answers"]["score"]["probabilities"]["0"] = invalid
        with pytest.raises(ProviderError, match="malformed_probability"):
            validate_provider_payload(request, payload)
    payload["answers"]["score"]["probabilities"] = probabilities | {"extra": 0.0}
    with pytest.raises(ProviderError, match="malformed_probabilities"):
        validate_provider_payload(request, payload)


@pytest.mark.asyncio
async def test_public_payload_is_allowlisted_and_replaces_player_names() -> None:
    game = prepared_trade_game()
    provider = FakeProvider([choice_payload] * 4)
    agent = GuidedJevAgent(0, "public-boundary", JevRuntime(provider))
    await agent.decide(game.decision_view(0))
    encoded = json.dumps(provider.requests[0].state, sort_keys=True)
    assert set(provider.requests[0].state) == {
        "protocol_version",
        "decision",
        "players",
        "properties",
        "legal_actions",
        "trade",
        "pending_offer",
        "strategy",
    }
    assert "PRIVATE_PLAYER_NAME" not in encoded
    assert "OPPONENT_CANARY" not in encoded
    assert "rng" not in encoded.lower()
    assert "deck" not in encoded.lower()
    assert "seed" not in encoded.lower()
    assert [item["label"] for item in provider.requests[0].state["players"]] == [
        "Seat 1",
        "Seat 2",
    ]


@pytest.mark.asyncio
async def test_custom_trade_can_use_exact_cash_outside_suggestion_limits() -> None:
    def select(question_id: str, criteria: dict[str, Any]) -> str:
        choices = list(criteria)
        if question_id == "objective":
            return "acquire"
        if question_id == "cash_reserve":
            return "reserve_200"
        if question_id == "action":
            return "consider_trade"
        if question_id == "trade_path":
            return "construct_offer_to_1"
        if question_id == "first_give_property":
            return "property_1"
        if question_id == "first_want_property":
            return "property_3"
        if question_id in {"second_give_property", "second_want_property"}:
            return "none"
        if question_id == "cash_direction":
            return "recipient_pays"
        if question_id == "cash_range":
            for key, description in criteria.items():
                low, high = (int(item) for item in re.findall(r"\$(\d+)", description))
                if low <= 777 <= high:
                    return key
        if question_id == "cash_amount":
            return "cash_777"
        if question_id == "complete_offer":
            return "propose"
        return choices[0]

    game = prepared_trade_game()
    provider = FakeProvider([lambda request: choice_payload(request, select)] * 12)
    agent = GuidedJevAgent(0, "custom-offer", JevRuntime(provider))
    outcome = await agent.decide(game.decision_view(0))
    assert isinstance(outcome.command, ProposeTrade)
    assert outcome.command.give_properties == [1]
    assert outcome.command.want_properties == [3]
    assert outcome.command.want_money == 777
    assert outcome.command.give_money == 0
    assert outcome.command.validate(game) == (True, "")
    game.apply_action(0, outcome.command, expected_revision=0)
    game.apply_action(1, RejectTrade(1, 0), expected_revision=1)
    assert game.decision_player == 0
    assert game.property_manager.properties[1].owner == 0
    assert game.property_manager.properties[3].owner == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cash_direction", ["proposer_pays", "recipient_pays"])
async def test_custom_trade_questions_receive_evolving_offer_context(
    cash_direction: str,
) -> None:
    def select(question_id: str, criteria: dict[str, Any]) -> str:
        selections = {
            "objective": "acquire",
            "cash_reserve": "reserve_200",
            "action": "consider_trade",
            "trade_path": "construct_offer_to_1",
            "first_give_property": "property_1",
            "first_want_property": "property_3",
            "second_give_property": "property_6",
            "second_want_property": "property_8",
            "cash_direction": cash_direction,
            "cash_amount": "cash_777",
            "complete_offer": "propose",
        }
        if question_id == "cash_range":
            return next(
                key
                for key, description in criteria.items()
                if int(re.findall(r"\$(\d+)", description)[0]) <= 777
                <= int(re.findall(r"\$(\d+)", description)[1])
            )
        return selections.get(question_id, next(iter(criteria)))

    game = prepared_trade_game()
    provider = FakeProvider(default_step=lambda request: choice_payload(request, select))
    outcome = await GuidedJevAgent(0, f"context-{cash_direction}", JevRuntime(provider)).decide(
        game.decision_view(0)
    )
    assert isinstance(outcome.command, ProposeTrade)
    assert outcome.command.give_properties == [1, 6]
    assert outcome.command.want_properties == [3, 8]
    expected_money = (777, 0) if cash_direction == "proposer_pays" else (0, 777)
    assert (outcome.command.give_money, outcome.command.want_money) == expected_money
    assert outcome.command.validate(game) == (True, "")

    requests = {
        next(iter(request.questions)): request
        for request in provider.requests
        if len(request.questions) == 1
    }
    partials = {
        question_id: requests[question_id].state["guided_step"]["partial_offer"]
        for question_id in (
            "first_give_property",
            "first_want_property",
            "second_give_property",
            "second_want_property",
            "cash_direction",
            "cash_range",
            "cash_amount",
            "complete_offer",
        )
    }
    assert partials["first_give_property"] == {
        "proposer": 0,
        "recipient": 1,
        "give_properties": [],
        "want_properties": [],
        "cash_direction": None,
        "cash_range": None,
        "give_money": None,
        "want_money": None,
    }
    assert partials["first_want_property"]["give_properties"] == [1]
    assert partials["second_give_property"]["want_properties"] == [3]
    assert partials["second_want_property"]["give_properties"] == [1, 6]
    assert partials["cash_direction"]["want_properties"] == [3, 8]
    assert partials["cash_range"]["cash_direction"] == cash_direction
    assert partials["cash_range"]["cash_range"] == {"minimum": 1, "maximum": 1500}
    selected_range = partials["cash_amount"]["cash_range"]
    assert selected_range["minimum"] <= 777 <= selected_range["maximum"]
    assert partials["complete_offer"]["cash_range"] == selected_range
    assert (
        partials["complete_offer"]["give_money"],
        partials["complete_offer"]["want_money"],
    ) == expected_money
    assert partials["first_want_property"]["give_properties"] == [1]


@pytest.mark.asyncio
async def test_malformed_incoming_trade_response_falls_back_to_reject() -> None:
    game = prepared_trade_game()
    game.apply_action(0, ProposeTrade(0, 1, [1], 0, [3], 0))
    provider = FakeProvider([{"not": "a response"}])
    agent = GuidedJevAgent(1, "incoming-fallback", JevRuntime(provider))
    outcome = await agent.decide(game.decision_view(1))
    assert isinstance(outcome.command, RejectTrade)
    assert outcome.source == "fallback"
    assert outcome.fallback_reason == "malformed_response"


@pytest.mark.asyncio
async def test_timeout_retries_are_bounded_and_visible_as_fallback() -> None:
    game = prepared_trade_game()
    provider = FakeProvider([choice_payload, choice_payload], delay_seconds=0.05)
    runtime = JevRuntime(provider, max_retries=1, attempt_timeout_seconds=0.005)
    agent = GuidedJevAgent(0, "timeout", runtime)
    outcome = await agent.decide(game.decision_view(0))
    assert isinstance(outcome.command, EndTurn)
    assert outcome.source == "fallback"
    assert outcome.fallback_reason == "timeout"
    assert len(provider.requests) == 2
    usage = runtime.game_usage("timeout")
    assert usage["attempts"] == 2
    assert usage["unknown_reserved_tokens"] == 128_000


@pytest.mark.asyncio
async def test_budget_exhaustion_uses_fallback_without_calling_provider() -> None:
    game = prepared_trade_game()
    provider = FakeProvider([choice_payload])
    runtime = JevRuntime(
        provider,
        game_limits=BudgetLimits(0, 5_000_000, 0.25),
    )
    outcome = await GuidedJevAgent(0, "budget", runtime).decide(game.decision_view(0))
    assert outcome.source == "fallback"
    assert outcome.fallback_reason == "budget_exhausted"
    assert provider.requests == []


@pytest.mark.asyncio
async def test_cancellation_retains_unknown_usage_reservation() -> None:
    provider = FakeProvider([choice_payload], delay_seconds=1.0)
    runtime = JevRuntime(provider)
    request = QuestionBatch(
        state={},
        questions={
            "action": {
                "type": "choice",
                "instructions": "Choose.",
                "criteria": {"a": "A"},
            }
        },
    )
    task = asyncio.create_task(
        runtime.evaluate(
            "cancelled",
            "session",
            request,
            DecisionCallBudget(),
            time.monotonic() + 5,
        )
    )
    while not provider.requests:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runtime.game_usage("cancelled")["unknown_reserved_tokens"] == 64_000


@pytest.mark.asyncio
async def test_http_answer_rejection_accounts_valid_usage_in_every_scope() -> None:
    request = simple_request()

    def respond(_: httpx.Request) -> httpx.Response:
        payload = choice_payload(request)
        payload["usage"] = {"input_tokens": 37, "output_tokens": 4}
        payload["answers"]["action"]["probabilities"] = {"a": 0.4, "b": 0.4}
        return httpx.Response(200, json=payload)

    provider = TypeSafeProvider("offline-key", transport=httpx.MockTransport(respond))
    runtime = JevRuntime(provider, max_retries=0)
    call_budget = DecisionCallBudget()
    with pytest.raises(ProviderError, match="malformed_probability_sum") as caught:
        await runtime.evaluate(
            "rejected", "session-rejected", request, call_budget, time.monotonic() + 2
        )
    assert caught.value.usage is not None
    assert caught.value.usage.input_tokens == 37
    assert call_budget.input_tokens == 37
    for usage in (
        runtime.game_usage("rejected"),
        runtime.session_usage("session-rejected"),
        runtime.process_usage(),
    ):
        assert usage["input_tokens"] == 37
        assert usage["unknown_reserved_tokens"] == 0
        assert usage["inflight_reserved_tokens"] == 0
    await runtime.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["malformed_json", "malformed_usage"])
async def test_http_untrustworthy_usage_retains_reservation(failure: str) -> None:
    request = simple_request()

    def respond(_: httpx.Request) -> httpx.Response:
        if failure == "malformed_json":
            return httpx.Response(200, content=b"not-json")
        payload = choice_payload(request)
        payload["usage"] = {"input_tokens": "unknown", "output_tokens": 1}
        return httpx.Response(200, json=payload)

    provider = TypeSafeProvider("offline-key", transport=httpx.MockTransport(respond))
    runtime = JevRuntime(provider, max_retries=0)
    with pytest.raises(ProviderError, match=failure):
        await runtime.evaluate(
            failure, f"session-{failure}", request, DecisionCallBudget(), time.monotonic() + 2
        )
    for usage in (
        runtime.game_usage(failure),
        runtime.session_usage(f"session-{failure}"),
        runtime.process_usage(),
    ):
        assert usage["input_tokens"] == 0
        assert usage["unknown_reserved_tokens"] == 64_000
        assert usage["inflight_reserved_tokens"] == 0
    await runtime.aclose()


@pytest.mark.asyncio
async def test_http_success_and_retry_account_once_per_attempt() -> None:
    request = simple_request()
    calls = 0

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(500)
        payload = choice_payload(request)
        payload["usage"] = {"input_tokens": 23, "output_tokens": 2}
        return httpx.Response(200, json=payload)

    provider = TypeSafeProvider("offline-key", transport=httpx.MockTransport(respond))
    runtime = JevRuntime(provider, max_retries=1)
    call_budget = DecisionCallBudget()
    result = await runtime.evaluate(
        "retried", "session-retried", request, call_budget, time.monotonic() + 2
    )
    assert result.usage.input_tokens == 23
    assert result.attempts == 2
    assert call_budget.attempts == 2
    assert call_budget.input_tokens == 23
    for usage in (
        runtime.game_usage("retried"),
        runtime.session_usage("session-retried"),
        runtime.process_usage(),
    ):
        assert usage["attempts"] == 2
        assert usage["input_tokens"] == 23
        assert usage["unknown_reserved_tokens"] == 64_000
        assert usage["inflight_reserved_tokens"] == 0
    await runtime.aclose()


@pytest.mark.asyncio
async def test_http_cancellation_after_dispatch_retains_reservation() -> None:
    request = simple_request()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def respond(_: httpx.Request) -> httpx.Response:
        entered.set()
        await release.wait()
        return httpx.Response(200, json=choice_payload(request))

    provider = TypeSafeProvider("offline-key", transport=httpx.MockTransport(respond))
    runtime = JevRuntime(provider, max_retries=0)
    task = asyncio.create_task(
        runtime.evaluate(
            "http-cancelled",
            "session-http-cancelled",
            request,
            DecisionCallBudget(),
            time.monotonic() + 2,
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runtime.game_usage("http-cancelled")["unknown_reserved_tokens"] == 64_000
    assert runtime.game_usage("http-cancelled")["inflight_reserved_tokens"] == 0
    await runtime.aclose()


@pytest.mark.asyncio
async def test_queued_request_expires_before_slot_releases_and_never_dispatches() -> None:
    request = simple_request()
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    provider_calls = 0

    async def respond(batch: QuestionBatch) -> dict[str, Any]:
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            first_entered.set()
            await release_first.wait()
        return choice_payload(batch)

    provider = FakeProvider(default_step=respond)
    runtime = JevRuntime(provider, max_concurrent=1, max_retries=0)
    first = asyncio.create_task(
        runtime.evaluate(
            "holder", "session-queue", request, DecisionCallBudget(), time.monotonic() + 2
        )
    )
    await first_entered.wait()
    queued_budget = DecisionCallBudget()
    with pytest.raises(ProviderError, match="decision_deadline_exceeded"):
        await runtime.evaluate(
            "queued",
            "session-queue",
            request,
            queued_budget,
            time.monotonic() + 0.02,
        )
    assert provider_calls == 1
    assert len(provider.requests) == 1
    assert queued_budget.attempts == 1
    assert runtime.game_usage("queued")["unknown_reserved_tokens"] == 0
    assert runtime.game_usage("queued")["inflight_reserved_tokens"] == 0
    release_first.set()
    await first
    await runtime.evaluate(
        "after-queue", "session-queue", request, DecisionCallBudget(), time.monotonic() + 2
    )
    assert provider_calls == 2
    await runtime.aclose()


@pytest.mark.asyncio
async def test_cancellation_while_queued_releases_reservation_without_dispatch() -> None:
    request = simple_request()
    first_entered = asyncio.Event()
    release_first = asyncio.Event()

    async def respond(batch: QuestionBatch) -> dict[str, Any]:
        if not first_entered.is_set():
            first_entered.set()
            await release_first.wait()
        return choice_payload(batch)

    provider = FakeProvider(default_step=respond)
    runtime = JevRuntime(provider, max_concurrent=1, max_retries=0)
    first = asyncio.create_task(
        runtime.evaluate(
            "queue-holder", "session", request, DecisionCallBudget(), time.monotonic() + 2
        )
    )
    await first_entered.wait()
    queued = asyncio.create_task(
        runtime.evaluate(
            "queue-cancelled", "session", request, DecisionCallBudget(), time.monotonic() + 2
        )
    )
    while runtime.game_usage("queue-cancelled")["inflight_reserved_tokens"] == 0:
        await asyncio.sleep(0)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert len(provider.requests) == 1
    usage = runtime.game_usage("queue-cancelled")
    assert usage["input_tokens"] == 0
    assert usage["unknown_reserved_tokens"] == 0
    assert usage["inflight_reserved_tokens"] == 0
    release_first.set()
    await first
    await runtime.aclose()


@pytest.mark.asyncio
async def test_cancellation_during_retry_backoff_has_no_open_reservation() -> None:
    request = simple_request()
    provider = FakeProvider(
        [ProviderError("retry", retryable=True, usage_unknown=False), choice_payload]
    )
    runtime = JevRuntime(provider, max_retries=1)
    task = asyncio.create_task(
        runtime.evaluate(
            "backoff-cancelled",
            "session",
            request,
            DecisionCallBudget(),
            time.monotonic() + 2,
        )
    )
    while len(provider.requests) < 1:
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    usage = runtime.game_usage("backoff-cancelled")
    assert usage["attempts"] == 1
    assert usage["unknown_reserved_tokens"] == 0
    assert usage["inflight_reserved_tokens"] == 0
    await runtime.aclose()


def test_strategy_memory_is_per_agent_and_resettable() -> None:
    runtime = JevRuntime(FakeProvider())
    first = GuidedJevAgent(0, "game-a", runtime)
    second = GuidedJevAgent(1, "game-a", runtime)
    first.memory = StrategyMemory(strategy_version=4, objective_id="liquidity")
    assert second.memory.strategy_version == 0
    first.reset()
    assert first.memory.strategy_version == 0
    assert second.memory.objective_id == "assess"
