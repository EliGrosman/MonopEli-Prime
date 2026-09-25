"""TypeSafe Jev HTTP adapter, validation, retries, and bounded usage accounting."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import httpx

from .types import (
    Answer,
    ChoiceAnswer,
    DecisionCallBudget,
    NoulAnswer,
    ProviderResult,
    ProviderUsage,
    QuestionBatch,
    ScoreAnswer,
)

logger = logging.getLogger(__name__)
PROBABILITY_SUM_TOLERANCE = 0.0001


class ProviderError(RuntimeError):
    """Sanitized provider failure safe for logs and UI status."""

    def __init__(
        self,
        code: str,
        *,
        retryable: bool = False,
        retry_after: float | None = None,
        usage_unknown: bool = False,
        usage: ProviderUsage | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.retry_after = retry_after
        self.usage_unknown = usage_unknown
        self.usage = usage


class ProviderBudgetExhaustedError(ProviderError):
    def __init__(self, code: str = "budget_exhausted") -> None:
        super().__init__(code)


class DecisionProvider(Protocol):
    async def evaluate(self, request: QuestionBatch) -> ProviderResult: ...

    async def aclose(self) -> None: ...


def _finite_probability(value: Any, field: str) -> float:
    if type(value) not in (int, float):
        raise ProviderError(f"malformed_{field}")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ProviderError(f"malformed_{field}")
    return number


def _probabilities(
    value: Any, expected: set[str], *, question_type: str
) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ProviderError("malformed_probabilities")
    result = {str(key): _finite_probability(item, "probability") for key, item in value.items()}
    total = math.fsum(result.values())
    deviation = abs(total - 1.0)
    if deviation > PROBABILITY_SUM_TOLERANCE:
        logger.warning(
            "Rejected provider probability sum: question_type=%s option_count=%d "
            "probability_sum=%.17g deviation=%.17g tolerance=%.17g",
            question_type,
            len(result),
            total,
            deviation,
            PROBABILITY_SUM_TOLERANCE,
        )
        raise ProviderError("malformed_probability_sum")
    return result


def _provider_usage(payload: Any) -> ProviderUsage:
    if not isinstance(payload, dict):
        raise ProviderError("malformed_usage", usage_unknown=True)
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise ProviderError("malformed_usage", usage_unknown=True)
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if type(input_tokens) is not int or input_tokens < 0:
        raise ProviderError("malformed_usage", usage_unknown=True)
    if type(output_tokens) is not int or output_tokens < 0:
        raise ProviderError("malformed_usage", usage_unknown=True)
    return ProviderUsage(input_tokens, output_tokens)


def validate_provider_payload(
    request: QuestionBatch, payload: Any, elapsed_seconds: float = 0.0
) -> ProviderResult:
    """Strictly validate the documented System One response shape."""
    try:
        usage = _provider_usage(payload)
        usage_error: ProviderError | None = None
    except ProviderError as error:
        usage = None
        usage_error = error

    try:
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), str):
            raise ProviderError("malformed_response")
        raw_answers = payload.get("answers")
        if not isinstance(raw_answers, dict) or set(raw_answers) != set(request.questions):
            raise ProviderError("malformed_answers")
        answers: dict[str, Answer] = {}
        for question_id, question in request.questions.items():
            raw = raw_answers.get(question_id)
            if not isinstance(raw, dict) or raw.get("type") != question.get("type"):
                raise ProviderError("malformed_answer_type")
            answer_type = question["type"]
            if answer_type == "choice":
                criteria = question.get("criteria")
                if not isinstance(criteria, dict) or not criteria:
                    raise ProviderError("malformed_choice_question")
                choice = raw.get("choice")
                if not isinstance(choice, str) or choice not in criteria:
                    raise ProviderError("unknown_choice")
                answers[question_id] = ChoiceAnswer(
                    choice=choice,
                    probabilities=_probabilities(
                        raw.get("probabilities"), set(criteria), question_type="choice"
                    ),
                    confidence=_finite_probability(raw.get("confidence"), "confidence"),
                )
            elif answer_type == "score":
                criteria = question.get("criteria")
                if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                    raise ProviderError("malformed_score_question")
                score = raw.get("score")
                if (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(float(score))
                ):
                    raise ProviderError("malformed_score")
                score_number = float(score)
                keys = {str(index) for index in range(len(criteria))}
                legend = raw.get("legend")
                if not isinstance(legend, dict) or set(legend) != keys:
                    raise ProviderError("malformed_legend")
                answers[question_id] = ScoreAnswer(
                    score=score_number,
                    probabilities=_probabilities(
                        raw.get("probabilities"), keys, question_type="score"
                    ),
                    confidence=_finite_probability(raw.get("confidence"), "confidence"),
                    legend={str(key): str(value) for key, value in legend.items()},
                )
            elif answer_type == "noul":
                answers[question_id] = NoulAnswer(
                    noul=_finite_probability(raw.get("noul"), "noul")
                )
            else:
                raise ProviderError("unsupported_question_type")
        request_id = payload.get("request_id")
        if request_id is not None and not isinstance(request_id, str):
            raise ProviderError("malformed_request_id")
    except ProviderError as error:
        error.usage = usage
        error.usage_unknown = usage is None
        raise

    if usage_error is not None:
        raise usage_error
    assert usage is not None
    return ProviderResult(
        model=payload["model"],
        answers=answers,
        usage=usage,
        elapsed_seconds=elapsed_seconds,
        request_id=request_id,
    )


class TypeSafeProvider:
    """Direct asynchronous adapter for the official TypeSafe endpoint."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.typesafe.ai",
        timeout_seconds: float = 5.0,
        connect_timeout_seconds: float = 2.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key or any(ch.isspace() for ch in api_key.strip()):
            raise ValueError("TYPESAFE_API_KEY is missing or invalid")
        timeout = httpx.Timeout(timeout_seconds, connect=connect_timeout_seconds)
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            timeout=timeout,
            transport=transport,
        )

    async def evaluate(self, request: QuestionBatch) -> ProviderResult:
        started = time.monotonic()
        try:
            response = await self._client.post("/v1/systemone", json=request.to_dict())
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "timeout", retryable=True, usage_unknown=True
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                "transport_error", retryable=True, usage_unknown=True
            ) from exc
        if response.status_code == 401:
            raise ProviderError("authentication_failed")
        if response.status_code == 422:
            raise ProviderError("request_rejected")
        if response.status_code in (429, 529) or response.status_code >= 500:
            retry_after = response.headers.get("retry-after")
            try:
                delay = float(retry_after) if retry_after is not None else None
            except ValueError:
                delay = None
            raise ProviderError(
                "rate_limited" if response.status_code == 429 else "provider_overloaded",
                retryable=True,
                retry_after=delay,
                usage_unknown=True,
            )
        if not response.is_success:
            raise ProviderError("provider_http_error", usage_unknown=True)
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError("malformed_json", usage_unknown=True) from exc
        return validate_provider_payload(request, payload, time.monotonic() - started)

    async def aclose(self) -> None:
        await self._client.aclose()


FakeStep = ProviderResult | dict[str, Any] | Exception | Callable[[QuestionBatch], Any]


class FakeProvider:
    """Scriptable provider used by offline tests and deterministic evaluation."""

    def __init__(
        self,
        steps: list[FakeStep] | None = None,
        *,
        delay_seconds: float = 0.0,
        default_step: FakeStep | None = None,
    ) -> None:
        self.steps: deque[FakeStep] = deque(steps or [])
        self.delay_seconds = delay_seconds
        self.default_step = default_step
        self.requests: list[QuestionBatch] = []
        self.closed = False

    async def evaluate(self, request: QuestionBatch) -> ProviderResult:
        self.requests.append(request)
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if not self.steps and self.default_step is None:
            raise ProviderError("fake_script_exhausted")
        step = self.steps.popleft() if self.steps else self.default_step
        assert step is not None
        if isinstance(step, Exception):
            raise step
        if callable(step):
            value = step(request)
            if inspect.isawaitable(value):
                value = await value
            step = value
        if isinstance(step, ProviderResult):
            return step
        return validate_provider_payload(request, step)

    async def aclose(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class BudgetLimits:
    max_requests: int
    max_tokens: int
    max_cost_usd: float


@dataclass
class UsageBudget:
    attempts: int = 0
    input_tokens: int = 0
    unknown_reserved_tokens: int = 0
    inflight_reserved_tokens: int = 0

    def exposure_tokens(self) -> int:
        return self.input_tokens + self.unknown_reserved_tokens + self.inflight_reserved_tokens


class JevRuntime:
    """Shared concurrency, retry, and usage boundary for all guided agents."""

    PRICE_PER_INPUT_TOKEN = 0.042 / 1_000_000
    RESERVATION_TOKENS = 64_000

    def __init__(
        self,
        provider: DecisionProvider,
        *,
        max_concurrent: int = 2,
        max_retries: int = 1,
        attempt_timeout_seconds: float = 5.0,
        game_limits: BudgetLimits = BudgetLimits(2_000, 5_000_000, 0.25),
        session_limits: BudgetLimits = BudgetLimits(20_000, 50_000_000, 3.0),
        process_limits: BudgetLimits = BudgetLimits(20_000, 50_000_000, 3.0),
    ) -> None:
        self.provider = provider
        self.max_retries = max_retries
        self.attempt_timeout_seconds = attempt_timeout_seconds
        self.game_limits = game_limits
        self.session_limits = session_limits
        self.process_limits = process_limits
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._game_usage: defaultdict[str, UsageBudget] = defaultdict(UsageBudget)
        self._session_usage: defaultdict[str, UsageBudget] = defaultdict(UsageBudget)
        self._process_usage = UsageBudget()
        self._game_failures: defaultdict[str, int] = defaultdict(int)
        self._disabled_games: set[str] = set()
        self.authentication_failed = False

    @staticmethod
    def _fits(usage: UsageBudget, limits: BudgetLimits) -> bool:
        projected_tokens = usage.exposure_tokens() + JevRuntime.RESERVATION_TOKENS
        return (
            usage.attempts + 1 <= limits.max_requests
            and projected_tokens <= limits.max_tokens
            and projected_tokens * JevRuntime.PRICE_PER_INPUT_TOKEN <= limits.max_cost_usd
        )

    def _reserve(self, game_id: str, session_id: str) -> None:
        """Atomically reserve one attempt before it enters the concurrency queue."""
        scopes = (
            (self._game_usage[game_id], self.game_limits),
            (self._session_usage[session_id], self.session_limits),
            (self._process_usage, self.process_limits),
        )
        if not all(self._fits(usage, limits) for usage, limits in scopes):
            self._disabled_games.add(game_id)
            raise ProviderBudgetExhaustedError()
        for usage, _ in scopes:
            usage.attempts += 1
            usage.inflight_reserved_tokens += self.RESERVATION_TOKENS

    def _reconcile(self, game_id: str, session_id: str, *, input_tokens: int | None) -> None:
        """Settle a reservation exactly once without a cancellation point."""
        for usage in (
            self._game_usage[game_id],
            self._session_usage[session_id],
            self._process_usage,
        ):
            usage.inflight_reserved_tokens -= self.RESERVATION_TOKENS
            if input_tokens is None:
                usage.unknown_reserved_tokens += self.RESERVATION_TOKENS
            else:
                usage.input_tokens += input_tokens

    async def evaluate(
        self,
        game_id: str,
        session_id: str,
        request: QuestionBatch,
        call_budget: DecisionCallBudget,
        deadline: float,
    ) -> ProviderResult:
        if self.authentication_failed:
            raise ProviderError("authentication_failed")
        if game_id in self._disabled_games:
            raise ProviderError("game_inference_disabled")
        if len(json.dumps(request.to_dict(), separators=(",", ":"))) > 120_000:
            raise ProviderError("request_too_large")
        last_error: ProviderError | None = None
        for retry in range(self.max_retries + 1):
            if call_budget.attempts >= call_budget.max_attempts:
                raise ProviderBudgetExhaustedError("decision_attempt_budget_exhausted")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderError("decision_deadline_exceeded")
            self._reserve(game_id, session_id)
            call_budget.attempts += 1
            acquired = False
            dispatched = False
            settled = False
            try:
                try:
                    await asyncio.wait_for(self._semaphore.acquire(), timeout=remaining)
                    acquired = True
                except asyncio.TimeoutError:
                    error = ProviderError("decision_deadline_exceeded")
                    self._reconcile(game_id, session_id, input_tokens=0)
                    settled = True
                    last_error = error
                    break

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProviderError("decision_deadline_exceeded")
                deadline_limited = remaining <= self.attempt_timeout_seconds
                timeout = min(self.attempt_timeout_seconds, remaining)
                dispatched = True
                try:
                    async with asyncio.timeout(timeout):
                        result = await self.provider.evaluate(request)
                except asyncio.TimeoutError:
                    error = ProviderError(
                        "decision_deadline_exceeded" if deadline_limited else "timeout",
                        retryable=not deadline_limited,
                        usage_unknown=True,
                    )
                    self._reconcile(game_id, session_id, input_tokens=None)
                    settled = True
                except ProviderError as caught:
                    error = caught
                    known_tokens = caught.usage.input_tokens if caught.usage is not None else None
                    if known_tokens is None and not caught.usage_unknown:
                        known_tokens = 0
                    self._reconcile(game_id, session_id, input_tokens=known_tokens)
                    settled = True
                    if caught.usage is not None:
                        call_budget.input_tokens += caught.usage.input_tokens
                except Exception as exc:
                    self._reconcile(game_id, session_id, input_tokens=None)
                    settled = True
                    error = ProviderError("unexpected_provider_error", usage_unknown=True)
                    error.__cause__ = exc
                else:
                    self._reconcile(
                        game_id, session_id, input_tokens=result.usage.input_tokens
                    )
                    settled = True
                    call_budget.input_tokens += result.usage.input_tokens
                    self._game_failures[game_id] = 0
                    return ProviderResult(
                        model=result.model,
                        answers=result.answers,
                        usage=result.usage,
                        elapsed_seconds=result.elapsed_seconds,
                        request_id=result.request_id,
                        attempts=retry + 1,
                    )
            except asyncio.CancelledError:
                if not settled:
                    self._reconcile(
                        game_id,
                        session_id,
                        input_tokens=None if dispatched else 0,
                    )
                raise
            except ProviderError as caught:
                error = caught
                if not settled:
                    known_tokens = caught.usage.input_tokens if caught.usage is not None else None
                    if known_tokens is None and (not dispatched or not caught.usage_unknown):
                        known_tokens = 0
                    self._reconcile(game_id, session_id, input_tokens=known_tokens)
                    settled = True
                    if caught.usage is not None:
                        call_budget.input_tokens += caught.usage.input_tokens
            finally:
                if acquired:
                    self._semaphore.release()
            last_error = error
            if error.code == "authentication_failed":
                self.authentication_failed = True
            if not error.retryable or retry >= self.max_retries:
                break
            delay = error.retry_after if error.retry_after is not None else min(0.1 * 2**retry, 1.0)
            if delay > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or delay >= remaining:
                    last_error = ProviderError("decision_deadline_exceeded")
                    break
                try:
                    await asyncio.wait_for(asyncio.sleep(delay), timeout=remaining)
                except asyncio.TimeoutError:
                    last_error = ProviderError("decision_deadline_exceeded")
                    break
        self._game_failures[game_id] += 1
        if self._game_failures[game_id] >= 3:
            self._disabled_games.add(game_id)
        assert last_error is not None
        raise last_error

    def _usage_summary(
        self, usage: UsageBudget, *, disabled: bool = False
    ) -> dict[str, int | float | bool]:
        exposure = usage.exposure_tokens()
        return {
            "attempts": usage.attempts,
            "input_tokens": usage.input_tokens,
            "unknown_reserved_tokens": usage.unknown_reserved_tokens,
            "inflight_reserved_tokens": usage.inflight_reserved_tokens,
            "cost_usd": usage.input_tokens * self.PRICE_PER_INPUT_TOKEN,
            "exposure_cost_usd": exposure * self.PRICE_PER_INPUT_TOKEN,
            "disabled": disabled,
        }

    def game_usage(self, game_id: str) -> dict[str, int | float | bool]:
        return self._usage_summary(
            self._game_usage[game_id], disabled=game_id in self._disabled_games
        )

    def session_usage(self, session_id: str) -> dict[str, int | float | bool]:
        return self._usage_summary(self._session_usage[session_id])

    def process_usage(self) -> dict[str, int | float | bool]:
        return self._usage_summary(self._process_usage)

    async def aclose(self) -> None:
        await self.provider.aclose()
