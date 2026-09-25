"""Typed contracts shared by the guided policy and provider adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from monopoly_engine import Action

DecisionSource = Literal["jev", "forced", "fallback"]


@dataclass(frozen=True)
class QuestionBatch:
    """One TypeSafe System One request."""

    state: dict[str, Any]
    questions: dict[str, dict[str, Any]]
    model: str = "jev-1.13.0"

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state, "model": self.model, "questions": self.questions}


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    probabilities: dict[str, float]
    confidence: float
    legend: dict[str, str]


@dataclass(frozen=True)
class NoulAnswer:
    noul: float


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer


@dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ProviderResult:
    model: str
    answers: dict[str, Answer]
    usage: ProviderUsage
    elapsed_seconds: float = 0.0
    request_id: str | None = None
    attempts: int = 1


@dataclass
class DecisionCallBudget:
    """Per-decision accounting; attempts include requests admitted while queued."""

    max_attempts: int = 14
    attempts: int = 0
    input_tokens: int = 0


@dataclass(frozen=True)
class StrategyUpdate:
    objective_id: str
    short_term_objective: str
    long_term_objective: str
    target_group: str | None
    cash_reserve_target: int
    revision: int
    own_turn: int
    public_fingerprint: str


@dataclass(frozen=True)
class DecisionOutcome:
    command: Action | None
    source: DecisionSource
    summary: str
    fallback_reason: str | None = None
    staged_strategy: StrategyUpdate | None = None
    provider_model: str | None = None
    answer_reference: str | None = None
    attempts: int = 0
    input_tokens: int = 0
    elapsed_seconds: float = 0.0
    memory_version: int = 0


@dataclass
class AgentInspection:
    player_id: int
    sequence: int = 0
    basis_revision: int = 0
    status: Literal["idle", "thinking", "fallback", "error", "disabled"] = "idle"
    short_term_objective: str = "Assess the board"
    long_term_objective: str = "Build a sustainable property position"
    cash_reserve_target: int = 200
    latest_summary: str = "Waiting for the first decision"
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TradeMemoryEntry:
    revision: int
    trade_id: int
    role: Literal["proposer", "recipient"]
    outcome: Literal["proposed", "accepted", "rejected"]
    from_player: int
    to_player: int
    give_properties: tuple[int, ...]
    give_money: int
    want_properties: tuple[int, ...]
    want_money: int


@dataclass
class StrategyMemory:
    strategy_version: int = 0
    objective_id: str = "assess"
    short_term_objective: str = "Assess the board"
    long_term_objective: str = "Build a sustainable property position"
    target_group: str | None = None
    cash_reserve_target: int = 200
    last_strategy_revision: int = -1
    last_strategy_own_turn: int = -1
    observed_public_fingerprint: str = ""
    recent_trades: list[TradeMemoryEntry] = field(default_factory=list)

    def public_dict(self) -> dict[str, Any]:
        return {
            "strategy_version": self.strategy_version,
            "objective_id": self.objective_id,
            "short_term_objective": self.short_term_objective,
            "long_term_objective": self.long_term_objective,
            "target_group": self.target_group,
            "cash_reserve_target": self.cash_reserve_target,
            "recent_trades": [asdict(item) for item in self.recent_trades],
        }
