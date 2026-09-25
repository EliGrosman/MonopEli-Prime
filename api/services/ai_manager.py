"""Server-side native and guided AI orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from typing import TYPE_CHECKING, Any

from agents.base import Agent
from agents.random_agent import RandomAgent
from agents.rule_based import AggressiveAgent, ConservativeAgent, RuleBasedAgent
from monopoly_engine.progress import ProgressGuard
from monopoly_gym.action_space import ActionEncoder

from ..config import Settings, get_settings

if TYPE_CHECKING:
    from agents.jev import DecisionProvider, GuidedJevAgent, JevRuntime
    from monopoly_engine import DecisionView, MonopolyGame

    from .game_manager import GameManager


AI_TYPES: dict[str, type[Agent]] = {
    "random": RandomAgent,
    "rule_based": RuleBasedAgent,
    "aggressive": AggressiveAgent,
    "conservative": ConservativeAgent,
}
TRADING_AI_TYPES = {
    "trading_random": RandomAgent,
    "trading_rule_based": RuleBasedAgent,
    "trading_aggressive": AggressiveAgent,
    "trading_conservative": ConservativeAgent,
}


class AIManager:
    """Own agents and run at most one tracked decision loop per game."""

    def __init__(
        self,
        think_delay_ms: int | None = None,
        *,
        settings: Settings | None = None,
        provider: DecisionProvider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.think_delay_ms = (
            think_delay_ms
            if think_delay_ms is not None
            else self.settings.ai_think_delay_ms
        )
        self._agents: dict[tuple[str, int], Agent | GuidedJevAgent] = {}
        self._agent_types: dict[tuple[str, int], str] = {}
        self._generations: dict[tuple[str, int], int] = {}
        self._guards: dict[str, ProgressGuard] = {}
        self._processing: set[str] = set()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._runtime: JevRuntime | None = None
        if provider is not None:
            self._runtime = self._build_runtime(provider)

    def _build_runtime(self, provider: DecisionProvider) -> JevRuntime:
        from agents.jev import JevRuntime
        from agents.jev.provider import BudgetLimits

        process_limits = BudgetLimits(
            self.settings.jev_process_max_requests,
            self.settings.jev_process_max_input_tokens,
            self.settings.jev_process_max_cost_usd,
        )
        return JevRuntime(
            provider,
            max_concurrent=self.settings.jev_max_concurrent_requests,
            max_retries=self.settings.jev_max_retries,
            attempt_timeout_seconds=self.settings.jev_attempt_timeout_seconds,
            game_limits=BudgetLimits(
                self.settings.jev_game_max_requests,
                self.settings.jev_game_max_input_tokens,
                self.settings.jev_game_max_cost_usd,
            ),
            session_limits=process_limits,
            process_limits=process_limits,
        )

    def _ensure_runtime(self) -> JevRuntime:
        if self._runtime is not None:
            return self._runtime
        if not self.settings.jev_enabled:
            raise ValueError("Guided Jev is disabled by server configuration")
        api_key = (
            self.settings.typesafe_api_key.get_secret_value()
            if self.settings.typesafe_api_key is not None
            else ""
        )
        if not api_key:
            raise ValueError("Guided Jev is unavailable: TYPESAFE_API_KEY is not configured")
        from agents.jev import TypeSafeProvider

        provider = TypeSafeProvider(
            api_key,
            timeout_seconds=self.settings.jev_attempt_timeout_seconds,
            connect_timeout_seconds=self.settings.jev_connect_timeout_seconds,
        )
        self._runtime = self._build_runtime(provider)
        return self._runtime

    @property
    def jev_available(self) -> bool:
        if self._runtime is not None:
            return not self._runtime.authentication_failed
        return self.settings.jev_enabled and self.settings.typesafe_api_key is not None

    def create_agent(
        self,
        game_id: str,
        player_id: int,
        ai_type: str = "rule_based",
        seed: int | None = None,
        *,
        session_budget_id: str | None = None,
    ) -> Agent | GuidedJevAgent:
        key = (game_id, player_id)
        generation = self._generations.get(key, 0) + 1
        self._generations[key] = generation
        if ai_type == "jev":
            from agents.jev import GuidedJevAgent

            agent = GuidedJevAgent(
                player_id,
                game_id,
                self._ensure_runtime(),
                model=self.settings.jev_model,
                session_budget_id=session_budget_id,
            )
        else:
            agent_cls = AI_TYPES.get(ai_type) or TRADING_AI_TYPES.get(ai_type)
            if agent_cls is None:
                valid = list(AI_TYPES) + list(TRADING_AI_TYPES) + ["jev"]
                raise ValueError(f"Unknown AI type: {ai_type}. Valid types: {valid}")
            base = (
                agent_cls(player_id=player_id, seed=seed)
                if agent_cls is RandomAgent
                else agent_cls(player_id=player_id)
            )
            if ai_type in TRADING_AI_TYPES:
                from agents.trading_agent import TradingAgent

                agent = TradingAgent(base, response="mutual")
            else:
                agent = base
        self._agents[key] = agent
        self._agent_types[key] = ai_type
        return agent

    def get_agent(self, game_id: str, player_id: int) -> Agent | GuidedJevAgent | None:
        return self._agents.get((game_id, player_id))

    def remove_agent(self, game_id: str, player_id: int) -> bool:
        key = (game_id, player_id)
        if key not in self._agents:
            return False
        del self._agents[key]
        self._agent_types.pop(key, None)
        self._generations[key] = self._generations.get(key, 0) + 1
        task = self._tasks.get(game_id)
        if task is not None:
            task.cancel()
        return True

    def remove_game_agents(self, game_id: str) -> int:
        self._guards.pop(game_id, None)
        task = self._tasks.pop(game_id, None)
        if task is not None:
            task.cancel()
        keys = [key for key in self._agents if key[0] == game_id]
        for key in keys:
            del self._agents[key]
            self._agent_types.pop(key, None)
            self._generations[key] = self._generations.get(key, 0) + 1
        return len(keys)

    def is_ai_player(self, game_id: str, player_id: int) -> bool:
        return (game_id, player_id) in self._agents

    def schedule_ai_turns(self, game_manager: GameManager, game_id: str) -> None:
        existing = self._tasks.get(game_id)
        if existing is not None and not existing.done():
            return
        task = asyncio.create_task(self.process_ai_turns_for_game(game_manager, game_id))
        self._tasks[game_id] = task

        def cleanup(completed: asyncio.Task[None]) -> None:
            if self._tasks.get(game_id) is completed:
                self._tasks.pop(game_id, None)
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(cleanup)

    async def _publish_inspection(
        self, game_manager: GameManager, game_id: str, agent: GuidedJevAgent
    ) -> None:
        await game_manager.broadcast_agent_update(game_id, agent.inspection.to_dict())

    async def _process_jev_decision(
        self,
        game_manager: GameManager,
        game_id: str,
        player_id: int,
        agent: GuidedJevAgent,
        generation: int,
    ) -> tuple[bool, str]:
        view = await game_manager.capture_decision(game_id, player_id)
        if view is None:
            return False, "stale_capture"
        agent.inspection.status = "thinking"
        agent.inspection.basis_revision = view.revision
        agent.inspection.sequence += 1
        agent.inspection.fallback_reason = None
        await self._publish_inspection(game_manager, game_id, agent)
        outcome = await agent.decide(view)
        if outcome.command is None:
            agent.inspection.status = "idle"
            agent.inspection.latest_summary = outcome.summary
            agent.inspection.sequence += 1
            await self._publish_inspection(game_manager, game_id, agent)
            return False, "no_command"
        request_id = (
            f"jev:{game_id}:{player_id}:{generation}:{view.revision}:{uuid.uuid4().hex}"
        )
        key = (game_id, player_id)
        metadata = {
            "source": outcome.source,
            "fallback_reason": outcome.fallback_reason,
            "summary": outcome.summary,
            "provider_model": outcome.provider_model,
            "answer_reference": outcome.answer_reference,
            "prompt_protocol": "guided-jev-v1",
            "attempts": outcome.attempts,
            "input_tokens": outcome.input_tokens,
            "latency_seconds": outcome.elapsed_seconds,
            "memory_version": outcome.memory_version,
            "memory_hash": hashlib.sha256(
                json.dumps(agent.memory.public_dict(), sort_keys=True).encode()
            ).hexdigest()[:16],
            "reported_cost_usd": outcome.input_tokens * (0.042 / 1_000_000),
            "usage": self._runtime.game_usage(game_id) if self._runtime else None,
        }
        success, message = await game_manager.execute_action(
            game_id,
            outcome.command,
            expected_revision=view.revision,
            request_id=request_id,
            commit_guard=lambda: (
                self._agents.get(key) is agent and self._generations.get(key) == generation
            ),
            action_metadata=metadata,
            post_commit=lambda revision, turn_number: agent.commit_outcome(
                outcome, revision, turn_number
            ),
        )
        if success:
            await self._publish_inspection(game_manager, game_id, agent)
            return True, ""
        agent.inspection.status = "error"
        agent.inspection.latest_summary = "Discarded a stale or invalid decision"
        agent.inspection.fallback_reason = (
            "stale_result" if "Stale" in message else "commit_failed"
        )
        agent.inspection.sequence += 1
        await self._publish_inspection(game_manager, game_id, agent)
        return False, message

    async def process_ai_turn(
        self,
        game_manager: GameManager,
        game_id: str,
        player_id: int,
        max_actions: int = 100,
    ) -> int:
        key = (game_id, player_id)
        agent = self._agents.get(key)
        if agent is None:
            return 0
        actions_taken = 0
        stale_attempts = 0
        while actions_taken < max_actions:
            active_game = await game_manager.get_game(game_id)
            if active_game is None:
                break
            game = active_game.game
            if game.decision_player != player_id or game.game_over:
                break
            if self.think_delay_ms > 0:
                await asyncio.sleep(self.think_delay_ms / 1000)
            if self._agent_types.get(key) == "jev":
                from agents.jev import GuidedJevAgent

                assert isinstance(agent, GuidedJevAgent)
                success, message = await self._process_jev_decision(
                    game_manager,
                    game_id,
                    player_id,
                    agent,
                    self._generations[key],
                )
                if not success:
                    if message.startswith("Stale") or message in {
                        "stale_capture",
                        "AI agent was replaced or removed",
                    }:
                        stale_attempts += 1
                        if stale_attempts < 3:
                            continue
                        break
                    if message == "no_command":
                        break
                    raise RuntimeError(f"AI action failed: {message}")
                actions_taken += 1
                continue
            encoder = ActionEncoder(rules_id=game.rules_id)
            mask = encoder.get_action_mask(game, player_id)
            if not mask.any():
                raise RuntimeError("Live AI decision has no legal actions")
            self._guards.setdefault(game_id, ProgressGuard()).check(game)
            revision = game.state.revision
            if game.state.phase == "trade_response" and not hasattr(
                agent, "choose_native_action"
            ):
                from monopoly_engine import RejectTrade

                action = RejectTrade(player_id, next(iter(game.state.pending_trades)))
            elif hasattr(agent, "choose_native_action"):
                action = agent.choose_native_action(game, encoder)
            else:
                action = agent.choose_decision(game.decision_view(player_id))
            success, message = await game_manager.execute_action(
                game_id,
                action,
                expected_revision=revision,
                action_metadata={"source": "native", "summary": type(action).__name__},
            )
            if not success:
                raise RuntimeError(f"AI action failed: {message}")
            actions_taken += 1
        return actions_taken

    async def process_ai_turns_for_game(
        self, game_manager: GameManager, game_id: str
    ) -> None:
        if game_id in self._processing:
            return
        self._processing.add(game_id)
        try:
            while True:
                active_game = await game_manager.get_game(game_id)
                if active_game is None or active_game.game.game_over:
                    break
                player_id = active_game.game.decision_player
                if not self.is_ai_player(game_id, player_id):
                    break
                count = await self.process_ai_turn(game_manager, game_id, player_id)
                if count == 0:
                    latest = await game_manager.get_game(game_id)
                    if (
                        latest is None
                        or latest.game.game_over
                        or not self.is_ai_player(game_id, latest.game.decision_player)
                    ):
                        break
                    if latest.game.decision_player != player_id:
                        continue
                    raise RuntimeError("AI made no progress")
                await asyncio.sleep(0.1)
        except Exception:
            active_game = await game_manager.get_game(game_id)
            if active_game is not None:
                player_id = active_game.game.decision_player
                candidate = self._agents.get((game_id, player_id))
                from agents.jev import GuidedJevAgent

                if isinstance(candidate, GuidedJevAgent):
                    candidate.inspection.status = "error"
                    candidate.inspection.latest_summary = "Agent orchestration failed"
                    candidate.inspection.fallback_reason = "orchestration_error"
                    candidate.inspection.sequence += 1
                    await self._publish_inspection(
                        game_manager, game_id, candidate
                    )
            raise
        finally:
            self._processing.discard(game_id)

    def observe_transition(
        self,
        game_id: str,
        before: DecisionView,
        after: DecisionView,
        structured_events: list[dict[str, Any]],
    ) -> None:
        from agents.jev import GuidedJevAgent

        for (candidate_game, _), agent in self._agents.items():
            if candidate_game == game_id and isinstance(agent, GuidedJevAgent):
                agent.observe_transition(before, after, structured_events)

    def public_inspections(self, game_id: str) -> dict[int, dict[str, Any]]:
        from agents.jev import GuidedJevAgent

        return {
            player_id: agent.inspection.to_dict()
            for (candidate_game, player_id), agent in self._agents.items()
            if candidate_game == game_id and isinstance(agent, GuidedJevAgent)
        }

    def usage_for_game(self, game_id: str) -> dict[str, int | float | bool] | None:
        return self._runtime.game_usage(game_id) if self._runtime is not None else None

    def _get_observation(self, game: MonopolyGame, player_id: int) -> dict[str, Any]:
        state = game.state.to_dict()
        return {
            "players": state["players"],
            "properties": state["properties"],
            "current_player": state["current_player"],
            "turn_number": state["turn_number"],
            "houses_remaining": state["houses_remaining"],
            "hotels_remaining": state["hotels_remaining"],
            "player_id": player_id,
        }

    def get_ai_type_for_agent(self, game_id: str, player_id: int) -> str | None:
        return self._agent_types.get((game_id, player_id))

    def agent_count(self) -> int:
        return len(self._agents)

    async def shutdown(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        if self._runtime is not None:
            await self._runtime.aclose()
