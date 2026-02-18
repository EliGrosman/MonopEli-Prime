#!/usr/bin/env python3
"""Evaluate the hybrid MCTS+LLM agent against baselines.

Runs games with HybridAgent (player 0) against configurable opponents
and reports win rate, trade metrics, game length, and token usage.

Usage:
    uv run python scripts/evaluate_hybrid.py --games 10
    uv run python scripts/evaluate_hybrid.py --opponents random rule_based --games 50
    uv run python scripts/evaluate_hybrid.py --llm ollama --llm-model gemma3:4b --games 10
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import Agent
from agents.hybrid_agent import HybridAgent, HybridAgentConfig
from agents.mcts_agent import MCTSAgent
from agents.random_agent import RandomAgent
from agents.rule_based import AggressiveAgent, ConservativeAgent, RuleBasedAgent
from mcts.llm.budget import TokenBudget
from mcts.llm.client import LLMClient, LLMConfig, create_client
from mcts.negotiation import NegotiationManager, NegotiationStatus
from monopoly_engine.game import MonopolyGame
from monopoly_gym.action_space import ActionEncoder

# ---------------------------------------------------------------------------
# Fake LLM client for evaluation (no real API calls)
# ---------------------------------------------------------------------------

class _EvalLLMClient(LLMClient):
    """LLM client that returns simple heuristic responses for evaluation.

    Returns ``no_trade`` for proposals and ``reject`` for evaluations,
    so the MCTS verifier drives all trade decisions.
    """

    def __init__(self) -> None:
        super().__init__(LLMConfig())
        self.call_count = 0
        self.total_tokens = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        self.total_tokens += 50  # Approximate
        return '{"no_trade": true, "reasoning": "eval mode"}'

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.call_count += 1
        self.total_tokens += 50
        # For trade proposals: suggest a simple trade if possible
        if "propose" in user_prompt.lower() or "suggest" in user_prompt.lower():
            return {"no_trade": True, "reasoning": "eval mode"}
        # For trade evaluation: reject to be conservative
        return {"decision": "reject", "reasoning": "eval mode"}

    def close(self) -> None:
        pass


class _TrackingLLMClient(LLMClient):
    """Wraps a real LLM client and tracks call count / approximate tokens."""

    def __init__(self, inner: LLMClient) -> None:
        super().__init__(inner.config)
        self._inner = inner
        self.call_count = 0
        self.total_tokens = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        result = self._inner.complete(system_prompt, user_prompt)
        # Approximate token count from character length
        self.total_tokens += (len(system_prompt) + len(user_prompt) + len(result)) // 4
        return result

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.call_count += 1
        result = self._inner.complete_json(system_prompt, user_prompt, schema)
        self.total_tokens += (len(system_prompt) + len(user_prompt)) // 4 + 50
        return result

    def close(self) -> None:
        self._inner.close()


def _make_llm_client(
    llm_provider: str | None,
    llm_model: str | None,
) -> _EvalLLMClient | _TrackingLLMClient:
    """Create an LLM client based on CLI args."""
    if llm_provider is None:
        return _EvalLLMClient()

    config = LLMConfig(
        provider=llm_provider,
        model=llm_model or ("gemma3:4b" if llm_provider == "ollama" else ""),
        temperature=0.3,
        max_tokens=512,
        timeout_seconds=30.0,
    )
    inner = create_client(config)
    return _TrackingLLMClient(inner)


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class GameResult:
    """Result of a single evaluation game."""

    winner: int | None
    action_count: int
    elapsed_sec: float
    trades_proposed: int
    trades_accepted: int
    tokens_used: int


@dataclass
class EvalResults:
    """Aggregated evaluation results for one matchup."""

    agent_name: str
    opponent_name: str
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_trades_proposed: int = 0
    total_trades_accepted: int = 0
    total_actions: int = 0
    total_tokens: int = 0
    total_time_sec: float = 0.0
    game_results: list[GameResult] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.games_played if self.games_played > 0 else 0.0

    @property
    def avg_trades_proposed(self) -> float:
        return (
            self.total_trades_proposed / self.games_played
            if self.games_played > 0
            else 0.0
        )

    @property
    def avg_trades_accepted(self) -> float:
        return (
            self.total_trades_accepted / self.games_played
            if self.games_played > 0
            else 0.0
        )

    @property
    def trade_acceptance_rate(self) -> float:
        return (
            self.total_trades_accepted / self.total_trades_proposed
            if self.total_trades_proposed > 0
            else 0.0
        )

    @property
    def avg_game_length(self) -> float:
        return (
            self.total_actions / self.games_played
            if self.games_played > 0
            else 0.0
        )

    @property
    def avg_tokens_used(self) -> float:
        return (
            self.total_tokens / self.games_played
            if self.games_played > 0
            else 0.0
        )


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def _create_opponent(opp_type: str, player_id: int) -> Agent:
    """Create an opponent agent by type name."""
    factories: dict[str, type[Agent]] = {
        "random": RandomAgent,
        "rule_based": RuleBasedAgent,
        "aggressive": AggressiveAgent,
        "conservative": ConservativeAgent,
    }
    if opp_type == "mcts":
        return MCTSAgent(player_id=player_id, num_simulations=50)
    cls = factories.get(opp_type)
    if cls is None:
        raise ValueError(f"Unknown opponent type: {opp_type}")
    return cls(player_id=player_id)


def play_eval_game(
    hybrid_agent: HybridAgent,
    opponents: list[Agent],
    llm_client: _EvalLLMClient | _TrackingLLMClient,
    max_turns: int = 500,
    seed: int | None = None,
    enable_trades: bool = False,
) -> GameResult:
    """Play a single evaluation game.

    HybridAgent is always player 0.
    """
    num_players = 1 + len(opponents)
    game = MonopolyGame(num_players=num_players, seed=seed)
    encoder = ActionEncoder(enable_trades=enable_trades)

    # Reset agents
    hybrid_agent.reset()
    for opp in opponents:
        opp.reset()

    # Reset LLM call counter for this game
    tokens_before = llm_client.total_tokens

    agents: dict[int, Agent] = {0: hybrid_agent}
    for i, opp in enumerate(opponents):
        agents[i + 1] = opp

    action_count = 0
    t0 = time.monotonic()

    while not game.game_over and action_count < max_turns:
        pid = game.current_player
        agent = agents[pid]
        mask = encoder.get_action_mask(game, pid)
        obs: dict[str, Any] = {}

        action_idx = agent.choose_action(obs, mask, game)
        action = encoder.decode(action_idx, pid, game)
        valid, _ = action.validate(game)
        if valid:
            action.execute(game)

        action_count += 1

    elapsed = time.monotonic() - t0

    # Gather trade metrics from negotiation manager
    mgr = hybrid_agent.negotiation_manager
    accepted = sum(
        1
        for n in mgr._negotiations.values()
        if n.status == NegotiationStatus.ACCEPTED
    )

    tokens_this_game = llm_client.total_tokens - tokens_before

    return GameResult(
        winner=game.winner,
        action_count=action_count,
        elapsed_sec=elapsed,
        trades_proposed=hybrid_agent.trades_proposed,
        trades_accepted=accepted,
        tokens_used=tokens_this_game,
    )


def run_evaluation(
    opponent_type: str,
    num_games: int,
    num_players: int = 4,
    mcts_simulations: int = 50,
    max_turns: int = 500,
    seed: int = 42,
    verbose: bool = True,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    enable_trades: bool = False,
) -> EvalResults:
    """Run evaluation of HybridAgent against a specific opponent type."""
    results = EvalResults(
        agent_name="HybridAgent",
        opponent_name=opponent_type,
    )

    for game_idx in range(num_games):
        game_seed = seed + game_idx

        # Create fresh agents for each game
        llm_client = _make_llm_client(llm_provider, llm_model)
        config = HybridAgentConfig(
            mcts_simulations=mcts_simulations,
            trade_eval_simulations=10,  # Faster for eval
            trade_check_interval=3,
            token_budget=TokenBudget(max_calls_per_game=10),
        )
        negotiation_mgr = NegotiationManager(max_rounds=3)
        hybrid_agent = HybridAgent(
            player_id=0,
            config=config,
            negotiation_manager=negotiation_mgr,
            llm_client=llm_client,
        )

        opponents = [
            _create_opponent(opponent_type, i + 1)
            for i in range(num_players - 1)
        ]

        game_result = play_eval_game(
            hybrid_agent,
            opponents,
            llm_client,
            max_turns=max_turns,
            seed=game_seed,
            enable_trades=enable_trades,
        )

        # Aggregate
        results.games_played += 1
        if game_result.winner == 0:
            results.wins += 1
        elif game_result.winner is None:
            results.draws += 1
        else:
            results.losses += 1

        results.total_trades_proposed += game_result.trades_proposed
        results.total_trades_accepted += game_result.trades_accepted
        results.total_actions += game_result.action_count
        results.total_tokens += game_result.tokens_used
        results.total_time_sec += game_result.elapsed_sec
        results.game_results.append(game_result)

        if verbose:
            w = "WIN" if game_result.winner == 0 else (
                "DRAW" if game_result.winner is None else "LOSS"
            )
            print(
                f"  Game {game_idx + 1:3d}/{num_games}: {w:4s} "
                f"| {game_result.action_count:4d} actions "
                f"| {game_result.trades_proposed} trades "
                f"| {game_result.elapsed_sec:.1f}s",
            )

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results_table(all_results: list[EvalResults]) -> None:
    """Print evaluation results as a formatted table."""
    print("\n" + "=" * 80)
    print(f"{'Opponent':<15} {'Games':>6} {'Win%':>7} {'Trades':>7} "
          f"{'Accept%':>8} {'AvgLen':>7} {'Tokens':>7} {'Time':>7}")
    print("-" * 80)

    for r in all_results:
        print(
            f"{r.opponent_name:<15} {r.games_played:>6} "
            f"{r.win_rate:>6.1%} "
            f"{r.avg_trades_proposed:>7.1f} "
            f"{r.trade_acceptance_rate:>7.1%} "
            f"{r.avg_game_length:>7.0f} "
            f"{r.avg_tokens_used:>7.0f} "
            f"{r.total_time_sec:>6.1f}s",
        )

    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate HybridAgent (MCTS+LLM) against baselines",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        nargs="+",
        default=["random", "rule_based"],
        choices=["random", "rule_based", "aggressive", "conservative", "mcts"],
        help="Opponent types (default: random rule_based)",
    )
    parser.add_argument(
        "--games", type=int, default=10,
        help="Games per opponent (default: 10)",
    )
    parser.add_argument(
        "--num-players", type=int, default=4,
        help="Number of players (default: 4)",
    )
    parser.add_argument(
        "--mcts-sims", type=int, default=50,
        help="MCTS simulations per move (default: 50)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=500,
        help="Max actions per game (default: 500)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-game output",
    )
    parser.add_argument(
        "--llm", type=str, default=None,
        choices=["ollama", "claude", "openai"],
        help="LLM provider for trade generation (default: fake/no-op client)",
    )
    parser.add_argument(
        "--llm-model", type=str, default=None,
        help="LLM model name (default: gemma3:4b for ollama)",
    )
    parser.add_argument(
        "--enable-trades", action="store_true",
        help="Enable 1-for-1 property trading for heuristic agents",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    llm_label = f"{args.llm} ({args.llm_model or 'default'})" if args.llm else "fake (no-op)"

    print("=" * 60)
    print("HybridAgent (MCTS + LLM) Evaluation")
    print("=" * 60)
    print(f"Games per opponent: {args.games}")
    print(f"Opponents: {', '.join(args.opponents)}")
    print(f"Players: {args.num_players}")
    print(f"MCTS simulations: {args.mcts_sims}")
    print(f"LLM: {llm_label}")
    print(f"Max turns: {args.max_turns}")
    print(f"Trades enabled: {args.enable_trades}")
    print()

    all_results: list[EvalResults] = []

    for opp_type in args.opponents:
        print(f"\n--- vs {opp_type} ---")
        result = run_evaluation(
            opponent_type=opp_type,
            num_games=args.games,
            num_players=args.num_players,
            mcts_simulations=args.mcts_sims,
            max_turns=args.max_turns,
            seed=args.seed,
            verbose=not args.quiet,
            llm_provider=args.llm,
            llm_model=args.llm_model,
            enable_trades=args.enable_trades,
        )
        all_results.append(result)

    print_results_table(all_results)


if __name__ == "__main__":
    main()
