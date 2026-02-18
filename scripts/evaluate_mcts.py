#!/usr/bin/env python3
"""Evaluate the MCTSAgent against baselines.

Runs games with MCTSAgent (player 0) against configurable opponents
and reports win rate, game length, and timing. Supports multiple
simulation budgets for scaling analysis.

Usage:
    uv run python scripts/evaluate_mcts.py --games 50
    uv run python scripts/evaluate_mcts.py --opponents random rule_based aggressive conservative --games 100
    uv run python scripts/evaluate_mcts.py --simulations 50 100 200 400 --games 50
    uv run python scripts/evaluate_mcts.py --network models/mcts_best/ --games 100
    uv run python scripts/evaluate_mcts.py --json results.json --games 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcts.eval import MCTSEvalResult


def print_results_table(
    all_results: list[tuple[int, dict[str, MCTSEvalResult]]],
) -> None:
    """Print evaluation results as a formatted table."""
    print("\n" + "=" * 90)
    print(
        f"{'Sims':>6} {'Opponent':<15} {'Games':>6} {'Win%':>7} "
        f"{'W/L/D':>11} {'AvgLen':>7} {'ms/move':>8} {'Time':>7}"
    )
    print("-" * 90)

    for sims, results_by_opp in all_results:
        for opp_type, result in results_by_opp.items():
            wld = f"{result.wins}W/{result.losses}L/{result.draws}D"
            print(
                f"{sims:>6} {opp_type:<15} {result.num_games:>6} "
                f"{result.win_rate:>6.1%} "
                f"{wld:>11} "
                f"{result.avg_game_length:>7.0f} "
                f"{result.avg_mcts_time_per_move * 1000:>7.1f} "
                f"{result.total_time_sec:>6.1f}s"
            )

    print("=" * 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MCTSAgent against baselines",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        nargs="+",
        default=["random", "rule_based"],
        choices=["random", "rule_based", "aggressive", "conservative"],
        help="Opponent types (default: random rule_based)",
    )
    parser.add_argument(
        "--games", type=int, default=50,
        help="Games per opponent per simulation budget (default: 50)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        nargs="+",
        default=[200],
        help="MCTS simulation budgets to test (default: 200). "
             "Pass multiple values for scaling analysis, e.g. --simulations 50 100 200 400",
    )
    parser.add_argument(
        "--num-players", type=int, default=4,
        help="Number of players (default: 4)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=500,
        help="Max actions per game (default: 500)",
    )
    parser.add_argument(
        "--network", type=str, default=None,
        help="Path to ValueNetwork checkpoint (default: None = random rollouts)",
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
        "--json", type=str, default=None,
        metavar="FILE",
        help="Write results to JSON file for later analysis",
    )
    parser.add_argument(
        "--enable-trades", action="store_true",
        help="Enable 1-for-1 property trading for heuristic agents",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("MCTSAgent Evaluation")
    print("=" * 60)
    print(f"Games per matchup:  {args.games}")
    print(f"Opponents:          {', '.join(args.opponents)}")
    print(f"Simulation budgets: {args.simulations}")
    print(f"Players:            {args.num_players}")
    print(f"Network:            {args.network or 'None (random rollouts)'}")
    print(f"Max turns:          {args.max_turns}")
    print(f"Trades enabled:     {args.enable_trades}")
    print()

    # We call evaluate_mcts_agent for each simulation budget.
    # It returns dict[str, dict[str, float]] but we also want the
    # MCTSEvalResult objects for the table. We'll call the lower-level
    # functions directly.
    from agents.mcts_agent import MCTSAgent
    from mcts.eval import MCTSEvalResult, _make_opponent, play_evaluation_game

    all_results: list[tuple[int, dict[str, MCTSEvalResult]]] = []
    json_data: dict[str, Any] = {
        "config": {
            "games": args.games,
            "opponents": args.opponents,
            "simulations": args.simulations,
            "num_players": args.num_players,
            "network": args.network,
            "max_turns": args.max_turns,
            "seed": args.seed,
            "enable_trades": args.enable_trades,
        },
        "results": [],
    }

    total_t0 = time.monotonic()

    for sims in args.simulations:
        print(f"\n{'='*60}")
        print(f"Simulation budget: {sims}")
        print(f"{'='*60}")

        mcts_agent = MCTSAgent(
            player_id=0,
            num_simulations=sims,
            temperature=0.0,
            network_path=args.network,
        )

        results_by_opp: dict[str, MCTSEvalResult] = {}

        for opp_type in args.opponents:
            print(f"\n--- {sims} sims vs {opp_type} ---")

            wins = 0
            losses = 0
            draws = 0
            total_actions = 0
            total_time = 0.0
            num_opponents = args.num_players - 1

            for game_idx in range(args.games):
                game_seed = (args.seed + game_idx) if args.seed is not None else None
                opp_agents = [
                    _make_opponent(opp_type, player_id=i + 1)
                    for i in range(num_opponents)
                ]

                winner, actions, elapsed = play_evaluation_game(
                    mcts_agent, opp_agents,
                    max_turns=args.max_turns, seed=game_seed,
                    enable_trades=args.enable_trades,
                )

                if winner == 0:
                    wins += 1
                elif winner is None:
                    draws += 1
                else:
                    losses += 1

                total_actions += actions
                total_time += elapsed

                if not args.quiet:
                    outcome = "WIN " if winner == 0 else (
                        "DRAW" if winner is None else "LOSS"
                    )
                    print(
                        f"  Game {game_idx + 1:3d}/{args.games}: {outcome} "
                        f"| {actions:4d} actions | {elapsed:.1f}s"
                    )

            result = MCTSEvalResult(
                opponent_type=opp_type,
                num_games=args.games,
                wins=wins,
                losses=losses,
                draws=draws,
                win_rate=wins / args.games,
                avg_game_length=total_actions / args.games,
                avg_time_sec=total_time / args.games,
                total_time_sec=total_time,
                avg_mcts_time_per_move=mcts_agent.search_stats.avg_time_per_move,
            )
            results_by_opp[opp_type] = result
            print(f"  {result}")

            json_data["results"].append({
                "simulations": sims,
                **result.to_dict(),
                "opponent_type": opp_type,
            })

        all_results.append((sims, results_by_opp))

    total_elapsed = time.monotonic() - total_t0

    print_results_table(all_results)
    print(f"\nTotal evaluation time: {total_elapsed:.1f}s")

    if args.json:
        json_data["total_time_sec"] = total_elapsed
        with open(args.json, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"Results written to {args.json}")


if __name__ == "__main__":
    main()
