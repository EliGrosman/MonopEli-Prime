#!/usr/bin/env python3
"""Evaluate a trained Monopoly agent.

Usage:
    uv run python scripts/evaluate_agent.py models/ppo_monopoly
    uv run python scripts/evaluate_agent.py models/ppo_monopoly --games 200
    uv run python scripts/evaluate_agent.py models/ppo_monopoly --opponents random rule_based
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained Monopoly agent")

    parser.add_argument(
        "model_path",
        type=str,
        help="Path to trained model",
    )

    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games per opponent (default: 100)",
    )

    parser.add_argument(
        "--opponents",
        type=str,
        nargs="+",
        default=["random", "rule_based"],
        choices=["random", "rule_based", "aggressive", "conservative"],
        help="Opponent types to evaluate against (default: random rule_based)",
    )

    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players (default: 2)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic (non-deterministic) actions",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    return parser.parse_args()


def main() -> None:
    """Main evaluation entry point."""
    args = parse_args()

    # Check model exists
    model_path = Path(args.model_path)
    if not model_path.exists() and not Path(f"{args.model_path}.zip").exists():
        print(f"Error: Model not found: {args.model_path}")
        sys.exit(1)

    print("=" * 60)
    print("Monopoly Agent Evaluation")
    print("=" * 60)
    print(f"\nModel: {args.model_path}")
    print(f"Games per opponent: {args.games}")
    print(f"Opponents: {', '.join(args.opponents)}")
    print(f"Players: {args.num_players}")
    print(f"Mode: {'Stochastic' if args.stochastic else 'Deterministic'}")

    try:
        from training import evaluate_agent
    except ImportError:
        print("\nError: Training dependencies not installed.")
        print("Install with: uv pip install stable-baselines3 sb3-contrib torch")
        sys.exit(1)

    print("\nRunning evaluation...")
    summary = evaluate_agent(
        model_path=args.model_path,
        num_games=args.games,
        opponents=args.opponents,
        num_players=args.num_players,
        seed=args.seed,
        deterministic=not args.stochastic,
        verbose=not args.quiet,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(summary)

    # Check against targets
    print("\n" + "-" * 60)
    print("Target Comparison:")

    random_result = summary.results.get("random")
    if random_result:
        target = 0.90
        status = "✓" if random_result.win_rate >= target else "✗"
        print(f"  {status} vs Random: {random_result.win_rate:.1%} (target: {target:.0%})")

    rulebased_result = summary.results.get("rule_based")
    if rulebased_result:
        target = 0.70
        status = "✓" if rulebased_result.win_rate >= target else "✗"
        print(f"  {status} vs RuleBased: {rulebased_result.win_rate:.1%} (target: {target:.0%})")


if __name__ == "__main__":
    main()
