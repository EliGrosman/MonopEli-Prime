"""CLI for behavioral cloning training on expert demonstrations.

Trains a MaskablePPO-compatible policy network to imitate expert actions,
then evaluates by playing games against random and rule_based opponents.

Usage:
    uv run python scripts/train_bc.py --data-dir data/expert/test --epochs 10

    # Quick smoke test
    uv run python scripts/train_bc.py --epochs 2 --max-samples 50000 --eval-games 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def evaluate_model(
    model_path: str,
    num_players: int,
    policy_kwargs: dict[str, Any] | None,
    n_games: int,
    seed: int,
    verbose: bool,
) -> dict[str, float]:
    """Evaluate a BC model by playing games against different opponents."""
    from sb3_contrib import MaskablePPO
    from training.pettingzoo_selfplay import SelfPlayEnv

    model = MaskablePPO.load(model_path)

    results = {}
    for opponent_type in ["random", "rule_based"]:
        eval_env = SelfPlayEnv(
            num_players=num_players,
            max_turns=300,
            opponent_type=opponent_type,
        )

        wins = 0
        for ep in range(n_games):
            obs, info = eval_env.reset(seed=seed + 10000 + ep)
            done = False
            steps = 0

            while not done and steps < 500:
                mask = eval_env.action_masks()
                action, _ = model.predict(
                    obs.reshape(1, -1),
                    action_masks=mask.reshape(1, -1),
                    deterministic=True,
                )
                obs, reward, terminated, truncated, info = eval_env.step(
                    int(action[0])
                )
                done = terminated or truncated
                steps += 1

            if reward > 0:
                wins += 1

        eval_env.close()
        win_rate = wins / n_games
        results[opponent_type] = win_rate

        if verbose:
            print(f"  vs {opponent_type:12s}: {win_rate:.1%} ({wins}/{n_games})")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Behavioral cloning: train policy from expert demonstrations"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/expert/test",
        help="Path to HDF5 data directory (default: data/expert/test)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/bc",
        help="Output directory for model and logs (default: models/bc)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Training batch size (default: 512)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience in epochs (default: 5)",
    )
    parser.add_argument(
        "--net-arch",
        type=str,
        choices=["default", "large", "separate-large"],
        default="separate-large",
        help="Network architecture (default: separate-large [256,256] each for pi/vf)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=4,
        help="Number of players (determines observation size, default: 4)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap training set size (default: use all samples)",
    )
    parser.add_argument(
        "--filter-winners",
        action="store_true",
        help="Only train on winning players' experiences",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=100,
        help="Games to play for post-training evaluation (default: 100, 0 to skip)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Resolve network architecture
    net_arch_map: dict[str, dict[str, Any] | None] = {
        "default": None,
        "large": {"net_arch": [256, 256]},
        "separate-large": {"net_arch": dict(pi=[256, 256], vf=[256, 256])},
    }
    policy_kwargs = net_arch_map.get(args.net_arch)

    # Create config
    from training.behavioral_cloning import BCConfig, BCTrainer

    config = BCConfig(
        data_dir=Path(args.data_dir),
        save_dir=Path(args.save_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        num_players=args.num_players,
        policy_kwargs=policy_kwargs,
        max_samples=args.max_samples,
        filter_winners=args.filter_winners,
        seed=args.seed,
        verbose=not args.quiet,
    )

    # Train
    trainer = BCTrainer(config)
    model_path = trainer.train()

    # Post-training evaluation
    if args.eval_games > 0:
        verbose = not args.quiet
        if verbose:
            print(f"\nPost-training evaluation ({args.eval_games} games each):")

        results = evaluate_model(
            model_path=str(model_path),
            num_players=args.num_players,
            policy_kwargs=policy_kwargs,
            n_games=args.eval_games,
            seed=args.seed,
            verbose=verbose,
        )

        if verbose:
            chance = 1.0 / args.num_players
            print(f"\n  Chance baseline: {chance:.1%}")
            for opp, rate in results.items():
                delta = rate - chance
                print(
                    f"  vs {opp}: {'+' if delta > 0 else ''}{delta:.1%} vs chance"
                )


if __name__ == "__main__":
    main()
