"""CLI for MCTS self-play training.

Usage:
    uv run python scripts/train_mcts.py --iterations 50 --games-per-iter 100
    uv run python scripts/train_mcts.py --resume models/mcts_iter_10/
    uv run python scripts/train_mcts.py --eval-only models/mcts_best/
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="MCTS Self-Play Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Self-play loop
    parser.add_argument(
        "--iterations", type=int, default=50, help="Number of self-play iterations"
    )
    parser.add_argument(
        "--games-per-iter", type=int, default=100, help="Games per iteration"
    )
    parser.add_argument(
        "--simulations", type=int, default=100, help="MCTS simulations per move"
    )
    parser.add_argument("--num-players", type=int, default=4, help="Players per game")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="MCTS temperature for data generation",
    )
    parser.add_argument(
        "--save-dir", type=str, default="models/mcts", help="Directory for checkpoints"
    )

    # Evaluation
    parser.add_argument(
        "--eval-freq", type=int, default=5, help="Evaluate every N iterations"
    )
    parser.add_argument(
        "--eval-games", type=int, default=50, help="Games for evaluation"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="rule_based",
        choices=["rule_based", "random"],
        help="Opponent policy during self-play",
    )

    # Checkpoint / resume
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint directory"
    )
    parser.add_argument(
        "--eval-only",
        type=str,
        default=None,
        help="Evaluate a checkpoint without training",
    )

    # Network training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Training epochs per iteration"
    )

    # Game settings
    parser.add_argument(
        "--max-turns", type=int, default=500, help="Max turns per game"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser


def _resolve_network_path(ckpt_path: str) -> Path:
    """Resolve a checkpoint path to a network weights file.

    If path is a directory, returns path/network.pt.
    If path is a file, returns it directly.
    """
    p = Path(ckpt_path)
    if p.is_dir():
        return p / "network.pt"
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # --eval-only mode: evaluate a checkpoint without running training
    if args.eval_only is not None:
        from mcts.eval import evaluate_mcts_agent

        network_path = _resolve_network_path(args.eval_only)
        resolved = network_path if network_path.exists() else None
        print(f"Evaluating checkpoint: {network_path}")
        if resolved is None:
            print("  (no network.pt found — using random rollouts)")

        results = evaluate_mcts_agent(
            network_path=resolved,
            num_simulations=args.simulations,
            num_games=args.eval_games,
            opponents=["random", "rule_based"],
            num_players=args.num_players,
            max_turns=args.max_turns,
            seed=args.seed,
            verbose=True,
        )

        print("\nEvaluation Results:")
        for opp_type, metrics in results.items():
            print(
                f"  vs {opp_type}: "
                f"{metrics['win_rate']:.1%} win rate "
                f"({metrics['wins']:.0f}W / {metrics['losses']:.0f}L / "
                f"{metrics['draws']:.0f}D)"
            )
        return

    # Training mode: build configs and run self-play loop
    from mcts.training import SelfPlayConfig, TrainingConfig, self_play_loop

    training_cfg = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
    )

    config = SelfPlayConfig(
        num_iterations=args.iterations,
        games_per_iteration=args.games_per_iter,
        mcts_simulations=args.simulations,
        num_players=args.num_players,
        temperature=args.temperature,
        eval_games=args.eval_games,
        eval_frequency=args.eval_freq,
        max_turns_per_game=args.max_turns,
        training_config=training_cfg,
    )

    print("Starting MCTS self-play training:")
    print(f"  Iterations:  {config.num_iterations}")
    print(f"  Games/iter:  {config.games_per_iteration}")
    print(f"  Simulations: {config.mcts_simulations}")
    print(f"  Players:     {config.num_players}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Save dir:    {args.save_dir}")
    if args.resume:
        print(f"  Resume from: {args.resume}")

    self_play_loop(
        config=config,
        save_dir=args.save_dir,
        resume_from=args.resume,
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
