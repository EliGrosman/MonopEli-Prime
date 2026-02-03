#!/usr/bin/env python3
"""CLI script for self-play training.

Usage:
    uv run python scripts/train_self_play.py --timesteps 500000
    uv run python scripts/train_self_play.py --timesteps 1000000 --checkpoint-freq 100000
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train a Monopoly agent using self-play",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic self-play training
    uv run python scripts/train_self_play.py --timesteps 500000

    # Longer training with more frequent checkpoints
    uv run python scripts/train_self_play.py --timesteps 1000000 --checkpoint-freq 25000

    # Train with higher past version probability
    uv run python scripts/train_self_play.py --timesteps 500000 --past-prob 0.7
        """,
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps (default: 500000)",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50_000,
        help="Save checkpoint every N steps (default: 50000)",
    )
    parser.add_argument(
        "--past-prob",
        type=float,
        default=0.5,
        help="Probability of playing against past version (default: 0.5)",
    )
    parser.add_argument(
        "--max-versions",
        type=int,
        default=10,
        help="Maximum past versions to keep (default: 10)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel environments (default: 8)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=25_000,
        help="Evaluate every N steps (default: 25000)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/self_play",
        help="Directory to save models (default: models/self_play)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    try:
        from training.self_play import SelfPlayConfig, SelfPlayTrainer
    except ImportError as e:
        print(f"Error importing training modules: {e}")
        print("Make sure sb3-contrib is installed: uv sync --extra training")
        sys.exit(1)

    # Create config
    config = SelfPlayConfig(
        total_timesteps=args.timesteps,
        num_envs=args.num_envs,
        checkpoint_freq=args.checkpoint_freq,
        past_version_prob=args.past_prob,
        max_past_versions=args.max_versions,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        save_dir=args.save_dir,
    )

    # Train
    trainer = SelfPlayTrainer(config)
    model = trainer.train(verbose=not args.quiet)

    # Print final stats
    if not args.quiet:
        print("\n=== Training Statistics ===")
        print(f"Total steps: {trainer.stats.total_steps:,}")
        print(f"Checkpoints saved: {trainer.stats.num_checkpoints}")
        print(f"Training time: {trainer.stats.training_time:.0f}s")

        if trainer.stats.win_rates:
            final_wr = trainer.stats.win_rates[-1]
            print(f"Final win rate vs random: {final_wr.get('vs_random', 0):.1%}")
            if "vs_past" in final_wr:
                print(f"Final win rate vs past: {final_wr.get('vs_past', 0):.1%}")


if __name__ == "__main__":
    main()
