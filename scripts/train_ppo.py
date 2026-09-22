#!/usr/bin/env python3
"""Train a Monopoly agent using MaskablePPO.

Usage:
    uv run python scripts/train_ppo.py
    uv run python scripts/train_ppo.py --timesteps 500000 --opponent rule_based
    uv run python scripts/train_ppo.py --curriculum
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a Monopoly agent using MaskablePPO")

    # Training mode
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Use curriculum learning",
    )

    # Training parameters
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps (default: 1M)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel environments (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    # Environment parameters
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "rule_based", "aggressive", "conservative"],
        help="Opponent type (default: random)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players (default: 2)",
    )
    parser.add_argument(
        "--reward-type",
        type=str,
        default="sparse",
        choices=["sparse", "dense"],
        help="Reward type (default: sparse)",
    )
    parser.add_argument(
        "--rules-id",
        choices=["foundation-v1", "foundation-trade-v1"],
        default="foundation-v1",
        help="Rules and learner encoding contract (default: foundation-v1)",
    )

    # Output paths
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/ppo_monopoly",
        help="Path to save model (default: models/ppo_monopoly)",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="logs/tensorboard",
        help="Path for TensorBoard logs (default: logs/tensorboard)",
    )

    # Evaluation
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50_000,
        help="Evaluate every N timesteps (default: 50000)",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=20,
        help="Number of evaluation games (default: 20)",
    )

    return parser.parse_args()


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    print("=" * 60)
    print("Monopoly PPO Training")
    print("=" * 60)

    try:
        # Check if training dependencies are available
        from sb3_contrib import MaskablePPO  # noqa: F401
    except ImportError:
        print("\nError: Training dependencies not installed.")
        print("Install with: uv pip install stable-baselines3 sb3-contrib torch tensorboard")
        sys.exit(1)

    if args.curriculum:
        print("\nMode: Curriculum Learning")
        from training import CurriculumTrainer

        trainer = CurriculumTrainer(
            base_save_path=args.save_path,
            tensorboard_log=args.log_path,
            seed=args.seed,
            verbose=True,
        )

        trainer.train()
        print("\n" + trainer.get_summary())

    else:
        print("\nMode: Standard Training")
        print(f"Opponent: {args.opponent}")
        print(f"Timesteps: {args.timesteps:,}")
        print(f"Parallel envs: {args.num_envs}")
        print(f"Seed: {args.seed}")

        from training import EnvironmentConfig, TrainingConfig, train_agent

        training_config = TrainingConfig(
            total_timesteps=args.timesteps,
            num_envs=args.num_envs,
            seed=args.seed,
        )

        env_config = EnvironmentConfig(
            num_players=args.num_players,
            opponent_type=args.opponent,
            reward_type=args.reward_type,
            rules_id=args.rules_id,
        )

        print("\nStarting training...")
        train_agent(
            training_config=training_config,
            env_config=env_config,
            save_path=args.save_path,
            tensorboard_log=args.log_path,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_games,
        )

        print(f"\nModel saved to: {args.save_path}")

    print("\nTraining complete!")
    print(f"View TensorBoard: tensorboard --logdir {args.log_path}")


if __name__ == "__main__":
    main()
