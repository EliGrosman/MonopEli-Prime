#!/usr/bin/env python3
"""CLI script for self-play training.

This script trains a Monopoly agent using proper PettingZoo-based self-play.

Usage:
    # Train vs random opponents (warmup)
    uv run python scripts/train_selfplay.py --opponent random --timesteps 200000

    # Train vs rule-based opponents
    uv run python scripts/train_selfplay.py --opponent rule_based --timesteps 500000

    # True self-play
    uv run python scripts/train_selfplay.py --opponent self --timesteps 1000000

    # Full curriculum (recommended)
    uv run python scripts/train_selfplay.py --curriculum --timesteps 2000000
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Monopoly agent with self-play",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test (10 min)
    uv run python scripts/train_selfplay.py --timesteps 50000 --eval-freq 10000

    # Train vs random (baseline)
    uv run python scripts/train_selfplay.py --opponent random --timesteps 200000

    # Train vs rule-based (target >50% win rate)
    uv run python scripts/train_selfplay.py --opponent rule_based --timesteps 500000

    # Long run with self-play (multi-hour)
    uv run python scripts/train_selfplay.py --opponent self --timesteps 2000000

    # View training in TensorBoard
    tensorboard --logdir models/selfplay/tensorboard --host 0.0.0.0 --port 6006
        """,
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps (default: 500000)",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        choices=["random", "rule_based", "aggressive", "conservative", "mixed", "self"],
        default="random",
        help="Opponent type (default: random)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel environments (default: 8)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=4,
        help="Number of players per game (default: 4)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=500,
        help="Maximum turns per game (default: 500)",
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
        default=50,
        help="Episodes per evaluation (default: 50)",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50_000,
        help="Save checkpoint every N steps (default: 50000)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/selfplay",
        help="Directory to save models (default: models/selfplay)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Use curriculum: random -> rule_based -> self",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Import training module
    try:
        from training.pettingzoo_selfplay import SelfPlayConfig, SelfPlayTrainer
    except ImportError as e:
        print(f"Error importing training module: {e}")
        print("Make sure sb3-contrib is installed: uv sync --extra training")
        sys.exit(1)

    if args.curriculum:
        # Run curriculum training
        run_curriculum(args)
    else:
        # Single opponent type training
        config = SelfPlayConfig(
            total_timesteps=args.timesteps,
            num_envs=args.num_envs,
            num_players=args.num_players,
            max_turns=args.max_turns,
            opponent_type=args.opponent,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            save_freq=args.save_freq,
            save_dir=args.save_dir,
            learning_rate=args.learning_rate,
            seed=args.seed,
            verbose=not args.quiet,
        )

        trainer = SelfPlayTrainer(config)
        trainer.train()


def run_curriculum(args: argparse.Namespace) -> None:
    """Run curriculum training: random -> rule_based -> self."""
    from training.pettingzoo_selfplay import SelfPlayConfig, SelfPlayTrainer

    total = args.timesteps
    stages = [
        ("random", int(total * 0.2)),        # 20% vs random
        ("rule_based", int(total * 0.4)),    # 40% vs rule_based
        ("self", int(total * 0.4)),          # 40% self-play
    ]

    print("=" * 60)
    print("CURRICULUM TRAINING")
    print("=" * 60)
    for stage, steps in stages:
        print(f"  {stage}: {steps:,} steps")
    print("=" * 60)
    print()

    model = None
    save_dir = Path(args.save_dir)

    for stage_name, stage_steps in stages:
        print(f"\n{'='*60}")
        print(f"STAGE: {stage_name.upper()} ({stage_steps:,} steps)")
        print(f"{'='*60}\n")

        config = SelfPlayConfig(
            total_timesteps=stage_steps,
            num_envs=args.num_envs,
            num_players=args.num_players,
            max_turns=args.max_turns,
            opponent_type=stage_name,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            save_freq=args.save_freq,
            save_dir=str(save_dir / stage_name),
            learning_rate=args.learning_rate,
            seed=args.seed,
            verbose=not args.quiet,
        )

        trainer = SelfPlayTrainer(config)

        # Load previous model if available
        if model is not None:
            trainer._model = model
            # Need to set up environment first
            from stable_baselines3.common.vec_env import DummyVecEnv
            from training.pettingzoo_selfplay import SelfPlayEnv

            def make_env(rank: int):
                def _init():
                    env = SelfPlayEnv(
                        num_players=config.num_players,
                        max_turns=config.max_turns,
                        opponent_type=config.opponent_type,
                    )
                    env.reset(seed=config.seed + rank)
                    return env
                return _init

            trainer._vec_env = DummyVecEnv([make_env(i) for i in range(config.num_envs)])
            trainer._model.set_env(trainer._vec_env)

        model = trainer.train()

    # Save final curriculum model
    final_path = save_dir / "curriculum_final"
    model.save(str(final_path))
    print(f"\nFinal curriculum model saved to: {final_path}")


if __name__ == "__main__":
    main()
