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
from typing import Any

import numpy as np

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
    # Mixed opponent training
    parser.add_argument(
        "--opponent-pool",
        nargs="+",
        type=str,
        choices=["random", "rule_based", "conservative", "aggressive"],
        default=None,
        help="Pool of opponent types to sample from (default: use --opponent only)",
    )
    parser.add_argument(
        "--opponent-weights",
        nargs="+",
        type=float,
        default=None,
        help="Sampling weights for opponent pool (must match pool length, default: uniform)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=16,
        help="Number of parallel environments (default: 16)",
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
        "--n-steps",
        type=int,
        default=2048,
        help="Steps per PPO update per env (default: 2048)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Minibatch size (default: 256)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="PPO epochs per update (default: 10)",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Entropy coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        choices=["linear", "constant", "cosine"],
        default="linear",
        help="Learning rate schedule (default: linear)",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=0.0,
        help="Minimum LR floor for linear schedule (default: 0.0)",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="Value function loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda (default: 0.95)",
    )
    parser.add_argument(
        "--net-arch",
        type=str,
        choices=["default", "large", "separate-large"],
        default="default",
        help="Network architecture (default: [64,64] shared)",
    )
    parser.add_argument(
        "--worth-scale",
        type=float,
        default=500.0,
        help="Divisor for dense reward net worth delta (default: 500.0)",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO clip range (default: 0.2, try 0.1 for stability)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Maximum gradient norm for clipping (default: 0.5)",
    )
    parser.add_argument(
        "--terminal-scale",
        type=float,
        default=5.0,
        help="Terminal reward magnitude for dense mode (+val win, -val loss) (default: 5.0)",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to pre-trained model to load (for transfer learning)",
    )
    parser.add_argument(
        "--reward-type",
        type=str,
        choices=["sparse", "dense"],
        default="dense",
        help="Reward type: sparse (win/loss only) or dense (per-step shaping) (default: dense)",
    )
    parser.add_argument(
        "--min-win-rate",
        type=float,
        default=0.0,
        help="Min win rate vs random to save checkpoint, 0=disabled (default: 0)",
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
    parser.add_argument(
        "--diagnostic-logging",
        action="store_true",
        default=True,
        help="Enable diagnostic logging (value stats, gradients, etc.)",
    )
    parser.add_argument(
        "--no-diagnostic-logging",
        dest="diagnostic_logging",
        action="store_false",
        help="Disable diagnostic logging",
    )
    parser.add_argument(
        "--vec-env",
        type=str,
        choices=["auto", "subproc", "dummy"],
        default="auto",
        help="Vectorization backend: auto (SubprocVecEnv except self-play), "
             "subproc (force SubprocVecEnv), dummy (force DummyVecEnv) (default: auto)",
    )
    parser.add_argument(
        "--normalize-env", action="store_true", default=False,
        help="Enable VecNormalize for observation and reward normalization",
    )
    parser.add_argument(
        "--no-normalize-env", dest="normalize_env", action="store_false",
        help="Disable VecNormalize (default)",
    )
    parser.add_argument(
        "--vf-lr-multiplier", type=float, default=1.0,
        help="Value function LR multiplier (e.g., 3.0 = 3x policy LR). Default: 1.0 (same LR).",
    )

    args = parser.parse_args()

    # Resolve network architecture
    net_arch_map: dict[str, dict[str, Any] | None] = {
        "default": None,
        "large": {"net_arch": [256, 256]},
        "separate-large": {"net_arch": dict(pi=[256, 256], vf=[256, 256])},
    }
    policy_kwargs = net_arch_map.get(args.net_arch)

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
        # Resolve vec-env flag
        use_subproc = args.vec_env != "dummy"  # "auto" and "subproc" both enable it

        config = SelfPlayConfig(
            total_timesteps=args.timesteps,
            num_envs=args.num_envs,
            num_players=args.num_players,
            max_turns=args.max_turns,
            opponent_type=args.opponent,
            reward_type=args.reward_type,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            save_freq=args.save_freq,
            save_dir=args.save_dir,
            learning_rate=args.learning_rate,
            lr_schedule=args.lr_schedule,
            lr_min=args.lr_min,
            clip_range=args.clip_range,
            max_grad_norm=args.max_grad_norm,
            terminal_win_reward=args.terminal_scale,
            terminal_loss_reward=-args.terminal_scale,
            load_model=args.load_model,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            policy_kwargs=policy_kwargs,
            worth_scale=args.worth_scale,
            checkpoint_min_win_rate=args.min_win_rate,
            seed=args.seed,
            verbose=not args.quiet,
            diagnostic_logging=args.diagnostic_logging,
            opponent_pool=args.opponent_pool,
            opponent_weights=args.opponent_weights,
            use_subproc=use_subproc,
            normalize_env=args.normalize_env,
            vf_lr_multiplier=args.vf_lr_multiplier,
        )

        # Validate opponent pool weights
        if config.opponent_pool and config.opponent_weights:
            if len(config.opponent_weights) != len(config.opponent_pool):
                raise ValueError(
                    f"opponent_weights length ({len(config.opponent_weights)}) "
                    f"must match opponent_pool length ({len(config.opponent_pool)})"
                )
            if not np.isclose(sum(config.opponent_weights), 1.0):
                raise ValueError(
                    f"opponent_weights must sum to 1.0, got {sum(config.opponent_weights)}"
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

    # Resolve network architecture
    net_arch_map_cur: dict[str, dict[str, Any] | None] = {
        "default": None,
        "large": {"net_arch": [256, 256]},
        "separate-large": {"net_arch": dict(pi=[256, 256], vf=[256, 256])},
    }
    policy_kwargs_cur = net_arch_map_cur.get(args.net_arch)

    use_subproc = args.vec_env != "dummy"

    for stage_name, stage_steps in stages:
        print(f"\n{'='*60}")
        print(f"STAGE: {stage_name.upper()} ({stage_steps:,} steps)")
        print(f"{'='*60}\n")

        # Only load external model for first stage when no prior model exists
        stage_load_model = args.load_model if model is None else None

        config = SelfPlayConfig(
            total_timesteps=stage_steps,
            num_envs=args.num_envs,
            num_players=args.num_players,
            max_turns=args.max_turns,
            opponent_type=stage_name,
            reward_type=args.reward_type,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            save_freq=args.save_freq,
            save_dir=str(save_dir / stage_name),
            learning_rate=args.learning_rate,
            lr_schedule=args.lr_schedule,
            lr_min=args.lr_min,
            clip_range=args.clip_range,
            max_grad_norm=args.max_grad_norm,
            terminal_win_reward=args.terminal_scale,
            terminal_loss_reward=-args.terminal_scale,
            load_model=stage_load_model,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            policy_kwargs=policy_kwargs_cur,
            worth_scale=args.worth_scale,
            seed=args.seed,
            verbose=not args.quiet,
            diagnostic_logging=args.diagnostic_logging,
            opponent_pool=args.opponent_pool,
            opponent_weights=args.opponent_weights,
            use_subproc=use_subproc,
            normalize_env=args.normalize_env,
            vf_lr_multiplier=args.vf_lr_multiplier,
        )

        trainer = SelfPlayTrainer(config)

        # Load previous model if available
        if model is not None:
            trainer._model = model
            # Need to set up environment first
            from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
            from training.pettingzoo_selfplay import SelfPlayEnv

            def make_env(rank: int):
                def _init():
                    env = SelfPlayEnv(
                        num_players=config.num_players,
                        max_turns=config.max_turns,
                        opponent_type=config.opponent_type,
                        reward_type=config.reward_type,
                        worth_scale=config.worth_scale,
                    )
                    env.reset(seed=config.seed + rank)
                    return env
                return _init

            env_fns = [make_env(i) for i in range(config.num_envs)]
            stage_use_subproc = (
                config.use_subproc
                and config.num_envs > 1
                and stage_name != "self"
            )
            if stage_use_subproc:
                trainer._vec_env = SubprocVecEnv(env_fns)
            else:
                trainer._vec_env = DummyVecEnv(env_fns)
            trainer._model.set_env(trainer._vec_env)

        model = trainer.train()

    # Save final curriculum model
    final_path = save_dir / "curriculum_final"
    model.save(str(final_path))
    print(f"\nFinal curriculum model saved to: {final_path}")


if __name__ == "__main__":
    main()
