#!/usr/bin/env python3
"""Benchmark training throughput across parallelism and hyperparameter configs.

Tests different combinations of num_envs, n_steps, batch_size, and
vectorization backend to find the optimal training configuration.

Usage:
    # Quick default sweep (tests key configs from OPT-1 findings)
    uv run python scripts/benchmark_envs.py

    # Custom configs
    uv run python scripts/benchmark_envs.py --configs 8:2048:64 16:2048:256 32:2048:256

    # Sweep env counts with fixed hyperparams
    uv run python scripts/benchmark_envs.py --sweep-envs 8 16 24 32

    # Shorter runs for quick iteration
    uv run python scripts/benchmark_envs.py --timesteps 25000
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def measure_training(
    num_envs: int,
    n_steps: int,
    batch_size: int,
    timesteps: int,
    use_subproc: bool,
    seed: int = 42,
) -> dict:
    """Run a short training and measure throughput."""
    import psutil

    from training.pettingzoo_selfplay import SelfPlayConfig, SelfPlayTrainer

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    config = SelfPlayConfig(
        total_timesteps=timesteps,
        num_envs=num_envs,
        n_steps=n_steps,
        batch_size=batch_size,
        num_players=2,
        max_turns=200,
        opponent_type="random",
        reward_type="dense",
        eval_freq=timesteps + 1,
        save_freq=timesteps + 1,
        save_dir=f"/tmp/bench_{num_envs}e_{n_steps}s_{batch_size}b",
        seed=seed,
        verbose=False,
        diagnostic_logging=False,
        use_subproc=use_subproc,
        lr_schedule="constant",
    )

    trainer = SelfPlayTrainer(config)

    start = time.perf_counter()
    trainer.train()
    wall_time = time.perf_counter() - start

    mem_after = process.memory_info().rss / (1024 * 1024)

    if trainer._vec_env is not None:
        trainer._vec_env.close()

    buffer = num_envs * n_steps
    grad_steps = (buffer // batch_size) * 10  # 10 = default n_epochs

    return {
        "num_envs": num_envs,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "buffer_size": buffer,
        "grad_steps": grad_steps,
        "backend": "subproc" if use_subproc else "dummy",
        "wall_time": wall_time,
        "steps_per_sec": timesteps / wall_time,
        "mem_delta_mb": mem_after - mem_before,
    }


def parse_config(spec: str) -> tuple[int, int, int]:
    """Parse 'envs:n_steps:batch_size' spec."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Config must be envs:n_steps:batch_size, got '{spec}'")
    return int(parts[0]), int(parts[1]), int(parts[2])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark training throughput",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Steps per benchmark run (default: 50000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--configs",
        nargs="+",
        type=str,
        metavar="E:S:B",
        help="Custom configs as envs:n_steps:batch_size (e.g. 16:2048:256)",
    )
    group.add_argument(
        "--sweep-envs",
        nargs="+",
        type=int,
        metavar="N",
        help="Sweep env counts with default n_steps=2048, batch_size=64",
    )

    parser.add_argument(
        "--backend",
        choices=["dummy", "subproc", "both"],
        default="dummy",
        help="Vectorization backend to test (default: dummy)",
    )

    args = parser.parse_args()

    # Build config list: (num_envs, n_steps, batch_size, use_subproc)
    configs: list[tuple[int, int, int, bool]] = []

    if args.configs:
        for spec in args.configs:
            envs, steps, batch = parse_config(spec)
            if args.backend in ("dummy", "both"):
                configs.append((envs, steps, batch, False))
            if args.backend in ("subproc", "both"):
                configs.append((envs, steps, batch, True))

    elif args.sweep_envs:
        for envs in args.sweep_envs:
            if args.backend in ("dummy", "both"):
                configs.append((envs, 2048, 64, False))
            if args.backend in ("subproc", "both"):
                configs.append((envs, 2048, 64, True))

    else:
        # Default: key configs from OPT-1 findings
        configs = [
            (8, 2048, 64, False),  # baseline
            (16, 2048, 128, False),  # 2x envs, 2x batch (same grad steps)
            (16, 2048, 256, False),  # recommended default
            (16, 2048, 512, False),  # fast mode
            (32, 2048, 256, False),  # more envs
        ]

    # Print header
    import psutil

    print("=" * 85)
    print("TRAINING THROUGHPUT BENCHMARK")
    print("=" * 85)
    print(
        f"CPU: {psutil.cpu_count(logical=False)} physical / "
        f"{psutil.cpu_count(logical=True)} logical cores"
    )
    print(
        f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB total, "
        f"{psutil.virtual_memory().available / (1024**3):.1f} GB available"
    )
    print(f"Timesteps per run: {args.timesteps:,}")
    print(f"Configs: {len(configs)}")
    print()

    # Run benchmarks
    results = []
    for num_envs, n_steps, batch_size, use_subproc in configs:
        backend = "subproc" if use_subproc else "dummy"
        label = f"{num_envs}e/{n_steps}s/{batch_size}b/{backend}"
        print(f"[{label}] Running... ", end="", flush=True)
        r = measure_training(
            num_envs,
            n_steps,
            batch_size,
            args.timesteps,
            use_subproc,
            args.seed,
        )
        print(f"{r['steps_per_sec']:.0f} steps/s ({r['wall_time']:.1f}s)")
        results.append(r)

    # Summary table
    print()
    print("=" * 85)
    print(
        f"{'Envs':>4} {'nSteps':>6} {'Batch':>5} {'Backend':>7} {'Buffer':>6} "
        f"{'GradSteps':>9} {'Steps/s':>8} {'Time':>7} {'Mem':>7} {'Speedup':>7}"
    )
    print("-" * 85)

    baseline_sps = results[0]["steps_per_sec"]
    for r in results:
        speedup = r["steps_per_sec"] / baseline_sps
        print(
            f"{r['num_envs']:>4} {r['n_steps']:>6} {r['batch_size']:>5} "
            f"{r['backend']:>7} {r['buffer_size']:>6} {r['grad_steps']:>9} "
            f"{r['steps_per_sec']:>8.0f} {r['wall_time']:>6.1f}s "
            f"{r['mem_delta_mb']:>+6.0f}MB {speedup:>6.2f}x"
        )

    best = max(results, key=lambda r: r["steps_per_sec"])
    print()
    print(
        f"BEST: {best['num_envs']}e/{best['n_steps']}s/{best['batch_size']}b/"
        f"{best['backend']} ({best['steps_per_sec']:.0f} steps/s, "
        f"{best['steps_per_sec'] / baseline_sps:.2f}x vs first config)"
    )


if __name__ == "__main__":
    main()
