#!/usr/bin/env python3
"""Benchmark suite for the Monopoly game engine.

This script measures performance metrics including:
- Games per second throughput
- Memory usage
- Serialization speed
- Action validation performance
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from monopoly_engine.game import MonopolyGame
from monopoly_engine.actions import RollDice, BuyProperty, EndTurn, PayJailFine
from monopoly_engine.types import SpaceType


def play_game_fast(seed: int, max_turns: int = 300) -> tuple[bool, int, float]:
    """Play a game as fast as possible with minimal strategy.

    Args:
        seed: Random seed
        max_turns: Maximum turns before stopping

    Returns:
        Tuple of (game_over, turns_played, elapsed_time)
    """
    start_time = time.perf_counter()

    game = MonopolyGame(num_players=2, seed=seed)

    for turn in range(max_turns):
        player_id = game.state.current_player
        player = game.state.players[player_id]

        if player.bankrupt:
            EndTurn(player_id=player_id).execute(game)
            continue

        # Handle jail
        if player.in_jail and player.money >= 50:
            PayJailFine(player_id=player_id).execute(game)

        # Roll dice
        RollDice(player_id=player_id).execute(game)

        # Buy if cheap and we have money
        if not player.bankrupt and not player.in_jail:
            space = game.board.get_space(player.position)
            if space.space_type in (SpaceType.PROPERTY, SpaceType.RAILROAD, SpaceType.UTILITY):
                prop = game.state.property_manager.get(player.position)
                if prop and prop.owner is None:
                    cost = getattr(space, "cost", 0)
                    if cost > 0 and cost <= 200 and player.money >= cost + 100:
                        BuyProperty(player_id=player_id, property_id=player.position).execute(
                            game
                        )

        EndTurn(player_id=player_id).execute(game)

        if game.state.game_over:
            break

    elapsed = time.perf_counter() - start_time
    return game.state.game_over, turn, elapsed


def benchmark_throughput(num_games: int = 1000, num_players: int = 2) -> dict[str, Any]:
    """Benchmark game throughput.

    Args:
        num_games: Number of games to run
        num_players: Number of players per game

    Returns:
        Dictionary with throughput metrics
    """
    print(f"Running {num_games} games with {num_players} players each...")

    # Force garbage collection before starting
    gc.collect()

    start_time = time.perf_counter()
    completed_games = 0
    total_turns = 0

    for seed in range(num_games):
        game_over, turns, _ = play_game_fast(seed)
        if game_over:
            completed_games += 1
        total_turns += turns

    elapsed = time.perf_counter() - start_time

    return {
        "num_games": num_games,
        "completed_games": completed_games,
        "total_turns": total_turns,
        "elapsed_seconds": elapsed,
        "games_per_second": num_games / elapsed,
        "avg_turns_per_game": total_turns / num_games,
        "avg_time_per_game_ms": (elapsed / num_games) * 1000,
    }


def benchmark_serialization(num_iterations: int = 1000) -> dict[str, Any]:
    """Benchmark state serialization/deserialization.

    Args:
        num_iterations: Number of serialize/deserialize cycles

    Returns:
        Dictionary with serialization metrics
    """
    print(f"Benchmarking serialization ({num_iterations} iterations)...")

    game = MonopolyGame(num_players=4, seed=42)

    # Play some turns to have realistic state
    for _ in range(20):
        RollDice(player_id=game.state.current_player).execute(game)
        EndTurn(player_id=game.state.current_player).execute(game)

    # Benchmark serialization
    serialize_times = []
    deserialize_times = []
    state_sizes = []

    for _ in range(num_iterations):
        # Serialize
        start = time.perf_counter()
        state_dict = game.to_dict()
        serialize_time = time.perf_counter() - start
        serialize_times.append(serialize_time)

        # Measure size
        json_str = json.dumps(state_dict)
        state_sizes.append(len(json_str))

        # Deserialize
        start = time.perf_counter()
        MonopolyGame.from_dict(state_dict)
        deserialize_time = time.perf_counter() - start
        deserialize_times.append(deserialize_time)

    avg_serialize = sum(serialize_times) / len(serialize_times)
    avg_deserialize = sum(deserialize_times) / len(deserialize_times)
    avg_size = sum(state_sizes) / len(state_sizes)

    return {
        "num_iterations": num_iterations,
        "avg_serialize_ms": avg_serialize * 1000,
        "avg_deserialize_ms": avg_deserialize * 1000,
        "avg_state_size_bytes": int(avg_size),
        "avg_state_size_kb": avg_size / 1024,
    }


def benchmark_action_validation(num_iterations: int = 10000) -> dict[str, Any]:
    """Benchmark action validation performance.

    Args:
        num_iterations: Number of validation checks to run

    Returns:
        Dictionary with validation metrics
    """
    print(f"Benchmarking action validation ({num_iterations} iterations)...")

    game = MonopolyGame(num_players=2, seed=42)

    # Benchmark RollDice validation (simple case)
    roll_times = []
    for _ in range(num_iterations):
        action = RollDice(player_id=0)
        start = time.perf_counter()
        action.validate(game)
        elapsed = time.perf_counter() - start
        roll_times.append(elapsed)

    # Benchmark BuyProperty validation (more complex)
    game.state.players[0].position = 1  # Mediterranean
    buy_times = []
    for _ in range(num_iterations):
        action = BuyProperty(player_id=0, property_id=1)
        start = time.perf_counter()
        action.validate(game)
        elapsed = time.perf_counter() - start
        buy_times.append(elapsed)

    avg_roll = sum(roll_times) / len(roll_times)
    avg_buy = sum(buy_times) / len(buy_times)

    return {
        "num_iterations": num_iterations,
        "avg_roll_dice_validation_us": avg_roll * 1_000_000,
        "avg_buy_property_validation_us": avg_buy * 1_000_000,
    }


def benchmark_memory_usage() -> dict[str, Any]:
    """Benchmark memory usage of game instances.

    Returns:
        Dictionary with memory metrics
    """
    import sys

    print("Benchmarking memory usage...")

    # Single game
    game = MonopolyGame(num_players=2, seed=42)
    single_size = sys.getsizeof(game)

    # Multiple games
    games = [MonopolyGame(num_players=2, seed=i) for i in range(100)]
    multi_size = sum(sys.getsizeof(g) for g in games)

    return {
        "single_game_bytes": single_size,
        "single_game_kb": single_size / 1024,
        "100_games_mb": multi_size / (1024 * 1024),
        "avg_game_kb": (multi_size / 100) / 1024,
    }


def run_all_benchmarks(
    throughput_games: int = 1000,
    serialize_iterations: int = 1000,
    validate_iterations: int = 10000,
) -> dict[str, Any]:
    """Run all benchmarks and return results.

    Args:
        throughput_games: Number of games for throughput test
        serialize_iterations: Number of iterations for serialization test
        validate_iterations: Number of iterations for validation test

    Returns:
        Dictionary with all benchmark results
    """
    results = {}

    # Throughput benchmark
    results["throughput"] = benchmark_throughput(num_games=throughput_games)

    # Serialization benchmark
    results["serialization"] = benchmark_serialization(num_iterations=serialize_iterations)

    # Validation benchmark
    results["validation"] = benchmark_action_validation(num_iterations=validate_iterations)

    # Memory benchmark
    results["memory"] = benchmark_memory_usage()

    return results


def print_results(results: dict[str, Any]) -> None:
    """Print benchmark results in a readable format.

    Args:
        results: Dictionary with benchmark results
    """
    print("\n" + "=" * 70)
    print("MONOPOLY ENGINE BENCHMARK RESULTS")
    print("=" * 70)

    # Throughput
    if "throughput" in results:
        t = results["throughput"]
        print("\n📊 THROUGHPUT")
        print(f"  Games played: {t['num_games']}")
        print(f"  Completed: {t['completed_games']} ({t['completed_games']/t['num_games']*100:.1f}%)")
        print(f"  Total turns: {t['total_turns']}")
        print(f"  Elapsed: {t['elapsed_seconds']:.2f}s")
        print(f"  → Games/second: {t['games_per_second']:.1f}")
        print(f"  → Avg turns/game: {t['avg_turns_per_game']:.1f}")
        print(f"  → Avg time/game: {t['avg_time_per_game_ms']:.2f}ms")

        # Check target
        target = 1000
        if t["games_per_second"] >= target:
            print(f"  ✓ PASSES target of {target} games/second")
        else:
            print(f"  ✗ FAILS target of {target} games/second")

    # Serialization
    if "serialization" in results:
        s = results["serialization"]
        print("\n💾 SERIALIZATION")
        print(f"  Iterations: {s['num_iterations']}")
        print(f"  → Avg serialize: {s['avg_serialize_ms']:.3f}ms")
        print(f"  → Avg deserialize: {s['avg_deserialize_ms']:.3f}ms")
        print(f"  → Avg state size: {s['avg_state_size_kb']:.2f}KB")

    # Validation
    if "validation" in results:
        v = results["validation"]
        print("\n⚡ ACTION VALIDATION")
        print(f"  Iterations: {v['num_iterations']}")
        print(f"  → RollDice validation: {v['avg_roll_dice_validation_us']:.2f}µs")
        print(f"  → BuyProperty validation: {v['avg_buy_property_validation_us']:.2f}µs")

    # Memory
    if "memory" in results:
        m = results["memory"]
        print("\n💿 MEMORY USAGE")
        print(f"  → Single game: {m['single_game_kb']:.2f}KB")
        print(f"  → 100 games: {m['100_games_mb']:.2f}MB")
        print(f"  → Avg per game: {m['avg_game_kb']:.2f}KB")

    print("\n" + "=" * 70)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Monopoly Engine Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark suite
  python scripts/benchmark.py

  # Quick test
  python scripts/benchmark.py --quick

  # Focus on throughput
  python scripts/benchmark.py --throughput 5000

  # Export results to JSON
  python scripts/benchmark.py --export results.json
        """,
    )

    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark (fewer iterations)"
    )
    parser.add_argument(
        "--throughput", type=int, help="Number of games for throughput test (default: 1000)"
    )
    parser.add_argument(
        "--serialize", type=int, help="Number of serialization iterations (default: 1000)"
    )
    parser.add_argument(
        "--validate", type=int, help="Number of validation iterations (default: 10000)"
    )
    parser.add_argument("--export", type=str, help="Export results to JSON file")

    args = parser.parse_args()

    # Set iteration counts
    if args.quick:
        throughput = 100
        serialize = 100
        validate = 1000
    else:
        throughput = args.throughput or 1000
        serialize = args.serialize or 1000
        validate = args.validate or 10000

    # Run benchmarks
    results = run_all_benchmarks(
        throughput_games=throughput,
        serialize_iterations=serialize,
        validate_iterations=validate,
    )

    # Print results
    print_results(results)

    # Export if requested
    if args.export:
        with open(args.export, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results exported to {args.export}")


if __name__ == "__main__":
    main()
