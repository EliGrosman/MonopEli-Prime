#!/usr/bin/env python3
"""Compare evaluation results across multiple JSON result files.

Reads JSON files produced by ``evaluate_mcts.py`` and ``evaluate_hybrid.py``
and prints a side-by-side comparison table.

Usage:
    uv run python scripts/compare_results.py --files results/*.json
    uv run python scripts/compare_results.py  # auto-discovers results/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_results(json_path: str) -> dict:
    """Load a JSON results file."""
    with open(json_path) as f:
        return json.load(f)


def compute_metrics(data: dict) -> list[dict[str, object]]:
    """Extract per-opponent metrics from a results file.

    Handles both MCTS and hybrid eval JSON formats.
    Returns a list of metric dicts, one per opponent entry.
    """
    rows: list[dict[str, object]] = []
    for entry in data.get("results", []):
        row: dict[str, object] = {
            "opponent": entry.get("opponent_type", "unknown"),
        }

        # Win rate — available directly or computed
        if "win_rate" in entry:
            row["win_pct"] = entry["win_rate"] * 100
        elif "wins" in entry and "games_played" in entry and entry["games_played"] > 0:
            row["win_pct"] = entry["wins"] / entry["games_played"] * 100
        elif "wins" in entry and "num_games" in entry and entry["num_games"] > 0:
            row["win_pct"] = entry["wins"] / entry["num_games"] * 100
        else:
            row["win_pct"] = 0.0

        # Draw rate
        games = entry.get("games_played", entry.get("num_games", 0))
        draws = entry.get("draws", 0)
        row["draw_pct"] = (draws / games * 100) if games > 0 else 0.0

        # Average game length
        row["avg_len"] = entry.get("avg_game_length", 0.0)

        # Trade metrics (hybrid only)
        trades_proposed = entry.get("total_trades_proposed", 0)
        row["trades"] = (
            trades_proposed / games if games > 0 else 0.0
        )
        row["accept_pct"] = entry.get("trade_acceptance_rate", 0.0) * 100

        # Simulations (MCTS only)
        row["sims"] = entry.get("simulations", None)

        rows.append(row)

    return rows


def print_comparison_table(runs: list[tuple[str, list[dict[str, object]]]]) -> None:
    """Print a formatted comparison table across all runs."""
    print()
    header = (
        f"{'Run':<28} {'Opp':<14} {'Win%':>6} {'Draw%':>6} "
        f"{'AvgLen':>7} {'Trades':>7} {'Accept%':>8}"
    )
    print(header)
    print("-" * len(header))

    for run_name, rows in runs:
        for row in rows:
            opp = str(row["opponent"])
            sims = row.get("sims")
            if sims is not None:
                opp = f"{opp} ({int(sims)}sim)"  # type: ignore[arg-type]

            win_pct: float = row["win_pct"]  # type: ignore[assignment]
            draw_pct: float = row["draw_pct"]  # type: ignore[assignment]
            avg_len: float = row["avg_len"]  # type: ignore[assignment]
            trades: float = row["trades"]  # type: ignore[assignment]
            accept_pct: float = row["accept_pct"]  # type: ignore[assignment]

            print(
                f"{run_name:<28} {opp:<14} "
                f"{win_pct:>5.1f}% {draw_pct:>5.1f}% "
                f"{avg_len:>7.0f} {trades:>7.1f} {accept_pct:>7.1f}%"
            )

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare evaluation results from JSON files",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="*",
        default=None,
        help="JSON result files to compare (default: auto-discover results/*.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Discover files
    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        results_dir = Path(__file__).parent.parent / "results"
        if not results_dir.exists():
            print(f"No results directory found at {results_dir}", file=sys.stderr)
            print("Use --files to specify JSON files, or run evaluations first.",
                  file=sys.stderr)
            sys.exit(1)
        paths = sorted(results_dir.glob("*.json"))

    if not paths:
        print("No JSON files found.", file=sys.stderr)
        sys.exit(1)

    runs: list[tuple[str, list[dict[str, object]]]] = []
    for path in paths:
        data = load_results(str(path))
        metrics = compute_metrics(data)
        run_name = path.stem
        runs.append((run_name, metrics))

    print_comparison_table(runs)


if __name__ == "__main__":
    main()
