"""Run the offline guided-Jev integration evaluation."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from evaluation.jev_runner import run_fake_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("fake",), required=True)
    parser.add_argument("--suite", choices=("smoke",), required=True)
    parser.add_argument("--seed-root", type=int, default=21_000_000)
    parser.add_argument("--horizon", type=int, default=1_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run_fake_smoke(args.output, args.seed_root, args.horizon))
    print(f"Wrote offline guided-Jev artifacts to {args.output}")


if __name__ == "__main__":
    main()
