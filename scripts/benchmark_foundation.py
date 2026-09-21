"""Run frozen, seat-rotated foundation matchups. No training or outcome adjudication."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.runner import play_game, summarize


def run(job):
    policies, seed, seat, horizon, capture = job
    return play_game(policies, seed, seat, horizon, capture)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument(
        "--namespace", choices=["development", "certification"], default="development"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.games <= 0 or args.games % 4:
        parser.error("games must be a positive multiple of four")
    args.output.mkdir(parents=True, exist_ok=False)
    root = 17000000 if args.namespace == "certification" else 11000000
    suites = {}
    for n in (2, 4):
        for focal in ("rule_based", "aggressive", "conservative"):
            for opponent in ("random", "rule_based"):
                label = f"{n}p-{focal}-vs-{opponent}"
                jobs = []
                for block in range(args.games // n):
                    for seat in range(n):
                        policies = [opponent] * n
                        policies[seat] = focal
                        jobs.append((policies, root + block, seat, args.horizon, block == 0))
                suites[label] = jobs

    def git(*parts):
        return subprocess.check_output(["git", *parts], text=True).strip()

    # Include untracked implementation files: git diff alone cannot identify a dirty run.
    sources = sorted(
        {
            p
            for directory in (
                "monopoly_engine",
                "monopoly_gym",
                "agents",
                "evaluation",
                "training",
                "scripts",
            )
            for p in Path(directory).rglob("*.py")
        }
    )
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    with tarfile.open(args.output / "source.tar.gz", "w:gz") as archive:
        for p in sources:
            archive.add(p, arcname=str(p))
        archive.add("uv.lock")
        archive.add("pyproject.toml")
    manifest = {
        "source_files": source_hashes,
        "source_archive_sha256": hashlib.sha256(
            (args.output / "source.tar.gz").read_bytes()
        ).hexdigest(),
        "commit": git("rev-parse", "HEAD"),
        "dirty": git("status", "--porcelain"),
        "lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        "source_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"])
        ).hexdigest(),
        "namespace": args.namespace,
        "suites": suites,
        "versions": [
            "foundation-v1",
            "action-v2",
            "observation-v2",
            "terminal-v1",
            "foundation-evaluation-v1",
        ],
        "bootstrap": {"samples": 2000, "seed": 12345, "unit": "seed block"},
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Frozen manifest:", hashlib.sha256(manifest_path.read_bytes()).hexdigest(), flush=True)
    summaries = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for name, jobs in suites.items():
            rows = list(pool.map(run, jobs, chunksize=10))
            with (args.output / f"{name}.jsonl").open("w") as out:
                for i, row in enumerate(rows):
                    if "replay" in row:
                        replay_path = args.output / f"{name}-{i}-replay.json"
                        replay_path.write_text(json.dumps(row.pop("replay")))
                        row["replay_path"] = replay_path.name
                        row["replay_sha256"] = hashlib.sha256(replay_path.read_bytes()).hexdigest()
                    out.write(json.dumps(row) + "\n")
            summaries[name] = summarize(rows)
            (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
            print(name, summaries[name], flush=True)


if __name__ == "__main__":
    main()
