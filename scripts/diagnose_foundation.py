"""Bounded DEVELOPMENT diagnostics; never a certification command.

Run: .venv/bin/python -m scripts.diagnose_foundation --output artifacts/milestone1b/pilot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tarfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from evaluation.economics import VERSION, diagnose_game


def run_job(job):
    return diagnose_game(**job)


def aggregate(records):
    groups = {}
    for record in records:
        key = f"{len(record['policies'])}p-{record['policies'][record['focal_seat']]}-vs-"
        others = [p for i, p in enumerate(record["policies"]) if i != record["focal_seat"]]
        key += others[0]
        group = groups.setdefault(key, {"games": 0, "statuses": Counter(), "by_status": {}})
        group["games"] += 1
        group["statuses"][record["status"]] += 1
        stats = group["by_status"].setdefault(
            record["status"],
            {
                "games": 0,
                "labels": Counter(),
                "flows": Counter(),
                "final_cash": 0,
                "survivors": 0,
                "declined_builds": 0,
                "rebuilds": 0,
                "defect_actions": 0,
                "transitions": 0,
            },
        )
        econ = record["economics"]
        stats["games"] += 1
        stats["labels"].update(econ["labels"])
        stats["flows"].update(econ["flows"])
        stats["final_cash"] += sum(econ["final"]["cash"])
        stats["survivors"] += len(econ["final"]["survivors"])
        stats["declined_builds"] += len(econ["declined_builds"])
        stats["rebuilds"] += len(econ["rebuilds"])
        stats["defect_actions"] += len(econ["defects"])
        stats["transitions"] += econ["ledger_checks"]
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12000000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--jobs-from", type=Path, help="Explicit selected development jobs JSON")
    args = parser.parse_args()
    if args.blocks < 1 or args.workers < 1 or args.horizon < 1:
        parser.error("blocks, workers and horizon must be positive")
    # Isolate this investigation from development-v1 and certification namespaces.
    if args.jobs_from:
        jobs = json.loads(args.jobs_from.read_text())
    else:
        jobs = []
        for n in (2, 4):
            for focal, opponent in (
                ("rule_based", "rule_based"),
                ("aggressive", "rule_based"),
                ("conservative", "rule_based"),
                ("rule_based", "random"),
            ):
                for seed in range(args.seed, args.seed + args.blocks):
                    for seat in range(n):
                        policies = [opponent] * n
                        policies[seat] = focal
                        jobs.append(
                            dict(
                                policies=policies,
                                seed=seed,
                                focal_seat=seat,
                                max_turns=args.horizon,
                                capture=seed == args.seed,
                            )
                        )
    if any(not 12000000 <= j["seed"] < 13000000 for j in jobs):
        parser.error("Only milestone1b development seeds [12000000, 13000000) are permitted")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        parser.error("Output already contains a run; use a fresh directory")
    sources = {}
    for directory in ("monopoly_engine", "monopoly_gym", "agents", "evaluation", "scripts"):
        for path in sorted(Path(directory).rglob("*.py")):
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "version": VERSION,
        "purpose": "development-only",
        "rules": "foundation-v1",
        "production_horizon_unchanged": 1000,
        "diagnostic_jobs": jobs,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--porcelain"], text=True),
        "python": platform.python_version(),
        "sources": sources,
        "uv_lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        "workers": args.workers,
    }
    source_archive = args.output / "source.tar.gz"
    with tarfile.open(source_archive, "w:gz") as archive:
        for name in sorted([*sources, "uv.lock", "pyproject.toml"]):
            archive.add(name, arcname=name)
    manifest["source_archive_sha256"] = hashlib.sha256(source_archive.read_bytes()).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    records, start = [], time.monotonic()
    with (
        ProcessPoolExecutor(args.workers) as pool,
        (args.output / "records.jsonl").open("w") as out,
    ):
        for index, result in enumerate(pool.map(run_job, jobs)):
            if "replay" in result:
                filename = f"replay-{index:04d}.json"
                replay = {
                    "replay": result.pop("replay"),
                    "final_snapshot": result.pop("final_snapshot"),
                    "transactions": result["economics"].pop("transactions", []),
                }
                data = json.dumps(replay, sort_keys=True)
                (args.output / filename).write_text(data)
                result["replay_file"] = filename
                result["replay_file_sha256"] = hashlib.sha256(data.encode()).hexdigest()
            out.write(json.dumps(result, sort_keys=True) + "\n")
            out.flush()
            records.append(result)
            if (index + 1) % 12 == 0:
                print(f"{index + 1}/{len(jobs)} games; {time.monotonic() - start:.1f}s", flush=True)
    summary = {
        "groups": aggregate(records),
        "runtime_seconds": time.monotonic() - start,
        "games": len(records),
        "transitions": sum(r["decisions"] for r in records),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if any(r["status"] in ("error", "stalled") for r in records):
        raise SystemExit("Diagnostic run contains errors/stalls; inspect records")


if __name__ == "__main__":
    main()
