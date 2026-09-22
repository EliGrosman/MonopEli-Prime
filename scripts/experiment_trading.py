"""Run frozen headless foundation-trade-v1 development/validation experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tarfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from evaluation.economics import diagnose_game

ARMS = {
    "baseline": {"rules_id": "foundation-v1", "trading": None},
    "rejection": {"rules_id": "foundation-trade-v1", "trading": "reject"},
    "mutual": {"rules_id": "foundation-trade-v1", "trading": "mutual"},
}
FOCAL_POLICIES = ("rule_based", "aggressive", "conservative")


def run_job(job):
    label = job.pop("job_label")
    result = diagnose_game(**job)
    result["job_label"] = label
    return result


def create_jobs(blocks: int, seed: int, arms: tuple[str, ...]):
    jobs = []
    for arm in arms:
        for players in (2, 4):
            for focal in FOCAL_POLICIES:
                for block in range(blocks):
                    for seat in range(players):
                        policies = ["rule_based"] * players
                        policies[seat] = focal
                        jobs.append(
                            {
                                "job_label": f"{arm}:{players}p:{focal}",
                                "policies": policies,
                                "seed": seed + block,
                                "focal_seat": seat,
                                "max_turns": 1000,
                                "capture": block == 0,
                                **ARMS[arm],
                            }
                        )
    return jobs


def summarize(records):
    groups = {}
    for label in sorted({r["job_label"] for r in records}):
        rows = [r for r in records if r["job_label"] == label]
        counts = Counter(r["status"] for r in rows)
        diagnostics = Counter()
        for row in rows:
            for agent in row["trading_diagnostics"]:
                diagnostics.update(agent)
        first_build_turns = [
            row["economics"].get("first_build_turn")
            for row in rows
            if row["economics"].get("first_build_turn") is not None
        ]
        first_group_turns = [
            row["economics"].get("first_complete_group_turn")
            for row in rows
            if row["economics"].get("first_complete_group_turn") is not None
        ]
        groups[label] = {
            "games": len(rows),
            "statuses": dict(counts),
            "completion_rate": counts["completed"] / len(rows),
            "wins": sum(r["win"] for r in rows),
            "eliminations": sum(r["eliminated"] for r in rows),
            "transitions": sum(r["decisions"] for r in rows),
            "split_group_cutoffs": sum(
                r["status"] == "cutoff" and "no_complete_color_group" in r["economics"]["labels"]
                for r in rows
            ),
            "median_first_complete_group_turn": (
                float(np.median(first_group_turns)) if first_group_turns else None
            ),
            "median_first_build_turn": (
                float(np.median(first_build_turns)) if first_build_turns else None
            ),
            "trade_diagnostics": dict(diagnostics),
        }

    indexed = defaultdict(dict)
    for row in records:
        arm = row["job_label"].split(":", 1)[0]
        key = (len(row["policies"]), row["seed"], row["focal_seat"], tuple(row["policies"]))
        indexed[key][arm] = row
    rejection_mismatches = []
    for key, arms in indexed.items():
        if not {"baseline", "rejection"} <= arms.keys():
            continue
        base, reject = arms["baseline"], arms["rejection"]
        fields = (
            "status",
            "winner",
            "turns",
            "diagnostic_cash",
            "elimination_order",
            "gameplay_trace_sha256",
        )
        if any(base[field] != reject[field] for field in fields):
            rejection_mismatches.append(
                {
                    "key": key,
                    "differences": {
                        field: [base[field], reject[field]]
                        for field in fields
                        if base[field] != reject[field]
                    },
                }
            )

    effects = {}
    rng = np.random.default_rng(130013)
    for players in (2, 4):
        block_values = defaultdict(lambda: {"baseline": [], "mutual": []})
        for row in records:
            arm = row["job_label"].split(":", 1)[0]
            if len(row["policies"]) == players and arm in ("baseline", "mutual"):
                block_values[row["seed"]][arm].append(int(row["status"] == "completed"))
        paired = [
            np.mean(v["mutual"]) - np.mean(v["baseline"])
            for v in block_values.values()
            if v["baseline"] and v["mutual"]
        ]
        if paired:
            values = np.asarray(paired)
            draws = rng.choice(values, (5000, len(values)), replace=True).mean(axis=1)
            effects[f"{players}p_completion_difference"] = {
                "point": float(values.mean()),
                "paired_block_95ci": np.quantile(draws, [0.025, 0.975]).tolist(),
                "blocks": len(values),
            }
    return {
        "groups": groups,
        "rejection_control_mismatches": rejection_mismatches,
        "paired_effects": effects,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13000000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=tuple(ARMS))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory must not already exist")
    if args.blocks < 1 or args.workers < 1:
        parser.error("blocks and workers must be positive")
    if not (13000000 <= args.seed < 15000000):
        parser.error("Trading experiments require the milestone 1c seed namespaces")
    arms = tuple(args.arms)
    jobs = create_jobs(args.blocks, args.seed, arms)
    args.output.mkdir(parents=True)

    sources = {}
    for directory in ("monopoly_engine", "monopoly_gym", "agents", "evaluation", "scripts"):
        for path in sorted(Path(directory).rglob("*.py")):
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    archive_path = args.output / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in [*sources, "uv.lock", "pyproject.toml"]:
            archive.add(path, arcname=path)
    manifest = {
        "purpose": "milestone-1c-headless-trading-experiment",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--porcelain"], text=True),
        "python": platform.python_version(),
        "jobs": jobs,
        "sources": sources,
        "source_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "uv_lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        "policy_constants": {
            "min_gain": 25,
            "cash_cap": 500,
            "cash_reserve": 200,
            "complete_group_bonus": "2x printed purchase price",
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))

    records, start = [], time.monotonic()
    with (
        ProcessPoolExecutor(args.workers) as pool,
        (args.output / "records.jsonl").open("w") as out,
    ):
        for index, result in enumerate(pool.map(run_job, [dict(job) for job in jobs])):
            if "replay" in result:
                replay_name = f"replay-{index:04d}.json"
                replay = {
                    "replay": result.pop("replay"),
                    "final_snapshot": result.pop("final_snapshot"),
                    "transactions": result["economics"].pop("transactions", []),
                }
                content = json.dumps(replay, sort_keys=True)
                (args.output / replay_name).write_text(content)
                result["replay_file"] = replay_name
                result["replay_file_sha256"] = hashlib.sha256(content.encode()).hexdigest()
            out.write(json.dumps(result, sort_keys=True) + "\n")
            out.flush()
            records.append(result)
            if (index + 1) % 30 == 0:
                print(f"{index + 1}/{len(jobs)} games; {time.monotonic() - start:.1f}s", flush=True)
    summary = summarize(records)
    summary.update(
        games=len(records),
        transitions=sum(r["decisions"] for r in records),
        runtime_seconds=time.monotonic() - start,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if any(r["status"] in ("error", "stalled") for r in records):
        raise SystemExit("Experiment contains errors or stalls")
    if "rejection" in arms and "baseline" in arms and summary["rejection_control_mismatches"]:
        raise SystemExit("Rejection control changed ordinary gameplay")


if __name__ == "__main__":
    main()
