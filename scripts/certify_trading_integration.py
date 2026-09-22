"""Freeze and run the foundation-trade-v1 AEC integration certification."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import subprocess
import tarfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from agents import AggressiveAgent, ConservativeAgent, RandomAgent, RuleBasedAgent
from agents.trading_agent import TradingAgent
from monopoly_engine import MonopolyGame
from monopoly_gym import MonopolyEnv

POLICIES = {
    "random": RandomAgent,
    "rule_based": RuleBasedAgent,
    "aggressive": AggressiveAgent,
    "conservative": ConservativeAgent,
}


def _agent(
    name: str, player_id: int, seed: int, response: str | None
) -> TradingAgent | RandomAgent | RuleBasedAgent | AggressiveAgent | ConservativeAgent:
    cls = POLICIES[name]
    base = cls(player_id, seed=seed) if cls is RandomAgent else cls(player_id)
    return TradingAgent(base, response=response) if response is not None else base


def run_game(job: dict[str, Any]) -> dict[str, Any]:
    policies = job["policies"]
    seed = job["seed"]
    focal = job["focal_seat"]
    rules_id = job["rules_id"]
    response = job["response"]
    env = MonopolyEnv(
        len(policies),
        max_turns=1000,
        cutoff_player=focal,
        rules_id=rules_id,
    )
    env.reset(options={"engine_seed": seed})
    assert env.game is not None
    initial = copy.deepcopy(env.game.to_dict())
    direct = MonopolyGame.from_dict(copy.deepcopy(initial))
    agents = [_agent(name, i, seed * 11 + i, response) for i, name in enumerate(policies)]
    for agent in agents:
        agent.reset()
    trace: list[dict[str, int]] = []
    error = None
    try:
        while not env.game.game_over and not any(env.truncations.values()):
            agent_name = env.agent_selection
            pid = env.agent_name_mapping[agent_name]
            if env.terminations.get(agent_name) or env.truncations.get(agent_name):
                env.step(None)
                continue
            if hasattr(agents[pid], "choose_native_action"):
                native = agents[pid].choose_native_action(env.game, env.action_encoder)
                action_index = env.action_encoder.encode_current(native, env.game, pid)
            else:
                native = agents[pid].choose_decision(env.game.decision_view(pid))
                action_index = env.action_encoder.encode_current(native, env.game, pid)
            direct_action = env.action_encoder.decode(action_index, pid, direct)
            direct.apply_action(pid, direct_action)
            env.step(action_index)
            if env.game.to_dict() != direct.to_dict():
                raise AssertionError("AEC and direct engine states diverged")
            trace.append({"actor": pid, "action": action_index})
    except Exception as exc:  # recorded as invalid certification evidence
        error = f"{type(exc).__name__}: {exc}"
    status = "error" if error else "completed" if env.game.game_over else "cutoff"
    structured = env.game.state.structured_event_log
    result: dict[str, Any] = {
        "label": job["label"],
        "arm": job["arm"],
        "rules_id": rules_id,
        "policies": policies,
        "seed": seed,
        "focal_seat": focal,
        "status": status,
        "winner": env.game.winner,
        "turns": env.game.turn_number,
        "decisions": len(trace),
        "error": error,
        "elimination_order": env.game.state.elimination_order,
        "cash": [player.money for player in env.game.players],
        "trading": [dict(getattr(agent, "stats", {})) for agent in agents],
        "proposals": sum(event["type"] == "trade_proposed" for event in structured),
        "acceptances": sum(event["type"] == "trade_accepted" for event in structured),
        "rejections": sum(event["type"] == "trade_rejected" for event in structured),
        "trace_sha256": hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest(),
    }
    if job["capture"]:
        result["replay"] = {
            "initial": initial,
            "trace": trace,
            "final": copy.deepcopy(env.game.to_dict()),
        }
    return result


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", choices=("development", "certification"), required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.games <= 0 or args.games % 4:
        parser.error("games must be a positive multiple of four")
    args.output.mkdir(parents=True, exist_ok=False)
    seed_root = 18_000_000 if args.namespace == "development" else 19_000_000
    jobs: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    for players in (2, 4):
        for focal_name in ("rule_based", "aggressive", "conservative"):
            for opponent in ("random", "rule_based"):
                label = f"{players}p-{focal_name}-vs-{opponent}"
                for block in range(args.games // players):
                    for seat in range(players):
                        policies = [opponent] * players
                        policies[seat] = focal_name
                        common = {
                            "label": label,
                            "block": block,
                            "policies": policies,
                            "seed": seed_root + block,
                            "focal_seat": seat,
                            "capture": block == 0 and seat == 0,
                        }
                        jobs.append(
                            {
                                **common,
                                "arm": "mutual",
                                "rules_id": "foundation-trade-v1",
                                "response": "mutual",
                            }
                        )
                        controls.append(
                            {
                                **common,
                                "arm": "no-trade",
                                "rules_id": "foundation-v1",
                                "response": None,
                            }
                        )
                        if args.namespace == "development" and block == 0:
                            controls.extend(
                                [
                                    {
                                        **common,
                                        "arm": "always-reject",
                                        "rules_id": "foundation-trade-v1",
                                        "response": "reject",
                                    },
                                    {
                                        **common,
                                        "arm": "mixed-response",
                                        "rules_id": "foundation-trade-v1",
                                        "response": "alternate",
                                    },
                                ]
                            )
    sources = sorted(
        path
        for root in (
            "monopoly_engine",
            "monopoly_gym",
            "agents",
            "api",
            "evaluation",
            "training",
            "scripts",
            "frontend/src",
            "frontend/e2e",
            "tests/foundation",
        )
        for path in Path(root).rglob("*")
        if path.is_file()
    )
    with tarfile.open(args.output / "source.tar.gz", "w:gz") as archive:
        for path in sources + [Path("uv.lock"), Path("frontend/package-lock.json")]:
            archive.add(path, arcname=str(path))
    manifest = {
        "namespace": args.namespace,
        "commit": _git("rev-parse", "HEAD"),
        "dirty": _git("status", "--porcelain"),
        "seed_root": seed_root,
        "games_per_matchup": args.games,
        "matchups": sorted({job["label"] for job in jobs}),
        "versions": [
            "foundation-trade-v1",
            "decision-contract-v1",
            "action-v3",
            "observation-v3",
            "foundation-benefit-candidates-v1",
            "terminal-v1",
        ],
        "reference_policies": {
            "ordinary": ["rule_based", "aggressive", "conservative", "random"],
            "trading_wrapper": "mutual-benefit-v1",
            "minimum_gain": 25,
            "cash_reserve": 200,
            "maximum_cash_adjustment": 500,
        },
        "seed_schedule": "seed_root + block; every focal seat shares its complete seed block",
        "horizon": 1000,
        "completion_gate": 0.95,
        "bootstrap": {"samples": 2000, "seed": 12345, "unit": "seed_block"},
        "control_games_per_matchup": args.games,
        "diagnostic_schedule": (
            "one seat-balanced seed block per matchup" if args.namespace == "development" else None
        ),
        "source_archive_sha256": hashlib.sha256(
            (args.output / "source.tar.gz").read_bytes()
        ).hexdigest(),
        "lock_sha256": {
            "uv.lock": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
            "frontend/package-lock.json": hashlib.sha256(
                Path("frontend/package-lock.json").read_bytes()
            ).hexdigest(),
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run_game, jobs, chunksize=4))
        control_rows = list(pool.map(run_game, controls, chunksize=4))
        repeat_rows = list(
            pool.map(run_game, [job for job in jobs if job["block"] < 2], chunksize=4)
        )
    replay_dir = args.output / "replays"
    replay_dir.mkdir()
    with (args.output / "records.jsonl").open("w") as output:
        for index, row in enumerate(rows):
            replay = row.pop("replay", None)
            if replay is not None:
                path = replay_dir / f"{index}.json"
                path.write_text(json.dumps(replay))
                row["replay_path"] = str(path.relative_to(args.output))
            output.write(json.dumps(row) + "\n")
    summary = {}
    bootstrap_rng = random.Random(12345)
    for label in manifest["matchups"]:
        group = [row for row in rows if row["label"] == label]
        completed = sum(row["status"] == "completed" for row in group)
        by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_seed[row["seed"]].append(row)
        seed_blocks = list(by_seed.values())
        bootstrap_rates = []
        for _ in range(2000):
            sampled = [bootstrap_rng.choice(seed_blocks) for _ in seed_blocks]
            sampled_rows = [row for block_rows in sampled for row in block_rows]
            bootstrap_rates.append(
                sum(row["status"] == "completed" for row in sampled_rows) / len(sampled_rows)
            )
        bootstrap_rates.sort()
        summary[label] = {
            "games": len(group),
            "completed": completed,
            "completion_rate": completed / len(group),
            "errors": sum(row["status"] == "error" for row in group),
            "cutoffs": sum(row["status"] == "cutoff" for row in group),
            "transitions": sum(row["decisions"] for row in group),
            "proposals": sum(row["proposals"] for row in group),
            "acceptances": sum(row["acceptances"] for row in group),
            "rejections": sum(row["rejections"] for row in group),
            "completion_bootstrap_95": [
                bootstrap_rates[49],
                bootstrap_rates[1949],
            ],
        }
    repeat_fields = (
        "label",
        "arm",
        "rules_id",
        "policies",
        "seed",
        "focal_seat",
        "status",
        "winner",
        "turns",
        "decisions",
        "error",
        "elimination_order",
        "cash",
        "trading",
        "proposals",
        "acceptances",
        "rejections",
        "trace_sha256",
    )
    original_repeats = {
        (row["label"], row["seed"], row["focal_seat"]): row
        for row in rows
        if row["seed"] < seed_root + 2
    }
    repeat_mismatches = []
    for repeated in repeat_rows:
        key = (repeated["label"], repeated["seed"], repeated["focal_seat"])
        original = original_repeats[key]
        differences = [field for field in repeat_fields if repeated[field] != original[field]]
        if differences:
            repeat_mismatches.append({"key": key, "fields": differences})
    repeat_report = {
        "games": len(repeat_rows),
        "maximum_blocks_per_matchup": 2,
        "mismatches": repeat_mismatches,
    }
    (args.output / "repeat-verification.json").write_text(
        json.dumps(repeat_report, indent=2) + "\n"
    )
    controls_dir = args.output / "controls"
    controls_dir.mkdir()
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in control_rows:
        by_arm[row["arm"]].append(row)
    for arm, arm_rows in by_arm.items():
        with (controls_dir / f"{arm}.jsonl").open("w") as output:
            for row in arm_rows:
                row.pop("replay", None)
                output.write(json.dumps(row) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
