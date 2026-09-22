"""Verify frozen milestone 1d records, gates, and captured AEC replays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from monopoly_engine import MonopolyGame
from monopoly_gym.action_space import ActionEncoder


def verify_learner_coverage(records: list[dict]) -> None:
    """Require the lifecycle cases that a consumer certification claims to exercise."""
    for consumer in ("single_agent", "self_play"):
        rows = [row for row in records if row["consumer"] == consumer]
        assert sum(row["learner_responses"] for row in rows) > 0, (consumer, "no responses")
        assert {row["learner_reward"] for row in rows} >= {-1.0, 1.0}, (
            consumer,
            "missing wins or losses",
        )
        assert any(
            row["label"].startswith("4p-")
            and row["learner_reward"] == -1
            and row["learner_terminal_revision"] < row["decisions"]
            for row in rows
        ), (consumer, "no four-player elimination before game completion")


def verify_consumers(run: Path, consumer_run: Path, *, allow_small: bool = False) -> int:
    from scripts.certify_trading_consumers import COMPARISON_FIELDS, CONSUMERS, select_records

    manifest = json.loads((consumer_run / "manifest.json").read_text())
    assert manifest["purpose"] == "milestone1d-shared-consumer-parity"
    assert set(manifest["consumers"]) == set(CONSUMERS)
    assert allow_small or manifest["blocks_per_matchup"] >= 20
    for name in ("records", "manifest"):
        suffix = "jsonl" if name == "records" else "json"
        assert (
            hashlib.sha256((run / f"{name}.{suffix}").read_bytes()).hexdigest()
            == manifest[f"reference_{name}_sha256"]
        )
    assert (
        hashlib.sha256((consumer_run / "source.tar.gz").read_bytes()).hexdigest()
        == manifest["source_archive_sha256"]
    )
    reference = select_records(run, manifest["blocks_per_matchup"])
    index = {(r["label"], r["seed"], r["focal_seat"]): r for r in reference}
    records = [
        json.loads(line) for line in (consumer_run / "records.jsonl").read_text().splitlines()
    ]
    expected = {(*key, consumer) for key in index for consumer in CONSUMERS}
    actual = {(r["label"], r["seed"], r["focal_seat"], r["consumer"]) for r in records}
    assert actual == expected and len(records) == len(expected), "Consumer coverage is incomplete"
    assert manifest["games_per_consumer"] == len(reference)
    for row in records:
        paired = index[(row["label"], row["seed"], row["focal_seat"])]
        assert all(row[field] == paired[field] for field in COMPARISON_FIELDS)
        if row["consumer"] != "api_bot":
            assert row["learner_boundaries"] > 0
            expected_reward = 1.0 if paired["winner"] == row["focal_seat"] else -1.0
            if paired["status"] == "completed":
                assert row["learner_reward"] == expected_reward
    summary = json.loads((consumer_run / "summary.json").read_text())
    assert summary["games"] == len(records) and summary["mismatches"] == 0
    assert summary["checked_transitions"] == sum(row["decisions"] for row in records)
    assert (
        summary["records_sha256"]
        == hashlib.sha256((consumer_run / "records.jsonl").read_bytes()).hexdigest()
    )
    if not allow_small:
        verify_learner_coverage(records)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--allow-small", action="store_true")
    parser.add_argument("--consumer-run", type=Path)
    parser.add_argument(
        "--aec-only",
        action="store_true",
        help="Verify historical AEC evidence only; does not certify shared consumers",
    )
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    assert (
        hashlib.sha256((args.run / "source.tar.gz").read_bytes()).hexdigest()
        == manifest["source_archive_sha256"]
    )
    for lock, expected in manifest["lock_sha256"].items():
        # The source archive is authoritative for frozen dependencies; current locks are
        # checked as an additional guard when verifying in the source checkout.
        current = Path(lock)
        if current.exists():
            assert hashlib.sha256(current.read_bytes()).hexdigest() == expected
    records = [json.loads(line) for line in (args.run / "records.jsonl").read_text().splitlines()]
    assert len(records) == manifest["games_per_matchup"] * 12
    assert not [row for row in records if row["status"] == "error"]
    assert args.allow_small or sum(row["decisions"] for row in records) >= 100_000
    control_rows = [
        json.loads(line)
        for line in (args.run / "controls" / "no-trade.jsonl").read_text().splitlines()
    ]
    assert len(control_rows) == len(records)
    repeat_report = json.loads((args.run / "repeat-verification.json").read_text())
    assert repeat_report["maximum_blocks_per_matchup"] == 2
    assert repeat_report["games"] > 0
    assert args.allow_small or repeat_report["games"] == 72
    assert not repeat_report["mismatches"]
    for label in manifest["matchups"]:
        group = [row for row in records if row["label"] == label]
        rate = sum(row["status"] == "completed" for row in group) / len(group)
        assert rate >= manifest["completion_gate"], (label, rate)
    for row in records:
        if "replay_path" not in row:
            continue
        replay = json.loads((args.run / row["replay_path"]).read_text())
        assert replay["initial"]["foundation"]["revision"] == 0
        game = MonopolyGame.from_dict(replay["initial"])
        encoder = ActionEncoder(rules_id="foundation-trade-v1")
        for item in replay["trace"]:
            action = encoder.decode(item["action"], item["actor"], game)
            game.apply_action(item["actor"], action)
        restored = json.loads(json.dumps(game.to_dict()))
        assert restored == replay["final"]
        digest = hashlib.sha256(json.dumps(replay["trace"], sort_keys=True).encode()).hexdigest()
        assert digest == row["trace_sha256"]
    if manifest["namespace"] == "development":
        baseline = {(row["label"], row["seed"], row["focal_seat"]): row for row in control_rows}
        rejected = [
            json.loads(line)
            for line in (args.run / "controls" / "always-reject.jsonl").read_text().splitlines()
        ]
        for row in rejected:
            paired = baseline[(row["label"], row["seed"], row["focal_seat"])]
            for field in ("status", "winner", "turns", "elimination_order", "cash"):
                assert row[field] == paired[field], (field, row, paired)
    print(f"verified {len(records)} AEC games and {sum(r['decisions'] for r in records)} decisions")
    if not args.aec_only:
        count = verify_consumers(
            args.run,
            args.consumer_run or args.run / "consumer-parity",
            allow_small=args.allow_small,
        )
        print(f"verified {count} learner/API consumer games")


if __name__ == "__main__":
    main()
