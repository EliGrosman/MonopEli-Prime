"""Verify guided-Jev execution artifacts without making provider calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evaluation.native_actions import decode_native_action
from evaluation.runner import invariant
from monopoly_engine import MonopolyGame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    assert manifest["evaluator_version"] == "guided-jev-evaluation-v1"
    assert manifest["live_inference"] is False
    records = [
        json.loads(line)
        for line in (args.run / "records.jsonl").read_text().splitlines()
        if line
    ]
    assert len(records) == manifest["games"]
    transitions = 0
    for record in records:
        replay_path = args.run / record["replay_path"]
        content = replay_path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == record["replay_sha256"]
        replay = json.loads(content)
        game = MonopolyGame.from_dict(replay["initial"])
        for item in replay["trace"]:
            assert game.state.revision == item["before_revision"]
            action = decode_native_action(item["action"])
            result = game.apply_action(
                item["actor"], action, expected_revision=item["before_revision"]
            )
            assert result.revision == item["after_revision"]
            assert list(result.events) == item["events"]
            assert list(result.structured_events) == item["structured_events"]
            invariant(game)
            transitions += 1
        assert json.dumps(game.to_dict(), sort_keys=True) == json.dumps(
            replay["final"], sort_keys=True
        )
        assert game.winner == record["winner"]
    print(f"Verified {len(records)} games and {transitions} executed commands")


if __name__ == "__main__":
    main()
