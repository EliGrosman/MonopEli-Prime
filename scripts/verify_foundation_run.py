"""Verify benchmark replay checksums and optionally compare repeated runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from monopoly_engine import MonopolyGame
from monopoly_gym.action_space import ActionEncoder

DETERMINISTIC_FIELDS = (
    "seed",
    "derived_seeds",
    "focal_seat",
    "policies",
    "status",
    "winner",
    "game_over",
    "win",
    "loss",
    "eliminated",
    "elimination_order",
    "turns",
    "decisions",
    "horizon",
    "overshoot",
    "error",
    "trace_sha256",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    games = replays = transitions = 0
    encoder = ActionEncoder()
    paths = sorted(args.run.glob("*.jsonl"))
    if not paths:
        raise ValueError("No result records")
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if args.compare:
            other = [
                json.loads(line) for line in (args.compare / path.name).read_text().splitlines()
            ]
            assert len(rows) == len(other)
            for a, b in zip(rows, other, strict=True):
                assert all(a[key] == b[key] for key in DETERMINISTIC_FIELDS), (
                    path.name,
                    a["seed"],
                    a["focal_seat"],
                )
        for row in rows:
            games += 1
            if "replay_path" not in row:
                continue
            content = (args.run / row["replay_path"]).read_bytes()
            assert hashlib.sha256(content).hexdigest() == row["replay_sha256"]
            replay = json.loads(content)
            game = MonopolyGame.from_dict(replay["initial"])
            for item in replay["trace"]:
                result = game.apply_action(
                    item["actor"], encoder.decode(item["action"], item["actor"], game)
                )
                assert tuple(item["events"]) == result.events
                assert item["revision"] == result.revision
                transitions += 1
            assert (game.winner, game.turn_number) == (row["winner"], row["turns"])
            replays += 1
    print(f"Verified {games} records, {replays} replays, {transitions} replay transitions")
    if args.compare:
        print("Repeated outcomes and trace hashes match exactly (timing excluded)")


if __name__ == "__main__":
    main()
