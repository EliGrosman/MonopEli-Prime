"""Summarize, select diagnostic extensions, and verify economic replays."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from evaluation.native_actions import decode_native_action
from evaluation.runner import invariant
from monopoly_engine import MonopolyGame
from monopoly_gym.action_space import ActionEncoder
from scripts.verify_foundation_run import DETERMINISTIC_FIELDS


def read_records(path):
    return [json.loads(line) for line in (path / "records.jsonl").read_text().splitlines()]


def select_long(records):
    """Purposive sample, never an estimate of population long-horizon completion.

    First three RB seed blocks in each observed structural category; then all
    unusual-color or recently-changing cutoff blocks across the three heuristics.
    Every selected block includes all focal seats, even those already completed.
    """
    blocks = {}
    for n in (2, 4):
        for label in ("no_complete_color_group", "developed_at_stop"):
            seeds = sorted(
                {
                    r["seed"]
                    for r in records
                    if len(r["policies"]) == n
                    and set(r["policies"]) == {"rule_based"}
                    and r["status"] == "cutoff"
                    and label in r["economics"]["labels"]
                }
            )[:3]
            for seed in seeds:
                blocks[n, seed, "rule_based"] = [label]
    for r in records:
        if r["status"] != "cutoff" or "random" in r["policies"]:
            continue
        final = r["economics"]["final"]
        colors = [g["color"] for g in final["groups"] if g["complete_owner"] is not None]
        reasons = []
        if final["turns_since_structure_change"] < 500:
            reasons.append("recent_structural_change")
        if any(c not in ("BROWN", "LIGHT_BLUE") for c in colors):
            reasons.append("other_developed_color")
        if reasons:
            blocks[len(r["policies"]), r["seed"], r["policies"][r["focal_seat"]]] = reasons
    jobs, selection = [], []
    for (n, seed, focal), reasons in sorted(blocks.items()):
        selection.append(dict(players=n, seed=seed, focal=focal, reasons=reasons))
        for seat in range(n):
            policies = ["rule_based"] * n
            policies[seat] = focal
            jobs.append(
                dict(policies=policies, seed=seed, focal_seat=seat, max_turns=10000, capture=True)
            )
    return jobs, selection


def prevalence(records):
    groups = {}
    for n in (2, 4):
        for opponent in ("heuristic", "random"):
            for status in ("completed", "cutoff", "error", "stalled"):
                rows = [
                    r
                    for r in records
                    if len(r["policies"]) == n
                    and r["status"] == status
                    and ("random" in r["policies"]) == (opponent == "random")
                ]
                if not rows:
                    continue
                labels = Counter()
                for r in rows:
                    labels.update(r["economics"]["labels"])
                group = {
                    "games": len(rows),
                    "seed_blocks": len({r["seed"] for r in rows}),
                    "labels": dict(labels),
                    "transitions": sum(r["decisions"] for r in rows),
                }
                group["median_final_cash"] = float(
                    np.median([sum(r["economics"]["final"]["cash"]) for r in rows])
                )
                group["final_complete_colors"] = dict(
                    Counter(
                        "+".join(
                            g["color"]
                            for g in r["economics"]["final"]["groups"]
                            if g["complete_owner"] is not None
                        )
                        or "none"
                        for r in rows
                    )
                )
                groups[f"{n}p-{opponent}-{status}"] = group
    return groups


def key(row):
    return len(row["policies"]), row["seed"], row["focal_seat"], tuple(row["policies"])


def paired_extensions(base, extended):
    indexed, pairs = {key(r): r for r in base}, []
    for r in extended:
        old = indexed[key(r)]
        a, b = old["economics"]["final"], r["economics"]["final"]
        pairs.append(
            {
                "seed": r["seed"],
                "players": len(r["policies"]),
                "seat": r["focal_seat"],
                "focal": r["policies"][r["focal_seat"]],
                "old_status": old["status"],
                "new_status": r["status"],
                "winner": r["winner"],
                "old_survivors": a["survivors"],
                "new_survivors": b["survivors"],
                "new_eliminations": len(a["survivors"]) - len(b["survivors"]),
                "cash_change": [y - x for x, y in zip(a["cash"], b["cash"], strict=True)],
                "same_final_property_structure": a["properties"] == b["properties"],
                "old_labels": old["economics"]["labels"],
                "new_labels": r["economics"]["labels"],
            }
        )
    return pairs


def verify_replays(path, records):
    encoder, checked, transitions = ActionEncoder(), 0, 0
    for row in records:
        if "replay_file" not in row:
            continue
        raw = (path / row["replay_file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row["replay_file_sha256"]
        saved = json.loads(raw)
        game = MonopolyGame.from_dict(saved["replay"]["initial"])
        digest = hashlib.sha256()
        for item in saved["replay"]["trace"]:
            action = (
                decode_native_action(item["native_action"])
                if "native_action" in item
                else encoder.decode(item["action"], item["actor"], game)
            )
            result = game.apply_action(item["actor"], action)
            assert result.events == tuple(item["events"])
            assert result.structured_events == tuple(item.get("structured_events", ()))
            assert result.revision == item["revision"]
            digest.update(json.dumps(item, sort_keys=True).encode())
            invariant(game)
            transitions += 1
        assert digest.hexdigest() == row["trace_sha256"]
        assert json.loads(json.dumps(game.to_dict())) == saved["final_snapshot"]
        assert game.winner == row["winner"]
        cash = [p["money"] for p in saved["replay"]["initial"]["players"]]
        for transaction in saved["transactions"]:
            if transaction["source"] is not None:
                cash[transaction["source"]] -= transaction["amount"]
            if transaction["target"] is not None:
                cash[transaction["target"]] += transaction["amount"]
        assert cash == [p.money for p in game.players]
        checked += 1
    return {"replays": checked, "transitions": transitions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--select-long", type=Path)
    parser.add_argument("--extended", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--verify-replays", action="store_true")
    args = parser.parse_args()
    records = read_records(args.run)
    output = {"prevalence": prevalence(records)}
    if args.select_long:
        jobs, selection = select_long(records)
        args.select_long.write_text(json.dumps(jobs, indent=2))
        output["long_selection"] = selection
    if args.extended:
        output["paired_extensions"] = paired_extensions(records, read_records(args.extended))
    if args.compare:
        other = read_records(args.compare)
        assert len(records) == len(other)
        for a, b in zip(records, other, strict=True):
            assert all(a[k] == b[k] for k in DETERMINISTIC_FIELDS), key(a)
        output["identical_repeated_games"] = len(records)
    if args.verify_replays:
        output["replay_verification"] = verify_replays(args.run, records)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
