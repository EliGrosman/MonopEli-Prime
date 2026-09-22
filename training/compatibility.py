"""Checkpoint compatibility metadata for learner-facing rule contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def learner_contract(rules_id: str, num_players: int) -> dict[str, Any]:
    if rules_id == "foundation-trade-v1":
        return {
            "rules_id": rules_id,
            "action_version": "action-v3",
            "observation_version": "observation-v3",
            "num_players": num_players,
            "candidate_provider": "foundation-benefit-candidates-v1",
            "candidate_capacity": 32,
        }
    if rules_id != "foundation-v1":
        raise ValueError(f"Unsupported rules ID: {rules_id}")
    return {
        "rules_id": rules_id,
        "action_version": "action-v2",
        "observation_version": "observation-v2",
        "num_players": num_players,
        "candidate_provider": None,
        "candidate_capacity": 0,
    }


def metadata_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    if path.suffix == ".zip":
        path = path.with_suffix("")
    return Path(f"{path}.metadata.json")


def write_checkpoint_metadata(model_path: str | Path, rules_id: str, num_players: int) -> None:
    metadata_path(model_path).write_text(
        json.dumps(learner_contract(rules_id, num_players), indent=2, sort_keys=True) + "\n"
    )


def validate_checkpoint_metadata(
    model_path: str | Path,
    rules_id: str,
    num_players: int,
    *,
    allow_legacy_v2: bool = False,
) -> None:
    expected = learner_contract(rules_id, num_players)
    path = metadata_path(model_path)
    if not path.exists():
        if allow_legacy_v2 and rules_id == "foundation-v1":
            return
        raise ValueError(f"Checkpoint compatibility metadata is missing: {path}")
    actual = json.loads(path.read_text())
    if actual != expected:
        raise ValueError(f"Checkpoint contract mismatch: expected {expected}, got {actual}")
