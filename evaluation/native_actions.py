"""Versioned replay codec for parameterized headless actions."""

from __future__ import annotations

from typing import Any

from monopoly_engine import AcceptTrade, ProposeTrade, RejectTrade

NATIVE_ACTION_VERSION = "native-action-v1"


def decode_native_action(data: dict[str, Any]):
    action_type = data.get("type")
    player_id = data.get("player_id")
    if type(player_id) is not int:
        raise ValueError("Native action requires an integer player_id")
    if action_type == "ProposeTrade":
        return ProposeTrade(
            player_id,
            data["to_player"],
            list(data["give_properties"]),
            data["give_money"],
            list(data["want_properties"]),
            data["want_money"],
        )
    if action_type == "AcceptTrade":
        return AcceptTrade(player_id, data["trade_id"])
    if action_type == "RejectTrade":
        return RejectTrade(player_id, data["trade_id"])
    raise ValueError(f"Unsupported native action type: {action_type}")
