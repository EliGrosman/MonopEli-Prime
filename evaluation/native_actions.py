"""Versioned replay codec for parameterized headless actions."""

from __future__ import annotations

from typing import Any

from monopoly_engine import (
    AcceptTrade,
    BuildHotel,
    BuildHouse,
    BuyProperty,
    DeclareBankruptcy,
    EndTurn,
    MortgageProperty,
    PayJailFine,
    ProposeTrade,
    RejectTrade,
    RollDice,
    SellHotel,
    SellHouse,
    UnmortgageProperty,
    UseJailCard,
)
from monopoly_engine.actions import PassBuy, SellBuildingGroup

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
    simple = {
        "RollDice": RollDice,
        "PayJailFine": PayJailFine,
        "UseJailCard": UseJailCard,
        "DeclareBankruptcy": DeclareBankruptcy,
        "EndTurn": EndTurn,
        "PassBuy": PassBuy,
    }
    if action_type in simple:
        return simple[action_type](player_id)
    property_actions = {
        "BuyProperty": BuyProperty,
        "BuildHouse": BuildHouse,
        "BuildHotel": BuildHotel,
        "SellHouse": SellHouse,
        "SellHotel": SellHotel,
        "SellBuildingGroup": SellBuildingGroup,
        "MortgageProperty": MortgageProperty,
        "UnmortgageProperty": UnmortgageProperty,
    }
    if action_type in property_actions:
        property_id = data.get("property_id")
        if type(property_id) is not int:
            raise ValueError("Property action requires an integer property_id")
        return property_actions[action_type](player_id, property_id)
    raise ValueError(f"Unsupported native action type: {action_type}")
