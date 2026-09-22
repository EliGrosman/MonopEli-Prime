"""
Pydantic models for game actions.

Maps client action requests to monopoly_engine Action objects.
"""

from pydantic import BaseModel, ConfigDict, Field

from monopoly_engine.actions import (
    Action,
    BuildHotel,
    BuildHouse,
    BuyProperty,
    DeclareBankruptcy,
    EndTurn,
    MortgageProperty,
    PassBuy,
    PayJailFine,
    RollDice,
    SellBuildingGroup,
    SellHotel,
    SellHouse,
    UnmortgageProperty,
    UseJailCard,
)


class ActionRequest(BaseModel):
    """Generic action request from client.

    This model handles all action types via a single flexible schema.
    The action_type field determines which monopoly_engine Action to create.
    """

    model_config = ConfigDict(extra="forbid")

    action_type: str
    property_position: int | None = None
    contract_version: str | None = None
    expected_revision: int | None = Field(default=None, ge=0)
    request_id: str | None = Field(default=None, min_length=1, max_length=100)
    trade_id: int | None = None
    to_player: int | None = None
    give_properties: list[int] = Field(default_factory=list, max_length=2)
    give_money: int = Field(default=0, ge=0)
    want_properties: list[int] = Field(default_factory=list, max_length=2)
    want_money: int = Field(default=0, ge=0)

    def to_engine_action(self, player_id: int) -> Action:
        """Convert to monopoly_engine Action object.

        Args:
            player_id: The player performing the action

        Returns:
            The appropriate Action subclass instance

        Raises:
            ValueError: If action_type is unknown or required params missing
        """
        match self.action_type:
            case "pass_buy":
                return PassBuy(player_id)
            case "sell_building_group":
                if self.property_position is None:
                    raise ValueError("property_position required")
                return SellBuildingGroup(player_id, self.property_position)

            case "roll_dice":
                return RollDice(player_id=player_id)

            case "buy_property":
                if self.property_position is None:
                    raise ValueError("property_position required for buy_property")
                return BuyProperty(
                    player_id=player_id,
                    property_id=self.property_position,
                )

            case "build_house":
                if self.property_position is None:
                    raise ValueError("property_position required for build_house")
                return BuildHouse(
                    player_id=player_id,
                    property_id=self.property_position,
                )

            case "build_hotel":
                if self.property_position is None:
                    raise ValueError("property_position required for build_hotel")
                return BuildHotel(
                    player_id=player_id,
                    property_id=self.property_position,
                )

            case "sell_house":
                if self.property_position is None:
                    raise ValueError("property_position required for sell_house")
                return SellHouse(
                    player_id=player_id,
                    property_id=self.property_position,
                )

            case "sell_hotel":
                if self.property_position is None:
                    raise ValueError("property_position required for sell_hotel")
                return SellHotel(
                    player_id=player_id,
                    property_id=self.property_position,
                )

            case "mortgage_property":
                if self.property_position is None:
                    raise ValueError("property_position required for mortgage_property")
                return MortgageProperty(
                    player_id=player_id,
                    property_id=self.property_position,
                )

            case "unmortgage_property":
                if self.property_position is None:
                    raise ValueError("property_position required for unmortgage_property")
                return UnmortgageProperty(
                    player_id=player_id,
                    property_id=self.property_position,
                )

            case "pay_jail_fine":
                return PayJailFine(player_id=player_id)

            case "use_jail_card":
                return UseJailCard(player_id=player_id)

            case "end_turn":
                return EndTurn(player_id=player_id)

            case "propose_trade":
                if self.to_player is None:
                    raise ValueError("to_player required for propose_trade")
                from monopoly_engine import ProposeTrade

                return ProposeTrade(
                    player_id,
                    self.to_player,
                    list(self.give_properties),
                    self.give_money,
                    list(self.want_properties),
                    self.want_money,
                )

            case "accept_trade" | "reject_trade":
                if self.trade_id is None:
                    raise ValueError("trade_id required for a trade response")
                from monopoly_engine import AcceptTrade, RejectTrade

                cls = AcceptTrade if self.action_type == "accept_trade" else RejectTrade
                return cls(player_id, self.trade_id)

            case "declare_bankruptcy":
                return DeclareBankruptcy(player_id=player_id)

            case _:
                raise ValueError(f"Unknown action type: {self.action_type}")


class ActionResponse(BaseModel):
    """Response after executing an action."""

    success: bool
    message: str = ""
    action_type: str | None = None
