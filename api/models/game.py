"""
Pydantic models for game state and configuration.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from monopoly_engine import MonopolyGame


class PlayerState(BaseModel):
    """Player state visible to clients."""

    id: int
    name: str
    money: int
    position: int
    in_jail: bool
    jail_turns: int
    jail_cards: int
    bankrupt: bool
    properties: list[int]  # Property positions owned


class PropertyState(BaseModel):
    """Property state visible to clients."""

    position: int
    owner: int | None
    houses: int
    mortgaged: bool


class PlayerSlot(BaseModel):
    """Player slot in a game/lobby."""

    player_id: int
    name: str
    session_id: str | None = None
    is_ai: bool = False
    ai_type: str | None = None
    is_ready: bool = False
    disconnected_at: datetime | None = None  # For reconnection window tracking


class GameState(BaseModel):
    """Full game state sent to clients."""

    players: list[PlayerState]
    properties: dict[str, PropertyState]  # String keys for JSON compatibility
    current_player: int
    turn_number: int
    houses_remaining: int
    hotels_remaining: int
    last_roll: tuple[int, int] | None
    doubles_count: int
    game_over: bool
    winner: int | None
    event_log: list[str]

    @classmethod
    def from_engine(
        cls,
        game: MonopolyGame,
        player_slots: dict[int, PlayerSlot] | None = None,
    ) -> "GameState":
        """Convert from engine game state.

        Args:
            game: The MonopolyGame instance
            player_slots: Optional player slot info (for session tracking)

        Returns:
            GameState instance
        """
        state = game.state.to_dict()
        pm = game.property_manager

        return cls(
            players=[
                PlayerState(
                    id=p["id"],
                    name=p["name"],
                    money=p["money"],
                    position=p["position"],
                    in_jail=p["in_jail"],
                    jail_turns=p["jail_turns"],
                    jail_cards=p["jail_cards"],
                    bankrupt=p["bankrupt"],
                    properties=pm.get_owned_by(p["id"]),
                )
                for p in state["players"]
            ],
            properties={
                str(pos): PropertyState(
                    position=pos,
                    owner=prop["owner"],
                    houses=prop["houses"],
                    mortgaged=prop["mortgaged"],
                )
                for pos, prop in state["properties"].items()
            },
            current_player=state["current_player"],
            turn_number=state["turn_number"],
            houses_remaining=state["houses_remaining"],
            hotels_remaining=state["hotels_remaining"],
            last_roll=tuple(state["last_roll"]) if state["last_roll"] else None,
            doubles_count=state["doubles_count"],
            game_over=state["game_over"],
            winner=state["winner"],
            event_log=state["event_log"],
        )


class GameInfo(BaseModel):
    """Summary info for game listing."""

    id: str
    num_players: int
    players_joined: int
    started: bool
    game_over: bool
    created_at: datetime


class CreateGameRequest(BaseModel):
    """Request to create a new game."""

    num_players: int = Field(ge=2, le=8, default=4)
    player_names: list[str] | None = None
    seed: int | None = None


class CreateGameResponse(BaseModel):
    """Response after creating a game."""

    id: str
    num_players: int
    players: list[PlayerSlot]
    created_at: datetime
    websocket_url: str
