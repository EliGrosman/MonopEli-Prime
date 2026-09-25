"""
Pydantic models for game state and configuration.
"""

from datetime import datetime
from typing import Any

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
    is_ai: bool = False
    ai_type: str | None = None


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


class AgentInspectionState(BaseModel):
    """Small public projection of a guided agent's current state."""

    player_id: int
    sequence: int
    basis_revision: int
    status: str
    short_term_objective: str
    long_term_objective: str
    cash_reserve_target: int
    latest_summary: str
    fallback_reason: str | None = None


class AgentActivity(BaseModel):
    """Public, bounded record of one server-side agent command."""

    actor: int
    before_revision: int
    after_revision: int
    source: str
    summary: str
    action_type: str
    fallback_reason: str | None = None


class GameActivity(BaseModel):
    """One confirmed action and its public outcomes, shared by all viewers."""

    id: str
    actor: int
    revision: int
    turn_number: int
    action_type: str
    summary: str
    details: list[str] = Field(default_factory=list)
    occurred_at: datetime


class GameState(BaseModel):
    """Full game state sent to clients."""

    rules_id: str = "foundation-v1"
    game_phase: str = "pre_roll"
    decision_player: int = 0
    revision: int = 0
    debt: dict[str, Any] | None = None
    legal_actions: list[dict[str, Any]] = Field(default_factory=list)
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
    decision_contract: dict[str, Any]
    agent_inspections: dict[int, AgentInspectionState] = Field(default_factory=dict)
    agent_activity: list[AgentActivity] = Field(default_factory=list)
    game_activity: list[GameActivity] = Field(default_factory=list)

    @classmethod
    def from_engine(
        cls,
        game: MonopolyGame,
        player_slots: dict[int, PlayerSlot] | None = None,
        agent_inspections: dict[int, dict[str, Any]] | None = None,
        agent_activity: list[dict[str, Any]] | None = None,
        game_activity: list[dict[str, Any]] | None = None,
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

        from monopoly_engine.foundation import legal_actions

        pid = game.decision_player
        return cls(
            rules_id=game.rules_id,
            game_phase=game.state.phase,
            decision_player=pid,
            revision=game.state.revision,
            debt=dict(game.state.obligations[0]) if game.state.obligations else None,
            legal_actions=[action.to_dict() for action in legal_actions(game, pid)],
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
                    is_ai=(
                        player_slots[p["id"]].is_ai
                        if player_slots and p["id"] in player_slots
                        else False
                    ),
                    ai_type=(
                        player_slots[p["id"]].ai_type
                        if player_slots and p["id"] in player_slots
                        else None
                    ),
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
            decision_contract=game.decision_view(None).to_dict(),
            agent_inspections=agent_inspections or {},
            agent_activity=agent_activity or [],
            game_activity=[GameActivity.model_validate(row) for row in game_activity or []],
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
    rules_id: str = Field(default="foundation-v1", pattern="^(foundation-v1|foundation-trade-v1)$")


class CreateGameResponse(BaseModel):
    """Response after creating a game."""

    id: str
    num_players: int
    players: list[PlayerSlot]
    created_at: datetime
    websocket_url: str
