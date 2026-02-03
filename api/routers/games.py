"""
Game management REST API endpoints.

Provides CRUD operations for game sessions.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import get_game_manager
from ..models.game import (
    CreateGameRequest,
    CreateGameResponse,
    GameInfo,
    GameState,
    PlayerSlot,
)
from ..services.game_manager import GameManager

router = APIRouter()


@router.post(
    "",
    response_model=CreateGameResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new game",
    description="Create a new Monopoly game session with the specified number of players.",
)
async def create_game(
    request: CreateGameRequest,
    game_manager: Annotated[GameManager, Depends(get_game_manager)],
) -> CreateGameResponse:
    """Create a new game session.

    Args:
        request: Game creation parameters
        game_manager: Injected game manager

    Returns:
        CreateGameResponse with game ID and player slots

    Raises:
        HTTPException: If max games reached or invalid parameters
    """
    try:
        game_id = await game_manager.create_game(
            num_players=request.num_players,
            player_names=request.player_names,
            seed=request.seed,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Get the created game
    active_game = await game_manager.get_game(game_id)
    if active_game is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Game creation failed",
        )

    return CreateGameResponse(
        id=game_id,
        num_players=request.num_players,
        players=list(active_game.player_slots.values()),
        created_at=active_game.created_at,
        websocket_url=f"/ws/games/{game_id}",
    )


@router.get(
    "",
    response_model=list[GameInfo],
    summary="List all games",
    description="Get a list of all active game sessions.",
)
async def list_games(
    game_manager: Annotated[GameManager, Depends(get_game_manager)],
) -> list[GameInfo]:
    """List all active games.

    Args:
        game_manager: Injected game manager

    Returns:
        List of GameInfo summaries
    """
    return await game_manager.list_games()


@router.get(
    "/{game_id}",
    response_model=GameState,
    summary="Get game state",
    description="Get the current state of a game session.",
)
async def get_game_state(
    game_id: str,
    game_manager: Annotated[GameManager, Depends(get_game_manager)],
) -> GameState:
    """Get the current state of a game.

    Args:
        game_id: The game ID
        game_manager: Injected game manager

    Returns:
        Current GameState

    Raises:
        HTTPException: If game not found
    """
    state = await game_manager.get_game_state(game_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Game not found",
        )
    return state


@router.delete(
    "/{game_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a game",
    description="Delete a game session.",
)
async def delete_game(
    game_id: str,
    game_manager: Annotated[GameManager, Depends(get_game_manager)],
) -> None:
    """Delete a game session.

    Args:
        game_id: The game ID
        game_manager: Injected game manager

    Raises:
        HTTPException: If game not found
    """
    deleted = await game_manager.delete_game(game_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Game not found",
        )


@router.get(
    "/{game_id}/players",
    response_model=list[PlayerSlot],
    summary="Get player slots",
    description="Get the player slots for a game session.",
)
async def get_players(
    game_id: str,
    game_manager: Annotated[GameManager, Depends(get_game_manager)],
) -> list[PlayerSlot]:
    """Get player slots for a game.

    Args:
        game_id: The game ID
        game_manager: Injected game manager

    Returns:
        List of PlayerSlot objects

    Raises:
        HTTPException: If game not found
    """
    active_game = await game_manager.get_game(game_id)
    if active_game is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Game not found",
        )
    return list(active_game.player_slots.values())
