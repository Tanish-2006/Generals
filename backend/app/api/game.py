from fastapi import APIRouter, HTTPException, BackgroundTasks
from ..schemas.game import (
    GameStartRequest,
    GameResponse,
    MoveRequest,
    MoveResponse,
    GameStatusResponse,
)
from ..game.manager import GameManager, get_game

router = APIRouter()


@router.post("/start", response_model=GameResponse)
async def start_game(request: GameStartRequest):
    game_id = GameManager.create_game(request.difficulty, request.human_player)

    # If AI goes first, we need to execute valid move before returning the initial state
    # so the user sees the board AFTER AI has moved.
    # We await this synchronous-ish check.
    await GameManager.handle_ai_first_move_if_needed(game_id)

    # Fetch state
    session = get_game(game_id)
    state_data = session.env.get_state_data()

    return GameResponse(game_id=game_id, state=state_data)


@router.post("/{game_id}/move", response_model=MoveResponse)
async def make_move(game_id: str, request: MoveRequest):
    # Process Human Move and immediatley trigger AI Move if game not done
    result = await GameManager.process_move(game_id, request.action)
    return result


@router.get("/{game_id}", response_model=GameStatusResponse)
async def get_game_state(game_id: str):
    return GameManager.get_game_state(game_id)


@router.post("/{game_id}/reset", response_model=GameResponse)
async def reset_game(game_id: str):
    await GameManager.reset_game(game_id)

    session = get_game(game_id)
    state_data = session.env.get_state_data()

    return GameResponse(game_id=game_id, state=state_data)
