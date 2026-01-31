from typing import List, Optional, Literal
from pydantic import BaseModel


class GameState(BaseModel):
    board: List[List[int]]
    current_player: str
    legal_actions: List[int]
    dice_value: int
    general_hits: int
    turn: int
    attacker: str
    defender: str
    message: Optional[str] = None
    last_move_description: Optional[str] = None


class GameStartRequest(BaseModel):
    difficulty: Literal["easy", "medium", "hard"]
    human_player: Literal["A", "B"]


class GameResponse(BaseModel):
    game_id: str
    state: GameState


class MoveRequest(BaseModel):
    action: int


class MoveResponse(BaseModel):
    human_move: int
    ai_move: Optional[int]
    state: GameState
    done: bool
    winner: Optional[str]


class GameStatusResponse(BaseModel):
    state: GameState
    done: bool
    winner: Optional[str]
