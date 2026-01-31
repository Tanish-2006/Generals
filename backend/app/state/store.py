import asyncio
from typing import Dict, Optional
from dataclasses import dataclass, field
from ..game.env import GameEnvironment
from ..core.difficulty import DifficultyConfig


@dataclass
class GameSession:
    game_id: str
    env: GameEnvironment
    difficulty_config: DifficultyConfig
    done: bool = False
    winner: Optional[str] = None
    human_player_side: int = 1  # 1 (A) or -1 (B)

    # Concurrency lock to prevent race conditions on move processing
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# Thread-safe in-memory store for asyncio event loop
# Note: For multi-process workers, this needs to be Redis/Memcached.
# For this task (one InferenceServer), one worker is implied or stickiness required.
GAMES: Dict[str, GameSession] = {}


def get_game(game_id: str) -> Optional[GameSession]:
    return GAMES.get(game_id)


def save_game(session: GameSession):
    GAMES[session.game_id] = session


def delete_game(game_id: str):
    if game_id in GAMES:
        del GAMES[game_id]
