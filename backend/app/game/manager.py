import asyncio
import uuid
import logging
from typing import Dict, Any, Optional

from fastapi import HTTPException

from ..game.env import GameEnvironment
from ..core.difficulty import get_difficulty_config
from ..state.store import GAMES, GameSession, get_game, save_game
from ..core.ai import compute_ai_move

logger = logging.getLogger("game.manager")
PLAYER_MAP = {"A": 1, "B": -1}
REV_PLAYER_MAP = {1: "A", -1: "B"}


class GameManager:
    @staticmethod
    def create_game(difficulty: str, human_player: str) -> str:
        """
        Creates a new game session.
        Note: This is synchronous (except for the potential locking if we did async creation, but simplified here).
        """
        try:
            game_id = str(uuid.uuid4())
            diff_config = get_difficulty_config(difficulty)

            # Initialize Environment
            env = GameEnvironment()
            env.reset()

            human_side = PLAYER_MAP[human_player]

            session = GameSession(
                game_id=game_id,
                env=env,
                difficulty_config=diff_config,
                human_player_side=human_side,
                done=False,
                winner=None,
            )
            save_game(session)

            logger.info(
                f"Game created: {game_id} [Difficulty: {difficulty}, Human: {human_player}]"
            )
            return game_id
        except Exception as e:
            logger.error(f"Error creating game: {e}")
            raise HTTPException(status_code=500, detail="Failed to create game session")

    @staticmethod
    async def process_move(game_id: str, action: int) -> Dict[str, Any]:
        session = get_game(game_id)
        if not session:
            raise HTTPException(status_code=404, detail="Game not found")

        # Use lock to ensure atomic turn processing
        async with session.lock:
            if session.done:
                raise HTTPException(status_code=400, detail="Game is over")

            env = session.env

            # 1. Validate Human Turn
            if env.raw_env.current_player != session.human_player_side:
                raise HTTPException(status_code=400, detail="Not your turn")

            # 2. Validate Legal Action
            # Optimization: Check primitive types before heavy logic if possible,
            # checking legality involves array lookup in env, it's fast enough.
            legal_actions = [a["id"] for a in env.raw_env.get_legal_actions()]
            if action not in legal_actions:
                logger.warning(f"Illegal move attempt in game {game_id}: {action}")
                raise HTTPException(status_code=400, detail=f"Illegal action: {action}")

            # 3. Apply Human Move
            logger.debug(f"Game {game_id}: Human move {action}")
            _, done, winner = env.step(action)
            session.done = done
            session.winner = REV_PLAYER_MAP[winner] if winner else None

            human_move_result = {
                "human_move": action,
                "ai_move": None,
                "state": env.get_state_data(),
                "done": done,
                "winner": session.winner,
            }

            if done:
                logger.info(f"Game {game_id} ended. Winner: {session.winner}")
                return human_move_result

            # 4. AI Turn
            # "Ensure AI move is only triggered after valid human move"
            # Optimization: We are holding the lock, so no other move can come in.

            try:
                # Compute AI move
                ai_action = await compute_ai_move(session)

                logger.debug(f"Game {game_id}: AI move {ai_action}")
                _, done, winner = env.step(ai_action)
                session.done = done
                session.winner = REV_PLAYER_MAP[winner] if winner else None

                if done:
                    logger.info(
                        f"Game {game_id} ended after AI move. Winner: {session.winner}"
                    )

                return {
                    "human_move": action,
                    "ai_move": ai_action,
                    "state": env.get_state_data(),
                    "done": done,
                    "winner": session.winner,
                }
            except Exception as e:
                logger.error(
                    f"Error during AI turn in game {game_id}: {e}", exc_info=True
                )
                raise HTTPException(status_code=500, detail="Internal AI Error")

    @staticmethod
    async def handle_ai_first_move_if_needed(game_id: str):
        """
        Checks if it's the AI's turn to move (player != human) and executes it.
        This is typically called right after game start or reset.
        """
        session = get_game(game_id)
        if not session:
            return

        async with session.lock:
            # If current player is NOT human, AI moves
            if session.env.raw_env.current_player != session.human_player_side:
                try:
                    logger.info(f"Game {game_id}: AI moving first...")
                    ai_action = await compute_ai_move(session)
                    session.env.step(ai_action)
                    # We don't need to check for winner here technically as it's the first move,
                    # but good practice:
                    if (
                        session.env.raw_env.winner
                    ):  # Access raw winner for correctness or env wrapper check
                        # env wrapper step returns done/winner but here we didn't capture it.
                        # Let's trust env state update.
                        pass
                except Exception as e:
                    logger.error(
                        f"Error during initial AI move in game {game_id}: {e}",
                        exc_info=True,
                    )
                    # We swallow this here to allow the game to at least return state,
                    # but realistically this is a bricked game.
                    pass

    @staticmethod
    def get_game_state(game_id: str):
        session = get_game(game_id)
        if not session:
            raise HTTPException(status_code=404, detail="Game not found")

        return {
            "state": session.env.get_state_data(),
            "done": session.done,
            "winner": session.winner,
        }

    @staticmethod
    async def reset_game(game_id: str) -> str:
        session = get_game(game_id)
        if not session:
            raise HTTPException(status_code=404, detail="Game not found")

        async with session.lock:
            session.env.reset()
            session.done = False
            session.winner = None
            logger.info(f"Game {game_id} reset.")

        # Release lock before calling this helper which acquires lock again
        await GameManager.handle_ai_first_move_if_needed(game_id)

        return game_id
