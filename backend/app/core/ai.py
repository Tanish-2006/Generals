import asyncio
from mcts.mcts import AsyncMCTS
from .inference import inference_manager
from ..state.store import GameSession


async def compute_ai_move(session: GameSession) -> int:
    """
    Computes the AI move using AsyncMCTS.
    Synchronously waited on by the request handler, but runs async internally using the server.
    """
    env = session.env.raw_env
    config = session.difficulty_config

    # Initialize MCTS
    # Note: inference_server is globally managed
    mcts = AsyncMCTS(
        env=env,
        inference_server=inference_manager.server,
        c_puct=1.0,  # Standard PUCT
    )

    # Run simulations
    # The prompt says: "AI move must be computed synchronously per request"
    # But strictly: "AI must use AsyncMCTS"
    # So we await the search here.

    sims = config["mcts_simulations"]
    await mcts.search(n_sims=sims)

    # Select action
    temp = config["temperature"]
    action = mcts.select_action(temperature=temp)

    return int(action)
