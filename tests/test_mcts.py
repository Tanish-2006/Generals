import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import numpy as np
from mcts.mcts import AsyncMCTS
from env.generals_env import GeneralsEnv


def test_mcts_basic():
    env = GeneralsEnv()
    env.reset()
    
    mcts = AsyncMCTS(env, net=None, c_puct=1.0)
    
    async def run_search():
        await mcts.search(n_sims=10)
        return mcts.get_action_probs(temperature=1.0)
    
    probs = asyncio.run(run_search())
    
    assert probs.shape == (env.ACTION_DIM,)
    assert np.sum(probs) > 0
    
    print(f"MCTS action probs shape: {probs.shape}")
    print(f"Non-zero actions: {np.count_nonzero(probs)}")
    print("MCTS basic test: PASSED")


def test_mcts_action_selection():
    env = GeneralsEnv()
    env.reset()
    
    mcts = AsyncMCTS(env, net=None, c_puct=1.0)
    
    async def run_and_select():
        await mcts.search(n_sims=20)
        return mcts.select_action(temperature=0, deterministic=True)
    
    action = asyncio.run(run_and_select())
    
    legal_actions = [a["id"] for a in env.get_legal_actions()]
    
    assert action in legal_actions, f"Selected action {action} is NOT legal"
    print(f"Selected action {action} is legal")
    print("MCTS action selection test: PASSED")


if __name__ == "__main__":
    test_mcts_basic()
    test_mcts_action_selection()
