import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.generals_env import GeneralsEnv
from selfplay.selfplay import SelfPlay
from utils.batched_inference import InferenceServer
from model.network import GeneralsNet


async def test_deadlock_fix():
    print("Testing deadlock fix with max_moves=5...")

    # Mock inference server
    net = GeneralsNet()
    inference_server = InferenceServer(net, batch_size=1)
    await inference_server.start()

    try:
        # Initialize SelfPlay with a very short max_moves
        sp = SelfPlay(
            GeneralsEnv,
            inference_server,
            games_per_iteration=1,
            mcts_simulations=10,  # low sims for speed
            temperature_threshold=5,
            max_moves=5,  # force early termination
        )

        print("Starting game...")
        states, policies, values = await sp.play_one_game()
        print("Game finished!")

        print(f"Number of moves: {len(states)}")
        print(f"Values (should be 0.0 for draw): {values}")

        if len(states) == 5 and all(v == 0.0 for v in values):
            print("SUCCESS: Game terminated at max_moves with draw result.")
        else:
            print("FAILURE: Game did not terminate correctly or values are wrong.")

    finally:
        await inference_server.stop()


if __name__ == "__main__":
    asyncio.run(test_deadlock_fix())
