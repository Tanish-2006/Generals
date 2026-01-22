import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate.evaluate import Arena
from config import PATHS


def run_arena_test():
    arena = Arena(
        model_A_path=str(PATHS.checkpoint_dir / "model_latest.pth"),
        model_B_path=str(PATHS.checkpoint_dir / "model_old.pth"),
        games=2,
        mcts_simulations=10
    )
    return asyncio.run(arena.run())


if __name__ == "__main__":
    run_arena_test()
