"""Test script for Arena evaluation between two models."""

import asyncio
import os
from evaluate.evaluate import Arena

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

arena = Arena(
    model_A_path=os.path.join(BASE_DIR, "data", "checkpoints", "model_latest.pth"),
    model_B_path=os.path.join(BASE_DIR, "data", "checkpoints", "model_old.pth"),
    games=2,
    mcts_simulations=10
)

if __name__ == "__main__":
    asyncio.run(arena.run())