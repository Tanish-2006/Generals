"""Test script for Arena evaluation between two models."""

from evaluate.evaluate import Arena

arena = Arena(
    model_A_path="data/checkpoints/model_latest.pth",
    model_B_path="data/checkpoints/model_old.pth",
    games=2,
    mcts_simulations=10
)

arena.run()