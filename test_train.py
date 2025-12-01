"""Test script for trainer functionality."""

from training.train import Trainer
import numpy as np

# Create trainer object
trainer = Trainer()

# Create dummy dataset to produce a first model
states = np.zeros((5, 17, 10, 10), dtype=np.float32)
policies = np.zeros((5, 10003), dtype=np.float32)
policies[:, 0] = 1.0   # fake best action for all samples
values = np.zeros(5, dtype=np.float32)

# Train one small batch and save the model
trainer.train(states, policies, values, save_name="model_latest.pth")