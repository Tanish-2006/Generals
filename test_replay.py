from training.replay_buffer import ReplayBuffer
import numpy as np

buffer = ReplayBuffer()

# Fake data for testing
states = [np.zeros((17,10,10), dtype=np.float32) for _ in range(5)]
policies = [np.zeros((10003,), dtype=np.float32) for _ in range(5)]
values = [0, 1, -1, 1, 0]

# Save batch
buffer.add_game(states, policies, values)

# Load everything
s, p, v = buffer.load_all()
print("States shape:", s.shape)
print("Policies shape:", p.shape)
print("Values shape:", v.shape)