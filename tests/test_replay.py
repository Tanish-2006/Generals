import sys
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from training.replay_buffer import ReplayBuffer


def test_replay_buffer():
    temp_dir = tempfile.mkdtemp()
    
    try:
        buffer = ReplayBuffer(save_dir=temp_dir)
        
        states = [np.zeros((17, 10, 10), dtype=np.float32) for _ in range(5)]
        policies = [np.zeros((10003,), dtype=np.float32) for _ in range(5)]
        values = [0, 1, -1, 1, 0]
        
        buffer.add_game(states, policies, values)
        
        s, p, v = buffer.load_all()
        print(f"States shape: {s.shape}")
        print(f"Policies shape: {p.shape}")
        print(f"Values shape: {v.shape}")
        
        assert s.shape == (5, 17, 10, 10)
        assert p.shape == (5, 10003)
        assert v.shape == (5,)
        
        print("ReplayBuffer test: PASSED")
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_replay_buffer()
