import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from training.train import Trainer


def test_trainer():
    trainer = Trainer()
    
    states = np.zeros((5, 17, 10, 10), dtype=np.float32)
    policies = np.zeros((5, 10003), dtype=np.float32)
    policies[:, 0] = 1.0
    values = np.zeros(5, dtype=np.float32)
    
    save_path = trainer.train(states, policies, values, save_name="test_model.pth")
    
    assert save_path is not None
    print(f"Model saved to: {save_path}")
    print("Trainer test: PASSED")


if __name__ == "__main__":
    test_trainer()
