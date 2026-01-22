import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from env.generals_env import GeneralsEnv


def test_conversion_rule():
    env = GeneralsEnv()
    
    env.board.fill(0)
    env.board[9][9] = -10
    env.board[8][9] = 10
    
    move = (8, 9, 9, 9)
    action_id = None
    for aid, m in env.ACTION_ID_TO_MOVE.items():
        if m == move:
            action_id = aid
            break
    
    assert action_id is not None, "Could not find action_id for move (8,9)->(9,9)"
        
    print(f"Testing Move (8,9)->(9,9) with P1(10) attacking P-1(10) in Garrison.")
    
    env.current_player = 1
    env.step(action_id)
    
    result_val = env.board[9][9]
    print(f"Result Value at (9,9): {result_val}")
    
    assert result_val == 2, f"Conversion Rule failed. Expected 2, got {result_val}"
    print("SUCCESS: Conversion Rule applied correctly. (10 vs 10 -> +2)")


if __name__ == "__main__":
    test_conversion_rule()
