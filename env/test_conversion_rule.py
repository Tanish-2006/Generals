
import numpy as np
import sys
import os

# Add parent directory to path to import env
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from env.generals_env import GeneralsEnv

def test_conversion_rule():
    env = GeneralsEnv()
    
    # Setup Board Override
    # Player 1 is invading Player -1's Garrison at (9,9).
    # P-1 Garrison Location is (9,9).
    
    # 1. Clear board
    env.board.fill(0)
    
    # 2. Place Player -1 (Defender/Owner) on Garrison
    # Let's say 10 units. (-10 value)
    env.board[9][9] = -10
    
    # 3. Place Player 1 (Invader) on adjacent square (8,9)
    # Let's say 10 units. (+10 value)
    env.board[8][9] = 10
    
    # 4. Define Move Action
    # Move from (8,9) to (9,9).
    # We need to find the action_id for this move.
    move = (8, 9, 9, 9)
    action_id = None
    for aid, m in env.ACTION_ID_TO_MOVE.items():
        if m == move:
            action_id = aid
            break
            
    if action_id is None:
        print("Error: Could not find action_id for move (8,9)->(9,9)")
        return
        
    print(f"Testing Move (8,9)->(9,9) with P1(10) attacking P-1(10) in Garrison.")
    
    # 5. Execute Step using internal helper or public step
    # We use step() but enforce current_player = 1
    env.current_player = 1
    env.step(action_id)
    
    # 6. Verify Result
    # Expected:
    # Before: P1 (+10), P-1 (-10)
    # Rule: P-1 loses 1 -> -9. P1 gains 1 -> +11.
    # Combat: |11| - |-9| = 2.
    # Sign: P1 (+).
    # Result: +2 at (9,9).
    
    result_val = env.board[9][9]
    print(f"Result Value at (9,9): {result_val}")
    
    if result_val == 2:
        print("SUCCESS: Conversion Rule applied correctly. (10 vs 10 -> +2)")
    elif result_val == 0:
        print("FAILURE: Seems standard combat (10 vs 10 -> Draw) was applied. Conversion missed.")
    else:
        print(f"FAILURE: Unexpected result {result_val}. Logic might be wrong.")

if __name__ == "__main__":
    test_conversion_rule()
