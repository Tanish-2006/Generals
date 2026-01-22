import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.generals_env import GeneralsEnv


def test_environment_basic():
    env = GeneralsEnv()
    state = env.reset()
    
    print("Environment reset successful")
    print(f"State shape: {state.shape}")
    
    actions = env.get_legal_actions(env.dice_value)
    print(f"Legal actions count: {len(actions)}")
    
    assert len(actions) > 0, "No legal actions found"
    
    action = actions[0]["id"]
    print(f"Taking action: {action}")
    
    next_state, reward, done, info = env.step(action)
    print(f"Next state shape: {next_state.shape}")
    print(f"Reward: {reward}")
    print(f"Game done: {done}")
    
    assert next_state.shape == (17, 10, 10)


def test_state_save_restore():
    env = GeneralsEnv()
    env.reset()
    
    saved = env.save_state()
    original_board = env.board.copy()
    
    actions = env.get_legal_actions()
    if actions:
        env.step(actions[0]["id"])
    
    env.restore_state(saved)
    
    assert (env.board == original_board).all(), "State save/restore failed"
    print("State save/restore: PASSED")


if __name__ == "__main__":
    test_environment_basic()
    test_state_save_restore()
