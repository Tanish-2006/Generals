import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.generals_env import GeneralsEnv


def test_conversion_rule():
    """Test that defender entering attacker garrison converts one attacker unit."""
    env = GeneralsEnv()
    env.reset()

    # Clear board for controlled test
    env.board.fill(0)

    # Place defender adjacent to attacker's top-left garrison
    # Defender at (1, 3) will move left into garrison at (1, 0) with dice=3
    env.board[1, 3] = env.defender_player

    # Place a single attacker in the top-left garrison
    env.board[0, 0] = env.attacker_player  # This is in GARRISON_CELLS[0]

    # Set current player to defender
    env.current_player = env.defender_player
    env.dice_value = 3

    # Find the move action (1, 3) -> (1, 0), direction left (index 2), distance 3
    action_key = (1, 3, 2, 3)  # row, col, dir_index, distance
    action_id = env.MOVE_TO_ACTION_ID.get(action_key)

    if action_id is None:
        print("WARNING: Could not find action for this test move")
        return

    print("Testing garrison conversion rule:")
    print("  Defender at (1, 3) moves to (1, 0) with dice=3")
    print("  Attacker unit at (0, 0) should be converted")

    # Execute the move
    env.step(action_id)

    # Check that attacker at (0, 0) is now converted to defender
    result_val = env.board[0, 0]
    print(f"  Result at (0, 0): {result_val}")

    # The attacker should be converted to defender
    assert result_val == env.defender_player, (
        f"Conversion Rule failed. Expected {env.defender_player}, got {result_val}"
    )

    print("SUCCESS: Conversion Rule applied correctly.")


def test_conversion_only_in_entered_zone():
    """Test that conversion only affects the specific garrison zone entered."""
    env = GeneralsEnv()
    env.reset()

    env.board.fill(0)

    # Place attacker units in BOTH top-left and top-right garrisons
    env.board[0, 0] = env.attacker_player  # Top-left garrison
    env.board[0, 9] = env.attacker_player  # Top-right garrison

    # Place defender to enter top-left garrison
    env.board[1, 3] = env.defender_player
    env.current_player = env.defender_player
    env.dice_value = 3

    action_key = (1, 3, 2, 3)  # Move left 3 to (1, 0)
    action_id = env.MOVE_TO_ACTION_ID.get(action_key)

    if action_id is None:
        print("WARNING: Could not find action for this test move")
        return

    env.step(action_id)

    # Top-left garrison unit should be converted
    assert env.board[0, 0] == env.defender_player, (
        "Unit in entered zone should be converted"
    )

    # Top-right garrison unit should NOT be converted
    assert env.board[0, 9] == env.attacker_player, (
        "Unit in OTHER zone should NOT be converted"
    )

    print("SUCCESS: Conversion only affects entered zone.")


if __name__ == "__main__":
    test_conversion_rule()
    test_conversion_only_in_entered_zone()
