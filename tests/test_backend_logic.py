import sys
from pathlib import Path
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.game.env import GameEnvironment


def test_backend_message_formatting():
    """
    Verify that GameEnvironment.step() correctly formats stalemate messages
    from the underlying env info.
    """
    game_env = GameEnvironment()
    game_env.reset()

    # Mock the internal env step return to simulate a stalemate reroll
    # Original step returns: (encoded_state, reward, done, info)

    # Mock return values
    mock_encoded_state = game_env.env.encode_state()
    mock_reward = 0.0
    mock_done = False
    mock_info = {"stalemate_rerolls": 5}

    # Patch the env.step method temporarily
    original_step = game_env.env.step
    game_env.env.step = lambda action: (
        mock_encoded_state,
        mock_reward,
        mock_done,
        mock_info,
    )

    try:
        # Call the wrapper step
        state_data, done, winner = game_env.step(0)

        # Verify message
        print(f"Message: {state_data.get('message')}")
        assert state_data["message"] == "Stalemate! Auto-rerolled 5 times."

    finally:
        # Restore original method
        game_env.env.step = original_step


def test_backend_schema_fields():
    """
    Verify all expected fields are present in state data.
    """
    game_env = GameEnvironment()
    state_data = game_env.reset()

    expected_fields = [
        "board",
        "current_player",
        "legal_actions",
        "dice_value",
        "general_hits",
        "turn",
        "attacker",
        "defender",
        "message",
    ]

    for field in expected_fields:
        assert field in state_data, f"Missing field: {field}"


if __name__ == "__main__":
    test_backend_message_formatting()
    test_backend_schema_fields()
    print("Backend logic tests PASSED")
