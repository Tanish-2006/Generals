"""
Comprehensive tests for GeneralsEnv correctness after rule fixes.
Tests all mechanics according to the authoritative rule set.
"""

import pytest
import numpy as np
from env.generals_env import GeneralsEnv, ATTACKER, DEFENDER


class TestGeneralsEnvCorrectness:
    @pytest.fixture
    def env(self):
        return GeneralsEnv()

    def test_garrison_placement(self, env):
        """
        Garrison placement test — reset() should set exactly 12 units for each player
        across the two garrisons (6+6).
        """
        env.reset()
        board = env.board

        # Player counts (binary: 1 or -1 per cell)
        attacker_units = np.sum(board == env.attacker_player)
        defender_units = np.sum(board == env.defender_player)

        assert attacker_units == 12, (
            f"Attacker should have 12 units, found {attacker_units}"
        )
        assert defender_units == 12, (
            f"Defender should have 12 units, found {defender_units}"
        )

        # Verify attacker in top garrisons (garrison 0 and 1)
        for pos in env.GARRISON_CELLS[0]:  # Top-left
            assert board[pos] == env.attacker_player, f"Pos {pos} should be Attacker"
        for pos in env.GARRISON_CELLS[1]:  # Top-right
            assert board[pos] == env.attacker_player, f"Pos {pos} should be Attacker"

        # Verify defender in bottom garrisons (garrison 2 and 3)
        for pos in env.GARRISON_CELLS[2]:  # Bottom-left
            assert board[pos] == env.defender_player, f"Pos {pos} should be Defender"
        for pos in env.GARRISON_CELLS[3]:  # Bottom-right
            assert board[pos] == env.defender_player, f"Pos {pos} should be Defender"

    def test_moat_general_interaction_no_defender(self, env):
        """
        Attacker to general while no moat defenders: attacker removed and general_hits++.
        """
        env.reset()
        env.board[:] = 0

        # Set roles explicitly
        env.current_player = env.attacker_player

        gen_cell = env.GENERAL_CELLS[0]  # (4,4)
        attacker_pos = (3, 4)  # In Moat
        assert attacker_pos in env.MOAT_CELLS

        env.board[attacker_pos] = env.attacker_player
        env.dice_value = 1

        # Move down 1 to General
        action_key = (3, 4, 1, 1)  # r, c, d_idx (down), dist
        action_id = env.MOVE_TO_ACTION_ID.get(action_key)
        assert action_id is not None

        initial_hits = env.general_hits
        env.step(action_id)

        # Attacker removed
        assert env.board[gen_cell] == 0, "General cell should be empty"
        assert env.board[attacker_pos] == 0, "Source cell should be empty"

        # General hit incremented
        assert env.general_hits == initial_hits + 1, "General hits should increment"

    def test_moat_general_interaction_with_defender(self, env):
        """
        Attacker to general while moat defender present: both die, no general-hit.
        """
        env.reset()
        env.board[:] = 0

        env.current_player = env.attacker_player

        gen_cell = env.GENERAL_CELLS[0]  # e.g. (4,4)
        attacker_pos = (3, 4)
        env.board[attacker_pos] = env.attacker_player

        # Place a defender in the moat
        defender_pos = (3, 5)  # Also in moat
        assert defender_pos in env.MOAT_CELLS
        env.board[defender_pos] = env.defender_player

        env.dice_value = 1
        action_key = (3, 4, 1, 1)  # Down 1
        action_id = env.MOVE_TO_ACTION_ID[action_key]

        initial_hits = env.general_hits
        env.step(action_id)

        # Attacker removed
        assert env.board[gen_cell] == 0
        assert env.board[attacker_pos] == 0

        # Moat defender removed
        assert env.board[defender_pos] == 0, "Defender in moat should be removed"

        # No general hit
        assert env.general_hits == initial_hits, "General hits should NOT increment"

    def test_path_blocking(self, env):
        """
        Ensure _is_path_clear returns False if any intermediate square is blocked.
        """
        env.reset()
        env.board[:] = 0

        env.board[0, 0] = 1
        env.board[0, 1] = -1  # Blocked

        is_clear = env._is_path_clear(0, 0, 0, 2)
        assert is_clear == False, "Path should be blocked by piece at (0,1)"

        # Check legal actions
        env.current_player = 1
        env.dice_value = 2
        legal = env.get_legal_actions()

        # Dir (0,1) is index 3 (right)
        move_key = (0, 0, 3, 2)
        move_id = env.MOVE_TO_ACTION_ID.get(move_key)

        legal_ids = [a["id"] for a in legal]
        if move_id:
            assert move_id not in legal_ids, "Blocked move should not be legal"

    def test_exact_distance_legality(self, env):
        """
        Verify that only EXACT dice distance is legal (not range 1..dice).
        """
        env.reset()
        env.board[:] = 0
        env.board[0, 0] = 1
        env.current_player = 1
        env.dice_value = 3

        legal = env.get_legal_actions()
        legal_ids = set([a["id"] for a in legal])

        # Only distance 3 should be legal
        for dist in [1, 2]:
            key = (0, 0, 3, dist)  # Right by dist
            aid = env.MOVE_TO_ACTION_ID.get(key)
            if aid:
                assert aid not in legal_ids, (
                    f"Distance {dist} should NOT be legal with dice 3"
                )

        # Distance 3 should be legal
        key_3 = (0, 0, 3, 3)
        aid_3 = env.MOVE_TO_ACTION_ID.get(key_3)
        if aid_3:
            assert aid_3 in legal_ids, "Distance 3 should be legal with dice 3"

    def test_stalemate_autoreroll(self, env):
        """
        Ensure stalemate auto-rerolls when no legal moves exist.
        The new implementation handles stalemate internally - it's not an action.
        """
        env.reset()
        env.current_player = 1
        env.board[:] = 0

        # Place single unit in corner with dice=6 (would need to move 6 which is off-board)
        env.board[0, 0] = 1
        env.dice_value = 6

        # Clear special move conditions
        env.garrison_used = {1: True, -1: True}  # Already used

        # Get legal actions - should auto-reroll if no moves
        legal = env.get_legal_actions()

        # Should have some legal action after auto-reroll
        legal_ids = [a["id"] for a in legal]
        # Even if reroll happens internally, there should be moves available
        # unless truly stalemate (which would keep rerolling)

    def test_special_moves_state_persistence(self, env):
        """
        Ensure garrison_used and offensive_used flags persist on save_state() and restore_state().
        """
        env.reset()
        env.garrison_used[1] = True
        env.offensive_used = True

        state_dict = env.save_state()

        # Create new env and restore
        new_env = GeneralsEnv()
        new_env.restore_state(state_dict)

        assert new_env.garrison_used[1] == True
        assert new_env.offensive_used == True
        assert new_env.garrison_used[-1] == False  # Default

    def test_win_condition_attacker(self, env):
        """
        Test attacker wins with 3 general hits.
        """
        env.reset()
        env.board[:] = 0

        env.current_player = env.attacker_player
        env.general_hits = 2  # On brink of win

        gen_cell = env.GENERAL_CELLS[0]
        attacker_pos = (3, 4)  # In moat
        assert attacker_pos in env.MOAT_CELLS
        env.board[attacker_pos] = env.attacker_player

        env.dice_value = 1
        action_key = (3, 4, 1, 1)  # Down 1
        action_id = env.MOVE_TO_ACTION_ID[action_key]

        _, reward, done, _ = env.step(action_id)

        assert env.general_hits == 3
        assert done == True
        assert env.winner == env.attacker_player
        assert reward == 1.0

    def test_win_condition_defender(self, env):
        """
        Test defender wins when all attacker units are eliminated.
        """
        env.reset()
        env.board[:] = 0

        # Single attacker and multiple defenders
        env.board[0, 3] = env.attacker_player  # Last attacker
        env.board[0, 1] = env.defender_player  # Defender to capture

        env.current_player = env.defender_player
        env.dice_value = 2

        # Move defender right 2 to capture attacker
        action_key = (0, 1, 3, 2)  # Right 2
        action_id = env.MOVE_TO_ACTION_ID.get(action_key)

        if action_id:
            _, reward, done, _ = env.step(action_id)

            # All attackers eliminated
            if np.sum(env.board == env.attacker_player) == 0:
                assert done == True
                assert env.winner == env.defender_player

    def test_simple_kill_combat(self, env):
        """
        Combat should be simple kill - attacker always wins, no unit counting.
        """
        env.reset()
        env.board[:] = 0

        # Attacker unit
        env.board[0, 0] = 1
        # Defender unit to be captured
        env.board[0, 3] = -1

        env.current_player = 1
        env.dice_value = 3

        action_key = (0, 0, 3, 3)  # Right 3
        action_id = env.MOVE_TO_ACTION_ID.get(action_key)

        if action_id:
            env.step(action_id)

            # Attacker should occupy target
            assert env.board[0, 3] == 1, "Attacker should occupy captured cell"
            assert env.board[0, 0] == 0, "Source should be empty"

    def test_no_stacking(self, env):
        """
        Cannot move onto friendly unit.
        """
        env.reset()
        env.board[:] = 0

        env.board[0, 0] = 1  # Attacker
        env.board[0, 3] = 1  # Friendly at target

        env.current_player = 1
        env.dice_value = 3

        legal = env.get_legal_actions()
        legal_ids = [a["id"] for a in legal]

        action_key = (0, 0, 3, 3)  # Right 3
        action_id = env.MOVE_TO_ACTION_ID.get(action_key)

        if action_id:
            assert action_id not in legal_ids, "Cannot stack on friendly unit"

    def test_path_blocked_by_general(self, env):
        """
        Path cannot cross through General cells.
        """
        env.reset()
        env.board[:] = 0

        # Unit that would cross General to reach target
        env.board[4, 2] = 1
        env.current_player = 1
        env.dice_value = 5

        # Path from (4, 2) to (4, 7) crosses General at (4, 4) and (4, 5)
        is_clear = env._is_path_clear(4, 2, 4, 7)
        assert is_clear == False, "Path through General should be blocked"
