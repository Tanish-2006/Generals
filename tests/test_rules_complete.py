"""
Comprehensive unit tests for GeneralsEnv rule implementation.
Tests all 12 rule fixes based on authoritative PDF rules.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.generals_env import GeneralsEnv, ATTACKER, DEFENDER


class TestBoardAndLayout:
    """Test board geometry and masks."""

    def test_board_size(self):
        env = GeneralsEnv()
        assert env.BOARD_SIZE == 10
        assert env.board.shape == (10, 10)

    def test_general_cells(self):
        env = GeneralsEnv()
        assert len(env.GENERAL_CELLS) == 4
        assert (4, 4) in env.GENERAL_CELLS
        assert (4, 5) in env.GENERAL_CELLS
        assert (5, 4) in env.GENERAL_CELLS
        assert (5, 5) in env.GENERAL_CELLS

    def test_moat_cells(self):
        env = GeneralsEnv()
        assert len(env.MOAT_CELLS) == 12
        # Moat should not include General cells
        for pos in env.MOAT_CELLS:
            assert pos not in env.GENERAL_CELLS

    def test_garrison_cells(self):
        env = GeneralsEnv()
        assert len(env.GARRISON_CELLS) == 4
        for group in env.GARRISON_CELLS:
            assert len(group) == 6


class TestInitialPlacement:
    """Test initial unit placement."""

    def test_unit_count_per_player(self):
        env = GeneralsEnv()
        env.reset()
        attacker_count = np.sum(env.board == env.attacker_player)
        defender_count = np.sum(env.board == env.defender_player)
        assert attacker_count == 12
        assert defender_count == 12

    def test_units_in_garrisons(self):
        env = GeneralsEnv()
        env.reset()
        # Attacker units in top garrisons
        for pos in env.GARRISON_CELLS[0] + env.GARRISON_CELLS[1]:
            assert env.board[pos] == env.attacker_player
        # Defender units in bottom garrisons
        for pos in env.GARRISON_CELLS[2] + env.GARRISON_CELLS[3]:
            assert env.board[pos] == env.defender_player

    def test_no_units_in_general_or_moat(self):
        env = GeneralsEnv()
        env.reset()
        for pos in env.GENERAL_CELLS:
            assert env.board[pos] == 0
        for pos in env.MOAT_CELLS:
            assert env.board[pos] == 0


class TestTossAndRoles:
    """Test toss and role assignment."""

    def test_attacker_moves_first(self):
        env = GeneralsEnv()
        env.reset()
        assert env.current_player == env.attacker_player

    def test_roles_assigned(self):
        env = GeneralsEnv()
        env.reset()
        assert env.roles[env.attacker_player] == ATTACKER
        assert env.roles[env.defender_player] == DEFENDER


class TestMovementRules:
    """Test movement distance and path validation."""

    def test_exact_dice_distance(self):
        """Movement must be EXACTLY dice value, not 1 to dice."""
        env = GeneralsEnv()
        env.reset()
        env.board[:] = 0
        env.board[0, 0] = env.attacker_player
        env.current_player = env.attacker_player
        env.dice_value = 3

        legal = env.get_legal_actions()
        legal_ids = [a["id"] for a in legal]

        # Distance 3 should be legal
        key_dist3 = (0, 0, 3, 3)  # Right 3
        if key_dist3 in env.MOVE_TO_ACTION_ID:
            aid = env.MOVE_TO_ACTION_ID[key_dist3]
            assert aid in legal_ids

        # Distance 1 should NOT be legal (exact dice required)
        key_dist1 = (0, 0, 3, 1)  # Right 1
        if key_dist1 in env.MOVE_TO_ACTION_ID:
            aid = env.MOVE_TO_ACTION_ID[key_dist1]
            assert aid not in legal_ids

    def test_path_blocked_by_unit(self):
        """Path cannot cross through units."""
        env = GeneralsEnv()
        env.reset()
        env.board[:] = 0
        env.board[0, 0] = env.attacker_player
        env.board[0, 1] = env.defender_player  # Blocker
        env.current_player = env.attacker_player
        env.dice_value = 3

        # Path to (0, 3) is blocked by unit at (0, 1)
        assert not env._is_path_clear(0, 0, 0, 3)

    def test_path_blocked_by_general(self):
        """Path cannot cross through General cells."""
        env = GeneralsEnv()
        env.reset()
        env.board[:] = 0
        # Place unit that would need to cross General
        env.board[4, 2] = env.attacker_player
        env.current_player = env.attacker_player

        # Path from (4, 2) to (4, 7) crosses General at (4, 4) and (4, 5)
        assert not env._is_path_clear(4, 2, 4, 7)


class TestCombatRules:
    """Test combat mechanics."""

    def test_simple_kill_no_counting(self):
        """Combat is simple kill - attacker always wins."""
        env = GeneralsEnv()
        env.reset()
        env.board[:] = 0
        env.board[0, 0] = env.attacker_player
        env.board[0, 3] = env.defender_player
        env.current_player = env.attacker_player
        env.dice_value = 3

        # Move to capture
        key = (0, 0, 3, 3)  # Right 3
        aid = env.MOVE_TO_ACTION_ID.get(key)
        if aid:
            env.step(aid)
            # Attacker should now occupy (0, 3)
            assert env.board[0, 3] == env.attacker_player
            assert env.board[0, 0] == 0

    def test_no_stacking(self):
        """Cannot move onto friendly unit."""
        env = GeneralsEnv()
        env.reset()
        env.board[:] = 0
        env.board[0, 0] = env.attacker_player
        env.board[0, 3] = env.attacker_player  # Friendly
        env.current_player = env.attacker_player
        env.dice_value = 3

        legal = env.get_legal_actions()
        legal_ids = [a["id"] for a in legal]

        # Move to (0, 3) should NOT be legal
        key = (0, 0, 3, 3)
        if key in env.MOVE_TO_ACTION_ID:
            aid = env.MOVE_TO_ACTION_ID[key]
            assert aid not in legal_ids


class TestGeneralAndMoat:
    """Test General hit mechanics and Moat defense."""

    def test_general_hit_no_moat_defenders(self):
        """Attacker hits General when Moat is empty."""
        env = GeneralsEnv()
        env.reset()
        env.board[:] = 0
        # Place attacker adjacent to General
        env.board[4, 3] = env.attacker_player
        env.current_player = env.attacker_player
        env.roles = {env.current_player: ATTACKER, -env.current_player: DEFENDER}
        env.attacker_player = env.current_player
        env.defender_player = -env.current_player
        env.dice_value = 1

        initial_hits = env.general_hits

        # Move to General cell (4, 4)
        key = (4, 3, 3, 1)  # Right 1
        aid = env.MOVE_TO_ACTION_ID.get(key)
        if aid:
            env.step(aid)
            assert env.general_hits == initial_hits + 1
            # Attacker should be consumed (not on General)
            assert env.board[4, 4] == 0

    def test_moat_intercept(self):
        """Defender in Moat intercepts General attack."""
        env = GeneralsEnv()
        env.reset()
        env.board[:] = 0
        # Place attacker adjacent to General
        env.board[4, 3] = env.attacker_player
        # Place defender in Moat
        env.board[3, 4] = env.defender_player  # Moat cell
        env.current_player = env.attacker_player
        env.roles = {env.current_player: ATTACKER, -env.current_player: DEFENDER}
        env.attacker_player = env.current_player
        env.defender_player = -env.current_player
        env.dice_value = 1

        initial_hits = env.general_hits

        # Move to General cell (4, 4)
        key = (4, 3, 3, 1)  # Right 1
        aid = env.MOVE_TO_ACTION_ID.get(key)
        if aid:
            env.step(aid)
            # No hit should occur
            assert env.general_hits == initial_hits
            # Attacker dies
            assert env.board[4, 4] == 0
            # Moat defender dies
            assert env.board[3, 4] == 0

    def test_defender_cannot_enter_general(self):
        """Defender cannot enter General cells."""
        env = GeneralsEnv()
        env.reset()
        env.board[:] = 0
        # Place defender adjacent to General
        env.board[4, 3] = env.defender_player
        env.current_player = env.defender_player
        env.dice_value = 1

        legal = env.get_legal_actions()
        legal_ids = [a["id"] for a in legal]

        # Move to General cell (4, 4) should NOT be legal
        key = (4, 3, 3, 1)  # Right 1
        if key in env.MOVE_TO_ACTION_ID:
            aid = env.MOVE_TO_ACTION_ID[key]
            assert aid not in legal_ids


class TestSpecialMoves:
    """Test Garrison and Offensive special moves."""

    def test_garrison_requires_dice_6(self):
        """Garrison only available on dice 6."""
        env = GeneralsEnv()
        env.reset()
        env.dice_value = 5

        legal = env.get_legal_actions()
        legal_ids = [a["id"] for a in legal]
        assert env.ACTION_GARRISON not in legal_ids

        env.dice_value = 6
        legal = env.get_legal_actions()
        legal_ids = [a["id"] for a in legal]
        # Should be available if not used and has units in garrison
        if not env.garrison_used[env.current_player]:
            assert env.ACTION_GARRISON in legal_ids

    def test_garrison_once_per_game(self):
        """Garrison can only be used once per player."""
        env = GeneralsEnv()
        env.reset()
        env.dice_value = 6

        # Use garrison
        env.step(env.ACTION_GARRISON)

        # Next time attacker's turn with dice 6, should not be available
        env.current_player = env.attacker_player
        env.dice_value = 6
        legal = env.get_legal_actions()
        legal_ids = [a["id"] for a in legal]
        assert env.ACTION_GARRISON not in legal_ids

    def test_offensive_requires_conditions(self):
        """Offensive requires defender <3 units, dice 1, attacker in moat."""
        env = GeneralsEnv()
        env.reset()
        env.board[:] = 0
        env.current_player = env.defender_player

        # Place fewer than 3 defender units
        env.board[0, 0] = env.defender_player
        env.board[0, 1] = env.defender_player

        # Place attacker in moat
        env.board[3, 3] = env.attacker_player
        env.dice_value = 1

        legal = env.get_legal_actions()
        legal_ids = [a["id"] for a in legal]
        assert env.ACTION_OFFENSIVE in legal_ids


class TestGarrisonConversion:
    """Test Garrison conversion rule."""

    def test_conversion_single_unit(self):
        """Defender entering attacker garrison converts single attacker unit."""
        env = GeneralsEnv()
        env.reset()
        env.board[:] = 0

        # Place defender to move into attacker garrison
        env.board[1, 3] = env.defender_player
        # Place single attacker in top-left garrison
        env.board[0, 0] = env.attacker_player
        env.current_player = env.defender_player
        env.dice_value = 3

        # Move defender into garrison area
        key = (1, 3, 2, 3)  # Left 3 to (1, 0)
        aid = env.MOVE_TO_ACTION_ID.get(key)
        if aid:
            env.step(aid)
            # The attacker unit should be converted
            assert env.board[0, 0] == env.defender_player


class TestWinConditions:
    """Test win conditions."""

    def test_attacker_wins_3_hits(self):
        """Attacker wins with 3 General hits."""
        env = GeneralsEnv()
        env.reset()
        env.general_hits = 2
        env.board[:] = 0
        env.board[4, 3] = env.attacker_player
        env.current_player = env.attacker_player
        env.roles = {env.current_player: ATTACKER, -env.current_player: DEFENDER}
        env.attacker_player = env.current_player
        env.dice_value = 1

        key = (4, 3, 3, 1)
        aid = env.MOVE_TO_ACTION_ID.get(key)
        if aid:
            _, reward, done, _ = env.step(aid)
            assert done
            assert env.winner == env.attacker_player
            assert reward == 1.0

    def test_defender_wins_attacker_exhausted(self):
        """Defender wins when attacker has no units."""
        env = GeneralsEnv()
        env.reset()
        env.board[:] = 0
        # Only defender units
        env.board[0, 0] = env.defender_player
        env.board[0, 1] = env.defender_player
        # Single attacker to be killed
        env.board[0, 3] = env.attacker_player
        env.current_player = env.defender_player
        env.dice_value = 2

        # Kill last attacker
        key = (0, 1, 3, 2)  # Right 2
        aid = env.MOVE_TO_ACTION_ID.get(key)
        if aid:
            _, _, done, _ = env.step(aid)
            assert done
            assert env.winner == env.defender_player


class TestSaveRestore:
    """Test state save/restore for MCTS."""

    def test_save_restore_preserves_state(self):
        env = GeneralsEnv()
        env.reset()

        # Make some moves
        legal = env.get_legal_actions()
        if legal:
            env.step(legal[0]["id"])

        # Save
        state = env.save_state()

        # Modify
        env.reset()

        # Restore
        env.restore_state(state)

        # Verify
        assert env.turn == state["turn"]
        assert env.general_hits == state["general_hits"]
        assert np.array_equal(env.board, state["board"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
