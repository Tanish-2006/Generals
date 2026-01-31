"""
Generals Game Environment - Complete Rule Implementation
Based on authoritative PDF rules provided by user.
"""

import numpy as np
import random

ATTACKER = 1
DEFENDER = -1


class GeneralsEnv:
    BOARD_SIZE = 10
    MAX_DICE = 6
    ACTION_DIM = 10004  # Max action ID + 1 for MCTS compatibility

    # Special Action IDs
    ACTION_GARRISON = 10001
    ACTION_OFFENSIVE = 10003

    # 8 directions for movement
    _DIRS = [
        (-1, 0),  # up
        (1, 0),  # down
        (0, -1),  # left
        (0, 1),  # right
        (-1, -1),  # up-left
        (-1, 1),  # up-right
        (1, -1),  # down-left
        (1, 1),  # down-right
    ]

    # Class-level mappings (built once)
    ACTION_ID_TO_MOVE = {}
    MOVE_TO_ACTION_ID = {}

    # Board regions
    GARRISON_CELLS = None
    GENERAL_CELLS = None
    MOAT_CELLS = None

    @classmethod
    def build_board_masks(cls):
        """Build static board masks for General, Moat, and Garrisons."""
        if cls.GENERAL_CELLS is not None:
            return

        center = cls.BOARD_SIZE // 2  # 5

        # General: 4 center cells (2x2)
        cls.GENERAL_CELLS = [
            (center - 1, center - 1),  # (4, 4)
            (center - 1, center),  # (4, 5)
            (center, center - 1),  # (5, 4)
            (center, center),  # (5, 5)
        ]
        cls.GENERAL_CELLS_SET = set(cls.GENERAL_CELLS)

        # Moat: 12 cells surrounding General (4x4 minus 2x2 center)
        moat = []
        for r in range(center - 2, center + 2):
            for c in range(center - 2, center + 2):
                if (r, c) not in cls.GENERAL_CELLS_SET:
                    moat.append((r, c))
        cls.MOAT_CELLS = moat
        cls.MOAT_CELLS_SET = set(moat)

        # Garrisons: 4 corners, 6 cells each
        cls.GARRISON_CELLS = [
            # Top-left garrison (triangular pattern)
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)],
            # Top-right garrison (triangular pattern)
            [(0, 7), (0, 8), (0, 9), (1, 8), (1, 9), (2, 9)],
            # Bottom-left garrison (triangular pattern)
            [(7, 0), (8, 0), (8, 1), (9, 0), (9, 1), (9, 2)],
            # Bottom-right garrison (triangular pattern)
            [(7, 9), (8, 8), (8, 9), (9, 7), (9, 8), (9, 9)],
        ]

        # Create sets for quick lookup
        cls.ATTACKER_GARRISON_CELLS = set(cls.GARRISON_CELLS[0] + cls.GARRISON_CELLS[1])
        cls.DEFENDER_GARRISON_CELLS = set(cls.GARRISON_CELLS[2] + cls.GARRISON_CELLS[3])
        cls.ALL_GARRISON_CELLS = (
            cls.ATTACKER_GARRISON_CELLS | cls.DEFENDER_GARRISON_CELLS
        )

    @classmethod
    def build_action_table(cls):
        """Build action ID mappings for all possible moves."""
        if cls.ACTION_ID_TO_MOVE:
            return
        cls.build_board_masks()

        action_id = 1
        for r in range(cls.BOARD_SIZE):
            for c in range(cls.BOARD_SIZE):
                for d_idx, (dr, dc) in enumerate(cls._DIRS):
                    for dist in range(1, cls.MAX_DICE + 1):
                        tr = r + dr * dist
                        tc = c + dc * dist
                        if 0 <= tr < cls.BOARD_SIZE and 0 <= tc < cls.BOARD_SIZE:
                            key = (r, c, d_idx, dist)
                            cls.ACTION_ID_TO_MOVE[action_id] = key
                            cls.MOVE_TO_ACTION_ID[key] = action_id
                            action_id += 1

        # Special actions
        cls.ACTION_ID_TO_MOVE[cls.ACTION_GARRISON] = ("GARRISON",)
        cls.ACTION_ID_TO_MOVE[cls.ACTION_OFFENSIVE] = ("OFFENSIVE",)

    def __init__(self):
        self.build_action_table()
        self.reset()

    def reset(self):
        # Board: -1 = Defender unit, 0 = Empty, 1 = Attacker unit
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)

        # Toss & Role Assignment
        toss_winner = random.choice([1, -1])
        self.roles = {toss_winner: ATTACKER, -toss_winner: DEFENDER}
        self.attacker_player = toss_winner
        self.defender_player = -toss_winner

        # Attacker always moves first
        self.current_player = self.attacker_player

        # Place units: Attacker in top garrisons, Defender in bottom garrisons
        for pos in self.GARRISON_CELLS[0]:  # Top-left
            self.board[pos] = self.attacker_player
        for pos in self.GARRISON_CELLS[1]:  # Top-right
            self.board[pos] = self.attacker_player
        for pos in self.GARRISON_CELLS[2]:  # Bottom-left
            self.board[pos] = self.defender_player
        for pos in self.GARRISON_CELLS[3]:  # Bottom-right
            self.board[pos] = self.defender_player

        # Game state
        self.general_hits = 0
        self.dice_value = self._roll_dice()
        self.turn = 0
        self.winner = None
        self.done = False

        # Special move tracking (once per game per player)
        self.garrison_used = {1: False, -1: False}
        self.offensive_used = False  # Only defender's general can use

        return self.encode_state()

    def _roll_dice(self):
        """Roll dice (1-6)."""
        return random.randint(1, self.MAX_DICE)

    def _count_units(self, player):
        """Count units belonging to a player."""
        return int(np.sum(self.board == player))

    def _is_path_clear(self, fr, fc, tr, tc):
        """
        Check if path from (fr, fc) to (tr, tc) is clear.
        Path cannot cross:
        - Other units (friendly or enemy)
        - General cells (never crossable)
        """
        dr = tr - fr
        dc = tc - fc
        steps = max(abs(dr), abs(dc))

        if steps <= 0:
            return False

        step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
        step_c = 0 if dc == 0 else (1 if dc > 0 else -1)

        r, c = fr + step_r, fc + step_c

        # Check all intermediate cells (not including target)
        for _ in range(steps - 1):
            # Cannot cross through any unit
            if self.board[r, c] != 0:
                return False
            # Cannot cross through General cells
            if (r, c) in self.GENERAL_CELLS_SET:
                return False
            r += step_r
            c += step_c

        return True

    def _get_movement_legal_actions(self):
        """
        Get all legal movement actions for current player.
        Movement distance must EXACTLY equal dice value.
        """
        legal = []
        cur_role = self.roles[self.current_player]
        dist = self.dice_value  # Exact distance required

        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                # Must be current player's unit
                if self.board[r, c] != self.current_player:
                    continue

                for d_idx, (dr, dc) in enumerate(self._DIRS):
                    tr = r + dr * dist
                    tc = c + dc * dist

                    # Check bounds
                    if not (0 <= tr < self.BOARD_SIZE and 0 <= tc < self.BOARD_SIZE):
                        continue

                    # Check path is clear
                    if not self._is_path_clear(r, c, tr, tc):
                        continue

                    # Target cell checks
                    target = self.board[tr, tc]

                    # Cannot move onto friendly unit (no stacking)
                    if target == self.current_player:
                        continue

                    # General cell restrictions
                    if (tr, tc) in self.GENERAL_CELLS_SET:
                        # Only Attacker can enter General cells
                        if cur_role == DEFENDER:
                            continue

                    key = (r, c, d_idx, dist)
                    aid = self.MOVE_TO_ACTION_ID.get(key)
                    if aid:
                        legal.append(aid)

        return legal

    def _can_garrison(self):
        if self.dice_value != 6:
            return False
        if self.garrison_used[self.current_player]:
            return False

        # Must have at least one unit in own garrison
        my_garrisons = (
            self.GARRISON_CELLS[0:2]
            if self.roles[self.current_player] == ATTACKER
            else self.GARRISON_CELLS[2:4]
        )
        for group in my_garrisons:
            for pos in group:
                if self.board[pos] == self.current_player:
                    return True
        return False

    def _can_offensive(self):
        # Only Defender's General can use this
        if self.roles[self.current_player] != DEFENDER:
            return False
        if self.offensive_used:
            return False
        if self.dice_value != 1:
            return False
        # Defender units on board must be fewer than 3
        if self._count_units(self.defender_player) >= 3:
            return False
        # Must have at least one attacker unit in Moat
        for pos in self.MOAT_CELLS:
            if self.board[pos] == self.attacker_player:
                return True
        return False

    def get_legal_actions(self, dice_value=None):
        """
        Get all legal actions for current player.
        Returns list of dicts with 'id' key for compatibility.
        """
        if dice_value is not None:
            old_dice = self.dice_value
            self.dice_value = dice_value

        legal = self._get_movement_legal_actions()

        # Garrison special move
        if self._can_garrison():
            legal.append(self.ACTION_GARRISON)

        # Offensive special move
        if self._can_offensive():
            legal.append(self.ACTION_OFFENSIVE)

        if dice_value is not None:
            self.dice_value = old_dice

        return [{"id": int(a)} for a in legal]

    def _apply_move(self, fr, fc, tr, tc):
        """
        Apply a movement from (fr, fc) to (tr, tc).
        Handles combat and General interactions.
        """
        mover = self.board[fr, fc]
        if mover == 0:
            return

        _target = self.board[tr, tc]  # Captured for potential future use

        # Combat: Simple kill - moving unit always wins
        # (Rules say no unit counting)
        self.board[fr, fc] = 0

        # Check if attacking General
        if (tr, tc) in self.GENERAL_CELLS_SET:
            # Only Attacker can hit General
            if self.roles[self.current_player] == ATTACKER:
                # Check for Defender units in Moat
                defenders_in_moat = []
                for pos in self.MOAT_CELLS:
                    if self.board[pos] == self.defender_player:
                        defenders_in_moat.append(pos)

                if len(defenders_in_moat) == 0:
                    # No defenders in moat - General takes a hit!
                    self.general_hits += 1
                    # Attacker unit is consumed (doesn't stay on General)
                    self.board[tr, tc] = 0
                else:
                    # Intercepted by Moat defender
                    # Both attacker and one defender die
                    self.board[tr, tc] = 0  # Attacker dies
                    # Remove one defender from moat (random selection per rules)
                    remove_pos = random.choice(defenders_in_moat)
                    self.board[remove_pos] = 0
            else:
                # Defender should never reach here (blocked in legal actions)
                pass
        else:
            # Normal move/capture
            self.board[tr, tc] = mover

    def _apply_garrison(self):
        """
        Execute Garrison special move.
        Move one unit from own garrison to any empty non-garrison cell.
        """
        # Find own garrison cells with units
        my_garrisons = (
            self.GARRISON_CELLS[0:2]
            if self.roles[self.current_player] == ATTACKER
            else self.GARRISON_CELLS[2:4]
        )
        source_cells = []
        for group in my_garrisons:
            for pos in group:
                if self.board[pos] == self.current_player:
                    source_cells.append(pos)

        if not source_cells:
            return False

        # Find all empty non-garrison cells
        target_cells = []
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if self.board[r, c] == 0:
                    if (r, c) not in self.ALL_GARRISON_CELLS:
                        if (r, c) not in self.GENERAL_CELLS_SET:
                            target_cells.append((r, c))

        if not target_cells:
            return False

        # Execute: move first available source to random target
        # (In actual game, player chooses; for AI, we pick randomly)
        src = random.choice(source_cells)
        tgt = random.choice(target_cells)
        self.board[src] = 0
        self.board[tgt] = self.current_player

        self.garrison_used[self.current_player] = True
        return True

    def _apply_offensive(self):
        """
        Execute Offensive special move.
        General eliminates one Attacker unit in Moat.
        """
        # Find attacker units in Moat
        targets = []
        for pos in self.MOAT_CELLS:
            if self.board[pos] == self.attacker_player:
                targets.append(pos)

        if not targets:
            return False

        # Remove one (random selection for AI)
        target = random.choice(targets)
        self.board[target] = 0

        self.offensive_used = True
        return True

    def _check_garrison_conversion(self, tr, tc):
        """
        Check and apply Garrison Conversion rule after Defender moves.
        If Defender enters Attacker's garrison:
        - If exactly 1 attacker unit in that garrison zone -> convert it
        - If >1 attacker units -> convert 1 (random for AI)
        """
        if self.roles[self.current_player] != DEFENDER:
            return

        # Determine which Attacker garrison zone (if any) the defender entered
        target_group = None
        if (tr, tc) in set(self.GARRISON_CELLS[0]):
            target_group = 0
        elif (tr, tc) in set(self.GARRISON_CELLS[1]):
            target_group = 1

        if target_group is None:
            return  # Not in attacker garrison

        # Count attacker units in that specific garrison zone
        group = self.GARRISON_CELLS[target_group]
        attacker_units_in_zone = []
        for pos in group:
            if self.board[pos] == self.attacker_player:
                attacker_units_in_zone.append(pos)

        if len(attacker_units_in_zone) == 1:
            # Convert it to defender
            pos = attacker_units_in_zone[0]
            self.board[pos] = self.defender_player
        elif len(attacker_units_in_zone) > 1:
            # Convert one (random for AI, player chooses in real game)
            pos = random.choice(attacker_units_in_zone)
            self.board[pos] = self.defender_player

    def _check_win_conditions(self):
        """Check if game has ended."""
        # Attacker wins: 3 General hits
        if self.general_hits >= 3:
            self.winner = self.attacker_player
            self.done = True
            return

        # Defender wins: All attacker units eliminated
        if self._count_units(self.attacker_player) == 0:
            self.winner = self.defender_player
            self.done = True
            return

    def step(self, action_id):
        """
        Execute an action and advance game state.
        Returns: (state, reward, done, info)
        """
        info = {}
        mover = self.current_player  # Store before potential flip

        if self.done:
            return self.encode_state(), 0.0, True, info

        # Execute action
        if action_id == self.ACTION_GARRISON:
            self._apply_garrison()

        elif action_id == self.ACTION_OFFENSIVE:
            self._apply_offensive()

        elif action_id in self.ACTION_ID_TO_MOVE:
            move = self.ACTION_ID_TO_MOVE[action_id]
            if isinstance(move, tuple) and len(move) == 4:
                fr, fc, d_idx, dist = move
                dr, dc = self._DIRS[d_idx]
                tr = fr + dr * dist
                tc = fc + dc * dist

                if self._is_path_clear(fr, fc, tr, tc):
                    self._apply_move(fr, fc, tr, tc)
                    self._check_garrison_conversion(tr, tc)

        # Check win conditions
        self._check_win_conditions()

        # Calculate reward (from mover's perspective)
        reward = 0.0
        if self.done:
            if self.winner == mover:
                reward = 1.0
            else:
                reward = -1.0

        # Advance turn if not done
        if not self.done:
            self.turn += 1
            self.current_player *= -1
            self.dice_value = self._roll_dice()

            # Handle STALEMATE: Auto-reroll if no legal moves
            # Keep rerolling until a legal move exists
            max_rerolls = 100  # Safety limit
            rerolls = 0
            while (
                len(self._get_movement_legal_actions()) == 0 and rerolls < max_rerolls
            ):
                # Check if any special moves are available
                if self._can_garrison() or self._can_offensive():
                    break
                self.dice_value = self._roll_dice()
                rerolls += 1

            if rerolls > 0:
                info["stalemate_rerolls"] = rerolls

        return self.encode_state(), reward, self.done, info

    def encode_state(self, dice_value=None):
        """
        Encode board state for neural network input.
        9 channels, 10x10 each.
        """
        if dice_value is None:
            dice_value = self.dice_value

        channels = 9
        state = np.zeros((channels, self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)

        cur = self.current_player
        opp = -cur

        # Ch 0: My units (binary)
        state[0] = (self.board == cur).astype(np.float32)

        # Ch 1: Opponent units (binary)
        state[1] = (self.board == opp).astype(np.float32)

        # Ch 2: General location
        for r, c in self.GENERAL_CELLS:
            state[2, r, c] = 1.0

        # Ch 3: My role (1.0 if Attacker)
        if self.roles[cur] == ATTACKER:
            state[3, :, :] = 1.0

        # Ch 4: Opponent role (1.0 if Attacker)
        if self.roles[opp] == ATTACKER:
            state[4, :, :] = 1.0

        # Ch 5: Board ownership (1 for mine, -1 for opponent, 0 empty)
        state[5] = np.sign(self.board * cur).astype(np.float32)

        # Ch 6: Dice value (normalized)
        state[6, :, :] = dice_value / float(self.MAX_DICE)

        # Ch 7: General hits (normalized, max 3)
        state[7, :, :] = self.general_hits / 3.0

        # Ch 8: Moat cells
        for r, c in self.MOAT_CELLS:
            state[8, r, c] = 1.0

        return state

    def save_state(self):
        return {
            "board": self.board.copy(),
            "roles": self.roles.copy(),
            "attacker_player": self.attacker_player,
            "defender_player": self.defender_player,
            "current_player": int(self.current_player),
            "general_hits": int(self.general_hits),
            "dice_value": int(self.dice_value),
            "turn": int(self.turn),
            "winner": self.winner,
            "done": self.done,
            "garrison_used": self.garrison_used.copy(),
            "offensive_used": self.offensive_used,
        }

    def restore_state(self, state):
        self.board = state["board"].copy()
        self.roles = state["roles"].copy()
        self.attacker_player = state["attacker_player"]
        self.defender_player = state["defender_player"]
        self.current_player = int(state["current_player"])
        self.general_hits = int(state["general_hits"])
        self.dice_value = int(state["dice_value"])
        self.turn = int(state["turn"])
        self.winner = state["winner"]
        self.done = state.get("done", False)
        self.garrison_used = state.get("garrison_used", {1: False, -1: False}).copy()
        self.offensive_used = state.get("offensive_used", False)

    def render(self):
        print(
            f"\nTurn {self.turn} | Player: {self.current_player} | Dice: {self.dice_value}"
        )
        print(f"Attacker: {self.attacker_player} | Defender: {self.defender_player}")
        print(f"General Hits: {self.general_hits}")
        print("-" * 21)
        for r in range(self.BOARD_SIZE):
            row_str = ""
            for c in range(self.BOARD_SIZE):
                val = self.board[r, c]
                if val == 1:
                    row_str += " A"
                elif val == -1:
                    row_str += " D"
                elif (r, c) in self.GENERAL_CELLS_SET:
                    row_str += " G"
                elif (r, c) in self.MOAT_CELLS_SET:
                    row_str += " M"
                else:
                    row_str += " ."
            print(row_str)
        print("-" * 21)
