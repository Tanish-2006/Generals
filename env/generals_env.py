import random
import numpy as np


class GeneralsEnv:
    BOARD_SIZE = 10

    ACTION_GARRISON = 10001
    ACTION_STALEMATE = 10002
    ACTION_DIM = 10003

    ACTION_ID_TO_MOVE = {}

    MOVE_SRCS = None  # (N_MOVES, 2) -> r, c
    MOVE_TGT_INDICES = None  # (N_MOVES, 2) -> tr, tc
    MOVE_IDS = None  # (N_MOVES,)

    GARRISON_LOCATIONS = {1: (0, 0), -1: (9, 9)}

    @classmethod
    def build_action_table(cls):
        if cls.ACTION_ID_TO_MOVE:
            return

        action_id = 1

        srcs = []
        tgts = []
        ids = []

        for r in range(cls.BOARD_SIZE):
            for c in range(cls.BOARD_SIZE):
                # Up
                if r > 0:
                    cls.ACTION_ID_TO_MOVE[action_id] = (r, c, r - 1, c)
                    srcs.append([r, c])
                    tgts.append([r - 1, c])
                    ids.append(action_id)
                    action_id += 1

                # Down
                if r < cls.BOARD_SIZE - 1:
                    cls.ACTION_ID_TO_MOVE[action_id] = (r, c, r + 1, c)
                    srcs.append([r, c])
                    tgts.append([r + 1, c])
                    ids.append(action_id)
                    action_id += 1

                # Left
                if c > 0:
                    cls.ACTION_ID_TO_MOVE[action_id] = (r, c, r, c - 1)
                    srcs.append([r, c])
                    tgts.append([r, c - 1])
                    ids.append(action_id)
                    action_id += 1

                # Right
                if c < cls.BOARD_SIZE - 1:
                    cls.ACTION_ID_TO_MOVE[action_id] = (r, c, r, c + 1)
                    srcs.append([r, c])
                    tgts.append([r, c + 1])
                    ids.append(action_id)
                    action_id += 1

        cls.ACTION_ID_TO_MOVE[cls.ACTION_GARRISON] = ("GARRISON",)
        cls.ACTION_ID_TO_MOVE[cls.ACTION_STALEMATE] = ("STALE",)

        cls.MOVE_SRCS = np.array(srcs, dtype=np.int32)
        cls.MOVE_TGTS = np.array(tgts, dtype=np.int32)
        cls.MOVE_IDS = np.array(ids, dtype=np.int32)

    def __init__(self):
        self.build_action_table()
        self.reset()

    def reset(self):
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int32)
        self.board[0, 0] = 1
        self.board[9, 9] = -1

        self.current_player = 1
        self.general_hits = 0
        self.dice_value = self.roll_dice()
        self.turn = 0
        self.MAX_TURNS = 300
        self.winner = None
        return self.encode_state()

    def roll_dice(self):
        return random.randint(1, 6)

    def encode_state(self, dice_value=None):
        if dice_value is None:
            dice_value = self.dice_value

        state = np.zeros((17, self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)

        state[0] = self.board.copy()
        state[1] = 0
        state[2] = dice_value / 6.0

        return state

    def get_legal_actions(self, dice_value=None):
        src_values = self.board[self.MOVE_SRCS[:, 0], self.MOVE_SRCS[:, 1]]
        has_piece = np.sign(src_values) == np.sign(self.current_player)
        legal_move_ids = self.MOVE_IDS[has_piece]

        legal = [{"id": int(aid)} for aid in legal_move_ids]

        legal.append({"id": self.ACTION_GARRISON})
        legal.append({"id": self.ACTION_STALEMATE})

        return legal

    def _apply_normal_move(self, fr, fc, tr, tc):
        source_val = self.board[fr][fc]
        target_val = self.board[tr][tc]

        moved_units = source_val

        enemy_player = -self.current_player
        is_enemy_garrison = False
        if enemy_player in self.GARRISON_LOCATIONS:
            gr, gc = self.GARRISON_LOCATIONS[enemy_player]
            if tr == gr and tc == gc:
                is_enemy_garrison = True

        if is_enemy_garrison:
            if target_val != 0 and np.sign(target_val) == np.sign(enemy_player):
                if abs(target_val) > 0:
                    target_val -= np.sign(target_val)
                    moved_units += np.sign(self.current_player)

        if moved_units == 0:
            self.board[fr][fc] = 0
            return

        if target_val == 0:
            self.board[tr][tc] = moved_units

        elif np.sign(target_val) == np.sign(moved_units):
            self.board[tr][tc] = target_val + moved_units

        else:
            diff = abs(moved_units) - abs(target_val)
            if diff > 0:
                self.board[tr][tc] = np.sign(moved_units) * diff
            elif diff < 0:
                self.board[tr][tc] = np.sign(target_val) * abs(diff)
            else:
                self.board[tr][tc] = 0

        self.board[fr][fc] = 0

    def step(self, action_id):
        reward = 0
        done = False
        info = {}

        if action_id not in self.ACTION_ID_TO_MOVE:
            return self.encode_state(), -0.01, False, {}

        if action_id == 0:
            reward = -0.001

        elif 1 <= action_id < self.ACTION_GARRISON:
            fr, fc, tr, tc = self.ACTION_ID_TO_MOVE[action_id]
            self._apply_normal_move(fr, fc, tr, tc)
            reward = 0.001

        elif action_id == self.ACTION_GARRISON:
            self.general_hits += 1
            reward = 0.1

        elif action_id == self.ACTION_STALEMATE:
            self.general_hits += 2
            reward = 0.2

        if self.general_hits >= 3:
            done = True
            reward = +1.0
            self.winner = self.current_player
            return self.encode_state(), reward, done, info

        self.turn += 1
        if self.turn >= self.MAX_TURNS:
            done = True
            reward = 0
            self.winner = 0
            return self.encode_state(), reward, done, info

        self.current_player *= -1
        self.dice_value = self.roll_dice()

        return self.encode_state(), reward, done, info

    def save_state(self):
        return {
            "board": self.board.copy(),
            "current_player": self.current_player,
            "general_hits": self.general_hits,
            "dice_value": self.dice_value,
            "turn": self.turn,
            "winner": self.winner,
        }

    def restore_state(self, state):
        self.board = state["board"].copy()
        self.current_player = state["current_player"]
        self.general_hits = state["general_hits"]
        self.dice_value = state["dice_value"]
        self.turn = state["turn"]
        self.winner = state["winner"]
