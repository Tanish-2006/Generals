# generals_env.py

import random
import numpy as np


class GeneralsEnv:
    BOARD_SIZE = 10

    ACTION_GARRISON = 10001
    ACTION_STALEMATE = 10002
    ACTION_DIM = 10003

    ACTION_ID_TO_MOVE = {}

    @classmethod
    def build_action_table(cls):
        if cls.ACTION_ID_TO_MOVE:
            return

        action_id = 1

        for r in range(cls.BOARD_SIZE):
            for c in range(cls.BOARD_SIZE):

                if r > 0:
                    cls.ACTION_ID_TO_MOVE[action_id] = (r, c, r - 1, c)
                    action_id += 1

                if r < cls.BOARD_SIZE - 1:
                    cls.ACTION_ID_TO_MOVE[action_id] = (r, c, r + 1, c)
                    action_id += 1

                if c > 0:
                    cls.ACTION_ID_TO_MOVE[action_id] = (r, c, r, c - 1)
                    action_id += 1

                if c < cls.BOARD_SIZE - 1:
                    cls.ACTION_ID_TO_MOVE[action_id] = (r, c, r, c + 1)
                    action_id += 1

        cls.ACTION_ID_TO_MOVE[cls.ACTION_GARRISON] = ("GARRISON",)
        cls.ACTION_ID_TO_MOVE[cls.ACTION_STALEMATE] = ("STALE",)

    def __init__(self):
        self.build_action_table()
        self.reset()

    def reset(self):
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int32)
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
        legal = []

        for action_id, move in self.ACTION_ID_TO_MOVE.items():
            if isinstance(move, tuple) and len(move) == 4:
                legal.append({"id": action_id})

        legal.append({"id": self.ACTION_GARRISON})
        legal.append({"id": self.ACTION_STALEMATE})

        return legal

    def _apply_normal_move(self, fr, fc, tr, tc):
        self.board[tr][tc] = self.board[fr][fc]
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
            'board': self.board.copy(),
            'current_player': self.current_player,
            'general_hits': self.general_hits,
            'dice_value': self.dice_value,
            'turn': self.turn,
            'winner': self.winner
        }

    def restore_state(self, state):
        self.board = state['board'].copy()
        self.current_player = state['current_player']
        self.general_hits = state['general_hits']
        self.dice_value = state['dice_value']
        self.turn = state['turn']
        self.winner = state['winner']