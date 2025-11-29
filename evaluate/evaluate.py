import copy
import torch
import numpy as np

from env.generals_env import GeneralsEnv
from mcts.mcts import MCTS
from model.network import GeneralsNet


class Arena:
    """
    Plays games between two models:
        model_A vs model_B

    Returns win rate of model_A.
    """

    def __init__(self, model_A_path, model_B_path, games=10, mcts_simulations=50):
        self.games = games
        self.mcts_simulations = mcts_simulations

        # Load both models
        self.model_A = GeneralsNet()
        self.model_A.load_state_dict(torch.load(model_A_path, map_location="cpu"))
        self.model_A.eval()

        self.model_B = GeneralsNet()
        self.model_B.load_state_dict(torch.load(model_B_path, map_location="cpu"))
        self.model_B.eval()

    # ----------------------------------------------------
    # Play one game A vs B
    # ----------------------------------------------------
    def play_one_game(self, first_player="A"):
        env = GeneralsEnv()
        state = env.reset()

        while True:
            # Whose turn?
            if (env.current_player == +1 and first_player == "A") or \
               (env.current_player == -1 and first_player == "B"):
                net = self.model_A
            else:
                net = self.model_B

            # Run MCTS with chosen network
            mcts = MCTS(env, net=net, c_puct=1.0)
            mcts.search(n_sims=self.mcts_simulations)

            action = mcts.select_action(temperature=0)  # deterministic

            next_state, reward, done, info = env.step(action)
            state = next_state

            if done:
                # winner is env.winner (1 or -1)
                if env.winner == +1 and first_player == "A":
                    return "A"
                if env.winner == -1 and first_player == "B":
                    return "A"

                return "B"

    # ----------------------------------------------------
    # Play N games and return results
    # ----------------------------------------------------
    def run(self):
        A_wins = 0
        B_wins = 0

        for i in range(self.games):
            # Alternate who starts
            first = "A" if i % 2 == 0 else "B"
            winner = self.play_one_game(first_player=first)

            if winner == "A":
                A_wins += 1
            else:
                B_wins += 1

            print(f"Game {i+1}/{self.games} → Winner: {winner}")

        print("\n=== Arena Results ===")
        print(f"Model A wins: {A_wins}")
        print(f"Model B wins: {B_wins}")
        print(f"Win rate A: {A_wins / self.games:.2f}")

        return A_wins / self.games