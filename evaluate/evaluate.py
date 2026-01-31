import copy
import torch
import numpy as np
import asyncio

from env.generals_env import GeneralsEnv
from mcts.mcts import AsyncMCTS
from model.network import GeneralsNet


class Arena:
    def __init__(self, model_A_path, model_B_path, games=10, mcts_simulations=50):
        self.games = games
        self.mcts_simulations = mcts_simulations

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Arena] Using device: {self.device}")

        self.model_A = GeneralsNet().to(self.device)
        self.model_A.load_state_dict(
            torch.load(model_A_path, map_location=self.device, weights_only=False)
        )
        self.model_A.eval()

        self.model_B = GeneralsNet().to(self.device)
        self.model_B.load_state_dict(
            torch.load(model_B_path, map_location=self.device, weights_only=False)
        )
        self.model_B.eval()

    async def play_one_game(self, first_player="A"):
        env = GeneralsEnv()
        state = env.reset()

        while True:
            if (env.current_player == +1 and first_player == "A") or (
                env.current_player == -1 and first_player == "B"
            ):
                net = self.model_A
            else:
                net = self.model_B

            mcts = AsyncMCTS(env, net=net, c_puct=1.0)
            await mcts.search(n_sims=self.mcts_simulations)

            action = mcts.select_action(temperature=0)

            next_state, reward, done, info = env.step(action)
            state = next_state

            if done:
                # Winner is stored as player ID in env.winner
                # Model A plays as the player assigned "first_player"
                # Model B plays as the other player
                if first_player == "A":
                    model_A_player = +1
                else:
                    model_A_player = -1

                if env.winner == model_A_player:
                    return "A"
                else:
                    return "B"

    async def run(self):
        A_wins = 0
        B_wins = 0

        for i in range(self.games):
            first = "A" if i % 2 == 0 else "B"
            winner = await self.play_one_game(first_player=first)

            if winner == "A":
                A_wins += 1
            else:
                B_wins += 1

            print(f"Game {i + 1}/{self.games} → Winner: {winner}")

        print("\n=== Arena Results ===")
        print(f"Model A wins: {A_wins}")
        print(f"Model B wins: {B_wins}")
        print(f"Win rate A: {A_wins / self.games:.2f}")

        return A_wins / self.games
