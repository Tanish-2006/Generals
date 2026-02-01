import asyncio
import numpy as np
from mcts.mcts import AsyncMCTS


class SelfPlay:
    def __init__(
        self,
        env_class,
        inference_server,
        games_per_iteration=1,
        mcts_simulations=100,
        temperature_threshold=12,
        max_moves=300,
    ):
        self.env_class = env_class
        self.inference_server = inference_server
        self.games_per_iteration = games_per_iteration
        self.mcts_simulations = mcts_simulations
        self.temperature_threshold = temperature_threshold
        self.max_moves = max_moves

    async def play_iteration(self):
        tasks = [self.play_one_game() for _ in range(self.games_per_iteration)]
        results = await asyncio.gather(*tasks)

        all_states = []
        all_policies = []
        all_values = []

        for s, p, v in results:
            all_states.extend(s)
            all_policies.extend(p)
            all_values.extend(v)

        return np.array(all_states), np.array(all_policies), np.array(all_values)

    async def play_one_game(self):
        env = self.env_class()
        state = env.reset()

        states = []
        policies = []
        current_players = []

        move_counter = 0

        while True:
            if move_counter >= self.max_moves:
                values = [0.0] * len(current_players)
                return states, policies, values

            mcts = AsyncMCTS(env, inference_server=self.inference_server, c_puct=1.0)
            await mcts.search(n_sims=self.mcts_simulations)

            if move_counter < self.temperature_threshold:
                temp = 1.0
            else:
                temp = 0.1

            pi = mcts.get_action_probs(temperature=temp)

            states.append(state)
            policies.append(pi)
            current_players.append(env.current_player)

            action = mcts.select_action(temperature=temp)

            next_state, reward, done, _ = env.step(action)

            state = next_state
            move_counter += 1

            if done:
                values = []
                for player in current_players:
                    if reward == 0:
                        z = 0
                    else:
                        z = reward if player == env.winner else -reward
                    values.append(z)

                return states, policies, values
