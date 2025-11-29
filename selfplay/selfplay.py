import numpy as np
from mcts.mcts import MCTS


class SelfPlay:
    """
    Generates training samples using MCTS + GeneralsEnv + Network.
    Each game returns:
        states  -> list of (17,10,10) numpy arrays
        policies -> list of length-10003 arrays (π)
        values -> list of floats (game result from player's perspective)
    """

    def __init__(self, env_class, net, games_per_iteration=1,
                 mcts_simulations=100, temperature_threshold=12):
        """
        env_class: reference to GeneralsEnv class
        net: neural network (GeneralsNet)
        games_per_iteration: how many full self-play games to generate
        mcts_simulations: number of MCTS simulations per move
        temperature_threshold: after this move, temperature=0 (deterministic)
        """
        self.env_class = env_class
        self.net = net
        self.games_per_iteration = games_per_iteration
        self.mcts_simulations = mcts_simulations
        self.temperature_threshold = temperature_threshold

    # -------------------------------------------------------------
    # Run self-play for N games
    # -------------------------------------------------------------
    def play_iteration(self):
        all_states = []
        all_policies = []
        all_values = []

        for g in range(self.games_per_iteration):
            s, p, v = self.play_one_game()
            all_states.extend(s)
            all_policies.extend(p)
            all_values.extend(v)

        return np.array(all_states), np.array(all_policies), np.array(all_values)

    # -------------------------------------------------------------
    # Play one full game
    # -------------------------------------------------------------
    def play_one_game(self):
        env = self.env_class()
        state = env.reset()

        states = []
        policies = []
        current_players = []

        move_counter = 0

        while True:
            # 1. Run MCTS on current state
            mcts = MCTS(env, net=self.net, c_puct=1.0)
            mcts.search(n_sims=self.mcts_simulations)

            # 2. Temperature rule (explore early, exploit late)
            if move_counter < self.temperature_threshold:
                temp = 1.0
            else:
                temp = 0.1     # almost deterministic

            pi = mcts.get_action_probs(temperature=temp)

            # 3. Save training data
            states.append(state)
            policies.append(pi)
            current_players.append(env.current_player)

            # 4. Choose action
            action = mcts.select_action(temperature=temp)

            # 5. Apply action
            next_state, reward, done, _ = env.step(action)

            state = next_state
            move_counter += 1

            if done:
                # convert final reward into value for each state
                values = []
                for player in current_players:
                    # value z = +1 if player won, -1 if lost
                    if reward == 0:
                        z = 0
                    else:
                        z = reward if player == env.winner else -reward
                    values.append(z)

                return states, policies, values