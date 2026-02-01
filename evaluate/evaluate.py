import torch
import os
import asyncio
from datetime import datetime

from env.generals_env import GeneralsEnv
from mcts.mcts import AsyncMCTS
from model.network import GeneralsNet


class Arena:
    def __init__(
        self,
        model_A_path,
        model_B_path,
        games=10,
        mcts_simulations=50,
        save_logs=True,
        log_dir="game_logs",
    ):
        self.games = games
        self.mcts_simulations = mcts_simulations
        self.save_logs = save_logs
        self.log_dir = log_dir

        if self.save_logs:
            os.makedirs(self.log_dir, exist_ok=True)

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

    def _render_board_to_string(self, env):
        lines = []
        lines.append(
            f"\nTurn {env.turn} | Player: {env.current_player} | Dice: {env.dice_value}"
        )
        lines.append(
            f"Attacker: {env.attacker_player} | Defender: {env.defender_player}"
        )
        lines.append(f"General Hits: {env.general_hits}")
        lines.append("-" * 21)
        for r in range(env.BOARD_SIZE):
            row_str = ""
            for c in range(env.BOARD_SIZE):
                val = env.board[r, c]
                if val == 1:
                    row_str += " A"
                elif val == -1:
                    row_str += " D"
                elif (r, c) in env.GENERAL_CELLS_SET:
                    row_str += " G"
                elif (r, c) in env.MOAT_CELLS_SET:
                    row_str += " M"
                else:
                    row_str += " ."
            lines.append(row_str)
        lines.append("-" * 21)
        return "\n".join(lines)

    async def _save_game_log(self, game_id, logs):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.log_dir, f"game_{game_id}_{timestamp}.txt")
        await asyncio.to_thread(self._write_log_file, filename, logs)

    def _write_log_file(self, filename, logs):
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(logs))

    async def play_one_game(self, game_id, first_player="A"):
        env = GeneralsEnv()
        _state = env.reset()
        game_logs = []

        if self.save_logs:
            game_logs.append(f"Game {game_id} Start. First Player: {first_player}")
            game_logs.append(self._render_board_to_string(env))

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
            _state = next_state

            if self.save_logs:
                game_logs.append(
                    f"\nAction taken: {action} (Player {env.current_player * -1})"
                )
                game_logs.append(self._render_board_to_string(env))

            if done:
                # Winner is stored as player ID in env.winner
                # Model A plays as the player assigned "first_player"
                # Model B plays as the other player
                if first_player == "A":
                    model_A_player = +1
                else:
                    model_A_player = -1

                if env.winner == model_A_player:
                    winner_name = "A"
                else:
                    winner_name = "B"

                if self.save_logs:
                    game_logs.append(f"\nGame Over. Winner: {winner_name}")
                    asyncio.create_task(self._save_game_log(game_id, game_logs))

                return winner_name

    async def run(self):
        A_wins = 0
        B_wins = 0

        for i in range(self.games):
            first = "A" if i % 2 == 0 else "B"
            winner = await self.play_one_game(game_id=i + 1, first_player=first)

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
