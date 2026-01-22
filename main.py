import asyncio
import os
import shutil
import time
from pathlib import Path

import numpy as np

from env.generals_env import GeneralsEnv
from selfplay.selfplay import SelfPlay
from training.replay_buffer import ReplayBuffer
from training.train import Trainer
from evaluate.evaluate import Arena
from utils.batched_inference import InferenceServer
from config import TRAINING, EVAL, PATHS


def ensure_dirs():
    PATHS.ensure_dirs()


def checkpoint_paths():
    latest = PATHS.checkpoint_dir / "model_latest.pth"
    old = PATHS.checkpoint_dir / "model_old.pth"
    return str(latest), str(old)


async def main_loop(max_iterations=None):
    ensure_dirs()
    latest_path, old_path = checkpoint_paths()

    if not os.path.exists(old_path):
        if os.path.exists(latest_path):
            shutil.copyfile(latest_path, old_path)
            print(f"[main] model_old did not exist - copied model_latest -> model_old")
        else:
            print("[main] No models found. Creating an initial dummy model.")
            trainer0 = Trainer(
                lr=TRAINING.learning_rate,
                weight_decay=TRAINING.weight_decay,
                batch_size=TRAINING.batch_size,
                epochs=TRAINING.train_epochs,
                checkpoint_dir=str(PATHS.checkpoint_dir)
            )
            states = np.zeros((8, 17, 10, 10), dtype=np.float32)
            policies = np.zeros((8, 10003), dtype=np.float32)
            policies[:, 0] = 1.0
            values = np.zeros((8,), dtype=np.float32)
            trainer0.train(states, policies, values, save_name="model_latest.pth")
            shutil.copyfile(latest_path, old_path)
            print("[main] Created initial model_latest.pth and copied to model_old.pth")

    replay = ReplayBuffer(
        save_dir=str(PATHS.replay_dir),
        max_batches=TRAINING.max_replay_batches
    )
    
    trainer = Trainer(
        lr=TRAINING.learning_rate,
        weight_decay=TRAINING.weight_decay,
        batch_size=TRAINING.batch_size,
        epochs=TRAINING.train_epochs,
        checkpoint_dir=str(PATHS.checkpoint_dir)
    )

    inference_server = InferenceServer(trainer.net, batch_size=32)
    await inference_server.start()

    iteration = 0
    try:
        while True:
            iteration += 1
            if max_iterations is not None and iteration > max_iterations:
                print(f"[main] reached max_iterations={max_iterations}. Stopping.")
                break

            print("\n" + "=" * 60)
            print(f"[main] ITERATION {iteration} - self-play {TRAINING.games_per_iter} games")
            print("=" * 60)

            sp = SelfPlay(
                GeneralsEnv,
                inference_server,
                games_per_iteration=TRAINING.games_per_iter,
                mcts_simulations=TRAINING.mcts_simulations,
                temperature_threshold=TRAINING.temperature_threshold
            )

            print(f"[main] Generating {TRAINING.games_per_iter} games concurrently...")
            states, policies, values = await sp.play_iteration()
            
            replay.add_game(states, policies, values)

            print("[main] Loading replay data to train")
            states_all, policies_all, values_all = replay.load_all()

            save_name = "model_latest.pth"
            print(f"[main] Training for {TRAINING.train_epochs} epochs...")
            trainer.train(states_all, policies_all, values_all, save_name=save_name)
            latest_path, old_path = checkpoint_paths()

            print("[main] Reloading InferenceServer with new model weights...")
            inference_server.reload_model(latest_path)

            print("[main] Evaluating new model vs previous model (Arena)")
            arena = Arena(
                model_A_path=latest_path,
                model_B_path=old_path,
                games=EVAL.eval_games,
                mcts_simulations=EVAL.mcts_simulations
            )

            win_rate = await arena.run()
            print(f"[main] Arena win rate for new model: {win_rate:.2f}")

            if win_rate > EVAL.acceptance_threshold:
                print("[main] New model accepted! Copying model_latest -> model_old")
                shutil.copyfile(latest_path, old_path)
            else:
                print("[main] New model rejected. Keeping previous model_old.")

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt - stopping training loop.")
    except Exception as e:
        print(f"[main] Exception occurred: {e}")
        raise
    finally:
        await inference_server.stop()


if __name__ == "__main__":
    print("=" * 60)
    print(" GENERAL - Strategic Game AI Training System")
    print("=" * 60)
    print(f" - Games per iteration: {TRAINING.games_per_iter}")
    print(f" - MCTS simulations: {TRAINING.mcts_simulations}")
    print(f" - Arena games: {EVAL.eval_games}")
    print(f" - Learning rate: {TRAINING.learning_rate}")
    print(f" - Network: 196 channels, 7 res blocks")
    print("=" * 60)
    asyncio.run(main_loop(max_iterations=None))
