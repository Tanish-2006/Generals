"""
AlphaZero-style training loop for Generals.io.
"""

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


PROJECT_ROOT = Path(__file__).parent.resolve()
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "checkpoints"
REPLAY_DIR = PROJECT_ROOT / "data" / "replay"

GAMES_PER_ITER = 32
MCTS_SIMS_SELFPLAY = 25
TEMPERATURE_THRESHOLD = 10

TRAIN_EPOCHS = 3
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4

EVAL_GAMES = 4
MCTS_SIMS_ARENA = 25
ACCEPTANCE_THRESHOLD = 0.55

SLEEP_BETWEEN_ITERS = 1.0


def ensure_dirs():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)


def checkpoint_paths():
    latest = CHECKPOINT_DIR / "model_latest.pth"
    old = CHECKPOINT_DIR / "model_old.pth"
    return str(latest), str(old)


async def main_loop(max_iterations=None):
    ensure_dirs()
    latest_path, old_path = checkpoint_paths()

    if not os.path.exists(old_path):
        if os.path.exists(latest_path):
            shutil.copyfile(latest_path, old_path)
            print(f"[main] model_old did not exist — copied model_latest -> model_old")
        else:
            print("[main] No models found. Creating an initial dummy model (small train).")
            trainer0 = Trainer(
                lr=LR,
                weight_decay=WEIGHT_DECAY,
                batch_size=BATCH_SIZE,
                epochs=TRAIN_EPOCHS,
                checkpoint_dir=str(CHECKPOINT_DIR)
            )
            states = np.zeros((8, 17, 10, 10), dtype=np.float32)
            policies = np.zeros((8, 10003), dtype=np.float32)
            policies[:, 0] = 1.0
            values = np.zeros((8,), dtype=np.float32)
            trainer0.train(states, policies, values, save_name="model_latest.pth")
            shutil.copyfile(latest_path, old_path)
            print("[main] Created initial model_latest.pth and copied to model_old.pth")

    replay = ReplayBuffer(save_dir=str(REPLAY_DIR))
    trainer = Trainer(
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
        epochs=TRAIN_EPOCHS,
        checkpoint_dir=str(CHECKPOINT_DIR)
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
            print(f"[main] ITERATION {iteration} — self-play {GAMES_PER_ITER} games")
            print("=" * 60)

            sp = SelfPlay(
                GeneralsEnv,
                inference_server,
                games_per_iteration=GAMES_PER_ITER,
                mcts_simulations=MCTS_SIMS_SELFPLAY,
                temperature_threshold=TEMPERATURE_THRESHOLD
            )

            print(f"[main] Generating {GAMES_PER_ITER} games concurrently...")
            states, policies, values = await sp.play_iteration()
            
            replay.add_game(states, policies, values)

            print("[main] Loading replay data to train")
            states_all, policies_all, values_all = replay.load_all()

            save_name = "model_latest.pth"
            print(f"[main] Training for {TRAIN_EPOCHS} epochs ...")
            trainer.train(states_all, policies_all, values_all, save_name=save_name)
            latest_path, old_path = checkpoint_paths()

            print("[main] Evaluating new model vs previous model (Arena)")
            arena = Arena(
                model_A_path=latest_path,
                model_B_path=old_path,
                games=EVAL_GAMES,
                mcts_simulations=MCTS_SIMS_ARENA
            )

            win_rate = await arena.run()
            print(f"[main] Arena win rate for new model: {win_rate:.2f}")

            if win_rate > ACCEPTANCE_THRESHOLD:
                print("[main] New model accepted! Copying model_latest -> model_old")
                shutil.copyfile(latest_path, old_path)
            else:
                print("[main] New model rejected. Keeping previous model_old.")

            time.sleep(SLEEP_BETWEEN_ITERS)

    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt — stopping training loop.")
    except Exception as e:
        print(f"[main] Exception occurred: {e}")
        raise
    finally:
        await inference_server.stop()


if __name__ == "__main__":
    print("="*60)
    print(" OPTIMIZED GENERALS RL TRAINING v2")
    print("="*60)
    print(f" - Games per iteration: {GAMES_PER_ITER} (parallel)")
    print(f" - MCTS simulations: {MCTS_SIMS_SELFPLAY} (balanced)")
    print(f" - Network: 196 channels, 7 res blocks")
    print(f" - Intermediate rewards: Enabled")
    print("="*60)
    asyncio.run(main_loop(max_iterations=None))
