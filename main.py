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

# ----------------------------
# Configuration (tweak these)
# ----------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "checkpoints"
REPLAY_DIR = PROJECT_ROOT / "data" / "replay"

# Self-play / MCTS settings (kept small for laptop)
GAMES_PER_ITER = 2              # how many games to generate per iteration
MCTS_SIMS_SELFPLAY = 20         # sims per move during self-play
TEMPERATURE_THRESHOLD = 10      # move after which temperature reduced

# Training settings
TRAIN_EPOCHS = 3
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Arena / evaluation
EVAL_GAMES = 4
MCTS_SIMS_ARENA = 25            # sims per move for evaluation (slightly higher)
ACCEPTANCE_THRESHOLD = 0.55     # new model must win > this to be accepted

# Misc
SLEEP_BETWEEN_ITERS = 1.0       # seconds between iterations


# ----------------------------
# Utilities
# ----------------------------
def ensure_dirs():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)


def checkpoint_paths():
    latest = CHECKPOINT_DIR / "model_latest.pth"
    old = CHECKPOINT_DIR / "model_old.pth"
    return str(latest), str(old)


# ----------------------------
# Main training loop
# ----------------------------
def main_loop(max_iterations=None):
    """
    Run the AlphaZero-style loop:
      self-play -> add to replay -> train -> evaluate -> accept/reject
    """
    ensure_dirs()
    latest_path, old_path = checkpoint_paths()

    # Ensure there is at least a baseline model_old
    if not os.path.exists(old_path):
        if os.path.exists(latest_path):
            shutil.copyfile(latest_path, old_path)
            print(f"[main] model_old did not exist — copied model_latest -> model_old")
        else:
            # If neither exists: create an initial dummy model by training on a tiny dummy set
            print("[main] No models found. Creating an initial dummy model (small train).")
            trainer0 = Trainer(lr=LR, weight_decay=WEIGHT_DECAY,
                               batch_size=BATCH_SIZE, epochs=TRAIN_EPOCHS,
                               checkpoint_dir=str(CHECKPOINT_DIR))
            # small fake dataset
            states = np.zeros((8, 17, 10, 10), dtype=np.float32)
            policies = np.zeros((8, 10003), dtype=np.float32)
            policies[:, 0] = 1.0
            values = np.zeros((8,), dtype=np.float32)
            trainer0.train(states, policies, values, save_name="model_latest.pth")
            shutil.copyfile(latest_path, old_path)
            print("[main] Created initial model_latest.pth and copied to model_old.pth")

    # Objects that will be re-used each iteration
    replay = ReplayBuffer(save_dir=str(REPLAY_DIR))
    trainer = Trainer(lr=LR, weight_decay=WEIGHT_DECAY,
                      batch_size=BATCH_SIZE, epochs=TRAIN_EPOCHS,
                      checkpoint_dir=str(CHECKPOINT_DIR))

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

            # -------------------------
            # Self-play: generate games
            # -------------------------
            sp = SelfPlay(GeneralsEnv, trainer.net,
                          games_per_iteration=1,
                          mcts_simulations=MCTS_SIMS_SELFPLAY,
                          temperature_threshold=TEMPERATURE_THRESHOLD)

            # We call play_one_game repeatedly to save each game separately
            for g in range(GAMES_PER_ITER):
                print(f"[main] Generating game {g+1}/{GAMES_PER_ITER}")
                states, policies, values = sp.play_one_game()
                # states, policies, values are lists (or arrays). Save single game
                replay.add_game(states, policies, values)

            # -------------------------
            # Train on all replay data
            # -------------------------
            print("[main] Loading replay data to train")
            states_all, policies_all, values_all = replay.load_all()

            # Train and save to model_latest.pth
            save_name = "model_latest.pth"
            print(f"[main] Training for {TRAIN_EPOCHS} epochs ...")
            trainer.train(states_all, policies_all, values_all, save_name=save_name)
            latest_path, old_path = checkpoint_paths()

            # -------------------------
            # Evaluate: Arena between latest (A) and old (B)
            # -------------------------
            print("[main] Evaluating new model vs previous model (Arena)")
            arena = Arena(model_A_path=latest_path,
                          model_B_path=old_path,
                          games=EVAL_GAMES,
                          mcts_simulations=MCTS_SIMS_ARENA)

            win_rate = arena.run()
            print(f"[main] Arena win rate for new model: {win_rate:.2f}")

            # -------------------------
            # Accept or reject
            # -------------------------
            if win_rate > ACCEPTANCE_THRESHOLD:
                print("[main] New model accepted! Copying model_latest -> model_old")
                shutil.copyfile(latest_path, old_path)
            else:
                print("[main] New model rejected. Keeping previous model_old.")

            # small sleep so logs are readable
            time.sleep(SLEEP_BETWEEN_ITERS)

    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt — stopping training loop.")
    except Exception as e:
        print(f"[main] Exception occurred: {e}")
        raise


if __name__ == "__main__":
    # Run only a few iterations by default so your laptop doesn't overheat.
    # Set max_iterations=None to run forever.
    main_loop(max_iterations=None)