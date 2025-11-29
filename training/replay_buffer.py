import os
import numpy as np


class ReplayBuffer:
    def __init__(self, save_dir="data/replay"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    # -----------------------------------------------
    def add_game(self, states, policies, values):
        states = np.array(states, dtype=np.float32)
        policies = np.array(policies, dtype=np.float32)
        values = np.array(values, dtype=np.float32)

        batch_id = self._next_batch_id()
        filename = f"batch_{batch_id:04d}.npz"
        path = os.path.join(self.save_dir, filename)

        np.savez_compressed(
            path,
            states=states,
            policies=policies,
            values=values
        )
        print(f"[ReplayBuffer] Saved → {path}")
        return path

    # -----------------------------------------------
    def load_all(self):
        files = sorted(
            f for f in os.listdir(self.save_dir)
            if f.endswith(".npz")
        )

        if not files:
            raise ValueError("Replay buffer is empty!")

        all_states = []
        all_policies = []
        all_values = []

        for f in files:
            data = np.load(os.path.join(self.save_dir, f))
            all_states.append(data["states"])
            all_policies.append(data["policies"])
            all_values.append(data["values"])

        states = np.concatenate(all_states)
        policies = np.concatenate(all_policies)
        values = np.concatenate(all_values)

        print(f"[ReplayBuffer] Loaded {len(files)} batches")
        return states, policies, values

    # -----------------------------------------------
    def _next_batch_id(self):
        files = [
            f for f in os.listdir(self.save_dir)
            if f.endswith(".npz")
        ]
        if not files:
            return 1

        ids = []
        for f in files:
            num = int(f.replace("batch_", "").replace(".npz", ""))
            ids.append(num)

        return max(ids) + 1