import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model.network import GeneralsNet
from config import TRAINING


class GenGameDataset(Dataset):
    def __init__(self, states, policies, values):
        self.states = states
        self.policies = policies
        self.values = values

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (self.states[idx], self.policies[idx], self.values[idx])


class Trainer:
    def __init__(
        self,
        action_dim=10004,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=64,
        epochs=5,
        checkpoint_dir="data/checkpoints",
    ):
        self.action_dim = action_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Using device: {self.device}")

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.use_amp = True
            self.scaler = torch.amp.GradScaler("cuda")
            print("[Trainer] AMP enabled for faster training")
        else:
            self.use_amp = False
            self.scaler = None

        self.net = GeneralsNet(action_dim=action_dim).to(self.device)

        # JIT Compilation (PyTorch 2.0+)
        try:
            if os.name != "nt":
                self.net = torch.compile(self.net)
                print("[Trainer] JIT compilation enabled (torch.compile)")
            else:
                print(
                    "[Trainer] JIT compilation skipped on Windows (Triton not supported)"
                )
        except Exception as e:
            print(f"[Trainer] JIT compilation skipped: {e}")

        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.policy_loss_fn = nn.KLDivLoss(reduction="batchmean")
        self.value_loss_fn = nn.MSELoss()

    def train(self, states, policies, values, save_name="model_latest.pth"):
        dataset_size = len(states)
        print(f"\n[Trainer] Training on {dataset_size} samples")

        dataset = GenGameDataset(states, policies, values)

        num_workers = TRAINING.num_workers
        persistent_workers = num_workers > 0

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=persistent_workers,
            drop_last=True,  # Avoid small straggler batches that might disturb Batch Norm
        )

        batches = len(dataloader)

        for epoch in range(self.epochs):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"{'=' * 60}")

            epoch_loss = 0.0

            for b, (batch_states, batch_policies, batch_values) in enumerate(
                dataloader
            ):
                batch_states = batch_states.to(self.device, non_blocking=True)
                batch_policies = batch_policies.to(self.device, non_blocking=True)
                batch_values = batch_values.unsqueeze(1).to(
                    self.device, non_blocking=True
                )

                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        logits, pred_values = self.net(batch_states)
                        log_probs = F.log_softmax(logits, dim=1)
                        loss_policy = self.policy_loss_fn(log_probs, batch_policies)
                        loss_value = self.value_loss_fn(pred_values, batch_values)
                        loss = loss_policy + loss_value

                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits, pred_values = self.net(batch_states)
                    log_probs = F.log_softmax(logits, dim=1)
                    loss_policy = self.policy_loss_fn(log_probs, batch_policies)
                    loss_value = self.value_loss_fn(pred_values, batch_values)
                    loss = loss_policy + loss_value

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                epoch_loss += loss.item()

                if b % 20 == 0:
                    print(
                        f"  Batch {b:3d}/{batches} | Loss: {loss.item():.4f} | "
                        f"Policy: {loss_policy.item():.4f} | Value: {loss_value.item():.4f}"
                    )

            avg_loss = epoch_loss / max(1, batches)
            print(f"\nEpoch {epoch + 1} Average Loss: {avg_loss:.4f}")

        save_path = os.path.join(self.checkpoint_dir, save_name)
        # Unwrap model if compiled
        model_to_save = (
            self.net._orig_mod if hasattr(self.net, "_orig_mod") else self.net
        )
        torch.save(model_to_save.state_dict(), save_path)
        print(f"\n[Trainer] Model saved to: {save_path}")

        return save_path

    def load_model(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

        state_dict = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        # Handle compiled model loading
        if hasattr(self.net, "_orig_mod"):
            self.net._orig_mod.load_state_dict(state_dict)
        else:
            self.net.load_state_dict(state_dict)

        print(f"[Trainer] Loaded model: {checkpoint_path}")

    def predict(self, state_numpy):
        model = self.net._orig_mod if hasattr(self.net, "_orig_mod") else self.net
        return model.predict(state_numpy)
