import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from model.network import GeneralsNet


class Trainer:
    def __init__(
        self,
        action_dim=10003,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=64,
        epochs=5,
        checkpoint_dir="data/checkpoints"
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
            self.scaler = torch.amp.GradScaler('cuda')
            print("[Trainer] AMP enabled for faster training")
        else:
            self.use_amp = False
            self.scaler = None

        self.net = GeneralsNet(action_dim=action_dim).to(self.device)
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.policy_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.value_loss_fn = nn.MSELoss()

    def train(self, states, policies, values, save_name="model_latest.pth"):
        dataset_size = len(states)
        print(f"\n[Trainer] Training on {dataset_size} samples")

        for epoch in range(self.epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"{'='*60}")

            indices = np.random.permutation(dataset_size)
            states = states[indices]
            policies = policies[indices]
            values = values[indices]

            batches = max(1, dataset_size // self.batch_size)
            actual_batch_size = min(self.batch_size, dataset_size)
            epoch_loss = 0.0

            for b in range(batches):
                start = b * actual_batch_size
                end = min(start + actual_batch_size, dataset_size)

                batch_states = torch.tensor(
                    states[start:end], dtype=torch.float32
                ).to(self.device)
                batch_policies = torch.tensor(
                    policies[start:end], dtype=torch.float32
                ).to(self.device)
                batch_values = torch.tensor(
                    values[start:end], dtype=torch.float32
                ).unsqueeze(1).to(self.device)

                if self.use_amp:
                    with torch.amp.autocast('cuda'):
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
                    print(f"  Batch {b:3d}/{batches} | Loss: {loss.item():.4f} | "
                          f"Policy: {loss_policy.item():.4f} | Value: {loss_value.item():.4f}")

            avg_loss = epoch_loss / batches
            print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}")

        save_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(self.net.state_dict(), save_path)
        print(f"\n[Trainer] Model saved to: {save_path}")

        return save_path

    def load_model(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint does not exist: {checkpoint_path}"
            )

        self.net.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        )
        print(f"[Trainer] Loaded model: {checkpoint_path}")

    def predict(self, state_numpy):
        return self.net.predict(state_numpy)