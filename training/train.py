import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model.network import GeneralsNet


class Trainer:
    """
    Trainer handles:
    - Loading training data from self-play
    - Training GeneralsNet (policy + value)
    - Saving checkpoints
    """

    def __init__(self,
                 action_dim=10003,
                 lr=1e-3,
                 weight_decay=1e-4,
                 batch_size=64,
                 epochs=5,
                 checkpoint_dir="data/checkpoints"):

        self.action_dim = action_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize network
        self.net = GeneralsNet(action_dim=action_dim)
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()

    # ------------------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------------------
    def train(self, states, policies, values, save_name="model_latest.pth"):

        """
        states:   (N, 17,10,10)
        policies: (N, 10003)
        values:   (N,)
        """

        dataset_size = len(states)

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")

            # Shuffle dataset
            indices = np.random.permutation(dataset_size)
            states = states[indices]
            policies = policies[indices]
            values = values[indices]

            # Mini-batches
            batches = dataset_size // self.batch_size

            for b in range(batches):
                start = b * self.batch_size
                end = start + self.batch_size

                batch_states = torch.tensor(states[start:end], dtype=torch.float32)
                batch_policies = torch.tensor(policies[start:end], dtype=torch.float32)
                batch_values = torch.tensor(values[start:end], dtype=torch.float32).unsqueeze(1)

                # Forward pass
                logits, pred_values = self.net(batch_states)

                # Policy loss: cross entropy
                target_policy = torch.argmax(batch_policies, dim=1)
                loss_policy = self.policy_loss_fn(logits, target_policy)

                # Value loss: MSE
                loss_value = self.value_loss_fn(pred_values, batch_values)

                # Total loss
                loss = loss_policy + loss_value

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if b % 20 == 0:
                    print(f"Batch {b}/{batches} | Loss: {loss.item():.4f}")

        # Save model
        save_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(self.net.state_dict(), save_path)
        print(f"\nModel saved to: {save_path}")

        return save_path

    # ------------------------------------------------------------------------
    # LOADING MODEL CHECKPOINT
    # ------------------------------------------------------------------------
    def load_model(self, checkpoint_path):
        """
        Load model weights into the network.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

        self.net.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded model: {checkpoint_path}")

    # ------------------------------------------------------------------------
    # EVALUATION PREDICT API
    # ------------------------------------------------------------------------
    def predict(self, state_numpy):
        """
        Wrapper for network.predict().
        """
        return self.net.predict(state_numpy)