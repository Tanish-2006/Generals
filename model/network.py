import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class GeneralsNet(nn.Module):
    """
    AlphaZero-style CNN for Generals.
    Input  : (17, 10, 10)
    Output : policy logits (10003), value scalar (-1..1)
    """

    def __init__(self, action_dim=10003, channels=128, num_res_blocks=5):
        super().__init__()

        self.action_dim = action_dim

        # ---- Initial convolution ----
        self.conv = nn.Conv2d(17, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)

        # ---- Residual blocks ----
        self.res_layers = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_res_blocks)
        ])

        # ---- Policy head ----
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 10 * 10, action_dim)

        # ---- Value head ----
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 10 * 10, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """
        x: tensor of shape (batch, 17, 10, 10)
        """
        out = F.relu(self.bn(self.conv(x)))

        for layer in self.res_layers:
            out = layer(out)

        # ---- Policy ----
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # ---- Value ----
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    # ---------------------------------------------------
    # METHOD USED BY MCTS
    # ---------------------------------------------------
    def predict(self, state_numpy):
        """
        state_numpy: (17,10,10) numpy array from env.encode_state()

        Returns:
            policy_logits (np array shape (ACTION_DIM,))
            value_scalar (float)
        """
        self.eval()

        # Convert to batch=1 tensor
        state = torch.tensor(state_numpy, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            policy_logits, value = self.forward(state)

        # Convert to numpy
        policy_logits = policy_logits.squeeze(0).cpu().numpy()
        value_scalar = float(value.squeeze(0).cpu().numpy())

        return policy_logits, value_scalar