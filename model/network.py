import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out, inplace=True)


class GeneralsNet(nn.Module):
    def __init__(self, action_dim=10004, channels=196, num_res_blocks=7):
        super().__init__()

        self.action_dim = action_dim

        self.conv = nn.Conv2d(9, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)

        self.res_layers = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 10 * 10, action_dim)

        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 10 * 10, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)), inplace=True)

        for layer in self.res_layers:
            out = layer(out)

        p = F.relu(self.policy_bn(self.policy_conv(out)), inplace=True)
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(out)), inplace=True)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v), inplace=True)
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    def predict(self, state_numpy):
        self.eval()

        device = next(self.parameters()).device

        state = torch.tensor(state_numpy, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            policy_logits, value = self.forward(state)

        policy_logits = policy_logits.squeeze(0).cpu().numpy()
        value_scalar = value.item()

        return policy_logits, value_scalar
