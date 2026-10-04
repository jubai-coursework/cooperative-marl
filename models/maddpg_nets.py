import numpy as np
import torch
import torch.nn as nn


class DDPGActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ):
        super().__init__()
        self.action_low = action_low
        self.action_high = action_high
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.uniform_(
                layer.weight,
                -1 / np.sqrt(layer.weight.shape[0]),
                1 / np.sqrt(layer.weight.shape[0]),
            )
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return self.action_low + (self.action_high - self.action_low) * (x + 1) / 2


class MADDPGCritic(nn.Module):
    def __init__(self, global_obs_dim: int, global_action_dim: int, hidden_dim: int = 128):
        super().__init__()
        input_dim = global_obs_dim + global_action_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.uniform_(
                layer.weight,
                -1 / np.sqrt(layer.weight.shape[0]),
                1 / np.sqrt(layer.weight.shape[0]),
            )
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, global_obs, global_actions):
        x = torch.cat([global_obs, global_actions], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
