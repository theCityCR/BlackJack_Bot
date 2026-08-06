"""Shared neural network modules for DQN-family agents."""

from __future__ import annotations

import torch
import torch.nn as nn


class DuelingDQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return value + advantages - advantages.mean(dim=1, keepdim=True)
