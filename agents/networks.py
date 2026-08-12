"""Shared neural network modules for DQN-family and policy-gradient agents."""

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


class BetPlayActorCritic(nn.Module):
    """Separate bet/play actors plus matching value heads for bet+play PG."""

    def __init__(
        self,
        *,
        shoe_size: int,
        state_size: int,
        bet_actions: int,
        play_actions: int,
    ):
        super().__init__()
        self.bet_body = nn.Sequential(
            nn.Linear(shoe_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.bet_policy = nn.Linear(64, bet_actions)
        self.bet_value = nn.Linear(64, 1)

        self.play_body = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.play_policy = nn.Linear(128, play_actions)
        self.play_value = nn.Linear(128, 1)

    def bet_logits_value(
        self, shoe: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.bet_body(shoe)
        return self.bet_policy(features), self.bet_value(features).squeeze(-1)

    def play_logits_value(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.play_body(state)
        return self.play_policy(features), self.play_value(features).squeeze(-1)
