"""Vanilla DQN Blackjack agent (target-net max + MSE)."""

from __future__ import annotations

import random
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from agents.common import ACTION_LIST, resolve_torch_device
from agents.neural_base import NeuralQAgent, init_neural_hyperparams
from config import (
    NEURAL_BATCH_SIZE,
    NEURAL_DISCOUNT_FACTOR,
    NEURAL_EPSILON_DECAY,
    NEURAL_EPSILON_MIN,
    NEURAL_EPSILON_START,
    NEURAL_LEARNING_RATE,
    NEURAL_MIN_REPLAY_SIZE,
    NEURAL_REPLAY_SIZE,
    NEURAL_TARGET_UPDATE_INTERVAL,
    NEURAL_TRAIN_UPDATES_PER_EPISODE,
)


class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.net(x)


class DeepQLearningAgent(NeuralQAgent):
    def __init__(
        self,
        learning_rate=NEURAL_LEARNING_RATE,
        discount_factor=NEURAL_DISCOUNT_FACTOR,
        epsilon=NEURAL_EPSILON_START,
        epsilon_min=NEURAL_EPSILON_MIN,
        epsilon_decay=NEURAL_EPSILON_DECAY,
        replay_size=NEURAL_REPLAY_SIZE,
        batch_size=NEURAL_BATCH_SIZE,
        target_update_interval=NEURAL_TARGET_UPDATE_INTERVAL,
        min_replay_size=NEURAL_MIN_REPLAY_SIZE,
        train_updates_per_episode=NEURAL_TRAIN_UPDATES_PER_EPISODE,
        device: Optional[str] = None,
    ):
        init_neural_hyperparams(
            self,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            batch_size=batch_size,
            target_update_interval=target_update_interval,
            min_replay_size=min_replay_size,
            train_updates_per_episode=train_updates_per_episode,
        )
        self.device = resolve_torch_device(device)
        self.model = DQN(self.input_size, self.output_size).to(self.device)
        self.target_model = DQN(self.input_size, self.output_size).to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = deque(maxlen=replay_size)
        self.use_shoe_features = True

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states = torch.stack([t.state for t in batch]).to(self.device)
        action_indices = torch.tensor(
            [t.action_index for t in batch],
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.tensor(
            [t.reward for t in batch],
            dtype=torch.float32,
            device=self.device,
        )

        current_q = self.model(states).gather(
            1, action_indices.unsqueeze(1)
        ).squeeze(1)
        targets = rewards.clone()

        non_done_indices = [i for i, t in enumerate(batch) if not t.done]
        if non_done_indices:
            next_states = torch.stack(
                [batch[i].next_state for i in non_done_indices]
            ).to(self.device)

            with torch.no_grad():
                next_q_values = self.target_model(next_states)
                masked = torch.full_like(next_q_values, float("-inf"))
                for row, batch_index in enumerate(non_done_indices):
                    legal = batch[batch_index].next_legal_action_indices
                    masked[row, legal] = next_q_values[row, legal]
                max_next_q = masked.max(dim=1).values

            non_done_tensor = torch.tensor(
                non_done_indices, dtype=torch.long, device=self.device
            )
            targets[non_done_tensor] += self.discount_factor * max_next_q

        loss = self.loss_fn(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_steps += 1
        self.maybe_sync_target()
