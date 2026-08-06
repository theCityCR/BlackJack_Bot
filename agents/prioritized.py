"""Dueling Double DQN with prioritized experience replay."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from agents.common import Transition, resolve_torch_device
from agents.networks import DuelingDQN
from agents.neural_base import NeuralQAgent, init_neural_hyperparams
from agents.replay import PrioritizedReplayBuffer
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


class PrioritizedDuelingDQNAgent(NeuralQAgent):
    def __init__(
        self,
        learning_rate: float = NEURAL_LEARNING_RATE,
        discount_factor: float = NEURAL_DISCOUNT_FACTOR,
        epsilon: float = NEURAL_EPSILON_START,
        epsilon_min: float = NEURAL_EPSILON_MIN,
        epsilon_decay: float = NEURAL_EPSILON_DECAY,
        replay_size: int = NEURAL_REPLAY_SIZE,
        batch_size: int = NEURAL_BATCH_SIZE,
        target_update_interval: int = NEURAL_TARGET_UPDATE_INTERVAL,
        min_replay_size: int = NEURAL_MIN_REPLAY_SIZE,
        train_updates_per_episode: int = NEURAL_TRAIN_UPDATES_PER_EPISODE,
        priority_alpha: float = 0.6,
        priority_beta_start: float = 0.4,
        priority_beta_increment: float = 0.00001,
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
        self.model = DuelingDQN(self.input_size, self.output_size).to(self.device)
        self.target_model = DuelingDQN(self.input_size, self.output_size).to(
            self.device
        )
        self.update_target_model()
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=replay_size,
            alpha=priority_alpha,
            beta_start=priority_beta_start,
            beta_increment=priority_beta_increment,
        )
        self.use_shoe_features = True

    def _store_transition(self, transition: Transition) -> None:
        self.replay_buffer.add(transition)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch, indices, weights = self.replay_buffer.sample(self.batch_size)
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
        weights = weights.to(self.device)

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
                next_q_main = self.model(next_states)
                masked = torch.full_like(next_q_main, float("-inf"))
                for row, batch_index in enumerate(non_done_indices):
                    legal = batch[batch_index].next_legal_action_indices
                    masked[row, legal] = next_q_main[row, legal]
                best_next = masked.argmax(dim=1)
                next_q = self.target_model(next_states).gather(
                    1, best_next.unsqueeze(1)
                ).squeeze(1)

            non_done_tensor = torch.tensor(
                non_done_indices, dtype=torch.long, device=self.device
            )
            targets[non_done_tensor] += self.discount_factor * next_q

        td_errors = targets.detach() - current_q.detach()
        elementwise_loss = nn.functional.smooth_l1_loss(
            current_q, targets, reduction="none"
        )
        loss = (weights * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, td_errors.cpu())
        self.training_steps += 1
        self.maybe_sync_target()
