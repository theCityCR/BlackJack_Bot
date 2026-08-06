"""Prioritized experience replay buffer."""

from __future__ import annotations

import torch

from agents.common import Transition


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_increment: float = 0.00001,
        epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self.buffer = []
        self.priorities = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer = []
        self.priorities = []
        self.position = 0

    def add(self, transition: Transition):
        max_priority = max(self.priorities, default=1.0)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        priority_tensor = torch.tensor(self.priorities, dtype=torch.float32)
        scaled_priorities = priority_tensor ** self.alpha
        probabilities = scaled_priorities / scaled_priorities.sum()

        indices = torch.multinomial(
            probabilities,
            batch_size,
            replacement=False,
        ).tolist()

        samples = [self.buffer[index] for index in indices]

        sample_probabilities = probabilities[indices]
        weights = (len(self.buffer) * sample_probabilities) ** (-self.beta)
        weights = weights / weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for index, td_error in zip(indices, td_errors):
            priority = abs(float(td_error)) + self.epsilon
            self.priorities[index] = priority
