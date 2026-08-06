"""Shared base class for neural Q-learning Blackjack agents."""

from __future__ import annotations

import random
from typing import Optional

import torch

from agents.common import (
    ACTION_LIST,
    ACTION_TO_INDEX,
    STATE_SIZE,
    Transition,
    encode_state,
)
from agents.episode import attribute_hand_transitions
from game import Action, BlackjackGame, GameState


class NeuralQAgent:
    """ε-greedy DQN agent with legal-action masking and split-aware episodes.

    Subclasses must set ``model``, ``target_model``, ``optimizer``,
    ``replay_buffer``, ``loss_fn`` (optional), and implement ``train_step``.
    Override ``_store_transition`` when the buffer uses ``add`` instead of
    ``append``.
    """

    # Populated by subclasses in __init__
    discount_factor: float
    epsilon: float
    epsilon_min: float
    epsilon_decay: float
    batch_size: int
    target_update_interval: int
    min_replay_size: int
    train_updates_per_episode: int
    input_size: int
    output_size: int
    device: torch.device
    model: torch.nn.Module
    target_model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    replay_buffer: object
    training_steps: int
    use_shoe_features: bool

    def encode_state(self, state: GameState) -> torch.Tensor:
        return encode_state(state, use_shoe_features=self.use_shoe_features)

    def legal_action_indices(self, available_actions) -> list[int]:
        return [ACTION_TO_INDEX[action] for action in available_actions]

    @staticmethod
    def mask_illegal_actions(
        q_values: torch.Tensor,
        legal_indices: list[int],
    ) -> torch.Tensor:
        masked = torch.full_like(q_values, float("-inf"))
        masked[legal_indices] = q_values[legal_indices]
        return masked

    def choose_action(self, state: GameState, available_actions) -> Action:
        legal_indices = self.legal_action_indices(available_actions)

        if random.random() < self.epsilon:
            return ACTION_LIST[random.choice(legal_indices)]

        return self.best_action(state, available_actions)

    def best_action(self, state: GameState, available_actions) -> Action:
        legal_indices = self.legal_action_indices(available_actions)

        with torch.no_grad():
            state_tensor = self.encode_state(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)[0]
            masked = self.mask_illegal_actions(q_values, legal_indices)
            action_index = masked.argmax().item()

        return ACTION_LIST[action_index]

    def _store_transition(self, transition: Transition) -> None:
        self.replay_buffer.append(transition)

    def remember(
        self,
        state: GameState,
        action: Action,
        reward: float,
        next_state: Optional[GameState],
        done: bool,
        next_available_actions,
    ):
        next_legal = (
            []
            if next_available_actions is None
            else self.legal_action_indices(next_available_actions)
        )
        self._store_transition(
            Transition(
                state=self.encode_state(state),
                action_index=ACTION_TO_INDEX[action],
                reward=reward,
                next_state=(
                    None if next_state is None else self.encode_state(next_state)
                ),
                done=done,
                next_legal_action_indices=next_legal,
            )
        )

    def update_target_model(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())

    def maybe_sync_target(self) -> None:
        if self.training_steps % self.target_update_interval == 0:
            self.update_target_model()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_step(self) -> None:
        raise NotImplementedError

    def train_one_episode(self, game: BlackjackGame) -> float:
        state = game.reset()

        if state is None:
            self.decay_epsilon()
            return game.round_reward

        transitions = []
        done = False
        round_reward = 0.0

        while not done:
            hand_index = game.active_hand_index
            available_actions = game.available_actions()
            action = self.choose_action(state, available_actions)

            next_state, round_reward, done = game.step(action)
            next_available_actions = None if done else game.available_actions()

            transitions.append(
                {
                    "hand_index": hand_index,
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                    "next_available_actions": next_available_actions,
                }
            )
            state = next_state

        attribute_hand_transitions(
            transitions,
            game.hand_rewards,
            self.remember,
        )

        if len(self.replay_buffer) >= self.min_replay_size:
            for _ in range(self.train_updates_per_episode):
                self.train_step()

        self.decay_epsilon()
        return round_reward

    def play_episode(self, game: BlackjackGame, render: bool = False) -> float:
        state = game.reset()

        if state is None:
            if render:
                game.render()
            return game.round_reward

        done = False
        reward = 0.0

        if render:
            game.render()

        while not done:
            available_actions = game.available_actions()
            action = self.best_action(state, available_actions)
            next_state, reward, done = game.step(action)

            if render:
                print(f"Action: {action.name}")
                game.render()
                print()

            state = next_state

        return reward


def init_neural_hyperparams(
    agent: NeuralQAgent,
    *,
    discount_factor: float,
    epsilon: float,
    epsilon_min: float,
    epsilon_decay: float,
    batch_size: int,
    target_update_interval: int,
    min_replay_size: int,
    train_updates_per_episode: int,
    input_size: int = STATE_SIZE,
) -> None:
    """Assign shared hyperparameter fields used by all neural agents."""
    agent.discount_factor = discount_factor
    agent.epsilon = epsilon
    agent.epsilon_min = epsilon_min
    agent.epsilon_decay = epsilon_decay
    agent.batch_size = batch_size
    agent.target_update_interval = target_update_interval
    agent.min_replay_size = min_replay_size
    agent.train_updates_per_episode = train_updates_per_episode
    agent.input_size = input_size
    agent.output_size = len(ACTION_LIST)
    agent.training_steps = 0
