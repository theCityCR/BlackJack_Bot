import random
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from agents.common import (
    ACTION_LIST,
    ACTION_TO_INDEX,
    STATE_SIZE,
    Transition,
    encode_state,
)
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
from game import Action, BlackjackGame, GameState


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


class PrioritizedDuelingDQNAgent:
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
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.min_replay_size = min_replay_size
        self.train_updates_per_episode = train_updates_per_episode

        self.input_size = STATE_SIZE
        self.output_size = len(ACTION_LIST)

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model = DuelingDQN(self.input_size, self.output_size).to(self.device)
        self.target_model = DuelingDQN(self.input_size, self.output_size).to(self.device)

        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=replay_size,
            alpha=priority_alpha,
            beta_start=priority_beta_start,
            beta_increment=priority_beta_increment,
        )

        self.training_steps = 0
        self.use_shoe_features = True

    def encode_state(self, state: GameState) -> torch.Tensor:
        return encode_state(state, use_shoe_features=self.use_shoe_features)

    def legal_action_indices(self, available_actions) -> list[int]:
        return [ACTION_TO_INDEX[action] for action in available_actions]

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
            masked_q_values = self._mask_illegal_actions(q_values, legal_indices)
            action_index = masked_q_values.argmax().item()

        return ACTION_LIST[action_index]

    def remember(
        self,
        state: GameState,
        action: Action,
        reward: float,
        next_state: Optional[GameState],
        done: bool,
        next_available_actions,
    ):
        next_legal_action_indices = (
            []
            if next_available_actions is None
            else self.legal_action_indices(next_available_actions)
        )

        transition = Transition(
            state=self.encode_state(state),
            action_index=ACTION_TO_INDEX[action],
            reward=reward,
            next_state=None if next_state is None else self.encode_state(next_state),
            done=done,
            next_legal_action_indices=next_legal_action_indices,
        )

        self.replay_buffer.add(transition)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch, indices, weights = self.replay_buffer.sample(self.batch_size)

        states = torch.stack([transition.state for transition in batch]).to(self.device)

        action_indices = torch.tensor(
            [transition.action_index for transition in batch],
            dtype=torch.long,
            device=self.device,
        )

        rewards = torch.tensor(
            [transition.reward for transition in batch],
            dtype=torch.float32,
            device=self.device,
        )

        weights = weights.to(self.device)

        current_q_values = self.model(states)

        current_q = current_q_values.gather(
            1,
            action_indices.unsqueeze(1),
        ).squeeze(1)

        targets = rewards.clone()

        non_done_indices = [
            index
            for index, transition in enumerate(batch)
            if not transition.done
        ]

        if non_done_indices:
            next_states = torch.stack([
                batch[index].next_state
                for index in non_done_indices
            ]).to(self.device)

            with torch.no_grad():
                next_q_main = self.model(next_states)

                masked_next_q_main = torch.full_like(
                    next_q_main,
                    float("-inf"),
                )

                for row, batch_index in enumerate(non_done_indices):
                    legal_indices = batch[batch_index].next_legal_action_indices
                    masked_next_q_main[row, legal_indices] = (
                        next_q_main[row, legal_indices]
                    )

                best_next_action_indices = masked_next_q_main.argmax(dim=1)

                next_q_target = self.target_model(next_states)

                next_q = next_q_target.gather(
                    1,
                    best_next_action_indices.unsqueeze(1),
                ).squeeze(1)

            non_done_tensor = torch.tensor(
                non_done_indices,
                dtype=torch.long,
                device=self.device,
            )

            targets[non_done_tensor] += self.discount_factor * next_q

        td_errors = targets.detach() - current_q.detach()

        elementwise_loss = nn.functional.smooth_l1_loss(
            current_q,
            targets,
            reduction="none",
        )

        loss = (weights * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, td_errors.cpu())

        self.training_steps += 1

        if self.training_steps % self.target_update_interval == 0:
            self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

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

            transitions.append({
                "hand_index": hand_index,
                "state": state,
                "action": action,
                "next_state": next_state,
                "next_available_actions": next_available_actions,
            })

            state = next_state

        hand_rewards = game.hand_rewards

        for index, transition in enumerate(transitions):
            hand_index = transition["hand_index"]
            hand_reward = hand_rewards[hand_index]

            is_last_transition = index == len(transitions) - 1

            next_transition_is_new_hand = (
                not is_last_transition
                and transitions[index + 1]["hand_index"] != hand_index
            )

            terminal_for_this_hand = (
                is_last_transition
                or next_transition_is_new_hand
            )

            if terminal_for_this_hand:
                self.remember(
                    transition["state"],
                    transition["action"],
                    hand_reward,
                    None,
                    True,
                    None,
                )
            else:
                self.remember(
                    transition["state"],
                    transition["action"],
                    0.0,
                    transition["next_state"],
                    False,
                    transition["next_available_actions"],
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

    def _mask_illegal_actions(
        self,
        q_values: torch.Tensor,
        legal_indices: list[int],
    ) -> torch.Tensor:
        masked_q_values = torch.full_like(q_values, float("-inf"))
        masked_q_values[legal_indices] = q_values[legal_indices]
        return masked_q_values

# Backward-compatible alias for older imports/tests.
DuelingDQNAgent = PrioritizedDuelingDQNAgent
