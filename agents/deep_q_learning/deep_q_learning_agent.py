import random
from collections import deque

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


class DeepQLearningAgent:
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

        self.model = DQN(self.input_size, self.output_size)
        self.target_model = DQN(self.input_size, self.output_size)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = deque(maxlen=replay_size)
        self.training_steps = 0

    def encode_state(self, state: GameState) -> torch.Tensor:
        return encode_state(state)

    def legal_action_indices(self, available_actions):
        return [ACTION_TO_INDEX[action] for action in available_actions]

    def choose_action(self, state: GameState, available_actions) -> Action:
        legal_indices = self.legal_action_indices(available_actions)

        if random.random() < self.epsilon:
            return ACTION_LIST[random.choice(legal_indices)]

        return self.best_action(state, available_actions)

    def best_action(self, state: GameState, available_actions) -> Action:
        legal_indices = self.legal_action_indices(available_actions)

        with torch.no_grad():
            state_tensor = self.encode_state(state).unsqueeze(0)
            q_values = self.model(state_tensor)[0]

            masked_q_values = torch.full_like(q_values, float("-inf"))
            masked_q_values[legal_indices] = q_values[legal_indices]

            action_index = torch.argmax(masked_q_values).item()

        return ACTION_LIST[action_index]

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        next_available_actions,
    ):
        next_legal_indices = (
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
            next_legal_action_indices=next_legal_indices,
        )

        self.replay_buffer.append(transition)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)

        states = torch.stack([t.state for t in batch])
        action_indices = torch.tensor([t.action_index for t in batch])
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)

        current_q_values = self.model(states)
        current_q = current_q_values.gather(
            1,
            action_indices.unsqueeze(1),
        ).squeeze(1)

        targets = rewards.clone()

        non_done_indices = [i for i, t in enumerate(batch) if not t.done]

        if non_done_indices:
            next_states = torch.stack([
                batch[i].next_state for i in non_done_indices
            ])

            with torch.no_grad():
                next_q_values = self.target_model(next_states)

                masked_next_q_values = torch.full_like(
                    next_q_values,
                    float("-inf"),
                )

                for row, batch_index in enumerate(non_done_indices):
                    legal_indices = batch[batch_index].next_legal_action_indices
                    masked_next_q_values[row, legal_indices] = next_q_values[
                        row,
                        legal_indices,
                    ]

                max_next_q = masked_next_q_values.max(dim=1).values

            targets[non_done_indices] += self.discount_factor * max_next_q

        loss = self.loss_fn(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_steps += 1

        if self.training_steps % self.target_update_interval == 0:
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

            transitions.append(
                {
                    "hand_index": hand_index,
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                    "done": done,
                    "next_available_actions": next_available_actions,
                }
            )

            state = next_state

        hand_rewards = game.hand_rewards

        for i, transition in enumerate(transitions):
            hand_index = transition["hand_index"]
            hand_reward = hand_rewards[hand_index]

            is_last = i == len(transitions) - 1
            next_is_different_hand = (
                not is_last
                and transitions[i + 1]["hand_index"] != hand_index
            )

            terminal_for_this_hand = is_last or next_is_different_hand

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

    def play_episode(self, game: BlackjackGame, render=False) -> float:
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