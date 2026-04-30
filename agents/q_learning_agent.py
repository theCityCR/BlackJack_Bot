import random
from collections import defaultdict

from config import (
    LEARNING_RATE,
    DISCOUNT_FACTOR,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY,
)

from game import Action, BlackjackGame, GameState


class QLearningAgent:
    """
    Q-learning agent for simple Blackjack.

    State:
        (player_value, dealer_upcard, usable_ace)

    Actions:
        HIT or STAND
    """

    def __init__(
        self,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = defaultdict(self._default_action_values)

    def _default_action_values(self):
        return {
            Action.HIT: 0.0,
            Action.STAND: 0.0,
        }

    def choose_action(self, state: GameState) -> Action:
        """
        Choose an action using epsilon-greedy exploration.
        """
        if random.random() < self.epsilon:
            return random.choice([Action.HIT, Action.STAND])

        return self.best_action(state)

    def best_action(self, state: GameState) -> Action:
        """
        Choose the action with the highest Q-value.
        """
        state_key = state.as_tuple()
        action_values = self.q_table[state_key]

        return max(action_values, key=action_values.get)

    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-table after one step.
        """
        state_key = state.as_tuple()

        current_q = self.q_table[state_key][action]

        if done:
            target = reward
        else:
            next_state_key = next_state.as_tuple()
            next_best_q = max(self.q_table[next_state_key].values())
            target = reward + self.discount_factor * next_best_q

        self.q_table[state_key][action] += self.learning_rate * (target - current_q)

        self._decay_epsilon()

    def _decay_epsilon(self):
        """
        Slowly reduce exploration.
        """
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    def train_one_episode(self, game: BlackjackGame) -> int:
        """
        Play one episode and learn from it.
        """
        state = game.reset()
        done = False

        while not done:
            action = self.choose_action(state)
            next_state, reward, done = game.step(action)

            self.learn(state, action, reward, next_state, done)

            state = next_state

        return reward

    def play_episode(self, game: BlackjackGame, render=False) -> int:
        """
        Play one episode without learning.
        """
        state = game.reset()
        done = False

        if render:
            game.render()

        while not done:
            action = self.best_action(state)
            next_state, reward, done = game.step(action)

            if render:
                print(f"Action: {action.name}")
                game.render()
                print()

            state = next_state

        return reward