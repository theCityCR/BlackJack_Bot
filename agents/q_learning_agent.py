import random
from collections import defaultdict
from typing import Iterable, Optional

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
    Q-learning agent for Blackjack with:
    - Hit
    - Stand
    - Double
    - Split

    Important:
    Split hands are trained using per-hand rewards, not the total round reward.
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
            Action.DOUBLE: 0.0,
            Action.SPLIT: 0.0,
        }

    def choose_action(
        self,
        state: GameState,
        available_actions: Optional[Iterable[Action]] = None,
    ) -> Action:
        legal_actions = self._normalize_available_actions(state, available_actions)

        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        return self.best_action(state, legal_actions)

    def best_action(
        self,
        state: GameState,
        available_actions: Optional[Iterable[Action]] = None,
    ) -> Action:
        legal_actions = self._normalize_available_actions(state, available_actions)
        state_key = state.as_tuple()
        action_values = self.q_table[state_key]

        best_value = max(action_values[action] for action in legal_actions)
        best_actions = [
            action for action in legal_actions
            if action_values[action] == best_value
        ]

        return random.choice(best_actions)

    def learn(
        self,
        state: GameState,
        action: Action,
        reward: float,
        next_state: Optional[GameState],
        done: bool,
        next_available_actions: Optional[Iterable[Action]] = None,
    ):
        """
        Update Q-table after one transition.

        Note:
        Epsilon is NOT decayed here. It is decayed once per episode instead.
        """
        state_key = state.as_tuple()
        current_q = self.q_table[state_key][action]

        if done:
            target = reward
        else:
            if next_state is None:
                raise ValueError("next_state cannot be None when done is False")

            legal_next_actions = self._normalize_available_actions(
                next_state,
                next_available_actions,
            )

            next_state_key = next_state.as_tuple()
            next_best_q = max(
                self.q_table[next_state_key][next_action]
                for next_action in legal_next_actions
            )

            target = reward + self.discount_factor * next_best_q

        self.q_table[state_key][action] += self.learning_rate * (
            target - current_q
        )

    def _normalize_available_actions(
        self,
        state: GameState,
        available_actions: Optional[Iterable[Action]],
    ) -> list[Action]:
        if available_actions is not None:
            legal_actions = list(available_actions)
        else:
            legal_actions = [Action.HIT, Action.STAND]

            if state.can_double:
                legal_actions.append(Action.DOUBLE)

            if state.can_split:
                legal_actions.append(Action.SPLIT)

            if state.player_value >= 21:
                legal_actions = [Action.STAND]

        if not legal_actions:
            raise ValueError("No legal actions available for this state")

        for action in legal_actions:
            if not isinstance(action, Action):
                raise ValueError(f"Invalid action in available_actions: {action}")

        return legal_actions

    def _decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    def train_one_episode(self, game: BlackjackGame) -> float:
        """
        Play one episode and learn from it.

        For split rounds, each hand's actions are trained using that hand's
        final reward, not the total round reward.
        """
        state = game.reset()

        # Dealer blackjack can end the round immediately before the player acts.
        if state is None:
            self._decay_epsilon()
            return game.round_reward

        transitions = []
        done = False
        round_reward = 0.0

        while not done:
            hand_index = game.active_hand_index
            available_actions = game.available_actions()
            action = self.choose_action(state, available_actions)

            next_state, round_reward, done = game.step(action)

            transitions.append(
                {
                    "hand_index": hand_index,
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                }
            )

            state = next_state

        hand_rewards = game.hand_rewards

        for index, transition in enumerate(transitions):
            state = transition["state"]
            action = transition["action"]
            next_state = transition["next_state"]
            hand_index = transition["hand_index"]

            hand_reward = hand_rewards[hand_index]

            is_last_transition = index == len(transitions) - 1

            next_is_different_hand = (
                not is_last_transition
                and transitions[index + 1]["hand_index"] != hand_index
            )

            terminal_for_this_hand = is_last_transition or next_is_different_hand

            if terminal_for_this_hand:
                self.learn(
                    state=state,
                    action=action,
                    reward=hand_reward,
                    next_state=None,
                    done=True,
                    next_available_actions=[],
                )
            else:
                self.learn(
                    state=state,
                    action=action,
                    reward=0.0,
                    next_state=next_state,
                    done=False,
                    next_available_actions=None,
                )

        self._decay_epsilon()
        return round_reward

    def play_episode(self, game: BlackjackGame, render=False) -> float:
        """
        Play one episode without learning.
        """
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