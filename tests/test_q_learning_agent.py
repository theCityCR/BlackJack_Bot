import pytest

from agents.q_learning_agent import QLearningAgent
from game import Action, GameState


def test_q_table_starts_with_zero_values():
    agent = QLearningAgent()

    state = GameState(
        player_value=15,
        dealer_upcard=10,
        usable_ace=False,
    )

    values = agent.q_table[state.as_tuple()]

    assert values[Action.HIT] == 0.0
    assert values[Action.STAND] == 0.0


def test_best_action_returns_action_with_highest_q_value():
    agent = QLearningAgent()

    state = GameState(
        player_value=18,
        dealer_upcard=6,
        usable_ace=False,
    )

    state_key = state.as_tuple()
    agent.q_table[state_key][Action.HIT] = -1.0
    agent.q_table[state_key][Action.STAND] = 1.0

    assert agent.best_action(state) == Action.STAND


def test_choose_action_with_epsilon_zero_uses_best_action():
    agent = QLearningAgent(epsilon=0.0)

    state = GameState(
        player_value=18,
        dealer_upcard=6,
        usable_ace=False,
    )

    state_key = state.as_tuple()
    agent.q_table[state_key][Action.HIT] = -1.0
    agent.q_table[state_key][Action.STAND] = 1.0

    assert agent.choose_action(state) == Action.STAND


def test_learn_terminal_state_updates_toward_reward():
    agent = QLearningAgent(
        learning_rate=0.5,
        discount_factor=1.0,
        epsilon=0.0,
    )

    state = GameState(
        player_value=20,
        dealer_upcard=10,
        usable_ace=False,
    )

    agent.learn(
        state=state,
        action=Action.STAND,
        reward=1,
        next_state=None,
        done=True,
    )

    assert agent.q_table[state.as_tuple()][Action.STAND] == pytest.approx(0.5)


def test_learn_non_terminal_state_uses_next_state_best_q_value():
    agent = QLearningAgent(
        learning_rate=0.5,
        discount_factor=1.0,
        epsilon=0.0,
    )

    state = GameState(
        player_value=12,
        dealer_upcard=10,
        usable_ace=False,
    )

    next_state = GameState(
        player_value=18,
        dealer_upcard=10,
        usable_ace=False,
    )

    agent.q_table[next_state.as_tuple()][Action.HIT] = -1.0
    agent.q_table[next_state.as_tuple()][Action.STAND] = 0.8

    agent.learn(
        state=state,
        action=Action.HIT,
        reward=0,
        next_state=next_state,
        done=False,
    )

    assert agent.q_table[state.as_tuple()][Action.HIT] == pytest.approx(0.4)


def test_epsilon_decays_after_learning():
    agent = QLearningAgent(
        learning_rate=0.5,
        discount_factor=1.0,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.5,
    )

    state = GameState(
        player_value=20,
        dealer_upcard=10,
        usable_ace=False,
    )

    agent.learn(
        state=state,
        action=Action.STAND,
        reward=1,
        next_state=None,
        done=True,
    )

    assert agent.epsilon == pytest.approx(0.5)


def test_epsilon_does_not_decay_below_minimum():
    agent = QLearningAgent(
        learning_rate=0.5,
        discount_factor=1.0,
        epsilon=0.2,
        epsilon_min=0.1,
        epsilon_decay=0.01,
    )

    state = GameState(
        player_value=20,
        dealer_upcard=10,
        usable_ace=False,
    )

    agent.learn(
        state=state,
        action=Action.STAND,
        reward=1,
        next_state=None,
        done=True,
    )

    assert agent.epsilon == pytest.approx(0.1)