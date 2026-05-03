import pytest

from agents.deprecated.q_learning_simple.q_learning_agent import QLearningAgent
from game import Action, GameState


def make_state(
    player_value=15,
    dealer_upcard=10,
    usable_ace=False,
    can_double=False,
    can_split=False,
    is_split_hand=False,
    active_hand_index=0,
    num_hands=1,
):
    return GameState(
        player_value=player_value,
        dealer_upcard=dealer_upcard,
        usable_ace=usable_ace,
        can_double=can_double,
        can_split=can_split,
        is_split_hand=is_split_hand,
        active_hand_index=active_hand_index,
        num_hands=num_hands,
    )


def test_q_table_starts_with_zero_values_for_all_actions():
    agent = QLearningAgent()
    state = make_state()

    values = agent.q_table[state.as_tuple()]

    assert values[Action.HIT] == 0.0
    assert values[Action.STAND] == 0.0
    assert values[Action.DOUBLE] == 0.0
    assert values[Action.SPLIT] == 0.0


def test_best_action_returns_highest_q_value_among_legal_actions():
    agent = QLearningAgent()
    state = make_state(player_value=11, dealer_upcard=6, can_double=True)

    state_key = state.as_tuple()
    agent.q_table[state_key][Action.HIT] = 0.1
    agent.q_table[state_key][Action.STAND] = -0.2
    agent.q_table[state_key][Action.DOUBLE] = 1.0
    agent.q_table[state_key][Action.SPLIT] = 5.0

    action = agent.best_action(
        state,
        available_actions=[Action.HIT, Action.STAND, Action.DOUBLE],
    )

    assert action == Action.DOUBLE


def test_best_action_masks_illegal_actions_even_if_illegal_action_has_highest_q():
    agent = QLearningAgent()
    state = make_state(player_value=16, dealer_upcard=10)

    state_key = state.as_tuple()
    agent.q_table[state_key][Action.HIT] = 0.2
    agent.q_table[state_key][Action.STAND] = -0.1
    agent.q_table[state_key][Action.DOUBLE] = 100.0
    agent.q_table[state_key][Action.SPLIT] = 100.0

    action = agent.best_action(
        state,
        available_actions=[Action.HIT, Action.STAND],
    )

    assert action == Action.HIT


def test_choose_action_with_epsilon_zero_uses_best_legal_action():
    agent = QLearningAgent(epsilon=0.0)
    state = make_state(player_value=16, dealer_upcard=10)

    state_key = state.as_tuple()
    agent.q_table[state_key][Action.HIT] = -1.0
    agent.q_table[state_key][Action.STAND] = 1.0
    agent.q_table[state_key][Action.DOUBLE] = 10.0

    action = agent.choose_action(
        state,
        available_actions=[Action.HIT, Action.STAND],
    )

    assert action == Action.STAND


def test_choose_action_with_epsilon_one_only_explores_legal_actions():
    agent = QLearningAgent(epsilon=1.0)
    state = make_state(player_value=8, dealer_upcard=6, can_split=True)

    legal_actions = [Action.HIT, Action.STAND, Action.SPLIT]

    for _ in range(100):
        action = agent.choose_action(state, available_actions=legal_actions)
        assert action in legal_actions


def test_choose_action_can_select_double_when_double_is_legal():
    agent = QLearningAgent(epsilon=0.0)
    state = make_state(player_value=11, dealer_upcard=6, can_double=True)

    state_key = state.as_tuple()
    agent.q_table[state_key][Action.HIT] = 0.0
    agent.q_table[state_key][Action.STAND] = 0.0
    agent.q_table[state_key][Action.DOUBLE] = 2.0

    action = agent.choose_action(
        state,
        available_actions=[Action.HIT, Action.STAND, Action.DOUBLE],
    )

    assert action == Action.DOUBLE


def test_choose_action_can_select_split_when_split_is_legal():
    agent = QLearningAgent(epsilon=0.0)
    state = make_state(player_value=16, dealer_upcard=10, can_split=True)

    state_key = state.as_tuple()
    agent.q_table[state_key][Action.HIT] = 0.0
    agent.q_table[state_key][Action.STAND] = 0.0
    agent.q_table[state_key][Action.SPLIT] = 2.0

    action = agent.choose_action(
        state,
        available_actions=[Action.HIT, Action.STAND, Action.SPLIT],
    )

    assert action == Action.SPLIT


def test_state_value_21_defaults_to_stand_when_available_actions_omitted():
    agent = QLearningAgent(epsilon=0.0)
    state = make_state(
        player_value=21,
        dealer_upcard=6,
        can_double=True,
        can_split=True,
    )

    state_key = state.as_tuple()
    agent.q_table[state_key][Action.HIT] = 10.0
    agent.q_table[state_key][Action.DOUBLE] = 10.0
    agent.q_table[state_key][Action.SPLIT] = 10.0
    agent.q_table[state_key][Action.STAND] = 0.0

    assert agent.best_action(state) == Action.STAND


def test_learn_terminal_state_updates_toward_reward():
    agent = QLearningAgent(
        learning_rate=0.5,
        discount_factor=1.0,
        epsilon=0.0,
    )
    state = make_state(player_value=20, dealer_upcard=10)

    agent.learn(
        state=state,
        action=Action.STAND,
        reward=1.0,
        next_state=None,
        done=True,
    )

    assert agent.q_table[state.as_tuple()][Action.STAND] == pytest.approx(0.5)


def test_learn_non_terminal_state_uses_best_legal_next_action_only():
    agent = QLearningAgent(
        learning_rate=0.5,
        discount_factor=1.0,
        epsilon=0.0,
    )

    state = make_state(player_value=12, dealer_upcard=10)
    next_state = make_state(player_value=18, dealer_upcard=10)

    next_key = next_state.as_tuple()
    agent.q_table[next_key][Action.HIT] = -1.0
    agent.q_table[next_key][Action.STAND] = 0.8
    agent.q_table[next_key][Action.DOUBLE] = 10.0
    agent.q_table[next_key][Action.SPLIT] = 20.0

    agent.learn(
        state=state,
        action=Action.HIT,
        reward=0.0,
        next_state=next_state,
        done=False,
        next_available_actions=[Action.HIT, Action.STAND],
    )

    assert agent.q_table[state.as_tuple()][Action.HIT] == pytest.approx(0.4)


def test_learn_non_terminal_requires_next_state():
    agent = QLearningAgent(epsilon=0.0)
    state = make_state(player_value=12, dealer_upcard=10)

    with pytest.raises(ValueError):
        agent.learn(
            state=state,
            action=Action.HIT,
            reward=0.0,
            next_state=None,
            done=False,
        )


def test_learn_does_not_decay_epsilon():
    agent = QLearningAgent(
        learning_rate=0.5,
        discount_factor=1.0,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.5,
    )

    state = make_state(player_value=20, dealer_upcard=10)

    agent.learn(
        state=state,
        action=Action.STAND,
        reward=1.0,
        next_state=None,
        done=True,
    )

    assert agent.epsilon == pytest.approx(1.0)


class ImmediateTerminalGame:
    def __init__(self):
        self.round_reward = -1.0
        self.step_called = False

    def reset(self):
        return None

    def step(self, action):
        self.step_called = True
        raise AssertionError("step() should not be called")

    def render(self):
        pass


def test_train_one_episode_handles_reset_returning_none():
    agent = QLearningAgent(
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.5,
    )
    game = ImmediateTerminalGame()

    reward = agent.train_one_episode(game)

    assert reward == pytest.approx(-1.0)
    assert game.step_called is False
    assert agent.epsilon == pytest.approx(0.5)


def test_play_episode_handles_reset_returning_none():
    agent = QLearningAgent(epsilon=0.0)
    game = ImmediateTerminalGame()

    reward = agent.play_episode(game)

    assert reward == pytest.approx(-1.0)
    assert game.step_called is False


class FakeSplitRewardGame:
    """
    Hand 0 gets +1.
    Hand 1 gets -2.
    Total round reward is -1.
    """

    def __init__(self):
        self.active_hand_index = 0
        self.hand_rewards = [1.0, -2.0]
        self.round_reward = -1.0
        self.step_count = 0

        self.state_0 = make_state(
            player_value=16,
            dealer_upcard=10,
            can_double=False,
            can_split=False,
            is_split_hand=True,
            active_hand_index=0,
            num_hands=2,
        )

        self.state_1 = make_state(
            player_value=18,
            dealer_upcard=10,
            can_double=False,
            can_split=False,
            is_split_hand=True,
            active_hand_index=1,
            num_hands=2,
        )

    def reset(self):
        return self.state_0

    def available_actions(self):
        return [Action.STAND]

    def step(self, action):
        assert action == Action.STAND

        self.step_count += 1

        if self.step_count == 1:
            self.active_hand_index = 1
            return self.state_1, 0.0, False

        return None, self.round_reward, True


def test_train_one_episode_uses_per_hand_rewards_for_split_hands():
    agent = QLearningAgent(
        learning_rate=1.0,
        discount_factor=1.0,
        epsilon=0.0,
        epsilon_decay=1.0,
    )

    game = FakeSplitRewardGame()

    reward = agent.train_one_episode(game)

    assert reward == pytest.approx(-1.0)

    q_hand_0 = agent.q_table[game.state_0.as_tuple()][Action.STAND]
    q_hand_1 = agent.q_table[game.state_1.as_tuple()][Action.STAND]

    assert q_hand_0 == pytest.approx(1.0)
    assert q_hand_1 == pytest.approx(-2.0)


def test_train_one_episode_decays_epsilon_once():
    agent = QLearningAgent(
        learning_rate=1.0,
        discount_factor=1.0,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.5,
    )

    game = FakeSplitRewardGame()

    agent.train_one_episode(game)

    assert agent.epsilon == pytest.approx(0.5)


def test_train_one_episode_epsilon_does_not_decay_below_minimum():
    agent = QLearningAgent(
        learning_rate=1.0,
        discount_factor=1.0,
        epsilon=0.2,
        epsilon_min=0.1,
        epsilon_decay=0.01,
    )

    game = FakeSplitRewardGame()

    agent.train_one_episode(game)

    assert agent.epsilon == pytest.approx(0.1)