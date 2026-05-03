import pytest
import torch

from agents.prioritized_replay.dueling_dqn_prioritized_agent import (
    ACTION_LIST,
    ACTION_TO_INDEX,
    DuelingDQNAgent,
    PrioritizedReplayBuffer,
    Transition,
)
from game import Action, BlackjackGame, GameState


def make_state(
    player_value=16,
    dealer_upcard=10,
    usable_ace=False,
    can_double=True,
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
        count_vector=(8, 8, 8, 8, 8, 8, 8, 8, 8, 32),
    )


def make_transition(reward=0.0, done=False):
    state = torch.zeros(19, dtype=torch.float32)
    next_state = None if done else torch.ones(19, dtype=torch.float32)

    return Transition(
        state=state,
        action_index=ACTION_TO_INDEX[Action.HIT],
        reward=reward,
        next_state=next_state,
        done=done,
        next_legal_action_indices=[
            ACTION_TO_INDEX[Action.HIT],
            ACTION_TO_INDEX[Action.STAND],
        ],
    )


def test_prioritized_replay_buffer_adds_transitions():
    buffer = PrioritizedReplayBuffer(capacity=3)

    buffer.add(make_transition())
    buffer.add(make_transition())
    buffer.add(make_transition())

    assert len(buffer) == 3
    assert len(buffer.buffer) == 3
    assert len(buffer.priorities) == 3


def test_prioritized_replay_buffer_overwrites_old_transitions():
    buffer = PrioritizedReplayBuffer(capacity=2)

    buffer.add(make_transition(reward=1.0))
    buffer.add(make_transition(reward=2.0))
    buffer.add(make_transition(reward=3.0))

    rewards = [transition.reward for transition in buffer.buffer]

    assert len(buffer) == 2
    assert 3.0 in rewards
    assert 1.0 not in rewards


def test_prioritized_replay_buffer_sample_returns_correct_shapes():
    buffer = PrioritizedReplayBuffer(capacity=10)

    for index in range(10):
        buffer.add(make_transition(reward=float(index)))

    batch, indices, weights = buffer.sample(batch_size=4)

    assert len(batch) == 4
    assert len(indices) == 4
    assert weights.shape == (4,)
    assert torch.all(weights > 0)
    assert torch.all(weights <= 1)


def test_prioritized_replay_buffer_updates_priorities():
    buffer = PrioritizedReplayBuffer(capacity=5)

    for _ in range(5):
        buffer.add(make_transition())

    old_priority = buffer.priorities[0]

    buffer.update_priorities(
        indices=[0],
        td_errors=torch.tensor([10.0]),
    )

    assert buffer.priorities[0] > old_priority


def test_agent_encodes_state_to_expected_size():
    agent = DuelingDQNAgent(device="cpu")
    state = make_state()

    encoded = agent.encode_state(state)

    assert encoded.shape == (19,)
    assert encoded.dtype == torch.float32


def test_agent_legal_action_indices_are_correct():
    agent = DuelingDQNAgent(device="cpu")

    indices = agent.legal_action_indices([
        Action.HIT,
        Action.STAND,
        Action.DOUBLE,
    ])

    assert indices == [
        ACTION_TO_INDEX[Action.HIT],
        ACTION_TO_INDEX[Action.STAND],
        ACTION_TO_INDEX[Action.DOUBLE],
    ]


def test_agent_choose_action_returns_legal_action_with_full_exploration():
    agent = DuelingDQNAgent(
        epsilon=1.0,
        device="cpu",
    )

    state = make_state()
    available_actions = [Action.HIT, Action.STAND]

    for _ in range(50):
        action = agent.choose_action(state, available_actions)
        assert action in available_actions


def test_agent_best_action_returns_legal_action():
    agent = DuelingDQNAgent(
        epsilon=0.0,
        device="cpu",
    )

    state = make_state()
    available_actions = [Action.HIT, Action.STAND]

    action = agent.best_action(state, available_actions)

    assert action in available_actions


def test_agent_remember_adds_transition_to_prioritized_buffer():
    agent = DuelingDQNAgent(device="cpu")

    state = make_state()
    next_state = make_state(player_value=18)

    agent.remember(
        state=state,
        action=Action.HIT,
        reward=0.0,
        next_state=next_state,
        done=False,
        next_available_actions=[Action.HIT, Action.STAND],
    )

    assert len(agent.replay_buffer) == 1

    stored = agent.replay_buffer.buffer[0]

    assert stored.action_index == ACTION_TO_INDEX[Action.HIT]
    assert stored.reward == 0.0
    assert stored.done is False
    assert stored.next_state is not None
    assert stored.next_legal_action_indices == [
        ACTION_TO_INDEX[Action.HIT],
        ACTION_TO_INDEX[Action.STAND],
    ]


def test_agent_remember_terminal_transition():
    agent = DuelingDQNAgent(device="cpu")

    state = make_state()

    agent.remember(
        state=state,
        action=Action.STAND,
        reward=1.0,
        next_state=None,
        done=True,
        next_available_actions=None,
    )

    stored = agent.replay_buffer.buffer[0]

    assert stored.action_index == ACTION_TO_INDEX[Action.STAND]
    assert stored.reward == 1.0
    assert stored.done is True
    assert stored.next_state is None
    assert stored.next_legal_action_indices == []


def test_agent_train_step_does_nothing_when_buffer_too_small():
    agent = DuelingDQNAgent(
        batch_size=4,
        device="cpu",
    )

    agent.remember(
        state=make_state(),
        action=Action.HIT,
        reward=0.0,
        next_state=make_state(player_value=17),
        done=False,
        next_available_actions=[Action.HIT, Action.STAND],
    )

    old_training_steps = agent.training_steps

    agent.train_step()

    assert agent.training_steps == old_training_steps


def test_agent_train_step_updates_model_when_buffer_large_enough():
    agent = DuelingDQNAgent(
        batch_size=4,
        min_replay_size=4,
        target_update_interval=100,
        device="cpu",
    )

    for _ in range(10):
        agent.remember(
            state=make_state(),
            action=Action.HIT,
            reward=0.0,
            next_state=make_state(player_value=17),
            done=False,
            next_available_actions=[Action.HIT, Action.STAND],
        )

    old_parameters = [
        parameter.detach().clone()
        for parameter in agent.model.parameters()
    ]

    agent.train_step()

    new_parameters = [
        parameter.detach().clone()
        for parameter in agent.model.parameters()
    ]

    assert agent.training_steps == 1

    assert any(
        not torch.equal(old, new)
        for old, new in zip(old_parameters, new_parameters)
    )


def test_agent_train_step_updates_priorities():
    agent = DuelingDQNAgent(
        batch_size=4,
        min_replay_size=4,
        device="cpu",
    )

    for index in range(10):
        agent.remember(
            state=make_state(player_value=12 + index % 8),
            action=Action.HIT,
            reward=float(index % 3 - 1),
            next_state=make_state(player_value=13 + index % 7),
            done=False,
            next_available_actions=[Action.HIT, Action.STAND],
        )

    old_priorities = list(agent.replay_buffer.priorities)

    agent.train_step()

    new_priorities = agent.replay_buffer.priorities

    assert old_priorities != new_priorities


def test_agent_target_model_updates_on_interval():
    agent = DuelingDQNAgent(
        batch_size=4,
        min_replay_size=4,
        target_update_interval=1,
        device="cpu",
    )

    for _ in range(10):
        agent.remember(
            state=make_state(),
            action=Action.HIT,
            reward=1.0,
            next_state=None,
            done=True,
            next_available_actions=None,
        )

    agent.train_step()

    for model_parameter, target_parameter in zip(
        agent.model.parameters(),
        agent.target_model.parameters(),
    ):
        assert torch.allclose(model_parameter, target_parameter)


def test_agent_decay_epsilon_respects_minimum():
    agent = DuelingDQNAgent(
        epsilon=0.06,
        epsilon_min=0.05,
        epsilon_decay=0.1,
        device="cpu",
    )

    agent.decay_epsilon()

    assert agent.epsilon == 0.05


def test_agent_can_train_one_episode():
    agent = DuelingDQNAgent(
        batch_size=4,
        min_replay_size=4,
        train_updates_per_episode=1,
        device="cpu",
    )

    game = BlackjackGame()

    reward = agent.train_one_episode(game)

    assert isinstance(reward, float)
    assert len(agent.replay_buffer) >= 0


def test_agent_can_play_one_episode_without_training():
    agent = DuelingDQNAgent(
        epsilon=0.0,
        device="cpu",
    )

    game = BlackjackGame()

    reward = agent.play_episode(game)

    assert isinstance(reward, float)