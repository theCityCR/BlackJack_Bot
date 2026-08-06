import pytest
import torch

from agents.common import ACTION_TO_INDEX, Transition
from agents.double_dqn import DoubleQNetworkLearningAgent
from conftest import make_state
from game import Action

def test_legal_action_indices_returns_correct_indices():
    agent = DoubleQNetworkLearningAgent()

    indices = agent.legal_action_indices([
        Action.HIT,
        Action.DOUBLE,
        Action.SPLIT,
    ])

    assert indices == [0, 2, 3]

def test_choose_action_with_epsilon_one_returns_legal_action():
    agent = DoubleQNetworkLearningAgent(epsilon=1.0)
    state = make_state()

    available_actions = [Action.STAND]

    for _ in range(20):
        assert agent.choose_action(state, available_actions) == Action.STAND

def test_best_action_only_returns_legal_action_even_if_illegal_q_is_high():
    agent = DoubleQNetworkLearningAgent(epsilon=0.0)
    state = make_state()

    with torch.no_grad():
        for parameter in agent.model.parameters():
            parameter.zero_()

        final_layer = agent.model.net[-1]
        final_layer.bias[ACTION_TO_INDEX[Action.SPLIT]] = 100.0
        final_layer.bias[ACTION_TO_INDEX[Action.STAND]] = 1.0

    # SPLIT has the highest Q-value, but it is illegal here.
    available_actions = [Action.HIT, Action.STAND]

    assert agent.best_action(state, available_actions) in available_actions
    assert agent.best_action(state, available_actions) != Action.SPLIT

def test_remember_adds_encoded_transition_to_replay_buffer():
    agent = DoubleQNetworkLearningAgent()

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

    transition = agent.replay_buffer[0]

    assert isinstance(transition, Transition)
    assert transition.state.shape == torch.Size([19])
    assert transition.action_index == ACTION_TO_INDEX[Action.HIT]
    assert transition.reward == 0.0
    assert transition.next_state.shape == torch.Size([19])
    assert transition.done is False
    assert transition.next_legal_action_indices == [
        ACTION_TO_INDEX[Action.HIT],
        ACTION_TO_INDEX[Action.STAND],
    ]

def test_remember_handles_terminal_transition():
    agent = DoubleQNetworkLearningAgent()
    state = make_state()

    agent.remember(
        state=state,
        action=Action.STAND,
        reward=1.0,
        next_state=None,
        done=True,
        next_available_actions=None,
    )

    transition = agent.replay_buffer[0]

    assert transition.next_state is None
    assert transition.done is True
    assert transition.next_legal_action_indices == []

def test_train_step_does_nothing_when_replay_buffer_too_small():
    agent = DoubleQNetworkLearningAgent(batch_size=4)

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

def test_train_step_updates_training_steps_when_enough_samples():
    agent = DoubleQNetworkLearningAgent(
        batch_size=4,
        min_replay_size=4,
    )

    for _ in range(4):
        agent.remember(
            state=make_state(),
            action=Action.STAND,
            reward=1.0,
            next_state=None,
            done=True,
            next_available_actions=None,
        )

    agent.train_step()

    assert agent.training_steps == 1

def test_decay_epsilon_respects_minimum():
    agent = DoubleQNetworkLearningAgent(
        epsilon=0.06,
        epsilon_min=0.05,
        epsilon_decay=0.5,
    )

    agent.decay_epsilon()

    assert agent.epsilon == pytest.approx(0.05)

def test_model_output_has_one_q_value_per_action():
    agent = DoubleQNetworkLearningAgent()
    state = make_state()

    encoded = agent.encode_state(state).unsqueeze(0).to(agent.device)
    output = agent.model(encoded)

    assert output.shape == torch.Size([1, 4])
