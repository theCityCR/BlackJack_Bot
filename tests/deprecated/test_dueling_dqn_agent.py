import pytest
import torch

from agents.dueling_dqn.dueling_dqn_agent import (
    ACTION_LIST,
    ACTION_TO_INDEX,
    DuelingDQN,
    DuelingDQNAgent,
    Transition,
)
from config import NUM_DECKS
from game import Action, GameState


def make_state(
    player_value=16,
    dealer_upcard=10,
    usable_ace=False,
    can_double=True,
    can_split=False,
    is_split_hand=False,
    active_hand_index=0,
    num_hands=1,
    count_vector=(4, 4, 4, 4, 4, 4, 4, 4, 4, 16),
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
        count_vector=count_vector,
    )


def test_action_list_order_matches_action_indices():
    assert ACTION_LIST == [
        Action.HIT,
        Action.STAND,
        Action.DOUBLE,
        Action.SPLIT,
    ]

    assert ACTION_TO_INDEX[Action.HIT] == 0
    assert ACTION_TO_INDEX[Action.STAND] == 1
    assert ACTION_TO_INDEX[Action.DOUBLE] == 2
    assert ACTION_TO_INDEX[Action.SPLIT] == 3


def test_dueling_network_outputs_one_q_value_per_action():
    model = DuelingDQN(input_size=19, output_size=4)
    batch = torch.zeros((5, 19))

    output = model(batch)

    assert output.shape == torch.Size([5, 4])


def test_dueling_network_has_value_and_advantage_streams():
    model = DuelingDQN(input_size=19, output_size=4)

    assert hasattr(model, "value_stream")
    assert hasattr(model, "advantage_stream")


def test_encode_state_returns_19_features():
    agent = DuelingDQNAgent()
    state = make_state()

    encoded = agent.encode_state(state)

    assert isinstance(encoded, torch.Tensor)
    assert encoded.shape == torch.Size([19])


def test_encode_state_normalizes_basic_features():
    agent = DuelingDQNAgent()
    state = make_state(
        player_value=21,
        dealer_upcard=10,
        usable_ace=True,
        can_double=True,
        can_split=True,
        is_split_hand=True,
        active_hand_index=2,
        num_hands=4,
    )

    encoded = agent.encode_state(state)

    assert encoded[0].item() == pytest.approx(1.0)
    assert encoded[1].item() == pytest.approx(1.0)
    assert encoded[2].item() == pytest.approx(1.0)
    assert encoded[3].item() == pytest.approx(1.0)
    assert encoded[4].item() == pytest.approx(1.0)
    assert encoded[5].item() == pytest.approx(1.0)
    assert encoded[6].item() == pytest.approx(0.5)
    assert encoded[7].item() == pytest.approx(1.0)


def test_encode_state_stores_cards_remaining_fraction():
    agent = DuelingDQNAgent()
    state = make_state(
        count_vector=(2, 2, 2, 2, 2, 2, 2, 2, 2, 8),
    )

    encoded = agent.encode_state(state)

    cards_remaining = sum(state.count_vector)
    expected_fraction = cards_remaining / (52 * NUM_DECKS)

    assert encoded[8].item() == pytest.approx(expected_fraction)


def test_encode_state_normalizes_count_vector_by_cards_remaining():
    agent = DuelingDQNAgent()
    state = make_state(
        count_vector=(2, 2, 2, 2, 2, 2, 2, 2, 2, 8),
    )

    encoded = agent.encode_state(state)

    normalized_counts = encoded[9:].tolist()
    cards_remaining = sum(state.count_vector)

    expected = [
        count / cards_remaining
        for count in state.count_vector
    ]

    assert normalized_counts == pytest.approx(expected)
    assert sum(normalized_counts) == pytest.approx(1.0)


def test_encode_state_handles_empty_count_vector():
    agent = DuelingDQNAgent()
    state = make_state(
        count_vector=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )

    encoded = agent.encode_state(state)

    assert encoded[8].item() == pytest.approx(0.0)
    assert encoded[9:].tolist() == pytest.approx([0.0] * 10)


def test_legal_action_indices_returns_correct_indices():
    agent = DuelingDQNAgent()

    indices = agent.legal_action_indices([
        Action.HIT,
        Action.DOUBLE,
        Action.SPLIT,
    ])

    assert indices == [0, 2, 3]


def test_choose_action_with_epsilon_one_returns_legal_action():
    agent = DuelingDQNAgent(epsilon=1.0)
    state = make_state()

    available_actions = [Action.STAND]

    for _ in range(20):
        assert agent.choose_action(state, available_actions) == Action.STAND


def test_best_action_only_returns_legal_action_even_if_illegal_q_is_high():
    agent = DuelingDQNAgent(epsilon=0.0)
    state = make_state()

    with torch.no_grad():
        for parameter in agent.model.parameters():
            parameter.zero_()

        final_advantage_layer = agent.model.advantage_stream[-1]
        final_value_layer = agent.model.value_stream[-1]

        final_value_layer.bias.fill_(0.0)
        final_advantage_layer.bias[ACTION_TO_INDEX[Action.SPLIT]] = 100.0
        final_advantage_layer.bias[ACTION_TO_INDEX[Action.STAND]] = 1.0

    action = agent.best_action(
        state,
        available_actions=[Action.HIT, Action.STAND],
    )

    assert action == Action.STAND


def test_remember_adds_transition_to_replay_buffer():
    agent = DuelingDQNAgent()
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
    assert transition.next_legal_action_indices == [0, 1]


def test_remember_terminal_transition_has_no_next_state():
    agent = DuelingDQNAgent()
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
    agent = DuelingDQNAgent(batch_size=4)

    agent.train_step()

    assert agent.training_steps == 0


def test_train_step_updates_training_steps_when_enough_samples():
    agent = DuelingDQNAgent(
        batch_size=4,
        target_update_interval=100,
    )

    state = make_state()
    next_state = make_state(player_value=18)

    for _ in range(4):
        agent.remember(
            state=state,
            action=Action.HIT,
            reward=0.0,
            next_state=next_state,
            done=False,
            next_available_actions=[Action.HIT, Action.STAND],
        )

    agent.train_step()

    assert agent.training_steps == 1


def test_train_step_updates_target_model_at_interval():
    agent = DuelingDQNAgent(
        batch_size=4,
        target_update_interval=1,
    )

    state = make_state()

    for _ in range(4):
        agent.remember(
            state=state,
            action=Action.STAND,
            reward=1.0,
            next_state=None,
            done=True,
            next_available_actions=None,
        )

    agent.train_step()

    for model_param, target_param in zip(
        agent.model.parameters(),
        agent.target_model.parameters(),
    ):
        assert torch.allclose(model_param, target_param)


def test_decay_epsilon_never_goes_below_minimum():
    agent = DuelingDQNAgent(
        epsilon=0.051,
        epsilon_min=0.05,
        epsilon_decay=0.1,
    )

    agent.decay_epsilon()

    assert agent.epsilon == pytest.approx(0.05)