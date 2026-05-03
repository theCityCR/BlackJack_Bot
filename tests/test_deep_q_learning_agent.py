import torch

from agents.deprecated.deep_q_learning.deep_q_learning_agent import DeepQLearningAgent, ACTION_LIST
from game import Action, BlackjackGame, GameState


def test_encode_state_returns_tensor_of_size_8():
    agent = DeepQLearningAgent()

    state = GameState(
        player_value=16,
        dealer_upcard=10,
        usable_ace=False,
        can_double=True,
        can_split=False,
        is_split_hand=False,
        active_hand_index=0,
        num_hands=1,
    )

    encoded = agent.encode_state(state)

    assert isinstance(encoded, torch.Tensor)
    assert encoded.shape == (8,)
    assert encoded.dtype == torch.float32


def test_choose_action_returns_legal_action_only():
    agent = DeepQLearningAgent(epsilon=1.0)

    state = GameState(
        player_value=12,
        dealer_upcard=6,
        usable_ace=False,
        can_double=False,
        can_split=False,
        is_split_hand=False,
        active_hand_index=0,
        num_hands=1,
    )

    legal_actions = [Action.HIT, Action.STAND]

    for _ in range(100):
        action = agent.choose_action(state, legal_actions)
        assert action in legal_actions


def test_best_action_respects_legal_action_mask():
    agent = DeepQLearningAgent(epsilon=0.0)

    state = GameState(
        player_value=12,
        dealer_upcard=6,
        usable_ace=False,
        can_double=False,
        can_split=False,
        is_split_hand=False,
        active_hand_index=0,
        num_hands=1,
    )

    with torch.no_grad():
        for param in agent.model.parameters():
            param.zero_()

        # Force illegal SPLIT to have high value.
        final_layer = agent.model.net[-1]
        final_layer.bias[ACTION_LIST.index(Action.SPLIT)] = 100.0

        # Legal STAND has lower value.
        final_layer.bias[ACTION_LIST.index(Action.STAND)] = 10.0

    action = agent.best_action(state, [Action.HIT, Action.STAND])

    assert action == Action.STAND


def test_remember_adds_transition_to_replay_buffer():
    agent = DeepQLearningAgent()

    state = GameState(
        player_value=10,
        dealer_upcard=6,
        usable_ace=False,
        can_double=True,
        can_split=False,
        is_split_hand=False,
        active_hand_index=0,
        num_hands=1,
    )

    next_state = GameState(
        player_value=18,
        dealer_upcard=6,
        usable_ace=False,
        can_double=False,
        can_split=False,
        is_split_hand=False,
        active_hand_index=0,
        num_hands=1,
    )

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
    assert transition.action_index == ACTION_LIST.index(Action.HIT)
    assert transition.reward == 0.0
    assert transition.done is False
    assert transition.next_state is not None


def test_train_step_runs_when_replay_buffer_has_enough_samples():
    agent = DeepQLearningAgent(batch_size=4)

    state = GameState(
        player_value=10,
        dealer_upcard=6,
        usable_ace=False,
        can_double=True,
        can_split=False,
        is_split_hand=False,
        active_hand_index=0,
        num_hands=1,
    )

    next_state = GameState(
        player_value=18,
        dealer_upcard=6,
        usable_ace=False,
        can_double=False,
        can_split=False,
        is_split_hand=False,
        active_hand_index=0,
        num_hands=1,
    )

    for _ in range(4):
        agent.remember(
            state=state,
            action=Action.HIT,
            reward=0.0,
            next_state=next_state,
            done=False,
            next_available_actions=[Action.HIT, Action.STAND],
        )

    old_steps = agent.training_steps
    agent.train_step()

    assert agent.training_steps == old_steps + 1


def test_train_one_episode_runs_without_crashing():
    agent = DeepQLearningAgent(batch_size=4)
    game = BlackjackGame()

    reward = agent.train_one_episode(game)

    assert isinstance(reward, float)


def test_play_episode_runs_without_learning():
    agent = DeepQLearningAgent(epsilon=0.0)
    game = BlackjackGame()

    old_steps = agent.training_steps
    reward = agent.play_episode(game)

    assert isinstance(reward, float)
    assert agent.training_steps == old_steps


def test_epsilon_decays_but_not_below_minimum():
    agent = DeepQLearningAgent(
        epsilon=0.1,
        epsilon_min=0.05,
        epsilon_decay=0.5,
    )

    agent.decay_epsilon()
    assert agent.epsilon == 0.05

    agent.decay_epsilon()
    assert agent.epsilon == 0.05