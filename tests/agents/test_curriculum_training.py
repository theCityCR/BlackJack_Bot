"""Tests for two-phase shoe curriculum training."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from agents.common import (
    HAND_FEATURE_COUNT,
    SHOE_FEATURE_COUNT,
    STATE_SIZE,
    clear_replay_buffer,
    encode_state,
    run_neural_training_loop,
)
from agents.dqn import DeepQLearningAgent
from agents.replay import PrioritizedReplayBuffer
from conftest import make_state

def test_state_size_splits_hand_and_shoe_features():
    assert HAND_FEATURE_COUNT + SHOE_FEATURE_COUNT == STATE_SIZE == 19

def test_encode_state_zeros_shoe_features_when_disabled():
    state = make_state()
    with_shoe = encode_state(state, use_shoe_features=True)
    without_shoe = encode_state(state, use_shoe_features=False)

    assert with_shoe.shape == without_shoe.shape == (STATE_SIZE,)
    assert without_shoe[:HAND_FEATURE_COUNT].tolist() == pytest.approx(
        with_shoe[:HAND_FEATURE_COUNT].tolist()
    )
    assert without_shoe[HAND_FEATURE_COUNT:].tolist() == pytest.approx(
        [0.0] * SHOE_FEATURE_COUNT
    )
    assert with_shoe[HAND_FEATURE_COUNT:].abs().sum().item() > 0

def test_agent_encode_state_respects_use_shoe_features_flag():
    agent = DeepQLearningAgent()
    state = make_state()

    agent.use_shoe_features = False
    encoded = agent.encode_state(state)
    assert encoded[HAND_FEATURE_COUNT:].tolist() == pytest.approx(
        [0.0] * SHOE_FEATURE_COUNT
    )

    agent.use_shoe_features = True
    encoded = agent.encode_state(state)
    assert encoded[8].item() > 0

def test_prioritized_replay_buffer_clear():
    buffer = PrioritizedReplayBuffer(capacity=10)
    dummy = MagicMock()
    buffer.add(dummy)
    buffer.add(dummy)
    assert len(buffer) == 2

    buffer.clear()
    assert len(buffer) == 0

def test_curriculum_enables_shoe_features_and_clears_replay(monkeypatch):
    agent = DeepQLearningAgent(min_replay_size=10_000, batch_size=4)
    game = MagicMock()

    rewards = iter([0.0, 0.5, -0.5, 1.0])

    def fake_train_one_episode(_game):
        # Populate buffer during phase A so clear is observable.
        agent.replay_buffer.append(
            MagicMock(state=torch.zeros(STATE_SIZE))
        )
        return next(rewards)

    monkeypatch.setattr(agent, "train_one_episode", fake_train_one_episode)
    monkeypatch.setattr(
        "agents.common.evaluate_greedy",
        lambda _agent, _episodes: (0.0, {"draw": 1}),
    )

    assert agent.use_shoe_features is True
    run_neural_training_loop(
        agent,
        game,
        num_episodes=4,
        print_interval=100,
        curriculum=True,
        phase_a_episodes=2,
        warmstart=False,
    )

    assert agent.use_shoe_features is True
    # Phase B starts at episode 3: buffer cleared then two more episodes append.
    assert len(agent.replay_buffer) == 2

def test_no_curriculum_keeps_shoe_features_from_start(monkeypatch):
    agent = DeepQLearningAgent()
    game = MagicMock()
    monkeypatch.setattr(agent, "train_one_episode", lambda _game: 0.0)
    monkeypatch.setattr(
        "agents.common.evaluate_greedy",
        lambda _agent, _episodes: (0.0, {"draw": 1}),
    )

    run_neural_training_loop(
        agent,
        game,
        num_episodes=3,
        print_interval=100,
        curriculum=False,
        warmstart=False,
    )

    assert agent.use_shoe_features is True

def test_clear_replay_buffer_helper_works_on_deque():
    agent = DeepQLearningAgent()
    agent.replay_buffer.append(MagicMock())
    clear_replay_buffer(agent)
    assert len(agent.replay_buffer) == 0
