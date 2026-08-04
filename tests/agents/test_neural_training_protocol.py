"""Tests for the shared neural training protocol."""

from __future__ import annotations

from agents.common import neural_training_kwargs
from agents.deep_q_learning.deep_q_learning_agent import DeepQLearningAgent
from agents.double_q_network_learning.double_q_network_learning_agent import (
    DoubleQNetworkLearningAgent,
)
from agents.dueling_dqn.dueling_dqn_agent import DuelingDQNAgent
from agents.prioritized_replay.dueling_dqn_prioritized_agent import (
    PrioritizedDuelingDQNAgent,
)
from config import (
    NEURAL_BATCH_SIZE,
    NEURAL_EPSILON_DECAY,
    NEURAL_LEARNING_RATE,
    NEURAL_MIN_REPLAY_SIZE,
    NEURAL_TARGET_UPDATE_INTERVAL,
    NEURAL_TRAIN_UPDATES_PER_EPISODE,
    NEURAL_TRAINING_EPISODES,
)


def test_neural_training_kwargs_match_config():
    kwargs = neural_training_kwargs()
    assert kwargs["learning_rate"] == NEURAL_LEARNING_RATE
    assert kwargs["batch_size"] == NEURAL_BATCH_SIZE
    assert kwargs["epsilon_decay"] == NEURAL_EPSILON_DECAY
    assert kwargs["target_update_interval"] == NEURAL_TARGET_UPDATE_INTERVAL
    assert kwargs["min_replay_size"] == NEURAL_MIN_REPLAY_SIZE
    assert kwargs["train_updates_per_episode"] == NEURAL_TRAIN_UPDATES_PER_EPISODE
    assert kwargs["train_updates_per_episode"] == 4
    assert kwargs["epsilon_decay"] == 0.99997


def test_neural_agents_share_default_training_hyperparameters():
    agents = [
        DeepQLearningAgent(),
        DoubleQNetworkLearningAgent(),
        DuelingDQNAgent(device="cpu"),
        PrioritizedDuelingDQNAgent(device="cpu"),
    ]

    reference = agents[0]
    for agent in agents[1:]:
        assert agent.batch_size == reference.batch_size == NEURAL_BATCH_SIZE
        assert agent.epsilon_decay == reference.epsilon_decay == NEURAL_EPSILON_DECAY
        assert (
            agent.target_update_interval
            == reference.target_update_interval
            == NEURAL_TARGET_UPDATE_INTERVAL
        )
        assert (
            agent.min_replay_size
            == reference.min_replay_size
            == NEURAL_MIN_REPLAY_SIZE
        )
        assert (
            agent.train_updates_per_episode
            == reference.train_updates_per_episode
            == NEURAL_TRAIN_UPDATES_PER_EPISODE
        )


def test_neural_training_episode_budget_is_shared():
    assert NEURAL_TRAINING_EPISODES == 200_000
