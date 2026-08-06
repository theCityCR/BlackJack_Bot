"""Tests for shared torch device selection."""

from __future__ import annotations

import torch

from agents.common import resolve_torch_device
from agents.double_q_network_learning.double_q_network_learning_agent import (
    DoubleQNetworkLearningAgent,
)
from game import Action, GameState


def test_resolve_torch_device_explicit_cpu():
    assert resolve_torch_device("cpu").type == "cpu"


def test_resolve_torch_device_prefers_accelerator_when_available():
    device = resolve_torch_device()
    if torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        # MPS is opt-in only; default without CUDA is CPU.
        assert device.type == "cpu"


def test_resolve_torch_device_honors_env(monkeypatch):
    monkeypatch.setenv("BLACKJACK_TORCH_DEVICE", "cpu")
    assert resolve_torch_device().type == "cpu"


def test_double_dqn_places_model_on_resolved_device():
    agent = DoubleQNetworkLearningAgent(batch_size=4, device=None)
    assert next(agent.model.parameters()).device.type == agent.device.type


def test_double_dqn_train_step_runs_on_selected_device():
    agent = DoubleQNetworkLearningAgent(
        batch_size=4,
        min_replay_size=4,
        train_updates_per_episode=1,
    )
    state = GameState(
        player_value=16,
        dealer_upcard=10,
        usable_ace=False,
        can_double=True,
        can_split=False,
        is_split_hand=False,
        active_hand_index=0,
        num_hands=1,
        count_vector=(4, 4, 4, 4, 4, 4, 4, 4, 4, 16),
    )
    for _ in range(8):
        agent.remember(state, Action.HIT, -1.0, None, True, None)

    before = agent.training_steps
    agent.train_step()
    assert agent.training_steps == before + 1
