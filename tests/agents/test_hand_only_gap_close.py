"""Tests for compact hand-only encoding and gap-close wiring."""

from __future__ import annotations

from agents.common import HAND_FEATURE_COUNT, STATE_SIZE, encode_state
from agents.double_q_network_learning.double_q_network_learning_agent import (
    DoubleQNetworkLearningAgent,
)
from game import GameState


def make_state() -> GameState:
    return GameState(
        player_value=16,
        dealer_upcard=10,
        usable_ace=False,
        can_double=True,
        can_split=False,
        is_split_hand=False,
        active_hand_index=0,
        num_hands=1,
        count_vector=(2, 2, 2, 2, 2, 2, 2, 2, 2, 8),
    )


def test_compact_hand_only_encode_is_8d():
    state = make_state()
    encoded = encode_state(state, compact_hand_only=True)
    assert encoded.shape == (HAND_FEATURE_COUNT,)
    full = encode_state(state, use_shoe_features=True)
    assert full.shape == (STATE_SIZE,)
    assert encoded.tolist() == full[:HAND_FEATURE_COUNT].tolist()


def test_hand_only_agent_uses_8d_network():
    agent = DoubleQNetworkLearningAgent(hand_only_encoder=True, batch_size=4)
    assert agent.input_size == HAND_FEATURE_COUNT
    assert agent.hand_only_encoder is True
    assert agent.use_shoe_features is False
    encoded = agent.encode_state(make_state())
    assert encoded.shape == (HAND_FEATURE_COUNT,)
