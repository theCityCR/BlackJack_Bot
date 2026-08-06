"""Tests for shared GameState encoding (agents.common.encode_state)."""

from __future__ import annotations

import pytest
import torch

from agents.common import (
    ACTION_LIST,
    ACTION_TO_INDEX,
    HAND_FEATURE_COUNT,
    STATE_SIZE,
    encode_state,
)
from config import NUM_DECKS
from conftest import make_state
from game import Action

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

def test_encode_state_returns_19_features():
    encoded = encode_state(make_state())
    assert isinstance(encoded, torch.Tensor)
    assert encoded.shape == torch.Size([STATE_SIZE])
    assert encoded.dtype == torch.float32

def test_encode_state_normalizes_basic_features():
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
    encoded = encode_state(state)
    assert encoded[0].item() == pytest.approx(1.0)
    assert encoded[1].item() == pytest.approx(1.0)
    assert encoded[2].item() == pytest.approx(1.0)
    assert encoded[3].item() == pytest.approx(1.0)
    assert encoded[4].item() == pytest.approx(1.0)
    assert encoded[5].item() == pytest.approx(1.0)
    assert encoded[6].item() == pytest.approx(0.5)
    assert encoded[7].item() == pytest.approx(1.0)

def test_encode_state_stores_cards_remaining_fraction():
    state = make_state(count_vector=(2, 2, 2, 2, 2, 2, 2, 2, 2, 8))
    encoded = encode_state(state)
    expected = sum(state.count_vector) / (52 * NUM_DECKS)
    assert encoded[8].item() == pytest.approx(expected)

def test_encode_state_normalizes_count_vector_by_cards_remaining():
    state = make_state(count_vector=(2, 2, 2, 2, 2, 2, 2, 2, 2, 8))
    encoded = encode_state(state)
    normalized = encoded[9:].tolist()
    expected = [count / sum(state.count_vector) for count in state.count_vector]
    assert normalized == pytest.approx(expected)
    assert sum(normalized) == pytest.approx(1.0)

def test_encode_state_handles_empty_count_vector():
    state = make_state(count_vector=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    encoded = encode_state(state)
    assert encoded[8].item() == pytest.approx(0.0)
    assert encoded[9:].tolist() == pytest.approx([0.0] * 10)

def test_encode_state_shoe_features_off_zeros_shoe_dims():
    state = make_state()
    encoded = encode_state(state, use_shoe_features=False)
    assert encoded.shape == torch.Size([STATE_SIZE])
    assert encoded[HAND_FEATURE_COUNT:].tolist() == pytest.approx([0.0] * 11)

def test_encode_state_compact_hand_only():
    encoded = encode_state(make_state(), compact_hand_only=True)
    assert encoded.shape == torch.Size([HAND_FEATURE_COUNT])
