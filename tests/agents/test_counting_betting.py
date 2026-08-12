"""Tests for Hi-Lo counting and true-count bet schedules."""

from unittest.mock import patch

import pytest

from agents.betting import FlatBetSchedule, TrueCountBetSchedule
from agents.counting import (
    full_shoe_count_vector,
    running_count,
    true_count,
    true_count_from_shoe,
)
from game import ShoeObservation


def test_full_shoe_running_count_is_zero():
    full = full_shoe_count_vector(2)
    assert running_count(full, num_decks=2) == 0.0
    assert true_count(full, sum(full), num_decks=2) == 0.0


def test_removing_low_cards_raises_running_count():
    counts = list(full_shoe_count_vector(2))
    counts[4] -= 5  # rank 5
    assert running_count(tuple(counts), num_decks=2) == 5.0


def test_true_count_divides_by_decks_remaining():
    counts = list(full_shoe_count_vector(2))
    counts[4] -= 5
    remaining = sum(counts)
    expected = 5.0 / (remaining / 52.0)
    assert true_count(tuple(counts), remaining, num_decks=2) == pytest.approx(expected)


def test_true_count_from_shoe_matches_helpers():
    shoe = ShoeObservation(count_vector=full_shoe_count_vector(2), cards_remaining=104)
    assert true_count_from_shoe(shoe, num_decks=2) == 0.0


def test_default_schedule_ramps_with_floor_true_count():
    schedule = TrueCountBetSchedule()
    shoe = ShoeObservation(count_vector=full_shoe_count_vector(2), cards_remaining=104)

    with patch("agents.betting.true_count_from_shoe", return_value=0.2):
        assert schedule.choose_bet(shoe) == 1.0
    with patch("agents.betting.true_count_from_shoe", return_value=1.9):
        assert schedule.choose_bet(shoe) == 2.0
    with patch("agents.betting.true_count_from_shoe", return_value=2.0):
        assert schedule.choose_bet(shoe) == 4.0
    with patch("agents.betting.true_count_from_shoe", return_value=3.1):
        assert schedule.choose_bet(shoe) == 6.0
    with patch("agents.betting.true_count_from_shoe", return_value=4.0):
        assert schedule.choose_bet(shoe) == 8.0


def test_flat_bet_schedule_ignores_shoe():
    shoe = ShoeObservation(count_vector=full_shoe_count_vector(2), cards_remaining=104)
    assert FlatBetSchedule(bet=3.0).choose_bet(shoe) == 3.0


def test_schedule_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        TrueCountBetSchedule(bet_min=0)
    with pytest.raises(ValueError):
        TrueCountBetSchedule(bet_min=5, bet_max=2)
